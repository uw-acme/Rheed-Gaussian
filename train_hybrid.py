"""Train the Gaussian-fitting QKeras model on a mix of real and synthetic spot crops.

Combines ``train_real.py`` and ``train_synthetic.py``. All constants live in a
YAML config file:

    python train_hybrid.py configs/hybrid.yaml

Pipeline:
    1. Real crops: split the growths into train / validation / test, sample
       crops, background-subtract, normalize and drop dim crops, exactly as
       ``train_real.py``.
    2. Synthetic crops: single rotated 2D Gaussians, exactly as
       ``train_synthetic.py``.
    3. Mix them: ``hybrid.synthetic_fraction`` of the training and validation
       sets is synthetic. The real crops are requested at ``dataset.*_size`` and
       filtered first; the synthetic count is then derived from the surviving
       real count so the share is exact (the total ends up a little below the
       configured size). Training reshuffles the mixed set every epoch.
    4. Build (or load) the quantized ``gaussian`` model and train it
       self-supervised through the fixed reconstruction layer, with the
       weighted reconstruction MSE, early stopping and LR reduction of
       ``train_real.py``.
    5. Test separately on real and synthetic test sets (``test_size`` crops
       each, before the real filter), then save the model, the config used,
       the training history, the test losses and a few plots per source.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from train_real import gen_data as gen_real
from train_real import load_config as load_real_config
from train_real import load_splits
from train_real import save_plots as save_real_plots
from train_synthetic import (
    SyntheticGenerator,
    build_model,
    load_model,
    make_datasets,
    make_gaussian_gen_tf,
    make_losses,
    make_training_model,
    parse_angle,
    setup_tensorflow,
    train,
)
from train_synthetic import save_plots as save_synthetic_plots


# ============================================================================
# Config
# ============================================================================


def load_config(path: str | Path) -> dict:
    cfg = load_real_config(path)

    g = cfg["gaussian"]
    g["min_theta"] = parse_angle(g["min_theta"])
    g["max_theta"] = parse_angle(g["max_theta"])

    frac = cfg["hybrid"]["synthetic_fraction"]
    if not 0.0 <= frac < 1.0:
        raise ValueError(
            "hybrid.synthetic_fraction must be in [0, 1); for purely synthetic training use "
            "train_synthetic.py"
        )

    return cfg


# ============================================================================
# Mixing
# ============================================================================


def synthetic_count(n_real: int, synthetic_fraction: float) -> int:
    """Number of synthetic crops so that synthetic / (real + synthetic) == synthetic_fraction."""
    return int(round(n_real * synthetic_fraction / (1.0 - synthetic_fraction)))


def mix(real: np.ndarray, synthetic: np.ndarray) -> np.ndarray:
    """Concatenate both sources. Not shuffled here: make_datasets shuffles the training set
    every epoch, and the order of validation doesn't matter."""
    return np.concatenate([real, synthetic])


def gen_mixed(real_pool, requested: int, rng, min_peak, frac, syn_gen, desc: str) -> np.ndarray:
    real = gen_real(requested, real_pool, rng, min_peak, desc=f"{desc} (real)")
    n_syn = synthetic_count(len(real), frac)
    syn, _ = syn_gen.gen_data(n_syn, desc=f"{desc} (synthetic)")
    print(f"[{desc} Mix]: {len(real)} real + {n_syn} synthetic "
          f"({100 * n_syn / max(len(real) + n_syn, 1):.1f}% synthetic)")
    return mix(real, syn)


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="Path to the YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rng = np.random.default_rng(cfg["seed"])

    # ---- Data -----------------------------------------------------------
    d = cfg["dataset"]
    frac = cfg["hybrid"]["synthetic_fraction"]
    min_peak = cfg["filter"]["min_peak"]

    splits = load_splits(cfg, rng)
    syn_gen = SyntheticGenerator(cfg, rng)

    train_img = gen_mixed(splits["Train"], d["training_size"], rng, min_peak, frac, syn_gen, "Train")
    val_img = gen_mixed(splits["Val"], d["validation_size"], rng, min_peak, frac, syn_gen, "Val")

    test_img = {"real": gen_real(d["test_size"], splits["Test"], rng, min_peak, desc="Test (real)")}
    test_img["synthetic"], test_label_syn = syn_gen.gen_data(d["test_size"], desc="Test (synthetic)")
    del splits

    # ---- Model ----------------------------------------------------------
    tf = setup_tensorflow(cfg["seed"])
    loss_fn, plain_loss_fn = make_losses(cfg)
    gaussian_gen_tf = make_gaussian_gen_tf(cfg)

    t = cfg["training"]
    if t.get("load_from"):
        model = load_model(t["load_from"])
    else:
        model = build_model(cfg)
    training_model = make_training_model(model, gaussian_gen_tf)

    # ---- Train ----------------------------------------------------------
    history = None
    if t["train"]:
        train_dataset, val_dataset = make_datasets(tf, cfg, train_img, val_img)
        history = train(tf, cfg, training_model, loss_fn, train_dataset, val_dataset)
    elif not t.get("load_from"):
        print("[Warning]: training.train is false and no training.load_from given; "
              "the model is untrained.")

    model.summary()

    # ---- Test (each source separately) ----------------------------------
    predictions, pred_recon, test_metrics = {}, {}, {}
    for source, img in test_img.items():
        predictions[source] = model.predict(img, batch_size=d["batch_size"])
        pred_recon[source] = gaussian_gen_tf(tf.constant(predictions[source])).numpy()
        loss = float(plain_loss_fn(tf.constant(img), tf.constant(pred_recon[source])).numpy())
        loss_train = float(loss_fn(tf.constant(img), tf.constant(pred_recon[source])).numpy())
        test_metrics[f"test_loss_recon_{source}"] = loss
        test_metrics[f"test_loss_recon_train_{source}"] = loss_train
        test_metrics[f"num_test_images_{source}"] = len(img)
        print(f"[Test Loss (recon, plain), {source}]: {loss:.6f}  [training loss]: {loss_train:.6f} "
              f"({len(img)} images)")

    # ---- Save -----------------------------------------------------------
    o = cfg["output"]
    model_dir = Path(o["models_dir"]) / o["model_name"]
    if o["save_model"]:
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(model_dir))
        print(f"[Saved Model]: {model_dir}")

        shutil.copy(args.config, model_dir / "config.yaml")
        if history is not None:
            with open(model_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump({k: [float(v) for v in vs] for k, vs in history.history.items()}, f, indent=2)
        with open(model_dir / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)

    if o["num_plot_examples"] > 0:
        plot_dir = (model_dir if o["save_model"] else Path(o["models_dir"])) / "plots"
        scaling = np.array(cfg["params"]["scaling"], dtype=np.float32)
        save_real_plots(plot_dir / "real", test_img["real"], pred_recon["real"],
                        predictions["real"], scaling, o["num_plot_examples"])
        true_recon_syn = gaussian_gen_tf(tf.constant(test_label_syn)).numpy()
        save_synthetic_plots(plot_dir / "synthetic", test_img["synthetic"], true_recon_syn,
                             pred_recon["synthetic"], o["num_plot_examples"])


if __name__ == "__main__":
    main()
