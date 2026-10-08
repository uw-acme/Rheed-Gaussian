"""Train the Gaussian-fitting QKeras model on real RHEED spot crops.

Script version of ``Real_Flow.ipynb`` (everything upstream of the hls4ml
conversion). All constants live in a YAML config file:

    python train_real.py configs/real.yaml

Pipeline:
    1. Split the growths of the h5 file into train / validation / test and
       sample spot crops from each.
    2. Draw crops with replacement, subtract each crop's minimum as background,
       normalize, and drop crops whose peak is too dim.
    3. Build (or load) the quantized ``gaussian`` model and train it
       self-supervised through the fixed reconstruction layer (no parameter
       labels exist for real data).
    4. Save the parameter model, the config used, the training history and a
       few image / predicted reconstruction plots.

The model, reconstruction layer, loss and training loop are imported from
``train_synthetic.py``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from train_synthetic import (  # noqa: E402
    NUM_PARAMS,
    build_model,
    load_model,
    make_datasets,
    make_gaussian_gen_tf,
    make_recon_loss,
    make_training_model,
    setup_tensorflow,
    train,
)


# ============================================================================
# Config
# ============================================================================


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if len(cfg["params"]["scaling"]) != NUM_PARAMS:
        raise ValueError(f"params.scaling must have {NUM_PARAMS} entries")

    m = cfg["model"]
    if len(m["conv_filters"]) != len(m["pool_sizes"]):
        raise ValueError("model.conv_filters and model.pool_sizes must have the same length")

    return cfg


# ============================================================================
# Real data loading
# ============================================================================


def load_splits(cfg: dict, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Shuffle the growths, split them, and sample spot crops from each split."""
    d = cfg["data"]
    splits = {}

    with h5py.File(d["path"], "r") as h5:
        growths = list(h5.keys())
        rng.shuffle(growths)  # Shuffle Growths

        n_train, n_val, n_test = d["training_growths"], d["validation_growths"], d["test_growths"]
        if n_train + n_val + n_test > len(growths):
            raise ValueError(
                f"Requested {n_train + n_val + n_test} growths but the file has {len(growths)}"
            )
        split_growths = {
            "Train": growths[:n_train],
            "Val": growths[n_train : n_train + n_val],
            "Test": growths[n_train + n_val : n_train + n_val + n_test],
        }

        for name, names in split_growths.items():
            print(f"Raw {name} Data Set:")
            crops = []
            for growth in names:
                for spot in d["spots"]:
                    indices = rng.choice(
                        h5[growth][spot].shape[0], size=d["images_per_growth"], replace=False
                    )
                    indices.sort()
                    crops.append(np.expand_dims(h5[growth][spot][indices], -1).astype(np.float32))
                    print(f"[Growth]: {growth:<12}, [Spot]: {spot}, [Shape]: {crops[-1].shape}")
            splits[name] = np.concatenate(crops)
            print(f"[{name} Data Set Shape]: {splits[name].shape}")

    expected = (cfg["image"]["y"], cfg["image"]["x"], 1)
    for name, data in splits.items():
        if data.shape[1:] != expected:
            raise ValueError(f"{name} crops have shape {data.shape[1:]}, expected {expected} "
                             "from image.y / image.x")
    return splits


def gen_data(
    num_images: int, data: np.ndarray, rng: np.random.Generator, min_peak: float, desc: str = ""
) -> np.ndarray:
    """Sample crops with replacement, background-subtract, normalize and filter."""
    idx = rng.integers(0, data.shape[0], size=num_images)
    imgs = data[idx].astype(np.float32)
    bg = imgs.reshape(num_images, -1).min(axis=1).reshape(num_images, 1, 1, 1)
    imgs = np.clip(imgs - bg, 0.0, None) / 256.0
    print(f"[{desc} Images Shape]: {imgs.shape}")

    # Filter out bad crops: peak must be bright enough to be a real spot
    valid_mask = imgs.reshape(num_images, -1).max(axis=1) >= min_peak
    print(f"[{desc} Valid]: {valid_mask.sum()} / {len(valid_mask)} ({100 * valid_mask.mean():.1f}%)")
    return imgs[valid_mask]


# ============================================================================
# Evaluation
# ============================================================================


def save_plots(out_dir: Path, test_img, pred_recon, predictions, scaling, num_examples: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(min(num_examples, len(test_img))):
        p = predictions[i] * scaling
        theta = 0.5 * np.arctan2(p[4], p[5])

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(test_img[i].squeeze(), cmap="viridis", interpolation="none")
        axes[0].set_title("Test Image")
        axes[1].imshow(pred_recon[i].squeeze(), cmap="viridis", interpolation="none")
        axes[1].set_title("QK Prediction")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(
            f"c=({p[0]:.1f}, {p[1]:.1f})  s=({p[2]:.1f}, {p[3]:.1f})  "
            f"theta={np.degrees(theta):.0f}deg  I={p[6]:.2f}",
            fontsize=9,
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"example_{i}.png", dpi=120)
        plt.close(fig)
    print(f"[Plots]: {out_dir}")


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
    splits = load_splits(cfg, rng)
    d = cfg["dataset"]
    min_peak = cfg["filter"]["min_peak"]
    train_img = gen_data(d["training_size"], splits["Train"], rng, min_peak, desc="Train")
    val_img = gen_data(d["validation_size"], splits["Val"], rng, min_peak, desc="Val")
    test_img = gen_data(d["test_size"], splits["Test"], rng, min_peak, desc="Test")
    del splits

    # ---- Model ----------------------------------------------------------
    tf = setup_tensorflow(cfg["seed"])
    loss_fn = make_recon_loss(cfg["loss"]["bg_weight"])
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

    # ---- Test -----------------------------------------------------------
    predictions = model.predict(test_img, batch_size=d["batch_size"])
    print(f"[Test Images Shape]: {test_img.shape}")
    print(f"[Prediction Shape]: {predictions.shape}")

    pred_recon = gaussian_gen_tf(tf.constant(predictions)).numpy()
    test_loss = float(loss_fn(tf.constant(test_img), tf.constant(pred_recon)).numpy())
    print(f"[Test Loss (recon)]: {test_loss:.6f}")

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
            json.dump({"test_loss_recon": test_loss, "num_test_images": len(test_img)}, f, indent=2)

    if o["num_plot_examples"] > 0:
        plot_dir = (model_dir if o["save_model"] else Path(o["models_dir"])) / "plots"
        scaling = np.array(cfg["params"]["scaling"], dtype=np.float32)
        save_plots(plot_dir, test_img, pred_recon, predictions, scaling, o["num_plot_examples"])


if __name__ == "__main__":
    main()
