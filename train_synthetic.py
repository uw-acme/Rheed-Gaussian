"""Train the Gaussian-fitting QKeras model on synthetic RHEED spot crops.

Script version of ``Generated_Flow.ipynb`` (everything upstream of the hls4ml
conversion). All constants live in a YAML config file:

    python train_synthetic.py configs/synthetic.yaml

Pipeline:
    1. Generate train / validation / test sets of single rotated 2D-Gaussian
       crops and their normalized 7-parameter labels.
    2. Build (or load) the quantized ``gaussian`` model, which predicts the
       normalized parameters [cx, cy, sx, sy, sin2t, cos2t, intensity].
    3. Train it self-supervised: a fixed reconstruction layer renders the
       predicted parameters back into an image, and a weighted MSE compares it
       to the input. Early stopping and LR reduction as in the notebook.
    4. Save the parameter model, the config used, the training history and a
       few image / true / predicted reconstruction plots.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


NUM_PARAMS = 7  # [cx, cy, sx, sy, sin2t, cos2t, intensity]


# ============================================================================
# Config
# ============================================================================


def parse_angle(value) -> float:
    """Accept a number or a string like ``pi``, ``pi/2`` or ``2*pi``."""
    if isinstance(value, str):
        return float(eval(value, {"__builtins__": {}}, {"pi": np.pi}))
    return float(value)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    g = cfg["gaussian"]
    g["min_theta"] = parse_angle(g["min_theta"])
    g["max_theta"] = parse_angle(g["max_theta"])

    if len(cfg["params"]["scaling"]) != NUM_PARAMS:
        raise ValueError(f"params.scaling must have {NUM_PARAMS} entries")

    m = cfg["model"]
    if len(m["conv_filters"]) != len(m["pool_sizes"]):
        raise ValueError("model.conv_filters and model.pool_sizes must have the same length")

    return cfg


# ============================================================================
# Synthetic data generation
# ============================================================================


class SyntheticGenerator:
    """Generates crops containing one rotated 2D Gaussian plus its parameters."""

    def __init__(self, cfg: dict, rng: np.random.Generator):
        self.rng = rng
        self.image_x = cfg["image"]["x"]
        self.image_y = cfg["image"]["y"]
        self.g = cfg["gaussian"]
        self.scaling = np.array(cfg["params"]["scaling"], dtype=np.float32)

        # Pixel grid reused by every gaussian_gen call
        self.X, self.Y = np.meshgrid(np.arange(self.image_x), np.arange(self.image_y))

    def gaussian_gen(
        self, center_x: float, center_y: float, std_x: float, std_y: float, theta: float
    ) -> np.ndarray:
        cos_theta_sqrd = np.cos(theta) ** 2
        sin_theta_sqrd = np.sin(theta) ** 2
        sin_cos_theta = np.sin(theta) * np.cos(theta)

        std_x_sqrd = std_x**2
        std_y_sqrd = std_y**2

        a = cos_theta_sqrd / (2 * std_x_sqrd) + sin_theta_sqrd / (2 * std_y_sqrd)
        b = -sin_cos_theta / (2 * std_x_sqrd) + sin_cos_theta / (2 * std_y_sqrd)
        c = sin_theta_sqrd / (2 * std_x_sqrd) + cos_theta_sqrd / (2 * std_y_sqrd)

        X, Y = self.X, self.Y
        gaussian = np.exp(
            -(
                a * (X - center_x) ** 2
                + 2 * b * (X - center_x) * (Y - center_y)
                + c * (Y - center_y) ** 2
            )
        )
        return np.expand_dims(gaussian, -1)

    def img_gen(self) -> tuple[np.ndarray, np.ndarray]:
        g, rng = self.g, self.rng

        center_x = self.image_x / 2 + rng.uniform(g["min_offset_x"], g["max_offset_x"])
        center_y = self.image_y / 2 + rng.uniform(g["min_offset_y"], g["max_offset_y"])

        std_x = rng.uniform(g["min_std_x"], g["max_std_x"])
        std_y = rng.uniform(g["min_std_y"], g["max_std_y"])
        theta = rng.uniform(g["min_theta"], g["max_theta"])

        intensity = g["min_intensity"] + rng.random() * (
            g["max_intensity"] - g["min_intensity"]
        )

        # Convert to 8 bit int
        gaussian = self.gaussian_gen(center_x, center_y, std_x, std_y, theta)
        gaussian = (gaussian * intensity * 255).astype(np.uint8)

        params = np.array(
            [
                center_x,
                center_y,
                std_x,
                std_y,
                np.sin(2 * theta),
                np.cos(2 * theta),
                intensity,
            ]
        )
        return gaussian, params

    def gen_data(self, num_images: int, desc: str = "") -> tuple[np.ndarray, np.ndarray]:
        img_arr = np.empty((num_images, self.image_y, self.image_x, 1), dtype=np.float32)
        label_arr = np.empty((num_images, NUM_PARAMS), dtype=np.float32)

        for i in tqdm(range(num_images), desc=desc):
            img, params = self.img_gen()
            img_arr[i] = img.astype(np.float32) / 256.0  # Normalize values to [0, 1)
            label_arr[i] = params.astype(np.float32) / self.scaling  # Normalize with scaling array

        print(f"[{desc} Images Shape]: {img_arr.shape}")
        print(f"[{desc} Labels Shape]: {label_arr.shape}")

        return img_arr, label_arr


# ============================================================================
# Model / loss
# ============================================================================


def setup_tensorflow(seed: int):
    import tensorflow as tf

    tf.random.set_seed(seed)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[Device]: GPU ({len(gpus)} found, CUDA build: {tf.test.is_built_with_cuda()})")
    else:
        print("[Device]: CPU (no CUDA GPU visible to TensorFlow)")

    return tf


def build_model(cfg: dict):
    from tensorflow.keras.layers import (
        Activation,
        BatchNormalization,
        Concatenate,
        Dense,
        Flatten,
        Input,
        MaxPool2D,
    )
    from tensorflow.keras.models import Model
    from qkeras import QActivation, QConv2D
    from qkeras.quantizers import quantized_bits, quantized_relu

    m = cfg["model"]
    total_bits, integer_bits = m["total_bits"], m["integer_bits"]

    w_quant = quantized_bits(
        total_bits, integer_bits, symmetric=False, keep_negative=True, alpha=1
    )
    relu_quant = quantized_relu(total_bits, integer_bits)

    input_layer = Input(shape=(cfg["image"]["y"], cfg["image"]["x"], 1), name="InputLayer")

    x = input_layer
    for filters, pool_size in zip(m["conv_filters"], m["pool_sizes"]):
        x = QConv2D(
            filters=filters,
            kernel_size=3,
            strides=1,
            padding="valid",
            kernel_quantizer=w_quant,
            kernel_initializer="lecun_uniform",
        )(x)
        x = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-03)(x)
        x = QActivation(relu_quant)(x)
        x = MaxPool2D(pool_size=pool_size, strides=pool_size)(x)

    if min(x.shape[1:3]) < 1:
        raise ValueError(
            f"Feature map collapsed to {tuple(x.shape[1:3])}; reduce model.pool_sizes "
            "or the number of conv blocks for this image size."
        )

    x = Flatten(name="flatten")(x)
    x = Dense(units=m["dense_units"], activation="relu")(x)

    # Split head to constrain outputs
    x_center = Activation("sigmoid")(Dense(2)(x))
    x_std = Activation("softplus")(Dense(2)(x))
    x_theta = Activation("tanh")(Dense(2)(x))
    x_intensity = Activation("sigmoid")(Dense(1)(x))

    x_p1 = Concatenate(axis=-1, name="params_2")([x_center, x_std])
    x_p2 = Concatenate(axis=-1, name="params_3")([x_theta, x_intensity])
    x_params = Concatenate(axis=-1, name="params")([x_p1, x_p2])  # 7-dim

    return Model(inputs=input_layer, outputs=x_params, name=m["name"])


def make_gaussian_gen_tf(cfg: dict):
    """Return a TF function rendering normalized parameters into images."""
    import tensorflow as tf

    scaling_tf = tf.constant(cfg["params"]["scaling"], dtype=tf.float32)
    X_mesh_tf, Y_mesh_tf = tf.meshgrid(
        tf.range(cfg["image"]["x"], dtype=tf.float32),
        tf.range(cfg["image"]["y"], dtype=tf.float32),
    )
    X_mesh_tf = X_mesh_tf[tf.newaxis, :, :, tf.newaxis]
    Y_mesh_tf = Y_mesh_tf[tf.newaxis, :, :, tf.newaxis]

    def gaussian_gen_tf(params_norm):
        # params_norm ordering: [cx, cy, sx, sy, sin2t, cos2t, intensity]
        params = params_norm * scaling_tf
        cx = tf.reshape(params[:, 0], (-1, 1, 1, 1))
        cy = tf.reshape(params[:, 1], (-1, 1, 1, 1))
        sx = tf.reshape(params[:, 2], (-1, 1, 1, 1))
        sy = tf.reshape(params[:, 3], (-1, 1, 1, 1))
        s2t = tf.reshape(params[:, 4], (-1, 1, 1, 1))
        c2t = tf.reshape(params[:, 5], (-1, 1, 1, 1))
        ity = tf.reshape(params[:, 6], (-1, 1, 1, 1))

        norm = tf.sqrt(tf.square(s2t) + tf.square(c2t) + 1e-6)
        s2t = s2t / norm
        c2t = c2t / norm

        cos_sqr = 0.5 * (1.0 + c2t)
        sin_sqr = 0.5 * (1.0 - c2t)
        sin_cos = 0.5 * s2t

        std_x_sqrd = tf.square(sx) + 1e-6
        std_y_sqrd = tf.square(sy) + 1e-6

        a = cos_sqr / (2.0 * std_x_sqrd) + sin_sqr / (2.0 * std_y_sqrd)
        b = -sin_cos / (2.0 * std_x_sqrd) + sin_cos / (2.0 * std_y_sqrd)
        c = sin_sqr / (2.0 * std_x_sqrd) + cos_sqr / (2.0 * std_y_sqrd)

        dx = X_mesh_tf - cx
        dy = Y_mesh_tf - cy
        return ity * tf.exp(-(a * tf.square(dx) + 2.0 * b * dx * dy + c * tf.square(dy)))

    return gaussian_gen_tf


def make_training_model(model, gaussian_gen_tf):
    """Wrap the parameter model with the fixed reconstruction layer."""
    import tensorflow as tf
    from tensorflow.keras.models import Model

    x_reconstruction = tf.keras.layers.Lambda(gaussian_gen_tf, name="reconstruction")(
        model.output
    )
    return Model(inputs=model.input, outputs=x_reconstruction, name=f"{model.name}_training")


def make_recon_loss(bg_weight: float):
    import tensorflow as tf

    def recon_loss(y_true, y_pred):
        W = y_true + bg_weight
        return tf.reduce_mean(tf.reduce_sum(W * tf.square(y_true - y_pred), axis=[1, 2, 3]))

    return recon_loss


def load_model(path: str):
    import tensorflow as tf
    from qkeras.utils import _add_supported_quantized_objects

    custom_objects = {}
    _add_supported_quantized_objects(custom_objects)
    print(f"[Load]: {path}")
    # Only the parameter model is saved (uncompiled); the loss lives on the training wrapper.
    return tf.keras.models.load_model(path, custom_objects=custom_objects, compile=False)


# ============================================================================
# Training / evaluation
# ============================================================================


def make_datasets(tf, cfg, train_img, val_img):
    batch_size = cfg["dataset"]["batch_size"]

    # Self-supervised: the target is the input image itself.
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((train_img, train_img))
        .shuffle(len(train_img), reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_dataset = (
        tf.data.Dataset.from_tensor_slices((val_img, val_img))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_dataset, val_dataset


def train(tf, cfg, training_model, loss_fn, train_dataset, val_dataset):
    t = cfg["training"]

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=t["learning_rate"],
        global_clipnorm=t["global_clipnorm"],
    )
    training_model.compile(optimizer=optimizer, loss=loss_fn)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=t["early_stopping"]["patience"],
        min_delta=t["early_stopping"]["min_delta"],
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=t["reduce_lr"]["factor"],
        patience=t["reduce_lr"]["patience"],
        min_lr=t["reduce_lr"]["min_lr"],
        verbose=1,
    )

    history = training_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=t["epochs"],
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )
    return history


def save_plots(out_dir: Path, test_img, true_recon, pred_recon, num_examples: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(min(num_examples, len(test_img))):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(test_img[i].squeeze(), cmap="viridis", interpolation="none")
        axes[0].set_title("Test Image")
        axes[1].imshow(true_recon[i].squeeze(), cmap="viridis", interpolation="none")
        axes[1].set_title("Label Reconstruction")
        axes[2].imshow(pred_recon[i].squeeze(), cmap="viridis", interpolation="none")
        axes[2].set_title("QK Prediction")
        for ax in axes:
            ax.axis("off")
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
    gen = SyntheticGenerator(cfg, rng)
    d = cfg["dataset"]
    train_img, _ = gen.gen_data(d["training_size"], desc="Train")
    val_img, _ = gen.gen_data(d["validation_size"], desc="Val")
    test_img, test_label = gen.gen_data(d["test_size"], desc="Test")

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
    print(f"[Test Shape]: {test_label.shape}")
    print(f"[Prediction Shape]: {predictions.shape}")

    pred_recon = gaussian_gen_tf(tf.constant(predictions)).numpy()
    true_recon = gaussian_gen_tf(tf.constant(test_label)).numpy()
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
            json.dump({"test_loss_recon": test_loss}, f, indent=2)

    if o["num_plot_examples"] > 0:
        plot_dir = (model_dir if o["save_model"] else Path(o["models_dir"])) / "plots"
        save_plots(plot_dir, test_img, true_recon, pred_recon, o["num_plot_examples"])


if __name__ == "__main__":
    main()
