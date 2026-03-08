"""
Fruit Classifier — Final Model Trainer
Author : Jin Somaang (67011125)

Everything considered:
1. Balanced dataset    — exactly 500 train / 100 val per class
2. Class weights       — extra insurance against imbalance
3. Random backgrounds  — replaces white bg so model works in real world
4. Strong augmentation — rotation, zoom, brightness, contrast, translation
5. IMG_SIZE = 128      — matches fruit_classifier.py exactly
6. Two-phase training  — head first, then fine-tune top 50 layers
7. Best model saved    — only saves when val_accuracy improves
"""

import os
import shutil
import zipfile
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ── MUST match fruit_classifier.py exactly ────────────────────
CLASSES     = ["apple", "banana", "blueberry", "guava", "orange", "pear", "strawberry"]
IMG_SIZE    = 128
BATCH_SIZE  = 32
DATASET_DIR = "fruits_dataset"
MODEL_OUT   = "fruit_model.keras"
MAX_TRAIN   = 500
MAX_VAL     = 100


# ──────────────────────────────────────────────
# Dataset download + balanced copy
# ──────────────────────────────────────────────
def find_training_folder(base: str) -> str:
    for root, dirs, _ in os.walk(base):
        if "Training" in dirs and "Test" in dirs:
            return root
    return None


def download_dataset():
    if os.path.exists(DATASET_DIR):
        print(f"'{DATASET_DIR}' already exists — skipping download.\n")
        for split in ["train", "val"]:
            for cls in CLASSES:
                p = os.path.join(DATASET_DIR, split, cls)
                n = len(os.listdir(p)) if os.path.exists(p) else 0
                print(f"  {split}/{cls:<12} {n}")
        return

    print("Downloading Fruits-360 from Kaggle...")
    os.system("kaggle datasets download -d moltean/fruits")
    if not os.path.exists("fruits.zip"):
        raise FileNotFoundError("fruits.zip not found — download failed.")

    print("Extracting...")
    with zipfile.ZipFile("fruits.zip", "r") as z:
        z.extractall("fruits_raw")

    raw_base = find_training_folder("fruits_raw")
    if raw_base is None:
        raise RuntimeError("Cannot find Training folder inside zip.")

    print(f"Found dataset at: {raw_base}\n")
    limits = {"train": MAX_TRAIN, "val": MAX_VAL}

    for split, raw_sub in [("train", "Training"), ("val", "Test")]:
        raw = os.path.join(raw_base, raw_sub)
        for cls in CLASSES:
            dst = os.path.join(DATASET_DIR, split, cls)
            os.makedirs(dst, exist_ok=True)

            all_files = []
            for folder in os.listdir(raw):
                if folder.lower().startswith(cls):
                    src = os.path.join(raw, folder)
                    if os.path.isdir(src):
                        for fname in os.listdir(src):
                            all_files.append(os.path.join(src, fname))

            random.seed(42)
            random.shuffle(all_files)
            chosen = all_files[:limits[split]]

            for fpath in chosen:
                shutil.copy(fpath, os.path.join(dst, os.path.basename(fpath)))

            print(f"  {split}/{cls:<12} {len(chosen)} images")

    shutil.rmtree("fruits_raw")
    print("\nDataset balanced and ready!")


# ──────────────────────────────────────────────
# Random background augmentation layer
# Fruits-360 has white backgrounds.
# This replaces white pixels with random solid
# colours so the model learns real-world scenes.
# ──────────────────────────────────────────────
class RandomBackground(layers.Layer):
    def call(self, images, training=None):
        if not training:
            return images
        bs        = tf.shape(images)[0]
        h         = tf.shape(images)[1]
        w         = tf.shape(images)[2]
        bg_colour = tf.random.uniform((bs, 1, 1, 3), 0.0, 255.0)
        bg        = tf.tile(bg_colour, [1, h, w, 1])
        is_white  = tf.reduce_all(images > 200, axis=-1, keepdims=True)
        is_white  = tf.cast(is_white, tf.float32)
        result    = images * (1.0 - is_white) + bg * is_white
        return tf.clip_by_value(result, 0.0, 255.0)


# ──────────────────────────────────────────────
# Data pipelines
# ──────────────────────────────────────────────
def build_datasets():
    augment = tf.keras.Sequential([
        RandomBackground(),
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.35),
        layers.RandomZoom(0.30),
        layers.RandomBrightness(0.40),
        layers.RandomContrast(0.40),
        layers.RandomTranslation(0.15, 0.15),
    ])

    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_DIR, "train"),
        class_names=CLASSES,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42,
    ).map(lambda x, y: (augment(x, training=True), y),
          num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_DIR, "val"),
        class_names=CLASSES,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
    ).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


# ──────────────────────────────────────────────
# Class weights — extra insurance against bias
# ──────────────────────────────────────────────
def compute_class_weights() -> dict:
    counts = []
    for cls in CLASSES:
        p = os.path.join(DATASET_DIR, "train", cls)
        counts.append(len(os.listdir(p)))
    total  = sum(counts)
    n_cls  = len(CLASSES)
    weights = {i: total / (n_cls * c) for i, c in enumerate(counts)}
    print("\nClass weights:")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:<12} {weights[i]:.3f}")
    return weights


# ──────────────────────────────────────────────
# Model — MobileNetV2 + custom head
# ──────────────────────────────────────────────
def build_model():
    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs  = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x       = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x       = base(x, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dropout(0.4)(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Fruit Classifier — Final Trainer")
    print(f"  Classes  : {CLASSES}")
    print(f"  Per class: {MAX_TRAIN} train / {MAX_VAL} val")
    print(f"  IMG_SIZE : {IMG_SIZE}x{IMG_SIZE}  (matches classifier)")
    print("=" * 55 + "\n")

    download_dataset()
    class_weights = compute_class_weights()
    train_ds, val_ds = build_datasets()
    model, base = build_model()

    callbacks_phase1 = [
        ModelCheckpoint(MODEL_OUT, save_best_only=True,
                        monitor="val_accuracy", verbose=1),
        EarlyStopping(patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=3, verbose=1),
    ]

    # Phase 1 — train head only
    print("\nPhase 1 — Training classifier head (base frozen)...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        class_weight=class_weights,
        callbacks=callbacks_phase1,
    )

    # Phase 2 — unfreeze top 50 layers and fine-tune
    print("\nPhase 2 — Fine-tuning top 50 layers of MobileNetV2...")
    base.trainable = True
    for layer in base.layers[:-50]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(2e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        class_weight=class_weights,
        callbacks=[
            ModelCheckpoint(MODEL_OUT, save_best_only=True,
                            monitor="val_accuracy", verbose=1),
            EarlyStopping(patience=6, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=3, verbose=1),
        ],
    )

    model.save(MODEL_OUT)
    print(f"\nDone! Model saved to: {MODEL_OUT}")
    print(f"IMG_SIZE used: {IMG_SIZE} — make sure fruit_classifier.py matches!")
    print("Now run:  python fruit_classifier.py <image_path>")
