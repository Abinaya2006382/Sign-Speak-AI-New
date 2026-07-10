"""
Sign Speak AI — Model Training Pipeline
========================================
Run this script from the project root AFTER collecting data with data_collection.py:
    python ai_engine/train_model.py

What it does:
  1. Loads .npy landmark files from ai_engine/dataset/
  2. Trains a TensorFlow/Keras Dense classifier
  3. Prints accuracy + loss curves (and saves a plot)
  4. Saves confusion matrix to ai_engine/model/confusion_matrix.png
  5. Saves trained model to ai_engine/model/gesture_classifier.keras
  6. Updates ai_engine/model/classes.txt
"""

import os
import sys

# Suppress TensorFlow C++ logging noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)
import matplotlib
matplotlib.use("Agg")           # headless backend — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR  = os.path.join(PROJECT_ROOT, "ai_engine", "dataset")
MODEL_DIR    = os.path.join(PROJECT_ROOT, "ai_engine", "model")
MODEL_PATH   = os.path.join(MODEL_DIR, "gesture_classifier.keras")
CLASSES_PATH = os.path.join(MODEL_DIR, "classes.txt")
CM_PATH      = os.path.join(MODEL_DIR, "confusion_matrix.png")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.png")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Training hyper-parameters ─────────────────────────────────────────────────
EPOCHS          = 200
BATCH_SIZE      = 32
LEARNING_RATE   = 1e-3
VAL_SPLIT       = 0.20
PATIENCE        = 20          # early-stopping patience
MIN_SAMPLES_WARN = 50         # warn if a class has fewer than this many samples

# ── 1. Load dataset ───────────────────────────────────────────────────────────
def load_dataset():
    """Scans DATASET_DIR for .npy files and returns (X, y, class_names)."""
    if not os.path.isdir(DATASET_DIR):
        print(f"\n❌ Dataset directory not found: {DATASET_DIR}")
        print("   Run  python ai_engine/data_collection.py  first.\n")
        sys.exit(1)

    npy_files = sorted(
        f for f in os.listdir(DATASET_DIR) if f.endswith(".npy")
    )
    if not npy_files:
        print(f"\n❌ No .npy files found in {DATASET_DIR}")
        print("   Run  python ai_engine/data_collection.py  first.\n")
        sys.exit(1)

    X_parts, y_parts = [], []
    class_names = []

    print("\n" + "═" * 60)
    print("  Loading dataset")
    print("═" * 60)

    for idx, fname in enumerate(npy_files):
        class_name = os.path.splitext(fname)[0]
        fpath      = os.path.join(DATASET_DIR, fname)
        data       = np.load(fpath).astype(np.float32)

        if data.ndim != 2 or data.shape[1] != 63:
            print(f"  ⚠ Skipping '{fname}': expected shape (N, 63), got {data.shape}")
            continue

        n = len(data)
        flag = "⚠ low" if n < MIN_SAMPLES_WARN else "✓"
        print(f"  {flag:6s}  {class_name:20s}  {n} samples")

        class_names.append(class_name)
        X_parts.append(data)
        y_parts.append(np.full(n, idx, dtype=np.int32))

    if len(class_names) < 2:
        print("\n❌ Need at least 2 gesture classes to train.\n")
        sys.exit(1)

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    print(f"\n  Total samples : {len(X)}")
    print(f"  Classes       : {len(class_names)}")
    print("═" * 60)
    return X, y, class_names

# ── 2. Build Keras model ──────────────────────────────────────────────────────
def build_model(num_classes: int) -> keras.Model:
    """
    Builds a deep Dense classifier suited for 63-dimensional hand landmark input.
    Architecture:
      Input(63) → Dense(256) → BN → Dropout(0.4)
               → Dense(128) → BN → Dropout(0.3)
               → Dense(64)  → BN → Dropout(0.2)
               → Dense(num_classes, softmax)
    """
    inp = keras.Input(shape=(63,), name="landmarks")

    x = layers.Dense(256, activation="relu", name="dense_1")(inp)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(0.4, name="drop_1")(x)

    x = layers.Dense(128, activation="relu", name="dense_2")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Dropout(0.3, name="drop_2")(x)

    x = layers.Dense(64, activation="relu", name="dense_3")(x)
    x = layers.BatchNormalization(name="bn_3")(x)
    x = layers.Dropout(0.2, name="drop_3")(x)

    out = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inp, outputs=out, name="GestureClassifier")
    return model

# ── 3. Plot training history ──────────────────────────────────────────────────
def plot_history(history, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Sign Speak AI — Training History", fontsize=13, fontweight="bold")

    # Accuracy
    axes[0].plot(history.history["accuracy"],     label="Train",      color="#4A90E2", lw=2)
    axes[0].plot(history.history["val_accuracy"], label="Validation",  color="#E94F37", lw=2)
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)
    axes[0].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"],     label="Train",      color="#4A90E2", lw=2)
    axes[1].plot(history.history["val_loss"], label="Validation",  color="#E94F37", lw=2)
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Training history plot saved → {save_path}")

# ── 4. Plot confusion matrix ──────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, class_names: list, save_path: str):
    cm = confusion_matrix(y_true, y_pred)
    # Normalise row-wise so each cell shows the recall fraction (0→1)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig_h = max(7, len(class_names) * 0.65)
    fig, ax = plt.subplots(figsize=(fig_h + 1, fig_h))

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".0%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="lightgray",
        ax=ax,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Recall fraction"},
    )
    ax.set_title("Confusion Matrix (normalised by true class)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  🗺  Confusion matrix saved      → {save_path}")

# ── 5. Main training routine ──────────────────────────────────────────────────
def train():
    # ── Load ──────────────────────────────────────────────────────────────────
    X, y, class_names = load_dataset()
    num_classes = len(class_names)

    # ── Split ─────────────────────────────────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=VAL_SPLIT,
        random_state=42,
        stratify=y,
    )
    print(f"\n  Train : {len(X_train)} samples")
    print(f"  Val   : {len(X_val)} samples\n")

    # ── Class weights (handle imbalanced collections) ─────────────────────────
    cw_array  = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    cw_dict   = {i: w for i, w in enumerate(cw_array)}

    # ── Build model ───────────────────────────────────────────────────────────
    model = build_model(num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=0,
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Training — press Ctrl+C to stop early")
    print("═" * 60 + "\n")

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        class_weight=cw_dict,
        callbacks=cb_list,
        verbose=1,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Evaluation on validation set")
    print("═" * 60)

    y_pred_proba = model.predict(X_val, verbose=0)
    y_pred       = np.argmax(y_pred_proba, axis=1)

    val_acc = accuracy_score(y_val, y_pred)
    print(f"\n  Validation accuracy: {val_acc * 100:.2f}%\n")
    print(classification_report(y_val, y_pred, target_names=class_names))

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_history(history, HISTORY_PATH)
    plot_confusion_matrix(y_val, y_pred, class_names, CM_PATH)

    # ── Save classes.txt ──────────────────────────────────────────────────────
    with open(CLASSES_PATH, "w") as f:
        f.write("\n".join(class_names))
    print(f"  📝 classes.txt saved           → {CLASSES_PATH}")

    # Model already saved via ModelCheckpoint; print confirmation
    print(f"  ✅ Best model saved            → {MODEL_PATH}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print(f"  Training complete!")
    print(f"  Classes   : {num_classes}")
    print(f"  Val acc   : {val_acc * 100:.1f}%")
    print(f"  Model     : {MODEL_PATH}")
    print("═" * 60)
    print("\nNext step: start the app with  python backend/app.py\n")


if __name__ == "__main__":
    train()
