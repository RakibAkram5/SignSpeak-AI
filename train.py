"""Training pipeline entry point.

Usage:
    python train.py

Loads dataset/train (and dataset/validation if present), extracts MediaPipe
landmark features, trains the SignLanguageModel MLP, evaluates it, saves
accuracy/loss plots, and persists the trained model + label encoder to
models/.
"""

from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")  # Headless-safe backend; plots are saved to disk, not shown interactively.
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from config import MODEL, PATHS
from dataset import DatasetSplit, load_dataset
from model import SignLanguageModel
from utils import setup_logger

logger = setup_logger(__name__)


def plot_history(history, accuracy_path=PATHS.accuracy_plot_file, loss_path=PATHS.loss_plot_file) -> None:
    """Save accuracy and loss curves from a Keras History object."""
    epochs = range(1, len(history.history["loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.history["accuracy"], label="Train Accuracy")
    plt.plot(epochs, history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(accuracy_path)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.history["loss"], label="Train Loss")
    plt.plot(epochs, history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_path)
    plt.close()

    logger.info("Saved accuracy plot to %s and loss plot to %s", accuracy_path, loss_path)


def main() -> int:
    """Run the full training pipeline. Returns a process exit code."""
    logger.info("Loading dataset...")
    try:
        train_split, val_split = load_dataset()
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1

    if len(train_split.X) < 20:
        logger.error(
            "Only %d usable training samples found. Add more images per class "
            "(50-100+ recommended) before training.",
            len(train_split.X),
        )
        return 1

    class_names = train_split.class_names

    if val_split is not None and len(val_split.X) > 0:
        X_train, y_train = train_split.X, train_split.y
        X_val, y_val = val_split.X, val_split.y
        # Carve a small held-out test set from the training data for a final,
        # unbiased accuracy number (validation data was already used to steer
        # early stopping, so it's not a clean test set).
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.1, random_state=MODEL.random_seed, stratify=y_train,
        )
    else:
        X_temp, X_test, y_temp, y_test = train_test_split(
            train_split.X, train_split.y,
            test_size=0.1, random_state=MODEL.random_seed, stratify=train_split.y,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=MODEL.validation_split, random_state=MODEL.random_seed, stratify=y_temp,
        )

    logger.info(
        "Dataset ready — train: %d, validation: %d, test: %d, classes: %d",
        len(X_train), len(X_val), len(X_test), len(class_names),
    )

    sign_model = SignLanguageModel(num_classes=len(class_names), class_names=class_names)
    sign_model.build_model()
    history = sign_model.train(X_train, y_train, X_val, y_val)

    plot_history(history)

    loss, accuracy = sign_model.evaluate(X_test, y_test)
    logger.info("Final held-out test accuracy: %.2f%% (loss: %.4f)", accuracy * 100, loss)

    sign_model.save_model()
    logger.info("Training pipeline complete. Model ready for predict.py / app.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
