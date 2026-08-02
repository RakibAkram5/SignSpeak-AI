"""Keras-based sign language classifier operating on MediaPipe landmark features.

The model is a compact multilayer perceptron (MLP) trained on normalized
126-dim hand-landmark feature vectors (see utils.landmarks_to_feature_vector),
not raw pixels. This keeps the model tiny and CPU-inference fast, which is
what makes real-time 30 FPS webcam prediction feasible without a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import MODEL, PATHS
from utils import setup_logger

logger = setup_logger(__name__)


class SignLanguageModel:
    """Builds, trains, evaluates, and serves predictions for the sign classifier."""

    def __init__(self, num_classes: int, class_names: Optional[list[str]] = None) -> None:
        """
        Args:
            num_classes: Number of distinct sign classes to predict.
            class_names: Ordered list of class labels matching model output
                indices (e.g., ["A", "B", "C", ...]). Required before calling
                predict() with human-readable labels; can be set later via
                load_model().
        """
        self.num_classes = num_classes
        self.class_names = class_names or []
        self.model: Optional[keras.Model] = None

        tf.random.set_seed(MODEL.random_seed)
        np.random.seed(MODEL.random_seed)

    def build_model(self) -> keras.Model:
        """Construct the MLP architecture and compile it.

        Returns:
            The compiled Keras model (also stored on self.model).
        """
        inputs = keras.Input(shape=(MODEL.input_dim,), name="landmark_features")
        x = inputs
        for units in MODEL.hidden_units:
            x = layers.Dense(units, activation="relu")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(MODEL.dropout_rate)(x)

        outputs = layers.Dense(self.num_classes, activation="softmax", name="predictions")(x)
        model = keras.Model(inputs, outputs, name="sign_language_mlp")

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=MODEL.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        self.model = model
        logger.info("Model built: input_dim=%d, classes=%d", MODEL.input_dim, self.num_classes)
        model.summary(print_fn=logger.info)
        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> keras.callbacks.History:
        """Train the model on landmark feature vectors.

        Args:
            X_train: Training features, shape (n_samples, 126).
            y_train: Integer-encoded training labels, shape (n_samples,).
            X_val: Optional validation features. If None, MODEL.validation_split
                is carved out of the training data instead.
            y_val: Optional validation labels, required if X_val is given.

        Returns:
            The Keras training History object (used to plot accuracy/loss).
        """
        if self.model is None:
            self.build_model()

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=MODEL.early_stopping_patience,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
            ),
        ]

        fit_kwargs = dict(
            x=X_train,
            y=y_train,
            batch_size=MODEL.batch_size,
            epochs=MODEL.epochs,
            callbacks=callbacks,
            verbose=2,
        )

        if X_val is not None and y_val is not None:
            fit_kwargs["validation_data"] = (X_val, y_val)
        else:
            fit_kwargs["validation_split"] = MODEL.validation_split

        logger.info("Starting training: %d samples, batch_size=%d, epochs=%d",
                    len(X_train), MODEL.batch_size, MODEL.epochs)
        history = self.model.fit(**fit_kwargs)
        logger.info("Training complete.")
        return history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> tuple[float, float]:
        """Evaluate the model on held-out data.

        Returns:
            (loss, accuracy) tuple.
        """
        if self.model is None:
            raise RuntimeError("Model has not been built or loaded yet.")
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        logger.info("Evaluation — loss: %.4f, accuracy: %.4f", loss, accuracy)
        return loss, accuracy

    def save_model(
        self,
        model_path: Path = PATHS.model_file,
        label_path: Path = PATHS.label_encoder_file,
    ) -> None:
        """Save the trained model and its class-name mapping to disk."""
        if self.model is None:
            raise RuntimeError("No model to save. Call build_model()/train() first.")

        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(model_path)

        with open(label_path, "w", encoding="utf-8") as f:
            json.dump({"class_names": self.class_names}, f, indent=2)

        logger.info("Model saved to %s, labels saved to %s", model_path, label_path)

    def load_model(
        self,
        model_path: Path = PATHS.model_file,
        label_path: Path = PATHS.label_encoder_file,
    ) -> None:
        """Load a previously trained model and its class-name mapping."""
        if not model_path.exists():
            raise FileNotFoundError(f"No trained model found at {model_path}. Run train.py first.")
        if not label_path.exists():
            raise FileNotFoundError(f"No label encoder found at {label_path}. Run train.py first.")

        self.model = keras.models.load_model(model_path)
        with open(label_path, "r", encoding="utf-8") as f:
            self.class_names = json.load(f)["class_names"]
        self.num_classes = len(self.class_names)

        logger.info("Loaded model from %s with %d classes", model_path, self.num_classes)

    def predict(self, feature_vector: np.ndarray) -> tuple[str, float]:
        """Predict the sign class for a single feature vector.

        Args:
            feature_vector: A (126,) or (1, 126) array of landmark features.

        Returns:
            (predicted_label, confidence) where confidence is in [0, 1].
        """
        if self.model is None:
            raise RuntimeError("Model has not been built or loaded yet.")
        if not self.class_names:
            raise RuntimeError("class_names is empty; cannot map prediction index to a label.")

        batch = feature_vector.reshape(1, -1) if feature_vector.ndim == 1 else feature_vector
        probabilities = self.model.predict(batch, verbose=0)[0]

        best_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[best_idx])
        label = self.class_names[best_idx]

        return label, confidence
