"""Real-time inference layer: loads the trained model and smooths raw
per-frame predictions into stable, debounced letter commits.

SignPredictor: thin wrapper that auto-loads the saved model/label encoder
and exposes a single predict() call for one feature vector.

StablePredictor: temporal smoothing on top of SignPredictor's raw output.
Implements the "avoid duplicate letters" and "configurable delay between
repeated predictions" requirements via a majority-vote buffer + cooldown.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from typing import Optional

import numpy as np

from config import PATHS, PREDICTION
from model import SignLanguageModel
from utils import setup_logger

logger = setup_logger(__name__)


class SignPredictor:
    """Loads the trained SignLanguageModel and predicts one frame at a time."""

    def __init__(self) -> None:
        """Auto-loads the model and label encoder from config.PATHS.

        Raises:
            FileNotFoundError: If no trained model exists yet (run train.py first).
        """
        self._model = SignLanguageModel(num_classes=0)
        self._model.load_model(PATHS.model_file, PATHS.label_encoder_file)
        logger.info("SignPredictor ready with classes: %s", self._model.class_names)

    @property
    def class_names(self) -> list[str]:
        return self._model.class_names

    def predict(self, feature_vector: np.ndarray) -> tuple[str, float]:
        """Predict the sign label for a single feature vector.

        Returns:
            (label, confidence) — confidence in [0, 1].
        """
        return self._model.predict(feature_vector)


class StablePredictor:
    """Smooths raw per-frame predictions into debounced, stable letter commits."""

    def __init__(
        self,
        stability_frames: int = PREDICTION.stability_frames,
        confidence_threshold: float = PREDICTION.confidence_threshold,
        cooldown_sec: float = PREDICTION.prediction_cooldown_sec,
        history_length: int = PREDICTION.history_length,
    ) -> None:
        """
        Args:
            stability_frames: Consecutive/majority frames required before a
                prediction is considered "stable" enough to display/commit.
            confidence_threshold: Minimum model confidence to even count a
                frame's prediction toward the stability vote.
            cooldown_sec: Minimum seconds between two commits of the same
                letter, preventing rapid duplicate letters from a held pose.
            history_length: Size of the rolling buffer used for the majority vote.
        """
        self._stability_frames = stability_frames
        self._confidence_threshold = confidence_threshold
        self._cooldown_sec = cooldown_sec

        self._history: deque[str] = deque(maxlen=history_length)
        self._last_committed_label: Optional[str] = None
        self._last_commit_time: float = 0.0

    def update(self, label: Optional[str], confidence: float) -> tuple[Optional[str], float, bool]:
        """Feed one frame's raw prediction into the smoothing buffer.

        Args:
            label: The raw predicted label this frame, or None if no hand
                was detected.
            confidence: The raw model confidence for that label.

        Returns:
            A tuple (stable_label, stable_confidence, should_commit):
                stable_label: The current majority-vote label to display
                    (may repeat across frames; None if no stable majority yet).
                stable_confidence: Confidence associated with the raw
                    prediction that produced this frame's update.
                should_commit: True exactly on the frame where this stable
                    label newly satisfies the stability + cooldown rules —
                    the signal the caller should use to append a letter to
                    a word/sentence buffer.
        """
        if label is not None and confidence >= self._confidence_threshold:
            self._history.append(label)
        else:
            self._history.clear()

        if not self._history:
            return None, 0.0, False

        counts = Counter(self._history)
        candidate_label, count = counts.most_common(1)[0]

        if count < self._stability_frames:
            return candidate_label, confidence, False

        now = time.monotonic()
        cooldown_elapsed = (now - self._last_commit_time) >= self._cooldown_sec
        is_new_label = candidate_label != self._last_committed_label

        if is_new_label or cooldown_elapsed:
            self._last_committed_label = candidate_label
            self._last_commit_time = now
            return candidate_label, confidence, True

        return candidate_label, confidence, False

    def reset(self) -> None:
        """Clear the smoothing buffer (e.g., after the sentence is cleared)."""
        self._history.clear()
        self._last_committed_label = None
        self._last_commit_time = 0.0
