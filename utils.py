"""Shared utility helpers: logging, FPS tracking, landmark normalization,
prediction logging, and screenshot/sentence persistence.

Centralizing the landmark-normalization function here (rather than
duplicating it in dataset.py and app.py) guarantees training and inference
see features computed the exact same way.
"""

from __future__ import annotations

import csv
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from config import HANDS, LOG_FORMAT, LOG_LEVEL, PATHS, UI


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logger(name: str) -> logging.Logger:
    """Create (or fetch) a logger that writes to both console and app.log.

    Args:
        name: Usually ``__name__`` of the calling module.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured (avoids duplicate handlers on repeated calls).
        return logger

    logger.setLevel(LOG_LEVEL)
    formatter = logging.Formatter(LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(PATHS.app_log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


# --------------------------------------------------------------------------- #
# FPS counter
# --------------------------------------------------------------------------- #
class FPSCounter:
    """Exponentially-smoothed FPS counter for on-screen display."""

    def __init__(self, smoothing: float = 0.9) -> None:
        """
        Args:
            smoothing: Weight given to the historical FPS value (0-1).
                Higher = smoother but slower to react to changes.
        """
        self._smoothing = smoothing
        self._prev_time: Optional[float] = None
        self._fps: float = 0.0

    def update(self) -> float:
        """Call once per frame. Returns the current smoothed FPS."""
        now = time.perf_counter()
        if self._prev_time is not None:
            instantaneous = 1.0 / max(now - self._prev_time, 1e-6)
            self._fps = (
                instantaneous
                if self._fps == 0.0
                else self._smoothing * self._fps + (1 - self._smoothing) * instantaneous
            )
        self._prev_time = now
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps


# --------------------------------------------------------------------------- #
# Landmark normalization (shared by dataset.py and app.py/predict.py)
# --------------------------------------------------------------------------- #
def normalize_single_hand(landmarks_xyz: np.ndarray) -> np.ndarray:
    """Normalize one hand's 21x3 landmark array to be translation/scale invariant.

    Args:
        landmarks_xyz: Array of shape (21, 3) with (x, y, z) per landmark,
            in MediaPipe's normalized image coordinates.

    Returns:
        Flattened array of shape (63,) — translated to the wrist and scaled
        by the wrist-to-middle-MCP distance.
    """
    wrist = landmarks_xyz[0]
    translated = landmarks_xyz - wrist

    middle_mcp = translated[9]
    scale = np.linalg.norm(middle_mcp)
    scale = scale if scale > 1e-6 else 1.0

    normalized = translated / scale
    return normalized.flatten().astype(np.float32)


def landmarks_to_feature_vector(
    hands_landmarks: Sequence[np.ndarray],
    handedness_labels: Sequence[str],
) -> np.ndarray:
    """Pack one or two detected hands into a fixed-size 126-dim feature vector.

    Order is always [Left(63), Right(63)] regardless of detection order, so
    the model sees a consistent feature layout. A missing hand is zero-padded.

    Args:
        hands_landmarks: List of (21, 3) arrays, one per detected hand.
        handedness_labels: Parallel list of "Left"/"Right" labels from MediaPipe.

    Returns:
        A (126,) float32 feature vector.
    """
    slot_dim = HANDS.num_landmarks * HANDS.num_coords_per_landmark
    vector = np.zeros(slot_dim * 2, dtype=np.float32)

    for landmarks, label in zip(hands_landmarks, handedness_labels):
        normalized = normalize_single_hand(landmarks)
        if label == "Left":
            vector[:slot_dim] = normalized
        else:  # "Right" (default fallback if label is ever ambiguous)
            vector[slot_dim:] = normalized

    return vector


# --------------------------------------------------------------------------- #
# Prediction CSV logging
# --------------------------------------------------------------------------- #
class PredictionLogger:
    """Appends timestamped predictions to a CSV log file."""

    def __init__(self, csv_path: Path = PATHS.predictions_log_file) -> None:
        self._csv_path = csv_path
        if not self._csv_path.exists():
            with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["timestamp", "prediction", "confidence"])

    def log(self, prediction: str, confidence: float) -> None:
        """Append one prediction row with an ISO-8601 timestamp."""
        with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([datetime.now().isoformat(timespec="seconds"), prediction, f"{confidence:.4f}"])


# --------------------------------------------------------------------------- #
# Screenshot / sentence persistence
# --------------------------------------------------------------------------- #
def save_screenshot(frame: np.ndarray, directory: Path = PATHS.screenshots_dir) -> Path:
    """Save the current frame as a timestamped PNG. Returns the saved path."""
    filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    path = directory / filename
    cv2.imwrite(str(path), frame)
    return path


def save_sentence(sentence: str, path: Path = PATHS.sentences_file) -> None:
    """Append a finalized sentence to the sentences log, one per line."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {sentence}\n")


# --------------------------------------------------------------------------- #
# OpenCV drawing helpers
# --------------------------------------------------------------------------- #
def draw_text_with_background(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font_scale: float = 0.8,
    color: tuple[int, int, int] = UI.text_color,
    thickness: int = 2,
    padding: int = 8,
) -> None:
    """Draw text with a semi-transparent dark background rectangle behind it,
    so overlay text stays legible over any webcam background.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x - padding, y - text_h - padding),
        (x + text_w + padding, y + baseline + padding),
        UI.panel_color,
        thickness=-1,
    )
    cv2.addWeighted(overlay, UI.panel_alpha, frame, 1 - UI.panel_alpha, 0, dst=frame)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
