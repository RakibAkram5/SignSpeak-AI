"""Central configuration for the Sign Language Recognition System.

All paths, model hyperparameters, camera settings, and runtime thresholds
are defined here so the rest of the codebase never hardcodes a magic
number or a path. Import what you need:

    from config import PATHS, CAMERA, HANDS, MODEL, PREDICTION, LABELS
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final


# --------------------------------------------------------------------------- #
# Base paths
# --------------------------------------------------------------------------- #
BASE_DIR: Final[Path] = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Paths:
    """Filesystem locations used across the project."""

    base_dir: Path = BASE_DIR
    models_dir: Path = BASE_DIR / "models"
    dataset_dir: Path = BASE_DIR / "dataset"
    train_dir: Path = BASE_DIR / "dataset" / "train"
    validation_dir: Path = BASE_DIR / "dataset" / "validation"
    logs_dir: Path = BASE_DIR / "logs"
    screenshots_dir: Path = BASE_DIR / "screenshots"
    assets_dir: Path = BASE_DIR / "assets"

    # Concrete file paths
    model_file: Path = BASE_DIR / "models" / "sign_model.keras"
    label_encoder_file: Path = BASE_DIR / "models" / "label_encoder.json"
    training_history_file: Path = BASE_DIR / "models" / "training_history.json"
    accuracy_plot_file: Path = BASE_DIR / "models" / "accuracy_plot.png"
    loss_plot_file: Path = BASE_DIR / "models" / "loss_plot.png"
    predictions_log_file: Path = BASE_DIR / "logs" / "predictions.csv"
    app_log_file: Path = BASE_DIR / "logs" / "app.log"
    sentences_file: Path = BASE_DIR / "logs" / "sentences.txt"


PATHS = Paths()


def ensure_directories() -> None:
    """Create every directory the project depends on if it does not exist."""
    for directory in (
        PATHS.models_dir,
        PATHS.dataset_dir,
        PATHS.train_dir,
        PATHS.validation_dir,
        PATHS.logs_dir,
        PATHS.screenshots_dir,
        PATHS.assets_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)


ensure_directories()


# --------------------------------------------------------------------------- #
# Camera / webcam settings
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CameraConfig:
    """Webcam capture settings."""

    device_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    flip_horizontal: bool = True  # Mirror view feels natural for a user-facing camera
    target_fps: int = 30


CAMERA = CameraConfig()


# --------------------------------------------------------------------------- #
# MediaPipe Hands settings
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HandsConfig:
    """Parameters passed to mediapipe.solutions.hands.Hands()."""

    max_num_hands: int = 2
    detection_confidence: float = 0.7
    tracking_confidence: float = 0.7
    model_complexity: int = 1
    static_image_mode: bool = False  # False = video stream tracking (faster, uses temporal info)

    num_landmarks: int = 21
    num_coords_per_landmark: int = 3  # x, y, z


HANDS = HandsConfig()


# --------------------------------------------------------------------------- #
# Labels (ASL alphabet)
# --------------------------------------------------------------------------- #
# J and Z are excluded by default: they are motion gestures in ASL and cannot
# be reliably distinguished from a single static frame of landmarks. If a
# dataset directory provides folders for them, dataset.py will use the
# dataset's own folder names instead of this default list.
DEFAULT_LABELS: Final[list[str]] = [c for c in string.ascii_uppercase if c not in ("J", "Z")]


# --------------------------------------------------------------------------- #
# Model / training hyperparameters
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelConfig:
    """Neural network architecture and training hyperparameters."""

    # Input: 2 hands * 21 landmarks * 3 coords = 126 features (single hand is
    # zero-padded to keep a fixed-size input vector).
    input_dim: int = HANDS.num_landmarks * HANDS.num_coords_per_landmark * 2

    hidden_units: tuple[int, ...] = (256, 128, 64)
    dropout_rate: float = 0.3
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 12
    validation_split: float = 0.15  # used only if no separate validation/ dir is found
    random_seed: int = 42


MODEL = ModelConfig()


# --------------------------------------------------------------------------- #
# Real-time prediction / word-formation settings
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PredictionConfig:
    """Controls smoothing and stability of real-time predictions."""

    confidence_threshold: float = 0.75  # Minimum confidence to accept a prediction
    stability_frames: int = 10          # Consecutive identical predictions required
    prediction_cooldown_sec: float = 1.2  # Min seconds before the same letter can repeat
    history_length: int = 20            # Frames kept for the stability/majority vote buffer


PREDICTION = PredictionConfig()


# --------------------------------------------------------------------------- #
# UI settings
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class UIConfig:
    """Colors (BGR, OpenCV convention) and layout constants for the overlay."""

    font: int = 0  # cv2.FONT_HERSHEY_SIMPLEX (kept as int to avoid importing cv2 here)
    text_color: tuple[int, int, int] = (255, 255, 255)
    accent_color: tuple[int, int, int] = (0, 200, 255)
    success_color: tuple[int, int, int] = (0, 220, 0)
    warning_color: tuple[int, int, int] = (0, 0, 255)
    panel_color: tuple[int, int, int] = (30, 30, 30)
    panel_alpha: float = 0.55


UI = UIConfig()


# --------------------------------------------------------------------------- #
# Keyboard shortcuts (OpenCV waitKey codes are resolved in utils.py; here we
# only define the semantic keys so app.py stays readable).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class KeyBindings:
    quit: str = "q"
    screenshot: str = "s"
    clear_sentence: str = "c"
    space: int = 32       # SPACE
    backspace: int = 8    # BACKSPACE
    enter: tuple[int, ...] = field(default_factory=lambda: (13, 10))  # ENTER (CR/LF, platform-dependent)


KEYS = KeyBindings()


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_LEVEL: Final[str] = "INFO"
