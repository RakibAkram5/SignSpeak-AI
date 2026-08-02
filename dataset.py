"""Dataset loading and preprocessing pipeline.

Walks a folder-per-class image dataset (see config.PATHS.train_dir /
validation_dir), runs MediaPipe hand detection on each image, and converts
detected landmarks into the fixed-size feature vectors the model trains on.

Expected layout:
    dataset/
        train/
            A/  *.jpg|*.png
            B/  ...
        validation/
            A/  ...
            B/  ...
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder

from config import PATHS
from detector import HandDetector
from utils import setup_logger

logger = setup_logger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class DatasetSplit:
    """Feature/label arrays for one dataset split (train or validation)."""

    X: np.ndarray  # shape (n_samples, 126)
    y: np.ndarray  # shape (n_samples,) integer-encoded labels
    class_names: list[str]  # index -> label name, shared across splits


def _discover_classes(root: Path) -> list[str]:
    """Return sorted class names from the immediate subdirectories of root."""
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def _load_split(
    root: Path,
    detector: HandDetector,
    label_encoder: LabelEncoder,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Extract feature vectors and labels for every image under root.

    Returns:
        (X, y, num_images_seen, num_skipped_no_hand)
    """
    features: list[np.ndarray] = []
    labels: list[str] = []
    num_seen = 0
    num_skipped = 0

    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        class_name = class_dir.name
        image_paths = [p for p in class_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS]

        for image_path in image_paths:
            num_seen += 1
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning("Could not read image %s, skipping.", image_path)
                num_skipped += 1
                continue

            hands = detector.detect_hands(image)
            if not hands:
                num_skipped += 1
                continue

            feature_vector = detector.get_feature_vector(hands)
            features.append(feature_vector)
            labels.append(class_name)

        logger.info("Processed class '%s': %d images found under %s", class_name, len(image_paths), class_dir)

    if not features:
        return np.empty((0, 126), dtype=np.float32), np.empty((0,), dtype=np.int64), num_seen, num_skipped

    X = np.stack(features).astype(np.float32)
    y = label_encoder.transform(labels).astype(np.int64)
    return X, y, num_seen, num_skipped


def load_dataset(
    train_dir: Path = PATHS.train_dir,
    validation_dir: Path = PATHS.validation_dir,
) -> tuple[DatasetSplit, Optional[DatasetSplit]]:
    """Load and preprocess the training (and optional validation) dataset.

    Args:
        train_dir: Directory containing one subfolder of images per class.
        validation_dir: Optional directory with the same layout. If it does
            not exist or has no class subfolders, validation is skipped and
            the caller should carve a validation split out of the training
            data instead (see config.MODEL.validation_split).

    Returns:
        (train_split, validation_split_or_None)

    Raises:
        FileNotFoundError: If train_dir does not exist or contains no classes.
    """
    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}. "
            f"Create it and add one subfolder of images per sign, e.g. dataset/train/A/."
        )

    class_names = _discover_classes(train_dir)
    if not class_names:
        raise FileNotFoundError(
            f"No class subfolders found in {train_dir}. "
            f"Expected structure: {train_dir}/A/, {train_dir}/B/, ..."
        )

    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)
    logger.info("Discovered %d classes: %s", len(class_names), class_names)

    with HandDetector(static_image_mode=True, mirrored_input=False) as detector:
        X_train, y_train, seen_train, skipped_train = _load_split(train_dir, detector, label_encoder)
        logger.info(
            "Train split: %d usable samples out of %d images (%d skipped, no hand detected).",
            len(X_train), seen_train, skipped_train,
        )

        val_split: Optional[DatasetSplit] = None
        val_classes = _discover_classes(validation_dir)
        if val_classes:
            X_val, y_val, seen_val, skipped_val = _load_split(validation_dir, detector, label_encoder)
            logger.info(
                "Validation split: %d usable samples out of %d images (%d skipped, no hand detected).",
                len(X_val), seen_val, skipped_val,
            )
            val_split = DatasetSplit(X=X_val, y=y_val, class_names=list(label_encoder.classes_))
        else:
            logger.info("No validation directory found at %s; will use a split of the training data.", validation_dir)

    train_split = DatasetSplit(X=X_train, y=y_train, class_names=list(label_encoder.classes_))
    return train_split, val_split
