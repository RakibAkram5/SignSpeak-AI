"""MediaPipe-based hand detection wrapper.

Exposes a single HandDetector class so the rest of the codebase never
imports mediapipe directly. Keeping the MediaPipe API surface contained
here makes it easy to swap detection backends later without touching
app.py, dataset.py, or predict.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np

from config import HANDS
from utils import landmarks_to_feature_vector, setup_logger

logger = setup_logger(__name__)


@dataclass
class DetectedHand:
    """Container for one detected hand's data for the current frame."""

    landmarks_xyz: np.ndarray  # shape (21, 3) — normalized image coords (x, y, z)
    landmarks_px: np.ndarray  # shape (21, 2) — pixel coords (x, y) for drawing/distance calc
    handedness: str  # "Left" or "Right" (already corrected for the mirrored view)
    confidence: float  # MediaPipe's handedness classification confidence


class HandDetector:
    """Detects hands and hand landmarks in a video frame using MediaPipe Hands."""

    # Fingertip and corresponding PIP-joint landmark indices, thumb first.
    _TIP_IDS = (4, 8, 12, 16, 20)
    _PIP_IDS = (3, 6, 10, 14, 18)

    def __init__(
        self,
        static_image_mode: bool = HANDS.static_image_mode,
        max_num_hands: int = HANDS.max_num_hands,
        model_complexity: int = HANDS.model_complexity,
        detection_confidence: float = HANDS.detection_confidence,
        tracking_confidence: float = HANDS.tracking_confidence,
        mirrored_input: bool = True,
    ) -> None:
        """
        Args:
            static_image_mode: True treats every frame independently (used for
                dataset image processing); False enables tracking between
                frames (used for the live webcam feed, faster and smoother).
            max_num_hands: Maximum number of hands to detect simultaneously.
            model_complexity: 0 (fastest) or 1 (more accurate landmarks).
            detection_confidence: Minimum confidence for initial hand detection.
            tracking_confidence: Minimum confidence to keep tracking a hand.
            mirrored_input: Set True if the frame passed to detect_hands() has
                already been horizontally flipped (selfie view). MediaPipe's
                Left/Right handedness label is corrected accordingly so it
                matches what the user visually sees.
        """
        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles

        self._hands = self._mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._mirrored_input = mirrored_input
        self._results = None
        self._frame_shape: tuple[int, int] = (0, 0)  # (height, width) of the last processed frame

        logger.info(
            "HandDetector initialized (max_hands=%d, complexity=%d, static_mode=%s)",
            max_num_hands,
            model_complexity,
            static_image_mode,
        )

    def detect_hands(self, frame: np.ndarray) -> List[DetectedHand]:
        """Run hand detection on a BGR frame and return structured hand data.

        Args:
            frame: BGR image (as read/produced by OpenCV).

        Returns:
            A list of DetectedHand, one per hand found (possibly empty).
        """
        self._frame_shape = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        self._results = self._hands.process(rgb_frame)

        detected: List[DetectedHand] = []
        if not self._results.multi_hand_landmarks:
            return detected

        height, width = self._frame_shape
        handedness_list = self._results.multi_handedness

        for hand_landmarks, handedness in zip(self._results.multi_hand_landmarks, handedness_list):
            xyz = np.array(
                [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark], dtype=np.float32
            )
            px = np.array(
                [(lm.x * width, lm.y * height) for lm in hand_landmarks.landmark], dtype=np.float32
            )

            label = handedness.classification[0].label  # "Left" or "Right"
            if self._mirrored_input:
                label = "Right" if label == "Left" else "Left"

            detected.append(
                DetectedHand(
                    landmarks_xyz=xyz,
                    landmarks_px=px,
                    handedness=label,
                    confidence=handedness.classification[0].score,
                )
            )

        return detected

    def draw_hands(self, frame: np.ndarray) -> np.ndarray:
        """Draw landmarks and connections for the most recently detected hands.

        Must be called after detect_hands() on the same frame. Returns the
        frame with drawings applied (drawn in place).
        """
        if self._results and self._results.multi_hand_landmarks:
            for hand_landmarks in self._results.multi_hand_landmarks:
                self._mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._mp_styles.get_default_hand_landmarks_style(),
                    self._mp_styles.get_default_hand_connections_style(),
                )
        return frame

    def get_feature_vector(self, hands: List[DetectedHand]) -> np.ndarray:
        """Convert detected hands into the fixed-size 126-dim model input vector.

        Args:
            hands: Output of detect_hands().

        Returns:
            A (126,) float32 feature vector (zero-padded if fewer than 2 hands).
        """
        landmarks = [h.landmarks_xyz for h in hands]
        handedness = [h.handedness for h in hands]
        return landmarks_to_feature_vector(landmarks, handedness)

    def fingers_up(self, hand: DetectedHand) -> List[int]:
        """Determine which fingers are extended for a single hand.

        Uses a simple geometric heuristic comparing each fingertip to its
        PIP joint. Intended for rule-based bonus gestures / debugging, not
        as the primary ASL classifier.

        Returns:
            List of 5 ints (1 = up, 0 = down), ordered [thumb, index, middle, ring, pinky].
        """
        px = hand.landmarks_px
        fingers = []

        # Thumb: compare x-position (accounts for left/right hand orientation).
        thumb_tip_x, thumb_pip_x = px[self._TIP_IDS[0]][0], px[self._PIP_IDS[0]][0]
        if hand.handedness == "Right":
            fingers.append(1 if thumb_tip_x > thumb_pip_x else 0)
        else:
            fingers.append(1 if thumb_tip_x < thumb_pip_x else 0)

        # Other four fingers: tip above PIP joint (smaller y) means extended.
        for tip_id, pip_id in zip(self._TIP_IDS[1:], self._PIP_IDS[1:]):
            fingers.append(1 if px[tip_id][1] < px[pip_id][1] else 0)

        return fingers

    def find_distance(
        self,
        point1_idx: int,
        point2_idx: int,
        hand: DetectedHand,
        frame: Optional[np.ndarray] = None,
    ) -> tuple[float, tuple[int, int], tuple[int, int]]:
        """Compute pixel distance between two landmarks on a hand.

        Args:
            point1_idx: Landmark index of the first point (0-20).
            point2_idx: Landmark index of the second point (0-20).
            hand: The DetectedHand to measure on.
            frame: If provided, draws the line and endpoints for visual debugging.

        Returns:
            (distance_in_pixels, point1_coords, point2_coords)
        """
        x1, y1 = hand.landmarks_px[point1_idx]
        x2, y2 = hand.landmarks_px[point2_idx]
        distance = float(np.hypot(x2 - x1, y2 - y1))

        if frame is not None:
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (int(x1), int(y1)), 6, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, (int(x2), int(y2)), 6, (255, 0, 255), cv2.FILLED)

        return distance, (int(x1), int(y1)), (int(x2), int(y2))

    def close(self) -> None:
        """Release MediaPipe resources. Call when done with the detector."""
        self._hands.close()

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
