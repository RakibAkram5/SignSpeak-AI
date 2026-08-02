"""Real-time Sign Language Recognition application.

Usage:
    python app.py [--camera 0]

Keyboard shortcuts:
    Q          Quit
    S          Save a screenshot of the current frame
    C          Clear the sentence and in-progress word
    SPACE      Manually add the currently displayed letter to the word
    BACKSPACE  Delete the last letter of the in-progress word
    ENTER      Finalize the in-progress word into the sentence (+ speak it)
"""

from __future__ import annotations

import argparse
import threading
from collections import deque
from typing import Optional

import cv2

from config import CAMERA, KEYS, PATHS, UI
from detector import HandDetector
from predict import SignPredictor, StablePredictor
from utils import (
    FPSCounter,
    PredictionLogger,
    draw_text_with_background,
    save_screenshot,
    save_sentence,
    setup_logger,
)

logger = setup_logger(__name__)

try:
    import pyttsx3

    _TTS_AVAILABLE = True
except ImportError:
    _TTS_AVAILABLE = False


class TextToSpeech:
    """Non-blocking text-to-speech using pyttsx3, run on a background thread.

    If pyttsx3 is not installed, speak() silently becomes a no-op so the
    rest of the application keeps working without the bonus feature.
    """

    def __init__(self) -> None:
        self._enabled = _TTS_AVAILABLE
        if not self._enabled:
            logger.warning("pyttsx3 not installed; text-to-speech is disabled. Run: pip install pyttsx3")

    def speak(self, text: str) -> None:
        if not self._enabled or not text.strip():
            return
        threading.Thread(target=self._speak_sync, args=(text,), daemon=True).start()

    @staticmethod
    def _speak_sync(text: str) -> None:
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception:  # pragma: no cover - defensive; TTS backends vary by OS
            logger.exception("Text-to-speech failed for text: %s", text)


class SentenceBuilder:
    """Manages the in-progress word buffer and the finalized sentence history."""

    def __init__(self) -> None:
        self.current_word: list[str] = []
        self.sentence_words: list[str] = []

    def add_letter(self, letter: str) -> None:
        self.current_word.append(letter)

    def backspace(self) -> None:
        if self.current_word:
            self.current_word.pop()
        elif self.sentence_words:
            # Nothing left to delete in the current word: reopen the last
            # finalized word for editing.
            self.current_word = list(self.sentence_words.pop())

    def finalize_word(self) -> Optional[str]:
        """Move the current word into the sentence. Returns the finalized word."""
        if not self.current_word:
            return None
        word = "".join(self.current_word)
        self.sentence_words.append(word)
        self.current_word = []
        return word

    def clear(self) -> None:
        self.current_word = []
        self.sentence_words = []

    @property
    def sentence_text(self) -> str:
        return " ".join(self.sentence_words)

    @property
    def current_word_text(self) -> str:
        return "".join(self.current_word)


def draw_overlay(
    frame,
    raw_label: Optional[str],
    raw_confidence: float,
    sentence_builder: SentenceBuilder,
    fps: float,
    model_loaded: bool,
    gesture_history: "deque[str]",
) -> None:
    """Draw all HUD text (prediction, confidence, word/sentence, FPS, help)."""
    height, width = frame.shape[:2]

    draw_text_with_background(frame, f"FPS: {fps:.1f}", (20, 40), font_scale=0.7)

    if not model_loaded:
        draw_text_with_background(
            frame,
            "No trained model found — run train.py first (showing landmarks only)",
            (20, 80),
            font_scale=0.65,
            color=UI.warning_color,
        )
    elif raw_label is not None:
        draw_text_with_background(
            frame, f"Prediction: {raw_label}", (20, 90), font_scale=1.1, color=UI.accent_color, thickness=3
        )
        draw_text_with_background(
            frame, f"Confidence: {raw_confidence * 100:.1f}%", (20, 130), font_scale=0.75
        )
    else:
        draw_text_with_background(frame, "Prediction: -", (20, 90), font_scale=1.0)

    if gesture_history:
        draw_text_with_background(
            frame, f"History: {' '.join(gesture_history)}", (20, 170), font_scale=0.6, color=UI.accent_color
        )

    draw_text_with_background(
        frame, f"Word: {sentence_builder.current_word_text}", (20, height - 90), font_scale=0.8, color=UI.success_color
    )
    sentence_display = sentence_builder.sentence_text or "(empty)"
    draw_text_with_background(
        frame, f"Sentence: {sentence_display}", (20, height - 50), font_scale=0.8
    )

    help_text = "Q:Quit  S:Screenshot  C:Clear  SPACE:Add  BACKSPACE:Delete  ENTER:Finalize"
    draw_text_with_background(frame, help_text, (20, height - 15), font_scale=0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time Sign Language Recognition")
    parser.add_argument("--camera", type=int, default=CAMERA.device_index, help="Webcam device index")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    predictor: Optional[SignPredictor] = None
    try:
        predictor = SignPredictor()
    except FileNotFoundError as exc:
        logger.warning("%s", exc)

    stable_predictor = StablePredictor()
    prediction_logger = PredictionLogger()
    sentence_builder = SentenceBuilder()
    tts = TextToSpeech()
    fps_counter = FPSCounter()
    gesture_history: deque[str] = deque(maxlen=10)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA.frame_height)
    if not cap.isOpened():
        logger.error("Could not open webcam at index %d.", args.camera)
        return 1

    window_name = "Sign Language Recognition"

    with HandDetector(mirrored_input=CAMERA.flip_horizontal) as detector:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame from webcam.")
                    break

                if CAMERA.flip_horizontal:
                    frame = cv2.flip(frame, 1)

                hands = detector.detect_hands(frame)
                detector.draw_hands(frame)

                raw_label: Optional[str] = None
                raw_confidence = 0.0

                if hands and predictor is not None:
                    feature_vector = detector.get_feature_vector(hands)
                    raw_label, raw_confidence = predictor.predict(feature_vector)

                stable_label, stable_confidence, should_commit = stable_predictor.update(
                    raw_label, raw_confidence
                )
                if should_commit and stable_label is not None:
                    sentence_builder.add_letter(stable_label)
                    prediction_logger.log(stable_label, stable_confidence)
                    gesture_history.append(stable_label)
                    logger.info("Committed letter '%s' (confidence %.2f%%)", stable_label, stable_confidence * 100)

                fps = fps_counter.update()
                draw_overlay(
                    frame, raw_label, raw_confidence, sentence_builder, fps, predictor is not None, gesture_history
                )

                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord(KEYS.quit):
                    break
                elif key == ord(KEYS.screenshot):
                    path = save_screenshot(frame)
                    logger.info("Screenshot saved to %s", path)
                elif key == ord(KEYS.clear_sentence):
                    sentence_builder.clear()
                    stable_predictor.reset()
                    logger.info("Sentence cleared.")
                elif key == KEYS.space:
                    if raw_label is not None:
                        sentence_builder.add_letter(raw_label)
                elif key == KEYS.backspace:
                    sentence_builder.backspace()
                elif key in KEYS.enter:
                    word = sentence_builder.finalize_word()
                    if word:
                        logger.info("Finalized word: %s", word)
                    if sentence_builder.sentence_text:
                        save_sentence(sentence_builder.sentence_text)
                        tts.speak(sentence_builder.sentence_text)
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Application closed cleanly.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
