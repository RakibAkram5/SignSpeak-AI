# SignSpeak-AI — Real-Time Sign Language Recognition

A production-ready, real-time Sign Language Recognition system built with **Python, OpenCV, MediaPipe, and TensorFlow/Keras**. It detects hand landmarks from a webcam feed, classifies them into ASL alphabet letters using a trained neural network, and lets you build words and sentences using only your hands and a few keyboard shortcuts.

## Overview

Instead of classifying raw video frames with a heavy CNN, this project uses **MediaPipe Hands** to extract 21 3D hand landmarks per hand, normalizes them into a translation/scale-invariant feature vector, and feeds that into a small, fast **MLP (multilayer perceptron)** classifier. This design is what makes real-time 30 FPS inference possible on a CPU, with no GPU required.

## Features

- Real-time webcam capture with live FPS display
- Detection and tracking of up to 2 hands, 21 landmarks each, drawn live on screen
- ASL alphabet recognition (A–Z, excluding J/Z — see [Dataset Preparation](#dataset-preparation))
- Prediction confidence displayed alongside each letter
- Automatic word formation with duplicate-letter suppression and a configurable cooldown
- Manual sentence builder via keyboard shortcuts (add / delete / finalize / clear)
- Full training pipeline: dataset loading → landmark extraction → training → validation → accuracy/loss plots → model saving
- Automatic model loading for inference — no manual wiring required
- CSV logging of every committed prediction with timestamps
- Screenshot capture
- Bonus: text-to-speech playback of finalized sentences (via `pyttsx3`, optional)

## Installation

**Requirements:** Python 3.12+ and a working webcam.

```bash
git clone https://github.com/RakibAkram5/SignSpeak-AI.git
cd SignSpeak-AI

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

## Folder Structure

```
sign-language-recognition/
│
├── app.py             # Main real-time application (webcam loop, prediction, sentence builder)
├── train.py            # Training pipeline entry point
├── predict.py          # Inference wrapper: model loading + temporal smoothing
├── detector.py          # HandDetector class (MediaPipe Hands wrapper)
├── model.py             # SignLanguageModel class (Keras MLP)
├── dataset.py            # Dataset loading and landmark feature extraction
├── utils.py              # Logging, FPS counter, normalization, CSV/screenshot/sentence helpers
├── config.py              # Central configuration (paths, hyperparameters, thresholds, UI)
├── requirements.txt
├── README.md
├── models/               # Saved model, label encoder, accuracy/loss plots
├── dataset/
│   ├── train/            # dataset/train/<LABEL>/*.jpg
│   └── validation/        # dataset/validation/<LABEL>/*.jpg (optional)
├── logs/                 # app.log, predictions.csv, sentences.txt
├── screenshots/           # Saved frames (press S)
└── assets/               # Any static assets (e.g. reference sign charts)
```

## Dataset Preparation

Organize images as one folder per class (letter) under `dataset/train/` and, optionally, `dataset/validation/`:

```
dataset/
    train/
        A/  img001.jpg  img002.jpg  ...
        B/  ...
        ...
    validation/
        A/  ...
        B/  ...
```

Notes:
- Class names are discovered automatically from folder names — you are not limited to the ASL alphabet. Any static hand-pose gesture set (ASL, PSL, custom gestures) works as long as folders follow this layout.
- **J and Z are excluded by default** because they are drawn as motion in ASL and cannot be distinguished from a single static frame of landmarks. If your dataset includes folders for them, they will still be picked up (a static approximation), but accuracy on those two classes will likely be lower.
- Images where MediaPipe cannot detect a hand are automatically skipped during training with a warning — this is expected for blurry/occluded images and does not indicate a bug.
- Recommended: 100+ images per class, varied hand positions/lighting/backgrounds, for good generalization.

You can build your own dataset quickly by running `app.py`, holding a sign, and pressing **S** to save labeled screenshots into the right folder — or write a small capture script using `detector.py` directly.

## Training Instructions

```bash
python train.py
```

This will:
1. Load and validate `dataset/train/` (and `dataset/validation/` if present)
2. Extract normalized MediaPipe landmark features for every image
3. Train the MLP with early stopping and learning-rate reduction on plateau
4. Evaluate on a held-out test split
5. Save `models/accuracy_plot.png` and `models/loss_plot.png`
6. Save the trained model to `models/sign_model.keras` and the label mapping to `models/label_encoder.json`

Tune hyperparameters (hidden layer sizes, dropout, learning rate, batch size, epochs) in `config.py` under `ModelConfig`.

## Running the Application

```bash
python app.py
# or specify a different webcam:
python app.py --camera 1
```

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Q` | Quit the application |
| `S` | Save a screenshot of the current frame |
| `C` | Clear the sentence and in-progress word |
| `SPACE` | Manually add the currently displayed letter to the word |
| `BACKSPACE` | Delete the last letter of the in-progress word |
| `ENTER` | Finalize the current word into the sentence, save it, and speak it |

If no trained model exists yet, `app.py` still runs — it shows the webcam feed with live hand landmarks and a banner prompting you to run `train.py` first.

## Troubleshooting

- **`Could not open webcam`**: another application may be using the camera, or the index is wrong — try `python app.py --camera 1`.
- **Low FPS**: lower `CAMERA.frame_width`/`frame_height` in `config.py`, or set `HandsConfig.model_complexity = 0` for faster (slightly less accurate) landmark detection.
- **`No trained model found` error from `predict.py`**: run `python train.py` first; it writes to `models/sign_model.keras`.
- **Poor accuracy / letters confused**: add more training images with varied lighting/backgrounds/hand positions, and check `logs/app.log` for how many images were skipped due to no hand detected.
- **`pyttsx3` errors on Linux**: install `espeak` (`sudo apt install espeak`); text-to-speech is optional and the app runs fine without it.
- **TensorFlow install issues on Windows**: ensure you're on a 64-bit Python 3.12 install; consider `pip install tensorflow-cpu` if you don't have a GPU.

## Future Improvements

- Sequence model (LSTM/Transformer over landmark sequences) to support motion signs (J, Z) and full ASL words/phrases
- Two-handed sign support for languages that require both hands per sign
- ONNX export and quantization for faster edge/mobile inference
- PSL and other regional sign language datasets
- Speech-to-text integration for two-way communication
- Tkinter/PyQt GUI as an alternative to the OpenCV window
- Live accuracy monitoring dashboard during training
