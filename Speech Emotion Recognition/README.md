# Speech Emotion Recognition

A deep-learning model that classifies the **emotion** expressed in a speech recording. The pipeline extracts MFCC features from audio and trains an LSTM network to predict one of seven emotion categories.

## Overview

The notebook covers the full sound-classification workflow:

1. **Load the dataset** — scan the audio files and parse the emotion label from each filename.
2. **Exploratory analysis** — inspect the class distribution and visualize waveforms and spectrograms for each emotion (with inline audio playback).
3. **Feature extraction** — compute MFCCs for each clip.
4. **Model** — train an LSTM classifier over the MFCC features.
5. **Evaluation** — plot training/validation accuracy and loss curves.

## Dataset

A speech emotion dataset (TESS-style, ~2800 clips) where the emotion label is encoded in each filename. The seven classes are: **angry, disgust, fear, happy, neutral, sad,** and **ps** (pleasant surprise). Labels are one-hot encoded for training.

## Feature Extraction

Each clip is loaded with librosa (3-second window, 0.5 s offset) and reduced to a **40-dimensional MFCC vector** by averaging the 40 MFCC coefficients over time. This gives one fixed-length feature vector per utterance, reshaped to `(40, 1)` as the LSTM input.

## Model

A Keras `Sequential` LSTM classifier:

```
LSTM(256)          # input shape (40, 1)
Dropout(0.2)
Dense(128, relu)
Dropout(0.2)
Dense(64, relu)
Dropout(0.2)
Dense(7, softmax)  # 7 emotion classes
```

- **Loss:** categorical cross-entropy
- **Optimizer:** Adam
- **Training:** 50 epochs, batch size 64, 20% validation split
- **Best validation accuracy:** ~72%

## Requirements

- Python 3.x
- `tensorflow` / `keras`
- `librosa`
- `numpy`, `pandas`, `scikit-learn`
- `matplotlib`, `seaborn`

```bash
pip install tensorflow librosa numpy pandas scikit-learn matplotlib seaborn
```

## Usage

1. Point the data-loading cell at your audio directory (the notebook walks `/kaggle/input`; change this to your dataset path).
2. Open `Speech_Emotion_Recognition_-_Sound_Classification.ipynb` and run the cells in order to load data, extract MFCCs, and train.
3. Review the accuracy/loss plots to inspect convergence.

## Notes

- Averaging MFCCs over time discards temporal dynamics; feeding the LSTM the full MFCC sequence (instead of a single averaged vector) would let it model how emotion unfolds over the utterance and likely improve accuracy.
- Accuracy could be further improved with data augmentation (pitch shift, time stretch, noise), added delta/delta-delta features, a model checkpoint to keep the best-validation weights, and learning-rate scheduling.
- This is a learning/portfolio project; the ~72% validation accuracy is a baseline rather than a tuned result.