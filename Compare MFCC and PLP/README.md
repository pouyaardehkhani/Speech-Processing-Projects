# MFCC vs PLP: Acoustic Features for Spoken-Digit Recognition

A comparison of two classic speech acoustic feature representations — **MFCC** (Mel-Frequency Cepstral Coefficients) and **PLP** (Perceptual Linear Prediction) — for spoken-digit classification, evaluated in both **clean** and **additive-noise** conditions. The goal is to see how each feature set performs on its own and how gracefully each degrades as noise increases.

## Overview

The pipeline extracts MFCC and PLP features from spoken-digit recordings, summarizes each utterance into a fixed-length vector, trains a simple classifier on clean speech, and then tests on both clean and noisy audio at several SNR levels:

1. Download and prepare the dataset.
2. Extract MFCC (via `librosa`) and PLP (via `spafe`) features per utterance.
3. Summarize variable-length frame features into a fixed vector: per-coefficient **mean + standard deviation** over time.
4. Train a classifier (KNN or linear SVM, with feature standardization) on clean features.
5. Evaluate on clean test data, then on test data corrupted with additive white Gaussian noise (AWGN) at each requested SNR.
6. Report accuracy and confusion matrices for each feature set and condition.

## Dataset

[Free Spoken Digit Dataset (FSDD)](https://github.com/Jakobovski/free-spoken-digit-dataset) — recordings of spoken digits 0–9 by multiple speakers. The label is parsed from each filename (`{digit}_{speaker}_{index}.wav`). The dataset is downloaded and extracted automatically. Audio is resampled to 16 kHz mono; the data is split into train/test (80/20, stratified by digit).

## Features

- **MFCC** — 13 coefficients, 25 ms window / 10 ms hop, computed with librosa.
- **PLP** — 13 cepstra, same framing, computed with spafe. The notebook calls spafe's PLP through a signature-robust wrapper that adapts to differences across spafe versions (parameter names, positional-only args).
- Both are reduced to a `2 × n_coeffs` vector (mean + std per coefficient), with padding/truncation to handle short utterances and orientation differences.

## Noise Model

Additive white Gaussian noise is injected on the waveform at a target SNR: from signal power `P_signal`, noise power is set to `P_signal / 10^(SNR/10)`, and the noisy signal is clipped to `[-1, 1]`. Crucially, the classifier is trained **only on clean speech** and tested on noisy speech — a deliberate train/test mismatch.

## Classifier

A scikit-learn `Pipeline` of `StandardScaler` + classifier:
- **KNN** (k = 7, default), or
- **Linear SVM** (`LinearSVC`).

## Results

In the **clean** condition, **MFCC + KNN** is highly accurate (~97.8%) with a near-diagonal confusion matrix — the spectral envelope captured by MFCCs is very discriminative for FSDD. **PLP + KNN** does much worse even on clean speech (~47.8%) in this setup, suggesting the PLP coefficients (as computed by the library version used) combined with the simple mean+std summarization and KNN are a poor match.

Under **noisy** test conditions, both feature sets collapse: MFCC accuracy falls to ~16% at 20 dB and toward chance (~10%) at 10 dB and 0 dB, with predictions heavily biased toward a single digit — a signature of train/test distribution shift, since the model never saw noise during training. PLP shows no robustness advantage here either (~10% across SNRs).

**Takeaway:** for this pipeline, MFCC is clearly superior on clean speech, but neither MFCC nor PLP is robust to additive noise **without noise-aware training or normalization**. Likely remedies include training on noisy/augmented data, cepstral mean–variance normalization (CMVN), delta/delta-delta features, or a model that captures temporal structure.

## Requirements

- Python 3.x
- `librosa`, `spafe`, `numpy`, `scikit-learn`, `requests`, `tqdm`

```bash
pip install librosa spafe numpy scikit-learn requests tqdm
```

## Usage

Run as a notebook cell or a script. The entry point accepts CLI-style arguments (notebook-safe):

```python
main(["--snr_list", "20", "10", "0", "--classifier", "knn"])
```

Options:
- `--snr_list` — SNR (dB) values for the noisy test (e.g. `20 10 0`)
- `--classifier` — `knn` or `linear`
- `--n_neighbors` — k for KNN
- `--sr`, `--test_size`, `--seed`, `--data_dir`

## Notes

- The mean+std utterance summarization discards temporal dynamics; adding delta features or a sequence model would likely help, especially for PLP.
- The PLP wrapper is intentionally defensive because spafe's PLP signature has changed across releases; if PLP extraction fails, check your installed spafe version.
- Results are illustrative of the clean-train / noisy-test mismatch problem, not a definitive ranking of MFCC vs PLP under matched conditions.