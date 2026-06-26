# Channel Noise in Representation Space

An experiment that simulates a **noisy communication channel in the audio representation domain** rather than on the raw waveform. The same Gaussian noise level (specified as an SNR in dB) is injected into three different internal audio representations, the signal is reconstructed back to audio, and the resulting distortion is compared across representations.

## Idea

Instead of corrupting the waveform directly, this notebook asks: *if you transmit audio as a feature representation (mel-spectrogram, STFT magnitude, or MFCC) over a noisy channel, which representation degrades most gracefully?* Each representation is corrupted at an identical representation-space SNR, then inverted back to audio so the channels can be compared on equal footing.

## The Three Channels

| Channel | Pipeline | Inversion |
|---------|----------|-----------|
| **A1 — Mel-Spectrogram** | `y → Mel(y) → Mel + N → mel_to_stft → audio` | Griffin-Lim |
| **A2 — STFT Magnitude** | `y → \|STFT(y)\| → \|STFT\| + N → audio` | Griffin-Lim (phase discarded) |
| **A3 — MFCC** | `y → MFCC(y) → MFCC + N → mfcc_to_mel → audio` | inverse MFCC → mel → audio |

Gaussian noise is added at a fixed SNR: from the signal power `P_signal`, the noise power is set as `P_noise = P_signal / 10^(SNR/10)`, so every channel sees the same relative noise level. Non-negativity is enforced where required (mel power and STFT magnitude cannot be negative).

## Workflow

1. Load Persian speech WAV files (first 20 of the dataset).
2. For each file, reconstruct audio through channels A1, A2, and A3 at the same representation SNR.
3. Save all reconstructions for listening.
4. Score each reconstruction by the L2 distance between the MFCCs of the original and reconstructed audio.
5. Visualize clean vs. noisy mel / STFT / MFCC representations for one example, plus the resulting MFCCs of each reconstruction.

## Dataset

[PersianSpeech](https://github.com/persiandataset/PersianSpeech) — a Persian speech corpus. The notebook also pulls a small tarball of WAV clips. Audio is processed at 16 kHz mono.

## Key Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `SR` | 16000 | Sample rate |
| `N_FFT` / `HOP` | 1024 / 256 | STFT window / hop |
| `N_MELS` | 80 | Mel bands |
| `N_MFCC` | 13 | MFCC coefficients |
| `NOISE_SNR_DB` | 10.0 | Representation-space SNR (higher = weaker noise) |

## Findings

At a fixed representation SNR, MFCC-based distortion (L2 between clean and reconstructed MFCCs) is consistently **lowest for the MFCC channel (A3)**, intermediate for the **STFT-magnitude channel (A2)**, and **highest for the mel channel (A1)**.

Qualitatively this matches how the reconstructions sound:

- **A1 (mel noise)** is the most degraded — strong broadband hiss with smeared speech harmonics, since noise is spread across a compact mel representation that is then doubly inverted (mel → STFT → waveform).
- **A2 (STFT noise)** is cleaner but still grainy; spectral structure survives better, with phase reconstructed via Griffin-Lim.
- **A3 (MFCC noise)** sounds least noisy but most "smoothed"/processed — MFCCs encode only the coarse spectral envelope, so noise in that compact space perturbs less audible detail (at the cost of losing fine structure).

In short, the more compact and envelope-focused the representation, the more robust it is to a fixed-SNR Gaussian perturbation — but that robustness comes from already discarding fine detail.

## Requirements

- Python 3.x
- `librosa >= 0.10` (needed for `mfcc_to_mel`)
- `soundfile`, `numpy`, `matplotlib`, `gdown`

```bash
pip install "librosa>=0.10" soundfile numpy matplotlib gdown
```

## Usage

1. Open `channel-noise.ipynb` (Google Colab works well).
2. Run the cells to download the dataset, set `DATA_DIR` to the extracted WAV folder, and process the files.
3. Reconstructed audio is written to `./recon_repr_noise/` as `<name>_A1.wav`, `<name>_A2.wav`, `<name>_A3.wav` for listening.
4. Adjust `NOISE_SNR_DB` to explore how each channel behaves under stronger or weaker noise.

## Notes

- Griffin-Lim discards original phase and estimates it iteratively; increasing `n_iter` (e.g. 32 → 64) improves reconstruction quality at the cost of speed.
- MFCC inversion is approximate (the DCT truncation to 13 coefficients is lossy), which is exactly why the MFCC channel sounds smoothed.
- The MFCC-L2 score is a convenient proxy for perceptual distortion, not a calibrated perceptual metric.