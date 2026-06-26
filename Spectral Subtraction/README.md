# Speech Enhancement via Spectral Subtraction

A classic **spectral subtraction** speech-enhancement experiment. Clean speech is mixed with real-world noise at controlled SNR levels, then denoised by estimating the noise spectrum and subtracting it in the power domain. The notebook measures the resulting SNR improvement and visualizes the effect on spectrograms.

## Overview

Spectral subtraction is one of the oldest single-channel denoising methods: estimate the noise power spectrum (from a noise-only reference), subtract a scaled version of it from the noisy signal's power spectrum, floor the result to avoid negative values, and resynthesize using the noisy phase. This notebook runs the full experiment on real datasets:

1. Download clean speech (LibriSpeech) and noise (NOISEX-92).
2. Mix a clean utterance with noise at target SNRs of 0, 5, and 10 dB.
3. Enhance each noisy mixture with spectral subtraction.
4. Measure input vs. output SNR and report the improvement.
5. Save the noisy/enhanced audio and plot clean/noisy/enhanced spectrograms.

## Method

Working on the STFT (1024-point FFT, 256 hop, Hann window):

1. **Noise PSD estimate** — average the power spectrum of a noise-only reference segment (first ~5 s) over time.
2. **Over-subtraction** — `|X|² = |Y|² − α · noise_psd`, with an over-subtraction factor `α` that is larger at lower SNR (interpolated from 3.0 at 0 dB to 2.0 at 10 dB) to subtract more aggressively when noise dominates.
3. **Spectral floor** — clamp the result to `β · noise_psd` (`β = 0.01`) to keep it non-negative and suppress "musical noise."
4. **Smoothing** — a light 5-point average across neighboring time/frequency bins to further reduce musical artifacts.
5. **Reconstruction** — combine the cleaned magnitude with the original noisy phase and inverse-STFT back to a waveform.

## Datasets

- **Clean speech:** [LibriSpeech](https://www.openslr.org/12) `dev-clean` (OpenSLR, CC BY 4.0) — FLAC recordings.
- **Noise:** [NOISEX-92](https://github.com/speechdnn/Noises) — a public WAV copy of the standard noise corpus.

Audio is loaded at 16 kHz mono and peak-normalized. Mixtures are created at a target SNR by scaling a random noise segment to the required power relative to the clean signal, with clipping protection.

## Key Parameters

| Parameter | Value | Role |
|-----------|-------|------|
| `n_fft` / `hop` / `win` | 1024 / 256 / 1024 | STFT framing |
| `alpha` | 2.0–3.0 (SNR-dependent) | Over-subtraction strength |
| `beta` | 0.01 | Spectral floor (musical-noise control) |
| `smooth` | True | Time–frequency smoothing |
| SNR targets | 0, 5, 10 dB | Mixing conditions |

## Output

For each target SNR the notebook prints measured input SNR, output SNR, and improvement; saves `noisy_<snr>dB.wav` and `enh_<snr>dB.wav`; and plays clean → noisy → enhanced audio inline. Spectrograms for the clean, noisy, and enhanced signals are plotted for qualitative comparison.

## Requirements

- Python 3.x (Google Colab works well)
- `librosa`, `soundfile`, `numpy`, `matplotlib`

```bash
pip install librosa soundfile numpy matplotlib
```

## Usage

1. Open `Spectral_Subtraction.ipynb` in Colab.
2. Run the cells — they download the datasets, pick a random speech/noise pair, run enhancement at each SNR, and display results.
3. Adjust `alpha`, `beta`, `snr_targets`, or the framing parameters to explore the noise-suppression vs. distortion trade-off.

## Notes

- Spectral subtraction reuses the **noisy phase**, which limits achievable quality and is a known weakness of the method.
- Larger `alpha` removes more noise but introduces more speech distortion; `beta` and the smoothing step trade residual noise against musical artifacts.
- The noise PSD is estimated from a fixed noise-only reference; performance degrades for non-stationary noise where the spectrum changes over time.