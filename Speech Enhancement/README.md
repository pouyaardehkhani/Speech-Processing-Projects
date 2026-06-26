# Speech Enhancement: Classical DSP vs. Deep Learning

A two-part study of single-channel speech enhancement — recovering clean, intelligible speech from noisy recordings. Part 1 implements classical DSP baselines (spectral subtraction, Wiener filtering); Part 2 designs and trains a deep learning model (a **Conformer U-Net** mask estimator). The two approaches are compared with SI-SDR and spectrogram visualizations on the VoiceBank + DEMAND benchmark.

> Developed as a practical assignment for the Speech Processing course (CE Department, Sharif University of Technology, Fall 2025).

## Dataset

[VoiceBank + DEMAND (16 kHz)](https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k) — a standard supervised speech-enhancement benchmark of clean utterances paired with versions corrupted by real-world noise at various SNRs. All audio is processed at 16 kHz mono.

## Part 1 — Classical DSP Baselines

Both methods operate on the STFT (512-point FFT, 128 hop, Hann window), enhance the magnitude, and reconstruct using the **noisy phase**.

- **Spectral Subtraction** — estimates a stationary noise magnitude from the first few frames and subtracts a scaled version (`|Ŝ| = max(|X| − α|N̂|, 0)`) from the noisy magnitude.
- **Wiener Filtering** — estimates speech and noise power spectra from the noisy signal and applies the MMSE gain `H = P_s / (P_s + P_n)`, a soft, continuous attenuation.

**Evaluation:** Scale-Invariant Signal-to-Distortion Ratio (SI-SDR), reported for noisy vs. enhanced along with the improvement, plus clean/noisy/enhanced spectrograms and audio playback.

The notebook also discusses the known limitations both methods share: musical noise (tonal "chirp" artifacts from hard flooring, mostly in pauses and high frequencies), failure at low SNR (unreliable noise estimates and heavy speech/noise overlap), and the quality ceiling imposed by reusing the noisy phase.

## Part 2 — Deep Learning: Conformer U-Net

The enhancement task is framed as **time-frequency mask estimation** rather than direct waveform prediction:

- **Input:** log-magnitude STFT of the noisy speech.
- **Target:** the Ideal Ratio Mask (IRM), `M = |S| / (|Y| + ε)`, clamped to `[0, 1]`.
- **Enhancement:** `|Ŝ| = M · |Y|`, reconstructed with the noisy phase.
- **Loss:** L1 between predicted and ideal masks.

**Architecture — Conformer U-Net:** a 2D convolutional U-Net (encoder/decoder with skip connections, GroupNorm + GELU conv blocks) whose bottleneck is processed by **Conformer blocks** operating over the time axis. Each Conformer block combines two half-step feed-forward layers, multi-head self-attention (long-range temporal dependencies), and a depthwise-separable convolution module with GLU (local features) — letting the model capture both local time-frequency structure and global context. The output is a sigmoid mask matched in size to the input spectrogram.

**Training:** fixed-length 2-second segments, Adam optimizer (lr 1e-3), gradient clipping, L1 mask loss over several epochs. Evaluation computes SI-SDR across the test set and visualizes the enhanced spectrogram.

## Key Configuration

| Setting | Value |
|---------|-------|
| Sample rate | 16 kHz |
| STFT (`n_fft`/`hop`/`win`) | 512 / 128 / 512 |
| Segment length | 2.0 s |
| Batch size | 8 |
| Learning rate | 1e-3 |
| Model dim / heads / conv kernel | 128 / 4 / 15 |

## Requirements

- Python 3.x with a **CUDA-enabled GPU** (Google Colab recommended)
- `torch`, `torchaudio`, `torchcodec`, `librosa`, `soundfile`
- `datasets` (Hugging Face), `pesq`, `pystoi`, `numpy`, `matplotlib`, `tqdm`
- `ffmpeg` (system package, for audio decoding)

## Usage

1. Open `Speech_Enhancement.ipynb` in Colab with a GPU runtime.
2. Run Part 0 to install dependencies and load VoiceBank + DEMAND.
3. Run Part 1 to apply spectral subtraction and Wiener filtering to an example and compare SI-SDR / spectrograms.
4. Run Part 2 to train the Conformer U-Net and evaluate it on the test set.

## Notes

- All three methods reuse the noisy phase, which caps achievable quality at low SNR; complex-mask or time-domain models would address this.
- The deep model predicts a bounded ratio mask, which avoids the negative-magnitude / hard-flooring artifacts that cause musical noise in the classical baselines.
- SI-SDR is amplitude-invariant and convenient, but does not perfectly track perceived quality — listening and metrics like PESQ/STOI complement it.