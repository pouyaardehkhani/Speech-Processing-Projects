# Speech Processing Projects

A collection of speech and audio processing projects spanning **classical DSP**, **acoustic feature analysis**, and **modern deep learning** — from hand-implemented signal-processing algorithms to transformer-based models for synthesis and recognition. Each project lives in its own directory with its own README, dataset notes, and runnable notebook/script.

## Projects

| Project | Area | Approach |
|---------|------|----------|
| [TD-PSOLA](https://github.com/pouyaardehkhani/Speech-Processing-Projects/tree/master/TD-PSOLA) | Speech manipulation | From-scratch pitch & duration modification via pitch-synchronous overlap-add |
| [Spectral Subtraction](https://github.com/pouyaardehkhani/Speech-Processing-Projects/tree/master/Spectral%20Subtraction) | Speech enhancement | Classic noise-spectrum subtraction (LibriSpeech + NOISEX-92) |
| [Speech Enhancement: DSP vs Deep Learning](https://github.com/pouyaardehkhani/Speech-Processing-Projects/tree/master/Speech%20Enhancement) | Speech enhancement | Spectral subtraction & Wiener filtering vs. a Conformer U-Net mask estimator |
| [Channel Noise in Representation Space](https://github.com/pouyaardehkhani/Speech-Processing-Projects/tree/master/Channel%20Noise%20Effect) | Robustness analysis | Gaussian noise injected into mel / STFT / MFCC representations and reconstructed |
| [MFCC vs PLP](https://github.com/pouyaardehkhani/Speech-Processing-Projects/tree/master/Compare%20MFCC%20and%20PLP) | Acoustic features | Feature comparison for spoken-digit recognition under clean and noisy conditions |
| [Speech Emotion Recognition](https://github.com/pouyaardehkhani/Speech-Processing-Projects/tree/master/Speech%20Emotion%20Recognition) | Sound classification | LSTM over MFCC features across seven emotions |
| [SpeechT5 Persian TTS](https://github.com/pouyaardehkhani/Speech-Processing-Projects/tree/master/Fine-tune%20SpeechT5) | Text-to-speech | Fine-tuning Microsoft SpeechT5 + HiFi-GAN for Persian, with x-vector speaker conditioning |
| [AudioLLM (Toy ASR)](https://github.com/pouyaardehkhani/Speech-Processing-Projects/tree/master/AudioLLM) | Speech recognition | Bridging frozen EnCodec tokens to a frozen LLM via a trainable mapper |

> Folder names above are placeholders — rename the links to match your actual directory names.

## What's Covered

Across these projects the repo touches the breadth of a speech-processing curriculum:

- **Classical DSP** — STFT analysis/synthesis, spectral subtraction, Wiener filtering, and a from-scratch TD-PSOLA implementation (pitch marks, grains, overlap-add).
- **Acoustic features** — MFCC, PLP, mel-spectrograms, and the trade-offs between compact, envelope-focused representations and fine-detail ones.
- **Speech enhancement** — both signal-model-based methods and a learned Conformer U-Net mask estimator, evaluated with SI-SDR.
- **Deep learning for speech** — TTS fine-tuning (SpeechT5 + HiFi-GAN), LSTM-based emotion classification, and an EnCodec→LLM bridging experiment for ASR.
- **Evaluation & robustness** — SI-SDR, WER, MFCC-distortion, and controlled clean/noisy comparisons that probe how each method degrades.
- **Tooling** — PyTorch, Hugging Face Transformers/Datasets, librosa, spafe, SpeechBrain, scikit-learn, and torchaudio.

## Repository Structure

```
Speech-Processing-Projects/
├── TD-PSOLA/
├── Spectral-Subtraction/
├── Speech-Enhancement/
├── Channel-Noise/
├── MFCC-vs-PLP/
├── Speech-Emotion-Recognition/
├── SpeechT5-Persian-TTS/
├── AudioLLM/
└── README.md
```

Each subdirectory contains its own `README.md` with dataset, method, requirements, and usage details specific to that project.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/pouyaardehkhani/Speech-Processing-Projects.git
   cd Speech-Processing-Projects
   ```
2. Open the project you're interested in and follow its README.
3. Most deep-learning projects are notebook-based and assume a CUDA-enabled GPU (Google Colab works well); the DSP projects run on CPU.

## Notes

- Several projects were developed as practical assignments for the Speech Processing course (CE Department, Sharif University of Technology); see each project's README for specifics.
- Environment requirements differ between projects (different audio/ML libraries and versions); install per-project as documented in each README.
