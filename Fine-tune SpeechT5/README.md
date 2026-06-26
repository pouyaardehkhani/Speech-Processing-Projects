# Fine-tuning SpeechT5 for Persian Text-to-Speech

Fine-tuning **Microsoft SpeechT5** for **Persian TTS** (text-to-speech). SpeechT5 is a transformer model that predicts mel-spectrograms from input text; a **HiFi-GAN** vocoder then converts those spectrograms into audible waveforms. This notebook covers the full pipeline from data loading and tokenizer extension through training and synthesis.

> Developed as a practical assignment for the Speech Processing course (CE Department, Sharif University of Technology, Fall 2025).

## Pipeline

1. **Install & verify dependencies** (transformers, datasets, speechbrain, torchcodec, etc.).
2. **Load the dataset** and cast audio to a consistent 16 kHz sampling rate.
3. **Clean and normalize text**, then **extend the tokenizer** with any Persian characters missing from SpeechT5's vocabulary (resizing the model's token embeddings to match).
4. **Extract speaker embeddings (x-vectors)** from each recording using a pretrained SpeechBrain speaker-recognition model.
5. **Prepare inputs/labels** — tokenize text, extract target spectrogram features, attach speaker embeddings, and filter out samples that exceed the model's text/speech length limits.
6. **Fine-tune** SpeechT5 with Hugging Face's `Seq2SeqTrainer` and a custom padding collator.
7. **Run inference** on test samples and listen to the synthesized speech.

## Models Used

- **`microsoft/speecht5_tts`** — the text-to-speech model being fine-tuned.
- **`microsoft/speecht5_hifigan`** — HiFi-GAN vocoder (spectrogram → waveform).
- **`speechbrain/spkrec-xvect-voxceleb`** — pretrained x-vector speaker encoder producing 512-dim speaker embeddings, L2-normalized for stability.

## Dataset

A Persian TTS dataset (`mohammadhossein/SP_HW5_PersianTTS`) of paired text transcripts and audio recordings, with per-speaker IDs (`client_id`). Audio is resampled to 16 kHz. The notebook analyzes the per-speaker sample distribution and splits the data 95% train / 5% test.

## Key Implementation Details

- **Tokenizer extension:** the dataset's character set is compared against the tokenizer's; certain characters are normalized (e.g. `š → ش`, `ā → آ`), and remaining missing Persian characters are added as new tokens, after which the model's embeddings are resized.
- **Speaker embeddings:** computed per utterance with the SpeechBrain x-vector model so the model learns speaker-conditioned synthesis.
- **Length filtering:** samples with more than 600 input tokens, or with target spectrograms longer than the model's `max_speech_positions` (adjusted to the decoder reduction factor), are dropped.
- **Data collator:** dynamically pads text and spectrogram targets, masks padded spectrogram frames with `-100.0` for correct loss, aligns target lengths to the reduction factor, and stacks speaker embeddings.

## Training Configuration

| Setting | Value |
|---------|-------|
| Batch size (train/eval) | 8 / 8 |
| Gradient accumulation | 2 |
| Learning rate | 1e-4 |
| Warmup steps | 200 |
| Max steps | 6000 |
| Eval / save interval | every 200 steps |
| Mixed precision | fp16 (if CUDA available) |
| Best-model selection | lowest `eval_loss` |

## Requirements

- Python 3.x with a **CUDA-enabled GPU** (Google Colab recommended)
- `transformers`, `datasets`, `soundfile`, `speechbrain`, `accelerate`, `torchcodec==0.7.0`, `torch`, `matplotlib`

```bash
pip install transformers datasets soundfile speechbrain accelerate "torchcodec==0.7.0"
```

## Usage

1. Open `fine_tune_speecht5.ipynb` in Colab with a GPU runtime.
2. Run the cells in order to install dependencies, load the model and dataset, extend the tokenizer, and prepare features.
3. Run the training cell (`trainer.train()`); checkpoints are written to `speecht5_persian_tts/`.
4. Run the inference cells to synthesize speech for random test samples and play the audio inline.

## Notes

- Speaker embeddings condition the voice — to synthesize new text in a target speaker's voice, reuse that speaker's embedding.
- `model.config.use_cache` is disabled during training (required when using gradient checkpointing / the trainer setup).
- Synthesis quality depends heavily on the amount and quality of training data and the number of training steps; 6000 steps is a starting point.