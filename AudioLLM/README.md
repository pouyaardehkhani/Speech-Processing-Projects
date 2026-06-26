# AudioLLM: Bridging EnCodec and a Frozen LLM for Toy ASR

A minimal, disk-light pipeline that connects a **frozen audio codec (EnCodec)** to a **frozen language model (DistilGPT-2)** through a small **trainable mapper**, demonstrating how to feed audio into a text LLM and have it generate transcripts. Only the mapper is trained — both the codec and the LLM stay frozen.

## How It Works

The pipeline builds a bridge from waveform to text in three stages:

1. **Audio → discrete tokens.** The raw waveform is resampled to 24 kHz and encoded by [EnCodec](https://huggingface.co/facebook/encodec_24khz) into discrete audio codes. The multi-codebook codes are flattened into a single sequence of audio token IDs (each codebook offset into its own ID range).
2. **Mapper → LLM embedding space.** A small trainable module — an embedding table followed by `LayerNorm + Linear` — projects the audio token IDs into DistilGPT-2's 768-dimensional input embedding space.
3. **Frozen LLM decodes text.** The model is fed the sequence `[audio_embeds] + "transcription: "` and generates the transcript autoregressively.

Because the LLM is frozen, training uses a standard language-model loss computed **only on the transcript tokens** (loss on the audio prefix and the prompt is masked with `-100`). The mapper thus learns to produce an audio-conditioned prefix the LLM can decode into text.

## Architecture

```
waveform (16k/…) ──resample──> 24 kHz
        │
   EnCodec encode (frozen)
        │
   flatten multi-codebook codes ──> audio token IDs
        │
   AudioTokenMapper (TRAINABLE):  Embedding → LayerNorm → Linear → 768-d
        │
   [audio_embeds] + emb("transcription: ")
        │
   DistilGPT-2 (frozen) ──greedy decode──> transcript
```

## Dataset

[LibriSpeech ASR](https://huggingface.co/datasets/openslr/librispeech_asr) (`clean`), loaded with `streaming=True` so the full dataset is never downloaded to disk. Training uses a subset of `train.100` and evaluation a subset of `test`. Transcripts are lowercased and stripped to `[a-z' ]` before scoring.

## Key Configuration

| Parameter | Default | Effect |
|-----------|---------|--------|
| `BANDWIDTH` | 3.0 kbps | Lower → fewer audio tokens → lower latency, lower fidelity |
| `MAX_AUDIO_SEC` | 2.0 s | Crops audio to keep token counts within the context window |
| `MAX_STEPS` | 200 | Training steps over the streamed data |
| `lr` | 2e-4 | AdamW learning rate (mapper only) |

## Results & Trade-offs

The notebook includes a bandwidth sweep that makes the **quality / latency / token-count** trade-off explicit. On a 2-second clip, the audio token count scales directly with bandwidth (≈300 tokens at 1.5 kbps up to ≈4800 at 24 kbps). Higher bandwidth gives a richer audio representation but produces far more tokens, and since transformer attention cost grows with sequence length, **latency is dominated by the LLM token count** rather than EnCodec's encode time (which stayed roughly constant).

A central constraint is DistilGPT-2's **1024-token context window**: high bandwidth or longer audio overflows it and forces truncation, which directly degrades transcription. As a toy ASR demo, the output quality is limited (greedy decoding tends toward repetition, and WER is high) — output quality is governed mainly by codec/token budget, mapper capacity, and the frozen LLM's decoding behavior.

## Requirements

- Python 3.x with a **CUDA-enabled GPU** recommended (Google Colab works)
- `transformers>=4.49.0`, `datasets`, `torchaudio`, `accelerate`, `jiwer`, `torch`

```bash
pip install "transformers>=4.49.0" datasets jiwer torchaudio accelerate
```

## Usage

1. Open `Simple_Pipeline.ipynb` in Google Colab with a GPU runtime.
2. Run the cells in order — they install dependencies, load frozen EnCodec + DistilGPT-2, train the mapper on streamed LibriSpeech, evaluate WER on a test subset, and print the bandwidth trade-off sweep.
3. Tune `BANDWIDTH`, `MAX_AUDIO_SEC`, `N_TRAIN`, and `MAX_STEPS` to explore the quality/latency trade-offs.

## Notes

- This is an educational demonstration of the audio-token → LLM-embedding bridging technique, not a production ASR system. Expect high WER.
- Quality could be improved with a larger/instruction-tuned LLM, a higher-capacity mapper, longer training, beam search or sampling instead of greedy decoding, and a model with a larger context window.
- Models still download (EnCodec + DistilGPT-2); only the dataset is streamed to stay disk-light.