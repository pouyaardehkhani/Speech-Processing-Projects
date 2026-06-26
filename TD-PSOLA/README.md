# TD-PSOLA: Time-Domain Pitch-Synchronous Overlap-Add

A from-scratch implementation of the **TD-PSOLA** algorithm for speech manipulation. TD-PSOLA modifies the **pitch** and **duration** of a speech signal independently while preserving its timbre and formant structure, by working directly in the time domain on pitch-synchronized grains. The core algorithm (pitch-mark detection, grain extraction, overlap-add synthesis) is implemented manually rather than using a ready-made PSOLA library.

## What It Does

- **Pitch modification** — raise or lower pitch (≈ ±50%) while keeping duration unchanged.
- **Duration modification** — speed up or slow down speech (≈ 0.5× to 2×) while keeping pitch unchanged.
- **Combined modification** — change pitch and duration independently in one pass.
- **Timbre/formant preservation** — because grains are taken directly from the original waveform and only repositioned, the spectral envelope (and therefore vowel identity / voice character) is preserved.

## How It Works

TD-PSOLA proceeds in four stages:

1. **F0 estimation** — fundamental frequency is estimated with librosa's pYIN, with NaN (unvoiced) values interpolated so a continuous period track is available.
2. **Pitch-mark detection** — pitch-synchronous points are located by interpolating F0 to sample resolution to get the expected local period, then searching for the strongest signal peak within ±30% of that period from the previous mark (with a uniform fallback when no peaks are found). This is the heart of the algorithm and is implemented manually.
3. **Grain extraction** — a Hanning-windowed segment ("grain") spanning roughly ±1 pitch period is extracted around each pitch mark.
4. **Overlap-add synthesis** — new pitch-mark positions are computed by rescaling the inter-mark periods (`new_period = period × time_stretch / pitch_shift`), and the grains are overlap-added at those new positions to produce the output.

For **pitch shifting**, grains are repositioned closer together (higher pitch) or farther apart (lower pitch); the time-stretch factor is set equal to the pitch-shift factor to cancel the duration change, so only pitch changes. For **time stretching**, the same grains are duplicated/spaced out without changing the spacing-derived pitch, so only duration changes.

## API

The module exposes a single `TDPSOLA` class:

```python
from td_psola import TDPSOLA
import librosa

sr = 16000
signal, _ = librosa.load("speech.wav", sr=sr)

psola = TDPSOLA(sr=sr)

# Raise pitch by ~20%, keep duration
higher = psola.modify_pitch(signal, pitch_shift=1.2)

# Slow down to 1.5× length, keep pitch
slower = psola.modify_duration(signal, time_stretch=1.5)

# Both at once
both = psola.modify_pitch_and_duration(signal, pitch_shift=0.8, time_stretch=1.3)
```

| Method | Effect |
|--------|--------|
| `estimate_f0(signal, fmin, fmax)` | F0 track via pYIN (autocorrelation-family) |
| `detect_pitch_marks(signal, f0_time, f0)` | Pitch-synchronous sample indices |
| `extract_grains(signal, pitch_marks)` | Hanning-windowed grains around each mark |
| `synthesize_psola(grains, marks, pitch_shift, time_stretch)` | Overlap-add resynthesis |
| `modify_pitch(signal, pitch_shift)` | Pitch change, duration preserved |
| `modify_duration(signal, time_stretch)` | Duration change, pitch preserved |
| `modify_pitch_and_duration(signal, pitch_shift, time_stretch)` | Independent pitch + duration change |

**Factor conventions:** `pitch_shift` of 2.0 = one octave up, 0.5 = one octave down; `time_stretch` of 2.0 = twice as long, 0.5 = half as long. Outputs are amplitude-normalized to the input's peak.

## Requirements

- Python 3.x
- `numpy`, `scipy`, `librosa`

```bash
pip install numpy scipy librosa
```

## Notes

- Default sampling rate is 16 kHz; pass `sr` to match your audio.
- Quality is best on clean, voiced speech; very large pitch/time factors and heavily unvoiced or noisy segments will show more artifacts, as is expected for TD-PSOLA.
- F0 estimation uses librosa's pYIN (permitted as a building block); the PSOLA core — pitch marks, grains, and overlap-add — is written from scratch.