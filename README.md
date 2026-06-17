# SpeechComprehension

Minimal intelligibility repair for accented English speech.

The goal is not accent conversion:

```text
Indian English -> American English
```

The goal is sparse repair:

```text
accented speaker audio
-> smallest intelligibility-improving edit
-> same speaker, same accent, same words
```

## Current Project State

The repo now has two layers.

1. Runnable baseline:
   - uses Whisper as an ASR/listener proxy
   - finds low-confidence spans
   - tries conservative local DSP edits
   - accepts only edits that improve the objective
   - produces JSON/CSV reports with WER and confidence deltas

2. Learned editor scaffold:
   - codec-latent editor: `z -> z'`
   - listener-proxy model to avoid the `z' = z` no-edit collapse
   - phoneme-conditioned editor scaffold
   - EnCodec and SoundStream dependency paths

The DSP edit bank is only a baseline. The intended research direction is the
codec-latent editor in [docs/codec_latent_editor.md](docs/codec_latent_editor.md).

## Install

```bash
cd /data/shubham/projects/github/SpeechComprehension
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Editable install with extras:

```bash
pip install -e ".[dev,whisper,codec,soundstream]"
```

Codec options:

```bash
pip install -e ".[codec]"       # EnCodec
pip install -e ".[soundstream]" # audiolm-pytorch SoundStream
```

Verify:

```bash
.venv/bin/python -m pytest -q
```

## Dataset Workflow

The Common Voice Kaggle data is expected at:

```text
../common-voice
```

Build all validated Common Voice rows:

```bash
.venv/bin/python -m speech_comprehension.datasets \
  --dataset common-voice \
  --root ../common-voice \
  --split all-valid \
  --output manifests/common_voice_all_valid_mp3.csv
```

This creates a manifest over:

```text
cv-valid-train
cv-valid-dev
cv-valid-test
```

To include every Common Voice CSV split, including `other` and `invalid`:

```bash
.venv/bin/python -m speech_comprehension.datasets \
  --dataset common-voice \
  --root ../common-voice \
  --split all \
  --output manifests/common_voice_all_mp3.csv
```

The pipeline can read MP3 directly through `ffmpeg`, so do not convert the full
dataset to WAV unless you have a specific reason. For small smoke tests only:

```bash
.venv/bin/python -m speech_comprehension.prepare_audio \
  manifests/common_voice_cv_valid_train_10_mp3.csv \
  manifests/common_voice_cv_valid_train_10_wav.csv \
  --audio-dir data/converted/common_voice_cv_valid_train_10
```

More dataset details are in [docs/datasets.md](docs/datasets.md).

## Run Baseline Experiment

Small smoke test:

```bash
.venv/bin/python -m speech_comprehension.experiment \
  manifests/common_voice_cv_valid_train_10_wav.csv \
  --output-dir outputs/common_voice_cv_valid_train_10 \
  --asr whisper \
  --model base
```

All validated Common Voice:

```bash
.venv/bin/python -m speech_comprehension.experiment \
  manifests/common_voice_all_valid_mp3.csv \
  --output-dir outputs/common_voice_all_valid \
  --asr whisper \
  --model base
```

Looser exploratory settings:

```bash
.venv/bin/python -m speech_comprehension.experiment \
  manifests/common_voice_all_valid_mp3.csv \
  --output-dir outputs/common_voice_all_valid_loose \
  --asr whisper \
  --model base \
  --max-edit-ratio 0.35 \
  --min-confidence-gain 0.005 \
  --edit-lambda 0.10 \
  --transcript-similarity-floor 0.94
```

The experiment writes:

```text
outputs/.../audio/*.repaired.wav
outputs/.../reports/*.json
outputs/.../summary.csv
outputs/.../summary.json
```

## Single-File Repair

```bash
.venv/bin/python -m speech_comprehension.cli \
  input.wav \
  outputs/repaired.wav \
  --asr whisper \
  --model base \
  --transcript-file transcript.txt \
  --report outputs/repair_report.json
```

Useful knobs:

```bash
--confidence-threshold 0.72
--min-confidence-gain 0.02
--edit-lambda 0.35
--wer-penalty 0.25
--max-edit-ratio 0.08
--transcript-similarity-floor 0.96
```

## Learned Codec-Latent Path

The promising architecture is:

```text
audio
-> codec encoder
-> latents z
-> StreamingCodecLatentEditor
-> edited latents z'
-> codec decoder
-> edited audio
```

The editor implements:

```text
z' = z + gate(z, accent) * residual(z, accent)
```

The missing signal in naive training is:

```text
edited audio should be more understandable than original audio
```

So the repo includes:

```text
LatentListenerProxy
```

Train it from ASR confidence / WER / listener labels, freeze it, then train the
editor with:

```text
score(z') > score(z)
+ minimal latent edit
+ sparse edit gate
+ residual smoothness
+ speaker preservation
```

Implemented modules:

```text
speech_comprehension/codec_editor.py
speech_comprehension/phoneme_editor.py
```

Architecture notes:

- [docs/codec_latent_editor.md](docs/codec_latent_editor.md)
- [docs/phoneme_level_editor.md](docs/phoneme_level_editor.md)
- [docs/research_baseline.md](docs/research_baseline.md)

## Codec Backends

EnCodec:

```python
from encodec import EncodecModel
model = EncodecModel.encodec_model_24khz()
```

SoundStream through `audiolm-pytorch`:

```python
from audiolm_pytorch import SoundStream
model = SoundStream(target_sample_hz=16000)
```

These packages are installed as optional dependencies. The actual codec dataset
builder is the next implementation step:

```text
manifest audio
-> codec latents z
-> ASR/listener confidence labels
-> train listener proxy
-> freeze listener proxy
-> train latent editor
```

## Report Shape

Each repair report contains:

- reference transcript when provided
- original and final transcript
- original and final ASR confidence
- original and final WER when a reference transcript is available
- detected low-confidence spans
- evaluated candidates
- accepted edits and objective scores

## GitHub

Remote:

```text
https://github.com/ShamSinha/SpeechComprehension.git
```

Latest workflow includes the codec-latent editor, SoundStream optional
dependency, full Common Voice manifests, and direct MP3 input support.
