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
   - EnCodec acoustic-token extraction path
   - SoundStream kept as an optional later codec path

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
pip install -e ".[dev,whisper,codec]"
```

Codec options:

```bash
pip install -e ".[codec]"       # EnCodec
pip install -e ".[soundstream]" # audiolm-pytorch SoundStream
pip install -e ".[logging]"     # optional TensorBoard logs
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

The current learned path uses pretrained EnCodec acoustic tokens:

```text
audio
-> pretrained EnCodec encoder
-> acoustic tokens z
-> StreamingCodecLatentEditor
-> edited acoustic tokens z'
-> EnCodec decoder
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

Use EnCodec first:

```python
from encodec import EncodecModel
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)
```

Extract EnCodec acoustic tokens from a manifest:

```bash
.venv/bin/python -m speech_comprehension.extract_encodec_tokens \
  manifests/common_voice_cv_valid_train_10_wav.csv \
  --output-dir data/encodec_tokens/common_voice_cv_valid_train_10 \
  --bandwidth 6.0
```

The extractor writes:

```text
data/encodec_tokens/.../tokens/*.pt
data/encodec_tokens/.../index.csv
data/encodec_tokens/.../metadata.json
```

For full validated Common Voice:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m speech_comprehension.extract_encodec_tokens \
  manifests/common_voice_all_valid_mp3.csv \
  --output-dir data/encodec_tokens/common_voice_all_valid \
  --bandwidth 6.0 \
  --device auto
```

SoundStream through `audiolm-pytorch` is optional/later:

```python
from audiolm_pytorch import SoundStream
model = SoundStream(target_sample_hz=16000, codebook_size=1024)
```

The upstream `audiolm-pytorch` path has pretrained EnCodec support, but its
native SoundStream path expects you to train SoundStream or load a checkpoint:

```python
from audiolm_pytorch import SoundStream

soundstream = SoundStream.init_and_load_from("runs/soundstream/soundstream.100000.pt")
```

The actual editor training dataset is now:

```text
manifest audio
-> EnCodec acoustic tokens z
-> ASR/listener confidence labels
-> train listener proxy
-> freeze listener proxy
-> train latent editor
```

## Optional: Train A SoundStream Codec

SoundStream gives acoustic codec tokens: compact reconstructable audio tokens
that preserve speaker, timing, prosody, and local pronunciation detail. A fresh
`audiolm-pytorch` SoundStream is randomly initialized, so train or load a codec
before treating its tokens as meaningful.

Train on the Common Voice valid-train audio folder:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m speech_comprehension.train_soundstream \
  --folder ../common-voice/cv-valid-train/cv-valid-train \
  --results-folder runs/soundstream_common_voice_valid_train \
  --num-train-steps 100000 \
  --batch-size 4 \
  --grad-accum-every 4 \
  --data-max-length-seconds 3 \
  --codebook-size 1024 \
  --rq-num-quantizers 8 \
  --multi-spectral-recon-loss-weight 0 \
  --mixed-precision fp16 \
  --tensorboard
```

For a smaller smoke test:

```bash
.venv/bin/python -m speech_comprehension.train_soundstream \
  --folder ../common-voice/cv-valid-train/cv-valid-train \
  --results-folder runs/soundstream_smoke \
  --num-train-steps 20 \
  --batch-size 1 \
  --data-max-length-seconds 1 \
  --multi-spectral-recon-loss-weight 0
```

The wrapper defaults `--multi-spectral-recon-loss-weight` to `0`. The upstream
`audiolm-pytorch` implementation computes `log(mel)` without an epsilon, so
near-silent speech chunks can make that auxiliary loss `inf` or `nan`. Once the
basic codec run is stable, try a small value such as `1e-6` or `1e-5` on a
cleaned/non-silent dataset.

To resume or fine-tune an existing SoundStream checkpoint:

```bash
.venv/bin/python -m speech_comprehension.train_soundstream \
  --folder ../common-voice/cv-valid-train/cv-valid-train \
  --soundstream-checkpoint runs/soundstream_common_voice_valid_train/soundstream.100000.pt \
  --results-folder runs/soundstream_finetune \
  --num-train-steps 10000
```

Each training run writes Lightning-style local logs:

```text
runs/.../logs/version_N/hparams.json
runs/.../logs/version_N/metrics.jsonl
runs/.../logs/version_N/metrics.csv
runs/.../logs/version_N/tensorboard/   # when --tensorboard is enabled
```

Open TensorBoard with:

```bash
.venv/bin/tensorboard --logdir runs
```

This trains the codec only. The intelligibility-repair model still needs the
next stages:

```text
pretrained EnCodec tokens
-> acoustic tokens or latents z
-> ASR confidence / WER labels
-> listener proxy
-> sparse latent editor
-> EnCodec decode
```

## Acoustic vs Semantic Tokens

For this project, acoustic tokens are required because they can be decoded back
to audio. EnCodec tokens are the default acoustic tokens now.

Semantic tokens are still useful, but as conditioning or supervision. In
`audiolm-pytorch`, semantic tokens usually come from:

```python
from audiolm_pytorch import HubertWithKmeans

semantic_tokenizer = HubertWithKmeans(
    checkpoint_path="checkpoints/hubert_base_ls960.pt",
    kmeans_path="checkpoints/hubert_base_ls960_L9_km500.bin",
    target_sample_hz=16000,
)
```

They represent content/phonetic structure more than waveform detail. They can
help the editor decide where a pronunciation is confusing, but they should not
replace EnCodec for reconstruction. The safest MVP is not full AudioLM
generation; it is sparse editing of original acoustic tokens, optionally
conditioned on semantic tokens.

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

Latest workflow uses pretrained EnCodec acoustic tokens, keeps SoundStream as
optional/later, supports full Common Voice manifests, and reads MP3 directly.
