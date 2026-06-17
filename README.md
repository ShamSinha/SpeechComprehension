# SpeechComprehension

Minimal intelligibility repair for accented speech.

This repository implements a conservative baseline for:

```text
input audio -> output audio'

same transcript
same speaker
same accent
higher intelligibility proxy score
minimal acoustic edit distance
```

The key distinction is that this is not accent conversion. The pipeline only
touches low-confidence regions and accepts an edit when it improves ASR
confidence enough to justify the acoustic change.

## Research Objective

For a candidate edited waveform `x'`, the baseline maximizes:

```text
score(x') = ASRConfidence(x') - lambda * EditCost(x, x')
```

with hard constraints:

- transcript similarity must stay high
- edited audio must remain close to the original
- only low-confidence spans are eligible for edits
- total edited duration is capped

That turns "make this sound American" into "make the confusing part slightly
clearer while preserving identity."

## What Is Implemented

- ASR backend abstraction with optional `whisper` and `faster-whisper` support
- low-confidence span detection from word or segment confidence
- conservative span-local DSP candidate edits
- objective-based candidate selection
- JSON repair reports with accepted and rejected candidates
- tests using a deterministic mock ASR backend

The included edit generator is intentionally simple. It tries small local
clarity repairs such as presence-band boost, consonant edge emphasis, and
gentle RMS lift. A learned phoneme/span repair model can replace this module
without changing the objective or evaluation loop.

The intended research direction is the learned phoneme-level editor described
in [docs/phoneme_level_editor.md](docs/phoneme_level_editor.md). The DSP edit
bank is a baseline, not the final speech editing approach.

## Install

```bash
cd SpeechComprehension
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Or install the practical experiment requirements directly:

```bash
pip install -r requirements.txt
```

For real ASR scoring, install one backend:

```bash
pip install -e ".[whisper]"
# or
pip install -e ".[faster-whisper]"
```

## Run

```bash
speech-repair input.wav outputs/repaired.wav \
  --asr whisper \
  --model small \
  --transcript-file transcript.txt \
  --report outputs/repair_report.json
```

Useful knobs:

```bash
speech-repair input.wav outputs/repaired.wav \
  --config configs/default.json \
  --confidence-threshold 0.72 \
  --min-confidence-gain 0.02 \
  --edit-lambda 0.35 \
  --wer-penalty 0.25 \
  --max-edit-ratio 0.06
```

If a dataset transcript is available, pass it with `--transcript` or
`--transcript-file`. The transcript becomes the semantic preservation target,
and the report includes `WER(original)` and `WER(edited)`. With a reference
transcript, candidate selection uses ASR confidence minus a small WER penalty;
without one, it falls back to confidence-only scoring.

## First Experiment

For Kaggle-hosted datasets, see [docs/datasets.md](docs/datasets.md).

Create a CSV manifest:

```csv
id,audio,transcript
sample-001,data/sample-001.wav,the world is changing
```

Then run:

```bash
speech-repair-experiment manifest.csv \
  --output-dir outputs/indian_english_mvp \
  --asr whisper \
  --model small
```

The experiment writes repaired audio, per-sample JSON reports, a summary CSV,
and an aggregate JSON file with mean confidence and WER deltas.

## Report Shape

The report contains:

- reference transcript when provided
- original and final transcript
- original and final ASR confidence
- original and final WER when a reference transcript is available
- detected low-confidence spans
- every candidate edit that was evaluated
- accepted edits and their objective scores

## Notes

ASR confidence is only a proxy for human comprehension. This baseline is useful
because it gives the project a measurable loop before human listener studies
exist. The next research step is to compare ASR-confidence improvements against
human comprehension ratings from Indian and non-Indian listeners.
