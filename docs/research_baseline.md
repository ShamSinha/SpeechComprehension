# Research Baseline

## Problem

The system should improve comprehension without converting an Indian accent into
an American accent.

Formally:

```text
Audio -> Audio'
```

subject to:

- same transcript
- same speaker identity
- same accent identity
- higher intelligibility proxy score
- minimal acoustic intervention

## Proxy

Until listener data exists, the baseline uses ASR confidence as the proxy for
human comprehension. Low-confidence words or segments become candidate repair
regions.

This is imperfect but useful: it gives the project a measurable loop while
keeping the objective aligned with minimal intelligibility repair.

## Optimization Loop

1. Transcribe the original audio.
2. Find low-confidence spans.
3. Generate conservative local edits for each span.
4. Re-transcribe each candidate.
5. Accept a candidate only when:
   - local ASR confidence improves
   - transcript similarity remains high against the dataset transcript when
     available, otherwise against the original ASR transcript
   - total edited duration remains under budget
   - `ComprehensionScore - lambda * EditCost` improves

Without a reference transcript, `ComprehensionScore` is ASR confidence. With a
reference transcript, it is ASR confidence minus a small WER penalty.

When a ground-truth transcript is provided, the baseline reports:

```text
WER(original ASR, reference)
vs
WER(edited ASR, reference)
```

This matters because the original ASR transcript may contain the very confusion
the system is supposed to repair.

## MVP Dataset Experiment

Use a CSV manifest with at least:

```csv
id,audio,transcript
sample-001,data/sample-001.wav,the world is changing
```

Run:

```bash
speech-repair-experiment manifest.csv --output-dir outputs/mvp
```

The pass/fail signal for the first experiment is:

- mean edited WER is lower than mean original WER
- mean ASR confidence increases
- accepted edits remain sparse
- listening checks confirm the speaker still sounds like the same accented
  speaker

## Why This Is Not Accent Conversion

Accent conversion typically learns a broad mapping:

```text
Indian pronunciation -> American pronunciation
```

This baseline instead searches for the smallest useful local repair:

```text
unclear Indian pronunciation -> slightly clearer Indian pronunciation
```

The edit cost and span budget are not implementation details; they are what keep
the project faithful to the product idea.

## Replaceable Module

The current edit bank is DSP-based. It can be replaced by a learned phoneme/span
repair model that proposes candidates, while the acceptance loop remains the
same.
