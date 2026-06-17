# Phoneme-Level Editor

The target system is not the DSP edit bank. The DSP code is only a conservative
baseline for measuring whether ASR-guided minimal repair is worth pursuing.

The intended model is a learned phoneme-conditioned editor:

```text
audio
-> speech encoder / codec encoder
-> frame features
-> phoneme alignment per frame
-> accent embedding
-> sparse residual editor
-> codec decoder / vocoder
-> edited audio
```

## What The Model Learns

The editor should learn:

```text
which phoneme frames reduce comprehension
how much to edit those frames
how to preserve the same speaker, accent, timing, emotion, and words
```

It should not learn a broad mapping like:

```text
Indian English -> American English
```

It should learn sparse repair:

```text
unclear /t/ closure, vowel contrast, fricative, stress cue, etc.
-> minimally clearer version of the same speaker's accented speech
```

## Current Scaffold

The repository now includes:

```text
speech_comprehension/phoneme_editor.py
```

It implements a trainable PyTorch module:

```text
StreamingPhonemeEditor
```

Inputs:

```text
acoustic_features: [batch, frames, acoustic_dim]
phoneme_ids:       [batch, frames]
accent_ids:        [batch]
```

Outputs:

```text
edited_features = acoustic_features + edit_gate * residual
```

The gate is important. It forces the model to learn:

```text
edit only where needed
```

not:

```text
rewrite the entire utterance
```

## Training Losses

The scaffold exposes losses for:

- intelligibility, supplied by ASR/CTC/transcript loss
- speaker preservation, via speaker embedding cosine distance
- minimal edit, via feature L1 distance
- gate sparsity, so edits stay local
- residual smoothness, so edits do not create artifacts

Conceptually:

```text
loss =
  ASR / phoneme intelligibility loss
  + speaker preservation
  + minimal feature edit
  + sparse edit gate
  + smooth residual
```

## Missing Pieces Before Real Training

To train this properly we still need:

1. A phoneme alignment stage.
   - forced alignment from transcript to frames, or
   - CTC phoneme alignment from a pretrained model

2. A feature/audio backend.
   - HuBERT / wav2vec2 features for analysis, or
   - neural codec features if we want clean reconstruction

3. A decoder.
   - codec decoder or vocoder that turns edited features into waveform

4. A real training script.
   - manifest -> aligned frames -> feature batches -> model/loss -> checkpoint

The important pivot is now represented in code: the learned editor predicts
phoneme-conditioned residual repairs rather than hardcoded acoustic effects.
