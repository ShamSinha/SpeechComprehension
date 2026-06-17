# Codec Latent Editor

This is the practical editing path.

The system should not learn direct waveform editing:

```text
audio samples -> neural net -> edited samples
```

Instead, use a pretrained neural codec:

```text
audio
-> codec encoder
-> continuous latents z
-> learned editor
-> edited latents z'
-> codec decoder
-> edited audio
```

Examples of suitable codecs include EnCodec and SoundStream-style models.

## Why Codec Latents

Codec latents are editable in a way waveforms are not. A one-second waveform at
16 kHz has 16,000 samples. A codec representation might have hundreds of frames
with a moderate feature dimension.

That changes the problem from:

```text
edit a raw waveform
```

to:

```text
edit a compact speech representation
```

## Current Implementation

The repo includes:

```text
speech_comprehension/codec_editor.py
```

The main editor is:

```text
StreamingCodecLatentEditor
```

It implements:

```text
z' = z + gate(z, accent) * residual(z, accent)
```

The explicit gate is important. It makes sparse editing trainable:

```text
gate near 0 -> keep original latent
gate near 1 -> apply repair residual
```

The editor is causal by default, so it can be adapted to streaming.

## The No-Edit Collapse

If the only losses are:

```text
transcript preservation
speaker preservation
minimal edit
```

then the best solution is:

```text
z' = z
```

That is why the implementation also includes:

```text
LatentListenerProxy
```

This proxy predicts a confidence/intelligibility score from latents. It can be
trained from offline labels such as:

```text
Whisper confidence
1 - WER
human listener score
```

Then freeze it and train the editor with a preference loss:

```text
score(z') > score(z)
```

while keeping edits small.

## Training Stages

Stage 1: train listener proxy.

```text
audio
-> codec encoder
-> z
-> listener proxy
-> predicted confidence

target = ASR confidence or 1 - WER
```

Stage 2: freeze listener proxy and train editor.

```text
z
-> editor
-> z'

loss =
  listener preference: score(z') > score(z)
  + minimal latent edit: z' close to z
  + gate sparsity
  + residual smoothness
  + speaker preservation
  + transcript preservation after decoding
```

Stage 3: decode and evaluate.

```text
z'
-> codec decoder
-> edited audio
-> ASR / listener eval
```

## What Is Still Needed

The repository now has the trainable editor and listener-proxy modules, but real
training still needs:

1. A codec backend wrapper around EnCodec or another codec.
2. A latent dataset builder:
   ```text
   manifest audio -> codec latents -> ASR confidence labels
   ```
3. A training loop for:
   - listener proxy pretraining
   - editor training with frozen listener proxy
4. Optional speaker embedding loss after decoding.

This is the right architecture for actual speech editing. The old DSP edit bank
should be treated as a quick baseline only.
