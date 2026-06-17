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

Use pretrained EnCodec first. It gives us acoustic tokens immediately, without
spending time training a codec from scratch. SoundStream stays as a later option
if we need a codec trained specifically on our data.

Install codec backends with:

```bash
pip install -e ".[codec]"
pip install -e ".[soundstream]"
```

The current MVP codec path is EnCodec:

```python
from encodec import EncodecModel

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)
```

Extract acoustic tokens from a manifest:

```bash
.venv/bin/python -m speech_comprehension.extract_encodec_tokens \
  manifests/common_voice_cv_valid_train_10_wav.csv \
  --output-dir data/encodec_tokens/common_voice_cv_valid_train_10 \
  --bandwidth 6.0
```

SoundStream through `audiolm-pytorch` is optional/later:

```python
from audiolm_pytorch import SoundStream
soundstream = SoundStream(target_sample_hz=16000, codebook_size=1024)
```

`audiolm-pytorch` also exposes AudioLM-style semantic-token components such as
`HubertWithKmeans`. Those are useful as conditioning signals, but the editable
representation should remain codec/acoustic latents because they can be decoded
back to audio.

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
