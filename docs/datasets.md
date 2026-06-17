# Dataset Manifests

The experiment runner expects a CSV with at least:

```csv
id,audio,transcript
sample-001,/path/to/sample.wav,the world is changing
```

The repository includes a local manifest builder for the two Kaggle datasets:

- Mozilla Common Voice: `mozillaorg/common-voice`
- L2-ARCTIC data: `divyamagg/l2-arctic-data`

Use the Kaggle environment for download commands:

```bash
/data/shubham/kaggle_env/bin/python -m kaggle datasets download \
  -d mozillaorg/common-voice \
  -p data/kaggle/common_voice \
  --unzip

/data/shubham/kaggle_env/bin/python -m kaggle datasets download \
  -d divyamagg/l2-arctic-data \
  -p data/kaggle/l2_arctic \
  --unzip
```

Kaggle credentials must already be available to that environment.

## Common Voice

Common Voice releases use different layouts. The Mozilla Kaggle dataset at
`mozillaorg/common-voice` uses CSV files such as `cv-valid-train.csv` and MP3
folders such as `cv-valid-train/cv-valid-train/`. Newer Common Voice releases
often use TSV files and a `clips/` directory. The manifest builder supports
both shapes.

Build an MP3 manifest from the Kaggle layout:

```bash
speech-build-manifest \
  --dataset common-voice \
  --root data/kaggle/common_voice \
  --split cv-valid-train \
  --max-rows 100 \
  --output manifests/common_voice_cv_valid_train_100_mp3.csv
```

Convert that manifest to mono 16 kHz WAV:

```bash
speech-convert-manifest-audio \
  manifests/common_voice_cv_valid_train_100_mp3.csv \
  manifests/common_voice_cv_valid_train_100_wav.csv \
  --audio-dir data/converted/common_voice_cv_valid_train_100
```

Then run:

```bash
speech-repair-experiment manifests/common_voice_cv_valid_train_100_wav.csv \
  --output-dir outputs/common_voice_cv_valid_train_100 \
  --asr whisper \
  --model small
```

## L2-ARCTIC

L2-ARCTIC is already WAV-based in common releases. The builder scans speaker
folders, finds WAV files, and matches transcripts from `txt.done.data` or
nearby transcript files.

Build a Hindi-filtered manifest if metadata is present:

```bash
speech-build-manifest \
  --dataset l2-arctic \
  --root data/kaggle/l2_arctic \
  --accent-contains hindi \
  --output manifests/l2_arctic_hindi.csv
```

If the local copy does not include speaker metadata, provide one:

```csv
speaker,accent
ABA,arabic
BWC,hindi
```

Then run:

```bash
speech-build-manifest \
  --dataset l2-arctic \
  --root data/kaggle/l2_arctic \
  --speaker-accent-map manifests/l2_arctic_speakers.csv \
  --accent-contains hindi \
  --output manifests/l2_arctic_hindi.csv
```

The output manifest can be passed directly to `speech-repair-experiment`.
