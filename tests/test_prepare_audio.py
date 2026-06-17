from __future__ import annotations

import csv
from pathlib import Path

from speech_comprehension.prepare_audio import convert_manifest_audio


def test_convert_manifest_audio_dry_run_rewrites_audio_paths(tmp_path: Path) -> None:
    source_audio = tmp_path / "source.mp3"
    source_audio.write_bytes(b"fake mp3")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "id,audio,transcript,accent\n"
        f"utt-1,{source_audio},hello world,india\n",
        encoding="utf-8",
    )
    output_manifest = tmp_path / "converted.csv"
    audio_dir = tmp_path / "wav"

    count = convert_manifest_audio(
        manifest,
        output_manifest,
        audio_dir=audio_dir,
        dry_run=True,
    )

    assert count == 1
    with output_manifest.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["audio"] == str(audio_dir / "utt-1.wav")
    assert row["transcript"] == "hello world"
    assert row["accent"] == "india"
