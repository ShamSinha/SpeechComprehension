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
    assert row["audio"] == "wav/utt-1.wav"
    assert row["transcript"] == "hello world"
    assert row["accent"] == "india"


def test_convert_manifest_audio_handles_repo_relative_input_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_audio = tmp_path / "data" / "source.mp3"
    source_audio.parent.mkdir()
    source_audio.write_bytes(b"fake mp3")
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    manifest = manifest_dir / "manifest.csv"
    manifest.write_text(
        "id,audio,transcript\n"
        "utt-1,data/source.mp3,hello world\n",
        encoding="utf-8",
    )
    output_manifest = manifest_dir / "converted.csv"

    monkeypatch.chdir(tmp_path)
    count = convert_manifest_audio(
        manifest,
        output_manifest,
        audio_dir=tmp_path / "wav",
        dry_run=True,
    )

    assert count == 1
    with output_manifest.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["audio"] == "../wav/utt-1.wav"
