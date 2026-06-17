from __future__ import annotations

import csv
from pathlib import Path

from speech_comprehension.datasets import common_voice_rows, l2_arctic_rows, write_manifest


def test_common_voice_manifest_rows_from_tsv(tmp_path: Path) -> None:
    root = tmp_path / "common_voice"
    clips = root / "clips"
    clips.mkdir(parents=True)
    (clips / "sample.wav").write_bytes(b"fake wav")
    (root / "validated.tsv").write_text(
        "client_id\tpath\tsentence\taccent\tlocale\n"
        "speaker-1\tsample.mp3\tThe world is changing.\tIndia\ten\n",
        encoding="utf-8",
    )

    rows = common_voice_rows(
        root,
        split="validated",
        audio_extension="wav",
        accent_contains="india",
    )

    assert len(rows) == 1
    assert rows[0].audio == str(clips / "sample.wav")
    assert rows[0].id == "validated_sample"
    assert rows[0].transcript == "The world is changing."
    assert rows[0].accent == "India"
    assert rows[0].speaker == "speaker-1"
    assert rows[0].source == "common-voice"


def test_common_voice_all_valid_combines_kaggle_csv_splits(tmp_path: Path) -> None:
    root = tmp_path / "common_voice"
    for split in ["cv-valid-train", "cv-valid-dev", "cv-valid-test"]:
        audio_dir = root / split / split
        audio_dir.mkdir(parents=True)
        (audio_dir / "sample.mp3").write_bytes(b"fake mp3")
        (root / f"{split}.csv").write_text(
            "filename,text,accent\n"
            f"{split}/sample.mp3,hello from {split},india\n",
            encoding="utf-8",
        )

    rows = common_voice_rows(root, split="all-valid")

    assert [row.split for row in rows] == [
        "cv-valid-dev",
        "cv-valid-test",
        "cv-valid-train",
    ]
    assert all(row.audio.endswith("sample.mp3") for row in rows)


def test_l2_arctic_manifest_rows_from_txt_done_data(tmp_path: Path) -> None:
    root = tmp_path / "l2_arctic"
    speaker = root / "ABA"
    wav_dir = speaker / "wav"
    etc_dir = speaker / "etc"
    wav_dir.mkdir(parents=True)
    etc_dir.mkdir()
    (wav_dir / "arctic_a0001.wav").write_bytes(b"fake wav")
    (etc_dir / "txt.done.data").write_text(
        '( arctic_a0001 "The birch canoe slid on the smooth planks." )\n',
        encoding="utf-8",
    )

    rows = l2_arctic_rows(root, accent_label="arabic")

    assert len(rows) == 1
    assert rows[0].id == "ABA_arctic_a0001"
    assert rows[0].audio == str(wav_dir / "arctic_a0001.wav")
    assert rows[0].transcript == "The birch canoe slid on the smooth planks."
    assert rows[0].accent == "arabic"
    assert rows[0].speaker == "ABA"
    assert rows[0].source == "l2-arctic"


def test_write_manifest_uses_experiment_columns(tmp_path: Path) -> None:
    rows = common_voice_rows(_common_voice_fixture(tmp_path), audio_extension="wav")
    output = tmp_path / "manifest.csv"

    write_manifest(output, rows)

    with output.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [
            "id",
            "audio",
            "transcript",
            "accent",
            "speaker",
            "source",
            "split",
        ]
        row = list(reader)[0]
    assert row["id"] == "validated_sample"
    assert row["transcript"] == "The world is changing."


def _common_voice_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "cv_fixture"
    clips = root / "clips"
    clips.mkdir(parents=True)
    (clips / "sample.wav").write_bytes(b"fake wav")
    (root / "validated.tsv").write_text(
        "client_id\tpath\tsentence\taccent\n"
        "speaker-1\tsample.mp3\tThe world is changing.\tIndia\n",
        encoding="utf-8",
    )
    return root
