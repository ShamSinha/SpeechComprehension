from __future__ import annotations

import csv
from pathlib import Path

from speech_comprehension.extract_encodec_tokens import (
    EncodecTokenRecord,
    _resolve_manifest_path,
    _safe_sample_id,
    _write_index,
    build_parser,
)


def test_extract_encodec_parser_defaults() -> None:
    args = build_parser().parse_args(["manifest.csv"])

    assert args.manifest == Path("manifest.csv")
    assert args.output_dir == Path("data/encodec_tokens")
    assert args.model == "24khz"
    assert args.bandwidth == 6.0
    assert args.device == "auto"


def test_safe_sample_id_removes_path_unfriendly_characters() -> None:
    assert _safe_sample_id(" speaker 1 / utt:42 ") == "speaker_1_utt_42"
    assert _safe_sample_id("...") == "sample"


def test_extract_encodec_resolves_repo_relative_manifest_audio(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audio = tmp_path / "data" / "sample.wav"
    audio.parent.mkdir()
    audio.write_bytes(b"fake wav")
    manifest = tmp_path / "manifests" / "manifest.csv"
    manifest.parent.mkdir()
    manifest.write_text("id,audio,transcript\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    assert _resolve_manifest_path(manifest, "data/sample.wav") == audio


def test_write_encodec_index(tmp_path: Path) -> None:
    index_path = tmp_path / "index.csv"
    _write_index(
        index_path,
        [
            EncodecTokenRecord(
                id="utt-1",
                audio="/audio/utt-1.wav",
                token_path="tokens/utt-1.pt",
                transcript="hello world",
                accent="india",
                speaker="spk",
                source="common-voice",
                split="train",
                duration_seconds=1.2,
                codec_model="encodec_24khz",
                sample_rate=24000,
                bandwidth=6.0,
                num_codebooks=8,
                num_frames=90,
            )
        ],
    )

    with index_path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))

    assert row["id"] == "utt-1"
    assert row["token_path"] == "tokens/utt-1.pt"
    assert row["num_codebooks"] == "8"
