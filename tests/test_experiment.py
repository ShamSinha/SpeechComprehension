from __future__ import annotations

from pathlib import Path

from speech_comprehension.experiment import _resolve_manifest_path


def test_experiment_resolves_repo_relative_manifest_audio(
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
