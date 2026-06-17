from __future__ import annotations

from types import SimpleNamespace

from speech_comprehension.asr import _whisper_supports_fp16


def test_whisper_fp16_enabled_only_on_cuda() -> None:
    assert _whisper_supports_fp16(SimpleNamespace(device=SimpleNamespace(type="cuda")))
    assert not _whisper_supports_fp16(SimpleNamespace(device=SimpleNamespace(type="cpu")))
    assert not _whisper_supports_fp16(object())
