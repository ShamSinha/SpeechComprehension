from __future__ import annotations

import inspect

import pytest

audiolm_pytorch = pytest.importorskip("audiolm_pytorch")


def test_audiolm_pytorch_exposes_soundstream() -> None:
    assert hasattr(audiolm_pytorch, "SoundStream")
    signature = inspect.signature(audiolm_pytorch.SoundStream)
    assert "target_sample_hz" in signature.parameters
