from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from speech_comprehension.audio import AudioSignal, read_wav, write_wav


def test_read_wav_decodes_mp3_with_ffmpeg(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")

    sample_rate = 16000
    t = np.arange(sample_rate // 4, dtype=np.float32) / sample_rate
    signal = AudioSignal(0.05 * np.sin(2.0 * np.pi * 220.0 * t), sample_rate)
    wav_path = tmp_path / "input.wav"
    mp3_path = tmp_path / "input.mp3"
    write_wav(wav_path, signal)

    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(wav_path),
            str(mp3_path),
        ],
        check=True,
    )

    decoded = read_wav(mp3_path)

    assert decoded.sample_rate == sample_rate
    assert decoded.samples.ndim == 1
    assert decoded.duration_seconds > 0.20
