from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AudioSignal:
    samples: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError("AudioSignal expects mono audio")
        object.__setattr__(self, "samples", np.clip(samples, -1.0, 1.0))

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return float(len(self.samples)) / float(self.sample_rate)

    def span_to_samples(self, start: float, end: float) -> tuple[int, int]:
        start_i = max(0, min(len(self.samples), int(round(start * self.sample_rate))))
        end_i = max(start_i, min(len(self.samples), int(round(end * self.sample_rate))))
        return start_i, end_i


def read_wav(path: str | Path) -> AudioSignal:
    path = Path(path)
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sample_width == 1:
        raw = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        audio = (raw - 128.0) / 128.0
    elif sample_width == 2:
        raw = np.frombuffer(frames, dtype="<i2").astype(np.float32)
        audio = raw / 32768.0
    elif sample_width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        signed = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
        audio = signed.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        raw = np.frombuffer(frames, dtype="<i4").astype(np.float32)
        audio = raw / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return AudioSignal(audio, sample_rate)


def write_wav(path: str | Path, signal: AudioSignal) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(signal.samples, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(signal.sample_rate)
        wav.writeframes(pcm16.tobytes())


def rms(samples: np.ndarray) -> float:
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float32))) + 1e-12))


def replace_span(
    signal: AudioSignal,
    start: float,
    end: float,
    replacement: np.ndarray,
    crossfade_seconds: float = 0.006,
) -> AudioSignal:
    start_i, end_i = signal.span_to_samples(start, end)
    original_len = end_i - start_i
    if original_len <= 0:
        return signal

    replacement = np.asarray(replacement, dtype=np.float32)
    replacement = match_length(replacement, original_len)
    replacement = np.clip(replacement, -1.0, 1.0)

    fade_n = min(int(round(crossfade_seconds * signal.sample_rate)), original_len // 2)
    if fade_n > 0:
        left = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
        right = np.linspace(1.0, 0.0, fade_n, dtype=np.float32)
        segment = signal.samples[start_i:end_i].copy()
        replacement[:fade_n] = replacement[:fade_n] * left + segment[:fade_n] * (1.0 - left)
        replacement[-fade_n:] = replacement[-fade_n:] * right + segment[-fade_n:] * (1.0 - right)

    samples = signal.samples.copy()
    samples[start_i:end_i] = replacement
    return AudioSignal(samples, signal.sample_rate)


def match_length(samples: np.ndarray, target_len: int) -> np.ndarray:
    if len(samples) == target_len:
        return samples.astype(np.float32)
    if target_len <= 0:
        return np.zeros(0, dtype=np.float32)
    if len(samples) == 0:
        return np.zeros(target_len, dtype=np.float32)

    source_x = np.linspace(0.0, 1.0, len(samples), dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    return np.interp(target_x, source_x, samples).astype(np.float32)


def windowed_fft_gain(
    samples: np.ndarray,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
    gain_db: float,
) -> np.ndarray:
    if len(samples) < 8:
        return samples.astype(np.float32)

    window = np.hanning(len(samples)).astype(np.float32)
    spectrum = np.fft.rfft(samples * window)
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / sample_rate)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    spectrum[mask] *= float(10.0 ** (gain_db / 20.0))
    repaired = np.fft.irfft(spectrum, n=len(samples)).astype(np.float32)

    original_rms = rms(samples)
    repaired_rms = rms(repaired)
    if repaired_rms > 1e-8 and original_rms > 1e-8:
        repaired *= min(1.35, original_rms / repaired_rms)

    blend = 0.85
    return (blend * repaired + (1.0 - blend) * samples).astype(np.float32)
