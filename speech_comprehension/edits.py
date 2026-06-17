from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from speech_comprehension.audio import AudioSignal, replace_span, rms, windowed_fft_gain


@dataclass(frozen=True)
class EditSpec:
    name: str
    strength: float
    description: str


class SpanEdit:
    spec: EditSpec

    def apply(self, signal: AudioSignal, start: float, end: float) -> AudioSignal:
        raise NotImplementedError


@dataclass(frozen=True)
class PresenceBoost(SpanEdit):
    spec: EditSpec
    low_hz: float = 1800.0
    high_hz: float = 4200.0

    def apply(self, signal: AudioSignal, start: float, end: float) -> AudioSignal:
        start_i, end_i = signal.span_to_samples(start, end)
        segment = signal.samples[start_i:end_i]
        repaired = windowed_fft_gain(
            segment,
            signal.sample_rate,
            low_hz=self.low_hz,
            high_hz=self.high_hz,
            gain_db=4.0 * self.spec.strength,
        )
        return replace_span(signal, start, end, repaired)


@dataclass(frozen=True)
class ConsonantEdgeBoost(SpanEdit):
    spec: EditSpec

    def apply(self, signal: AudioSignal, start: float, end: float) -> AudioSignal:
        start_i, end_i = signal.span_to_samples(start, end)
        segment = signal.samples[start_i:end_i]
        if len(segment) < 3:
            return signal

        diff = np.diff(segment, prepend=segment[0])
        emphasized = segment + (0.45 * self.spec.strength * diff)
        emphasized = _match_rms(emphasized, segment, max_gain=1.25)
        return replace_span(signal, start, end, emphasized.astype(np.float32))


@dataclass(frozen=True)
class GentleRmsLift(SpanEdit):
    spec: EditSpec
    target_dbfs: float = -24.0

    def apply(self, signal: AudioSignal, start: float, end: float) -> AudioSignal:
        start_i, end_i = signal.span_to_samples(start, end)
        segment = signal.samples[start_i:end_i]
        current = rms(segment)
        target = float(10.0 ** (self.target_dbfs / 20.0))
        if current <= 1e-8 or current >= target:
            return signal

        gain = min(1.0 + 0.75 * self.spec.strength, target / current)
        repaired = np.clip(segment * gain, -1.0, 1.0)
        return replace_span(signal, start, end, repaired.astype(np.float32))


@dataclass(frozen=True)
class MudReduction(SpanEdit):
    spec: EditSpec

    def apply(self, signal: AudioSignal, start: float, end: float) -> AudioSignal:
        start_i, end_i = signal.span_to_samples(start, end)
        segment = signal.samples[start_i:end_i]
        if len(segment) < 8:
            return signal

        window = np.hanning(len(segment)).astype(np.float32)
        spectrum = np.fft.rfft(segment * window)
        freqs = np.fft.rfftfreq(len(segment), d=1.0 / signal.sample_rate)
        mask = (freqs >= 180.0) & (freqs <= 520.0)
        spectrum[mask] *= 1.0 - min(0.45, 0.14 * self.spec.strength)
        repaired = np.fft.irfft(spectrum, n=len(segment)).astype(np.float32)
        repaired = _match_rms(repaired, segment, max_gain=1.20)
        return replace_span(signal, start, end, repaired)


def default_edit_bank() -> list[SpanEdit]:
    return [
        PresenceBoost(EditSpec("presence_boost", 0.5, "Small 1.8-4.2 kHz clarity lift")),
        PresenceBoost(EditSpec("presence_boost", 1.0, "Moderate 1.8-4.2 kHz clarity lift")),
        ConsonantEdgeBoost(EditSpec("consonant_edge", 0.5, "Tiny consonant onset emphasis")),
        ConsonantEdgeBoost(EditSpec("consonant_edge", 1.0, "Moderate consonant onset emphasis")),
        GentleRmsLift(EditSpec("rms_lift", 0.5, "Gentle local loudness lift")),
        GentleRmsLift(EditSpec("rms_lift", 1.0, "Moderate local loudness lift")),
        MudReduction(EditSpec("mud_reduction", 0.5, "Small low-mid reduction")),
        MudReduction(EditSpec("mud_reduction", 1.0, "Moderate low-mid reduction")),
    ]


def _match_rms(candidate: np.ndarray, reference: np.ndarray, max_gain: float) -> np.ndarray:
    candidate_rms = rms(candidate)
    reference_rms = rms(reference)
    if candidate_rms <= 1e-8 or reference_rms <= 1e-8:
        return candidate.astype(np.float32)
    gain = min(max_gain, reference_rms / candidate_rms)
    return np.clip(candidate * gain, -1.0, 1.0).astype(np.float32)
