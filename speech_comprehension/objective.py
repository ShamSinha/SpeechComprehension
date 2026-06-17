from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from speech_comprehension.audio import AudioSignal, match_length, rms


@dataclass(frozen=True)
class ChangedRegion:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class EditCost:
    total: float
    acoustic_distance: float
    changed_ratio: float
    peak_delta: float


def compute_edit_cost(
    original: AudioSignal,
    candidate: AudioSignal,
    changed_regions: list[ChangedRegion],
    changed_ratio_weight: float = 0.20,
    peak_delta_weight: float = 0.05,
) -> EditCost:
    if original.sample_rate != candidate.sample_rate:
        raise ValueError("Edit cost expects matching sample rates")

    candidate_samples = match_length(candidate.samples, len(original.samples))
    diff = candidate_samples - original.samples
    reference_rms = max(rms(original.samples), 1e-6)
    acoustic_distance = float(np.sqrt(np.mean(np.square(diff))) / reference_rms)
    changed_ratio = _changed_ratio(changed_regions, original.duration_seconds)
    peak_delta = float(abs(np.max(np.abs(candidate_samples)) - np.max(np.abs(original.samples))))
    total = (
        acoustic_distance
        + changed_ratio_weight * changed_ratio
        + peak_delta_weight * peak_delta
    )
    return EditCost(
        total=float(total),
        acoustic_distance=float(acoustic_distance),
        changed_ratio=float(changed_ratio),
        peak_delta=float(peak_delta),
    )


def objective_score(confidence: float, edit_cost: EditCost, edit_lambda: float) -> float:
    return float(confidence - edit_lambda * edit_cost.total)


def _changed_ratio(changed_regions: list[ChangedRegion], duration_seconds: float) -> float:
    if duration_seconds <= 0.0 or not changed_regions:
        return 0.0

    merged = []
    for region in sorted(changed_regions, key=lambda item: (item.start, item.end)):
        if not merged or region.start > merged[-1].end:
            merged.append(region)
        else:
            previous = merged[-1]
            merged[-1] = ChangedRegion(previous.start, max(previous.end, region.end))
    changed_seconds = sum(region.duration for region in merged)
    return min(1.0, changed_seconds / duration_seconds)
