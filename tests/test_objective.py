from __future__ import annotations

import numpy as np

from speech_comprehension.audio import AudioSignal
from speech_comprehension.objective import ChangedRegion, compute_edit_cost


def test_edit_cost_increases_with_larger_audio_change() -> None:
    original = AudioSignal(np.zeros(16000, dtype=np.float32), 16000)
    tiny = AudioSignal(np.full(16000, 0.001, dtype=np.float32), 16000)
    larger = AudioSignal(np.full(16000, 0.01, dtype=np.float32), 16000)

    region = [ChangedRegion(0.25, 0.35)]
    tiny_cost = compute_edit_cost(original, tiny, region)
    larger_cost = compute_edit_cost(original, larger, region)

    assert larger_cost.total > tiny_cost.total
    assert round(tiny_cost.changed_ratio, 3) == 0.1


def test_changed_regions_are_merged_for_ratio() -> None:
    original = AudioSignal(np.zeros(1000, dtype=np.float32), 1000)
    candidate = AudioSignal(np.zeros(1000, dtype=np.float32), 1000)
    cost = compute_edit_cost(
        original,
        candidate,
        [ChangedRegion(0.1, 0.3), ChangedRegion(0.2, 0.4)],
    )

    assert round(cost.changed_ratio, 3) == 0.3
