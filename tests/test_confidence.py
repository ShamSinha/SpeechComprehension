from __future__ import annotations

from speech_comprehension.asr import ASRResult, ASRSegment, WordTiming
from speech_comprehension.confidence import local_confidence


def test_local_confidence_weights_words_by_overlap() -> None:
    result = ASRResult(
        text="world is",
        segments=[
            ASRSegment(
                text="world is",
                start=0.0,
                end=1.0,
                confidence=0.75,
                words=[
                    WordTiming("world", 0.00, 0.90, 0.50),
                    WordTiming("is", 0.89, 1.00, 0.95),
                ],
            )
        ],
    )

    assert round(local_confidence(result, 0.0, 0.90), 3) == 0.505
