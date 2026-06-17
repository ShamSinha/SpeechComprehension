from __future__ import annotations

from dataclasses import dataclass

from speech_comprehension.asr import ASRResult, WordTiming


@dataclass(frozen=True)
class LowConfidenceSpan:
    start: float
    end: float
    confidence: float
    text: str
    source: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def global_confidence(result: ASRResult) -> float:
    words = result.words
    if words:
        return _mean([word.confidence for word in words])
    return _mean([segment.confidence for segment in result.segments])


def local_confidence(result: ASRResult, start: float, end: float) -> float:
    words = [
        (word.confidence, overlap)
        for word in result.words
        if (overlap := _overlap(word.start, word.end, start, end)) > 0.0
    ]
    if words:
        return _weighted_mean(words)

    segments = [
        (segment.confidence, overlap)
        for segment in result.segments
        if (overlap := _overlap(segment.start, segment.end, start, end)) > 0.0
    ]
    if segments:
        return _weighted_mean(segments)
    return global_confidence(result)


def find_low_confidence_spans(
    result: ASRResult,
    threshold: float,
    padding_seconds: float,
    min_span_seconds: float,
    max_span_seconds: float,
    audio_duration_seconds: float | None = None,
) -> list[LowConfidenceSpan]:
    words = result.words
    if words:
        spans = _word_spans(words, threshold)
    else:
        spans = [
            LowConfidenceSpan(
                start=segment.start,
                end=segment.end,
                confidence=segment.confidence,
                text=segment.text,
                source="segment",
            )
            for segment in result.segments
            if segment.confidence < threshold
        ]

    padded = [
        LowConfidenceSpan(
            start=max(0.0, span.start - padding_seconds),
            end=_clip_end(span.end + padding_seconds, audio_duration_seconds),
            confidence=span.confidence,
            text=span.text,
            source=span.source,
        )
        for span in spans
    ]
    merged = _merge_spans(padded)

    output: list[LowConfidenceSpan] = []
    for span in merged:
        if span.duration < min_span_seconds:
            center = 0.5 * (span.start + span.end)
            half = 0.5 * min_span_seconds
            span = LowConfidenceSpan(
                start=max(0.0, center - half),
                end=_clip_end(center + half, audio_duration_seconds),
                confidence=span.confidence,
                text=span.text,
                source=span.source,
            )
        if span.duration > max_span_seconds:
            output.extend(_split_span(span, max_span_seconds))
        else:
            output.append(span)
    return output


def _word_spans(words: list[WordTiming], threshold: float) -> list[LowConfidenceSpan]:
    return [
        LowConfidenceSpan(
            start=word.start,
            end=word.end,
            confidence=word.confidence,
            text=word.word,
            source="word",
        )
        for word in words
        if word.confidence < threshold
    ]


def _merge_spans(spans: list[LowConfidenceSpan]) -> list[LowConfidenceSpan]:
    if not spans:
        return []

    spans = sorted(spans, key=lambda span: (span.start, span.end))
    merged = [spans[0]]
    for span in spans[1:]:
        previous = merged[-1]
        if span.start <= previous.end:
            merged[-1] = LowConfidenceSpan(
                start=previous.start,
                end=max(previous.end, span.end),
                confidence=min(previous.confidence, span.confidence),
                text=f"{previous.text} {span.text}".strip(),
                source=previous.source if previous.source == span.source else "mixed",
            )
        else:
            merged.append(span)
    return merged


def _split_span(span: LowConfidenceSpan, max_span_seconds: float) -> list[LowConfidenceSpan]:
    pieces: list[LowConfidenceSpan] = []
    start = span.start
    while start < span.end:
        end = min(span.end, start + max_span_seconds)
        pieces.append(
            LowConfidenceSpan(
                start=start,
                end=end,
                confidence=span.confidence,
                text=span.text,
                source=span.source,
            )
        )
        start = end
    return pieces


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _weighted_mean(values: list[tuple[float, float]]) -> float:
    total_weight = sum(weight for _, weight in values)
    if total_weight <= 0.0:
        return 0.0
    return float(sum(value * weight for value, weight in values) / total_weight)


def _overlap(left_start: float, left_end: float, right_start: float, right_end: float) -> float:
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def _clip_end(end: float, audio_duration_seconds: float | None) -> float:
    if audio_duration_seconds is None:
        return end
    return min(audio_duration_seconds, end)
