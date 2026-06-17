from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol


@dataclass(frozen=True)
class WordTiming:
    word: str
    start: float
    end: float
    confidence: float


@dataclass(frozen=True)
class ASRSegment:
    text: str
    start: float
    end: float
    confidence: float
    words: list[WordTiming] = field(default_factory=list)


@dataclass(frozen=True)
class ASRResult:
    text: str
    segments: list[ASRSegment]

    @property
    def words(self) -> list[WordTiming]:
        return [word for segment in self.segments for word in segment.words]


class ASRBackend(Protocol):
    def transcribe(self, audio_path: str | Path) -> ASRResult:
        ...


class WhisperBackend:
    """Adapter for the openai-whisper package."""

    def __init__(self, model_name: str = "small", language: str | None = None) -> None:
        try:
            import whisper
        except ImportError as exc:
            raise RuntimeError(
                "openai-whisper is not installed. Install with `pip install -e .[whisper]`."
            ) from exc

        self.model = whisper.load_model(model_name)
        self.language = language
        self.fp16 = _whisper_supports_fp16(self.model)

    def transcribe(self, audio_path: str | Path) -> ASRResult:
        options: dict[str, object] = {
            "word_timestamps": True,
            "fp16": self.fp16,
        }
        if self.language:
            options["language"] = self.language
        raw = self.model.transcribe(str(audio_path), **options)

        segments: list[ASRSegment] = []
        for segment in raw.get("segments", []):
            segment_confidence = _avg_logprob_to_confidence(segment.get("avg_logprob"))
            words = [
                WordTiming(
                    word=str(word.get("word", "")).strip(),
                    start=float(word.get("start", segment.get("start", 0.0))),
                    end=float(word.get("end", segment.get("end", 0.0))),
                    confidence=float(word.get("probability", segment_confidence)),
                )
                for word in segment.get("words", [])
                if str(word.get("word", "")).strip()
            ]
            segments.append(
                ASRSegment(
                    text=str(segment.get("text", "")).strip(),
                    start=float(segment.get("start", 0.0)),
                    end=float(segment.get("end", 0.0)),
                    confidence=segment_confidence,
                    words=words,
                )
            )

        return ASRResult(text=str(raw.get("text", "")).strip(), segments=segments)


class FasterWhisperBackend:
    """Adapter for faster-whisper."""

    def __init__(
        self,
        model_name: str = "small",
        language: str | None = None,
        device: str = "auto",
        compute_type: str = "default",
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Install with `pip install -e .[faster-whisper]`."
            ) from exc

        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self.language = language

    def transcribe(self, audio_path: str | Path) -> ASRResult:
        raw_segments, _info = self.model.transcribe(
            str(audio_path),
            language=self.language,
            word_timestamps=True,
        )

        segments: list[ASRSegment] = []
        for segment in raw_segments:
            segment_confidence = _avg_logprob_to_confidence(getattr(segment, "avg_logprob", None))
            words = [
                WordTiming(
                    word=str(word.word).strip(),
                    start=float(word.start),
                    end=float(word.end),
                    confidence=float(getattr(word, "probability", segment_confidence)),
                )
                for word in (segment.words or [])
                if str(word.word).strip()
            ]
            segments.append(
                ASRSegment(
                    text=str(segment.text).strip(),
                    start=float(segment.start),
                    end=float(segment.end),
                    confidence=segment_confidence,
                    words=words,
                )
            )

        return ASRResult(
            text=" ".join(segment.text for segment in segments).strip(),
            segments=segments,
        )


class MockASRBackend:
    """Deterministic backend for tests and objective experiments."""

    def __init__(self, scorer: Callable[[str | Path], ASRResult]) -> None:
        self.scorer = scorer

    def transcribe(self, audio_path: str | Path) -> ASRResult:
        return self.scorer(audio_path)


def build_asr_backend(
    backend: str,
    model_name: str = "small",
    language: str | None = None,
) -> ASRBackend:
    if backend == "whisper":
        return WhisperBackend(model_name=model_name, language=language)
    if backend == "faster-whisper":
        return FasterWhisperBackend(model_name=model_name, language=language)
    raise ValueError(f"Unsupported ASR backend: {backend}")


def _avg_logprob_to_confidence(avg_logprob: object) -> float:
    if avg_logprob is None:
        return 0.5
    try:
        return max(0.0, min(1.0, math.exp(float(avg_logprob))))
    except (TypeError, ValueError, OverflowError):
        return 0.5


def _whisper_supports_fp16(model: object) -> bool:
    device = getattr(model, "device", None)
    return getattr(device, "type", None) == "cuda"
