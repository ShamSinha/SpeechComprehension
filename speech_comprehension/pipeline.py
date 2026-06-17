from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from speech_comprehension.asr import ASRBackend, ASRResult
from speech_comprehension.audio import AudioSignal, read_wav, write_wav
from speech_comprehension.confidence import (
    LowConfidenceSpan,
    find_low_confidence_spans,
    global_confidence,
    local_confidence,
)
from speech_comprehension.edits import SpanEdit, default_edit_bank
from speech_comprehension.objective import ChangedRegion, compute_edit_cost, objective_score
from speech_comprehension.text import transcript_similarity, word_error_rate


@dataclass(frozen=True)
class RepairConfig:
    confidence_threshold: float = 0.72
    min_confidence_gain: float = 0.02
    edit_lambda: float = 0.35
    wer_penalty: float = 0.25
    max_edit_ratio: float = 0.08
    transcript_similarity_floor: float = 0.96
    span_padding_seconds: float = 0.045
    min_span_seconds: float = 0.10
    max_span_seconds: float = 1.20
    max_spans: int = 24


@dataclass(frozen=True)
class CandidateRecord:
    span_index: int
    edit_name: str
    strength: float
    local_confidence_before: float
    local_confidence_after: float
    global_confidence_after: float
    transcript_similarity: float
    edit_cost: float
    acoustic_distance: float
    changed_ratio: float
    objective_before: float
    objective_after: float
    accepted: bool
    reason: str


@dataclass(frozen=True)
class AcceptedEdit:
    span_index: int
    start: float
    end: float
    text: str
    edit_name: str
    strength: float
    confidence_gain: float
    edit_cost: float


@dataclass(frozen=True)
class RepairReport:
    input_path: str
    output_path: str
    reference_transcript: str | None
    original_transcript: str
    final_transcript: str
    original_confidence: float
    final_confidence: float
    transcript_similarity: float
    original_wer: float | None
    final_wer: float | None
    low_confidence_spans: list[dict[str, object]]
    accepted_edits: list[AcceptedEdit]
    candidates: list[CandidateRecord]
    config: dict[str, object] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "reference_transcript": self.reference_transcript,
            "original_transcript": self.original_transcript,
            "final_transcript": self.final_transcript,
            "original_confidence": self.original_confidence,
            "final_confidence": self.final_confidence,
            "transcript_similarity": self.transcript_similarity,
            "original_wer": self.original_wer,
            "final_wer": self.final_wer,
            "low_confidence_spans": self.low_confidence_spans,
            "accepted_edits": [asdict(edit) for edit in self.accepted_edits],
            "candidates": [asdict(candidate) for candidate in self.candidates],
            "config": self.config,
        }


class RepairPipeline:
    def __init__(
        self,
        asr_backend: ASRBackend,
        config: RepairConfig | None = None,
        edit_bank: list[SpanEdit] | None = None,
    ) -> None:
        self.asr_backend = asr_backend
        self.config = config or RepairConfig()
        self.edit_bank = edit_bank or default_edit_bank()

    def repair(
        self,
        input_path: str | Path,
        output_path: str | Path,
        report_path: str | Path | None = None,
        reference_transcript: str | None = None,
    ) -> RepairReport:
        input_path = Path(input_path)
        output_path = Path(output_path)

        original_audio = read_wav(input_path)
        original_result = self.asr_backend.transcribe(input_path)
        clean_reference = _clean_optional_transcript(reference_transcript)
        preservation_transcript = clean_reference or original_result.text
        current_audio = original_audio
        current_result = original_result

        original_global_confidence = global_confidence(original_result)
        changed_regions: list[ChangedRegion] = []
        accepted_edits: list[AcceptedEdit] = []
        candidate_records: list[CandidateRecord] = []

        spans = find_low_confidence_spans(
            original_result,
            threshold=self.config.confidence_threshold,
            padding_seconds=self.config.span_padding_seconds,
            min_span_seconds=self.config.min_span_seconds,
            max_span_seconds=self.config.max_span_seconds,
            audio_duration_seconds=original_audio.duration_seconds,
        )[: self.config.max_spans]

        for span_index, span in enumerate(spans):
            current_local_confidence = local_confidence(current_result, span.start, span.end)
            best_audio: AudioSignal | None = None
            best_result: ASRResult | None = None
            best_record: CandidateRecord | None = None

            for edit in self.edit_bank:
                candidate_audio = edit.apply(current_audio, span.start, span.end)
                candidate_regions = changed_regions + [ChangedRegion(span.start, span.end)]
                candidate_cost = compute_edit_cost(original_audio, candidate_audio, candidate_regions)

                if candidate_cost.changed_ratio > self.config.max_edit_ratio:
                    record = self._candidate_record(
                        span_index,
                        edit,
                        current_local_confidence,
                        current_result,
                        current_result,
                        preservation_transcript,
                        candidate_cost,
                        accepted=False,
                        reason="max_edit_ratio",
                    )
                    candidate_records.append(record)
                    continue

                candidate_result = self._transcribe_signal(candidate_audio)
                candidate_local_confidence = local_confidence(candidate_result, span.start, span.end)
                similarity = transcript_similarity(preservation_transcript, candidate_result.text)
                current_comprehension = _comprehension_score(
                    current_local_confidence,
                    clean_reference,
                    current_result.text,
                    self.config.wer_penalty,
                )
                candidate_comprehension = _comprehension_score(
                    candidate_local_confidence,
                    clean_reference,
                    candidate_result.text,
                    self.config.wer_penalty,
                )
                objective_before = objective_score(
                    current_comprehension,
                    compute_edit_cost(original_audio, current_audio, changed_regions),
                    self.config.edit_lambda,
                )
                objective_after = objective_score(
                    candidate_comprehension,
                    candidate_cost,
                    self.config.edit_lambda,
                )

                accepted, reason = self._acceptance_reason(
                    current_local_confidence=current_local_confidence,
                    candidate_local_confidence=candidate_local_confidence,
                    transcript_similarity_score=similarity,
                    objective_before=objective_before,
                    objective_after=objective_after,
                )

                record = CandidateRecord(
                    span_index=span_index,
                    edit_name=edit.spec.name,
                    strength=edit.spec.strength,
                    local_confidence_before=current_local_confidence,
                    local_confidence_after=candidate_local_confidence,
                    global_confidence_after=global_confidence(candidate_result),
                    transcript_similarity=similarity,
                    edit_cost=candidate_cost.total,
                    acoustic_distance=candidate_cost.acoustic_distance,
                    changed_ratio=candidate_cost.changed_ratio,
                    objective_before=objective_before,
                    objective_after=objective_after,
                    accepted=accepted,
                    reason=reason,
                )
                candidate_records.append(record)

                if accepted and (
                    best_record is None or record.objective_after > best_record.objective_after
                ):
                    best_audio = candidate_audio
                    best_result = candidate_result
                    best_record = record

            if best_audio is not None and best_result is not None and best_record is not None:
                current_audio = best_audio
                current_result = best_result
                changed_regions.append(ChangedRegion(span.start, span.end))
                accepted_edits.append(
                    AcceptedEdit(
                        span_index=span_index,
                        start=span.start,
                        end=span.end,
                        text=span.text,
                        edit_name=best_record.edit_name,
                        strength=best_record.strength,
                        confidence_gain=(
                            best_record.local_confidence_after
                            - best_record.local_confidence_before
                        ),
                        edit_cost=best_record.edit_cost,
                    )
                )

        write_wav(output_path, current_audio)
        final_confidence = global_confidence(current_result)
        report = RepairReport(
            input_path=str(input_path),
            output_path=str(output_path),
            reference_transcript=clean_reference,
            original_transcript=original_result.text,
            final_transcript=current_result.text,
            original_confidence=original_global_confidence,
            final_confidence=final_confidence,
            transcript_similarity=transcript_similarity(
                preservation_transcript,
                current_result.text,
            ),
            original_wer=_maybe_wer(clean_reference, original_result.text),
            final_wer=_maybe_wer(clean_reference, current_result.text),
            low_confidence_spans=[_span_to_dict(span) for span in spans],
            accepted_edits=accepted_edits,
            candidates=candidate_records,
            config=asdict(self.config),
        )

        if report_path is not None:
            report_path = Path(report_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report.to_json_dict(), indent=2), encoding="utf-8")

        return report

    def _transcribe_signal(self, signal: AudioSignal) -> ASRResult:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as handle:
            write_wav(handle.name, signal)
            return self.asr_backend.transcribe(handle.name)

    def _acceptance_reason(
        self,
        current_local_confidence: float,
        candidate_local_confidence: float,
        transcript_similarity_score: float,
        objective_before: float,
        objective_after: float,
    ) -> tuple[bool, str]:
        gain = candidate_local_confidence - current_local_confidence
        if transcript_similarity_score < self.config.transcript_similarity_floor:
            return False, "transcript_changed"
        if gain < self.config.min_confidence_gain:
            return False, "insufficient_confidence_gain"
        if objective_after <= objective_before:
            return False, "objective_not_improved"
        return True, "accepted"

    def _candidate_record(
        self,
        span_index: int,
        edit: SpanEdit,
        current_local_confidence: float,
        current_result: ASRResult,
        candidate_result: ASRResult,
        preservation_transcript: str,
        candidate_cost,
        accepted: bool,
        reason: str,
    ) -> CandidateRecord:
        confidence_after = local_confidence(candidate_result, 0.0, float("inf"))
        objective_before = objective_score(
            current_local_confidence,
            candidate_cost,
            self.config.edit_lambda,
        )
        objective_after = objective_score(confidence_after, candidate_cost, self.config.edit_lambda)
        return CandidateRecord(
            span_index=span_index,
            edit_name=edit.spec.name,
            strength=edit.spec.strength,
            local_confidence_before=current_local_confidence,
            local_confidence_after=confidence_after,
            global_confidence_after=global_confidence(candidate_result),
            transcript_similarity=transcript_similarity(
                preservation_transcript,
                candidate_result.text,
            ),
            edit_cost=candidate_cost.total,
            acoustic_distance=candidate_cost.acoustic_distance,
            changed_ratio=candidate_cost.changed_ratio,
            objective_before=objective_before,
            objective_after=objective_after,
            accepted=accepted,
            reason=reason,
        )


def _span_to_dict(span: LowConfidenceSpan) -> dict[str, object]:
    return {
        "start": span.start,
        "end": span.end,
        "duration": span.duration,
        "confidence": span.confidence,
        "text": span.text,
        "source": span.source,
    }


def _clean_optional_transcript(transcript: str | None) -> str | None:
    if transcript is None:
        return None
    transcript = transcript.strip()
    return transcript or None


def _maybe_wer(reference_transcript: str | None, hypothesis: str) -> float | None:
    reference_transcript = _clean_optional_transcript(reference_transcript)
    if reference_transcript is None:
        return None
    return word_error_rate(reference_transcript, hypothesis)


def _comprehension_score(
    confidence: float,
    reference_transcript: str | None,
    hypothesis: str,
    wer_penalty: float,
) -> float:
    reference_transcript = _clean_optional_transcript(reference_transcript)
    if reference_transcript is None:
        return confidence
    return confidence - wer_penalty * word_error_rate(reference_transcript, hypothesis)
