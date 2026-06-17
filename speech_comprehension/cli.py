from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from speech_comprehension.asr import build_asr_backend
from speech_comprehension.pipeline import RepairConfig, RepairPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal ASR-confidence intelligibility repair for speech audio."
    )
    parser.add_argument("input", type=Path, help="Input mono or stereo WAV file")
    parser.add_argument("output", type=Path, help="Output repaired WAV file")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON report path")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config path")
    parser.add_argument(
        "--transcript",
        default=None,
        help="Optional reference transcript for WER and transcript-preservation checks",
    )
    parser.add_argument(
        "--transcript-file",
        type=Path,
        default=None,
        help="Optional UTF-8 file containing the reference transcript",
    )
    parser.add_argument(
        "--asr",
        choices=["whisper", "faster-whisper"],
        default="whisper",
        help="ASR backend used as the intelligibility proxy",
    )
    parser.add_argument("--model", default="small", help="ASR model name")
    parser.add_argument("--language", default=None, help="Optional ASR language hint")
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--min-confidence-gain", type=float, default=None)
    parser.add_argument("--edit-lambda", type=float, default=None)
    parser.add_argument("--wer-penalty", type=float, default=None)
    parser.add_argument("--max-edit-ratio", type=float, default=None)
    parser.add_argument("--transcript-similarity-floor", type=float, default=None)
    parser.add_argument("--max-spans", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reference_transcript = _load_reference_transcript(args.transcript, args.transcript_file)
    config_values = asdict(RepairConfig())
    if args.config is not None:
        config_values.update(json.loads(args.config.read_text(encoding="utf-8")))
    overrides = {
        "confidence_threshold": args.confidence_threshold,
        "min_confidence_gain": args.min_confidence_gain,
        "edit_lambda": args.edit_lambda,
        "wer_penalty": args.wer_penalty,
        "max_edit_ratio": args.max_edit_ratio,
        "transcript_similarity_floor": args.transcript_similarity_floor,
        "max_spans": args.max_spans,
    }
    config_values.update({key: value for key, value in overrides.items() if value is not None})
    config = RepairConfig(**config_values)
    backend = build_asr_backend(args.asr, model_name=args.model, language=args.language)
    pipeline = RepairPipeline(backend, config=config)
    report = pipeline.repair(
        args.input,
        args.output,
        report_path=args.report,
        reference_transcript=reference_transcript,
    )

    print(f"Original confidence: {report.original_confidence:.3f}")
    print(f"Final confidence:    {report.final_confidence:.3f}")
    if report.original_wer is not None and report.final_wer is not None:
        print(f"Original WER:        {report.original_wer:.3f}")
        print(f"Final WER:           {report.final_wer:.3f}")
    print(f"Accepted edits:      {len(report.accepted_edits)}")
    if args.report:
        print(f"Report:              {args.report}")
    return 0


def _load_reference_transcript(transcript: str | None, transcript_file: Path | None) -> str | None:
    if transcript is not None and transcript_file is not None:
        raise SystemExit("Use either --transcript or --transcript-file, not both.")
    if transcript_file is not None:
        return transcript_file.read_text(encoding="utf-8").strip()
    if transcript is not None:
        return transcript.strip()
    return None


if __name__ == "__main__":
    raise SystemExit(main())
