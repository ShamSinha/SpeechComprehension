from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict
from pathlib import Path

from speech_comprehension.asr import build_asr_backend
from speech_comprehension.pipeline import RepairConfig, RepairPipeline, RepairReport


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a minimal intelligibility-repair experiment over a CSV manifest."
    )
    parser.add_argument("manifest", type=Path, help="CSV with audio and transcript columns")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiment"))
    parser.add_argument("--summary", type=Path, default=None, help="Optional summary CSV path")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config path")
    parser.add_argument("--audio-column", default="audio")
    parser.add_argument("--transcript-column", default="transcript")
    parser.add_argument(
        "--transcript-file-column",
        default="transcript_file",
        help="Fallback column containing a UTF-8 transcript file path",
    )
    parser.add_argument("--id-column", default="id")
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
    manifest_path = args.manifest
    config = _load_config(args)
    backend = build_asr_backend(args.asr, model_name=args.model, language=args.language)
    pipeline = RepairPipeline(backend, config=config)

    output_dir = args.output_dir
    audio_dir = output_dir / "audio"
    reports_dir = output_dir / "reports"
    audio_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_manifest(manifest_path)
    results: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        audio_path = _resolve_manifest_path(manifest_path, _required(row, args.audio_column))
        reference = _reference_transcript(row, args, manifest_path)
        sample_id = _sample_id(row, args.id_column, audio_path, index)
        output_path = audio_dir / f"{sample_id}.repaired.wav"
        report_path = reports_dir / f"{sample_id}.json"

        report = pipeline.repair(
            audio_path,
            output_path,
            report_path=report_path,
            reference_transcript=reference,
        )
        results.append(_summary_row(sample_id, audio_path, output_path, report_path, report))
        print(
            f"{sample_id}: WER { _format_optional(report.original_wer) }"
            f" -> { _format_optional(report.final_wer) }, edits={len(report.accepted_edits)}"
        )

    summary_path = args.summary or (output_dir / "summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    _write_summary_csv(summary_path, results)
    aggregate_path = summary_path.with_suffix(".json")
    aggregate_path.write_text(
        json.dumps(_aggregate(results, config), indent=2),
        encoding="utf-8",
    )

    print(f"Summary CSV:         {summary_path}")
    print(f"Aggregate JSON:      {aggregate_path}")
    return 0


def _load_config(args: argparse.Namespace) -> RepairConfig:
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
    return RepairConfig(**config_values)


def _load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"Manifest has no header row: {path}")
        return [dict(row) for row in reader]


def _required(row: dict[str, str], column: str) -> str:
    value = row.get(column, "").strip()
    if not value:
        raise SystemExit(f"Missing required manifest column value: {column}")
    return value


def _reference_transcript(row: dict[str, str], args: argparse.Namespace, manifest_path: Path) -> str:
    transcript = row.get(args.transcript_column, "").strip()
    if transcript:
        return transcript

    transcript_file = row.get(args.transcript_file_column, "").strip()
    if transcript_file:
        path = _resolve_manifest_path(manifest_path, transcript_file)
        return path.read_text(encoding="utf-8").strip()

    raise SystemExit(
        f"Missing transcript in columns {args.transcript_column!r}"
        f" or {args.transcript_file_column!r}"
    )


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def _sample_id(row: dict[str, str], id_column: str, audio_path: Path, index: int) -> str:
    raw = row.get(id_column, "").strip() or audio_path.stem or f"sample-{index}"
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw).strip("._-")
    return safe or f"sample-{index}"


def _summary_row(
    sample_id: str,
    audio_path: Path,
    output_path: Path,
    report_path: Path,
    report: RepairReport,
) -> dict[str, object]:
    original_wer = report.original_wer
    final_wer = report.final_wer
    return {
        "id": sample_id,
        "input_path": str(audio_path),
        "output_path": str(output_path),
        "report_path": str(report_path),
        "original_confidence": report.original_confidence,
        "final_confidence": report.final_confidence,
        "confidence_delta": report.final_confidence - report.original_confidence,
        "original_wer": original_wer,
        "final_wer": final_wer,
        "wer_delta": None if original_wer is None or final_wer is None else final_wer - original_wer,
        "transcript_similarity": report.transcript_similarity,
        "accepted_edits": len(report.accepted_edits),
        "low_confidence_spans": len(report.low_confidence_spans),
    }


def _write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "id",
        "input_path",
        "output_path",
        "report_path",
        "original_confidence",
        "final_confidence",
        "confidence_delta",
        "original_wer",
        "final_wer",
        "wer_delta",
        "transcript_similarity",
        "accepted_edits",
        "low_confidence_spans",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict[str, object]], config: RepairConfig) -> dict[str, object]:
    return {
        "num_samples": len(rows),
        "mean_original_confidence": _mean_numeric(rows, "original_confidence"),
        "mean_final_confidence": _mean_numeric(rows, "final_confidence"),
        "mean_confidence_delta": _mean_numeric(rows, "confidence_delta"),
        "mean_original_wer": _mean_numeric(rows, "original_wer"),
        "mean_final_wer": _mean_numeric(rows, "final_wer"),
        "mean_wer_delta": _mean_numeric(rows, "wer_delta"),
        "total_accepted_edits": sum(int(row["accepted_edits"]) for row in rows),
        "config": asdict(config),
    }


def _mean_numeric(rows: list[dict[str, object]], key: str) -> float | None:
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    if not values:
        return None
    return float(sum(float(value) for value in values) / len(values))


def _format_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
