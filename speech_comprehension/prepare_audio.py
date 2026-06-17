from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path

from speech_comprehension.datasets import MANIFEST_FIELDS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert manifest audio to mono WAV files and write a new manifest."
    )
    parser.add_argument("input_manifest", type=Path)
    parser.add_argument("output_manifest", type=Path)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--audio-column", default="audio")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the converted manifest without running ffmpeg",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    count = convert_manifest_audio(
        input_manifest=args.input_manifest,
        output_manifest=args.output_manifest,
        audio_dir=args.audio_dir,
        audio_column=args.audio_column,
        id_column=args.id_column,
        sample_rate=args.sample_rate,
        ffmpeg=args.ffmpeg,
        max_rows=args.max_rows,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(f"Wrote {count} rows to {args.output_manifest}")
    return 0


def convert_manifest_audio(
    input_manifest: Path,
    output_manifest: Path,
    audio_dir: Path,
    audio_column: str = "audio",
    id_column: str = "id",
    sample_rate: int = 16000,
    ffmpeg: str = "ffmpeg",
    max_rows: int | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> int:
    rows, fieldnames = _read_manifest(input_manifest)
    if audio_column not in fieldnames:
        raise SystemExit(f"Manifest is missing audio column {audio_column!r}")

    audio_dir.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, str]] = []

    for index, row in enumerate(rows, start=1):
        if max_rows is not None and len(output_rows) >= max_rows:
            break

        source = _resolve_manifest_path(input_manifest, row[audio_column])
        sample_id = _sample_id(row, id_column, source, index)
        destination = audio_dir / f"{sample_id}.wav"
        if not dry_run and (overwrite or not destination.exists()):
            _run_ffmpeg(ffmpeg, source, destination, sample_rate, overwrite=overwrite)

        converted = dict(row)
        converted[audio_column] = str(destination)
        output_rows.append(converted)

    _write_rows(output_manifest, fieldnames, output_rows)
    return len(output_rows)


def _read_manifest(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"Manifest has no header row: {path}")
        return [dict(row) for row in reader], list(reader.fieldnames)


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    output_fields = _output_fieldnames(fieldnames)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)


def _output_fieldnames(fieldnames: list[str]) -> list[str]:
    fields = list(fieldnames)
    for field in MANIFEST_FIELDS:
        if field not in fields:
            fields.append(field)
    return fields


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def _sample_id(row: dict[str, str], id_column: str, audio_path: Path, index: int) -> str:
    raw = row.get(id_column, "").strip() or audio_path.stem or f"sample-{index}"
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw).strip("._-")
    return safe or f"sample-{index}"


def _run_ffmpeg(
    ffmpeg: str,
    source: Path,
    destination: Path,
    sample_rate: int,
    overwrite: bool,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(destination),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
