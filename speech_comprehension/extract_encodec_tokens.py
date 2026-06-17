from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EncodecTokenRecord:
    id: str
    audio: str
    token_path: str
    transcript: str
    accent: str
    speaker: str
    source: str
    split: str
    duration_seconds: float
    codec_model: str
    sample_rate: int
    bandwidth: float
    num_codebooks: int
    num_frames: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract pretrained EnCodec acoustic tokens for a speech manifest."
    )
    parser.add_argument("manifest", type=Path, help="CSV with audio/transcript rows")
    parser.add_argument("--output-dir", type=Path, default=Path("data/encodec_tokens"))
    parser.add_argument("--audio-column", default="audio")
    parser.add_argument("--transcript-column", default="transcript")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--accent-column", default="accent")
    parser.add_argument("--speaker-column", default="speaker")
    parser.add_argument("--source-column", default="source")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--model",
        choices=["24khz", "48khz"],
        default="24khz",
        help="Pretrained EnCodec model to use. 24khz is mono and recommended for speech.",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=6.0,
        help="Target EnCodec bandwidth. For 24khz: 1.5, 3, 6, 12, or 24.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract token files that already exist.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = extract_encodec_tokens(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        audio_column=args.audio_column,
        transcript_column=args.transcript_column,
        id_column=args.id_column,
        accent_column=args.accent_column,
        speaker_column=args.speaker_column,
        source_column=args.source_column,
        split_column=args.split_column,
        max_rows=args.max_rows,
        model_name=args.model,
        bandwidth=args.bandwidth,
        device=args.device,
        overwrite=args.overwrite,
    )
    print(f"Wrote {len(records)} EnCodec token records to {args.output_dir / 'index.csv'}")
    return 0


def extract_encodec_tokens(
    manifest_path: Path,
    output_dir: Path,
    audio_column: str = "audio",
    transcript_column: str = "transcript",
    id_column: str = "id",
    accent_column: str = "accent",
    speaker_column: str = "speaker",
    source_column: str = "source",
    split_column: str = "split",
    max_rows: int | None = None,
    model_name: str = "24khz",
    bandwidth: float = 6.0,
    device: str = "auto",
    overwrite: bool = False,
) -> list[EncodecTokenRecord]:
    try:
        import torch
        from encodec import EncodecModel
    except ImportError as exc:
        raise SystemExit(
            "EnCodec token extraction requires encodec. Install it with "
            '`pip install -e ".[codec]"`.'
        ) from exc

    manifest_path = manifest_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    token_dir = output_dir / "tokens"
    token_dir.mkdir(parents=True, exist_ok=True)

    device_obj = _resolve_device(device, torch)
    model = _load_encodec_model(EncodecModel, model_name)
    model.set_target_bandwidth(bandwidth)
    model.to(device_obj)
    model.eval()

    rows = _load_manifest(manifest_path)
    if max_rows is not None:
        rows = rows[:max_rows]

    records: list[EncodecTokenRecord] = []
    for index, row in enumerate(rows, start=1):
        audio_path = _resolve_manifest_path(manifest_path, _required(row, audio_column))
        sample_id = _safe_sample_id(row.get(id_column, "").strip() or audio_path.stem or f"sample-{index}")
        token_path = token_dir / f"{sample_id}.pt"

        if token_path.exists() and not overwrite:
            record = _record_from_existing_token(
                row=row,
                sample_id=sample_id,
                audio_path=audio_path,
                token_path=token_path,
                output_dir=output_dir,
                transcript_column=transcript_column,
                accent_column=accent_column,
                speaker_column=speaker_column,
                source_column=source_column,
                split_column=split_column,
            )
            records.append(record)
            print(f"[{index}/{len(rows)}] skipped existing {sample_id}")
            continue

        wav = _load_audio_for_encodec(audio_path, model.sample_rate, model.channels)
        duration_seconds = float(wav.shape[-1]) / float(model.sample_rate)
        wav = wav.to(device_obj)

        with torch.no_grad():
            encoded_frames = model.encode(wav)

        codes = torch.cat([frame_codes for frame_codes, _ in encoded_frames], dim=-1)
        token_payload = {
            "id": sample_id,
            "audio": str(audio_path),
            "transcript": row.get(transcript_column, "").strip(),
            "accent": row.get(accent_column, "").strip(),
            "speaker": row.get(speaker_column, "").strip(),
            "source": row.get(source_column, "").strip(),
            "split": row.get(split_column, "").strip(),
            "codec_model": f"encodec_{model_name}",
            "sample_rate": model.sample_rate,
            "channels": model.channels,
            "bandwidth": float(bandwidth),
            "duration_seconds": duration_seconds,
            "codes": codes.squeeze(0).cpu(),
            "encoded_frames": [
                (
                    frame_codes.squeeze(0).cpu(),
                    None if scale is None else scale.squeeze(0).cpu(),
                )
                for frame_codes, scale in encoded_frames
            ],
        }
        torch.save(token_payload, token_path)

        record = EncodecTokenRecord(
            id=sample_id,
            audio=str(audio_path),
            token_path=_relative_to(output_dir, token_path),
            transcript=row.get(transcript_column, "").strip(),
            accent=row.get(accent_column, "").strip(),
            speaker=row.get(speaker_column, "").strip(),
            source=row.get(source_column, "").strip(),
            split=row.get(split_column, "").strip(),
            duration_seconds=duration_seconds,
            codec_model=f"encodec_{model_name}",
            sample_rate=int(model.sample_rate),
            bandwidth=float(bandwidth),
            num_codebooks=int(codes.shape[1]),
            num_frames=int(codes.shape[-1]),
        )
        records.append(record)
        print(
            f"[{index}/{len(rows)}] {sample_id}: "
            f"{record.num_codebooks} codebooks x {record.num_frames} frames"
        )

    _write_index(output_dir / "index.csv", records)
    _write_metadata(
        output_dir / "metadata.json",
        {
            "manifest": str(manifest_path),
            "num_records": len(records),
            "codec_model": f"encodec_{model_name}",
            "sample_rate": int(model.sample_rate),
            "channels": int(model.channels),
            "bandwidth": float(bandwidth),
            "device": str(device_obj),
        },
    )
    return records


def _load_encodec_model(encodec_model_cls: Any, model_name: str) -> Any:
    if model_name == "24khz":
        return encodec_model_cls.encodec_model_24khz()
    if model_name == "48khz":
        return encodec_model_cls.encodec_model_48khz()
    raise ValueError(f"Unsupported EnCodec model: {model_name}")


def _resolve_device(device: str, torch_module: Any) -> Any:
    if device == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    return torch_module.device(device)


def _load_audio_for_encodec(path: Path, sample_rate: int, channels: int) -> Any:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-",
    ]
    try:
        completed = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg failed to decode {path}: {stderr}") from exc

    samples = np.frombuffer(completed.stdout, dtype=np.float32)
    if samples.size == 0:
        raise RuntimeError(f"ffmpeg decoded no audio samples from {path}")
    samples = samples.reshape(-1, channels).T.copy()

    import torch

    return torch.from_numpy(samples).unsqueeze(0)


def _record_from_existing_token(
    row: dict[str, str],
    sample_id: str,
    audio_path: Path,
    token_path: Path,
    output_dir: Path,
    transcript_column: str,
    accent_column: str,
    speaker_column: str,
    source_column: str,
    split_column: str,
) -> EncodecTokenRecord:
    import torch

    payload = torch.load(token_path, map_location="cpu")
    codes = payload["codes"]
    return EncodecTokenRecord(
        id=sample_id,
        audio=str(audio_path),
        token_path=_relative_to(output_dir, token_path),
        transcript=row.get(transcript_column, "").strip(),
        accent=row.get(accent_column, "").strip(),
        speaker=row.get(speaker_column, "").strip(),
        source=row.get(source_column, "").strip(),
        split=row.get(split_column, "").strip(),
        duration_seconds=float(payload.get("duration_seconds", 0.0)),
        codec_model=str(payload.get("codec_model", "encodec")),
        sample_rate=int(payload.get("sample_rate", 0)),
        bandwidth=float(payload.get("bandwidth", 0.0)),
        num_codebooks=int(codes.shape[0]),
        num_frames=int(codes.shape[-1]),
    )


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


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    manifest_relative = manifest_path.parent / path
    if manifest_relative.exists():
        return manifest_relative
    cwd_relative = Path.cwd() / path
    if cwd_relative.exists():
        return cwd_relative
    return manifest_relative


def _safe_sample_id(raw: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw).strip("._-")
    return safe or "sample"


def _write_index(path: Path, records: list[EncodecTokenRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(records[0]).keys()) if records else list(EncodecTokenRecord.__annotations__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def _write_metadata(path: Path, metadata: dict[str, object]) -> None:
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def _relative_to(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
