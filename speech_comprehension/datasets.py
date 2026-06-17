from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


MANIFEST_FIELDS = ["id", "audio", "transcript", "accent", "speaker", "source", "split"]
TRANSCRIPT_EXTENSIONS = {".txt", ".lab", ".transcript"}


@dataclass(frozen=True)
class ManifestRow:
    id: str
    audio: str
    transcript: str
    accent: str = ""
    speaker: str = ""
    source: str = ""
    split: str = ""

    def to_csv_row(self) -> dict[str, str]:
        return {
            "id": self.id,
            "audio": self.audio,
            "transcript": self.transcript,
            "accent": self.accent,
            "speaker": self.speaker,
            "source": self.source,
            "split": self.split,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a speech-repair experiment manifest from local dataset files."
    )
    parser.add_argument(
        "--dataset",
        choices=["common-voice", "l2-arctic"],
        required=True,
        help="Dataset layout to scan",
    )
    parser.add_argument("--root", type=Path, required=True, help="Unzipped dataset root")
    parser.add_argument("--output", type=Path, required=True, help="Output manifest CSV")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--accent-contains",
        default=None,
        help="Case-insensitive filter applied to the manifest accent field",
    )

    common_voice = parser.add_argument_group("Common Voice")
    common_voice.add_argument(
        "--split",
        default="validated",
        help=(
            "Common Voice metadata split, such as validated, cv-valid-train, "
            "all-valid, or all"
        ),
    )
    common_voice.add_argument(
        "--audio-root",
        type=Path,
        default=None,
        help="Override directory containing audio files",
    )
    common_voice.add_argument(
        "--audio-extension",
        default=None,
        help="Replace Common Voice clip suffix, useful after converting mp3 clips to wav",
    )

    l2_arctic = parser.add_argument_group("L2-ARCTIC")
    l2_arctic.add_argument(
        "--accent-label",
        default=None,
        help="Fallback accent label for L2-ARCTIC rows when no metadata is found",
    )
    l2_arctic.add_argument(
        "--speaker-accent-map",
        type=Path,
        default=None,
        help="Optional CSV/TSV with speaker and accent/L1 columns",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dataset == "common-voice":
        rows = common_voice_rows(
            root=args.root,
            split=args.split,
            audio_root=args.audio_root,
            audio_extension=args.audio_extension,
            accent_contains=args.accent_contains,
            max_rows=args.max_rows,
        )
    else:
        rows = l2_arctic_rows(
            root=args.root,
            accent_label=args.accent_label,
            speaker_accent_map=args.speaker_accent_map,
            accent_contains=args.accent_contains,
            max_rows=args.max_rows,
        )

    write_manifest(args.output, rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


def common_voice_rows(
    root: Path,
    split: str = "validated",
    audio_root: Path | None = None,
    audio_extension: str | None = None,
    accent_contains: str | None = None,
    max_rows: int | None = None,
) -> list[ManifestRow]:
    root = root.expanduser().resolve()
    rows: list[ManifestRow] = []

    for metadata_path in _find_common_voice_metadata_paths(root, split):
        split_name = metadata_path.stem
        resolved_audio_root = (
            audio_root.expanduser().resolve()
            if audio_root
            else _find_common_voice_audio_root(metadata_path, root, split_name)
        )
        with metadata_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=_csv_delimiter(metadata_path))
            for raw in reader:
                clip_path = _first_present(raw, ["path", "filename", "file", "audio"])
                transcript = _first_present(raw, ["sentence", "text", "transcript"])
                if not clip_path or not transcript:
                    continue

                audio_path = _common_voice_audio_path(
                    root,
                    resolved_audio_root,
                    clip_path,
                    audio_extension=audio_extension,
                )
                accent = _first_present(raw, ["accent", "locale", "variant"])
                if not _matches_filter(accent, accent_contains):
                    continue

                rows.append(
                    ManifestRow(
                        id=_safe_id(f"{split_name}_{Path(clip_path).stem}"),
                        audio=str(audio_path),
                        transcript=transcript,
                        accent=accent,
                        speaker=raw.get("client_id", "").strip(),
                        source="common-voice",
                        split=split_name,
                    )
                )
                if max_rows is not None and len(rows) >= max_rows:
                    return rows

    return rows


def l2_arctic_rows(
    root: Path,
    accent_label: str | None = None,
    speaker_accent_map: Path | None = None,
    accent_contains: str | None = None,
    max_rows: int | None = None,
) -> list[ManifestRow]:
    root = root.expanduser().resolve()
    transcript_index = _build_l2_transcript_index(root)
    accent_by_speaker = _load_accent_map(speaker_accent_map) if speaker_accent_map else _discover_accent_map(root)
    rows: list[ManifestRow] = []

    for wav_path in sorted(root.rglob("*.wav")):
        if _is_hidden_path(wav_path):
            continue
        transcript = _lookup_l2_transcript(wav_path, transcript_index)
        if not transcript:
            continue

        speaker = _infer_l2_speaker(root, wav_path)
        accent = accent_by_speaker.get(speaker.lower(), accent_label or "l2-english")
        if not _matches_filter(accent, accent_contains):
            continue

        rows.append(
            ManifestRow(
                id=_safe_id(f"{speaker}_{wav_path.stem}"),
                audio=str(wav_path),
                transcript=transcript,
                accent=accent,
                speaker=speaker,
                source="l2-arctic",
                split="",
            )
        )
        if max_rows is not None and len(rows) >= max_rows:
            break

    return rows


def write_manifest(path: Path, rows: list[ManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(row.to_csv_row() for row in rows)


def _find_common_voice_metadata_paths(root: Path, split: str) -> list[Path]:
    normalized = split.lower()
    if normalized in {"all-valid", "valid-all"}:
        matches = _metadata_glob(root, "cv-valid-*") or [
            path
            for name in ["validated", "train", "dev", "test"]
            for path in _metadata_glob(root, name)
        ]
        if matches:
            return matches
    elif normalized == "all":
        matches = sorted(
            path
            for path in root.glob("*")
            if path.suffix.lower() in {".csv", ".tsv"}
        )
        if matches:
            return matches

    return [_find_common_voice_metadata(root, split)]


def _metadata_glob(root: Path, stem_pattern: str) -> list[Path]:
    return sorted(
        path
        for suffix in [".csv", ".tsv"]
        for path in root.glob(f"{stem_pattern}{suffix}")
    )


def _find_common_voice_metadata(root: Path, split: str) -> Path:
    candidates = [
        root / f"{split}.tsv",
        root / f"{split}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(root.rglob(f"{split}.tsv")) + sorted(root.rglob(f"{split}.csv"))
    if matches:
        return matches[0]
    raise SystemExit(f"Could not find Common Voice metadata for split {split!r} under {root}")


def _find_common_voice_audio_root(metadata_path: Path, root: Path, split: str) -> Path:
    candidates = [
        metadata_path.parent / "clips",
        root / "clips",
        root / split / split,
        root / split,
    ]
    candidates.extend(sorted(root.rglob("clips")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return tsv_path.parent / "clips"


def _common_voice_audio_path(
    root: Path,
    audio_root: Path,
    clip_path: str,
    audio_extension: str | None,
) -> Path:
    path = Path(clip_path)
    if audio_extension:
        suffix = audio_extension if audio_extension.startswith(".") else f".{audio_extension}"
        path = path.with_suffix(suffix)
    if path.is_absolute():
        return path

    candidates = [audio_root / path, audio_root / path.name, root / path]
    if path.parts:
        candidates.append(root / path.parts[0] / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _csv_delimiter(path: Path) -> str:
    return "\t" if path.suffix.lower() == ".tsv" else ","


def _build_l2_transcript_index(root: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or _is_hidden_path(path):
            continue
        if path.name == "txt.done.data":
            index.update(_read_txt_done_data(path))
        elif path.suffix.lower() in TRANSCRIPT_EXTENSIONS and _looks_like_transcript_path(path):
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                index.setdefault(path.stem.lower(), _clean_transcript_text(text))
    return index


def _read_txt_done_data(path: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = re.match(r'\s*\(?\s*([^\s()]+)\s+"(.*)"\s*\)?\s*$', line)
        if not match:
            continue
        utterance_id, transcript = match.groups()
        transcript = _clean_transcript_text(transcript)
        if transcript:
            index[Path(utterance_id).stem.lower()] = transcript
    return index


def _lookup_l2_transcript(wav_path: Path, transcript_index: dict[str, str]) -> str:
    stem = wav_path.stem.lower()
    if stem in transcript_index:
        return transcript_index[stem]

    for directory in [wav_path.parent, wav_path.parent.parent]:
        for suffix in TRANSCRIPT_EXTENSIONS:
            candidate = directory / f"{wav_path.stem}{suffix}"
            if candidate.exists():
                text = candidate.read_text(encoding="utf-8", errors="ignore").strip()
                if text:
                    return _clean_transcript_text(text)
    return ""


def _looks_like_transcript_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return bool(parts & {"etc", "txt", "text", "texts", "prompt", "prompts", "transcript", "transcripts"})


def _discover_accent_map(root: Path) -> dict[str, str]:
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".csv", ".tsv"}:
            continue
        mapping = _load_accent_map(path)
        if mapping:
            return mapping
    return {}


def _load_accent_map(path: Path) -> dict[str, str]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if reader.fieldnames is None:
                return {}
            speaker_column = _find_column(reader.fieldnames, ["speaker", "speaker_id", "id", "name"])
            accent_column = _find_column(
                reader.fieldnames,
                ["accent", "l1", "native_language", "first_language", "language"],
            )
            if speaker_column is None or accent_column is None:
                return {}
            return {
                row[speaker_column].strip().lower(): row[accent_column].strip()
                for row in reader
                if row.get(speaker_column, "").strip() and row.get(accent_column, "").strip()
            }
    except UnicodeDecodeError:
        return {}


def _find_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {column.lower().strip(): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _infer_l2_speaker(root: Path, wav_path: Path) -> str:
    relative = wav_path.relative_to(root)
    if len(relative.parts) >= 3 and relative.parts[1].lower() in {"wav", "wavs", "audio"}:
        return relative.parts[0]
    return wav_path.parent.name


def _first_present(row: dict[str, str], keys: list[str]) -> str:
    for key in keys:
        value = row.get(key, "").strip()
        if value:
            return value
    return ""


def _matches_filter(value: str, contains: str | None) -> bool:
    if contains is None:
        return True
    return contains.lower() in value.lower()


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("._-")
    return safe or "sample"


def _clean_transcript_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r'^\(?\s*[^\s()]+\s+"(.*)"\s*\)?$', r"\1", text)
    text = text.replace('\\"', '"')
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip('"')


def _is_hidden_path(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


if __name__ == "__main__":
    raise SystemExit(main())
