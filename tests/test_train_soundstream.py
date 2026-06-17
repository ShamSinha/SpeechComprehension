from __future__ import annotations

from pathlib import Path

import pytest

from speech_comprehension.train_soundstream import (
    LocalTrainingLogger,
    _next_run_dir,
    build_parser,
)


def test_train_soundstream_parser_defaults() -> None:
    args = build_parser().parse_args(["--folder", "audio"])

    assert args.folder == Path("audio")
    assert args.results_folder == Path("runs/soundstream")
    assert args.sample_rate == 16000
    assert args.codebook_size == 1024
    assert args.rq_num_quantizers == 8
    assert args.data_max_length_seconds == 3.0
    assert args.multi_spectral_recon_loss_weight == 0.0


def test_next_run_dir_uses_lightning_style_versions(tmp_path: Path) -> None:
    assert _next_run_dir(tmp_path, None) == tmp_path / "version_0"
    (tmp_path / "version_0").mkdir()
    assert _next_run_dir(tmp_path, None) == tmp_path / "version_1"
    assert _next_run_dir(tmp_path, "named") == tmp_path / "named"


def test_local_training_logger_writes_hparams_jsonl_and_csv(tmp_path: Path) -> None:
    logger = LocalTrainingLogger(
        run_dir=tmp_path / "version_0",
        hparams={"batch_size": 1, "folder": "audio"},
    )

    logger.log({"loss": 1.25, "custom metric": 0.5}, step=3)
    logger.close()

    run_dir = tmp_path / "version_0"
    assert (run_dir / "hparams.json").exists()
    assert '"batch_size": 1' in (run_dir / "hparams.json").read_text(encoding="utf-8")
    assert '"loss": 1.25' in (run_dir / "metrics.jsonl").read_text(encoding="utf-8")
    assert "custom metric" in (run_dir / "metrics.csv").read_text(encoding="utf-8")


def test_local_training_logger_stops_on_nonfinite_loss(tmp_path: Path) -> None:
    logger = LocalTrainingLogger(run_dir=tmp_path / "version_0", hparams={})

    with pytest.raises(FloatingPointError, match="Non-finite SoundStream loss"):
        logger.log({"loss": float("inf"), "recon_loss": 0.1}, step=5)

    logger.close()
