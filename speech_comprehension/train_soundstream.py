from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train an audiolm-pytorch SoundStream codec on a local audio folder. "
            "This is codec pretraining; it does not train the intelligibility editor."
        )
    )
    parser.add_argument(
        "--folder",
        type=Path,
        required=True,
        help="Audio folder to scan recursively. Supported by audiolm-pytorch: flac, wav, mp3, webm.",
    )
    parser.add_argument(
        "--results-folder",
        type=Path,
        default=Path("runs/soundstream"),
        help="Folder for checkpoints, samples, and trainer state.",
    )
    parser.add_argument("--num-train-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-every", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--data-max-length-seconds", type=float, default=3.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--codebook-size", type=int, default=1024)
    parser.add_argument("--rq-num-quantizers", type=int, default=8)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument(
        "--soundstream-checkpoint",
        type=Path,
        default=None,
        help="Optional SoundStream checkpoint to load before training/fine-tuning.",
    )
    parser.add_argument("--valid-frac", type=float, default=0.05)
    parser.add_argument("--save-results-every", type=int, default=100)
    parser.add_argument("--save-model-every", type=int, default=1000)
    parser.add_argument("--log-losses-every", type=int, default=1)
    parser.add_argument("--dl-num-workers", type=int, default=0)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Local metrics log root. Defaults to RESULTS_FOLDER/logs.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name under the log directory. Defaults to version_N.",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Also write TensorBoard event files. Requires the tensorboard package.",
    )
    parser.add_argument(
        "--mixed-precision",
        choices=["no", "fp16", "bf16"],
        default="no",
        help="Accelerate mixed precision mode.",
    )
    parser.add_argument(
        "--use-wandb-tracking",
        action="store_true",
        help="Enable audiolm-pytorch/accelerate W&B tracking.",
    )
    parser.add_argument(
        "--force-clear-prev-results",
        action="store_true",
        help="Let the trainer clear an existing results folder before training.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    folder = args.folder.expanduser().resolve()
    if not folder.exists():
        parser.error(f"--folder does not exist: {folder}")
    if not folder.is_dir():
        parser.error(f"--folder must be a directory: {folder}")

    try:
        import torch
        from audiolm_pytorch import SoundStream, SoundStreamTrainer
    except ImportError as exc:
        raise SystemExit(
            "SoundStream training requires audiolm-pytorch. Install it with "
            '`pip install -e ".[soundstream]"`.'
        ) from exc

    results_folder = args.results_folder.expanduser().resolve()
    results_folder.mkdir(parents=True, exist_ok=True)

    if args.soundstream_checkpoint:
        checkpoint = args.soundstream_checkpoint.expanduser().resolve()
        if not checkpoint.exists():
            parser.error(f"--soundstream-checkpoint does not exist: {checkpoint}")
        soundstream = SoundStream.init_and_load_from(str(checkpoint))
    else:
        soundstream = SoundStream(
            target_sample_hz=args.sample_rate,
            codebook_size=args.codebook_size,
            rq_num_quantizers=args.rq_num_quantizers,
            channels=args.channels,
        )

    accelerate_kwargs = {}
    if args.mixed_precision != "no":
        accelerate_kwargs["mixed_precision"] = args.mixed_precision

    log_root = (args.log_dir or (results_folder / "logs")).expanduser().resolve()
    run_dir = _next_run_dir(log_root, args.run_name)
    logger = LocalTrainingLogger(
        run_dir=run_dir,
        hparams=_json_ready(vars(args))
        | {
            "resolved_folder": str(folder),
            "resolved_results_folder": str(results_folder),
            "cuda_available": torch.cuda.is_available(),
        },
        tensorboard=args.tensorboard,
    )

    print(f"Training SoundStream codec from: {folder}")
    print(f"Writing trainer outputs to: {results_folder}")
    print(f"Writing metrics logs to: {run_dir}")
    print(f"CUDA available to PyTorch: {torch.cuda.is_available()}")
    print(
        "Codec config: "
        f"{args.sample_rate} Hz, codebook_size={args.codebook_size}, "
        f"rq_num_quantizers={args.rq_num_quantizers}"
    )

    trainer = SoundStreamTrainer(
        soundstream,
        folder=str(folder),
        results_folder=str(results_folder),
        num_train_steps=args.num_train_steps,
        batch_size=args.batch_size,
        grad_accum_every=args.grad_accum_every,
        lr=args.lr,
        data_max_length_seconds=args.data_max_length_seconds,
        valid_frac=args.valid_frac,
        save_results_every=args.save_results_every,
        save_model_every=args.save_model_every,
        log_losses_every=args.log_losses_every,
        dl_num_workers=args.dl_num_workers,
        use_wandb_tracking=args.use_wandb_tracking,
        force_clear_prev_results=args.force_clear_prev_results,
        accelerate_kwargs=accelerate_kwargs,
    )
    try:
        if args.use_wandb_tracking:
            with trainer.wandb_tracker(project="soundstream", run=args.run_name):
                trainer.train(log_fn=lambda logs: logger.log(logs, step=_trainer_log_step(trainer)))
        else:
            trainer.train(log_fn=lambda logs: logger.log(logs, step=_trainer_log_step(trainer)))
    finally:
        logger.close()
    return 0


class LocalTrainingLogger:
    def __init__(
        self,
        run_dir: Path,
        hparams: dict[str, Any],
        tensorboard: bool = False,
    ) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.started_at = time.time()
        self.jsonl_path = self.run_dir / "metrics.jsonl"
        self.csv_path = self.run_dir / "metrics.csv"
        self.jsonl_handle = self.jsonl_path.open("a", encoding="utf-8")
        self.csv_handle = self.csv_path.open("a", encoding="utf-8", newline="")
        self.csv_writer = csv.DictWriter(
            self.csv_handle,
            fieldnames=[
                "step",
                "time_seconds",
                "loss",
                "recon_loss",
                "multi_spectral_recon_loss",
                "adversarial_loss",
                "feature_loss",
                "all_commitment_loss",
                "extra_metrics_json",
            ],
        )
        if self.csv_path.stat().st_size == 0:
            self.csv_writer.writeheader()

        (self.run_dir / "hparams.json").write_text(
            json.dumps(hparams, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.writer = _make_tensorboard_writer(self.run_dir) if tensorboard else None

    def log(self, logs: dict[str, Any], step: int) -> None:
        row = {
            "step": step,
            "time_seconds": round(time.time() - self.started_at, 3),
            **_numeric_logs(logs),
        }
        self.jsonl_handle.write(json.dumps(row, sort_keys=True) + "\n")
        self.jsonl_handle.flush()

        known_csv_keys = set(self.csv_writer.fieldnames or [])
        csv_row = {key: row.get(key, "") for key in known_csv_keys}
        csv_row["extra_metrics_json"] = json.dumps(
            {
                key: value
                for key, value in row.items()
                if key not in known_csv_keys and key not in {"step", "time_seconds"}
            },
            sort_keys=True,
        )
        self.csv_writer.writerow(csv_row)
        self.csv_handle.flush()

        if self.writer is not None:
            for key, value in row.items():
                if key in {"step", "time_seconds"}:
                    continue
                self.writer.add_scalar(_tensorboard_tag(key), value, step)
            self.writer.flush()

    def close(self) -> None:
        self.jsonl_handle.close()
        self.csv_handle.close()
        if self.writer is not None:
            self.writer.close()


def _next_run_dir(log_root: Path, run_name: str | None) -> Path:
    log_root.mkdir(parents=True, exist_ok=True)
    if run_name:
        return log_root / run_name

    version = 0
    while True:
        candidate = log_root / f"version_{version}"
        if not candidate.exists():
            return candidate
        version += 1


def _make_tensorboard_writer(run_dir: Path) -> Any:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise SystemExit(
            "TensorBoard logging requested, but tensorboard is not installed. "
            "Install it with `pip install tensorboard`, or run without --tensorboard."
        ) from exc
    return SummaryWriter(log_dir=str(run_dir / "tensorboard"))


def _trainer_log_step(trainer: Any) -> int:
    return max(int(trainer.steps.item()) - 1, 0)


def _numeric_logs(logs: dict[str, Any]) -> dict[str, float]:
    numeric: dict[str, float] = {}
    for key, value in logs.items():
        try:
            numeric[key] = float(value)
        except (TypeError, ValueError):
            continue
    return numeric


def _tensorboard_tag(key: str) -> str:
    return key.replace(" ", "_").replace("(", "").replace(")", "")


def _json_ready(values: dict[str, Any]) -> dict[str, Any]:
    ready: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, Path):
            ready[key] = str(value)
        else:
            ready[key] = value
    return ready


if __name__ == "__main__":
    raise SystemExit(main())
