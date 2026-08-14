"""Encoder-agnostic Figure-6 reproduction orchestrator.

Same flow as ``run_figure6.py`` (train -> alignments -> ROC) but with a
``--encoder`` switch and end-to-end benchmarking. Existing
``run_figure6.py`` is left untouched.

Examples
--------
CNN baseline (parity with ``run_figure6.py --train``)::

    python run_figure6_ablation.py --train --encoder cnn \
        --train-data-dir ../../data/TrainPatches \
        --patch-dir ../../data/Patches --epochs 38 \
        --output-dir ../../outputs/cnn_ablation

PoseFormer ablation::

    python run_figure6_ablation.py --train --encoder poseformer \
        --train-data-dir ../../data/TrainPatches \
        --patch-dir ../../data/Patches --epochs 38 \
        --output-dir ../../outputs/poseformer_ablation
"""

from __future__ import annotations

import datetime as _dt
import json
from argparse import ArgumentParser
from pathlib import Path
import subprocess
import sys

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _maybe_load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


class _Tee:
    """Minimal tee: write to multiple text streams, flush eagerly."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass
        return len(data)

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        return False


def _run_and_tee(cmd, log_fh):
    """Run a subprocess, streaming combined stdout/stderr to terminal + log."""
    log_fh.write(f"\n$ {' '.join(cmd)}\n")
    log_fh.flush()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.__stdout__.write(line)
        sys.__stdout__.flush()
        log_fh.write(line)
        log_fh.flush()
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def main(
    encoder_name: str,
    model_path: Path | None,
    patch_dir: Path,
    tmm100_path: Path,
    keyposes_path: Path,
    output_dir: Path,
    do_train: bool,
    train_data_dir: Path | None,
    epochs: int,
    seed: int | None,
    use_original_hparams: bool = False,
    num_workers: int = 0,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_basename = f"MyMocapKeyposeaccuracy_{encoder_name}"

    # ------------------------------------------------------------------
    # Log file: <output_dir>/logs/<encoder>_<timestamp>.log
    # Captures both this orchestrator's prints AND the training
    # subprocess's stdout/stderr so failures are diagnosable.
    # ------------------------------------------------------------------
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_id = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{encoder_name}_{run_id}.log"
    log_fh = open(log_path, "w", buffering=1)
    print(f"Logging to {log_path}")
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    sys.stderr = _Tee(sys.__stderr__, log_fh)

    # ------------------------------------------------------------------
    # Stage 1: train (optional).
    # ------------------------------------------------------------------
    train_metrics_path: Path | None = None
    if do_train:
        if train_data_dir is None:
            raise ValueError("--train-data-dir required when --train is set")
        print("=" * 60)
        print(f"Stage 1: training Phase-1 LOO-10 encoder ({encoder_name})")
        print("=" * 60)
        train_output = output_dir / "phase1"
        cmd = [
            sys.executable,
            "-u",
            str(SRC_DIR / "train_phase_1_ablation.py"),
            "--output-dir", str(train_output),
            "--data-dir", str(train_data_dir),
            "--epochs", str(epochs),
            "--encoder", encoder_name,
            "--num-workers", str(num_workers),
        ]
        if use_original_hparams:
            cmd.append("--use-original-hparams")
        if seed is not None:
            cmd += ["--seed", str(seed)]
        print(" ".join(cmd))
        _run_and_tee(cmd, log_fh)
        model_path = train_output / "model.pt"
        train_metrics_path = train_output / "benchmarks.json"

    if model_path is None:
        raise ValueError("Must provide --model or set --train.")

    # ------------------------------------------------------------------
    # Stage 2: alignments + inference benchmarking.
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Stage 2: computing DTW alignments + inference benchmarks")
    print("=" * 60)
    from compute_alignments_ablation import compute_alignments  # noqa: E402

    alignments_path = output_dir / "MyMocapAlignments.mat"
    align_metrics_path = output_dir / "alignment_benchmarks.json"
    compute_alignments(
        model_path=model_path,
        patch_dir=patch_dir,
        tmm100_path=tmm100_path,
        output_path=alignments_path,
        encoder_name=encoder_name,
        metrics_path=align_metrics_path,
    )

    # ------------------------------------------------------------------
    # Stage 3: ROC evaluation (unchanged module).
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Stage 3: keypose-transfer evaluation")
    print("=" * 60)
    from evaluate_keypose_transfer import evaluate  # noqa: E402

    result = evaluate(
        alignments_path=alignments_path,
        keyposes_path=keyposes_path,
        tmm100_path=tmm100_path,
        output_dir=output_dir,
        output_basename=output_basename,
    )
    print("Results:", result)

    # ------------------------------------------------------------------
    # Combined summary.
    # ------------------------------------------------------------------
    summary = {
        "encoder": encoder_name,
        "epochs": epochs,
        "outputs": {
            "alignments_mat": str(alignments_path),
            "roc_eps": str(result["eps_path"]),
            "roc_png": str(result["png_path"]),
            "roc_npz": str(result["roc_npz_path"]),
        },
        "roc": {
            "auc_norm": float(result["auc_norm"]),
            "halfsec_pct": float(result["halfsec_pct"]),
            "onesec_pct": float(result["onesec_pct"]),
            "thresholds_seconds": [
                float(t) / 50.0 for t in result["thresholds_frames"]
            ],
            "counts_pct": [float(c) * 100 for c in result["counts"]],
        },
        "training": _maybe_load_json(train_metrics_path) if train_metrics_path else {},
        "alignment": _maybe_load_json(align_metrics_path),
    }
    with open(output_dir / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=float)
    print(f"\nWrote {output_dir / 'ablation_summary.json'}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--encoder",
        choices=["cnn", "poseformer", "mixste", "poseformerv2"],
        default="cnn",
        help="Which encoder to use inside the Siamese network.",
    )
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--train-data-dir", type=Path, default=None)
    parser.add_argument(
        "--epochs", type=int, default=38,
        help="Phase-1 training epochs (capped at 50).",
    )
    parser.add_argument(
        "--use-original-hparams", action="store_true",
        help=("Use each encoder's author-recommended optimizer/LR/weight-"
              "decay/lr-decay recipe instead of the CNN baseline's SGD "
              "recipe. See ORIGINAL_HPARAMS in train_phase_1_ablation.py."),
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader num_workers passed to the Phase-1 training script.",
    )
    parser.add_argument("--patch-dir", type=Path, required=True)
    parser.add_argument(
        "--tmm100", type=Path,
        default=Path("../../data/eval/tmm100performances.mat"),
    )
    parser.add_argument(
        "--keyposes", type=Path,
        default=Path("../../data/eval/tmmkeyposes.mat"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("../outputs"))
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if not args.train and args.model is None:
        parser.error("Must provide --model or set --train.")

    if args.epochs > 50:
        print(f"[run_figure6_ablation] Capping --epochs from {args.epochs} to 50.")
        args.epochs = 50

    main(
        encoder_name=args.encoder,
        model_path=args.model,
        patch_dir=args.patch_dir,
        tmm100_path=args.tmm100,
        keyposes_path=args.keyposes,
        output_dir=args.output_dir,
        do_train=args.train,
        train_data_dir=args.train_data_dir,
        epochs=args.epochs,
        seed=args.seed,
        use_original_hparams=args.use_original_hparams,
        num_workers=args.num_workers,
    )
