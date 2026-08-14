"""End-to-end Figure 6 reproduction orchestrator.

Stages:
    1. (optional) Train a fresh LOO-10 Phase-1 Siamese encoder.
    2. Compute alignments between atlas (S7T2) and all valid PSU-TMM100
       performances using the trained encoder.
    3. Run keypose-transfer evaluation -> ROC-style curve + EPS plot.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import subprocess
import sys

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main(
    model_path: Path,
    patch_dir: Path,
    tmm100_path: Path,
    keyposes_path: Path,
    output_dir: Path,
    do_train: bool,
    train_data_dir: Path | None,
    epochs: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    if do_train:
        if train_data_dir is None:
            raise ValueError("--train-data-dir required when --train is set")
        print("=" * 60)
        print("Stage 1: training Phase-1 LOO-10 encoder")
        print("=" * 60)
        train_output = output_dir / "phase1"
        cmd = [
            sys.executable,
            str(SRC_DIR / "train_phase_1.py"),
            "--output-dir", str(train_output),
            "--data-dir", str(train_data_dir),
            "--epochs", str(epochs),
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        model_path = train_output / "model.pt"

    print("=" * 60)
    print("Stage 2: computing DTW alignments")
    print("=" * 60)
    from compute_alignments import compute_alignments  # noqa: E402

    alignments_path = output_dir / "MyMocapAlignments.mat"
    compute_alignments(
        model_path=model_path,
        patch_dir=patch_dir,
        tmm100_path=tmm100_path,
        output_path=alignments_path,
    )

    print("=" * 60)
    print("Stage 3: keypose-transfer evaluation")
    print("=" * 60)
    from evaluate_keypose_transfer import evaluate  # noqa: E402

    result = evaluate(
        alignments_path=alignments_path,
        keyposes_path=keyposes_path,
        tmm100_path=tmm100_path,
        output_dir=output_dir,
        output_basename="MyMocapKeyposeaccuracy",
    )
    print("Results:", result)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=Path, default=None,
                        help="Path to trained model.pt. If --train is set, this is overwritten.")
    parser.add_argument("--train", action="store_true",
                        help="Train a fresh Phase-1 LOO-10 encoder before evaluation.")
    parser.add_argument("--train-data-dir", type=Path, default=None,
                        help="Patches directory used for TRAINING (PRML 20-take subset).")
    parser.add_argument("--epochs", type=int, default=38)
    parser.add_argument("--patch-dir", type=Path, required=True,
                        help="Patches directory used for EVALUATION (full valid set).")
    parser.add_argument(
        "--tmm100", type=Path,
        default=Path("../../data/eval/tmm100performances.mat"),
    )
    parser.add_argument(
        "--keyposes", type=Path,
        default=Path("../../data/eval/tmmkeyposes.mat"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("../outputs"))
    args = parser.parse_args()

    if not args.train and args.model is None:
        parser.error("Must provide --model or set --train.")

    main(
        model_path=args.model,
        patch_dir=args.patch_dir,
        tmm100_path=args.tmm100,
        keyposes_path=args.keyposes,
        output_dir=args.output_dir,
        do_train=args.train,
        train_data_dir=args.train_data_dir,
        epochs=args.epochs,
    )
