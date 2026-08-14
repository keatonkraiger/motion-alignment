"""Phase-1 LOO-10 Siamese training, ENCODER-AGNOSTIC.

Mirrors ``train_phase_1.py`` (same data split, same loss, same SGD/AdamW
recipe, same batch size, same loss-plot output) but adds:

* ``--encoder {cnn,poseformer}`` to swap encoders at the call site so
  baseline and ablation runs share an identical code path -> a fair
  apples-to-apples comparison.
* Total + per-epoch training-time tracking (``benchmark.TrainingTimer``).
* Total / trainable parameter counts.
* Peak GPU memory during training.
* A ``benchmarks.json`` next to the model with all of the above.

Existing scripts (``train_phase_1.py``, ``run_figure6.py``, ...) are
untouched. To reproduce the previous CNN-baseline results pass
``--encoder cnn`` (default).
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from data_loading import load_all_takes_except, load_subject
from loss import CrossCorrLoss
from model import Encoder
from model_poseformer import PoseFormerEncoder
from model_mixste import MixSTEEncoder
from model_poseformerv2 import PoseFormerV2Encoder
from benchmark import (
    TrainingTimer,
    count_flops,
    count_parameters,
    state_dict_bytes,
    train_model_benchmarked,
    write_metrics,
)


def build_encoder(name: str) -> torch.nn.Module:
    if name == "cnn":
        return Encoder()
    if name == "poseformer":
        return PoseFormerEncoder()
    if name == "mixste":
        return MixSTEEncoder()
    if name == "poseformerv2":
        return PoseFormerV2Encoder()
    raise ValueError(
        f"Unknown encoder: {name!r}. Expected one of "
        "'cnn', 'poseformer', 'mixste', 'poseformerv2'."
    )


# Per-encoder optimizer presets pulled directly from each method's own
# training script (see other_methods/<method>/...). These are used when
# ``--use-original-hparams`` is set so each ablation can be run with its
# author-recommended optimizer + LR + weight-decay + per-epoch decay
# instead of the CNN baseline's SGD recipe.
#
# NOTE: ``lr_div_batchsize`` mirrors the existing CNN baseline quirk
# (``lr / batch_size``). For the transformer methods their original code
# uses the LR as-is, so we set it to False there.
ORIGINAL_HPARAMS = {
    "cnn": {
        "optimizer": "SGD",
        "lr": 0.01,
        "weight_decay": 5e-4,
        "lr_decay": 1.0,
        "lr_div_batchsize": True,
    },
    "poseformer": {
        "optimizer": "AdamW",
        "lr": 1e-4,
        "weight_decay": 0.1,
        "lr_decay": 0.99,
        "lr_div_batchsize": False,
    },
    "poseformerv2": {
        "optimizer": "AdamW",
        "lr": 1e-4,
        "weight_decay": 0.1,
        "lr_decay": 0.99,
        "lr_div_batchsize": False,
    },
    "mixste": {
        "optimizer": "AdamW",
        "lr": 4e-5,
        "weight_decay": 0.1,
        "lr_decay": 0.99,
        "lr_div_batchsize": False,
    },
}


def main(
    output_dir: Path,
    data_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    optim_name: str,
    print_freq: int,
    encoder_name: str,
    seed: int | None,
    use_original_hparams: bool = False,
    lr_decay: float = 1.0,
    num_workers: int = 0,
):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(output_dir / "logs.txt")
    if log_file.exists():
        log_file.unlink()
    log_file.touch()

    # Resolve hyperparameters: when --use-original-hparams is set, the
    # encoder-specific preset overrides whatever the CLI passed in.
    lr_div_batchsize = True  # baseline behavior (CNN recipe)
    if use_original_hparams:
        preset = ORIGINAL_HPARAMS[encoder_name]
        optim_name = preset["optimizer"]
        lr = preset["lr"]
        weight_decay = preset["weight_decay"]
        lr_decay = preset["lr_decay"]
        lr_div_batchsize = preset["lr_div_batchsize"]
        msg = (
            f"[hparams] using original {encoder_name} preset: "
            f"optimizer={optim_name} lr={lr} weight_decay={weight_decay} "
            f"lr_decay={lr_decay} lr_div_batchsize={lr_div_batchsize}"
        )
        print(msg)
        with open(log_file, "a") as writer:
            writer.write(msg + "\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using", device)
    with open(log_file, "a") as writer:
        writer.write(f"Using {device}\n")
        writer.write(f"Encoder: {encoder_name}\n")

    # --- Data. -----------------------------------------------------------
    test_subject = 10
    print("Training stage 1 without subject", test_subject)
    with open(log_file, "a") as writer:
        writer.write(f"Training stage 1 without subject {test_subject}\n")
    train_val_a, train_val_b, _ = load_all_takes_except(
        data_dir, test_subject, wrapper_fn=torch.tensor
    )
    test_a, test_b, _ = load_subject(data_dir, test_subject, wrapper_fn=torch.tensor)

    all_indices = np.random.permutation(train_val_a.shape[0])
    cutoff = int(all_indices.shape[0] * 0.9)
    train_a = train_val_a[all_indices[:cutoff]]
    train_b = train_val_b[all_indices[:cutoff]]
    val_a = train_val_a[all_indices[cutoff:]]
    val_b = train_val_b[all_indices[cutoff:]]

    train_loader = DataLoader(TensorDataset(train_a, train_b), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device == "cuda"))
    val_loader = DataLoader(TensorDataset(val_a, val_b), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda"))
    test_loader = DataLoader(TensorDataset(test_a, test_b), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda"))

    # --- Model + optimizer. ---------------------------------------------
    model = build_encoder(encoder_name).to(device)
    param_counts = count_parameters(model)
    # Single-sample forward FLOPs (input shape (1, 1, 75, 117)).
    flops_info = count_flops(model, (1, 1, 75, 117), device=device)
    print(
        f"Encoder={encoder_name} | params total={param_counts['total']:,} "
        f"trainable={param_counts['trainable']:,} | "
        f"flops/sample={flops_info.get('flops')}"
    )
    with open(log_file, "a") as writer:
        writer.write(
            f"Params total={param_counts['total']} "
            f"trainable={param_counts['trainable']}\n"
        )

    criterion = CrossCorrLoss()

    effective_lr = (lr / batch_size) if lr_div_batchsize else lr
    if optim_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=effective_lr,
            momentum=0.9,
            weight_decay=weight_decay,
            nesterov=False,
        )
    elif optim_name == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=effective_lr,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optim_name}")

    scheduler = None
    if lr_decay is not None and lr_decay != 1.0:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

    # --- Train. ---------------------------------------------------------
    timer = TrainingTimer()
    model, train_loss_history, val_loss_history, test_loss = train_model_benchmarked(
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        test_loader,
        epochs,
        device,
        output_dir,
        log_file,
        print_freq,
        timer,
        scheduler=scheduler,
    )

    timing_summary = timer.summary()
    print(
        f"Training time: total={timing_summary['total_seconds']:.1f}s, "
        f"per-epoch mean={timing_summary['epoch_seconds_mean']:.2f}s "
        f"(peak GPU mem {timing_summary['peak_train_memory_mb']:.1f} MB)"
    )

    # --- Loss plot. -----------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    if train_loss_history:
        x_vals, y_vals = zip(*train_loss_history)
        ax[0].plot(x_vals, y_vals, label="Train loss")
        ax[1].loglog(x_vals, y_vals, label="Train loss")
    if val_loss_history:
        x_vals, y_vals = zip(*val_loss_history)
        ax[0].plot(x_vals, y_vals, label="Val loss")
        ax[1].loglog(x_vals, y_vals, label="Val loss")
    ax[0].plot(epochs * len(train_loader), test_loss, marker="o", label="Test loss")
    ax[1].loglog(epochs * len(train_loader), test_loss, marker="o", label="Test loss")
    for a in ax:
        a.set_title("Loss vs step")
        a.set_xlabel("Steps")
        a.set_ylabel("Loss")
        a.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_plot.png")
    plt.close(fig)

    # --- Save model + benchmarks. ---------------------------------------
    torch.save(model, output_dir / "model.pt")
    torch.save(model.state_dict(), output_dir / "model_state_dict.pt")

    metrics = {
        "phase": 1,
        "encoder": encoder_name,
        "device": device,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "lr_effective": effective_lr,
        "lr_decay": lr_decay,
        "lr_div_batchsize": lr_div_batchsize,
        "use_original_hparams": use_original_hparams,
        "weight_decay": weight_decay,
        "optimizer": optim_name,
        "num_workers": num_workers,
        "test_subject": test_subject,
        "n_train_pairs": int(train_a.shape[0]),
        "n_val_pairs": int(val_a.shape[0]),
        "n_test_pairs": int(test_a.shape[0]),
        "params": param_counts,
        "flops": flops_info,
        "model_state_dict_bytes": state_dict_bytes(model),
        "training": timing_summary,
        "final_train_loss": float(train_loss_history[-1][1]) if train_loss_history else None,
        "final_val_loss": float(val_loss_history[-1][1]) if val_loss_history else None,
        "test_loss": float(test_loss),
    }
    write_metrics(output_dir / "benchmarks.json", metrics)
    print(f"Saved benchmarks: {output_dir / 'benchmarks.json'}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default="outputs_phase_1_ablation")
    parser.add_argument(
        "--data-dir", type=Path,
        default="../data/TrainPatches",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--optimizer", choices=["SGD", "AdamW"], default="SGD")
    parser.add_argument(
        "--lr-decay", type=float, default=1.0,
        help="Per-epoch ExponentialLR gamma. 1.0 disables decay.",
    )
    parser.add_argument(
        "--use-original-hparams", action="store_true",
        help=("Override --optimizer/--lr/--weight-decay/--lr-decay with the "
              "encoder's author-recommended training recipe (see "
              "ORIGINAL_HPARAMS at the top of this file)."),
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader num_workers for train/val/test loaders.",
    )
    parser.add_argument("--print-freq", type=int, default=10)
    parser.add_argument(
        "--encoder",
        choices=["cnn", "poseformer", "mixste", "poseformerv2"],
        default="cnn",
        help="Which encoder to use inside the Siamese network.",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    main(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optim_name=args.optimizer,
        print_freq=args.print_freq,
        encoder_name=args.encoder,
        seed=args.seed,
        use_original_hparams=args.use_original_hparams,
        lr_decay=args.lr_decay,
        num_workers=args.num_workers,
    )
