"""Single entry point for the whole pipeline: given a YAML config, build
the model it names and run Phase 1 (train the Siamese encoder), Phase 2
(DTW-align every performance against the atlas + keypose-transfer ROC
evaluation), or both.

    python train.py configs/cnn.yaml
    python train.py configs/poseformer.yaml --phase 1
    python train.py configs/mixste.yaml --phase 2 --model outputs/mixste_ablation/phase1/model.pt

The ROC curve (what the PSU-TMM100 paper calls "Figure 6") isn't a
separate step -- it's just what phase 2's evaluation writes out.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models import ORIGINAL_HPARAMS, build_model  # noqa: E402
from models.loss import CrossCorrLoss  # noqa: E402
from data.datasets import load_all_takes_except, load_subject  # noqa: E402
from benchmark import (  # noqa: E402
    TrainingTimer,
    count_flops,
    count_parameters,
    state_dict_bytes,
    train_model_benchmarked,
    write_metrics,
)
from evaluation.alignments import compute_alignments  # noqa: E402
from evaluation.keypose_eval import evaluate as evaluate_keypose_transfer  # noqa: E402

log = logging.getLogger(__name__)


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    fmt = logging.Formatter("%(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    root.addHandler(stream)
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


def resolve_training_hparams(cfg: dict) -> dict:
    """Apply the ORIGINAL_HPARAMS preset if training.use_original_hparams
    is set, matching the model named in cfg["model"]["name"]."""
    training = dict(cfg["training"])
    if training.get("use_original_hparams"):
        preset = ORIGINAL_HPARAMS[cfg["model"]["name"]]
        training["optimizer"] = preset["optimizer"]
        training["lr"] = preset["lr"]
        training["weight_decay"] = preset["weight_decay"]
        training["lr_decay"] = preset["lr_decay"]
        training["lr_div_batchsize"] = preset["lr_div_batchsize"]
        log.info("[hparams] using original %s preset: %s", cfg["model"]["name"], preset)
    return training


def run_phase1(cfg: dict, output_dir: Path) -> Path:
    """Train the Phase-1 LOO-10 Siamese encoder. Returns the model checkpoint path."""
    log.info("=" * 60)
    log.info("Phase 1: training Siamese encoder (%s)", cfg["model"]["name"])
    log.info("=" * 60)

    seed = cfg.get("seed")
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    phase1_dir = output_dir / "phase1"
    phase1_dir.mkdir(parents=True, exist_ok=True)

    training = resolve_training_hparams(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Using %s", device)
    log.info("Encoder: %s", cfg["model"]["name"])

    # --- Data. -------------------------------------------------------
    test_subject = training.get("test_subject", 10)
    log.info("Training stage 1 without subject %s", test_subject)
    data_dir = Path(cfg["data"]["train_patches_dir"])
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

    batch_size = training.get("batch_size", 128)
    num_workers = training.get("num_workers", 0)
    pin_memory = device == "cuda"
    train_loader = DataLoader(TensorDataset(train_a, train_b), batch_size=batch_size,
                               shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(TensorDataset(val_a, val_b), batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(TensorDataset(test_a, test_b), batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # --- Model + optimizer. -------------------------------------------
    model = build_model(cfg["model"]["name"], cfg["model"].get("params")).to(device)
    param_counts = count_parameters(model)
    flops_info = count_flops(model, (1, 1, 75, 117), device=device)
    log.info(
        "Encoder=%s | params total=%s trainable=%s | flops/sample=%s",
        cfg["model"]["name"], f"{param_counts['total']:,}",
        f"{param_counts['trainable']:,}", flops_info.get("flops"),
    )

    criterion = CrossCorrLoss()
    lr = training.get("lr", 0.01)
    lr_div_batchsize = training.get("lr_div_batchsize", True)
    effective_lr = (lr / batch_size) if lr_div_batchsize else lr
    weight_decay = training.get("weight_decay", 5e-4)
    optim_name = training.get("optimizer", "SGD")
    if optim_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=effective_lr, momentum=0.9,
                               weight_decay=weight_decay, nesterov=False)
    elif optim_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optim_name}")

    lr_decay = training.get("lr_decay", 1.0)
    scheduler = None
    if lr_decay is not None and lr_decay != 1.0:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

    # --- Train. --------------------------------------------------------
    epochs = min(training.get("epochs", 38), 50)
    timer = TrainingTimer()
    model, train_loss_history, val_loss_history, test_loss = train_model_benchmarked(
        model, optimizer, criterion, train_loader, val_loader, test_loader,
        epochs, device, training.get("print_freq", 10), timer, scheduler=scheduler,
    )

    timing_summary = timer.summary()
    log.info(
        "Training time: total=%.1fs, per-epoch mean=%.2fs (peak GPU mem %.1f MB)",
        timing_summary["total_seconds"], timing_summary["epoch_seconds_mean"],
        timing_summary["peak_train_memory_mb"],
    )

    # --- Loss plot. ------------------------------------------------------
    import matplotlib.pyplot as plt
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
    fig.savefig(phase1_dir / "loss_plot.png")
    plt.close(fig)

    # --- Save model + benchmarks. -----------------------------------------
    model_path = phase1_dir / "model.pt"
    torch.save(model, model_path)
    torch.save(model.state_dict(), phase1_dir / "model_state_dict.pt")

    metrics = {
        "phase": 1,
        "encoder": cfg["model"]["name"],
        "model_params": cfg["model"].get("params") or {},
        "device": device,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "lr_effective": effective_lr,
        "lr_decay": lr_decay,
        "lr_div_batchsize": lr_div_batchsize,
        "use_original_hparams": training.get("use_original_hparams", False),
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
    write_metrics(phase1_dir / "benchmarks.json", metrics)
    log.info("Saved benchmarks: %s", phase1_dir / "benchmarks.json")
    return model_path


def run_phase2(cfg: dict, model_path: Path, output_dir: Path) -> dict:
    """DTW-align every performance against the atlas, then run the
    keypose-transfer evaluation. The ROC curve is produced here, as a
    direct product of evaluation -- not a separate step."""
    log.info("=" * 60)
    log.info("Phase 2: computing DTW alignments")
    log.info("=" * 60)

    evaluation_cfg = cfg.get("evaluation", {})
    encoder_name = cfg["model"]["name"]
    alignments_path = output_dir / "MyMocapAlignments.mat"
    align_metrics_path = output_dir / "alignment_benchmarks.json"
    compute_alignments(
        model_path=model_path,
        patch_dir=Path(cfg["data"]["patches_dir"]),
        tmm100_path=Path(cfg["data"]["tmm100_path"]),
        output_path=alignments_path,
        encoder_name=encoder_name,
        model_params=cfg["model"].get("params"),
        metrics_path=align_metrics_path,
        benchmark_take_count=evaluation_cfg.get("benchmark_take_count", 5),
        atlas_subj=evaluation_cfg.get("atlas_subject", 7),
        atlas_take=evaluation_cfg.get("atlas_take", 2),
    )

    log.info("=" * 60)
    log.info("Phase 2: keypose-transfer evaluation (ROC curve)")
    log.info("=" * 60)
    result = evaluate_keypose_transfer(
        alignments_path=alignments_path,
        keyposes_path=Path(cfg["data"]["keyposes_path"]),
        tmm100_path=Path(cfg["data"]["tmm100_path"]),
        output_dir=output_dir,
        output_basename=f"MyMocapKeyposeaccuracy_{encoder_name}",
    )
    log.info("Results: AUC=%.4f, within 0.5s=%.2f%%, within 1.0s=%.2f%%",
              result["auc_norm"], result["halfsec_pct"], result["onesec_pct"])

    summary = {
        "encoder": encoder_name,
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
            "thresholds_seconds": [float(t) / 50.0 for t in result["thresholds_frames"]],
            "counts_pct": [float(c) * 100 for c in result["counts"]],
        },
        "alignment": json.load(open(align_metrics_path)) if align_metrics_path.exists() else {},
    }
    phase1_metrics_path = output_dir / "phase1" / "benchmarks.json"
    if phase1_metrics_path.exists():
        summary["training"] = json.load(open(phase1_metrics_path))
    with open(output_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=float)
    log.info("Wrote %s", output_dir / "run_summary.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to a YAML config (see configs/).")
    parser.add_argument("--phase", choices=["1", "2", "both"], default=None,
                         help="Override the config's `phase` field.")
    parser.add_argument("--model", type=Path, default=None,
                         help="Checkpoint to evaluate; only used with --phase 2 "
                              "(overrides evaluation.model_path in the config).")
    parser.add_argument("--seed", type=int, default=None,
                         help="Override the config's `seed` field.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    phase = args.phase or str(cfg.get("phase", "both"))

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "train.log")
    log.info("Config: %s | phase=%s | output_dir=%s", args.config, phase, output_dir)

    model_path = args.model
    if phase in ("1", "both"):
        model_path = run_phase1(cfg, output_dir)
    elif model_path is None:
        model_path_cfg = cfg.get("evaluation", {}).get("model_path")
        if model_path_cfg is None:
            parser.error("phase 2 alone requires --model or evaluation.model_path in the config.")
        model_path = Path(model_path_cfg)

    if phase in ("2", "both"):
        run_phase2(cfg, model_path, output_dir)


if __name__ == "__main__":
    main()
