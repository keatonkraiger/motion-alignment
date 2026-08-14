"""Benchmarking utilities for encoder ablations.

Tracks the metrics the user wants to compare across encoders
(CNN baseline vs. PoseFormer):

* Total / trainable parameter counts.
* Per-encoder size on disk (bytes of state_dict).
* Training wall-clock time (total + per-epoch hook).
* Inference latency: per-sample, per-batch, per-take, throughput.
* Peak GPU memory during training and inference (when CUDA is available).

These utilities are pure-Python and have no third-party deps beyond
torch + numpy. Both `train_phase_1_ablation.py` and
`compute_alignments_ablation.py` use them.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch


def count_parameters(model: torch.nn.Module) -> dict:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def state_dict_bytes(model: torch.nn.Module) -> int:
    """Approximate state_dict size in bytes (sum of tensor.numel * element_size)."""
    return int(sum(t.numel() * t.element_size() for t in model.state_dict().values()))


def count_flops(
    model: torch.nn.Module,
    input_shape: tuple,
    device: str | torch.device = "cpu",
) -> dict:
    """Count multiply-accumulate FLOPs for a single forward pass.

    Uses ``fvcore.nn.FlopCountAnalysis`` (counts multiply-adds, i.e. one
    multiply + one add ~= 1 flop in fvcore's accounting). Returns a dict
    with ``flops`` (total), ``mult_adds`` (alias) and ``per_module`` (top
    contributors). Returns ``{"flops": None, "error": ...}`` if fvcore is
    not available.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception as e:  # pragma: no cover
        return {"flops": None, "error": f"fvcore unavailable: {e}"}

    was_training = model.training
    model.eval()
    x = torch.zeros(input_shape, device=device)
    try:
        analysis = FlopCountAnalysis(model, x)
        analysis.unsupported_ops_warnings(False)
        analysis.uncalled_modules_warnings(False)
        total = int(analysis.total())
        per_mod = {k: int(v) for k, v in analysis.by_module().items()}
        # Keep only the largest-cost top-level / leaf modules to bound size.
        top = dict(
            sorted(per_mod.items(), key=lambda kv: -kv[1])[:25]
        )
    finally:
        if was_training:
            model.train()
    return {
        "flops": total,
        "mult_adds": total,
        "input_shape": [int(x) for x in input_shape],
        "top_modules": top,
    }


@contextmanager
def cuda_sync(device: str | torch.device | None) -> Iterator[None]:
    """Context that forces a CUDA sync at entry and exit so wall-clock
    times around a block include all queued GPU work."""
    is_cuda = torch.cuda.is_available() and (
        device is None or str(device).startswith("cuda")
    )
    if is_cuda:
        torch.cuda.synchronize()
    try:
        yield
    finally:
        if is_cuda:
            torch.cuda.synchronize()


@dataclass
class TrainingTimer:
    """Tracks per-epoch and total training time."""

    epoch_seconds: list[float] = field(default_factory=list)
    total_seconds: float = 0.0
    peak_train_memory_mb: float = 0.0

    @contextmanager
    def epoch(self, device: str | torch.device | None = None) -> Iterator[None]:
        with cuda_sync(device):
            t0 = time.perf_counter()
        try:
            yield
        finally:
            with cuda_sync(device):
                t1 = time.perf_counter()
            self.epoch_seconds.append(t1 - t0)

    @contextmanager
    def run(self, device: str | torch.device | None = None) -> Iterator[None]:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        with cuda_sync(device):
            t0 = time.perf_counter()
        try:
            yield
        finally:
            with cuda_sync(device):
                t1 = time.perf_counter()
            self.total_seconds = t1 - t0
            if torch.cuda.is_available():
                self.peak_train_memory_mb = (
                    torch.cuda.max_memory_allocated() / (1024 ** 2)
                )

    def summary(self) -> dict:
        return {
            "total_seconds": float(self.total_seconds),
            "num_epochs": len(self.epoch_seconds),
            "epoch_seconds": [float(x) for x in self.epoch_seconds],
            "epoch_seconds_mean": (
                float(np.mean(self.epoch_seconds)) if self.epoch_seconds else 0.0
            ),
            "epoch_seconds_std": (
                float(np.std(self.epoch_seconds)) if self.epoch_seconds else 0.0
            ),
            "peak_train_memory_mb": float(self.peak_train_memory_mb),
        }


def _time_forward(
    model: torch.nn.Module,
    sample: torch.Tensor,
    device: str | torch.device,
    n_warmup: int,
    n_iters: int,
) -> list[float]:
    """Run ``n_iters`` forward passes on ``sample`` and return per-iter wall-clock times."""
    model.eval()
    times: list[float] = []
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(sample)
        with cuda_sync(device):
            pass
        for _ in range(n_iters):
            with cuda_sync(device):
                t0 = time.perf_counter()
            _ = model(sample)
            with cuda_sync(device):
                t1 = time.perf_counter()
            times.append(t1 - t0)
    return times


def benchmark_inference(
    model: torch.nn.Module,
    take_tensors: Iterable[torch.Tensor],
    device: str | torch.device,
    batch_size: int = 128,
    single_sample_iters: int = 50,
    single_sample_warmup: int = 10,
    take_iters: int = 5,
    take_warmup: int = 2,
) -> dict:
    """Measure inference latency in several useful ways.

    Parameters
    ----------
    model : nn.Module
        The encoder. Must accept inputs of shape ``(N, 1, S, F)``.
    take_tensors : iterable of torch.Tensor
        One tensor per take, each of shape ``(N_i, 1, S, F)`` already on
        ``device`` or convertible to it. Latencies are reported across takes.
    batch_size : int
        Batch size used for the "batched" measurement. Should match the
        training batch size for an apples-to-apples comparison.
    single_sample_iters / single_sample_warmup : int
        Iteration counts for the per-sample latency measurement.
    take_iters / take_warmup : int
        Iteration counts for the per-take latency measurement.

    Returns
    -------
    dict with sub-dicts: ``per_sample`` (single-sample fwd), ``per_batch``
    (single batch fwd), ``per_take`` (one full take), and a summary
    ``throughput_samples_per_sec``.
    """
    model = model.to(device)
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    take_tensors = [t.to(device) for t in take_tensors]
    if not take_tensors:
        raise ValueError("benchmark_inference requires at least one take tensor")
    sample_shape = take_tensors[0].shape[1:]

    # Per-sample (batch of 1).
    one_sample = take_tensors[0][:1]
    sample_times = _time_forward(
        model, one_sample, device, single_sample_warmup, single_sample_iters
    )

    # Per-batch.
    if take_tensors[0].shape[0] >= batch_size:
        one_batch = take_tensors[0][:batch_size]
    else:
        # Tile the take to reach batch_size.
        reps = (batch_size + take_tensors[0].shape[0] - 1) // take_tensors[0].shape[0]
        one_batch = take_tensors[0].repeat(reps, 1, 1, 1)[:batch_size]
    batch_times = _time_forward(
        model, one_batch, device, single_sample_warmup, single_sample_iters
    )

    # Per-take (full take, batched internally at ``batch_size``).
    per_take_times: list[float] = []
    per_take_samples: list[int] = []
    for take in take_tensors:
        N = take.shape[0]
        # Warmup
        with torch.no_grad():
            for _ in range(take_warmup):
                for s in range(0, N, batch_size):
                    _ = model(take[s : s + batch_size])
        with torch.no_grad():
            for _ in range(take_iters):
                with cuda_sync(device):
                    t0 = time.perf_counter()
                for s in range(0, N, batch_size):
                    _ = model(take[s : s + batch_size])
                with cuda_sync(device):
                    t1 = time.perf_counter()
                per_take_times.append(t1 - t0)
                per_take_samples.append(N)

    per_take_arr = np.asarray(per_take_times)
    per_take_samp_arr = np.asarray(per_take_samples)

    peak_inference_memory_mb = (
        float(torch.cuda.max_memory_allocated() / (1024 ** 2))
        if torch.cuda.is_available()
        else 0.0
    )

    def _stats(times: list[float]) -> dict:
        a = np.asarray(times)
        return {
            "mean_seconds": float(a.mean()),
            "median_seconds": float(np.median(a)),
            "std_seconds": float(a.std()),
            "min_seconds": float(a.min()),
            "max_seconds": float(a.max()),
            "n_iters": int(a.size),
        }

    return {
        "input_shape": [int(x) for x in sample_shape],
        "device": str(device),
        "batch_size": int(batch_size),
        "per_sample": _stats(sample_times),
        "per_batch": _stats(batch_times),
        "per_take": {
            **_stats(per_take_times),
            "samples_per_take_mean": float(per_take_samp_arr.mean()),
            "samples_per_take_min": int(per_take_samp_arr.min()),
            "samples_per_take_max": int(per_take_samp_arr.max()),
            "throughput_samples_per_sec_mean": float(
                (per_take_samp_arr / per_take_arr).mean()
            ),
        },
        "throughput_samples_per_sec_batched": float(
            one_batch.shape[0] / np.median(batch_times)
        ),
        "peak_inference_memory_mb": peak_inference_memory_mb,
    }


def write_metrics(path: Path, metrics: dict) -> None:
    """Pretty-print metrics to disk as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True, default=float)


def train_model_benchmarked(
    model: torch.nn.Module,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    test_loader,
    epochs: int,
    device,
    output_dir: Path,
    log_file: Path,
    print_freq: int,
    timer: TrainingTimer,
    scheduler=None,
):
    """Same semantics as ``train.train_model`` but with per-epoch timing.

    Mirrors the existing loop verbatim (running-mean loss, same logging
    format, same train/val/test sequence) so ablation runs are
    apples-to-apples with the baseline pipeline.
    """
    train_loss_history = []
    val_loss_history = []

    with timer.run(device):
        for epoch in range(epochs):
            with timer.epoch(device):
                header = f"Epoch {epoch + 1}/{epochs} {'-' * 20}"
                print(header)
                with open(log_file, "a") as writer:
                    writer.write(header + "\n")

                # --- Train. ----------------------------------------------
                model.train()
                running_loss_avg = 0.0
                running_count = 0
                for i, (sample_a, sample_b) in enumerate(train_loader):
                    optimizer.zero_grad()
                    sample_a = sample_a.to(device)
                    sample_b = sample_b.to(device)
                    za = model(sample_a)
                    zb = model(sample_b)
                    loss = criterion(za, zb)
                    n = running_count
                    m = n + sample_a.shape[0]
                    running_loss_avg = ((n * running_loss_avg) + loss.item()) / m
                    running_count = m
                    if (i + 1) % print_freq == 0:
                        msg = (
                            f"Training | Epoch: {epoch + 1}/{epochs} | "
                            f"Step: {i + 1}/{len(train_loader)} | "
                            f"Loss: {running_loss_avg}"
                        )
                        print(msg)
                        with open(log_file, "a") as writer:
                            writer.write(msg + "\n")
                        train_loss_history.append(
                            (epoch * len(train_loader) + (i + 1), running_loss_avg)
                        )
                    loss.backward()
                    optimizer.step()

                # --- Val. ------------------------------------------------
                model.eval()
                val_running_loss_avg = 0.0
                val_running_count = 0
                with torch.no_grad():
                    for sample_a, sample_b in val_loader:
                        sample_a = sample_a.to(device)
                        sample_b = sample_b.to(device)
                        za = model(sample_a)
                        zb = model(sample_b)
                        loss = criterion(za, zb)
                        n = val_running_count
                        m = n + sample_a.shape[0]
                        val_running_loss_avg = (
                            (n * val_running_loss_avg) + loss.item()
                        ) / m
                        val_running_count = m
                msg = (
                    f"Validation | Epoch: {epoch + 1} | "
                    f"Loss: {val_running_loss_avg}"
                )
                print(msg)
                with open(log_file, "a") as writer:
                    writer.write(msg + "\n")
                val_loss_history.append(
                    ((epoch + 1) * len(train_loader), val_running_loss_avg)
                )

                if scheduler is not None:
                    scheduler.step()
                    cur_lr = optimizer.param_groups[0]["lr"]
                    msg = f"LR after epoch {epoch + 1}: {cur_lr:.3e}"
                    print(msg)
                    with open(log_file, "a") as writer:
                        writer.write(msg + "\n")

    # --- Test. -----------------------------------------------------------
    print("-" * 20)
    with open(log_file, "a") as writer:
        writer.write(("-" * 20) + "\n")
    model.eval()
    test_running_loss_avg = 0.0
    test_running_count = 0
    with torch.no_grad():
        for sample_a, sample_b in test_loader:
            sample_a = sample_a.to(device)
            sample_b = sample_b.to(device)
            za = model(sample_a)
            zb = model(sample_b)
            loss = criterion(za, zb)
            n = test_running_count
            m = n + sample_a.shape[0]
            test_running_loss_avg = (
                (n * test_running_loss_avg) + loss.item()
            ) / m
            test_running_count = m
    msg = f"Testing | Loss: {test_running_loss_avg}"
    print(msg)
    with open(log_file, "a") as writer:
        writer.write(msg + "\n")

    return model, train_loss_history, val_loss_history, test_running_loss_avg
