"""Phase 2: DTW alignment of every performance against a fixed atlas
performance, using a trained Siamese encoder, plus inference benchmarking.

* End-to-end alignment wall-clock time.
* Total embedding (forward-pass only) wall-clock time across all valid takes.
* Per-take embedding latency.
* Detailed inference-latency benchmark via
  ``benchmark.benchmark_inference`` (per-sample, per-batch, per-take,
  throughput, peak GPU mem).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat, savemat

from models import build_model
from benchmark import (
    benchmark_inference,
    count_flops,
    count_parameters,
    cuda_sync,
    state_dict_bytes,
    write_metrics,
)

from .dtw import dtw_from_distmat
from .valid_performances import load_all_performances, valid_mask

log = logging.getLogger(__name__)


def _load_patches(patch_dir: Path, subj: int, take: int):
    data = loadmat(str(patch_dir / f"patches_{subj}_{take}.mat"))
    a = data["A"].transpose((3, 2, 0, 1))  # (N, 1, S, F)
    t0 = np.asarray(data["t0"]).squeeze()  # (N,)
    return a.astype(np.float32), t0.astype(np.float64)


def _embed(model: torch.nn.Module, a_np: np.ndarray, device: str) -> np.ndarray:
    a = torch.from_numpy(a_np).to(device)
    with torch.no_grad():
        z = model(a)
        z = F.normalize(z, p=2, dim=1)
    return z.cpu().numpy().astype(np.float64)


def _cosine_distmat(atlas: np.ndarray, query: np.ndarray) -> np.ndarray:
    return 1.0 - atlas @ query.T


def _load_model(
    model_path: Path, encoder_name: str, model_params: dict, device: str
) -> torch.nn.Module:
    """Try state_dict first, fall back to pickled module."""
    state_dict_path = model_path.with_name("model_state_dict.pt")
    if state_dict_path.exists():
        log.info("Loading state dict: %s", state_dict_path)
        model = build_model(encoder_name, model_params)
        model.load_state_dict(torch.load(state_dict_path, map_location=device))
    else:
        log.info("Loading pickled model: %s", model_path)
        model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    return model


def compute_alignments(
    model_path: Path,
    patch_dir: Path,
    tmm100_path: Path,
    output_path: Path,
    encoder_name: str,
    model_params: dict | None = None,
    metrics_path: Path | None = None,
    benchmark_take_count: int = 5,
    atlas_subj: int = 7,
    atlas_take: int = 2,
    device: str | None = None,
) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s | encoder=%s", device, encoder_name)

    model = _load_model(model_path, encoder_name, model_params or {}, device)
    param_counts = count_parameters(model)
    flops_info = count_flops(model, (1, 1, 75, 117), device=device)
    log.info(
        "Encoder=%s | params total=%s trainable=%s | flops/sample=%s",
        encoder_name, f"{param_counts['total']:,}", f"{param_counts['trainable']:,}",
        flops_info.get("flops"),
    )

    # Embed atlas (timed for benchmarking).
    log.info("Embedding atlas: subject %s, take %s", atlas_subj, atlas_take)
    atlas_a, atlas_t0 = _load_patches(patch_dir, atlas_subj, atlas_take)
    with cuda_sync(device):
        t0 = time.perf_counter()
    atlas_vecs = _embed(model, atlas_a, device)
    with cuda_sync(device):
        atlas_embed_seconds = time.perf_counter() - t0

    subjectdata = load_all_performances(tmm100_path)
    keep = valid_mask(subjectdata)
    log.info(
        "Total performances: %d, valid (kept): %d, partials skipped: %d",
        subjectdata.shape[0], keep.sum(), (~keep).sum(),
    )

    pathcells = []
    costmat = np.zeros(subjectdata.shape[0])
    costmatavg = np.zeros(subjectdata.shape[0])

    per_take_embed_seconds: list[float] = []
    per_take_dtw_seconds: list[float] = []
    per_take_n_samples: list[int] = []
    benchmark_take_tensors: list[torch.Tensor] = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    overall_t0 = time.perf_counter()

    for sstind in range(subjectdata.shape[0]):
        subj, take = int(subjectdata[sstind, 0]), int(subjectdata[sstind, 1])
        if not keep[sstind]:
            log.info("  [skip partial] subj %d take %d", subj, take)
            pathcells.append(
                [
                    np.zeros((0, 2), dtype=np.int64),
                    np.zeros((0,), dtype=np.float64),
                    np.zeros((0, 0), dtype=np.float64),
                    np.array([subj, take], dtype=np.int64),
                ]
            )
            continue

        try:
            patches_a, t0_list = _load_patches(patch_dir, subj, take)
        except FileNotFoundError:
            log.info("  [missing patches] subj %d take %d - skipping", subj, take)
            pathcells.append(
                [
                    np.zeros((0, 2), dtype=np.int64),
                    np.zeros((0,), dtype=np.float64),
                    np.zeros((0, 0), dtype=np.float64),
                    np.array([subj, take], dtype=np.int64),
                ]
            )
            continue

        # --- Time the encoder forward pass on this take. ---------------
        with cuda_sync(device):
            t_e = time.perf_counter()
        vecs = _embed(model, patches_a, device)
        with cuda_sync(device):
            per_take_embed_seconds.append(time.perf_counter() - t_e)
        per_take_n_samples.append(int(patches_a.shape[0]))

        # --- Time DTW on this take. ------------------------------------
        distmat = _cosine_distmat(atlas_vecs, vecs)
        t_d = time.perf_counter()
        path = dtw_from_distmat(distmat)
        per_take_dtw_seconds.append(time.perf_counter() - t_d)

        idx_rows = path[:, 0] - 1
        idx_cols = path[:, 1] - 1
        total_cost = float(distmat[idx_rows, idx_cols].sum())
        costmat[sstind] = total_cost
        costmatavg[sstind] = total_cost / path.shape[0]

        pathcells.append(
            [
                path.astype(np.int64),
                t0_list.astype(np.float64),
                distmat.astype(np.float64),
                np.array([subj, take], dtype=np.int64),
            ]
        )

        # Cache a few take tensors for the structured inference benchmark below.
        if len(benchmark_take_tensors) < benchmark_take_count:
            benchmark_take_tensors.append(torch.from_numpy(patches_a))

        log.info(
            "  subj %d take %d: N=%d embed=%.1f ms dtw=%.1f ms path_len=%d avg_cost=%.4f",
            subj, take, patches_a.shape[0],
            per_take_embed_seconds[-1] * 1000, per_take_dtw_seconds[-1] * 1000,
            path.shape[0], costmatavg[sstind],
        )

    overall_seconds = time.perf_counter() - overall_t0

    peak_inference_memory_mb = (
        float(torch.cuda.max_memory_allocated() / (1024 ** 2))
        if torch.cuda.is_available()
        else 0.0
    )

    # --- Save alignments .mat (unchanged format). --------------------------
    pathcells_obj = np.empty((len(pathcells),), dtype=object)
    for i, entry in enumerate(pathcells):
        cell = np.empty((1, 4), dtype=object)
        cell[0, 0] = entry[0]
        cell[0, 1] = entry[1].reshape(1, -1)
        cell[0, 2] = entry[2]
        cell[0, 3] = entry[3].reshape(1, -1)
        pathcells_obj[i] = cell

    output_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(
        str(output_path),
        {
            "costmat": costmat.reshape(1, -1),
            "costmatavg": costmatavg.reshape(1, -1),
            "pathcells": pathcells_obj,
            "subjectdata": subjectdata,
            "atlassubsesstake": np.array([atlas_subj, atlas_take], dtype=np.int64),
            "atlast0": atlas_t0.reshape(1, -1),
        },
    )

    valid_costs = costmatavg[keep]
    log.info(
        "Saved alignments to %s\n  median avg cost (valid): %.4f\n"
        "  mean   avg cost (valid): %.4f\n  total alignment wall-clock: %.1fs",
        output_path, np.median(valid_costs), np.mean(valid_costs), overall_seconds,
    )

    # --- Structured inference benchmark on a few cached takes. -----------
    if benchmark_take_tensors:
        log.info(
            "Running structured inference benchmark on %d take(s)...",
            len(benchmark_take_tensors),
        )
        inference_metrics = benchmark_inference(
            model=model,
            take_tensors=benchmark_take_tensors,
            device=device,
        )
    else:
        inference_metrics = {}

    embed_seconds_arr = np.asarray(per_take_embed_seconds)
    n_samples_arr = np.asarray(per_take_n_samples)
    dtw_seconds_arr = np.asarray(per_take_dtw_seconds)
    metrics = {
        "encoder": encoder_name,
        "device": device,
        "params": param_counts,
        "flops": flops_info,
        "model_state_dict_bytes": state_dict_bytes(model),
        "alignment_run": {
            "total_seconds": float(overall_seconds),
            "atlas_embed_seconds": float(atlas_embed_seconds),
            "n_takes_processed": int(len(per_take_embed_seconds)),
            "embed_seconds_total": float(embed_seconds_arr.sum()),
            "embed_seconds_per_take_mean": float(embed_seconds_arr.mean()) if embed_seconds_arr.size else 0.0,
            "embed_seconds_per_take_median": float(np.median(embed_seconds_arr)) if embed_seconds_arr.size else 0.0,
            "embed_seconds_per_take_max": float(embed_seconds_arr.max()) if embed_seconds_arr.size else 0.0,
            "embed_throughput_samples_per_sec": (
                float(n_samples_arr.sum() / embed_seconds_arr.sum())
                if embed_seconds_arr.size and embed_seconds_arr.sum() > 0
                else 0.0
            ),
            "dtw_seconds_per_take_mean": float(dtw_seconds_arr.mean()) if dtw_seconds_arr.size else 0.0,
            "samples_per_take_mean": float(n_samples_arr.mean()) if n_samples_arr.size else 0.0,
            "peak_inference_memory_mb": peak_inference_memory_mb,
        },
        "inference_benchmark": inference_metrics,
        "alignment_quality": {
            "median_avg_cost_valid": float(np.median(valid_costs)) if valid_costs.size else 0.0,
            "mean_avg_cost_valid": float(np.mean(valid_costs)) if valid_costs.size else 0.0,
            "n_valid_performances": int(keep.sum()),
        },
    }

    if metrics_path is not None:
        write_metrics(metrics_path, metrics)
        log.info("Saved alignment benchmarks: %s", metrics_path)
