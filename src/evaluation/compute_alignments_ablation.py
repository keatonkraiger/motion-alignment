"""Encoder-agnostic version of ``compute_alignments.py`` with inference
benchmarking.

Same alignment / DTW / cost-matrix output as the original
``compute_alignments.py`` (file format unchanged so
``evaluate_keypose_transfer.py`` works as-is) plus:

* End-to-end alignment wall-clock time.
* Total embedding (forward-pass only) wall-clock time across all valid takes.
* Per-take embedding latency.
* Detailed inference-latency benchmark via
  ``benchmark.benchmark_inference`` (per-sample, per-batch, per-take,
  throughput, peak GPU mem).

Use with ``--encoder cnn`` to get baseline numbers, ``--encoder
poseformer`` for the ablation.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat, savemat

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import Encoder  # noqa: E402
from model_poseformer import PoseFormerEncoder  # noqa: E402
from model_mixste import MixSTEEncoder  # noqa: E402
from model_poseformerv2 import PoseFormerV2Encoder  # noqa: E402

from dtw_from_distmat import dtw_from_distmat  # noqa: E402
from valid_performances import load_all_performances, valid_mask  # noqa: E402

from benchmark import (  # noqa: E402
    benchmark_inference,
    count_flops,
    count_parameters,
    cuda_sync,
    state_dict_bytes,
    write_metrics,
)


def _load_patches(patch_dir: Path, subj: int, take: int):
    data = loadmat(str(patch_dir / f"patches_{subj}_{take}.mat"))
    a = data["A"].transpose((3, 2, 0, 1))  # (N, 1, S, F)
    t0 = np.asarray(data["t0"]).squeeze()  # (N,)
    return a.astype(np.float32), t0.astype(np.float64)


def _build_encoder(name: str) -> torch.nn.Module:
    if name == "cnn":
        return Encoder()
    if name == "poseformer":
        return PoseFormerEncoder()
    if name == "mixste":
        return MixSTEEncoder()
    if name == "poseformerv2":
        return PoseFormerV2Encoder()
    raise ValueError(f"Unknown encoder: {name!r}")


def _embed(model: torch.nn.Module, a_np: np.ndarray, device: str) -> np.ndarray:
    a = torch.from_numpy(a_np).to(device)
    with torch.no_grad():
        z = model(a)
        z = F.normalize(z, p=2, dim=1)
    return z.cpu().numpy().astype(np.float64)


def _cosine_distmat(atlas: np.ndarray, query: np.ndarray) -> np.ndarray:
    return 1.0 - atlas @ query.T


def _load_model(model_path: Path, encoder_name: str, device: str) -> torch.nn.Module:
    """Try state_dict first, fall back to pickled module."""
    state_dict_path = model_path.with_name("model_state_dict.pt")
    if state_dict_path.exists():
        print(f"Loading state dict: {state_dict_path}")
        model = _build_encoder(encoder_name)
        model.load_state_dict(torch.load(state_dict_path, map_location=device))
    else:
        print(f"Loading pickled model: {model_path}")
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
    metrics_path: Path | None = None,
    benchmark_take_count: int = 5,
    atlas_subj: int = 7,
    atlas_take: int = 2,
    device: str | None = None,
) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | encoder={encoder_name}")

    model = _load_model(model_path, encoder_name, device)
    param_counts = count_parameters(model)
    flops_info = count_flops(model, (1, 1, 75, 117), device=device)
    print(
        f"Encoder={encoder_name} | params total={param_counts['total']:,} "
        f"trainable={param_counts['trainable']:,} | "
        f"flops/sample={flops_info.get('flops')}"
    )

    # Embed atlas (timed for benchmarking).
    print(f"Embedding atlas: subject {atlas_subj}, take {atlas_take}")
    atlas_a, atlas_t0 = _load_patches(patch_dir, atlas_subj, atlas_take)
    with cuda_sync(device):
        t0 = time.perf_counter()
    atlas_vecs = _embed(model, atlas_a, device)
    with cuda_sync(device):
        atlas_embed_seconds = time.perf_counter() - t0

    subjectdata = load_all_performances(tmm100_path)
    keep = valid_mask(subjectdata)
    print(
        f"Total performances: {subjectdata.shape[0]}, "
        f"valid (kept): {keep.sum()}, partials skipped: {(~keep).sum()}"
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
            print(f"  [skip partial] subj {subj} take {take}")
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
            print(f"  [missing patches] subj {subj} take {take} — skipping")
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

        print(
            f"  subj {subj} take {take}: N={patches_a.shape[0]} "
            f"embed={per_take_embed_seconds[-1] * 1000:.1f} ms "
            f"dtw={per_take_dtw_seconds[-1] * 1000:.1f} ms "
            f"path_len={path.shape[0]} avg_cost={costmatavg[sstind]:.4f}"
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
    print(
        f"Saved alignments to {output_path}\n"
        f"  median avg cost (valid): {np.median(valid_costs):.4f}\n"
        f"  mean   avg cost (valid): {np.mean(valid_costs):.4f}\n"
        f"  total alignment wall-clock: {overall_seconds:.1f}s"
    )

    # --- Structured inference benchmark on a few cached takes. -----------
    if benchmark_take_tensors:
        print(
            "Running structured inference benchmark on "
            f"{len(benchmark_take_tensors)} take(s)..."
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
        print(f"Saved alignment benchmarks: {metrics_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--patch-dir", type=Path, required=True)
    parser.add_argument(
        "--encoder",
        choices=["cnn", "poseformer", "mixste", "poseformerv2"],
        default="cnn",
    )
    parser.add_argument(
        "--tmm100", type=Path,
        default=Path("../../data/eval/tmm100performances.mat"),
    )
    parser.add_argument("--output", type=Path, default=Path("../outputs/MyMocapAlignments.mat"))
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--atlas-subject", type=int, default=7)
    parser.add_argument("--atlas-take", type=int, default=2)
    parser.add_argument("--benchmark-take-count", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    compute_alignments(
        model_path=args.model,
        patch_dir=args.patch_dir,
        tmm100_path=args.tmm100,
        output_path=args.output,
        encoder_name=args.encoder,
        metrics_path=args.metrics,
        benchmark_take_count=args.benchmark_take_count,
        atlas_subj=args.atlas_subject,
        atlas_take=args.atlas_take,
        device=args.device,
    )
