"""Batch-generate the full patches_<subj>_<take>.mat dataset from the
PSU100 SMPL-X MoSh++ release, using this repo's own SMPL-X body-joint
extraction (extract_smplx_patches.py / smplx_kinematics.py).

Mirrors generate_patch_dataset.py's marker-based pipeline exactly (same
window/seed conventions, same TrainPatches subset via datasets.ALL_TAKES,
same output .mat schema) so the two representations are drop-in
interchangeable for train.py -- only the config's data paths and the
model's num_markers/num_joints need to change.

Scans <raw-root>/Subject<N>/Subject<N>_MOCAP_MRK_<take>_gt_stageii.npz for
every subject to build the (subject, take) performance list. Skips the
`_every_2_frame` decimated variants and the `*_stagei.npz` shape-fit files.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_smplx_patches import (
    JointPatchOptions,
    extract_smplx_patches,
    extract_smplx_patches_shape_augmented,
)
from smplx_kinematics import compute_skeleton_scale, load_body_model, load_take_joints
from datasets import ALL_TAKES

TAKE_RE = re.compile(r"^Subject(\d+)_MOCAP_MRK_(\d+)_gt_stageii\.npz$")


def discover_performances(raw_root: Path) -> list[tuple[int, int]]:
    perfs = []
    for subj_dir in sorted(raw_root.glob("Subject*")):
        if not re.match(r"^Subject\d+$", subj_dir.name):
            continue
        for f in subj_dir.iterdir():
            m = TAKE_RE.match(f.name)
            if m:
                perfs.append((int(m.group(1)), int(m.group(2))))
    return sorted(perfs)


def _get_body_model(body_models_dir: Path, npz_path: Path, device: str, body_model_cache: dict):
    npz_head = np.load(npz_path, allow_pickle=True)
    cache_key = (str(npz_head["surface_model_type"]), str(npz_head["gender"]))
    if cache_key not in body_model_cache:
        body_model_cache[cache_key] = load_body_model(
            body_models_dir, cache_key[0], cache_key[1], device=device
        )
    return body_model_cache[cache_key], npz_head


def _make_opts(subject: int, take: int, scale_norm: str, body_model, real_betas: np.ndarray) -> JointPatchOptions:
    skeleton_scale = compute_skeleton_scale(body_model, real_betas) if scale_norm == "skeleton" else None
    return JointPatchOptions(
        samplerateseconds=0.5,
        removelowaccelerations=False,
        padborders=True,
        seed=subject * 1000 + take,
        scale_norm=scale_norm,
        skeleton_scale=skeleton_scale,
    )


def generate_eval_patches(raw_root: Path, body_models_dir: Path, subject: int, take: int, device: str,
                           body_model_cache: dict, scale_norm: str = "window_std"):
    """Real betas only -- this is what phase-2 DTW/keypose evaluation and
    the atlas performance are scored against, so it must stay tied to the
    real performers regardless of any training-side augmentation. Must use
    the same `scale_norm` as the paired generate_train_patches call, since
    that's a representation choice (not an augmentation) and train/eval
    have to agree on it."""
    npz_path = raw_root / f"Subject{subject}" / f"Subject{subject}_MOCAP_MRK_{take}_gt_stageii.npz"
    body_model, npz_head = _get_body_model(body_models_dir, npz_path, device, body_model_cache)
    real_betas = np.asarray(npz_head["betas"])[:16]
    joints, fps = load_take_joints(npz_path, body_models_dir, device=device, body_model=body_model)
    opts = _make_opts(subject, take, scale_norm, body_model, real_betas)
    return extract_smplx_patches(joints, fps, opts)


def generate_train_patches(raw_root: Path, body_models_dir: Path, subject: int, take: int, device: str,
                            body_model_cache: dict, beta_augment_k: int, beta_augment_std: float,
                            beta_augment_pair: bool, scale_norm: str = "window_std"):
    """Real betas, plus `beta_augment_k` re-posings of the same motion on
    jittered body shapes (betas + N(0, beta_augment_std) per component) --
    increases training-set size ~(k+1)x and diversifies the body shapes
    the encoder is trained against, without touching the eval set. With
    beta_augment_k=0 this is byte-for-byte generate_eval_patches's output.
    """
    npz_path = raw_root / f"Subject{subject}" / f"Subject{subject}_MOCAP_MRK_{take}_gt_stageii.npz"
    body_model, npz_head = _get_body_model(body_models_dir, npz_path, device, body_model_cache)
    real_betas = np.asarray(npz_head["betas"])[:16]
    opts = _make_opts(subject, take, scale_norm, body_model, real_betas)

    if beta_augment_k <= 0:
        joints, fps = load_take_joints(npz_path, body_models_dir, device=device, body_model=body_model)
        return extract_smplx_patches(joints, fps, opts)

    joints_real, fps = load_take_joints(npz_path, body_models_dir, device=device, body_model=body_model)
    variants = [joints_real]
    rng_beta = np.random.default_rng(subject * 100_000 + take * 100)
    for _ in range(beta_augment_k):
        synthetic_betas = real_betas + rng_beta.normal(0.0, beta_augment_std, size=real_betas.shape)
        joints_v, _ = load_take_joints(
            npz_path, body_models_dir, device=device, body_model=body_model,
            betas_override=synthetic_betas,
        )
        variants.append(joints_v)
    return extract_smplx_patches_shape_augmented(variants, fps, opts, pair_across_shapes=beta_augment_pair)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path, default=Path("/mnt/d/Data/PSU100/SMPLX"))
    ap.add_argument("--body-models-dir", type=Path, default=Path("/mnt/e/Research/SMPL/body_models"))
    ap.add_argument("--patches-out", type=Path, required=True,
                     help="output dir for the full performance set (PatchesSMPLX/)")
    ap.add_argument("--trainpatches-out", type=Path, required=True,
                     help="output dir for the training subset (TrainPatchesSMPLX/)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--beta-augment-k", type=int, default=0,
                     help="number of extra jittered-body-shape re-posings per training take "
                          "(0 = disabled, matches the un-augmented pipeline exactly). Only "
                          "applied to the TrainPatchesSMPLX subset, never to eval Patches.")
    ap.add_argument("--beta-augment-std", type=float, default=1.0,
                     help="std of the per-component Gaussian jitter added to each training "
                          "take's real betas to synthesize a shape variant")
    ap.add_argument("--beta-augment-pair", action="store_true",
                     help="let a single A/B patch pair draw from two different shape variants "
                          "(directly trains shape-invariant matching), instead of every pair "
                          "sharing one shape")
    ap.add_argument("--scale-norm", choices=["window_std", "skeleton"], default="window_std",
                     help="'window_std' (default): per-axis std of each 75-frame window, "
                          "matching the marker pipeline. 'skeleton': fixed per-take isotropic "
                          "scale from the subject's real rest-pose bone lengths. Applied "
                          "identically to eval Patches and TrainPatches (must match, unlike "
                          "--beta-augment-*, which is train-only).")
    args = ap.parse_args()

    args.patches_out.mkdir(parents=True, exist_ok=True)
    args.trainpatches_out.mkdir(parents=True, exist_ok=True)

    performances = discover_performances(args.raw_root)
    print(f"Discovered {len(performances)} performances across "
          f"{len({s for s, _ in performances})} subjects | device={args.device} | "
          f"scale_norm={args.scale_norm} | "
          f"beta_augment_k={args.beta_augment_k} std={args.beta_augment_std} pair={args.beta_augment_pair}")

    train_set = set(ALL_TAKES)
    body_model_cache: dict = {}

    for i, (subject, take) in enumerate(performances, 1):
        t0 = time.perf_counter()
        is_train = (subject, take) in train_set

        result = generate_eval_patches(args.raw_root, args.body_models_dir, subject, take, args.device,
                                        body_model_cache, scale_norm=args.scale_norm)
        out_path = args.patches_out / f"patches_{subject}_{take}.mat"
        savemat(str(out_path), {"A": result["A"], "B": result["B"], "t0": result["t0"]})

        train_shape = None
        if is_train:
            train_result = generate_train_patches(
                args.raw_root, args.body_models_dir, subject, take, args.device, body_model_cache,
                args.beta_augment_k, args.beta_augment_std, args.beta_augment_pair,
                scale_norm=args.scale_norm,
            )
            savemat(
                str(args.trainpatches_out / f"patches_{subject}_{take}.mat"),
                {"A": train_result["A"], "B": train_result["B"], "t0": train_result["t0"]},
            )
            train_shape = train_result["A"].shape

        dt = time.perf_counter() - t0
        print(f"[{i}/{len(performances)}] subj {subject} take {take}: "
              f"eval A{result['A'].shape} ({dt:.1f}s) -> {out_path.name}"
              + (f"  train A{train_shape} -> TrainPatchesSMPLX" if is_train else ""))

    n_train_written = len(list(args.trainpatches_out.glob("*.mat")))
    print(f"\nDone. Patches: {len(performances)} files in {args.patches_out}")
    print(f"TrainPatchesSMPLX: {n_train_written} files in {args.trainpatches_out} "
          f"(expected {len(train_set)})")


if __name__ == "__main__":
    main()
