"""SMPL-X analogue of extract_mocap_patches.py: raw SMPL-X body-joint
sequence -> normalized 3-second patches, using the same windowing,
augmentation, and per-axis normalization recipe as the marker pipeline so
the two representations are directly comparable.

Operates on the 22-joint body-only chain (pelvis + 21 ``body_pose``
joints) computed by smplx_kinematics.py -- the subset shared identically
across SMPL, SMPL-H, and SMPL-X. No hands, no face.

Differences from the marker version, all consequences of the input being
a clean MoSh++ fit rather than raw occluded marker trajectories:
  - no `fill_missing_nearest` gap-filling (SMPL-X joint positions have no
    dropout);
  - the pelvis-center / yaw-alignment reference frame uses the pelvis
    joint and the left/right hip joints directly, in place of the
    LASI/LPSI/RASI/RPSI marker average and RASI-LASI vector.
Window arithmetic, augmentation (tshift/tscale), resampling, and per-axis
std normalization are otherwise identical to extract_mocap_patches.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from extract_mocap_patches import (
    NEW_FPS,
    OLD_FPS,
    TWINDOW_SECONDS,
    _low_accel_mask,
    _patch_frame_indices,
    matlab_round,
    resampindex,
)
from smplx_kinematics import BODY_JOINT_NAMES


def joint_names_xyz(names: Sequence[str] = BODY_JOINT_NAMES) -> list[str]:
    out = []
    for n in names:
        out += [f"{n}X", f"{n}Y", f"{n}Z"]
    return out


@dataclass
class JointPatchOptions:
    samplerateseconds: float = 0.5
    specificframes: Optional[np.ndarray] = None  # 1-based frame numbers
    removelowaccelerations: bool = True
    padborders: bool = True
    seed: Optional[int] = None
    # "window_std" (default): per-axis std of *this window's* 75 frames,
    #   exactly mirroring extract_mocap_patches.py's marker normalization.
    # "skeleton": a single isotropic scale, fixed per take, from the
    #   subject's real rest-pose bone lengths (see
    #   smplx_kinematics.compute_skeleton_scale) -- a lower-variance
    #   alternative that doesn't re-estimate scale from just 75 frames,
    #   only available because SMPL-X gives us the real skeleton for free.
    scale_norm: str = "window_std"
    skeleton_scale: Optional[float] = None  # required when scale_norm == "skeleton"


def _normalize_patch(
    patchdata: np.ndarray, name_to_idx: dict, halfn: int,
    scale_norm: str = "window_std", skeleton_scale: Optional[float] = None,
) -> np.ndarray:
    patchdata = patchdata.astype(np.float64)
    ref_row = halfn - 1  # halfn is MATLAB's 1-based row index -> 0-based

    def get_xyz(name):
        j = name_to_idx[name]
        return patchdata[ref_row, j * 3:j * 3 + 3]

    center = get_xyz("pelvis")
    left_hip, right_hip = get_xyz("left_hip"), get_xyz("right_hip")

    xyz = patchdata.reshape(patchdata.shape[0], -1, 3) - center

    vec = (right_hip - left_hip).copy()
    vec[2] = 0.0
    xvec = vec / np.linalg.norm(vec)
    zvec = np.array([0.0, 0.0, 1.0])
    yvec = np.cross(zvec, xvec)
    Rt = np.stack([xvec, yvec, zvec], axis=0)

    xyz = xyz @ Rt.T
    patchdata = xyz.reshape(patchdata.shape[0], -1)

    if scale_norm == "skeleton":
        if skeleton_scale is None:
            raise ValueError("skeleton_scale is required when scale_norm='skeleton'")
        patchdata /= max(0.001, skeleton_scale)
    elif scale_norm == "window_std":
        stdx = patchdata[:, 0::3].std(ddof=1)
        stdy = patchdata[:, 1::3].std(ddof=1)
        stdz = patchdata[:, 2::3].std(ddof=1)
        patchdata[:, 0::3] /= max(0.001, stdx)
        patchdata[:, 1::3] /= max(0.001, stdy)
        patchdata[:, 2::3] /= max(0.001, stdz)
    else:
        raise ValueError(f"Unknown scale_norm: {scale_norm!r}")

    return patchdata.astype(np.float32)


def extract_smplx_patches(joints: np.ndarray, fps: int, options: Optional[JointPatchOptions] = None):
    """joints: (numframes, 22, 3) body joint positions (pelvis + 21 joints).

    Returns a dict with keys 'A', 'B' (nframes, 3*22, 1, npatches float32),
    't0' (npatches,) and 'joint_names_xyz'.
    """
    if options is None:
        options = JointPatchOptions()
    if fps != OLD_FPS:
        raise ValueError(
            f"extract_smplx_patches expects {OLD_FPS} fps input (matching the "
            f"marker pipeline's window arithmetic), got {fps}"
        )
    rng = np.random.default_rng(options.seed)

    modata = np.asarray(joints, dtype=np.float32).reshape(joints.shape[0], -1)  # joint-major, xyz-minor
    T = modata.shape[0]

    nframes = int(matlab_round(TWINDOW_SECONDS * NEW_FPS))  # 75
    halfn = nframes // 2  # 37
    tmesh = np.arange(-halfn, halfn + 1, dtype=np.float64)

    if options.specificframes is not None and len(options.specificframes) > 0:
        t0list = np.asarray(options.specificframes, dtype=np.float64) / OLD_FPS
    else:
        if options.removelowaccelerations:
            enoughaccel = _low_accel_mask(modata)
        else:
            enoughaccel = np.ones(T, dtype=np.float64)
        nwindows = int(np.floor(T / OLD_FPS / options.samplerateseconds))
        tmpinds = resampindex(enoughaccel, nwindows, rng=rng)
        t0list = tmpinds / OLD_FPS

    name_to_idx = {n: i for i, n in enumerate(BODY_JOINT_NAMES)}

    patches_a, patches_b, used_t0 = [], [], []
    for t0 in t0list:
        idx_a = _patch_frame_indices(t0, 0.0, 1.0, tmesh)
        if options.padborders:
            idx_a = np.clip(idx_a, 0, T - 1)
        elif idx_a.min() < 0 or idx_a.max() > T - 1:
            continue

        tshift = 0.5 * (rng.random() * 2 - 1)
        tscale = 1.0 + (1.0 / 3.0) * (rng.random() * 2 - 1)
        idx_b = _patch_frame_indices(t0, tshift, tscale, tmesh)
        if options.padborders:
            idx_b = np.clip(idx_b, 0, T - 1)
        elif idx_b.min() < 0 or idx_b.max() > T - 1:
            continue

        patch_a = _normalize_patch(modata[idx_a], name_to_idx, halfn,
                                    options.scale_norm, options.skeleton_scale)
        patch_b = _normalize_patch(modata[idx_b], name_to_idx, halfn,
                                    options.scale_norm, options.skeleton_scale)
        patches_a.append(patch_a)
        patches_b.append(patch_b)
        used_t0.append(t0)

    A = np.stack(patches_a, axis=-1)[:, :, None, :].astype(np.float32)
    B = np.stack(patches_b, axis=-1)[:, :, None, :].astype(np.float32)
    t0_arr = np.asarray(used_t0, dtype=np.float64)
    return {"A": A, "B": B, "t0": t0_arr, "joint_names_xyz": joint_names_xyz(BODY_JOINT_NAMES)}


def extract_smplx_patches_shape_augmented(
    joints_variants: Sequence[np.ndarray],
    fps: int,
    options: Optional[JointPatchOptions] = None,
    pair_across_shapes: bool = False,
) -> dict:
    """Shape-augmented version of extract_smplx_patches: emits one patch
    pair per (window, shape variant) instead of one per window.

    ``joints_variants`` must all be re-posings of the *same* underlying
    performance (identical T / timing) -- ``joints_variants[0]`` is
    expected to be the take's real betas, and ``joints_variants[1:]``
    jittered-beta re-posings of the same root_orient/pose_body/trans
    (see smplx_kinematics.load_take_joints's betas_override) -- so window
    selection (which depends on acceleration, computed off variant 0
    only) and frame indices carry over identically across variants.

    Window/temporal-augmentation sampling is otherwise identical to
    extract_smplx_patches, just repeated once per variant with its own
    fresh tshift/tscale draw. With a single variant this reduces to
    exactly extract_smplx_patches's output (up to RNG draw order).

    If ``pair_across_shapes``, A and B of a single pair are independently
    drawn from the variant list (so a pair may combine two different body
    shapes performing the same motion) -- this trains the embedding to
    match across shape changes directly, not just across temporal
    augmentation. If False (default), A and B of a given repetition
    always share one shape, and it's the *set* of patches across the
    training run that gains shape diversity.
    """
    if options is None:
        options = JointPatchOptions()
    if fps != OLD_FPS:
        raise ValueError(
            f"extract_smplx_patches_shape_augmented expects {OLD_FPS} fps input, got {fps}"
        )
    n_variants = len(joints_variants)
    T = joints_variants[0].shape[0]
    modata_variants = []
    for j in joints_variants:
        if j.shape[0] != T:
            raise ValueError("all shape variants must share the same frame count/timing")
        modata_variants.append(np.asarray(j, dtype=np.float32).reshape(T, -1))

    rng = np.random.default_rng(options.seed)

    nframes = int(matlab_round(TWINDOW_SECONDS * NEW_FPS))  # 75
    halfn = nframes // 2  # 37
    tmesh = np.arange(-halfn, halfn + 1, dtype=np.float64)

    if options.specificframes is not None and len(options.specificframes) > 0:
        t0list = np.asarray(options.specificframes, dtype=np.float64) / OLD_FPS
    else:
        if options.removelowaccelerations:
            enoughaccel = _low_accel_mask(modata_variants[0])
        else:
            enoughaccel = np.ones(T, dtype=np.float64)
        nwindows = int(np.floor(T / OLD_FPS / options.samplerateseconds))
        tmpinds = resampindex(enoughaccel, nwindows, rng=rng)
        t0list = tmpinds / OLD_FPS

    name_to_idx = {n: i for i, n in enumerate(BODY_JOINT_NAMES)}

    patches_a, patches_b, used_t0 = [], [], []
    for t0 in t0list:
        idx_a = _patch_frame_indices(t0, 0.0, 1.0, tmesh)
        if options.padborders:
            idx_a = np.clip(idx_a, 0, T - 1)
        elif idx_a.min() < 0 or idx_a.max() > T - 1:
            continue

        for rep in range(n_variants):
            tshift = 0.5 * (rng.random() * 2 - 1)
            tscale = 1.0 + (1.0 / 3.0) * (rng.random() * 2 - 1)
            idx_b = _patch_frame_indices(t0, tshift, tscale, tmesh)
            if options.padborders:
                idx_b = np.clip(idx_b, 0, T - 1)
            elif idx_b.min() < 0 or idx_b.max() > T - 1:
                continue

            if pair_across_shapes:
                va = rng.integers(0, n_variants)
                vb = rng.integers(0, n_variants)
            else:
                va = vb = rep

            patch_a = _normalize_patch(modata_variants[va][idx_a], name_to_idx, halfn,
                                        options.scale_norm, options.skeleton_scale)
            patch_b = _normalize_patch(modata_variants[vb][idx_b], name_to_idx, halfn,
                                        options.scale_norm, options.skeleton_scale)
            patches_a.append(patch_a)
            patches_b.append(patch_b)
            used_t0.append(t0)

    A = np.stack(patches_a, axis=-1)[:, :, None, :].astype(np.float32)
    B = np.stack(patches_b, axis=-1)[:, :, None, :].astype(np.float32)
    t0_arr = np.asarray(used_t0, dtype=np.float64)
    return {"A": A, "B": B, "t0": t0_arr, "joint_names_xyz": joint_names_xyz(BODY_JOINT_NAMES)}


if __name__ == "__main__":
    import argparse

    import torch

    from smplx_kinematics import load_take_joints

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smplx_npz", help="path to a Subject<N>_MOCAP_MRK_<take>_gt_stageii.npz file")
    parser.add_argument("output_npz", help="output .npz path")
    parser.add_argument("--body-models-dir", type=Path, required=True)
    parser.add_argument("--samplerateseconds", type=float, default=0.5)
    parser.add_argument("--removelowaccelerations", action="store_true")
    parser.add_argument("--no-padborders", dest="padborders", action="store_false")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    joints, fps = load_take_joints(args.smplx_npz, args.body_models_dir, device=args.device)
    opts = JointPatchOptions(
        samplerateseconds=args.samplerateseconds,
        removelowaccelerations=args.removelowaccelerations,
        padborders=args.padborders,
        seed=args.seed,
    )
    result = extract_smplx_patches(joints, fps, opts)
    np.savez(args.output_npz, A=result["A"], B=result["B"], t0=result["t0"],
              joint_names_xyz=np.array(result["joint_names_xyz"]))
    print(f"Wrote {args.output_npz}: A{result['A'].shape} B{result['B'].shape} t0{result['t0'].shape}")
