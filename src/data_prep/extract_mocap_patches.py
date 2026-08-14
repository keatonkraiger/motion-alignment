"""Python port of updated_eval_code/extractMocapPatches_tmm.m.

Reads a performance's mocap marker sequence (frames x 39 markers x
[x, y, z, valid]), pulls 3-second windows out of it while converting
50fps -> 25fps, and normalizes each window (pelvis-center translation,
Z-axis rotation to face +X, per-axis std normalization). Emits two
paired sets of windows: "A" is the un-augmented window, "B" is the same
window with a random time shift/scale applied (data augmentation).

Faithfully mirrors the MATLAB implementation, including its quirks:
  - the pelvis-center/rotation reference frame is `halfn - 1` (0-based),
    i.e. one frame *before* the true temporal center of the window;
  - std normalization is computed over the whole window (all frames,
    all markers) per axis, after zeroing removed markers but before
    dividing;
  - `round()` uses MATLAB's round-half-away-from-zero, not numpy's
    round-half-to-even.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

MARKER_NAMES = [
    "LFHD", "RFHD", "LBHD", "RBHD",
    "C7", "T10", "CLAV", "STRN",
    "RBAK", "LSHO", "LUPA", "LELB",
    "LFRM", "LWRA", "LWRB", "LFIN",
    "RSHO", "RUPA", "RELB", "RFRM",
    "RWRA", "RWRB", "RFIN", "LASI",
    "RASI", "LPSI", "RPSI", "LTHI",
    "LKNE", "LTIB", "LANK", "LHEE",
    "LTOE", "RTHI", "RKNE", "RTIB",
    "RANK", "RHEE", "RTOE",
]

OLD_FPS = 50  # TMM release marker data is 50 frames per second
NEW_FPS = 25
TWINDOW_SECONDS = 3.0


def marker_names_xyz(names: Sequence[str] = MARKER_NAMES) -> list[str]:
    out = []
    for n in names:
        out += [f"{n}X", f"{n}Y", f"{n}Z"]
    return out


def matlab_round(x):
    """MATLAB round(): half away from zero (numpy rounds half to even)."""
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def fill_missing_nearest(modata: np.ndarray) -> np.ndarray:
    """Replace exact-zero entries in each column with the value from the
    nearest (by frame index) non-zero entry in the same column. Ties
    resolve to the earlier frame, matching MATLAB's min() first-match
    behavior on the sorted list of observed indices."""
    out = modata.copy()
    n = out.shape[0]
    for c in range(out.shape[1]):
        col = out[:, c]
        obs_idx = np.flatnonzero(col != 0)
        if obs_idx.size == 0 or obs_idx.size == n:
            continue
        zero_idx = np.flatnonzero(col == 0)
        pos = np.searchsorted(obs_idx, zero_idx)
        m = obs_idx.size
        has_left = pos - 1 >= 0
        has_right = pos <= m - 1
        left_val = obs_idx[np.clip(pos - 1, 0, m - 1)]
        right_val = obs_idx[np.clip(pos, 0, m - 1)]
        left_dist = np.where(has_left, np.abs(zero_idx - left_val), np.inf)
        right_dist = np.where(has_right, np.abs(zero_idx - right_val), np.inf)
        chosen = np.where(left_dist <= right_dist, left_val, right_val)
        col[zero_idx] = col[chosen]
    return out


def resampindex(weights: np.ndarray, n: Optional[int] = None,
                 rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Port of resampindex.m: one random draw seeds a deterministic
    systematic sample of `n` indices from the (normalized) `weights`
    distribution. Returns 1-based frame numbers (matching the MATLAB
    convention `indices(j) = i - 1`, which is already usable directly
    as a 1-based index)."""
    weights = np.maximum(0.0, np.asarray(weights, dtype=np.float64))
    weights = weights / weights.sum()
    if n is None:
        n = weights.size
    if rng is None:
        rng = np.random.default_rng()
    cumprob = np.concatenate(([0.0], np.cumsum(weights)))
    u1 = rng.random() / n
    uj = u1 + np.arange(n) / n
    return np.searchsorted(cumprob, uj, side="left")


@dataclass
class PatchOptions:
    samplerateseconds: float = 0.5
    specificframes: Optional[np.ndarray] = None  # 1-based frame numbers
    removelowaccelerations: bool = True
    padborders: bool = True
    markerstoremove: Sequence[str] = field(default_factory=list)
    seed: Optional[int] = None


def _low_accel_mask(modata: np.ndarray) -> np.ndarray:
    from scipy.ndimage import correlate1d

    kernel = np.array([1, 0, 0, 0, -2, 0, 0, 0, 1], dtype=np.float64)
    moaccel = correlate1d(modata.astype(np.float64), kernel, axis=0, mode="nearest")
    moaccelmed = np.median(np.abs(moaccel), axis=1)
    sortaccel = np.sort(moaccelmed)
    thresh_idx = max(0, int(matlab_round(0.1 * sortaccel.size)) - 1)
    threshold = sortaccel[thresh_idx]
    return (moaccelmed >= threshold).astype(np.float64)


def _patch_frame_indices(t0, tshift, tscale, tmesh) -> np.ndarray:
    """1-based MATLAB frame numbers for a patch window, converted to
    0-based indices."""
    idx_1based = matlab_round((t0 + tshift) * OLD_FPS + (tscale * OLD_FPS / NEW_FPS) * tmesh)
    return (idx_1based - 1).astype(np.int64)


def _normalize_patch(patchdata: np.ndarray, name_to_idx: dict, halfn: int,
                       markers_to_remove: Sequence[str], monamesxyz: Sequence[str]) -> np.ndarray:
    patchdata = patchdata.astype(np.float64)
    ref_row = halfn - 1  # halfn is MATLAB's 1-based row index -> 0-based

    def get_xyz(name):
        m = name_to_idx[name]
        return patchdata[ref_row, m * 3:m * 3 + 3]

    LASI, LPSI = get_xyz("LASI"), get_xyz("LPSI")
    RASI, RPSI = get_xyz("RASI"), get_xyz("RPSI")
    center = (LASI + LPSI + RASI + RPSI) / 4.0

    xyz = patchdata.reshape(patchdata.shape[0], -1, 3) - center

    vec = (RASI - LASI).copy()
    vec[2] = 0.0
    xvec = vec / np.linalg.norm(vec)
    zvec = np.array([0.0, 0.0, 1.0])
    yvec = np.cross(zvec, xvec)
    Rt = np.stack([xvec, yvec, zvec], axis=0)

    xyz = xyz @ Rt.T
    patchdata = xyz.reshape(patchdata.shape[0], -1)

    if markers_to_remove:
        remove_cols = [i for i, name in enumerate(monamesxyz)
                        if any(tok in name for tok in markers_to_remove)]
        patchdata[:, remove_cols] = 0.0

    stdx = patchdata[:, 0::3].std(ddof=1)
    stdy = patchdata[:, 1::3].std(ddof=1)
    stdz = patchdata[:, 2::3].std(ddof=1)
    patchdata[:, 0::3] /= max(0.001, stdx)
    patchdata[:, 1::3] /= max(0.001, stdy)
    patchdata[:, 2::3] /= max(0.001, stdz)

    return patchdata.astype(np.float32)


def extract_mocap_patches(mcmarkers: np.ndarray, options: Optional[PatchOptions] = None):
    """mcmarkers: (numframes, nummarkers, 4) array of [x, y, z, valid].

    Returns a dict with keys 'A', 'B' (nframes, 3*nmarkers, 1, npatches
    float32), 't0' (npatches,) and 'marker_names_xyz'.
    """
    if options is None:
        options = PatchOptions()
    rng = np.random.default_rng(options.seed)

    mocap = np.asarray(mcmarkers, dtype=np.float32)
    modata = mocap[:, :, :3].reshape(mocap.shape[0], -1)  # marker-major, xyz-minor
    modata = fill_missing_nearest(modata)
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

    name_to_idx = {n: i for i, n in enumerate(MARKER_NAMES)}
    monamesxyz = marker_names_xyz(MARKER_NAMES)

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

        patch_a = _normalize_patch(modata[idx_a], name_to_idx, halfn, options.markerstoremove, monamesxyz)
        patch_b = _normalize_patch(modata[idx_b], name_to_idx, halfn, options.markerstoremove, monamesxyz)
        patches_a.append(patch_a)
        patches_b.append(patch_b)
        used_t0.append(t0)

    A = np.stack(patches_a, axis=-1)[:, :, None, :].astype(np.float32)
    B = np.stack(patches_b, axis=-1)[:, :, None, :].astype(np.float32)
    t0_arr = np.asarray(used_t0, dtype=np.float64)
    return {"A": A, "B": B, "t0": t0_arr, "marker_names_xyz": monamesxyz}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mocap_npy", help="path to a MOCAP_MRK_*.npy file (T, 39, 4)")
    parser.add_argument("output_npz", help="output .npz path")
    parser.add_argument("--samplerateseconds", type=float, default=0.5)
    parser.add_argument("--removelowaccelerations", action="store_true")
    parser.add_argument("--no-padborders", dest="padborders", action="store_false")
    parser.add_argument("--markerstoremove", nargs="*", default=["RBAK"])
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    mcmarkers = np.load(args.mocap_npy)
    opts = PatchOptions(
        samplerateseconds=args.samplerateseconds,
        removelowaccelerations=args.removelowaccelerations,
        padborders=args.padborders,
        markerstoremove=args.markerstoremove,
        seed=args.seed,
    )
    result = extract_mocap_patches(mcmarkers, opts)
    np.savez(args.output_npz, A=result["A"], B=result["B"], t0=result["t0"],
              marker_names_xyz=np.array(result["marker_names_xyz"]))
    print(f"Wrote {args.output_npz}: A{result['A'].shape} B{result['B'].shape} t0{result['t0'].shape}")
