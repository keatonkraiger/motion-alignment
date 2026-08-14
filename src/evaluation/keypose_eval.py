"""Port of the second half of updated_eval_code/evaluateKeyposeAlignmentMocap.m.

Given a precomputed alignments file (from MATLAB or our Python pipeline),
transfer atlas keypose times into every other performance via the DTW path,
compare to ground truth, and produce the ROC-style "% keyposes within
threshold" curve and a normalized AUC.

The MATLAB script's logic is replicated very closely (including its two
sequential ``unique`` calls for monotone path dedup and its boundary points
for ``interp1``) so the Python output can be checked byte-for-byte against
the published curve.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat

from .valid_performances import partial_mask

log = logging.getLogger(__name__)

FRAMES_PER_SECOND = 50
KEYPOSES_TO_USE = np.arange(1, 42)  # MATLAB 2:42 (1-indexed) -> Python 1:42
THRESHOLDS_FRAMES = np.arange(5, 91, 5)  # MATLAB 5:5:90


def _load_alignments(alignments_path: Path):
    """Load either the MATLAB MyMocapAlignments.mat or a savemat-mirror of
    our Python output.

    Returns
    -------
    pathcells : list of tuples (path, subj_t0, distmat, [subj, take])
    subjectdata : np.ndarray (P, 2)
    atlassubsesstake : np.ndarray (2,)
    atlas_t0 : np.ndarray
    """
    foo = loadmat(str(alignments_path))
    raw = foo["pathcells"]  # cell array of cells
    # MATLAB cell-of-cells -> object ndarray. Normalize to a flat list.
    cells = np.atleast_1d(raw.squeeze())

    pathcells = []
    for entry in cells:
        # Each entry is itself a 1x4 cell: {path, t0, distmat, [subj take]}
        e = np.atleast_1d(entry.squeeze())
        path = np.asarray(e[0], dtype=np.int64)
        t0 = np.asarray(e[1]).squeeze()
        distmat = np.asarray(e[2])
        st = np.asarray(e[3]).squeeze().astype(np.int64)
        pathcells.append((path, t0, distmat, st))

    subjectdata = np.asarray(foo["subjectdata"], dtype=np.int64)
    if subjectdata.shape[1] != 2 and subjectdata.shape[0] == 2:
        subjectdata = subjectdata.T
    atlassubsesstake = np.asarray(foo["atlassubsesstake"]).squeeze().astype(np.int64)
    atlas_t0 = np.asarray(foo["atlast0"]).squeeze()
    return pathcells, subjectdata, atlassubsesstake, atlas_t0


def _monotone_unique_dedup(path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replicate the MATLAB `unique` dedup pattern exactly.

    MATLAB:
        [frominds, index] = unique(subjpath(:,1));
        toinds = subjpath(index, 2);
        [toinds, index] = unique(toinds);
        frominds = frominds(index);

    `unique` in MATLAB returns sorted unique values by default and the
    index of the *last* occurrence is NOT what's used: MATLAB's default
    returns the index of the first occurrence in the sorted output. For
    our integer paths this is equivalent to ``np.unique(..., return_index=True)``.
    """
    col1 = path[:, 0]
    frominds, idx = np.unique(col1, return_index=True)
    toinds_pre = path[idx, 1]
    toinds, idx2 = np.unique(toinds_pre, return_index=True)
    frominds = frominds[idx2]
    return frominds, toinds


def evaluate(
    alignments_path: Path,
    keyposes_path: Path,
    tmm100_path: Path,
    output_dir: Path,
    output_basename: str = "MyMocapKeyposeaccuracy",
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    pathcells, subjectdata, atlassubsesstake, atlas_t0 = _load_alignments(alignments_path)

    kp_data = loadmat(str(keyposes_path))
    keyposes = np.asarray(kp_data["keyposes"], dtype=np.float64)  # (43, P_full)

    perf_data = loadmat(str(tmm100_path))
    tmmperformances = np.asarray(perf_data["tmmperformances"], dtype=np.int64)
    if tmmperformances.shape[0] != 2:
        tmmperformances = tmmperformances.T  # (2, P_full)

    atlas_subj, atlas_take = int(atlassubsesstake[0]), int(atlassubsesstake[1])
    matches = np.where(
        (tmmperformances[0] == atlas_subj) & (tmmperformances[1] == atlas_take)
    )[0]
    if matches.size != 1:
        raise RuntimeError(
            f"Atlas (subj={atlas_subj}, take={atlas_take}) not uniquely "
            f"found in tmmperformances: matches={matches}"
        )
    atlas_col = int(matches[0])
    atlas_keyframes = keyposes[:, atlas_col]
    atlas_pose_time = (atlas_keyframes - 1) / FRAMES_PER_SECOND  # seconds

    # Skip partials.
    bad = partial_mask(subjectdata)

    all_frame_errors: list[np.ndarray] = []
    all_abs_errors: list[np.ndarray] = []
    errors_by_subject = np.zeros((len(KEYPOSES_TO_USE), subjectdata.shape[0]))

    for subjind in range(subjectdata.shape[0]):
        if bad[subjind]:
            continue

        path, subj_t0, _distmat, st = pathcells[subjind]
        subj, take = int(st[0]), int(st[1])
        # Note: subjectdata[subjind] should equal st; trust pathcells per MATLAB.

        frominds, toinds = _monotone_unique_dedup(path)
        # Indices are 1-based (MATLAB). Convert when indexing python arrays.
        from_t0 = atlas_t0[frominds - 1]
        to_t0 = subj_t0[toinds - 1]

        # Build interp1 endpoints exactly as MATLAB does.
        atlaspts = np.concatenate(([0.0], from_t0, [from_t0.max() + 50.0]))
        subjpts = np.concatenate(([0.0], to_t0, [to_t0.max() + 50.0]))

        # MATLAB interp1 default = linear, with NaN for out-of-range queries.
        # np.interp clamps to endpoints — use a custom routine that returns NaN
        # for values strictly outside [atlaspts[0], atlaspts[-1]].
        pose_time = np.interp(atlas_pose_time, atlaspts, subjpts)
        out_of_range = (atlas_pose_time < atlaspts[0]) | (atlas_pose_time > atlaspts[-1])
        pose_time = np.where(out_of_range, np.nan, pose_time)

        # Frames (round, then 1-indexed: round(t*fps + 1)).
        pose_frames = np.round(pose_time * FRAMES_PER_SECOND + 1)

        tmm_col_matches = np.where(
            (tmmperformances[0] == subj) & (tmmperformances[1] == take)
        )[0]
        if tmm_col_matches.size != 1:
            raise RuntimeError(
                f"(subj={subj}, take={take}) not uniquely found in tmmperformances"
            )
        tmm_col = int(tmm_col_matches[0])
        gt_keypose_frames = keyposes[:, tmm_col]

        pose_frames_used = pose_frames[KEYPOSES_TO_USE]
        gt_used = gt_keypose_frames[KEYPOSES_TO_USE]

        diff = pose_frames_used - gt_used
        all_frame_errors.append(diff)
        all_abs_errors.append(np.abs(diff))
        errors_by_subject[:, subjind] = diff

    all_frame_errors_arr = np.concatenate(all_frame_errors)
    all_abs_errors_arr = np.concatenate(all_abs_errors)

    # Drop NaNs (matches MATLAB).
    all_abs_errors_arr = all_abs_errors_arr[~np.isnan(all_abs_errors_arr)]
    all_frame_errors_arr = all_frame_errors_arr[~np.isnan(all_frame_errors_arr)]

    # Save raw errors mirror.
    savemat(
        str(output_dir / "mocapErrors.mat"),
        {
            "errorsbysubject": errors_by_subject,
            "allabserrors": all_abs_errors_arr,
            "allframeerrors": all_frame_errors_arr,
        },
    )

    # ROC-style curve.
    counts = np.array(
        [np.sum(all_abs_errors_arr <= x) for x in THRESHOLDS_FRAMES],
        dtype=np.float64,
    )
    counts = counts / len(all_abs_errors_arr)

    auc = np.trapezoid(counts, THRESHOLDS_FRAMES)
    auc_norm = auc / (THRESHOLDS_FRAMES.max() - THRESHOLDS_FRAMES.min())
    log.info("Normalized AUC = %.4f", auc_norm)

    # Plot.
    fig = plt.figure(figsize=(6.86, 2.7))
    plt.plot(THRESHOLDS_FRAMES / FRAMES_PER_SECOND, counts * 100, "*-", linewidth=2)
    plt.plot([1], [101], "w.")  # MATLAB hack to extend ylim
    # Annotate every other point in the [10:2:16] MATLAB range (10, 12, 14, 16
    # are 1-indexed -> Python 9, 11, 13, 15).
    for i in range(9, 16, 2):
        plt.text(
            THRESHOLDS_FRAMES[i] / FRAMES_PER_SECOND - 0.03,
            95,
            f"{counts[i] * 100:.2f}",
            fontweight="bold",
        )
    plt.grid(True)
    plt.yticks(np.arange(10, 101, 10))
    plt.xlabel("Threshold in Seconds")
    plt.ylabel("Percent Keyposes Correct")
    plt.title(f"Normalized AUC = {auc_norm:.4f}")
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    fig.tight_layout()

    eps_path = output_dir / f"{output_basename}.eps"
    png_path = output_dir / f"{output_basename}.png"
    fig.savefig(eps_path, format="eps")
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    # Save the numeric data needed to reproduce the ROC curve (without
    # rerunning the full pipeline) and to optionally bootstrap STD shading.
    roc_npz_path = output_dir / f"{output_basename}_roc.npz"
    np.savez(
        roc_npz_path,
        thresholds_frames=THRESHOLDS_FRAMES,
        thresholds_seconds=THRESHOLDS_FRAMES / FRAMES_PER_SECOND,
        counts=counts,
        counts_pct=counts * 100,
        all_abs_errors=all_abs_errors_arr,
        all_frame_errors=all_frame_errors_arr,
        errors_by_subject=errors_by_subject,
        keyposes_used=KEYPOSES_TO_USE,
        auc_norm=np.float64(auc_norm),
        halfsec_pct=np.float64(np.mean(all_abs_errors_arr <= 25) * 100),
        onesec_pct=np.float64(np.mean(all_abs_errors_arr <= 50) * 100),
        fps=np.int64(FRAMES_PER_SECOND),
    )

    # Print the headline numbers from the paper's narrative.
    halfsec_pct = float(np.mean(all_abs_errors_arr <= 25) * 100)
    onesec_pct = float(np.mean(all_abs_errors_arr <= 50) * 100)
    log.info("%% within 0.5 s : %.2f", halfsec_pct)
    log.info("%% within 1.0 s : %.2f", onesec_pct)

    return {
        "thresholds_frames": THRESHOLDS_FRAMES,
        "counts": counts,
        "auc_norm": auc_norm,
        "halfsec_pct": halfsec_pct,
        "onesec_pct": onesec_pct,
        "eps_path": eps_path,
        "png_path": png_path,
        "roc_npz_path": roc_npz_path,
        "errors_by_subject": errors_by_subject,
        "all_abs_errors": all_abs_errors_arr,
    }
