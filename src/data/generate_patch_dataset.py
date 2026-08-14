"""Batch-generate the full patches_<subj>_<take>.mat dataset directly from
raw PSU-TMM100 mocap marker data, using only this repo's own Python
extraction port (extract_mocap_patches.py).

Scans <raw-root>/Subject<N>/MOCAP_MRK_<take>.npy for every subject to
build the (subject, take) performance list, extracts patches for each
with a deterministic per-take seed, and writes them out as
patches_<subj>_<take>.mat (keys A, B, t0) -- the format
data/datasets.py and evaluation/alignments.py expect.

TrainPatches is not independently resampled: it's the same extracted
patches for the 20-take training subset (datasets.ALL_TAKES), just also
written into a separate directory.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_mocap_patches import PatchOptions, extract_mocap_patches
from datasets import ALL_TAKES

TAKE_RE = re.compile(r"^MOCAP_MRK_(\d+)\.npy$")


def discover_performances(raw_root: Path) -> list[tuple[int, int]]:
    perfs = []
    for subj_dir in sorted(raw_root.glob("Subject*")):
        m = re.match(r"^Subject(\d+)$", subj_dir.name)
        if not m:
            continue
        subject = int(m.group(1))
        for f in subj_dir.iterdir():
            tm = TAKE_RE.match(f.name)
            if tm:
                perfs.append((subject, int(tm.group(1))))
    return sorted(perfs)


def generate_one(raw_root: Path, subject: int, take: int, markers_to_remove):
    mocap_path = raw_root / f"Subject{subject}" / f"MOCAP_MRK_{take}.npy"
    mcmarkers = np.load(mocap_path)
    opts = PatchOptions(
        samplerateseconds=0.5,
        removelowaccelerations=False,
        padborders=True,
        markerstoremove=markers_to_remove,
        seed=subject * 1000 + take,
    )
    result = extract_mocap_patches(mcmarkers, opts)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path,
                     default=Path("/mnt/d/Data/PSU_NPY/Subject_wise"))
    ap.add_argument("--patches-out", type=Path, required=True,
                     help="output dir for the full performance set (Patches/)")
    ap.add_argument("--trainpatches-out", type=Path, required=True,
                     help="output dir for the training subset (TrainPatches/)")
    ap.add_argument("--markerstoremove", nargs="*", default=["RBAK"])
    args = ap.parse_args()

    args.patches_out.mkdir(parents=True, exist_ok=True)
    args.trainpatches_out.mkdir(parents=True, exist_ok=True)

    performances = discover_performances(args.raw_root)
    print(f"Discovered {len(performances)} performances across "
          f"{len({s for s, _ in performances})} subjects")

    train_set = set(ALL_TAKES)

    for i, (subject, take) in enumerate(performances, 1):
        result = generate_one(args.raw_root, subject, take, args.markerstoremove)
        mat_dict = {"A": result["A"], "B": result["B"], "t0": result["t0"]}
        out_path = args.patches_out / f"patches_{subject}_{take}.mat"
        savemat(str(out_path), mat_dict)
        if (subject, take) in train_set:
            savemat(str(args.trainpatches_out / f"patches_{subject}_{take}.mat"), mat_dict)
        print(f"[{i}/{len(performances)}] subj {subject} take {take}: "
              f"A{result['A'].shape} B{result['B'].shape} -> {out_path.name}"
              + ("  (+ TrainPatches)" if (subject, take) in train_set else ""))

    n_train_written = len(list(args.trainpatches_out.glob("*.mat")))
    print(f"\nDone. Patches: {len(performances)} files in {args.patches_out}")
    print(f"TrainPatches: {n_train_written} files in {args.trainpatches_out} "
          f"(expected {len(train_set)})")


if __name__ == "__main__":
    main()
