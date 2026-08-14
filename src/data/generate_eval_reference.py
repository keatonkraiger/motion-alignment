"""Generate tmm100performances.mat and tmmkeyposes.mat from our own
sources.

tmm100performances.mat's `tmmperformances` is just the (subject, take)
performance list; we derive it the same way generate_patch_dataset.py
does, by scanning which raw MOCAP_MRK_<take>.npy files actually exist.

tmmkeyposes.mat's `keyposes` (43, 100) is built from assets/keyposes.csv
(subject, take, class_index, frame_idx rows). The CSV labels all 45
classes; the standard 43-class evaluation set merges classes 33 and 35
("Turn, CH, and Left HK" / "GRS on Left Leg") into their neighboring
same-pose classes rather than keeping them as separate rows -- keeping
every class except {33, 35}, in order, reproduces that merge's net effect
on the stored frame numbers exactly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).parent))
from generate_patch_dataset import discover_performances

EXCLUDED_CLASSES = {33, 35}
NUM_CLASSES = 45  # class_index range in keyposes.csv is [0, 44]


def build_tmm100performances(performances: list[tuple[int, int]]) -> dict:
    tmm = np.array(performances, dtype=np.uint8).T  # (2, P)
    return {"tmmperformances": tmm}


def build_tmmkeyposes(keyposes_csv: Path, performances: list[tuple[int, int]]) -> dict:
    df = pd.read_csv(keyposes_csv, index_col=0)
    lookup = {(int(r.subject), int(r.take), int(r.class_index)): float(r.frame_idx)
              for r in df.itertuples()}

    classes = [c for c in range(NUM_CLASSES) if c not in EXCLUDED_CLASSES]
    keyposes = np.full((len(classes), len(performances)), np.nan, dtype=np.float64)
    for p, (subj, take) in enumerate(performances):
        for i, c in enumerate(classes):
            v = lookup.get((subj, take, c))
            if v is not None:
                keyposes[i, p] = v

    keyposecolst = np.array(performances, dtype=np.uint8).T  # (2, P)
    return {"keyposes": keyposes, "keyposecolst": keyposecolst}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path,
                     default=Path("/mnt/d/Data/PSU_NPY/Subject_wise"))
    ap.add_argument("--keyposes-csv", type=Path,
                     default=Path(__file__).resolve().parents[2] / "assets" / "keyposes.csv")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    performances = discover_performances(args.raw_root)
    print(f"{len(performances)} performances discovered from raw mocap data")

    tmm100 = build_tmm100performances(performances)
    savemat(str(args.out_dir / "tmm100performances.mat"), tmm100)
    print(f"Wrote {args.out_dir / 'tmm100performances.mat'}: "
          f"tmmperformances{tmm100['tmmperformances'].shape}")

    keyposes = build_tmmkeyposes(args.keyposes_csv, performances)
    savemat(str(args.out_dir / "tmmkeyposes.mat"), keyposes)
    print(f"Wrote {args.out_dir / 'tmmkeyposes.mat'}: "
          f"keyposes{keyposes['keyposes'].shape} "
          f"({np.isnan(keyposes['keyposes']).sum()} NaN cells)")


if __name__ == "__main__":
    main()
