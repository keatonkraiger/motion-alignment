"""Compute pairwise DTW alignments between an atlas performance and every
valid PSU-TMM100 performance, using a trained Siamese encoder.

Mirrors the first half of updated_eval_code/evaluateKeyposeAlignmentMocap.m,
but using the PyTorch encoder from motion_alignment/src/model.py and the
verbatim DTW port in evaluation/dtw_from_distmat.py.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat, savemat

# Make sibling modules importable when run as a script.
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import Encoder  # noqa: E402

from dtw_from_distmat import dtw_from_distmat  # noqa: E402
from valid_performances import load_all_performances, valid_mask  # noqa: E402


def _load_patches(patch_dir: Path, subj: int, take: int):
    data = loadmat(str(patch_dir / f"patches_{subj}_{take}.mat"))
    a = data["A"].transpose((3, 2, 0, 1))  # (N, 1, S, F)
    t0 = np.asarray(data["t0"]).squeeze()  # (N,)
    return a.astype(np.float32), t0.astype(np.float64)


def _embed(model: torch.nn.Module, a_np: np.ndarray, device: str) -> np.ndarray:
    """Run encoder and return L2-normalized embeddings of shape (N, D)."""
    a = torch.from_numpy(a_np).to(device)
    with torch.no_grad():
        z = model(a)  # (N, D)
        z = F.normalize(z, p=2, dim=1)
    return z.cpu().numpy().astype(np.float64)


def _cosine_distmat(atlas: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Cosine *distance* (1 - cos sim) between row vectors of `atlas` and `query`.

    Inputs are already L2-normalized, so this is `1 - atlas @ query.T`.
    Matches MATLAB ``pdist2(..., 'cosine')``.
    """
    sims = atlas @ query.T
    return 1.0 - sims


def compute_alignments(
    model_path: Path,
    patch_dir: Path,
    tmm100_path: Path,
    output_path: Path,
    atlas_subj: int = 7,
    atlas_take: int = 2,
    device: str | None = None,
) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model. Try state_dict first; fall back to pickled module.
    state_dict_path = model_path.with_name("model_state_dict.pt")
    if state_dict_path.exists():
        print(f"Loading state dict: {state_dict_path}")
        model = Encoder()
        model.load_state_dict(torch.load(state_dict_path, map_location=device))
    else:
        print(f"Loading pickled model: {model_path}")
        model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    # Embed atlas.
    print(f"Embedding atlas: subject {atlas_subj}, take {atlas_take}")
    atlas_a, atlas_t0 = _load_patches(patch_dir, atlas_subj, atlas_take)
    atlas_vecs = _embed(model, atlas_a, device)  # (Na, D)

    # Iterate valid performances.
    subjectdata = load_all_performances(tmm100_path)  # (P, 2)
    keep = valid_mask(subjectdata)
    print(
        f"Total performances: {subjectdata.shape[0]}, "
        f"valid (kept): {keep.sum()}, partials skipped: {(~keep).sum()}"
    )

    pathcells = []  # list of [path, t0, distmat, [subj, take]]
    costmat = np.zeros(subjectdata.shape[0])
    costmatavg = np.zeros(subjectdata.shape[0])

    for sstind in range(subjectdata.shape[0]):
        subj, take = int(subjectdata[sstind, 0]), int(subjectdata[sstind, 1])
        if not keep[sstind]:
            print(f"  [skip partial] subj {subj} take {take}")
            # Still keep a placeholder so indices align with subjectdata.
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

        vecs = _embed(model, patches_a, device)  # (Nq, D)
        distmat = _cosine_distmat(atlas_vecs, vecs)  # (Na, Nq)
        path = dtw_from_distmat(distmat)  # 1-indexed (L, 2)

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
        print(
            f"  subj {subj} take {take}: path_len={path.shape[0]}, "
            f"avg_cost={costmatavg[sstind]:.4f}"
        )

    # Save in a format compatible with the MATLAB script (cell-of-cells).
    pathcells_obj = np.empty((len(pathcells),), dtype=object)
    for i, entry in enumerate(pathcells):
        cell = np.empty((1, 4), dtype=object)
        cell[0, 0] = entry[0]
        cell[0, 1] = entry[1].reshape(1, -1)  # row vector for MATLAB t0 layout
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
        f"  mean   avg cost (valid): {np.mean(valid_costs):.4f}"
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=Path, required=True,
                        help="Path to model.pt (state_dict or pickled module).")
    parser.add_argument("--patch-dir", type=Path, required=True,
                        help="Directory containing patches_{s}_{t}.mat files.")
    parser.add_argument(
        "--tmm100", type=Path,
        default=Path("../../data/eval/tmm100performances.mat"),
    )
    parser.add_argument("--output", type=Path, default=Path("../outputs/MyMocapAlignments.mat"))
    parser.add_argument("--atlas-subject", type=int, default=7)
    parser.add_argument("--atlas-take", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    compute_alignments(
        model_path=args.model,
        patch_dir=args.patch_dir,
        tmm100_path=args.tmm100,
        output_path=args.output,
        atlas_subj=args.atlas_subject,
        atlas_take=args.atlas_take,
        device=args.device,
    )
