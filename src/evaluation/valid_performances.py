"""Determine the set of "valid" performances used in the keypose evaluation.

Mirrors the partial-performance bookkeeping in the second half of
updated_eval_code/evaluateKeyposeAlignmentMocap.m (lines 143-152):

    partials = [   2 2 2 4 5 9 9 9 9; ...
                   3 5 8 6 6 1 5 6 12 ];
    partials = [partials [2 2; 10 11]];
    partials = [partials [2 2; 7  9]];
"""

from __future__ import annotations

import numpy as np
from scipy.io import loadmat


# (subject, take) pairs to exclude from evaluation.
PARTIAL_PERFORMANCES: list[tuple[int, int]] = [
    (2, 3), (2, 5), (2, 8),
    (4, 6),
    (5, 6),
    (9, 1), (9, 5), (9, 6), (9, 12),
    # No-foot-pressure takes:
    (2, 10), (2, 11),
    # Pressures look wrong:
    (2, 7), (2, 9),
]


def load_all_performances(tmm100_mat_path) -> np.ndarray:
    """Load tmmperformances and return shape (P, 2) array of (subj, take)."""
    foo = loadmat(str(tmm100_mat_path))
    tmm = foo["tmmperformances"]  # MATLAB shape (2, P)
    if tmm.shape[0] != 2:
        tmm = tmm.T
    return tmm.T.astype(np.int64)  # (P, 2)


def partial_mask(subjectdata: np.ndarray) -> np.ndarray:
    """Return a boolean mask that is True for partial (excluded) performances."""
    bad = set(PARTIAL_PERFORMANCES)
    return np.array([(int(s), int(t)) in bad for s, t in subjectdata], dtype=bool)


def valid_mask(subjectdata: np.ndarray) -> np.ndarray:
    return ~partial_mask(subjectdata)
