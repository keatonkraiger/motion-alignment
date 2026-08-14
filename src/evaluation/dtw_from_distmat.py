"""Verbatim Python port of updated_eval_code/dtwFromDistmat.m.

3-step DTW with no diagonal weighting (the MATLAB code's chosen variant),
producing the same path representation: an (L, 2) array of 1-indexed
(row, col) pairs in temporal order, with the (0, 0) sentinel removed and
the path ``flipud``-reversed to start near (1, 1).
"""

from __future__ import annotations

import numpy as np


def dtw_from_distmat(distmat: np.ndarray) -> np.ndarray:
    """Compute DTW alignment path from a precomputed distance matrix.

    Parameters
    ----------
    distmat : np.ndarray
        Pairwise distance matrix of shape (N, M).

    Returns
    -------
    path : np.ndarray
        Alignment path, shape (L, 2), 1-indexed, in temporal order.
    """
    distmat = np.asarray(distmat)
    N, M = distmat.shape

    # Cumulative table with sentinel row/col padded by +Inf, except (0,0)=0.
    INF = np.inf
    dtw = np.full((N + 1, M + 1), INF)
    dtw[0, 0] = 0.0
    # Direction codes: 1=from (i-1,j), 2=from (i,j-1), 3=from (i-1,j-1).
    dtwdirs = np.zeros((N + 1, M + 1), dtype=np.int8)

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            # Same triplet ordering as MATLAB: [up, left, diag].
            up = dtw[i - 1, j]
            left = dtw[i, j - 1]
            diag = dtw[i - 1, j - 1]
            # MATLAB min returns the FIRST minimum; replicate via argmin
            # over [up, left, diag].
            triplet = (up, left, diag)
            minind = int(np.argmin(triplet))  # 0,1,2
            minval = triplet[minind]
            dtw[i, j] = distmat[i - 1, j - 1] + minval
            dtwdirs[i, j] = minind + 1  # MATLAB cases are 1,2,3.

    # Backtrack.
    path_list: list[tuple[int, int]] = []
    i, j = N, M
    while True:
        path_list.append((i, j))
        d = dtwdirs[i, j]
        if d == 0:
            break
        elif d == 1:
            i -= 1
        elif d == 2:
            j -= 1
        elif d == 3:
            i -= 1
            j -= 1
        else:
            raise RuntimeError(f"Unexpected direction code {d} at ({i},{j})")

    path = np.array(path_list, dtype=np.int64)
    # MATLAB does flipud then drops the (0,0) sentinel.
    path = path[::-1]
    path = path[1:]
    return path
