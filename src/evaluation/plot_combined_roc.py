"""Combined ROC figure for CNN, PoseFormer, and PoseFormerV2 ablations.

Reads the per-encoder ROC ``.npz`` files written by phase 2's
``evaluation/keypose_eval.py`` and produces a single, paper-ready figure
(PDF + EPS, vector text) overlaying the three curves with markers at
1.00, 1.25, 1.50, and 1.75 seconds.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "outputs" / "content"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHODS = [
    {
        "label": "CNN (ours)",
        "npz": REPO / "outputs/cnn_ablation/MyMocapKeyposeaccuracy_cnn_roc.npz",
        "color": "#1f77b4",
        "linestyle": "-",
    },
    {
        "label": "PoseFormer",
        "npz": REPO / "outputs/poseformer_ablation/MyMocapKeyposeaccuracy_poseformer_roc.npz",
        "color": "#d62728",
        "linestyle": "--",
    },
    {
        "label": "PoseFormerV2",
        "npz": REPO / "outputs/poseformerv2_ablation/MyMocapKeyposeaccuracy_poseformerv2_roc.npz",
        "color": "#ff7f0e",
        "linestyle": "-.",
    },
]

MARKER_SECONDS = np.array([1.00, 1.25, 1.50, 1.75])


def _pct_at(errs_frames: np.ndarray, fps: float, secs: np.ndarray) -> np.ndarray:
    """Percent of |errors| within each threshold (in seconds)."""
    thr_frames = secs * fps
    return np.array([(errs_frames <= t).mean() for t in thr_frames]) * 100.0


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,   # keep text editable / selectable in PDF
        "ps.fonttype": 42,    # keep text editable / selectable in EPS
    })

    fig, ax = plt.subplots(figsize=(8.0, 2.6))

    # Draw CNN (ours) last so it renders on top of the transformer baselines.
    for i, m in enumerate(METHODS):
        is_cnn = m["label"].startswith("CNN")
        line_z = 4 if is_cnn else 2 + i * 0.1
        marker_z = 6 if is_cnn else 5
        d = np.load(m["npz"])
        thr_s = d["thresholds_seconds"].astype(float)
        counts_pct = d["counts_pct"].astype(float)
        fps = float(d["fps"])
        errs = d["all_abs_errors"].astype(float)
        auc = float(d["auc_norm"])

        ax.plot(
            thr_s,
            counts_pct,
            color=m["color"],
            linestyle=m["linestyle"],
            linewidth=1.8,
            label=f"{m['label']} (AUC = {auc:.3f})",
            zorder=line_z,
        )

        # Markers at requested seconds (computed exactly from raw errors).
        marker_pct = _pct_at(errs, fps, MARKER_SECONDS)
        ax.plot(
            MARKER_SECONDS,
            marker_pct,
            marker="*",
            linestyle="none",
            markersize=9,
            markerfacecolor=m["color"],
            markeredgecolor="black",
            markeredgewidth=0.5,
            zorder=marker_z,
        )

    ax.set_xlim(0.0, 1.85)
    ax.set_ylim(20, 101)
    ax.set_yticks(np.arange(20, 101, 10))
    ax.set_xticks(np.arange(0.0, 1.81, 0.25))
    ax.set_xlabel("Threshold (seconds)")
    ax.set_ylabel("Percent Keyposes Correct")
    ax.grid(True, which="major", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)

    fig.tight_layout()

    pdf_path = OUT_DIR / "roc_combined.pdf"
    eps_path = OUT_DIR / "roc_combined.eps"
    fig.savefig(pdf_path)
    fig.savefig(eps_path, format="eps")
    plt.close(fig)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {eps_path}")


if __name__ == "__main__":
    main()
