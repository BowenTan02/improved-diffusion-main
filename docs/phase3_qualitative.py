"""Render docs/phase3_qualitative.png from docs/phase3_cubes.npz.

We pick one spatial "discontinuity event" frame — the time-slice with the
largest mean spatial gradient in the ground-truth log-flux cube — and show:

    rows:    GT | α_s = 0.0 | α_s = 0.5
    cols:    time-slice | spatial residual (pred - gt) | |∇_xy log-flux|

The middle column makes spatial artefacts from per-pixel 1D priors visible
as speckle; the right column makes the smoothing effect of α_s > 0 legible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def spatial_grad_mag(x: np.ndarray) -> np.ndarray:
    """[H, W] -> [H, W] magnitude of discrete spatial gradient."""
    dy = np.pad(np.abs(np.diff(x, axis=0)), ((0, 1), (0, 0)))
    dx = np.pad(np.abs(np.diff(x, axis=1)), ((0, 0), (0, 1)))
    return dy + dx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubes", type=Path,
                    default=REPO_ROOT / "docs" / "phase3_cubes.npz")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "docs" / "phase3_qualitative.png")
    args = ap.parse_args()

    data = np.load(args.cubes)
    recon_a0 = data["recon_a0.0"]        # [1, H, W, T] log-flux
    recon_a5 = data["recon_a0.5"]
    gt = data["gt"]                     # [1, H, W, T] linear flux

    gt_log = np.log(np.clip(gt, 1e-8, None))

    # Event frame: highest mean spatial-gradient magnitude in GT log-flux.
    sg_per_t = np.stack([spatial_grad_mag(gt_log[0, :, :, t]).mean()
                         for t in range(gt_log.shape[-1])])
    t_event = int(np.argmax(sg_per_t))
    print(f"event frame t={t_event}  |∇| mean={sg_per_t[t_event]:.3f}")

    gt_f = gt_log[0, :, :, t_event]
    a0_f = recon_a0[0, :, :, t_event]
    a5_f = recon_a5[0, :, :, t_event]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vmin = float(min(gt_f.min(), a0_f.min(), a5_f.min()))
    vmax = float(max(gt_f.max(), a0_f.max(), a5_f.max()))
    res0 = a0_f - gt_f
    res5 = a5_f - gt_f
    rmax = float(max(np.abs(res0).max(), np.abs(res5).max(), 1e-6))
    gmax = float(max(
        spatial_grad_mag(gt_f).max(),
        spatial_grad_mag(a0_f).max(),
        spatial_grad_mag(a5_f).max(),
    ))

    rows = [("GT",            gt_f, np.zeros_like(gt_f)),
            ("α_s = 0.0",     a0_f, res0),
            ("α_s = 0.5",     a5_f, res5)]

    fig, ax = plt.subplots(3, 3, figsize=(11, 10))
    for i, (label, f, r) in enumerate(rows):
        ax[i, 0].imshow(f, vmin=vmin, vmax=vmax, cmap="viridis")
        ax[i, 0].set_ylabel(label, fontsize=11)
        ax[i, 1].imshow(r, vmin=-rmax, vmax=rmax, cmap="RdBu_r")
        ax[i, 2].imshow(spatial_grad_mag(f), vmin=0, vmax=gmax, cmap="magma")
        for j in range(3):
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        if i == 0:
            ax[i, 0].set_title("log-flux frame", fontsize=10)
            ax[i, 1].set_title("residual (pred - GT)", fontsize=10)
            ax[i, 2].set_title("|∇ log-flux|", fontsize=10)

    fig.suptitle(f"Phase 3 qualitative — PPP=0.03, event frame t={t_event}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
