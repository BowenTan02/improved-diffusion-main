"""Visual verification of the 2D-prior bridge on real SPAD log-flux frames.

Loads a handful of clean frames from one of the Phase-1 flux `.npy` files,
adds q(x_t|x_0) noise at three diffusion timesteps, runs `denoise_frame_2d`,
and writes a 4-column grid PNG: [clean | noisy | denoised | residual].

Prints PSNR of noisy-vs-clean and denoised-vs-clean at each t.

Requires:
- `spatial_prior.py` at repo root (already in place after Phase 2).
- guided-diffusion source at `../guided-diffusion/` (already present).
- 256x256_diffusion_uncond.pt checkpoint. If not found at any of the
  candidate paths, this script exits with a `curl` command to fetch it
  (~2.1 GB, so NOT auto-downloaded).

Run from repo root:
    python docs/phase2_verify.py
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from spatial_prior import (  # noqa: E402
    NormalizationParams,
    to_2d_space,
    from_2d_space,
    denoise_frame_2d,
)

DEFAULT_FLUX_NPY = Path(
    "/Users/tan583/Documents/3D Probing/Quantitative/QNN_128K_0.1ppp/"
    "Raw Flux/wineglassfall-fps1000_rife4_linear8_ppp0.1.zarr_reconstruction.npy"
)

CHECKPOINT_URL = (
    "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/"
    "256x256_diffusion_uncond.pt"
)


def find_checkpoint() -> Optional[Path]:
    candidates = [
        REPO_ROOT / "checkpoints" / "256x256_diffusion_uncond.pt",
        REPO_ROOT.parent / "guided-diffusion" / "models" / "256x256_diffusion_uncond.pt",
        REPO_ROOT.parent / "DiffPIR-main" / "model_zoo" / "256x256_diffusion_uncond.pt",
        Path(os.environ.get("IMAGENET256_UNCOND_CKPT", "")),
    ]
    for p in candidates:
        if p and p.is_file():
            return p
    return None


def load_imagenet256_uncond(ckpt_path: Path, device: torch.device):
    gd_repo = REPO_ROOT.parent / "guided-diffusion"
    if str(gd_repo) not in sys.path:
        sys.path.insert(0, str(gd_repo))
    from guided_diffusion.script_util import (  # type: ignore
        create_model_and_diffusion,
        model_and_diffusion_defaults,
    )

    cfg = model_and_diffusion_defaults()
    cfg.update(dict(
        image_size=256,
        num_channels=256,
        num_head_channels=64,
        num_res_blocks=2,
        attention_resolutions="32,16,8",
        resblock_updown=True,
        use_scale_shift_norm=True,
        learn_sigma=True,
        noise_schedule="linear",
        diffusion_steps=1000,
        class_cond=False,
        use_fp16=False,
    ))
    model, diffusion = create_model_and_diffusion(**cfg)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    return model, diffusion


def center_crop(arr: np.ndarray, size: int) -> np.ndarray:
    """Center-crop an `[H, W]` array to `[size, size]`."""
    H, W = arr.shape[-2:]
    assert H >= size and W >= size, f"Array too small: {arr.shape} < {size}"
    y0 = (H - size) // 2
    x0 = (W - size) // 2
    return arr[..., y0:y0 + size, x0:x0 + size]


def load_clean_frames(flux_npy: Path, n_frames: int, crop: int) -> np.ndarray:
    """Load `n_frames` evenly-spaced frames from a flux `.npy`, crop, and
    return log-flux as `[n_frames, crop, crop]` float32.
    """
    data = np.load(flux_npy, mmap_mode="r")  # [T, H, W]
    T = data.shape[0]
    idxs = np.linspace(0, T - 1, n_frames, dtype=int)
    frames = np.stack([np.array(data[i]) for i in idxs], axis=0)  # [n, H, W]
    frames = center_crop(frames, crop).astype(np.float32)
    # log-flux with small epsilon to handle zeros.
    log_flux = np.log(frames + 1e-6)
    return log_flux


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float) -> float:
    mse = ((pred - target) ** 2).mean().item()
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10((data_range ** 2) / mse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flux-npy", type=Path, default=DEFAULT_FLUX_NPY)
    parser.add_argument("--n-frames", type=int, default=3,
                        help="Number of clean frames to sample from the .npy.")
    parser.add_argument("--crop", type=int, default=256,
                        help="Center-crop size fed to the 256-px 2D UNet.")
    parser.add_argument("--source-min", type=float, default=-14.0,
                        help="log-flux value mapped to -1 in 2D space.")
    parser.add_argument("--source-max", type=float, default=-1.0,
                        help="log-flux value mapped to +1 in 2D space.")
    parser.add_argument("--t-fracs", type=float, nargs="+", default=[0.25, 0.5, 0.75],
                        help="Fractions of T_diff at which to inject noise.")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "docs" / "phase2_verify.png")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt = find_checkpoint()
    if ckpt is None:
        msg = (
            "\n[phase2_verify] 256x256_diffusion_uncond.pt not found.\n"
            "Download (2.1 GB) to one of:\n"
            f"  {REPO_ROOT / 'checkpoints'}/\n"
            f"  {REPO_ROOT.parent / 'guided-diffusion' / 'models'}/\n\n"
            "    mkdir -p checkpoints && \\\n"
            f"    curl -L -o checkpoints/256x256_diffusion_uncond.pt \\\n"
            f"      '{CHECKPOINT_URL}'\n\n"
            "Or set IMAGENET256_UNCOND_CKPT=<path> and re-run.\n"
        )
        print(msg)
        sys.exit(2)

    if not args.flux_npy.is_file():
        print(f"[phase2_verify] Flux file not found: {args.flux_npy}", file=sys.stderr)
        sys.exit(2)

    device = torch.device(args.device)
    print(f"[phase2_verify] device: {device}")
    print(f"[phase2_verify] checkpoint: {ckpt}")
    print(f"[phase2_verify] flux:       {args.flux_npy}")

    # Load frames
    log_flux_np = load_clean_frames(args.flux_npy, args.n_frames, args.crop)
    clean = torch.from_numpy(log_flux_np).to(device)  # [n, H, W]
    print(f"[phase2_verify] clean log-flux: shape={tuple(clean.shape)}, "
          f"range=[{clean.min().item():.3f}, {clean.max().item():.3f}]")

    # Build 2D model + diffusion
    print("[phase2_verify] loading 2D model...")
    model, diffusion = load_imagenet256_uncond(ckpt, device)
    T_diff = diffusion.num_timesteps
    ts = [int(round(f * T_diff)) for f in args.t_fracs]
    ts = [min(max(t, 0), T_diff - 1) for t in ts]
    print(f"[phase2_verify] T_diff={T_diff}, timesteps={ts}")

    params = NormalizationParams(source_min=args.source_min, source_max=args.source_max)
    data_range = args.source_max - args.source_min

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = args.n_frames * len(ts)
    fig, axes = plt.subplots(n_rows, 4, figsize=(12, 3 * n_rows), squeeze=False)
    col_titles = ["clean", "noisy (x_t)", "denoised (x̂_0)", "residual"]

    row = 0
    for t in ts:
        # Add noise in 2D [-1,1] space so variance matches the checkpoint's
        # schedule exactly, then map back to log-flux for the wrapper input.
        clean_2d = to_2d_space(clean, params)
        noise = torch.randn_like(clean_2d)
        t_vec = torch.full((clean.shape[0],), t, dtype=torch.long, device=device)
        xt_2d = diffusion.q_sample(clean_2d, t_vec, noise=noise)
        xt_logflux = from_2d_space(xt_2d, params)

        with torch.no_grad():
            x0_hat = denoise_frame_2d(
                xt_logflux, t=t, model_2d=model, diffusion_2d=diffusion,
                normalization_params=params,
            )

        for i in range(args.n_frames):
            c = clean[i].cpu().numpy()
            n = xt_logflux[i].cpu().numpy()
            d = x0_hat[i].cpu().numpy()
            r = d - c

            p_noisy = psnr(xt_logflux[i], clean[i], data_range)
            p_denoi = psnr(x0_hat[i], clean[i], data_range)
            print(
                f"  frame {i}, t={t:4d}:  "
                f"PSNR noisy={p_noisy:6.2f} dB   denoised={p_denoi:6.2f} dB   "
                f"Δ={p_denoi - p_noisy:+.2f} dB"
            )

            vmin, vmax = args.source_min, args.source_max
            axes[row, 0].imshow(c, vmin=vmin, vmax=vmax, cmap="viridis")
            axes[row, 1].imshow(n, vmin=vmin, vmax=vmax, cmap="viridis")
            axes[row, 2].imshow(d, vmin=vmin, vmax=vmax, cmap="viridis")
            rmax = max(abs(r.min()), abs(r.max()), 1e-6)
            axes[row, 3].imshow(r, vmin=-rmax, vmax=rmax, cmap="RdBu_r")

            axes[row, 0].set_ylabel(f"frame {i}\nt={t}", fontsize=9)
            for c_idx in range(4):
                axes[row, c_idx].set_xticks([])
                axes[row, c_idx].set_yticks([])
                if row == 0:
                    axes[row, c_idx].set_title(col_titles[c_idx], fontsize=10)
            row += 1

    fig.suptitle(
        f"Phase 2 verification — ImageNet-256 uncond as 2D prior\n"
        f"source_min={args.source_min}, source_max={args.source_max}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"\n[phase2_verify] wrote grid to {args.out}")


if __name__ == "__main__":
    main()
