"""Phase-3 sweep: factored 1D+2D DiffPIR reconstruction on SPAD cubes.

Sweep: PPP  ∈ {0.01, 0.03, 0.05, 0.1}
       α_s  ∈ {0.0, 0.25, 0.5, 0.75, 1.0}

Inputs
------
data/flux_0.npy  : ground-truth linear-flux video, shape [T_raw, H_raw, W_raw],
                   float32. Provided by the user.

Pipeline (per PPP)
------------------
1. Load flux_0.npy. Center-crop spatial to --crop × --crop (must be a
   multiple of 32 so the 2D UNet's 5 downsampling stages are well-formed).
   Take the first T=sequence_length frames along the time axis to form a
   cube `[B=1, H, W, T]`.
2. `simulate_spad_cube` → per-pixel binary detections at SPAD rate
   (n_spad_frames), then `bin_binary_time` → counts at sequence_length.
3. For each α_s, run `factored_sample_flux` starting from the same seed
   and record 5 metrics:
     corr, rel_mae, rel_mse    (pixel-time flattened, vs. GT flux)
     spatial_coherence         (mean abs spatial-gradient of reconstructed
                                log-flux frame-by-frame — lower is smoother;
                                we report the ratio vs. α_s=0 so the sign of
                                the effect is legible)
4. Append a row to docs/phase3_results.csv.

Checkpoints
-----------
Requires a trained 1D temporal UNet at --ckpt_1d. The 2D checkpoint defaults
to the guided-diffusion ImageNet-256 unconditional model at
`checkpoints/256x256_diffusion_uncond.pt` (already present; see Phase 2).

Run:
    python evaluate_factored.py \
      --flux_npy data/flux_0.npy \
      --ckpt_1d PATH/TO/1d_checkpoint.pt \
      --crop 128 \
      --sequence_length 1024 \
      --n_spad_frames 10000 \
      --num_sampling_steps 50
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from factored_diffpir import (  # noqa: E402
    FactoredConfig,
    bin_binary_time,
    factored_sample_flux,
    simulate_spad_cube,
)
from spatial_prior import NormalizationParams  # noqa: E402


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_1d_model(ckpt_path: Path, *, sequence_length: int, num_channels: int,
                  device: torch.device):
    from improved_diffusion.temporal_script_util import (
        create_temporal_model_and_diffusion,
        temporal_model_and_diffusion_defaults,
    )
    cfg = temporal_model_and_diffusion_defaults()
    cfg.update(dict(
        sequence_length=sequence_length,
        num_channels=num_channels,
        diffusion_steps=1000,
        noise_schedule="linear",
        learn_sigma=False,
    ))
    model, diffusion = create_temporal_model_and_diffusion(**cfg)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, diffusion


def load_2d_model(ckpt_path: Path, device: torch.device):
    gd_repo = REPO_ROOT.parent / "guided-diffusion"
    if str(gd_repo) not in sys.path:
        sys.path.insert(0, str(gd_repo))
    from guided_diffusion.script_util import (
        create_model_and_diffusion,
        model_and_diffusion_defaults,
    )
    cfg = model_and_diffusion_defaults()
    cfg.update(dict(
        image_size=256, num_channels=256, num_head_channels=64,
        num_res_blocks=2, attention_resolutions="32,16,8",
        resblock_updown=True, use_scale_shift_norm=True,
        learn_sigma=True, noise_schedule="linear",
        diffusion_steps=1000, class_cond=False, use_fp16=False,
    ))
    model, diffusion = create_model_and_diffusion(**cfg)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, diffusion


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_flux_cube(flux_npy: Path, crop: int, sequence_length: int,
                   t_start: int = 0) -> np.ndarray:
    """Return a [1, H, W, T] float32 linear-flux cube."""
    data = np.load(flux_npy, mmap_mode="r")  # [T_raw, H_raw, W_raw]
    T_raw, H_raw, W_raw = data.shape
    if H_raw < crop or W_raw < crop:
        raise ValueError(f"crop={crop} exceeds volume spatial {H_raw}x{W_raw}")
    if T_raw - t_start < sequence_length:
        raise ValueError(
            f"Need {sequence_length} frames from t_start={t_start}, have {T_raw}"
        )
    y0 = (H_raw - crop) // 2
    x0 = (W_raw - crop) // 2
    block = np.array(data[t_start:t_start + sequence_length,
                          y0:y0 + crop, x0:x0 + crop])  # [T, H, W]
    cube = block.transpose(1, 2, 0)[None]  # [1, H, W, T]
    return cube.astype(np.float32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def cube_metrics(pred_log_flux: np.ndarray, gt_flux: np.ndarray):
    """`pred_log_flux`: [B,H,W,T] log-flux. `gt_flux`: [B,H,W,T] linear flux."""
    pred = np.exp(pred_log_flux.astype(np.float64))
    gt = gt_flux.astype(np.float64)
    p, g = pred.ravel(), gt.ravel()

    mse = float(np.mean((p - g) ** 2))
    denom_sq = float(np.mean(g ** 2))
    rel_mse = float(mse / denom_sq) if denom_sq > 0 else float("nan")
    mae = float(np.mean(np.abs(p - g)))
    denom_abs = float(np.mean(np.abs(g)))
    rel_mae = float(mae / denom_abs) if denom_abs > 0 else float("nan")

    # Correlation on centered log-scales (numerically safer for wide-dynamic data).
    pl = np.log(pred.clip(min=1e-8)).ravel()
    gl = np.log(gt.clip(min=1e-8)).ravel()
    pl -= pl.mean(); gl -= gl.mean()
    denom = float(np.sqrt((pl ** 2).sum() * (gl ** 2).sum()))
    corr = float((pl * gl).sum() / denom) if denom > 0 else float("nan")

    # Spatial coherence: mean |∂x/∂H|+|∂x/∂W| averaged over batch and time.
    lf = pred_log_flux.astype(np.float64)  # log-flux space is smoother to diff
    dy = np.abs(np.diff(lf, axis=1))  # [B, H-1, W, T]
    dx = np.abs(np.diff(lf, axis=2))  # [B, H, W-1, T]
    spatial_coherence = float(dy.mean() + dx.mean())

    return dict(corr=corr, rel_mae=rel_mae, rel_mse=rel_mse,
                spatial_coherence=spatial_coherence)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_float_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--flux_npy", type=Path, default=REPO_ROOT / "data" / "flux_0.npy")
    p.add_argument("--ckpt_1d", type=Path, required=True,
                   help="Trained 1D temporal UNet checkpoint.")
    p.add_argument("--ckpt_2d", type=Path,
                   default=REPO_ROOT / "checkpoints" / "256x256_diffusion_uncond.pt")
    p.add_argument("--num_channels_1d", type=int, default=64)
    p.add_argument("--crop", type=int, default=128,
                   help="Spatial crop size (must be a multiple of 32 for the 2D UNet).")
    p.add_argument("--sequence_length", type=int, default=1024)
    p.add_argument("--t_start", type=int, default=0)
    p.add_argument("--n_spad_frames", type=int, default=10_000)
    p.add_argument("--num_sampling_steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.85)
    p.add_argument("--source_min", type=float, default=-3.0)
    p.add_argument("--source_max", type=float, default=2.0)
    p.add_argument("--ppps", type=parse_float_list,
                   default=parse_float_list("0.01,0.03,0.05,0.1"))
    p.add_argument("--alphas", type=parse_float_list,
                   default=parse_float_list("0.0,0.25,0.5,0.75,1.0"))
    p.add_argument("--chunk_1d", type=int, default=2048)
    p.add_argument("--chunk_2d", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_csv", type=Path,
                   default=REPO_ROOT / "docs" / "phase3_results.csv")
    p.add_argument("--out_cubes", type=Path,
                   default=REPO_ROOT / "docs" / "phase3_cubes.npz",
                   help="Save reconstructed cubes for the qualitative plot.")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.crop % 32 != 0:
        raise ValueError(f"--crop must be divisible by 32 (2D UNet); got {args.crop}")

    device = torch.device(args.device)
    print(f"[eval] device={device}")

    # Data
    flux_cube = load_flux_cube(args.flux_npy, args.crop, args.sequence_length,
                               t_start=args.t_start)
    print(f"[eval] flux cube: shape={flux_cube.shape}, "
          f"range=[{flux_cube.min():.3f}, {flux_cube.max():.3f}]")

    # Models
    print(f"[eval] loading 1D: {args.ckpt_1d}")
    model_1d, diffusion_1d = load_1d_model(
        args.ckpt_1d,
        sequence_length=args.sequence_length,
        num_channels=args.num_channels_1d,
        device=device,
    )
    print(f"[eval] loading 2D: {args.ckpt_2d}")
    model_2d, diffusion_2d = load_2d_model(args.ckpt_2d, device=device)
    norm_params = NormalizationParams(
        source_min=args.source_min, source_max=args.source_max,
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.out_csv.exists()
    csv_f = args.out_csv.open("a", newline="")
    writer = csv.writer(csv_f)
    if new_file:
        writer.writerow(["ppp", "alpha_s", "corr", "rel_mae",
                         "rel_mse", "spatial_coherence"])

    saved_cubes: dict = {}

    for ppp in args.ppps:
        # Simulate SPAD once per PPP; reuse across alphas so differences are
        # purely from the prior composition, not observation noise.
        rng = np.random.default_rng(args.seed + int(round(ppp * 1e6)))
        binary, flux_gt_hi = simulate_spad_cube(
            flux_cube, target_ppp=ppp,
            n_spad_frames=args.n_spad_frames,
            dark_count=0.0, T_exp=1.0, rng=rng,
        )
        counts = bin_binary_time(binary, args.sequence_length)
        counts_t = torch.from_numpy(counts).to(device)
        # Ground truth at seq_length via time-averaging of flux_gt_hi.
        bin_size = args.n_spad_frames // args.sequence_length
        dt_spad = 1.0 / args.n_spad_frames
        gt_seq = flux_gt_hi.reshape(*flux_gt_hi.shape[:-1],
                                    args.sequence_length, bin_size).mean(axis=-1)

        for alpha_s in args.alphas:
            print(f"[eval] ppp={ppp}  α_s={alpha_s}")
            cfg = FactoredConfig(
                sequence_length=args.sequence_length,
                num_sampling_steps=args.num_sampling_steps,
                eta=args.eta, lambda_data=1.0,
                pp_iters=5, pp_lr_scale=0.5,
                alpha_s=alpha_s,
                chunk_1d=args.chunk_1d, chunk_2d=args.chunk_2d,
            )
            out_log = factored_sample_flux(
                counts_binned=counts_t,
                model_1d=model_1d, diffusion_1d=diffusion_1d,
                model_2d=model_2d, diffusion_2d=diffusion_2d,
                norm_params=norm_params,
                cfg=cfg,
                bin_size=bin_size, dt_spad=dt_spad,
                device=device, seed=args.seed, verbose=True,
            ).cpu().numpy()

            m = cube_metrics(out_log, gt_seq)
            print(f"       corr={m['corr']:.4f}  rel_mae={m['rel_mae']:.4f}  "
                  f"rel_mse={m['rel_mse']:.4f}  sc={m['spatial_coherence']:.4f}")
            writer.writerow([ppp, alpha_s, m["corr"], m["rel_mae"],
                             m["rel_mse"], m["spatial_coherence"]])
            csv_f.flush()

            # Save the PPP=0.03 cubes for the qualitative plot.
            if abs(ppp - 0.03) < 1e-9 and alpha_s in (0.0, 0.5):
                saved_cubes[f"recon_a{alpha_s}"] = out_log
                saved_cubes["gt"] = gt_seq

    csv_f.close()
    print(f"[eval] wrote {args.out_csv}")

    if saved_cubes:
        args.out_cubes.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out_cubes, **saved_cubes)
        print(f"[eval] wrote {args.out_cubes}")


if __name__ == "__main__":
    main()
