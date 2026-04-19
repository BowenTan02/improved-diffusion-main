"""Factored DiffPIR sampler: 1D temporal prior + 2D spatial prior via convex
Tweedie combination.

The forward observation model is the same as the existing 1D pipeline — per
pixel SPAD binary detections are simulated, then binned to `sequence_length`
time bins and used as a Poisson-like data term (via the Anscombe VST).

At every reverse diffusion step we compute TWO Tweedie x0 estimates:
  - x0_temporal: the 1D UNet applied per pixel across the time axis
  - x0_spatial:  the 2D UNet applied per time-slice across the spatial axes
                 (via spatial_prior.denoise_frame_2d)
and blend them with a user-controlled weight:
    x0_combined = alpha_s * x0_spatial + (1 - alpha_s) * x0_temporal
The data subproblem and the DDIM update then proceed as in vanilla DiffPIR.

When alpha_s == 0.0 no 2D forward pass is performed, and the loop reduces
*exactly* to 1D-only DiffPIR (see tests/test_factored_alpha0_equivalence.py).

Primary tensor layout: `x_t` is kept as [B, H, W, T] log-flux. Reshapes into
1D [B*H*W, 1, T] and 2D [B*T, H, W] views happen only around model calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from spatial_prior import NormalizationParams, denoise_frame_2d


# ---------------------------------------------------------------------------
# SPAD forward simulation on a spatio-temporal cube.
# ---------------------------------------------------------------------------

def simulate_spad_cube(
    flux_cube: np.ndarray,
    target_ppp: float,
    n_spad_frames: int,
    dark_count: float = 0.0,
    T_exp: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate SPAD binary detections for a [B, H, W, T_gt] linear-flux cube.

    Scales each pixel's flux so that the CUBE-mean matches `target_ppp` per
    frame (i.e. one global scaling — not per-pixel — so spatial brightness
    contrast is preserved; per-pixel normalization would flatten every pixel).

    Returns
    -------
    binary : uint8 [B, H, W, n_spad_frames]
    flux_gt : float32 [B, H, W, n_spad_frames]   ground-truth flux at SPAD rate
    """
    if rng is None:
        rng = np.random.default_rng(0)

    B, H, W, T_gt = flux_cube.shape
    flux = flux_cube.astype(np.float64)

    # Temporal interp from T_gt -> n_spad_frames on the time axis.
    if T_gt != n_spad_frames:
        t_src = np.linspace(0.0, T_exp, T_gt)
        t_dst = np.linspace(0.0, T_exp, n_spad_frames)
        flat = flux.reshape(-1, T_gt)
        interp = np.empty((flat.shape[0], n_spad_frames), dtype=np.float64)
        for i in range(flat.shape[0]):
            interp[i] = np.interp(t_dst, t_src, flat[i])
        flux_hi = interp.reshape(B, H, W, n_spad_frames)
    else:
        flux_hi = flux

    dt_frame = T_exp / n_spad_frames
    cube_mean = flux_hi.mean()
    if cube_mean > 0:
        scale = target_ppp / (cube_mean * dt_frame)
    else:
        scale = 1.0
    flux_gt = flux_hi * scale
    N_t = flux_gt * dt_frame  # expected photons per SPAD frame, per pixel

    det_prob = 1.0 - np.exp(-(N_t + dark_count))
    binary = (rng.random(det_prob.shape) < det_prob).astype(np.uint8)
    return binary, flux_gt.astype(np.float32)


def bin_binary_time(binary: np.ndarray, target_bins: int) -> np.ndarray:
    """[B, H, W, n_spad_frames] uint8 -> [B, H, W, target_bins] float32 counts."""
    B, H, W, T = binary.shape
    if T == target_bins:
        return binary.astype(np.float32)
    if T % target_bins == 0:
        m = T // target_bins
        return binary.reshape(B, H, W, target_bins, m).sum(axis=-1).astype(np.float32)
    edges = np.linspace(0, T, target_bins + 1, dtype=np.int64)
    out = np.empty((B, H, W, target_bins), dtype=np.float32)
    flat = binary.reshape(-1, T).astype(np.int64)
    for i in range(flat.shape[0]):
        sums = np.add.reduceat(flat[i], edges[:-1].astype(np.intp))
        out.reshape(-1, target_bins)[i] = sums[:target_bins].astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Anscombe-VST Poisson data step.
# ---------------------------------------------------------------------------

def anscombe_data_step(
    x0_pred: torch.Tensor,          # [N, 1, T] log-flux
    counts: torch.Tensor,           # [N, 1, T] non-negative float
    *,
    bin_size: float,                # SPAD frames per bin (integer in practice)
    dt_spad: float,                 # seconds per SPAD frame
    rho_t: torch.Tensor,            # scalar
    sigma_t_bar: float,
    n_iter: int = 5,
    lr_scale: float = 0.5,
) -> torch.Tensor:
    """Gradient-based data step with Anscombe-transformed Poisson likelihood.

    Under the Anscombe VST, `y_ans = 2*sqrt(counts + 3/8) ~ N(2*sqrt(lam+3/8), 1)`
    where `lam = exp(x) * bin_size * dt_spad` is the expected count per bin.
    We minimize `0.5*(y_ans - y_hat(x))^2 + rho_t/2 * ||x - x0_pred||^2`
    with a few proximal-gradient steps.
    """
    x = x0_pred.clone().requires_grad_(True)
    y_ans = 2.0 * torch.sqrt(counts + 0.375)
    step = lr_scale * (sigma_t_bar ** 2) / (2.0 * rho_t + 1e-8)
    for _ in range(n_iter):
        phi = torch.exp(x.clamp(-15.0, 10.0))
        lam = phi * (bin_size * dt_spad)
        y_hat = 2.0 * torch.sqrt(lam + 0.375)
        nll = 0.5 * ((y_ans - y_hat) ** 2).sum()
        prox = 0.5 * rho_t * ((x - x0_pred) ** 2).sum()
        loss = nll + prox
        grad = torch.autograd.grad(loss, x)[0]
        with torch.no_grad():
            x = (x - step * grad).clamp(-10.0, 10.0)
        x = x.detach().requires_grad_(True)
    return x.detach()


# ---------------------------------------------------------------------------
# Factored sampler.
# ---------------------------------------------------------------------------

@dataclass
class FactoredConfig:
    sequence_length: int = 1024
    num_sampling_steps: int = 100
    eta: float = 0.85
    lambda_data: float = 1.0
    pp_iters: int = 5
    pp_lr_scale: float = 0.5
    alpha_s: float = 0.0
    chunk_1d: int = 4096    # pixels per 1D forward chunk
    chunk_2d: int = 8       # time-slices per 2D forward chunk


def _chunked_forward_1d(model_1d, x_1d, t_tensor_scalar, *, chunk, device):
    """x_1d: [N, 1, T]. Returns eps: [N, 1, T]."""
    N = x_1d.shape[0]
    out = torch.empty_like(x_1d)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        tt = torch.full((e - s,), int(t_tensor_scalar), dtype=torch.long, device=device)
        out[s:e] = model_1d(x_1d[s:e], tt)
    return out


def _chunked_denoise_2d(model_2d, diffusion_2d, norm_params, x_2d_flat, t_scalar, *, chunk):
    """x_2d_flat: [N, H, W] in log-flux space. Returns x0_hat: [N, H, W]."""
    N = x_2d_flat.shape[0]
    out = torch.empty_like(x_2d_flat)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        out[s:e] = denoise_frame_2d(
            x_2d_flat[s:e], t=int(t_scalar),
            model_2d=model_2d, diffusion_2d=diffusion_2d,
            normalization_params=norm_params,
        )
    return out


def factored_sample_flux(
    counts_binned: torch.Tensor,    # [B, H, W, T_seq] float32, counts per seq-bin
    *,
    model_1d,
    diffusion_1d,
    model_2d=None,
    diffusion_2d=None,
    norm_params: Optional[NormalizationParams] = None,
    cfg: FactoredConfig = FactoredConfig(),
    bin_size: int = 1,              # n_spad_frames per seq-bin
    dt_spad: float = 1e-5,          # seconds per SPAD frame
    device: Optional[torch.device] = None,
    seed: int = 0,
    verbose: bool = True,
):
    """Run factored DiffPIR on a cube; returns estimated log-flux [B, H, W, T]."""
    if device is None:
        device = counts_binned.device
    B, H, W, T = counts_binned.shape
    assert T == cfg.sequence_length, f"seq mismatch: {T} vs {cfg.sequence_length}"

    use_2d = cfg.alpha_s > 0.0
    if use_2d and (model_2d is None or diffusion_2d is None or norm_params is None):
        raise ValueError("alpha_s > 0 requires model_2d, diffusion_2d, and norm_params")

    # Move schedule tensors to device.
    alphas_cumprod = torch.from_numpy(diffusion_1d.alphas_cumprod).to(device).float()

    # Reproducible init noise AND DDIM noise — alpha_s=0 equivalence requires
    # that the RNG stream consumed during sampling is independent of whether
    # the 2D branch is present. We seed both the CPU `Generator` used for the
    # initial noise and the global torch RNG used inside DDIM.
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x_t = torch.randn((B, H, W, T), generator=gen).to(device)
    torch.manual_seed(seed + 1)

    counts = counts_binned.to(device)

    timesteps = np.linspace(diffusion_1d.num_timesteps - 1, 0,
                            cfg.num_sampling_steps, dtype=int)
    it = enumerate(timesteps)
    if verbose:
        it = tqdm(list(it), total=len(timesteps), desc=f"factored(α_s={cfg.alpha_s})")

    counts_flat = counts.permute(0, 3, 1, 2).reshape(B * T, H, W)   # unused; kept for clarity
    # Flat views reused every step:
    # 1D layout: [B*H*W, 1, T]; 2D layout: [B*T, H, W]

    for step_idx, t in it:
        t_int = int(t)
        alpha_bar_t = alphas_cumprod[t_int]
        sqrt_ab = torch.sqrt(alpha_bar_t)
        sqrt_1mab = torch.sqrt(1.0 - alpha_bar_t)

        # --- 1D branch -----------------------------------------------------
        x_t_1d = x_t.reshape(B * H * W, 1, T)
        with torch.no_grad():
            eps_1d = _chunked_forward_1d(
                model_1d, x_t_1d, t_int, chunk=cfg.chunk_1d, device=device,
            )
            x0_1d_flat = (x_t_1d - sqrt_1mab * eps_1d) / (sqrt_ab + 1e-8)
        x0_temporal = x0_1d_flat.reshape(B, H, W, T)

        # --- 2D branch (skipped entirely when alpha_s == 0) ----------------
        if use_2d:
            x_t_2d = x_t.permute(0, 3, 1, 2).reshape(B * T, H, W)
            with torch.no_grad():
                x0_2d_flat = _chunked_denoise_2d(
                    model_2d, diffusion_2d, norm_params, x_t_2d, t_int,
                    chunk=cfg.chunk_2d,
                )
            x0_spatial = x0_2d_flat.reshape(B, T, H, W).permute(0, 2, 3, 1)
            x0_combined = cfg.alpha_s * x0_spatial + (1.0 - cfg.alpha_s) * x0_temporal
        else:
            x0_combined = x0_temporal

        # --- Data step (on binned counts, shared across batch) -------------
        x0_flat = x0_combined.reshape(B * H * W, 1, T)
        counts_1d = counts.reshape(B * H * W, 1, T)
        sigma_t_bar = float(sqrt_1mab.item())
        rho_t = torch.tensor(cfg.lambda_data / (sigma_t_bar ** 2 + 1e-8), device=device)
        x0_hat_flat = anscombe_data_step(
            x0_flat, counts_1d,
            bin_size=float(bin_size), dt_spad=float(dt_spad),
            rho_t=rho_t, sigma_t_bar=sigma_t_bar,
            n_iter=cfg.pp_iters, lr_scale=cfg.pp_lr_scale,
        )
        x0_hat = x0_hat_flat.reshape(B, H, W, T)

        # --- DDIM update ---------------------------------------------------
        if step_idx < len(timesteps) - 1:
            t_prev = int(timesteps[step_idx + 1])
            ab_prev = alphas_cumprod[t_prev]
            sqrt_ab_p = torch.sqrt(ab_prev)
            eps_hat = (x_t - sqrt_ab * x0_hat) / (sqrt_1mab + 1e-8)
            sigma_t = cfg.eta * torch.sqrt(
                (1.0 - ab_prev) / (1.0 - alpha_bar_t)
                * (1.0 - alpha_bar_t / ab_prev)
            )
            dir_xt = torch.sqrt(torch.clamp(1.0 - ab_prev - sigma_t ** 2, min=0.0)) * eps_hat
            if cfg.eta > 0:
                # Match the 1D-only DiffPIR noise shape so alpha_s=0 matches exactly.
                noise = torch.randn_like(x_t)
            else:
                noise = 0.0
            x_t = sqrt_ab_p * x0_hat + dir_xt + sigma_t * noise
        else:
            x_t = x0_hat

    return x_t.detach()
