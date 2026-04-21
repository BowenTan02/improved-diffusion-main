"""Factored DiffPIR sampler: 1D temporal prior + 2D spatial prior via convex
Tweedie combination.

The forward observation model is the same as the validated 1D pipeline — per
pixel SPAD binary detections are simulated, then binned to `sequence_length`
time bins and used as a Binomial (per-SPAD-frame Bernoulli) likelihood in the
data subproblem. Flux is parameterised in log1p space to match the 1D model's
training transform (`phi = expm1(x).clamp(min=0)`).

At every reverse diffusion step we compute TWO Tweedie x0 estimates:
  - x0_temporal: the 1D UNet applied per pixel across the time axis
  - x0_spatial:  the 2D UNet applied per time-slice across the spatial axes
                 (via spatial_prior.denoise_frame_2d)
and blend them with a user-controlled weight:
    x0_combined = alpha_s * x0_spatial + (1 - alpha_s) * x0_temporal
The data subproblem and the DDIM update then proceed as in vanilla DiffPIR.

When alpha_s == 0.0 no 2D forward pass is performed, and the loop reduces
*exactly* to 1D-only DiffPIR (see tests/test_factored_alpha0_equivalence.py).

Primary tensor layout: `x_t` is kept as [B, H, W, T] log1p-flux. Reshapes into
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
    frame (one global scaling — spatial brightness contrast is preserved).

    Returns
    -------
    binary : uint8 [B, H, W, n_spad_frames]
    flux_gt : float32 [B, H, W, n_spad_frames]   ground-truth flux at SPAD rate
    """
    if rng is None:
        rng = np.random.default_rng(0)

    B, H, W, T_gt = flux_cube.shape
    flux = flux_cube.astype(np.float64)

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
    N_t = flux_gt * dt_frame

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
# Poisson (Binomial) data step — matches the validated 1D pipeline.
# Uses `phi = expm1(x).clamp(min=0)` because x is log1p-flux (the 1D model's
# training transform). Anscombe VST intentionally dropped.
# ---------------------------------------------------------------------------

def poisson_data_step(
    x0_pred: torch.Tensor,          # [N, 1, T] log1p-flux
    counts: torch.Tensor,           # [N, 1, T] photon counts per bin
    *,
    bin_size: float,                # SPAD frames per bin
    dt_spad: float,                 # seconds per SPAD frame
    rho_t: torch.Tensor,            # scalar proximal weight
    sigma_t_bar: float,             # sqrt(1 - alpha_bar_t); unused here, kept for API parity
    n_iter: int = 5,
    lr_scale: float = 0.5,
    dark_count: float = 0.0,
    total_count_weight: float = 1e-3,
    T_exp: float = 1.0,
) -> torch.Tensor:
    """Proximal-gradient Binomial likelihood data step.

    Each bin covers `bin_size` SPAD frames of duration `dt_spad`. Per-frame
    detection probability `p = 1 - exp(-(N + dark_count))` with
    `N = phi * dt_spad` and `phi = expm1(x).clamp(min=0)` (linear flux — `x`
    is log1p-flux, matching the training-time transform of the 1D model).

    The bin count is modelled as Binomial(bin_size, p). We minimise
        NLL + total_count_weight · (∫phi dt − N_obs)² + (rho_t / 2) · ||x − x0||²
    with a few proximal-gradient steps.
    """
    x = x0_pred.detach().clone().clamp(-10.0, 15.0).requires_grad_(True)
    T = x.shape[-1]

    # Total-count anchor: observed detections -> per-frame mean -> expected total.
    total_det = counts.sum(-1).mean().item()
    total_frames = float(bin_size) * T
    det_rate = float(np.clip(total_det / (total_frames + 1e-8), 1e-8, 1.0 - 1e-3))
    N_per_frame = -np.log(1.0 - det_rate)
    mean_flux_target = max((N_per_frame - dark_count) / (dt_spad + 1e-10), 1.0)
    N_obs = mean_flux_target * T_exp

    for _ in range(n_iter):
        phi = torch.expm1(x.clamp(-15.0, 10.0)).clamp(min=0.0)
        N_k = phi * dt_spad + dark_count
        p_k = (1.0 - torch.exp(-N_k)).clamp(1e-8, 1.0 - 1e-8)
        nll = -(
            counts * torch.log(p_k)
            + (bin_size - counts) * torch.log(1.0 - p_k)
        ).sum(-1).mean()

        flux_integral = (phi * dt_spad * bin_size).sum(-1)
        count_penalty = total_count_weight * (
            (flux_integral - N_obs) ** 2 / (N_obs + 1e-8)
        ).mean()

        prox = 0.5 * rho_t * ((x - x0_pred) ** 2).sum(dim=[1, 2]).mean()
        loss = nll + count_penalty + prox

        grad = torch.autograd.grad(loss, x)[0]
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        with torch.no_grad():
            # Lipschitz-ish step: dominated by bin_size · N_k^2 term.
            L_nll = (bin_size * (dt_spad * phi) ** 2 * torch.exp(-N_k)).max()
            L_nll = torch.nan_to_num(L_nll, nan=0.0, posinf=1e6, neginf=0.0)
            step = lr_scale / (L_nll + rho_t + 1e-8)
            x = (x - step * grad).clamp(-10.0, 10.0)
        x = x.detach().requires_grad_(True)
    return x.detach()


# ---------------------------------------------------------------------------
# Factored sampler.
# ---------------------------------------------------------------------------

@dataclass
class FactoredConfig:
    sequence_length: int = 1024         # 1D UNet was trained at T=1024; keep fixed
    num_sampling_steps: int = 100
    eta: float = 0.85
    lambda_data: float = 1.0
    pp_iters: int = 5
    pp_lr_scale: float = 0.5
    alpha_s: float = 0.0
    chunk_1d: int = 4096                # pixels per 1D forward chunk
    chunk_2d: int = 32                  # time-slices per 2D forward chunk

    # Noise-aligned 2D call: when True, the 2D UNet is invoked at t' such
    # that 1 − alpha_bar_{t'} = scale^2 · (1 − alpha_bar_t). Required whenever
    # the bridge scale != 1 (currently scale ≈ 0.217 for [0, 9.2103] -> [-1, +1]).
    align_2d_noise: bool = True

    # --- Speedups for the 2D branch -------------------------------------
    # Compute the 2D Tweedie estimate on every `frame_stride_2d`-th frame and
    # broadcast it to its neighbours (nearest). stride=1 is the original
    # behaviour; stride=4 gives a ~4x speedup on the 2D branch with little
    # quality loss because the 2D prior varies slowly along the time axis.
    frame_stride_2d: int = 1
    # Skip the 2D branch entirely when the *mapped* 2D step t' <= this cut-off.
    # At low t the 1D data step dominates and the 2D prior's x0 ≈ x_t contributes
    # little; skipping saves many of the expensive 256² forwards.
    t_prime_skip_below: int = 1

    # Data-step likelihood parameters.
    dark_count: float = 0.0
    total_count_weight: float = 1e-3
    T_exp: float = 1.0


def _chunked_forward_1d(model_1d, x_1d, t_tensor_scalar, *, chunk, device):
    """x_1d: [N, 1, T]. Returns eps: [N, 1, T]."""
    N = x_1d.shape[0]
    out = torch.empty_like(x_1d)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        tt = torch.full((e - s,), int(t_tensor_scalar), dtype=torch.long, device=device)
        out[s:e] = model_1d(x_1d[s:e], tt)
    return out


def _chunked_denoise_2d(model_2d, diffusion_2d, norm_params, x_2d_flat, t_scalar,
                         *, chunk, align_noise=False, alphas_cumprod_1d=None):
    """x_2d_flat: [N, H, W] in log1p-flux space. Returns x0_hat: [N, H, W]."""
    N = x_2d_flat.shape[0]
    out = torch.empty_like(x_2d_flat)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        out[s:e] = denoise_frame_2d(
            x_2d_flat[s:e], t=int(t_scalar),
            model_2d=model_2d, diffusion_2d=diffusion_2d,
            normalization_params=norm_params,
            align_noise=align_noise,
            alphas_cumprod_1d=alphas_cumprod_1d,
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
    bin_size: int = 1,
    dt_spad: float = 1e-5,
    device: Optional[torch.device] = None,
    seed: int = 0,
    verbose: bool = True,
):
    """Run factored DiffPIR on a cube; returns estimated log1p-flux [B, H, W, T]."""
    if device is None:
        device = counts_binned.device
    B, H, W, T = counts_binned.shape
    assert T == cfg.sequence_length, (
        f"seq mismatch: input T={T} vs cfg.sequence_length={cfg.sequence_length}"
    )

    use_2d = cfg.alpha_s > 0.0
    if use_2d and (model_2d is None or diffusion_2d is None or norm_params is None):
        raise ValueError("alpha_s > 0 requires model_2d, diffusion_2d, and norm_params")

    # The t-shift handles mismatched schedules correctly as long as both
    # `alphas_cumprod` arrays are monotone in `1 - alpha_bar`. We warn — not
    # assert — when they differ, because the correct cross-schedule behaviour
    # is to searchsorted on the 2D schedule inside `t_prime_from_t`. Setting
    # `align_2d_noise=False` with mismatched schedules is a silent bug — keep
    # the hard stop for that combination only.
    if use_2d:
        schedules_match = (
            diffusion_1d.num_timesteps == diffusion_2d.num_timesteps
            and np.allclose(
                diffusion_1d.alphas_cumprod, diffusion_2d.alphas_cumprod, rtol=1e-6,
            )
        )
        if not schedules_match and not cfg.align_2d_noise:
            raise ValueError(
                "1D and 2D schedules differ "
                f"(1D: T={diffusion_1d.num_timesteps}, "
                f"2D: T={diffusion_2d.num_timesteps}) "
                "but `align_2d_noise=False`. Enable `align_2d_noise=True` — "
                "the t-shift handles cross-schedule variance matching "
                "(e.g. 1D linear T=1000 vs 2D cosine T=4000)."
            )

    alphas_cumprod = torch.from_numpy(diffusion_1d.alphas_cumprod).to(device).float()

    gen = torch.Generator(device="cpu").manual_seed(seed)
    x_t = torch.randn((B, H, W, T), generator=gen).to(device)
    torch.manual_seed(seed + 1)

    counts = counts_binned.to(device)

    timesteps = np.linspace(diffusion_1d.num_timesteps - 1, 0,
                            cfg.num_sampling_steps, dtype=int)
    it = enumerate(timesteps)
    if verbose:
        it = tqdm(list(it), total=len(timesteps), desc=f"factored(α_s={cfg.alpha_s})")

    # Precompute which frame indices get an actual 2D call.
    stride = max(1, int(cfg.frame_stride_2d))
    frame_idx = torch.arange(T, device=device)
    if stride > 1:
        anchor_idx = torch.arange(0, T, stride, device=device)
        # Nearest anchor for each frame (left-biased).
        nearest = (frame_idx // stride).clamp(max=len(anchor_idx) - 1)
    else:
        anchor_idx = frame_idx
        nearest = frame_idx

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
            # Compute t' for the noise-aligned call so we can also gate on it.
            # Search on the *2D* schedule (which may differ from 1D's).
            if cfg.align_2d_noise:
                from spatial_prior import t_prime_from_t
                t_prime = t_prime_from_t(
                    t_int,
                    norm_params.scale,
                    alphas_cumprod_2d=diffusion_2d.alphas_cumprod,
                    alphas_cumprod_1d=diffusion_1d.alphas_cumprod,
                )
            else:
                t_prime = t_int

            if t_prime < cfg.t_prime_skip_below:
                # 2D would be called at t' ≈ 0 — Tweedie ≈ x_t, little benefit.
                x0_combined = x0_temporal
            else:
                # Only denoise anchor frames; broadcast to neighbours via `nearest`.
                x_t_bht = x_t.permute(0, 3, 1, 2)                       # [B, T, H, W]
                x_t_anchors = x_t_bht.index_select(1, anchor_idx)        # [B, T_a, H, W]
                x_t_2d = x_t_anchors.reshape(-1, H, W)
                with torch.no_grad():
                    x0_2d_flat = _chunked_denoise_2d(
                        model_2d, diffusion_2d, norm_params, x_t_2d, t_int,
                        chunk=cfg.chunk_2d,
                        align_noise=cfg.align_2d_noise,
                        alphas_cumprod_1d=diffusion_1d.alphas_cumprod,
                    )
                x0_2d_anchors = x0_2d_flat.reshape(B, len(anchor_idx), H, W)
                # Nearest-neighbour upsample back to T frames.
                x0_spatial = x0_2d_anchors.index_select(1, nearest).permute(0, 2, 3, 1)
                x0_combined = cfg.alpha_s * x0_spatial + (1.0 - cfg.alpha_s) * x0_temporal
        else:
            x0_combined = x0_temporal

        # --- Data step (on binned counts, shared across batch) -------------
        x0_flat = x0_combined.reshape(B * H * W, 1, T)
        counts_1d = counts.reshape(B * H * W, 1, T)
        sigma_t_bar = float(sqrt_1mab.item())
        rho_t = torch.tensor(cfg.lambda_data / (sigma_t_bar ** 2 + 1e-8), device=device)
        x0_hat_flat = poisson_data_step(
            x0_flat, counts_1d,
            bin_size=float(bin_size), dt_spad=float(dt_spad),
            rho_t=rho_t, sigma_t_bar=sigma_t_bar,
            n_iter=cfg.pp_iters, lr_scale=cfg.pp_lr_scale,
            dark_count=cfg.dark_count,
            total_count_weight=cfg.total_count_weight,
            T_exp=cfg.T_exp,
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
                noise = torch.randn_like(x_t)
            else:
                noise = 0.0
            x_t = sqrt_ab_p * x0_hat + dir_xt + sigma_t * noise
        else:
            x_t = x0_hat

    return x_t.detach()
