"""
Video inference: recover per-pixel photon flux from 1D SPAD observations.

Pipeline
--------
Input : a log_flux .pt tensor of shape [T_src, H, W]  (log-flux video).

Stage A  (simulation):
  1. exp(log_flux) -> linear flux, normalize per-pixel to [0, flux_peak].
  2. RIFE x (2**rife_exp) temporal interpolation.
  3. Linear x (linear_factor) temporal interpolation.
  4. -> hi-FPS flux  [T_hi, H, W]      where T_hi ~= fps * t_total
  5. SPAD binary sampling (Poisson)    [T_hi, H, W]  uint8
  6. Save raw binary bit-packed to .bin (+ sidecar .json with shape/fps).

Stage B  (reconstruction):
  7. Bin binary [T_hi,H,W] -> counts [SEQ_LEN,H,W] for the diffusion model.
  8. Bin flux   [T_hi,H,W] -> integrated GT [SEQ_LEN,H,W] (avg-in-bin then
     per-pixel renormalize to flux_peak). Save as .npy.
  9. Save binned-binary MP4 (each SEQ_LEN frame clamped to {0,1}*255).
  10. DiffPIR with binomial-likelihood data step -> log-flux_hat [SEQ_LEN,H,W].
  11. Convert to linear flux. Save MP4 + .npy.

Stage C  (metrics):
  12. PSNR / SSIM / LPIPS per SEQ_LEN frame, recon vs integrated-GT.
  13. Dump metrics.json.

Example
-------
python video_inference.py \
  --input ./log_flux.pt \
  --output_dir ./outputs/vid_run_01 \
  --target_ppp 0.05 \
  --fps 100000 \
  --rife_exp 4 --linear_factor 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import gc
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# =============================================================================
# MODEL CONFIG   (edit here when switching checkpoints / architectures)
# =============================================================================
MODEL_CHECKPOINT      = "./models_1024/model200000.pt"
SEQUENCE_LENGTH       = 1024
NUM_DIFFUSION_STEPS   = 1000
CHANNEL_MULT          = "1,2,3,4"
ATTENTION_RESOLUTIONS = "256, 128"
NOISE_SCHEDULE        = "linear"
NUM_CHANNELS          = 64
LEARN_SIGMA           = False
# =============================================================================


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def maybe_add_to_syspath(path: Optional[str]) -> None:
    if path and os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)


def flux_from_x(x: torch.Tensor, x_param: str) -> torch.Tensor:
    if x_param == "log":
        return torch.exp(x)
    if x_param == "log1p":
        return torch.expm1(x)
    raise ValueError(f"Unknown x_param={x_param!r}")


def ensure_parent(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Diffusion model loading
# ---------------------------------------------------------------------------

def load_temporal_diffusion_model(
    *,
    checkpoint_path: str,
    sequence_length: int,
    num_channels: int,
    diffusion_steps: int,
    channel_mult: str,
    attention_resolutions: str,
    noise_schedule: str,
    learn_sigma: bool,
    device: torch.device,
) -> Tuple[torch.nn.Module, object]:
    from improved_diffusion.temporal_script_util import (
        temporal_model_and_diffusion_defaults,
        create_temporal_model_and_diffusion,
    )

    defaults = temporal_model_and_diffusion_defaults()
    defaults.update(
        {
            "sequence_length":       sequence_length,
            "num_channels":          num_channels,
            "diffusion_steps":       diffusion_steps,
            "channel_mult":          channel_mult,
            "attention_resolutions": attention_resolutions,
            "noise_schedule":        noise_schedule,
            "learn_sigma":           learn_sigma,
        }
    )
    model, diffusion = create_temporal_model_and_diffusion(**defaults)
    if not (checkpoint_path and os.path.exists(checkpoint_path)):
        raise FileNotFoundError(f"Checkpoint not found: `{checkpoint_path}`")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, diffusion


# ---------------------------------------------------------------------------
# RIFE  (adapted from generate_spad_dataset.py)
# ---------------------------------------------------------------------------

def load_rife_model(rife_dir: str, model_dir: str):
    """
    Import RIFE from `rife_dir` and load weights from `model_dir`.

    Tries RIFE_HDv2 -> RIFE_HDv3 -> RIFE_HD -> RIFE in that order.
    """
    if not os.path.isdir(rife_dir):
        raise FileNotFoundError(f"RIFE repo dir not found: {rife_dir}")
    if rife_dir not in sys.path:
        sys.path.insert(0, rife_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    model = None
    cwd = os.getcwd()
    try:
        os.chdir(rife_dir)
        try:
            try:
                from model.RIFE_HDv2 import Model
                model = Model(); model.load_model(model_dir, -1)
                print("Loaded RIFE HDv2.")
            except Exception:
                from train_log.RIFE_HDv3 import Model
                model = Model(); model.load_model(model_dir, -1)
                print("Loaded RIFE HDv3.")
        except Exception:
            try:
                from model.RIFE_HD import Model
                model = Model(); model.load_model(model_dir, -1)
                print("Loaded RIFE HD.")
            except Exception:
                from model.RIFE import Model
                model = Model(); model.load_model(model_dir, -1)
                print("Loaded RIFE ArXiv.")
    finally:
        os.chdir(cwd)

    model.eval()
    model.device()
    return model, device


def _rife_interpolate_rgb(
    model, device, frames_u8: np.ndarray, exp: int, scale: float = 1.0
) -> np.ndarray:
    """
    RIFE x (2**exp) on RGB uint8 frames [T, H, W, 3]. Returns uint8 [T_out, H, W, 3].
    T_out = (T - 1) * (2**exp) + 1.
    """
    T, H, W, _ = frames_u8.shape
    tmp = max(32, int(32 / scale))
    ph = ((H - 1) // tmp + 1) * tmp
    pw = ((W - 1) // tmp + 1) * tmp
    padding = (0, pw - W, 0, ph - H)

    def make_inference(I0_t, I1_t, n):
        if n == 0:
            return []
        middle = model.inference(I0_t, I1_t, scale)
        if n == 1:
            return [middle]
        a = make_inference(I0_t, middle, n=n // 2)
        b = make_inference(middle, I1_t, n=n // 2)
        return [*a, middle, *b] if n % 2 else [*a, *b]

    out = []
    I0_rgb = frames_u8[0].astype(np.float32) / 255.0
    I0 = torch.from_numpy(np.transpose(I0_rgb, (2, 0, 1))).unsqueeze(0).float().to(device)
    I0 = F.pad(I0, padding)
    out.append((I0[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)[:H, :W])

    for i in tqdm(range(T - 1), desc="RIFE"):
        I0_rgb = frames_u8[i].astype(np.float32) / 255.0
        I1_rgb = frames_u8[i + 1].astype(np.float32) / 255.0
        I0 = torch.from_numpy(np.transpose(I0_rgb, (2, 0, 1))).unsqueeze(0).float().to(device)
        I1 = torch.from_numpy(np.transpose(I1_rgb, (2, 0, 1))).unsqueeze(0).float().to(device)
        I0 = F.pad(I0, padding); I1 = F.pad(I1, padding)
        mids = make_inference(I0, I1, 2 ** exp - 1)
        for m in mids:
            out.append((m[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)[:H, :W])
        out.append((I1[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)[:H, :W])

    return np.asarray(out, dtype=np.uint8)


def rife_interpolate_flux(
    flux: np.ndarray,  # [T, H, W]  float, range roughly [0, flux_peak]
    flux_peak: float,
    rife_exp: int,
    rife_dir: str,
    rife_model_dir: str,
) -> np.ndarray:
    """
    Apply RIFE x (2**rife_exp) on a single-channel flux video by:
      flux / flux_peak -> [0,1] -> replicate to 3 channels -> RIFE -> mean -> rescale.

    Returns float32 flux [T_out, H, W].
    """
    if rife_exp <= 0:
        return flux.astype(np.float32, copy=False)

    T, H, W = flux.shape
    g = np.clip(flux / max(flux_peak, 1e-12), 0.0, 1.0)
    rgb_u8 = (g[..., None].repeat(3, axis=-1) * 255.0).astype(np.uint8)

    rife_model, rife_device = load_rife_model(rife_dir, rife_model_dir)
    try:
        rgb_out = _rife_interpolate_rgb(rife_model, rife_device, rgb_u8, exp=rife_exp)
    finally:
        del rife_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    g_out = rgb_out.astype(np.float32).mean(axis=-1) / 255.0  # [T_out, H, W]
    return (g_out * flux_peak).astype(np.float32)


def linear_interp_flux(flux: np.ndarray, factor: int, chunk_pairs: int = 256) -> np.ndarray:
    """
    Linearly interpolate along time: T -> 1 + (T-1)*factor.
    Vectorised with a chunked-in-pairs reduction so peak RAM stays bounded
    (chunk_pairs * factor * H * W floats at a time).
    """
    if factor <= 1:
        return flux.astype(np.float32, copy=False)
    T, H, W = flux.shape
    T_out = 1 + (T - 1) * factor
    out = np.empty((T_out, H, W), dtype=np.float32)
    out[0] = flux[0]

    alphas = (np.arange(1, factor + 1, dtype=np.float32) / factor).reshape(1, factor, 1, 1)
    one_minus = 1.0 - alphas
    pos = 1
    for p0 in tqdm(range(0, T - 1, chunk_pairs), desc="Linear interp"):
        p1 = min(p0 + chunk_pairs, T - 1)
        I0 = flux[p0:p1    ].astype(np.float32)[:, None]   # [m, 1, H, W]
        I1 = flux[p0 + 1:p1 + 1].astype(np.float32)[:, None]  # [m, 1, H, W]
        inter = (one_minus * I0 + alphas * I1).reshape(-1, H, W)  # [m*factor, H, W]
        n = inter.shape[0]
        out[pos:pos + n] = inter
        pos += n
    return out


# ---------------------------------------------------------------------------
# SPAD simulation at hi-FPS
# ---------------------------------------------------------------------------

def simulate_spad_binary(
    flux_hi: np.ndarray,  # [T, H, W] float, per-pixel max == flux_peak
    target_ppp: float,
    dark_count: float,
    flux_peak: float,
    rng: np.random.Generator,
    chunk_frames: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate SPAD binary detections frame-by-frame.

    Per pixel: I_norm(t) = flux(t)/flux_peak  in [0,1]
               a = target_ppp / mean_t(I_norm)        (per pixel)
               N(t) = a * I_norm(t) + dark_count
               binary(t) ~ Bernoulli(1 - exp(-N(t)))

    Returns
    -------
    binary     : uint8 [T, H, W]
    ppp_scale  : float32 [H, W]  =  a / flux_peak  (maps raw flux -> N)
    """
    T, H, W = flux_hi.shape
    I_norm = np.clip(flux_hi / max(flux_peak, 1e-12), 0.0, 1.0)

    I_mean = I_norm.mean(axis=0)                # [H, W]
    safe_mean = np.where(I_mean > 0, I_mean, 1.0)
    a = np.float32(target_ppp) / safe_mean.astype(np.float32)   # [H, W]
    ppp_scale = (a / max(flux_peak, 1e-12)).astype(np.float32)  # [H, W]

    binary = np.empty((T, H, W), dtype=np.uint8)
    for t0 in tqdm(range(0, T, chunk_frames), desc="SPAD sample"):
        t1 = min(t0 + chunk_frames, T)
        chunk = I_norm[t0:t1]
        N = a[None] * chunk + np.float32(dark_count)
        p = 1.0 - np.exp(-N)
        u = rng.random(chunk.shape, dtype=np.float32)
        binary[t0:t1] = (u < p).astype(np.uint8)
    return binary, ppp_scale


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_binary_bitpacked(binary_thw: np.ndarray, bin_path: str, t_total: float) -> None:
    """
    Save [T,H,W] uint8/bool binary as bit-packed raw bytes to `bin_path`.
    Also writes a sidecar `.json` with shape/dtype/fps so it can be loaded back.
    """
    ensure_parent(bin_path)
    data = binary_thw.astype(np.uint8, copy=False)
    T, H, W = data.shape
    packed = np.packbits(data.reshape(-1))
    packed.tofile(bin_path)

    sidecar = os.path.splitext(bin_path)[0] + ".json"
    with open(sidecar, "w") as f:
        json.dump(
            {
                "shape":   [int(T), int(H), int(W)],
                "order":   "T,H,W (C-contiguous)",
                "dtype":   "uint8 (packed via np.packbits over flat T*H*W)",
                "bits":    int(T * H * W),
                "t_total": float(t_total),
                "fps":     float(T / t_total),
            },
            f,
            indent=2,
        )
    print(f"  wrote {bin_path}  ({packed.nbytes / 1e6:.1f} MB, {T}x{H}x{W})")


def save_video_mp4(frames_u8: np.ndarray, path: str, fps: int = 30) -> None:
    """Write [T,H,W] uint8 grayscale frames as MP4."""
    ensure_parent(path)
    T, H, W = frames_u8.shape
    try:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, fps, (W, H), isColor=False)
        if not writer.isOpened():
            raise RuntimeError("cv2.VideoWriter failed to open")
        for t in range(T):
            writer.write(frames_u8[t])
        writer.release()
        print(f"  wrote {path} via OpenCV  ({T} frames @ {fps} fps)")
        return
    except Exception as e:
        print(f"  OpenCV unavailable ({e}); falling back to imageio.")
    import imageio.v3 as iio
    rgb = np.stack([frames_u8] * 3, axis=-1)
    iio.imwrite(path, rgb, fps=fps, codec="libx264")
    print(f"  wrote {path} via imageio  ({T} frames @ {fps} fps)")


def flux_to_u8_frames_global(flux_thw: np.ndarray) -> np.ndarray:
    """Global min-max scaling to uint8 for visualisation."""
    lo, hi = float(flux_thw.min()), float(flux_thw.max())
    if hi - lo < 1e-12:
        return np.zeros_like(flux_thw, dtype=np.uint8)
    return ((flux_thw - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

def bin_along_time(arr_thw: np.ndarray, seq_len: int, reduce: str) -> np.ndarray:
    """
    Bin a [T,H,W] array into [seq_len,H,W] along time using `reduce` in
    {'sum','mean'}. If T % seq_len != 0, uses np.linspace bin edges.
    """
    T = arr_thw.shape[0]
    edges = np.linspace(0, T, seq_len + 1, dtype=np.int64)
    bin_sizes = np.diff(edges).astype(np.float32)  # [seq_len]

    out = np.empty((seq_len,) + arr_thw.shape[1:], dtype=np.float32)
    for i in range(seq_len):
        s, e = int(edges[i]), int(edges[i + 1])
        chunk = arr_thw[s:e]
        if chunk.shape[0] == 0:
            out[i] = 0.0
            continue
        if reduce == "sum":
            out[i] = chunk.sum(axis=0, dtype=np.float32)
        elif reduce == "mean":
            out[i] = chunk.mean(axis=0, dtype=np.float32)
        else:
            raise ValueError(reduce)
    return out, bin_sizes


def integrated_gt_flux(
    flux_hi: np.ndarray,  # [T, H, W]
    seq_len: int,
    flux_peak: float,
) -> np.ndarray:
    """
    Average hi-FPS flux into [seq_len, H, W] bins, then renormalize per pixel
    so each pixel's max over time equals flux_peak. (Averaging contracts range;
    renormalizing keeps the integrated video on the same scale the diffusion
    model was trained on.)
    """
    gt, _ = bin_along_time(flux_hi, seq_len=seq_len, reduce="mean")  # [S,H,W] float32
    mx = gt.max(axis=0, keepdims=True)  # [1,H,W]
    mx = np.where(mx > 0, mx, 1.0).astype(np.float32)
    return (gt / mx * np.float32(flux_peak)).astype(np.float32)


# ---------------------------------------------------------------------------
# DiffPIR core  (analytical gradients, identical math to batch_inference)
# ---------------------------------------------------------------------------

@torch.no_grad()
def spad_data_step_binomial(
    x0_pred: torch.Tensor,         # [B,1,L]
    bin_counts: torch.Tensor,      # [B,L]
    bin_sizes: torch.Tensor,       # [B,L]
    rho_t: torch.Tensor,
    *,
    ppp_scale,                     # scalar / [B] / [B,1]
    dark_count: float = 7.74e-4,
    n_iter: int = 10,
    lr_scale: float = 0.75,
    total_count_weight: float = 0.001,
    t_total: float = 1.0,
    x_param: str = "log",
) -> torch.Tensor:
    B, _, L = x0_pred.shape
    x = x0_pred.clone().clamp(-10, 15)

    ppp_t = torch.as_tensor(ppp_scale, device=x.device, dtype=x.dtype)
    if ppp_t.ndim == 0:
        ppp_t = ppp_t.expand(B).unsqueeze(1)
    elif ppp_t.ndim == 1:
        ppp_t = ppp_t.unsqueeze(1)

    dt = t_total / L
    total_det = bin_counts.sum(-1)
    total_frm = bin_sizes.sum(-1)
    det_rate = torch.clamp(total_det / (total_frm + 1e-8), 1e-8, 1 - 1e-3)
    N_per_frame = -torch.log(1 - det_rate)
    mean_flux_target = torch.clamp(
        (N_per_frame - dark_count) / (ppp_t.squeeze(1) + 1e-10), min=1.0
    )
    N_obs = mean_flux_target * t_total

    for _ in range(n_iter):
        x_sq = x.squeeze(1)
        if x_param == "log":
            phi = torch.exp(x_sq); dphi_dx = phi
        else:
            phi = torch.expm1(x_sq); dphi_dx = phi + 1.0
        phi = torch.clamp(phi, min=0.0)

        N_k = ppp_t * phi + dark_count
        exp_neg_Nk = torch.exp(-N_k)
        p_k = torch.clamp(1.0 - exp_neg_Nk, 1e-8, 1 - 1e-8)

        grad_nll_phi = (-bin_counts * exp_neg_Nk / p_k
                        + (bin_sizes - bin_counts)) * ppp_t
        grad_nll_x = grad_nll_phi * dphi_dx

        flux_integral = (phi * dt).sum(-1)
        residual = flux_integral - N_obs
        grad_count_x = (
            total_count_weight * 2.0 * residual / (N_obs + 1e-8)
        ).unsqueeze(1) * dt * dphi_dx

        grad_prox_x = rho_t * (x_sq - x0_pred.squeeze(1))

        grad = (grad_nll_x + grad_count_x + grad_prox_x).unsqueeze(1)
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

        L_nll = (bin_sizes * (ppp_t * phi) ** 2 * exp_neg_Nk).max(dim=-1).values
        L_nll = torch.nan_to_num(L_nll, nan=0.0, posinf=1e6, neginf=0.0)
        step = lr_scale / (L_nll + rho_t + 1e-8)
        step = step.view(B, 1, 1)
        x = (x - step * grad).clamp(-10, 10)

    return x


@torch.no_grad()
def sample_diffpir(
    *,
    model,
    diffusion,
    bin_counts: torch.Tensor,
    bin_sizes: torch.Tensor,
    ppp_scale,
    dark_count: float,
    num_steps: int,
    diffusion_steps: int,
    lambda_data: float,
    eta: float,
    pp_solver_iters: int,
    pp_lr_scale: float,
    t_total: float,
    x_param: str,
    sequence_length: int,
    device: torch.device,
    show_progress: bool = False,
    use_amp: bool = False,
) -> torch.Tensor:
    B = bin_counts.shape[0]
    x_t = torch.randn((B, 1, sequence_length), device=device)
    alphas_cumprod = torch.from_numpy(diffusion.alphas_cumprod).to(device)
    timesteps = np.linspace(diffusion_steps - 1, 0, num_steps, dtype=int)
    iterator = (
        tqdm(list(enumerate(timesteps)), total=len(timesteps), desc="DiffPIR")
        if show_progress else enumerate(timesteps)
    )

    for step_idx, t in iterator:
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                model_output = model(x_t, t_tensor)
            model_output = model_output.float()
        else:
            model_output = model(x_t, t_tensor)
        model_output = torch.nan_to_num(model_output, nan=0.0, posinf=0.0, neginf=0.0)

        abar = alphas_cumprod[t]
        sab = torch.sqrt(abar)
        somb = torch.sqrt(1 - abar)

        x0_pred = (x_t - somb * model_output) / sab
        x0_pred = torch.nan_to_num(x0_pred, nan=0.0, posinf=15.0, neginf=-10.0).clamp(-10, 15)

        sigma_k_t = somb / (sab + 1e-8)
        rho_t = lambda_data / (sigma_k_t ** 2 + 1e-8)

        x0_hat = spad_data_step_binomial(
            x0_pred, bin_counts, bin_sizes, rho_t,
            ppp_scale=ppp_scale, dark_count=dark_count,
            n_iter=pp_solver_iters, lr_scale=pp_lr_scale,
            t_total=t_total, x_param=x_param,
        )
        x0_hat = torch.nan_to_num(x0_hat, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10, 15)

        if step_idx < len(timesteps) - 1:
            t_prev = timesteps[step_idx + 1]
            abar_prev = alphas_cumprod[t_prev]
            sab_prev = torch.sqrt(abar_prev)
            eps_hat = (x_t - sab * x0_hat) / (somb + 1e-8)
            sigma_t = eta * torch.sqrt(torch.clamp(
                (1 - abar_prev) / (1 - abar) * (1 - abar / abar_prev), min=0.0))
            dir_xt = torch.sqrt(torch.clamp(
                1 - abar_prev - sigma_t ** 2, min=0.0)) * eps_hat
            noise = torch.randn_like(x_t) if eta > 0 else 0
            x_t = sab_prev * x0_hat + dir_xt + sigma_t * noise
            x_t = torch.nan_to_num(x_t, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            x_t = x0_hat
    return x_t


def chunked_indices(n: int, batch_size: int) -> Iterable[slice]:
    for i in range(0, n, batch_size):
        yield slice(i, min(i + batch_size, n))


# ---------------------------------------------------------------------------
# Metrics:  PSNR / SSIM / LPIPS
# ---------------------------------------------------------------------------

def compute_video_metrics(
    recon_thw: np.ndarray,  # [T, H, W]  float
    gt_thw:    np.ndarray,  # [T, H, W]  float
    device: torch.device,
) -> dict:
    """
    PSNR / SSIM / LPIPS per-frame and aggregate (mean over frames).
    Both inputs are sanitized (NaN/Inf removed) then min-max normalized to
    [0,1] using the GT's own range so the comparison is well-defined
    regardless of absolute scale.
    """
    assert recon_thw.shape == gt_thw.shape
    T = recon_thw.shape[0]

    # Sanitize — np.clip does NOT remove NaN, and np.exp can produce ±Inf
    # from extreme log-flux values. Without this, PSNR/SSIM collapse to NaN.
    gt64    = np.nan_to_num(gt_thw.astype(np.float64, copy=False),
                            nan=0.0, posinf=0.0, neginf=0.0)
    recon64 = np.nan_to_num(recon_thw.astype(np.float64, copy=False),
                            nan=0.0, posinf=0.0, neginf=0.0)
    n_bad = int(np.sum(~np.isfinite(recon_thw)))
    if n_bad:
        print(f"  [metrics] sanitized {n_bad} non-finite recon entries "
              f"({100 * n_bad / recon_thw.size:.4f}%).")

    lo = float(gt64.min()); hi = float(gt64.max())
    denom = max(hi - lo, 1e-12)
    gt01    = np.clip((gt64    - lo) / denom, 0.0, 1.0).astype(np.float32)
    recon01 = np.clip((recon64 - lo) / denom, 0.0, 1.0).astype(np.float32)

    # -- PSNR / SSIM via skimage ----------------------------------------------
    psnr_fn = ssim_fn = None
    try:
        from skimage.metrics import peak_signal_noise_ratio as _psnr
        from skimage.metrics import structural_similarity as _ssim
        psnr_fn, ssim_fn = _psnr, _ssim
    except Exception as e:
        print(f"  [metrics] skimage import failed ({e!r}); PSNR/SSIM disabled.")

    psnr = np.full(T, np.nan, dtype=np.float64)
    ssim = np.full(T, np.nan, dtype=np.float64)
    if psnr_fn is not None:
        # SSIM default win_size=7 requires H,W >= 7; pick safe odd win_size.
        H, W = gt01.shape[1:]
        win = min(7, H if H % 2 == 1 else H - 1, W if W % 2 == 1 else W - 1)
        win = max(3, win - (1 - win % 2))  # force odd, >= 3
        n_fail = 0; first_err = None
        for t in range(T):
            try:
                psnr[t] = psnr_fn(gt01[t], recon01[t], data_range=1.0)
            except Exception as e:
                n_fail += 1
                if first_err is None:
                    first_err = repr(e)
            try:
                ssim[t] = ssim_fn(gt01[t], recon01[t], data_range=1.0, win_size=win)
            except Exception as e:
                n_fail += 1
                if first_err is None:
                    first_err = repr(e)
        if n_fail:
            print(f"  [metrics] {n_fail} per-frame PSNR/SSIM call(s) failed "
                  f"(first error: {first_err}).")

    # -- LPIPS via lpips package (AlexNet) ------------------------------------
    lpips_vals: np.ndarray
    try:
        import lpips
        net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        x = torch.from_numpy(recon01).to(device)  # [T,H,W]
        y = torch.from_numpy(gt01).to(device)
        # [T,1,H,W] -> [T,3,H,W], map [0,1] -> [-1,1]
        x = (x.unsqueeze(1).repeat(1, 3, 1, 1) * 2.0) - 1.0
        y = (y.unsqueeze(1).repeat(1, 3, 1, 1) * 2.0) - 1.0
        chunk = 16
        out = []
        with torch.no_grad():
            for t0 in range(0, T, chunk):
                out.append(net(x[t0:t0 + chunk], y[t0:t0 + chunk]).flatten().cpu().numpy())
        lpips_vals = np.concatenate(out).astype(np.float64)
        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  [metrics] LPIPS unavailable ({e}); LPIPS set to NaN.")
        lpips_vals = np.full(T, np.nan)

    return {
        "psnr_per_frame":  psnr.tolist(),
        "ssim_per_frame":  ssim.tolist(),
        "lpips_per_frame": lpips_vals.tolist(),
        "psnr_mean":  float(np.nanmean(psnr)),
        "ssim_mean":  float(np.nanmean(ssim)),
        "lpips_mean": float(np.nanmean(lpips_vals)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Video DiffPIR reconstruction with RIFE+linear interpolation."
    )
    # I/O
    ap.add_argument("--input",       type=str, required=True,
                    help="Path to log_flux .pt file of shape [T_src, H, W]")
    ap.add_argument("--output_dir",  type=str, required=True)
    ap.add_argument("--improved_diffusion_root", type=str, default=None)

    # Model (defaults come from top-of-file CONFIG; CLI flags override)
    ap.add_argument("--checkpoint",            type=str, default=MODEL_CHECKPOINT)
    ap.add_argument("--sequence_length",       type=int, default=SEQUENCE_LENGTH)
    ap.add_argument("--diffusion_steps",       type=int, default=NUM_DIFFUSION_STEPS)
    ap.add_argument("--num_channels",          type=int, default=NUM_CHANNELS)
    ap.add_argument("--channel_mult",          type=str, default=CHANNEL_MULT)
    ap.add_argument("--attention_resolutions", type=str, default=ATTENTION_RESOLUTIONS)
    ap.add_argument("--noise_schedule",        type=str, default=NOISE_SCHEDULE)
    ap.add_argument("--learn_sigma",           action="store_true", default=LEARN_SIGMA)

    # Sampling
    ap.add_argument("--sampling_steps",   type=int,   default=100)
    ap.add_argument("--lambda_data",      type=float, default=1.0)
    ap.add_argument("--eta",              type=float, default=0.85)
    ap.add_argument("--pp_solver_iters",  type=int,   default=10)
    ap.add_argument("--pp_lr_scale",      type=float, default=0.75)
    ap.add_argument("--x_param",          type=str,   default="log",
                    choices=["log", "log1p"])

    # Interpolation / SPAD
    ap.add_argument("--rife_dir",        type=str, default="./ECCV2022-RIFE",
                    help="Path to the RIFE repo (contains `model/` and `train_log/`).")
    ap.add_argument("--rife_model_dir",  type=str, default="./ECCV2022-RIFE/train_log")
    ap.add_argument("--rife_exp",        type=int, default=4,
                    help="RIFE multiplier = 2**rife_exp (set 0 to skip RIFE).")
    ap.add_argument("--linear_factor",   type=int, default=4,
                    help="Linear interpolation multiplier applied after RIFE (set 1 to skip).")
    ap.add_argument("--fps",             type=int, default=None,
                    help="If set, warns when RIFE*linear does not hit this FPS. "
                         "FPS is determined by --rife_exp and --linear_factor; "
                         "this flag is for annotation only.")
    ap.add_argument("--target_ppp",      type=float, default=0.05)
    ap.add_argument("--dark_count",      type=float, default=7.74e-4)
    ap.add_argument("--t_total",         type=float, default=1.0)
    ap.add_argument("--flux_peak",       type=float, default=10000.0)
    ap.add_argument("--no_normalize_flux", action="store_true")

    # Runtime
    ap.add_argument("--infer_batch_size", type=int, default=1024,
                    help="Pixels per DiffPIR batch. Larger = faster if VRAM allows. "
                         "Try 512 if OOM, 2048/4096 on an A100.")
    ap.add_argument("--seed",             type=int, default=42)
    ap.add_argument("--use_amp",          dest="use_amp", action="store_true",
                    default=True,
                    help="Mixed-precision model forward (default: on for CUDA).")
    ap.add_argument("--no_amp",           dest="use_amp", action="store_false",
                    help="Disable AMP and run the model in full fp32.")
    ap.add_argument("--compile_model",    action="store_true")
    ap.add_argument("--mp4_fps",          type=int, default=30)
    ap.add_argument("--save_raw_binary",  action="store_true", default=True,
                    help="Save hi-FPS binary as bit-packed .bin (default: on).")
    ap.add_argument("--no_save_raw_binary", dest="save_raw_binary", action="store_false")

    args = ap.parse_args()
    maybe_add_to_syspath(args.improved_diffusion_root)
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    normalize_flux = not args.no_normalize_flux
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- 1) Load log_flux  [T_src, H, W] ------------------------------------
    raw = torch.load(args.input, map_location="cpu")
    if not isinstance(raw, torch.Tensor) or raw.ndim != 3:
        raise ValueError(f"Expected [T,H,W] tensor in {args.input}, got "
                         f"{type(raw)} / {getattr(raw, 'shape', None)}")
    T_src, H, W = int(raw.shape[0]), int(raw.shape[1]), int(raw.shape[2])
    print(f"Input: {args.input}   shape=({T_src},{H},{W})")

    # log_flux -> linear flux; per-pixel normalize to flux_peak (matches training)
    flux_src = np.exp(raw.numpy().astype(np.float64))  # [T_src, H, W]
    if normalize_flux:
        mx = flux_src.max(axis=0, keepdims=True)
        mx = np.where(mx > 0, mx, 1.0)
        flux_src = (flux_src / mx) * float(args.flux_peak)
    flux_src = flux_src.astype(np.float32)
    print(f"Source flux range: [{flux_src.min():.2f}, {flux_src.max():.2f}]")

    # --- 2) RIFE interpolation ---------------------------------------------
    if args.rife_exp > 0:
        print(f"\nRIFE x {2**args.rife_exp} (exp={args.rife_exp}) ...")
        flux_rife = rife_interpolate_flux(
            flux_src, flux_peak=float(args.flux_peak),
            rife_exp=args.rife_exp,
            rife_dir=args.rife_dir, rife_model_dir=args.rife_model_dir,
        )
    else:
        flux_rife = flux_src
    print(f"After RIFE:   T = {flux_rife.shape[0]}")
    del flux_src; gc.collect()

    # --- 3) Linear interpolation -------------------------------------------
    flux_hi = linear_interp_flux(flux_rife, factor=args.linear_factor)
    del flux_rife; gc.collect()
    T_hi = flux_hi.shape[0]
    eff_fps = T_hi / float(args.t_total)
    print(f"After Linear: T = {T_hi}  (effective FPS = {eff_fps:.1f})")
    if args.fps is not None and abs(eff_fps - args.fps) / max(args.fps, 1) > 0.1:
        print(f"  [warn] requested --fps={args.fps} but got {eff_fps:.1f} "
              f"from rife_exp={args.rife_exp}, linear_factor={args.linear_factor}")

    # --- 4) SPAD binary sampling ------------------------------------------
    print(f"\nSimulating SPAD (PPP={args.target_ppp}, dark={args.dark_count}) ...")
    rng = np.random.default_rng(args.seed)
    binary_hi, ppp_scale_hw = simulate_spad_binary(
        flux_hi, target_ppp=float(args.target_ppp),
        dark_count=float(args.dark_count),
        flux_peak=float(args.flux_peak),
        rng=rng,
    )
    detections = int(binary_hi.sum())
    print(f"  detections: {detections:,}  ({detections / binary_hi.size * 100:.3f}% of frames)")

    # --- 5) Save raw bit-packed binary -------------------------------------
    seq_len = int(args.sequence_length)
    tag = f"ppp{args.target_ppp:g}_fps{int(round(eff_fps))}_T{T_hi}_H{H}_W{W}"
    if args.save_raw_binary:
        bin_path = os.path.join(args.output_dir, f"binary_{tag}.bin")
        save_binary_bitpacked(binary_hi, bin_path, t_total=float(args.t_total))

    # --- 6) Bin to SEQ_LEN and save integrated GT + binary MP4 -------------
    print(f"\nBinning {T_hi} hi-FPS frames -> {seq_len} bins ...")
    bin_counts, bin_sizes_np = bin_along_time(binary_hi, seq_len=seq_len, reduce="sum")
    # [S, H, W] float32 for feeding into DiffPIR (per-pixel)
    gt_flux_bins = integrated_gt_flux(flux_hi, seq_len=seq_len,
                                      flux_peak=float(args.flux_peak))

    np.save(os.path.join(args.output_dir, f"gt_flux_bins_{tag}_S{seq_len}.npy"),
            gt_flux_bins.astype(np.float32))
    print(f"  saved integrated GT: gt_flux_bins_{tag}_S{seq_len}.npy "
          f"({seq_len},{H},{W})   range=[{gt_flux_bins.min():.2f},{gt_flux_bins.max():.2f}]")

    # Binned-binary MP4: any bin with >=1 detection -> 1 (i.e. 255)
    bb_frames = ((bin_counts >= 1).astype(np.uint8)) * 255  # [S,H,W]
    save_video_mp4(bb_frames,
                   os.path.join(args.output_dir, f"binned_binary_{tag}_S{seq_len}.mp4"),
                   fps=args.mp4_fps)

    # No longer need hi-FPS arrays.
    del binary_hi, flux_hi; gc.collect()

    # --- 7) DiffPIR ---------------------------------------------------------
    print("\nLoading diffusion model ...")
    model, diffusion = load_temporal_diffusion_model(
        checkpoint_path=args.checkpoint,
        sequence_length=seq_len,
        num_channels=args.num_channels,
        diffusion_steps=args.diffusion_steps,
        channel_mult=args.channel_mult,
        attention_resolutions=args.attention_resolutions,
        noise_schedule=args.noise_schedule,
        learn_sigma=bool(args.learn_sigma),
        device=device,
    )
    if args.compile_model:
        try:
            model = torch.compile(model)
            print("  model compiled.")
        except Exception as e:
            print(f"  torch.compile skipped ({e}).")

    # [S,H,W] -> [N, S]   (N = H*W pixels, S = seq_len)
    N = H * W
    bc_flat = bin_counts.reshape(seq_len, N).T.astype(np.float32)         # [N, S]
    bs_flat_row = bin_sizes_np.astype(np.float32)                         # [S]
    ppp_flat = ppp_scale_hw.reshape(-1).astype(np.float32)                # [N]

    log_flux_hat = np.empty((N, seq_len), dtype=np.float32)
    print(f"Running DiffPIR  ({args.sampling_steps} steps, batch={args.infer_batch_size}) ...")
    batches = list(chunked_indices(N, args.infer_batch_size))
    for bidx, sl in enumerate(tqdm(batches, desc="Inference")):
        bc = torch.from_numpy(bc_flat[sl]).to(device)
        bs_t = torch.from_numpy(
            np.broadcast_to(bs_flat_row[None, :], (bc.shape[0], seq_len)).copy()
        ).to(device)
        ppp_s = torch.from_numpy(ppp_flat[sl]).to(device)

        x_hat = sample_diffpir(
            model=model, diffusion=diffusion,
            bin_counts=bc, bin_sizes=bs_t,
            ppp_scale=ppp_s, dark_count=float(args.dark_count),
            num_steps=int(args.sampling_steps),
            diffusion_steps=int(args.diffusion_steps),
            lambda_data=float(args.lambda_data), eta=float(args.eta),
            pp_solver_iters=int(args.pp_solver_iters),
            pp_lr_scale=float(args.pp_lr_scale),
            t_total=float(args.t_total),
            x_param=str(args.x_param),
            sequence_length=seq_len,
            device=device,
            show_progress=(bidx == 0),
            use_amp=bool(args.use_amp),
        )
        log_flux_hat[sl] = x_hat[:, 0, :].cpu().numpy().astype(np.float32)

    # --- 8) Reshape reconstruction & save ----------------------------------
    # Clip log-flux before exp to avoid +Inf overflows from runaway samples.
    log_flux_hat_c = np.clip(log_flux_hat.astype(np.float64), -30.0, 25.0)
    if args.x_param == "log":
        flux_hat_flat = np.exp(log_flux_hat_c)
    else:
        flux_hat_flat = np.expm1(log_flux_hat_c)
    flux_hat_flat = np.nan_to_num(flux_hat_flat, nan=0.0, posinf=0.0, neginf=0.0)
    flux_hat_flat = np.maximum(flux_hat_flat, 0.0)
    recon_thw = flux_hat_flat.T.reshape(seq_len, H, W).astype(np.float32)

    np.save(os.path.join(args.output_dir, f"recon_flux_{tag}_S{seq_len}.npy"), recon_thw)
    save_video_mp4(flux_to_u8_frames_global(recon_thw),
                   os.path.join(args.output_dir, f"recon_{tag}_S{seq_len}.mp4"),
                   fps=args.mp4_fps)
    save_video_mp4(flux_to_u8_frames_global(gt_flux_bins),
                   os.path.join(args.output_dir, f"gt_{tag}_S{seq_len}.mp4"),
                   fps=args.mp4_fps)

    # --- 9) Metrics --------------------------------------------------------
    print("\nComputing PSNR / SSIM / LPIPS ...")
    metrics = compute_video_metrics(recon_thw, gt_flux_bins, device=device)
    print(f"  PSNR  mean = {metrics['psnr_mean']:.3f} dB")
    print(f"  SSIM  mean = {metrics['ssim_mean']:.4f}")
    print(f"  LPIPS mean = {metrics['lpips_mean']:.4f}")

    out = {
        "config": {
            "input": args.input,
            "checkpoint": args.checkpoint,
            "sequence_length": seq_len,
            "diffusion_steps": args.diffusion_steps,
            "sampling_steps":  args.sampling_steps,
            "channel_mult": args.channel_mult,
            "attention_resolutions": args.attention_resolutions,
            "noise_schedule": args.noise_schedule,
            "num_channels": args.num_channels,
            "learn_sigma": bool(args.learn_sigma),
            "rife_exp": args.rife_exp,
            "linear_factor": args.linear_factor,
            "effective_fps": eff_fps,
            "target_ppp": args.target_ppp,
            "dark_count": args.dark_count,
            "t_total": args.t_total,
            "flux_peak": args.flux_peak,
            "normalize_flux": normalize_flux,
            "T_src": T_src, "H": H, "W": W, "T_hi": T_hi,
            "x_param": args.x_param,
        },
        "metrics": metrics,
        "tag": tag,
    }
    with open(os.path.join(args.output_dir, f"metrics_{tag}_S{seq_len}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote metrics + recon + GT to {args.output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
