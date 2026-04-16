"""
Video inference: recover per-pixel photon flux from 1D SPAD observations.

Input : a `log_flux.pt` tensor of shape [T, H, W]  (log-flux video).
Output: a reconstructed MP4 video at 30 fps.

Pipeline (per pixel):
  1. Convert log-flux → linear flux, normalize to [0, flux_peak].
  2. Simulate SPAD binary detections (n_spad_frames frames).
  3. Bin detections into `sequence_length` bins.
  4. Run DiffPIR with binomial-likelihood data step to recover flux.
  5. Convert recovered log-flux → linear flux.
All H*W pixels are batched together; no random sampling, no file saving
except the final MP4.

Example
-------
python video_inference.py \
  --input  ./log_flux.pt \
  --checkpoint ./models_1024/model200000.pt \
  --output ./reconstructed.mp4 \
  --target_ppp 0.05 \
  --n_spad_frames 100000 \
  --infer_batch_size 256
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers copied / adapted from batch_inference.py
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def flux_from_x(x: torch.Tensor, x_param: str) -> torch.Tensor:
    if x_param == "log":
        return torch.exp(x)
    if x_param == "log1p":
        return torch.expm1(x)
    raise ValueError(f"Unknown x_param={x_param!r}")


def x_from_flux(flux: np.ndarray, x_param: str) -> np.ndarray:
    flux = np.asarray(flux, dtype=np.float64)
    if x_param == "log":
        return np.log(np.maximum(flux, 1e-12))
    if x_param == "log1p":
        return np.log1p(np.maximum(flux, 0.0))
    raise ValueError(f"Unknown x_param={x_param!r}")


def maybe_add_to_syspath(path: Optional[str]) -> None:
    if path and os.path.isdir(path):
        if path not in sys.path:
            sys.path.insert(0, path)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_temporal_diffusion_model(
    *,
    checkpoint_path: str,
    sequence_length: int,
    num_channels: int,
    diffusion_steps: int,
    device: torch.device,
) -> Tuple[torch.nn.Module, object]:
    from improved_diffusion.temporal_script_util import (
        temporal_model_and_diffusion_defaults,
        create_temporal_model_and_diffusion,
    )

    defaults = temporal_model_and_diffusion_defaults()
    defaults.update(
        {
            "sequence_length": sequence_length,
            "num_channels": num_channels,
            "diffusion_steps": diffusion_steps,
            "noise_schedule": "linear",
            "learn_sigma": False,
            "attention_resolutions": "256, 128",
        }
    )
    model, diffusion = create_temporal_model_and_diffusion(**defaults)
    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Checkpoint not found at `{checkpoint_path}`")

    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, diffusion


# ---------------------------------------------------------------------------
# SPAD simulation & binning  (vectorised over many pixels at once)
# ---------------------------------------------------------------------------

def generate_spad_binary_batch(
    flux_batch: np.ndarray,
    target_ppp: float,
    n_spad_frames: int,
    dark_count: float = 7.74e-4,
    T: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate SPAD binary detections for a batch of 1-D flux signals.

    Parameters
    ----------
    flux_batch : ndarray [N, L]
        Linear-space flux for each pixel (already normalised to [0, flux_peak]).
    target_ppp : float
        Target average photons per pixel.
    n_spad_frames : int
        Number of binary frames to simulate per pixel.
    dark_count : float
        Spurious detection rate per frame.
    T : float
        Total observation time.

    Returns
    -------
    binary_batch : ndarray [N, n_spad_frames]  uint8
    ppp_scales   : ndarray [N]                  float32
    """
    N, L = flux_batch.shape
    flux_batch = flux_batch.astype(np.float64)

    # Interpolate each pixel's flux from L time-bins to n_spad_frames bins.
    t_src = np.linspace(0.0, T, L)
    t_dst = np.linspace(0.0, T, n_spad_frames)
    # Vectorised interpolation row-by-row.
    flux_interp = np.empty((N, n_spad_frames), dtype=np.float64)
    for i in range(N):
        flux_interp[i] = np.interp(t_dst, t_src, flux_batch[i])

    # Normalise each pixel to [0, 1].
    mx = flux_interp.max(axis=1, keepdims=True)  # [N, 1]
    mx = np.where(mx > 0, mx, 1.0)
    flux_norm = flux_interp / mx

    # Scaling factor a so that mean(a * I) = target_ppp.
    I_mean = flux_norm.mean(axis=1, keepdims=True)  # [N, 1]
    I_mean = np.where(I_mean > 0, I_mean, 1.0)
    a = target_ppp / I_mean                          # [N, 1]

    # ppp_scale = a / flux_max  (maps raw flux → expected photons per frame)
    ppp_scales = (a / mx).ravel().astype(np.float32)  # [N]

    # Scaled flux: N(t) = a * I_norm(t) + d
    flux_scaled = a * flux_norm + dark_count  # [N, n_spad_frames]

    # Detection probability: 1 - exp(-N(t))
    det_prob = 1.0 - np.exp(-flux_scaled)

    # Sample binary detections.
    binary = (np.random.random((N, n_spad_frames)) < det_prob).astype(np.uint8)

    return binary, ppp_scales


def bin_binary_batch(
    binary: np.ndarray,
    bin_edges: np.ndarray,
) -> np.ndarray:
    """
    Bin [N, n_spad_frames] binary arrays into [N, num_bins] counts.
    """
    N = binary.shape[0]
    num_bins = bin_edges.shape[0] - 1
    n_frames = binary.shape[1]

    # Fast path for equal-size bins.
    bin_sizes = np.diff(bin_edges)
    if np.all(bin_sizes == bin_sizes[0]) and (n_frames % num_bins == 0):
        m = n_frames // num_bins
        return binary.reshape(N, num_bins, m).sum(axis=2).astype(np.float32)

    # General path (row-by-row reduceat).
    out = np.empty((N, num_bins), dtype=np.float32)
    starts = bin_edges[:-1].astype(np.intp)
    for i in range(N):
        counts = np.add.reduceat(binary[i].astype(np.int64), starts)
        out[i] = counts[:num_bins].astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# DiffPIR core (identical to batch_inference.py)
# ---------------------------------------------------------------------------

def spad_data_step_binomial(
    x0_pred: torch.Tensor,
    bin_counts: torch.Tensor,
    bin_sizes: torch.Tensor,
    rho_t: torch.Tensor,
    *,
    ppp_scale,
    dark_count: float = 7.74e-4,
    n_iter: int = 10,
    lr_scale: float = 0.75,
    total_count_weight: float = 0.001,
    t_total: float = 1.0,
    x_param: str = "log",
):
    batch_size = x0_pred.shape[0]
    seq_len = x0_pred.shape[2]
    x = x0_pred.clone().detach().clamp(-10, 15).requires_grad_(True)

    ppp_scale_t = torch.as_tensor(ppp_scale, device=x.device, dtype=x.dtype)
    if ppp_scale_t.ndim == 0:
        ppp_scale_t = ppp_scale_t.expand(batch_size).unsqueeze(1)
    elif ppp_scale_t.ndim == 1:
        ppp_scale_t = ppp_scale_t.unsqueeze(1)

    dt = t_total / seq_len
    total_det = bin_counts.sum(-1)
    total_frm = bin_sizes.sum(-1)
    det_rate = torch.clamp(total_det / (total_frm + 1e-8), 1e-8, 1 - 1e-3)
    N_per_frame = -torch.log(1 - det_rate)
    mean_flux_target = torch.clamp(
        (N_per_frame - dark_count) / (ppp_scale_t.squeeze(1) + 1e-10), min=1.0
    )
    N_obs = mean_flux_target * t_total

    for _ in range(n_iter):
        phi = flux_from_x(x, x_param=x_param).squeeze(1)
        phi = torch.clamp(phi, min=0.0)

        N_k = ppp_scale_t * phi + dark_count
        p_k = torch.clamp(1.0 - torch.exp(-N_k), 1e-8, 1 - 1e-8)

        nll_per = -(bin_counts * torch.log(p_k)
                    + (bin_sizes - bin_counts) * torch.log(1 - p_k)).sum(-1)

        flux_integral = (phi * dt).sum(-1)
        count_penalty_per = total_count_weight * (
            (flux_integral - N_obs) ** 2 / (N_obs + 1e-8)
        )

        prox_per = 0.5 * rho_t * ((x - x0_pred) ** 2).sum(dim=[1, 2])
        total_loss = (nll_per + count_penalty_per + prox_per).sum()

        grad = torch.autograd.grad(total_loss, x)[0]
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

        with torch.no_grad():
            L_nll_per = (bin_sizes * (ppp_scale_t * phi) ** 2
                         * torch.exp(-N_k)).max(dim=-1).values
            L_nll_per = torch.nan_to_num(L_nll_per, nan=0.0, posinf=1e6, neginf=0.0)
            step = lr_scale / (L_nll_per + rho_t + 1e-8)
            step = step.view(batch_size, 1, 1)
            x = (x - step * grad).clamp(-10, 10)
        x = x.detach().requires_grad_(True)

    return x.detach()


def sample_diffpir_photon_flux(
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
    t_total: float = 1.0,
    x_param: str,
    sequence_length: int,
    device: torch.device,
    show_progress: bool = False,
) -> torch.Tensor:
    batch_size = bin_counts.shape[0]
    shape = (batch_size, 1, sequence_length)

    x_t = torch.randn(shape, device=device)

    alphas_cumprod = torch.from_numpy(diffusion.alphas_cumprod).to(device)
    sqrt_alphas_cumprod = torch.from_numpy(diffusion.sqrt_alphas_cumprod).to(device)
    sqrt_one_minus_alphas_cumprod = torch.from_numpy(
        diffusion.sqrt_one_minus_alphas_cumprod
    ).to(device)

    timesteps = np.linspace(diffusion_steps - 1, 0, num_steps, dtype=int)

    iterator = (
        tqdm(list(enumerate(timesteps)), total=len(timesteps), desc="DiffPIR")
        if show_progress
        else enumerate(timesteps)
    )

    for step_idx, t in iterator:
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

        with torch.no_grad():
            model_output = model(x_t, t_tensor)
            model_output = torch.nan_to_num(model_output, nan=0.0, posinf=0.0, neginf=0.0)

            alpha_bar_t = alphas_cumprod[t]
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)

            x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * model_output) / sqrt_alpha_bar_t
            x0_pred = torch.nan_to_num(x0_pred, nan=0.0, posinf=15.0, neginf=-10.0)
            x0_pred = x0_pred.clamp(-10, 15)

        sigma_k_t = sqrt_one_minus_alpha_bar_t / (sqrt_alpha_bar_t + 1e-8)
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
            alpha_bar_t_prev = alphas_cumprod[t_prev]
            sqrt_alpha_bar_t_prev = torch.sqrt(alpha_bar_t_prev)

            eps_hat = (x_t - sqrt_alpha_bar_t * x0_hat) / (sqrt_one_minus_alpha_bar_t + 1e-8)

            sigma_t = eta * torch.sqrt(torch.clamp(
                (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev),
                min=0.0,
            ))

            dir_xt = torch.sqrt(torch.clamp(
                1 - alpha_bar_t_prev - sigma_t ** 2, min=0.0,
            )) * eps_hat
            noise = torch.randn_like(x_t) if eta > 0 else 0

            x_t = sqrt_alpha_bar_t_prev * x0_hat + dir_xt + sigma_t * noise
            x_t = torch.nan_to_num(x_t, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            x_t = x0_hat

    return x_t


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

def flux_to_video_frames(flux: np.ndarray) -> np.ndarray:
    """
    Convert a [T, H, W] linear-flux array to uint8 grayscale frames [T, H, W]
    by normalising to [0, 255] globally.
    """
    fmin, fmax = float(flux.min()), float(flux.max())
    if fmax - fmin < 1e-12:
        return np.zeros_like(flux, dtype=np.uint8)
    normed = (flux - fmin) / (fmax - fmin)
    return (normed * 255.0).clip(0, 255).astype(np.uint8)


def save_video_mp4(frames: np.ndarray, path: str, fps: int = 30) -> None:
    """
    Write [T, H, W] uint8 grayscale frames to an MP4 file.
    Tries cv2 (OpenCV) first, then falls back to imageio.
    """
    T, H, W = frames.shape
    try:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, fps, (W, H), isColor=False)
        if not writer.isOpened():
            raise RuntimeError("cv2.VideoWriter failed to open")
        for t in range(T):
            writer.write(frames[t])
        writer.release()
        print(f"Saved video via OpenCV: {path}")
        return
    except Exception as e:
        print(f"OpenCV video write failed ({e}), trying imageio ...")

    import imageio.v3 as iio
    # imageio expects [T, H, W] for grayscale or [T, H, W, 3] for colour.
    # Convert grayscale to 3-channel so most codecs are happy.
    frames_rgb = np.stack([frames, frames, frames], axis=-1)  # [T, H, W, 3]
    iio.imwrite(path, frames_rgb, fps=fps, codec="libx264")
    print(f"Saved video via imageio: {path}")


# ---------------------------------------------------------------------------
# Chunked index iterator
# ---------------------------------------------------------------------------

def chunked_indices(n: int, batch_size: int) -> Iterable[slice]:
    for i in range(0, n, batch_size):
        yield slice(i, min(i + batch_size, n))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Reconstruct a video from per-pixel 1D SPAD observations via DiffPIR."
    )
    # I/O
    ap.add_argument("--input", type=str, required=True,
                     help="Path to log_flux .pt file of shape [T, H, W]")
    ap.add_argument("--checkpoint", type=str, required=True,
                     help="Path to diffusion model checkpoint .pt")
    ap.add_argument("--output", type=str, default="reconstructed.mp4",
                     help="Output MP4 path (default: reconstructed.mp4)")
    ap.add_argument("--improved_diffusion_root", type=str, default=None,
                     help="Optional path to add to sys.path for improved_diffusion")

    # Model / diffusion
    ap.add_argument("--sequence_length", type=int, default=1024)
    ap.add_argument("--num_channels", type=int, default=64)
    ap.add_argument("--diffusion_steps", type=int, default=1000)
    ap.add_argument("--sampling_steps", type=int, default=100)
    ap.add_argument("--lambda_data", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=0.85)

    # Data-step solver
    ap.add_argument("--pp_solver_iters", type=int, default=10)
    ap.add_argument("--pp_lr_scale", type=float, default=0.75)

    # SPAD simulation
    ap.add_argument("--t_total", type=float, default=1.0)
    ap.add_argument("--dark_count", type=float, default=7.74e-4)
    ap.add_argument("--n_spad_frames", type=int, default=100_000)
    ap.add_argument("--target_ppp", type=float, default=0.05)

    # Parameterisation & normalisation
    ap.add_argument("--x_param", type=str, default="log", choices=["log", "log1p"])
    ap.add_argument("--flux_peak", type=float, default=10000.0,
                     help="Peak value after per-pixel normalisation (default: 10000)")
    ap.add_argument("--no_normalize_flux", action="store_true",
                     help="Disable per-pixel flux normalisation")

    # Runtime
    ap.add_argument("--infer_batch_size", type=int, default=256,
                     help="Number of pixels processed per GPU batch")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps", type=int, default=30, help="Output video frame rate")

    args = ap.parse_args()
    maybe_add_to_syspath(args.improved_diffusion_root)
    set_seed(args.seed)
    normalize_flux = not args.no_normalize_flux

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ------------------------------------------------------------------
    # 1.  Load data  [T, H, W]
    # ------------------------------------------------------------------
    raw = torch.load(args.input, map_location="cpu")
    if not isinstance(raw, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor in `{args.input}`, got {type(raw)}")
    if raw.ndim != 3:
        raise ValueError(f"Expected shape [T, H, W], got {tuple(raw.shape)}")

    T_vid, H, W = raw.shape
    seq_len = args.sequence_length
    num_pixels = H * W
    print(f"Input  : {args.input}  shape=({T_vid}, {H}, {W})  pixels={num_pixels}")

    # The model operates on sequences of length `sequence_length`.
    # T must be >= sequence_length; we take the first sequence_length frames.
    if T_vid < seq_len:
        raise ValueError(
            f"Temporal dimension T={T_vid} is shorter than "
            f"sequence_length={seq_len}. Cannot proceed."
        )
    if T_vid > seq_len:
        print(f"Note   : T={T_vid} > sequence_length={seq_len}; "
              f"using first {seq_len} frames only.")
        raw = raw[:seq_len]  # [seq_len, H, W]

    # Reshape to [num_pixels, seq_len].
    log_flux_2d = raw.numpy().astype(np.float64).reshape(seq_len, -1).T  # [N, seq_len]

    # Convert to linear flux.
    flux_all = np.exp(log_flux_2d)  # [N, seq_len]

    # Per-pixel normalisation to [0, flux_peak].
    if normalize_flux:
        mx = flux_all.max(axis=1, keepdims=True)
        mx = np.where(mx > 0, mx, 1.0)
        flux_all = (flux_all / mx) * float(args.flux_peak)

    flux_for_infer = flux_all  # [N, seq_len]

    print(f"Flux range (linear): [{flux_for_infer.min():.1f}, {flux_for_infer.max():.1f}]")

    # ------------------------------------------------------------------
    # 2.  Load diffusion model
    # ------------------------------------------------------------------
    model, diffusion = load_temporal_diffusion_model(
        checkpoint_path=args.checkpoint,
        sequence_length=seq_len,
        num_channels=args.num_channels,
        diffusion_steps=args.diffusion_steps,
        device=device,
    )

    # ------------------------------------------------------------------
    # 3.  Simulate SPAD & bin  (all pixels)
    # ------------------------------------------------------------------
    print(f"\nSimulating SPAD observations (PPP={args.target_ppp}, "
          f"frames={args.n_spad_frames}) ...")

    bin_edges = np.linspace(0, args.n_spad_frames, seq_len + 1, dtype=np.int64)
    bin_sizes_np = np.diff(bin_edges).astype(np.float32)  # [seq_len]

    # Process SPAD simulation in chunks to limit RAM usage.
    spad_chunk = max(args.infer_batch_size, 1024)
    bin_counts_all = np.empty((num_pixels, seq_len), dtype=np.float32)
    ppp_scales_all = np.empty((num_pixels,), dtype=np.float32)

    for sl in tqdm(list(chunked_indices(num_pixels, spad_chunk)), desc="SPAD sim"):
        flux_chunk = flux_for_infer[sl]
        binary_chunk, ppp_chunk = generate_spad_binary_batch(
            flux_chunk,
            target_ppp=args.target_ppp,
            n_spad_frames=args.n_spad_frames,
            dark_count=args.dark_count,
            T=args.t_total,
        )
        bin_counts_all[sl] = bin_binary_batch(binary_chunk, bin_edges)
        ppp_scales_all[sl] = ppp_chunk

    # ------------------------------------------------------------------
    # 4.  DiffPIR inference  (batched over pixels)
    # ------------------------------------------------------------------
    print(f"\nRunning DiffPIR inference ({args.sampling_steps} steps, "
          f"batch_size={args.infer_batch_size}) ...")

    log_flux_hat = np.empty((num_pixels, seq_len), dtype=np.float32)

    batches = list(chunked_indices(num_pixels, args.infer_batch_size))
    with torch.set_grad_enabled(True):
        for batch_idx, sl in enumerate(tqdm(batches, desc="Inference")):
            bc = torch.from_numpy(bin_counts_all[sl]).to(device)
            bs_tensor = torch.from_numpy(
                np.broadcast_to(bin_sizes_np[None, :], (bc.shape[0], seq_len)).copy()
            ).to(device)
            ppp_s = torch.from_numpy(ppp_scales_all[sl]).to(device)

            x_hat = sample_diffpir_photon_flux(
                model=model,
                diffusion=diffusion,
                bin_counts=bc,
                bin_sizes=bs_tensor,
                ppp_scale=ppp_s,
                dark_count=float(args.dark_count),
                num_steps=int(args.sampling_steps),
                diffusion_steps=int(args.diffusion_steps),
                lambda_data=float(args.lambda_data),
                eta=float(args.eta),
                pp_solver_iters=int(args.pp_solver_iters),
                pp_lr_scale=float(args.pp_lr_scale),
                t_total=float(args.t_total),
                x_param=str(args.x_param),
                sequence_length=seq_len,
                device=device,
                show_progress=(batch_idx == 0),  # progress bar for the first batch only
            )
            log_flux_hat[sl] = x_hat[:, 0, :].detach().cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # 5.  Convert recovered log-flux → linear flux → video frames
    # ------------------------------------------------------------------
    if args.x_param == "log":
        flux_hat = np.exp(log_flux_hat.astype(np.float64))
    elif args.x_param == "log1p":
        flux_hat = np.expm1(log_flux_hat.astype(np.float64))
    else:
        flux_hat = np.exp(log_flux_hat.astype(np.float64))
    flux_hat = np.maximum(flux_hat, 0.0)  # [num_pixels, seq_len]

    # Reshape to [seq_len, H, W].
    flux_video = flux_hat.T.reshape(seq_len, H, W)  # [seq_len, H, W]
    print(f"\nReconstructed flux range: [{flux_video.min():.1f}, {flux_video.max():.1f}]")

    # ------------------------------------------------------------------
    # 6.  Save MP4
    # ------------------------------------------------------------------
    frames = flux_to_video_frames(flux_video)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_video_mp4(frames, args.output, fps=args.fps)
    print("Done.")


if __name__ == "__main__":
    main()
