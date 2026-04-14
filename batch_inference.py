"""
Batch inference for 1D temporal photon flux estimation with DiffPIR + SPAD binomial likelihood.

What this script does:
- Loads a `log_flux_dataset.pt` tensor of shape [N, L] (log(flux) samples).
- Randomly samples `--num_samples` items.
- For each TARGET_PPP level:
  - Simulates SPAD binary detections with `--n_spad_frames` frames.
  - Bins detections to `--sequence_length` bins (the diffusion model length).
  - Runs DiffPIR inference in batches.
  - Computes metrics, dropping any sample with any NaN/inf metric entry.
  - Saves detections + binned data + per-sample metrics.

python batch_inference.py \
  --dataset "./data/step_log_flux_dataset.pt" \
  --checkpoint "./models_1024/model200000.pt" \
  --output_dir "./outputs/batch_run_01" \
  --num_samples 2000 \
  --target_ppp "0.05" \
  --n_spad_frames 100000 \
  --infer_batch_size 256
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

np.random.seed(42)
@dataclass(frozen=True)
class Metrics:
    mse: float
    rel_mse: float
    mae: float
    rel_mae: float
    corr: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "MSE": float(self.mse),
            "Relative MSE": float(self.rel_mse),
            "MAE": float(self.mae),
            "Relative MAE": float(self.rel_mae),
            "Correlation": float(self.corr),
        }


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def flux_from_x(x: torch.Tensor, x_param: str) -> torch.Tensor:
    """
    Map model variable x -> physical flux phi.
    - x_param='log'   : x = log(phi)      => phi = exp(x)
    - x_param='log1p' : x = log(1 + phi)  => phi = expm1(x)
    """
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


def parse_ppp_list(ppp: str) -> List[float]:
    vals: List[float] = []
    for part in ppp.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise ValueError("No PPP values parsed. Example: --target_ppp 0.05,0.01")
    return vals


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def maybe_add_to_syspath(path: Optional[str]) -> None:
    if path and os.path.isdir(path):
        if path not in sys.path:
            sys.path.insert(0, path)


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
            'attention_resolutions': '256, 128',
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


def generate_photon_arrivals_spad(
    flux,
    target_ppp: float = 1.0,
    d: float = 7.74e-4,
    T: float = 1.0,
                                   return_binary=False, return_flux_scaled=False, seq_length=None):
    """
    Generate photon arrivals using SPAD-style Poisson sampling.
    
    This follows the SPAD dataset generation pipeline:
    1. Scale flux to achieve target average photons per pixel (PPP)
    2. Apply Poisson sampling: Pr{Φ(x,t)=1} = 1 - exp(-N(x,t))
    3. Convert binary detections to arrival times
    
    Args:
        flux: [seq_length] photon flux function (intensity values, e.g., grayscale [0,1])
        target_ppp: Target average photons per pixel/time-bin (default 1.0)
                   Lower values = sparser detections (e.g., 0.1, 0.01, 0.001)
        d: Spurious detection rate per frame (dark counts, default 7.74e-4)
        T: Total observation time in seconds
        return_binary: If True, also return the binary detection array
        return_flux_scaled: If True, also return the scaled flux (N values)
        seq_length: Number of discrete time bins over [0, T]. If None, uses len(flux).
                    If it differs from len(flux), flux is linearly interpolated in time.
    
    Returns:
        arrivals: array of photon arrival times
        (optional) binary: [seq_length] binary detection array
        (optional) flux_scaled: [seq_length] scaled flux values (N = a*I + d)
    
    Example PPP values:
        - PPP=1.0: ~1 photon per time bin on average (high light)
        - PPP=0.1: ~0.1 photons per time bin (moderate light)
        - PPP=0.01: ~0.01 photons per time bin (low light)
        - PPP=0.001: ~0.001 photons per time bin (very low light, sparse)
    """
    flux = np.asarray(flux, dtype=np.float64).ravel()
    n_src = len(flux)
    if seq_length is None:
        seq_length = n_src
    elif seq_length != n_src:
        t_src = np.linspace(0.0, T, n_src)
        t_dst = np.linspace(0.0, T, seq_length)
        flux = np.interp(t_dst, t_src, flux)

    # Normalize flux to [0, 1] if not already
    flux_normalized = flux.copy()
    if flux_normalized.max() > 1.0:
        flux_normalized = flux_normalized / flux_normalized.max()
    
    # Calculate scaling factor 'a' so that mean(a * I) = target_ppp
    I_mean = flux_normalized.mean()
    if I_mean > 0:
        a = target_ppp / I_mean
    else:
        a = 1.0
    
    # Calculate scaled flux: N(t) = a * I(t) + d
    flux_scaled = a * flux_normalized + d
    
    # Poisson sampling: Pr{detection at t} = 1 - exp(-N(t))
    detection_prob = 1.0 - np.exp(-flux_scaled)
    
    # Sample binary detections
    binary = (np.random.random(seq_length) < detection_prob).astype(np.uint8)
    
    # Convert binary detections to arrival times
    detection_indices = np.where(binary == 1)[0]
    dt = T / (seq_length - 1) if seq_length > 1 else T
    arrivals = detection_indices * dt
    
    # Build return tuple
    result = [arrivals]
    if return_binary:
        result.append(binary)
    if return_flux_scaled:
        result.append(flux_scaled)
    
    if len(result) == 1:
        return result[0]
    return tuple(result)


def bin_spad_binary(binary, num_bins=1024):
    """
    Bin SPAD binary detections into num_bins equal-width bins.
    
    Args:
        binary: [n_frames] binary detection array (0 or 1)
        num_bins: number of output bins
    
    Returns:
        bin_counts: [num_bins] number of detections per bin
        bin_sizes: [num_bins] number of frames per bin
    """
    n_frames = len(binary)
    bin_edges = np.linspace(0, n_frames, num_bins + 1, dtype=int)
    bin_counts = np.array([binary[bin_edges[i]:bin_edges[i+1]].sum()
                           for i in range(num_bins)], dtype=np.float32)
    bin_sizes = np.diff(bin_edges).astype(np.float32)
    return bin_counts, bin_sizes


def bin_spad_binary_with_edges(binary: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """
    Faster binner when you already have bin_edges.
    Returns counts only.
    """
    # Try the fast path for equal-size bins.
    n_frames = int(binary.shape[0])
    num_bins = int(bin_edges.shape[0] - 1)
    bin_sizes = np.diff(bin_edges)
    if np.all(bin_sizes == bin_sizes[0]) and (n_frames % num_bins == 0):
        m = n_frames // num_bins
        return binary.reshape(num_bins, m).sum(axis=1, dtype=np.int64).astype(np.float32)

    # General path.
    # Using reduceat over edges[:-1]; last bin ends at n_frames automatically.
    counts = np.add.reduceat(binary.astype(np.int64), bin_edges[:-1])
    # reduceat includes partial sums beyond the end if the last index equals n_frames;
    # we ensure edges[-1] == n_frames by construction, so it's safe.
    return counts[:num_bins].astype(np.float32)


def compute_ppp_scale_for_flux(flux: np.ndarray, target_ppp: float) -> float:
    """
    Reproduces the mapping implied by SPAD generation:
      flux_normalized = flux / flux_max
      a = target_ppp / mean(flux_normalized)
      N = a * flux_normalized + d = (a/flux_max) * flux + d
    So `ppp_scale = a/flux_max`.
    """
    flux = np.asarray(flux, dtype=np.float64).ravel()
    flux_max = float(np.max(flux)) if flux.size else 1.0
    if not np.isfinite(flux_max) or flux_max <= 0:
        flux_max = 1.0
    flux_normalized = flux / flux_max
    I_mean = float(np.mean(flux_normalized)) if flux_normalized.size else 0.0
    if not np.isfinite(I_mean) or I_mean <= 0:
        I_mean = 1.0
    a_scale = float(target_ppp) / I_mean
    return float(a_scale / flux_max)


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
    """
    Solve data subproblem with Binomial likelihood for binned SPAD data.

    Minimizes: NLL + total_count_penalty + proximal
    where total_count_penalty = w * (sum(exp(x_i)*dt) - N_obs)^2 / N_obs
    anchors the integrated intensity from the Poisson compensator equation.

    Uses adaptive step size based on the Lipschitz constant of the NLL
    gradient to ensure stable convergence.
    """
    batch_size = x0_pred.shape[0]
    seq_len = x0_pred.shape[2]
    x = x0_pred.clone().detach().clamp(-10, 15).requires_grad_(True)

    # ppp_scale: scalar or [batch] -> [batch, 1] for broadcasting over seq_len.
    ppp_scale_t = torch.as_tensor(ppp_scale, device=x.device, dtype=x.dtype)
    if ppp_scale_t.ndim == 0:
        ppp_scale_t = ppp_scale_t.expand(batch_size).unsqueeze(1)  # [B, 1]
    elif ppp_scale_t.ndim == 1:
        ppp_scale_t = ppp_scale_t.unsqueeze(1)  # [B, 1]

    # Compute target flux integral from observed SPAD detections (per-sample).
    dt = t_total / seq_len
    total_det = bin_counts.sum(-1)                                    # [B]
    total_frm = bin_sizes.sum(-1)                                     # [B]
    det_rate = torch.clamp(total_det / (total_frm + 1e-8), 1e-8, 1 - 1e-3)  # [B]
    N_per_frame = -torch.log(1 - det_rate)                            # [B]
    mean_flux_target = torch.clamp(
        (N_per_frame - dark_count) / (ppp_scale_t.squeeze(1) + 1e-10), min=1.0
    )                                                                 # [B]
    N_obs = mean_flux_target * t_total                                # [B]

    for _ in range(n_iter):
        phi = flux_from_x(x, x_param=x_param).squeeze(1)  # [B, seq_len]
        phi = torch.clamp(phi, min=0.0)

        N_k = ppp_scale_t * phi + dark_count                       # [B, seq_len]
        p_k = torch.clamp(1.0 - torch.exp(-N_k), 1e-8, 1 - 1e-8)

        # Per-sample NLL (sum over seq_len only, NOT over batch).
        nll_per = -(bin_counts * torch.log(p_k) +
                    (bin_sizes - bin_counts) * torch.log(1 - p_k)).sum(-1)  # [B]

        # Total count constraint (Poisson compensator) per sample.
        flux_integral = (phi * dt).sum(-1)                             # [B]
        count_penalty_per = total_count_weight * (
            (flux_integral - N_obs) ** 2 / (N_obs + 1e-8)
        )                                                              # [B]

        # Per-sample proximal (sum over channels + seq_len only).
        prox_per = 0.5 * rho_t * ((x - x0_pred) ** 2).sum(dim=[1, 2])     # [B]

        # Sum (not mean!) so each sample's gradient is independent of batch size.
        total_loss = (nll_per + count_penalty_per + prox_per).sum()

        grad = torch.autograd.grad(total_loss, x)[0]
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

        with torch.no_grad():
            # Per-sample Lipschitz constant -> per-sample step size [B, 1, 1].
            L_nll_per = (bin_sizes * (ppp_scale_t * phi) ** 2
                         * torch.exp(-N_k)).max(dim=-1).values   # [B]
            L_nll_per = torch.nan_to_num(L_nll_per, nan=0.0, posinf=1e6, neginf=0.0)
            step = lr_scale / (L_nll_per + rho_t + 1e-8)        # [B]
            step = step.view(batch_size, 1, 1)                   # [B, 1, 1]

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
    show_progress: bool,
) -> torch.Tensor:
    """
    DiffPIR sampling for photon flux estimation from binned SPAD data.
    
    Args:
        model: pre-trained 1D diffusion model
        diffusion: Gaussian diffusion object
        bin_counts: [batch, seq_len] detection counts per bin
        bin_sizes: [batch, seq_len] number of frames per bin
        ppp_scale: scaling factor mapping flux to expected photons per frame
        dark_count: spurious detection rate per frame
        num_steps: number of reverse diffusion steps
        lambda_data: data fidelity weight
        eta: DDIM parameter (0=deterministic, 1=stochastic)
        verbose: show progress bar
    
    Returns:
        x0_final: [batch, 1, seq_len] estimated log-flux
        trajectory: list of intermediate estimates
    """
    batch_size = bin_counts.shape[0]
    shape = (batch_size, 1, sequence_length)
    
    # Start from random noise
    x_t = torch.randn(shape, device=device)
    
    # Get diffusion schedule parameters
    alphas_cumprod = torch.from_numpy(diffusion.alphas_cumprod).to(device)
    sqrt_alphas_cumprod = torch.from_numpy(diffusion.sqrt_alphas_cumprod).to(device)
    sqrt_one_minus_alphas_cumprod = torch.from_numpy(diffusion.sqrt_one_minus_alphas_cumprod).to(device)
    
    # Create sampling timesteps (uniform spacing)
    timesteps = np.linspace(diffusion_steps - 1, 0, num_steps, dtype=int)

    iterator: Iterable[Tuple[int, int]]
    if show_progress:
        iterator = tqdm(list(enumerate(timesteps)), total=len(timesteps), desc="DiffPIR Sampling")
    else:
        iterator = enumerate(timesteps)
    
    for step_idx, t in iterator:
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        
        # === Step 1: Predict clean x0 from noisy x_t ===
        with torch.no_grad():
            model_output = model(x_t, t_tensor)
            model_output = torch.nan_to_num(model_output, nan=0.0, posinf=0.0, neginf=0.0)

            alpha_bar_t = alphas_cumprod[t]
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)

            x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * model_output) / sqrt_alpha_bar_t
            x0_pred = torch.nan_to_num(x0_pred, nan=0.0, posinf=15.0, neginf=-10.0)
            x0_pred = x0_pred.clamp(-10, 15)
        
        # === Step 2: Data subproblem with Binomial likelihood ===
        sigma_k_t = sqrt_one_minus_alpha_bar_t / (sqrt_alpha_bar_t + 1e-8)
        rho_t = lambda_data / (sigma_k_t ** 2 + 1e-8)
        
        x0_hat = spad_data_step_binomial(
            x0_pred,
            bin_counts,
            bin_sizes,
            rho_t,
            ppp_scale=ppp_scale,
            dark_count=dark_count,
            n_iter=pp_solver_iters,
            lr_scale=pp_lr_scale,
            t_total=t_total,
            x_param=x_param,
        )
        x0_hat = torch.nan_to_num(x0_hat, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10, 15)

        # === Step 3: DDIM update ===
        if step_idx < len(timesteps) - 1:
            t_prev = timesteps[step_idx + 1]
            alpha_bar_t_prev = alphas_cumprod[t_prev]
            sqrt_alpha_bar_t_prev = torch.sqrt(alpha_bar_t_prev)
            
            eps_hat = (x_t - sqrt_alpha_bar_t * x0_hat) / (sqrt_one_minus_alpha_bar_t + 1e-8)
            
            sigma_t = eta * torch.sqrt(torch.clamp(
                (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev),
                min=0.0
            ))

            dir_xt = torch.sqrt(torch.clamp(
                1 - alpha_bar_t_prev - sigma_t ** 2, min=0.0
            )) * eps_hat
            noise = torch.randn_like(x_t) if eta > 0 else 0

            x_t = sqrt_alpha_bar_t_prev * x0_hat + dir_xt + sigma_t * noise
            x_t = torch.nan_to_num(x_t, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            x_t = x0_hat

    return x_t


def compute_metrics(flux_true: np.ndarray, flux_pred: np.ndarray) -> Metrics:
    """
    Compute reconstruction metrics.
    """
    # Mean Squared Error
    mse = np.mean((flux_true - flux_pred) ** 2)
    
    # Relative MSE
    rel_mse = mse / np.mean(flux_true ** 2)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(flux_true - flux_pred))
    
    # Relative MAE
    rel_mae = mae / np.mean(np.abs(flux_true))
    
    # Correlation coefficient
    corr = np.corrcoef(flux_true, flux_pred)[0, 1]
    
    return Metrics(mse=mse, rel_mse=rel_mse, mae=mae, rel_mae=rel_mae, corr=corr)


def is_finite_metrics(m: Metrics) -> bool:
    arr = np.array([m.mse, m.rel_mse, m.mae, m.rel_mae, m.corr], dtype=np.float64)
    return bool(np.all(np.isfinite(arr)))


def load_log_flux_dataset(path: str) -> torch.Tensor:
    ds = torch.load(path, map_location="cpu")
    if isinstance(ds, torch.Tensor):
        return ds
    raise TypeError(f"Expected a torch.Tensor in `{path}`, got {type(ds)}")


def sample_indices(n_total: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if n > n_total:
        raise ValueError(f"--num_samples {n} exceeds dataset size {n_total}")
    return rng.choice(n_total, size=n, replace=False)


def chunked_indices(n: int, batch_size: int) -> Iterable[slice]:
    for i in range(0, n, batch_size):
        yield slice(i, min(i + batch_size, n))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, help="Path to log_flux_dataset.pt (tensor [N, L])")
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to diffusion model checkpoint .pt")
    ap.add_argument("--improved_diffusion_root", type=str, default=None, help="Optional path to add to sys.path")
    ap.add_argument("--output_dir", type=str, required=True, help="Directory to save detections/metrics")

    ap.add_argument("--num_samples", type=int, required=True, help="Number of random samples to run")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--sequence_length", type=int, default=1024)
    ap.add_argument("--num_channels", type=int, default=64)
    ap.add_argument("--diffusion_steps", type=int, default=1000)
    ap.add_argument("--sampling_steps", type=int, default=100)
    ap.add_argument("--lambda_data", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=0.85)

    ap.add_argument("--pp_solver_iters", type=int, default=10)
    ap.add_argument("--pp_lr_scale", type=float, default=0.75)

    ap.add_argument("--t_total", type=float, default=1.0)
    ap.add_argument("--dark_count", type=float, default=7.74e-4)
    ap.add_argument("--n_spad_frames", type=int, default=100_000)
    ap.add_argument("--target_ppp", type=str, default="0.05", help="Comma-separated, e.g. 0.05,0.01,0.005")
    ap.add_argument("--infer_batch_size", type=int, default=4)
    ap.add_argument("--save_binary", action="store_true", help="Also save raw binary detections [num_samples, n_spad_frames]")
    ap.add_argument(
        "--x_param",
        type=str,
        default="log",
        choices=["log", "log1p"],
        help="Model parameterization for x; 'log' means phi=exp(x) (default, matches most training).",
    )
    ap.add_argument(
        "--normalize_flux",
        action="store_true",
        default=True,
        help="Normalize each flux sample by its max then scale to --flux_peak (default: True).",
    )
    ap.add_argument(
        "--no_normalize_flux",
        action="store_true",
        help="Disable flux normalization.",
    )
    ap.add_argument(
        "--flux_peak",
        type=float,
        default=10000.0,
        help="Peak value after normalization (default: 10000, matching notebook).",
    )

    args = ap.parse_args()

    if args.no_normalize_flux:
        args.normalize_flux = False

    ensure_dir(args.output_dir)
    maybe_add_to_syspath(args.improved_diffusion_root)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset and choose samples.
    log_flux_ds = load_log_flux_dataset(args.dataset)
    if log_flux_ds.ndim != 2:
        raise ValueError(f"Expected dataset with shape [N, L], got {tuple(log_flux_ds.shape)}")
    n_total, l_total = int(log_flux_ds.shape[0]), int(log_flux_ds.shape[1])
    if args.sequence_length > l_total:
        raise ValueError(f"--sequence_length {args.sequence_length} exceeds dataset length {l_total}")
    idx = sample_indices(n_total, args.num_samples, args.seed)
    np.save(os.path.join(args.output_dir, "sample_indices.npy"), idx.astype(np.int64))

    # Load model once.
    model, diffusion = load_temporal_diffusion_model(
        checkpoint_path=args.checkpoint,
        sequence_length=args.sequence_length,
        num_channels=args.num_channels,
        diffusion_steps=args.diffusion_steps,
        device=device,
    )

    # Precompute bin edges/sizes.
    bin_edges = np.linspace(0, args.n_spad_frames, args.sequence_length + 1, dtype=np.int64)
    bin_sizes = np.diff(bin_edges).astype(np.float32)  # [seq_len]

    # Prepare the chosen flux samples.
    # Dataset contains log(flux). Convert to linear-space flux.
    log_flux_raw = log_flux_ds[idx, : args.sequence_length].numpy().astype(np.float64)
    flux_sel = np.exp(log_flux_raw)

    # Match notebook convention: normalize each sample to [0, flux_peak].
    if args.normalize_flux:
        mx = np.max(flux_sel, axis=1, keepdims=True)
        mx = np.where(mx > 0, mx, 1.0)
        flux_sel = (flux_sel / mx) * float(args.flux_peak)

    # Recompute log-flux from the (possibly rescaled) linear flux.
    # This is what the model's output space represents.
    log_flux_sel = x_from_flux(flux_sel, args.x_param)

    print(f"x_param: {args.x_param}")
    print(f"normalize_flux: {args.normalize_flux}, flux_peak: {args.flux_peak}")
    print(f"Flux range across dataset: [{flux_sel.min():.1f}, {flux_sel.max():.1f}]")
    print(f"x(flux) range: [{log_flux_sel.min():.2f}, {log_flux_sel.max():.2f}]")

    target_ppps = parse_ppp_list(args.target_ppp)
    for ppp in target_ppps:
        print(f"\n=== PPP={ppp} | frames={args.n_spad_frames} | samples={args.num_samples} ===")

        # 1) Generate detections (binary) and binned counts.
        binary_all: Optional[np.ndarray] = None
        if args.save_binary:
            binary_all = np.zeros((args.num_samples, args.n_spad_frames), dtype=np.uint8)
        bin_counts_all = np.zeros((args.num_samples, args.sequence_length), dtype=np.float32)
        ppp_scales = np.zeros((args.num_samples,), dtype=np.float32)

        for i in tqdm(range(args.num_samples), desc="Simulating SPAD", total=args.num_samples):
            flux_i = flux_sel[i]
            ppp_scales[i] = compute_ppp_scale_for_flux(flux_i, float(ppp))
            _, binary_i = generate_photon_arrivals_spad(
                flux_i,
                target_ppp=float(ppp),
                d=float(args.dark_count),
                T=float(args.t_total),
                seq_length=int(args.n_spad_frames),
                return_binary=True,
            )
            if binary_all is not None:
                binary_all[i] = binary_i
            bin_counts_all[i] = bin_spad_binary_with_edges(binary_i, bin_edges)

        # Save detections.
        tag = f"ppp{ppp:g}_frames{args.n_spad_frames}_bins{args.sequence_length}_n{args.num_samples}"
        np.save(os.path.join(args.output_dir, f"bin_counts_{tag}.npy"), bin_counts_all.astype(np.float32))
        np.save(os.path.join(args.output_dir, f"bin_sizes_{tag}.npy"), bin_sizes.astype(np.float32))
        np.save(os.path.join(args.output_dir, f"ppp_scale_{tag}.npy"), ppp_scales.astype(np.float32))
        if binary_all is not None:
            np.save(os.path.join(args.output_dir, f"binary_{tag}.npy"), binary_all)

        # 2) Run batched inference.
        log_flux_hat = np.zeros((args.num_samples, args.sequence_length), dtype=np.float32)
        with torch.set_grad_enabled(True):
            for sl in tqdm(list(chunked_indices(args.num_samples, args.infer_batch_size)), desc="Inference"):
                bc = torch.from_numpy(bin_counts_all[sl]).to(device)
                bs = torch.from_numpy(np.repeat(bin_sizes[None, :], bc.shape[0], axis=0)).to(device)
                ppp_s = torch.from_numpy(ppp_scales[sl]).to(device)

                x_hat = sample_diffpir_photon_flux(
                    model=model,
                    diffusion=diffusion,
                    bin_counts=bc,
                    bin_sizes=bs,
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
                    sequence_length=int(args.sequence_length),
                    device=device,
                    show_progress=False,
                )
                log_flux_hat[sl] = x_hat[:, 0, :].detach().cpu().numpy().astype(np.float32)

        np.save(os.path.join(args.output_dir, f"log_flux_hat_{tag}.npy"), log_flux_hat)

        # 3) Metrics (drop any sample with any NaN/inf metric).
        per_sample = {
            "MSE": np.full((args.num_samples,), np.nan, dtype=np.float64),
            "Relative MSE": np.full((args.num_samples,), np.nan, dtype=np.float64),
            "MAE": np.full((args.num_samples,), np.nan, dtype=np.float64),
            "Relative MAE": np.full((args.num_samples,), np.nan, dtype=np.float64),
            "Correlation": np.full((args.num_samples,), np.nan, dtype=np.float64),
            "valid": np.zeros((args.num_samples,), dtype=np.bool_),
        }

        for i in range(args.num_samples):
            flux_true_i = flux_sel[i].astype(np.float64)
            x_hat_i = log_flux_hat[i].astype(np.float64)
            if args.x_param == "log":
                flux_hat_i = np.exp(x_hat_i)
            elif args.x_param == "log1p":
                flux_hat_i = np.expm1(x_hat_i)
            else:
                flux_hat_i = np.exp(x_hat_i)
            flux_hat_i = np.maximum(flux_hat_i, 0.0)
            m = compute_metrics(flux_true_i, flux_hat_i)
            if not is_finite_metrics(m):
                continue
            d = m.as_dict()
            for k in ["MSE", "Relative MSE", "MAE", "Relative MAE", "Correlation"]:
                per_sample[k][i] = d[k]
            per_sample["valid"][i] = True

        valid_mask = per_sample["valid"]
        n_valid = int(valid_mask.sum())
        print(f"Valid metric samples: {n_valid}/{args.num_samples} (dropped {args.num_samples - n_valid})")

        # Aggregate over valid only.
        agg: Dict[str, float] = {}
        for k in ["MSE", "Relative MSE", "MAE", "Relative MAE", "Correlation"]:
            vals = per_sample[k][valid_mask]
            agg[k] = float(np.mean(vals)) if vals.size else float("nan")

        print("Average metrics (valid only):")
        for k, v in agg.items():
            print(f"  {k:14s}: {v:.6f}")

        np.savez(
            os.path.join(args.output_dir, f"metrics_{tag}.npz"),
            **{k: v for k, v in per_sample.items()},
            avg_MSE=agg["MSE"],
            avg_Relative_MSE=agg["Relative MSE"],
            avg_MAE=agg["MAE"],
            avg_Relative_MAE=agg["Relative MAE"],
            avg_Correlation=agg["Correlation"],
        )


if __name__ == "__main__":
    main()