"""
Batch inference for 1D temporal photon flux estimation with UWB (1D NUFFT + CFAR).

What this script does:
- Loads a `log_flux_dataset.pt` tensor of shape [N, L] (log(flux) samples).
- Randomly samples `--num_samples` items.
- For each TARGET_PPP level:
  - Simulates SPAD binary detections with `--n_spad_frames` frames.
  - Converts binary detections to arrival timestamps.
  - Runs UWB 1D reconstruction (NUFFT + CFAR thresholding + inverse NUFFT).
  - Inverts the SPAD forward model to recover flux from the estimated rate.
  - Computes metrics, dropping any sample with any NaN/inf metric entry.
  - Saves detections + reconstructions + per-sample metrics.

python batch_inference_uwb1d.py \
  --dataset "./data/step_log_flux_dataset.pt" \
  --output_dir "./outputs/batch_uwb1d_01" \
  --num_samples 2000 \
  --target_ppp "0.1" \
  --n_spad_frames 100000 \
  --save_binary
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

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


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


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


# ---------------------------------------------------------------------------
# SPAD simulation (identical to batch_inference.py)
# ---------------------------------------------------------------------------

def generate_photon_arrivals_spad(
    flux,
    target_ppp: float = 1.0,
    d: float = 7.74e-4,
    T: float = 1.0,
    return_binary=False,
    return_flux_scaled=False,
    seq_length=None,
):
    """
    Generate photon arrivals using SPAD-style Poisson sampling.

    1. Scale flux to achieve target average photons per pixel (PPP).
    2. Apply Poisson sampling: Pr{detection} = 1 - exp(-N(t)).
    3. Convert binary detections to arrival times.
    """
    flux = np.asarray(flux, dtype=np.float64).ravel()
    n_src = len(flux)
    if seq_length is None:
        seq_length = n_src
    elif seq_length != n_src:
        t_src = np.linspace(0.0, T, n_src)
        t_dst = np.linspace(0.0, T, seq_length)
        flux = np.interp(t_dst, t_src, flux)

    flux_normalized = flux.copy()
    if flux_normalized.max() > 1.0:
        flux_normalized = flux_normalized / flux_normalized.max()

    I_mean = flux_normalized.mean()
    a = target_ppp / I_mean if I_mean > 0 else 1.0

    flux_scaled = a * flux_normalized + d
    detection_prob = 1.0 - np.exp(-flux_scaled)
    binary = (np.random.random(seq_length) < detection_prob).astype(np.uint8)

    detection_indices = np.where(binary == 1)[0]
    dt = T / (seq_length - 1) if seq_length > 1 else T
    arrivals = detection_indices * dt

    result = [arrivals]
    if return_binary:
        result.append(binary)
    if return_flux_scaled:
        result.append(flux_scaled)
    return result[0] if len(result) == 1 else tuple(result)


def compute_ppp_scale_for_flux(flux: np.ndarray, target_ppp: float) -> float:
    """
    Reproduces the mapping implied by SPAD generation:
      flux_normalized = flux / flux_max
      a = target_ppp / mean(flux_normalized)
      ppp_scale = a / flux_max
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


# ---------------------------------------------------------------------------
# UWB 1D reconstruction
# ---------------------------------------------------------------------------

def simple_ihpp_uwb1d(
    stamps,
    T_exp: float = 1.0,
    freqs=(0, 5000, 1),
    cfarscale: float = 2.0,
    percentage_in: float = 0.99,
    target_len: int = 1024,
    verbose: bool = False,
):
    """
    Minimal 1D UWB via NUFFT -> CFAR threshold -> inverse NUFFT.

    Args:
        stamps: 1D array of arrival timestamps (seconds).
        T_exp: total duration of the trace (seconds).
        freqs: (start, stop, step) tuple for probing frequencies.
        cfarscale: scales the CFAR amplitude bound.
        percentage_in: percentage of true frequencies to keep in bound calc.
        target_len: length of reconstructed time grid.
        verbose: pass-through to uwb_utils functions.

    Returns:
        recon: time-domain reconstruction (len=target_len)
        F: raw NUFFT coefficients
        F_thresh: thresholded coefficients
        probed_freqs: frequency grid
        bound: amplitude threshold used
    """
    from uwb3d import uwb_utils

    stamps = np.asarray(stamps, dtype=np.float64).ravel()
    if stamps.size == 0:
        return (
            np.zeros(target_len),
            np.array([]),
            np.array([]),
            np.array([]),
            np.nan,
        )

    stamps = np.clip(stamps - stamps.min(), 0.0, T_exp)

    F, probed_freqs = uwb_utils.probe_frequencies_sweep(
        stamps=stamps,
        time_total=T_exp,
        freqs=freqs,
        return_freqs=True,
        verbose=verbose,
    )
    amplitudes = np.abs(F)
    bound = uwb_utils.compute_amplitude_bound(stamps, T_exp, percentage_in) * cfarscale
    keep_mask = amplitudes > bound
    F_thresh = F * keep_mask

    t_grid = np.linspace(0, T_exp, target_len, endpoint=False)
    recon = uwb_utils.reconstruct_rate_function(
        t_grid, probed_freqs, F_thresh, verbose=verbose,
    ).real

    return recon, F, F_thresh, probed_freqs, bound


def uwb_rate_to_flux(
    recon: np.ndarray,
    fps: float,
    ppp_scale: float,
    dark_count: float = 7.74e-4,
) -> np.ndarray:
    """
    Invert SPAD forward model to recover physical flux from UWB rate estimate.

    The SPAD model gives:
        detection_prob(t) = 1 - exp(-(ppp_scale * flux(t) + dark_count))
        rate(t) = detection_prob(t) * fps

    Inversion:
        detection_prob = rate / fps
        N = -log(1 - detection_prob)
        flux = (N - dark_count) / ppp_scale
    """
    p_hat = np.clip(recon / fps, 1e-12, 1 - 1e-8)
    N_hat = -np.log(1.0 - p_hat)
    flux_hat = np.maximum((N_hat - dark_count) / ppp_scale, 0.0)
    return flux_hat


# ---------------------------------------------------------------------------
# Metrics (identical to batch_inference.py)
# ---------------------------------------------------------------------------

def compute_metrics(flux_true: np.ndarray, flux_pred: np.ndarray) -> Metrics:
    mse = np.mean((flux_true - flux_pred) ** 2)
    rel_mse = mse / np.mean(flux_true ** 2)
    mae = np.mean(np.abs(flux_true - flux_pred))
    rel_mae = mae / np.mean(np.abs(flux_true))
    corr = np.corrcoef(flux_true, flux_pred)[0, 1]
    return Metrics(mse=mse, rel_mse=rel_mse, mae=mae, rel_mae=rel_mae, corr=corr)


def is_finite_metrics(m: Metrics) -> bool:
    arr = np.array([m.mse, m.rel_mse, m.mae, m.rel_mae, m.corr], dtype=np.float64)
    return bool(np.all(np.isfinite(arr)))


# ---------------------------------------------------------------------------
# Dataset loading (identical to batch_inference.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, help="Path to log_flux_dataset.pt (tensor [N, L])")
    ap.add_argument("--output_dir", type=str, required=True, help="Directory to save reconstructions/metrics")
    ap.add_argument("--uwb3d_root", type=str, default=None, help="Optional path to add to sys.path for uwb3d")

    ap.add_argument("--num_samples", type=int, required=True, help="Number of random samples to run")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--sequence_length", type=int, default=1024)

    # UWB parameters
    ap.add_argument("--cfarscale", type=float, default=2.0, help="CFAR amplitude bound scaling factor")
    ap.add_argument("--percentage_in", type=float, default=None,
                    help="Percentage for CFAR bound calc (default: auto = 1 - 1/n_freqs)")
    ap.add_argument("--freq_step", type=int, default=1, help="Step between probed frequencies")
    ap.add_argument("--max_freq", type=float, default=None,
                    help="Max probing frequency in Hz (default: Nyquist = n_spad_frames / (2 * t_total))")

    # SPAD simulation parameters
    ap.add_argument("--t_total", type=float, default=1.0)
    ap.add_argument("--dark_count", type=float, default=7.74e-4)
    ap.add_argument("--n_spad_frames", type=int, default=100_000)
    ap.add_argument("--target_ppp", type=str, default="0.05", help="Comma-separated, e.g. 0.05,0.01,0.005")
    ap.add_argument("--save_binary", action="store_true", help="Also save raw binary detections [num_samples, n_spad_frames]")

    ap.add_argument(
        "--normalize_flux",
        action="store_true",
        default=True,
        help="Normalize each flux sample by its max then scale to --flux_peak (default: True).",
    )
    ap.add_argument("--no_normalize_flux", action="store_true", help="Disable flux normalization.")
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
    maybe_add_to_syspath(args.uwb3d_root)
    set_seed(args.seed)

    # ------------------------------------------------------------------
    # Load dataset and choose samples.
    # ------------------------------------------------------------------
    log_flux_ds = load_log_flux_dataset(args.dataset)
    if log_flux_ds.ndim != 2:
        raise ValueError(f"Expected dataset with shape [N, L], got {tuple(log_flux_ds.shape)}")
    n_total, l_total = int(log_flux_ds.shape[0]), int(log_flux_ds.shape[1])
    if args.sequence_length > l_total:
        raise ValueError(f"--sequence_length {args.sequence_length} exceeds dataset length {l_total}")
    idx = sample_indices(n_total, args.num_samples, args.seed)
    np.save(os.path.join(args.output_dir, "sample_indices.npy"), idx.astype(np.int64))

    # Prepare the chosen flux samples (dataset stores log(flux)).
    log_flux_raw = log_flux_ds[idx, : args.sequence_length].numpy().astype(np.float64)
    flux_sel = np.exp(log_flux_raw)

    if args.normalize_flux:
        mx = np.max(flux_sel, axis=1, keepdims=True)
        mx = np.where(mx > 0, mx, 1.0)
        flux_sel = (flux_sel / mx) * float(args.flux_peak)

    print(f"normalize_flux: {args.normalize_flux}, flux_peak: {args.flux_peak}")
    print(f"Flux range across dataset: [{flux_sel.min():.1f}, {flux_sel.max():.1f}]")

    # ------------------------------------------------------------------
    # UWB frequency setup.
    # ------------------------------------------------------------------
    fps = args.n_spad_frames / args.t_total
    max_freq = args.max_freq if args.max_freq is not None else fps / 2.0
    freq_slice = (0, int(max_freq) + args.freq_step, args.freq_step)
    n_freqs = len(np.arange(*freq_slice))

    # Percentage_in: default to 1 - 1/n_freqs (matches notebook convention).
    pct_in = args.percentage_in if args.percentage_in is not None else 1.0 - 1.0 / max(1, n_freqs)

    print(f"fps: {fps:.1f}")
    print(f"Frequency range: (0, {int(max_freq) + args.freq_step}, {args.freq_step})  "
          f"({n_freqs} frequencies)")
    print(f"CFAR scale: {args.cfarscale}, percentage_in: {pct_in:.8f}")

    # ------------------------------------------------------------------
    # Loop over PPP levels.
    # ------------------------------------------------------------------
    target_ppps = parse_ppp_list(args.target_ppp)
    all_ppp_results: List[Dict] = []  # collect results for consolidated log
    for ppp in target_ppps:
        print(f"\n=== PPP={ppp} | frames={args.n_spad_frames} | samples={args.num_samples} ===")

        # 1) Generate SPAD binary detections and extract arrival timestamps.
        binary_all: Optional[np.ndarray] = None
        if args.save_binary:
            binary_all = np.zeros((args.num_samples, args.n_spad_frames), dtype=np.uint8)
        ppp_scales = np.zeros((args.num_samples,), dtype=np.float32)
        all_stamps: List[np.ndarray] = []

        for i in tqdm(range(args.num_samples), desc="Simulating SPAD", total=args.num_samples):
            flux_i = flux_sel[i]
            ppp_scales[i] = compute_ppp_scale_for_flux(flux_i, float(ppp))
            arrivals_i, binary_i = generate_photon_arrivals_spad(
                flux_i,
                target_ppp=float(ppp),
                d=float(args.dark_count),
                T=float(args.t_total),
                seq_length=int(args.n_spad_frames),
                return_binary=True,
            )
            if binary_all is not None:
                binary_all[i] = binary_i
            all_stamps.append(arrivals_i)

        # Save detections.
        tag = f"ppp{ppp:g}_frames{args.n_spad_frames}_len{args.sequence_length}_n{args.num_samples}"
        np.save(os.path.join(args.output_dir, f"ppp_scale_{tag}.npy"), ppp_scales.astype(np.float32))
        if binary_all is not None:
            np.save(os.path.join(args.output_dir, f"binary_{tag}.npy"), binary_all)

        # 2) Run UWB 1D inference.
        recon_rate_all = np.zeros((args.num_samples, args.sequence_length), dtype=np.float64)
        flux_hat_all = np.zeros((args.num_samples, args.sequence_length), dtype=np.float64)

        for i in tqdm(range(args.num_samples), desc="UWB Inference", total=args.num_samples):
            recon, _, _, _, _ = simple_ihpp_uwb1d(
                stamps=all_stamps[i],
                T_exp=args.t_total,
                freqs=freq_slice,
                cfarscale=args.cfarscale,
                percentage_in=pct_in,
                target_len=args.sequence_length,
                verbose=False,
            )
            recon_rate_all[i] = recon
            flux_hat_all[i] = uwb_rate_to_flux(
                recon, fps, float(ppp_scales[i]), float(args.dark_count),
            )

        np.save(os.path.join(args.output_dir, f"recon_rate_{tag}.npy"), recon_rate_all.astype(np.float32))
        np.save(os.path.join(args.output_dir, f"flux_hat_{tag}.npy"), flux_hat_all.astype(np.float32))

        # 3) Metrics (drop any sample with any NaN/inf metric).
        per_sample: Dict[str, np.ndarray] = {
            "MSE": np.full((args.num_samples,), np.nan, dtype=np.float64),
            "Relative MSE": np.full((args.num_samples,), np.nan, dtype=np.float64),
            "MAE": np.full((args.num_samples,), np.nan, dtype=np.float64),
            "Relative MAE": np.full((args.num_samples,), np.nan, dtype=np.float64),
            "Correlation": np.full((args.num_samples,), np.nan, dtype=np.float64),
            "valid": np.zeros((args.num_samples,), dtype=np.bool_),
        }

        for i in range(args.num_samples):
            flux_true_i = flux_sel[i].astype(np.float64)
            flux_hat_i = np.maximum(flux_hat_all[i], 0.0)
            m = compute_metrics(flux_true_i, flux_hat_i)
            if not is_finite_metrics(m):
                continue
            d = m.as_dict()
            for k in ["MSE", "Relative MSE", "MAE", "Relative MAE", "Correlation"]:
                per_sample[k][i] = d[k]
            per_sample["valid"][i] = True

        valid_mask = per_sample["valid"]
        n_valid = int(valid_mask.sum())
        print(f"Valid metric samples: {n_valid}/{args.num_samples} "
              f"(dropped {args.num_samples - n_valid})")

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

        # Per-PPP JSON log.
        ppp_entry = {
            "ppp": ppp,
            "n_valid": n_valid,
            "n_total": args.num_samples,
            "n_dropped": args.num_samples - n_valid,
            "avg_metrics": agg,
        }
        all_ppp_results.append(ppp_entry)

        log_path = os.path.join(args.output_dir, f"metrics_{tag}.json")
        with open(log_path, "w") as f:
            json.dump({"timestamp": datetime.now().isoformat(),
                        "config": vars(args), "results": ppp_entry}, f, indent=2)
        print(f"Log saved to {log_path}")

    # ------------------------------------------------------------------
    # Consolidated log: all PPP levels in one file.
    # ------------------------------------------------------------------
    consolidated = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "dataset": args.dataset,
            "num_samples": args.num_samples,
            "seed": args.seed,
            "sequence_length": args.sequence_length,
            "n_spad_frames": args.n_spad_frames,
            "t_total": args.t_total,
            "dark_count": args.dark_count,
            "cfarscale": args.cfarscale,
            "freq_step": args.freq_step,
            "max_freq": args.max_freq,
            "normalize_flux": args.normalize_flux,
            "flux_peak": args.flux_peak,
        },
        "results": all_ppp_results,
    }
    consolidated_path = os.path.join(args.output_dir, "metrics_all.json")
    with open(consolidated_path, "w") as f:
        json.dump(consolidated, f, indent=2)

    # Print summary table.
    print(f"\n{'=' * 70}")
    print(f"{'PPP':>8s} | {'MSE':>14s} | {'Rel MSE':>12s} | {'MAE':>12s} | {'Rel MAE':>12s} | {'Corr':>10s}")
    print(f"{'-' * 70}")
    for entry in all_ppp_results:
        m = entry["avg_metrics"]
        print(f"{entry['ppp']:8g} | {m['MSE']:14.6f} | {m['Relative MSE']:12.6f} | "
              f"{m['MAE']:12.6f} | {m['Relative MAE']:12.6f} | {m['Correlation']:10.6f}")
    print(f"{'=' * 70}")
    print(f"Consolidated log saved to {consolidated_path}")


if __name__ == "__main__":
    main()
