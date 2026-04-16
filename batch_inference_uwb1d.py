"""
Batch inference for 1D temporal photon flux estimation with UWB (uwb1d_quanta).

What this script does:
- Loads a `log_flux_dataset.pt` tensor of shape [N, L] (log(flux) samples).
- Randomly samples `--num_samples` items.
- For each TARGET_PPP level:
  - Simulates SPAD binary detections using the physical model
    (flux scaled so mean(φ)*dt_frame = target_ppp).
  - Runs UWB 1D reconstruction via ihpp_fft.uwb1d_quanta on binary frames.
  - Computes metrics between the UWB output and the physically-rescaled
    ground-truth flux (flux_gt), matching the notebook convention.
  - Saves downsampled reconstructions + per-sample metrics.
- Writes a consolidated metrics_all.json after all PPP levels.

python batch_inference_uwb1d.py \\
  --dataset "./data/step_log_flux_dataset.pt" \\
  --output_dir "./outputs/batch_uwb1d_01" \\
  --num_samples 2000 \\
  --target_ppp "0.01,0.05,0.1,0.5,1.0" \\
  --n_spad_frames 100000
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
# SPAD simulation — physical parameterization (matches updated notebook)
# ---------------------------------------------------------------------------

def generate_photon_arrivals_spad_physical(
    flux,
    target_ppp: float = 1.0,
    d: float = 0.0,
    T: float = 1.0,
    seq_length=None,
    return_binary: bool = False,
):
    """
    Physical SPAD model: scale flux so mean(φ) * dt_frame = target_ppp.

    Returns:
        arrivals: photon arrival times (seconds)
        flux_gt: rescaled ground-truth flux (photons/sec) at seq_length resolution
        (optional) binary: [seq_length] binary detection array
    """
    flux = np.asarray(flux, dtype=np.float64).ravel()
    n_src = len(flux)
    if seq_length is None:
        seq_length = n_src

    dt_frame = T / seq_length

    # Interpolate flux onto frame grid if needed.
    if seq_length != n_src:
        t_src = np.linspace(0, T, n_src)
        t_dst = np.linspace(0, T, seq_length)
        flux = np.interp(t_dst, t_src, flux)

    # Scale flux so that mean(φ) * dt_frame = target_ppp.
    flux_mean = flux.mean()
    if flux_mean > 0:
        flux_gt = flux * (target_ppp / (flux_mean * dt_frame))
    else:
        flux_gt = flux.copy()

    # Physical expected photons per frame: N(t) = φ(t) * dt + d.
    N_t = flux_gt * dt_frame  # mean(N_t) = target_ppp by construction

    # SPAD detection model.
    detection_prob = 1.0 - np.exp(-(N_t + d))
    binary = (np.random.random(seq_length) < detection_prob).astype(np.uint8)

    # Arrival times.
    detection_indices = np.where(binary == 1)[0]
    dt = T / (seq_length - 1) if seq_length > 1 else T
    arrivals = detection_indices * dt

    result = [arrivals, flux_gt]
    if return_binary:
        result.append(binary)
    return tuple(result)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    flux_true: np.ndarray,
    flux_pred: np.ndarray,
    T_exp: Optional[float] = None,
) -> Metrics:
    """
    Reconstruction metrics.  If lengths differ, resample flux_pred onto the
    ground-truth time grid (matching the notebook convention).
    """
    flux_true = np.asarray(flux_true, dtype=np.float64).ravel()
    flux_pred = np.asarray(flux_pred, dtype=np.float64).ravel()

    if flux_true.shape != flux_pred.shape:
        if T_exp is None:
            raise ValueError(
                "flux_true and flux_pred have different lengths; pass T_exp to resample."
            )
        t_gt = np.linspace(0, T_exp, len(flux_true))
        t_pr = np.linspace(0, T_exp, len(flux_pred))
        flux_pred = np.interp(t_gt, t_pr, flux_pred)

    mse = np.mean((flux_true - flux_pred) ** 2)
    denom_sq = np.mean(flux_true ** 2)
    rel_mse = mse / denom_sq if denom_sq > 0 else np.nan
    mae = np.mean(np.abs(flux_true - flux_pred))
    denom_abs = np.mean(np.abs(flux_true))
    rel_mae = mae / denom_abs if denom_abs > 0 else np.nan
    corr = np.corrcoef(flux_true, flux_pred)[0, 1]
    return Metrics(mse=mse, rel_mse=rel_mse, mae=mae, rel_mae=rel_mae, corr=corr)


def is_finite_metrics(m: Metrics) -> bool:
    arr = np.array([m.mse, m.rel_mse, m.mae, m.rel_mae, m.corr], dtype=np.float64)
    return bool(np.all(np.isfinite(arr)))


# ---------------------------------------------------------------------------
# Dataset loading
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
    ap.add_argument("--dataset", type=str, required=True,
                    help="Path to log_flux_dataset.pt (tensor [N, L])")
    ap.add_argument("--output_dir", type=str, required=True,
                    help="Directory to save reconstructions/metrics")
    ap.add_argument("--uwb3d_root", type=str, default=None,
                    help="Optional path to add to sys.path for uwb3d")

    ap.add_argument("--num_samples", type=int, required=True,
                    help="Number of random samples to run")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--sequence_length", type=int, default=1024,
                    help="Ground-truth flux length (from dataset)")

    # UWB parameters
    ap.add_argument("--nt", type=int, default=None,
                    help="UWB output time-grid resolution "
                         "(default: same as --n_spad_frames, matching notebook)")
    ap.add_argument("--cuda", action="store_true", default=True,
                    help="Use CUDA for uwb1d_quanta (default: True)")
    ap.add_argument("--no_cuda", action="store_true",
                    help="Disable CUDA for uwb1d_quanta")

    # SPAD simulation parameters
    ap.add_argument("--t_total", type=float, default=1.0)
    ap.add_argument("--dark_count", type=float, default=0.0,
                    help="Spurious detection rate per frame (default: 0.0, matching notebook)")
    ap.add_argument("--n_spad_frames", type=int, default=100_000)
    ap.add_argument("--target_ppp", type=str, default="0.05",
                    help="Comma-separated, e.g. 0.05,0.01,0.005")
    ap.add_argument("--save_binary", action="store_true",
                    help="Also save raw binary detections [num_samples, n_spad_frames]")

    ap.add_argument(
        "--normalize_flux",
        action="store_true",
        default=False,
        help="Normalize each flux sample by its max then scale to --flux_peak "
             "(default: False, matching updated notebook).",
    )
    ap.add_argument(
        "--flux_peak",
        type=float,
        default=10000.0,
        help="Peak value after normalization (only used with --normalize_flux).",
    )

    args = ap.parse_args()

    if args.no_cuda:
        args.cuda = False

    # Nt defaults to n_spad_frames (matches notebook: Nt = 100_000 = N_SPAD_FRAMES).
    if args.nt is None:
        args.nt = args.n_spad_frames

    ensure_dir(args.output_dir)
    maybe_add_to_syspath(args.uwb3d_root)
    set_seed(args.seed)

    # Lazy import — uwb3d_root may have been added to sys.path above.
    from uwb3d import ihpp_fft  # noqa: E402

    # ------------------------------------------------------------------
    # Load dataset and choose samples.
    # ------------------------------------------------------------------
    log_flux_ds = load_log_flux_dataset(args.dataset)
    if log_flux_ds.ndim != 2:
        raise ValueError(f"Expected dataset with shape [N, L], got {tuple(log_flux_ds.shape)}")
    n_total, l_total = int(log_flux_ds.shape[0]), int(log_flux_ds.shape[1])
    if args.sequence_length > l_total:
        raise ValueError(
            f"--sequence_length {args.sequence_length} exceeds dataset length {l_total}"
        )
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
    print(f"Nt (UWB output resolution): {args.nt}")
    print(f"CUDA: {args.cuda}")

    # ------------------------------------------------------------------
    # Loop over PPP levels.
    # ------------------------------------------------------------------
    target_ppps = parse_ppp_list(args.target_ppp)
    all_ppp_results: List[Dict] = []  # collect results for consolidated log

    for ppp in target_ppps:
        print(f"\n=== PPP={ppp} | frames={args.n_spad_frames} | samples={args.num_samples} ===")

        tag = (f"ppp{ppp:g}_frames{args.n_spad_frames}"
               f"_len{args.sequence_length}_n{args.num_samples}")

        # Downsampled arrays for storage (at sequence_length resolution).
        recon_ds_all = np.zeros(
            (args.num_samples, args.sequence_length), dtype=np.float32
        )
        flux_gt_ds_all = np.zeros(
            (args.num_samples, args.sequence_length), dtype=np.float32
        )

        per_sample: Dict[str, np.ndarray] = {
            "MSE": np.full((args.num_samples,), np.nan, dtype=np.float64),
            "Relative MSE": np.full((args.num_samples,), np.nan, dtype=np.float64),
            "MAE": np.full((args.num_samples,), np.nan, dtype=np.float64),
            "Relative MAE": np.full((args.num_samples,), np.nan, dtype=np.float64),
            "Correlation": np.full((args.num_samples,), np.nan, dtype=np.float64),
            "valid": np.zeros((args.num_samples,), dtype=np.bool_),
        }

        # Optional full-resolution binary storage.
        binary_all: Optional[np.ndarray] = None
        if args.save_binary:
            binary_all = np.zeros(
                (args.num_samples, args.n_spad_frames), dtype=np.uint8
            )

        # ---- Per-sample: SPAD simulation → UWB inference → metrics ----
        for i in tqdm(range(args.num_samples), desc="SPAD + UWB",
                      total=args.num_samples):
            flux_i = flux_sel[i]

            # 1) Physical SPAD simulation.
            result_i = generate_photon_arrivals_spad_physical(
                flux_i,
                target_ppp=float(ppp),
                d=float(args.dark_count),
                T=float(args.t_total),
                seq_length=int(args.n_spad_frames),
                return_binary=True,
            )
            _arrivals_i, flux_gt_i, binary_i = result_i

            if binary_all is not None:
                binary_all[i] = binary_i

            # 2) UWB reconstruction on binary frames.
            frames_1d = binary_i.reshape(-1, 1, 1)
            recon = ihpp_fft.uwb1d_quanta(
                frames_1d,
                T_exp=float(args.t_total),
                Nt=int(args.nt),
                cuda=args.cuda,
                load_frames_on_vram=True,
                load_output_on_vram=True,
            )
            recon = np.asarray(np.squeeze(recon), dtype=np.float64).ravel()

            # 3) Metrics at full resolution (matching notebook).
            m = compute_metrics(flux_gt_i, recon, T_exp=float(args.t_total))
            if is_finite_metrics(m):
                d_m = m.as_dict()
                for k in ["MSE", "Relative MSE", "MAE", "Relative MAE",
                           "Correlation"]:
                    per_sample[k][i] = d_m[k]
                per_sample["valid"][i] = True

            # 4) Downsample for storage.
            t_ds = np.linspace(0, args.t_total, args.sequence_length)
            t_gt = np.linspace(0, args.t_total, len(flux_gt_i))
            t_rc = np.linspace(0, args.t_total, len(recon))
            flux_gt_ds_all[i] = np.interp(t_ds, t_gt, flux_gt_i).astype(
                np.float32
            )
            recon_ds_all[i] = np.interp(t_ds, t_rc, recon).astype(np.float32)

        # ---- Save arrays ----
        np.save(
            os.path.join(args.output_dir, f"recon_rate_{tag}.npy"),
            recon_ds_all,
        )
        np.save(
            os.path.join(args.output_dir, f"flux_gt_{tag}.npy"),
            flux_gt_ds_all,
        )
        if binary_all is not None:
            np.save(
                os.path.join(args.output_dir, f"binary_{tag}.npy"), binary_all
            )

        # ---- Aggregate metrics ----
        valid_mask = per_sample["valid"]
        n_valid = int(valid_mask.sum())
        print(
            f"Valid metric samples: {n_valid}/{args.num_samples} "
            f"(dropped {args.num_samples - n_valid})"
        )

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
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "config": vars(args),
                    "results": ppp_entry,
                },
                f,
                indent=2,
            )
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
            "nt": args.nt,
            "t_total": args.t_total,
            "dark_count": args.dark_count,
            "normalize_flux": args.normalize_flux,
            "flux_peak": args.flux_peak,
            "cuda": args.cuda,
        },
        "results": all_ppp_results,
    }
    consolidated_path = os.path.join(args.output_dir, "metrics_all.json")
    with open(consolidated_path, "w") as f:
        json.dump(consolidated, f, indent=2)

    # Print summary table.
    print(f"\n{'=' * 70}")
    print(
        f"{'PPP':>8s} | {'MSE':>14s} | {'Rel MSE':>12s} | "
        f"{'MAE':>12s} | {'Rel MAE':>12s} | {'Corr':>10s}"
    )
    print(f"{'-' * 70}")
    for entry in all_ppp_results:
        m = entry["avg_metrics"]
        print(
            f"{entry['ppp']:8g} | {m['MSE']:14.6f} | "
            f"{m['Relative MSE']:12.6f} | {m['MAE']:12.6f} | "
            f"{m['Relative MAE']:12.6f} | {m['Correlation']:10.6f}"
        )
    print(f"{'=' * 70}")
    print(f"Consolidated log saved to {consolidated_path}")


if __name__ == "__main__":
    main()
