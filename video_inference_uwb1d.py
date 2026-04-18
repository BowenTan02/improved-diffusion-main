"""
UWB1D video inference: recover per-pixel photon flux from 1D SPAD observations
using the Ultra-Wideband 1D (UWB1D) FFT baseline.

Input : a `log_flux.pt` tensor of shape [T, H, W]  (log-flux video).
Output: a reconstructed MP4 video at 30 fps, plus a SPAD-observation video.

Pipeline (per pixel):
  1. Convert log-flux -> linear flux, normalize to [0, flux_peak].
  2. Simulate SPAD binary detections (n_spad_frames frames).
  3. Bin detections into `sequence_length` bins (for the SPAD observation video).
  4. Run UWB1D FFT reconstruction on the raw binary frames.
  5. Normalize and save as MP4.

No diffusion model is used. Pixels are processed in spatial chunks to keep
memory usage bounded.

Example
-------
python video_inference_uwb1d.py \\
  --input        ./log_flux.pt \\
  --output       ./reconstructed_uwb1d.mp4 \\
  --uwb_root     /path/to/photon-testing-grounds \\
  --target_ppp   0.05 \\
  --n_spad_frames 100000 \\
  --spatial_chunk 1024
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
# Misc helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def maybe_add_to_syspath(path: Optional[str]) -> None:
    if path and os.path.isdir(path):
        if path not in sys.path:
            sys.path.insert(0, path)


def chunked_indices(n: int, chunk_size: int) -> Iterable[slice]:
    for i in range(0, n, chunk_size):
        yield slice(i, min(i + chunk_size, n))


# ---------------------------------------------------------------------------
# SPAD simulation & binning  (vectorised over a batch of pixels)
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
    flux_batch : ndarray [N, seq_len]
        Linear-space flux for each pixel (already normalised to [0, flux_peak]).
    target_ppp : float
        Target average photons per pixel per time-bin.
    n_spad_frames : int
        Number of binary frames to simulate per pixel.
    dark_count : float
        Spurious detection rate per frame.
    T : float
        Total observation time (seconds).

    Returns
    -------
    binary_batch : ndarray [N, n_spad_frames]  uint8
    ppp_scales   : ndarray [N]                 float32  (unused by UWB1D, kept for
                                                          SPAD-video consistency)
    """
    N, L = flux_batch.shape
    flux_batch = flux_batch.astype(np.float64)

    # Interpolate each pixel's flux from seq_len bins -> n_spad_frames bins.
    t_src = np.linspace(0.0, T, L)
    t_dst = np.linspace(0.0, T, n_spad_frames)
    flux_interp = np.empty((N, n_spad_frames), dtype=np.float64)
    for i in range(N):
        flux_interp[i] = np.interp(t_dst, t_src, flux_batch[i])

    # Normalise each pixel to [0, 1].
    mx = flux_interp.max(axis=1, keepdims=True)
    mx = np.where(mx > 0, mx, 1.0)
    flux_norm = flux_interp / mx

    # Scaling factor a so that mean(a * I) = target_ppp.
    I_mean = flux_norm.mean(axis=1, keepdims=True)
    I_mean = np.where(I_mean > 0, I_mean, 1.0)
    a = target_ppp / I_mean                             # [N, 1]

    # ppp_scale = a / flux_max  (unused by UWB1D but kept for reference)
    ppp_scales = (a / mx).ravel().astype(np.float32)   # [N]

    # Scaled flux: N(t) = a * I_norm(t) + d
    flux_scaled = a * flux_norm + dark_count            # [N, n_spad_frames]

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
    Used only for the SPAD-observation video, not for UWB1D.
    """
    N = binary.shape[0]
    num_bins = bin_edges.shape[0] - 1
    n_frames = binary.shape[1]

    bin_sizes = np.diff(bin_edges)
    if np.all(bin_sizes == bin_sizes[0]) and (n_frames % num_bins == 0):
        m = n_frames // num_bins
        return binary.reshape(N, num_bins, m).sum(axis=2).astype(np.float32)

    out = np.empty((N, num_bins), dtype=np.float32)
    starts = bin_edges[:-1].astype(np.intp)
    for i in range(N):
        counts = np.add.reduceat(binary[i].astype(np.int64), starts)
        out[i] = counts[:num_bins].astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Temporal binning  (average-bin a high-FPS flux signal down to fewer frames)
# ---------------------------------------------------------------------------

def temporal_bin_chunk(flux_chunk: np.ndarray, target_bins: int) -> np.ndarray:
    """
    Average-bin flux_chunk from its current temporal resolution down to
    target_bins time bins.

    Parameters
    ----------
    flux_chunk : ndarray [N, T_full]
        High-resolution flux estimates (one row per pixel).
    target_bins : int
        Desired number of output time bins.

    Returns
    -------
    ndarray [N, target_bins]  float32
        Temporally averaged flux.
    """
    N, T_full = flux_chunk.shape
    if T_full == target_bins:
        return flux_chunk.astype(np.float32)

    if T_full % target_bins == 0:
        # Fast path: equal-size bins — sum across each bin.
        m = T_full // target_bins
        return flux_chunk.reshape(N, target_bins, m).sum(axis=2).astype(np.float32)

    # General path: use np.add.reduceat with pre-computed edges — sum, no division.
    edges = np.linspace(0, T_full, target_bins + 1, dtype=np.int64)
    out = np.empty((N, target_bins), dtype=np.float32)
    for i in range(N):
        sums = np.add.reduceat(flux_chunk[i].astype(np.float64), edges[:-1].astype(np.intp))
        out[i] = sums[:target_bins].astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# UWB1D reconstruction  (thin wrapper around ihpp_fft.uwb1d_quanta)
# ---------------------------------------------------------------------------

def reconstruct_uwb1d_chunk(
    binary_chunk: np.ndarray,
    *,
    T_exp: float,
    Nt: int,
    alpha: Optional[float],
    use_cuda: bool,
) -> np.ndarray:
    """
    Run UWB1D on a spatial chunk of binary SPAD frames.

    Parameters
    ----------
    binary_chunk : ndarray [N, n_spad_frames]  uint8
        Binary detections for N pixels.
    T_exp : float
        Total observation time (seconds).
    Nt : int
        Number of output time points (= sequence_length for the video).
    alpha : float or None
        CFAR significance level. None uses the Bonferroni default (1/num_freqs).
    use_cuda : bool
        Pass frames/output through GPU FFT if True.

    Returns
    -------
    flux_chunk : ndarray [N, Nt]  float32
        Reconstructed linear flux for each pixel at full Nt resolution.
        Caller is responsible for temporal binning if a lower output
        resolution is required.
    """
    from uwb3d import ihpp_fft  # imported lazily so --uwb_root takes effect first

    N, n_spad_frames = binary_chunk.shape

    # UWB1D expects shape (T, H, W); we treat pixels as (T, N, 1).
    frames_3d = binary_chunk.T[:, :, np.newaxis]  # [n_spad_frames, N, 1]

    flux_3d = ihpp_fft.uwb1d_quanta(
        frames_3d,
        T_exp=T_exp,
        Nt=Nt,
        alpha=alpha,
        cuda=use_cuda,
        load_frames_on_vram=False,  # we manage memory ourselves
        load_output_on_vram=False,
    )  # -> [Nt, N, 1]

    flux_3d = np.asarray(flux_3d, dtype=np.float32)
    return flux_3d[:, :, 0].T  # [N, Nt]


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

    Strategy (in order):
      1. ffmpeg subprocess — pipes raw 'gray' pixels directly into libx264
         with pix_fmt yuv420p.  This is the only reliable path for true
         grayscale at arbitrary FPS: no fake-RGB conversion, no chroma noise.
      2. OpenCV VideoWriter (isColor=False) — fallback if ffmpeg is absent.
      3. imageio — last resort; stacks to RGB before encoding.
    """
    import subprocess

    T, H, W = frames.shape
    frames = np.ascontiguousarray(frames, dtype=np.uint8)

    # ------------------------------------------------------------------
    # 1. ffmpeg (primary)
    # ------------------------------------------------------------------
    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{W}x{H}",
            "-pix_fmt", "gray",
            "-r", str(fps),
            "-i", "pipe:",
            "-an",
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for t in range(T):
            proc.stdin.write(frames[t].tobytes())
        proc.stdin.close()
        proc.wait()
        if proc.returncode == 0:
            print(f"Saved video via ffmpeg  : {path}")
            return
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")
    except Exception as e:
        print(f"ffmpeg failed ({e}), trying OpenCV ...")

    # ------------------------------------------------------------------
    # 2. OpenCV (fallback)
    # ------------------------------------------------------------------
    try:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, fps, (W, H), isColor=False)
        if not writer.isOpened():
            raise RuntimeError("cv2.VideoWriter failed to open")
        for t in range(T):
            writer.write(frames[t])
        writer.release()
        print(f"Saved video via OpenCV  : {path}")
        return
    except Exception as e:
        print(f"OpenCV failed ({e}), trying imageio ...")

    # ------------------------------------------------------------------
    # 3. imageio (last resort)
    # ------------------------------------------------------------------
    import imageio.v3 as iio
    frames_rgb = np.stack([frames, frames, frames], axis=-1)  # [T, H, W, 3]
    iio.imwrite(path, frames_rgb, fps=fps, codec="libx264")
    print(f"Saved video via imageio : {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Reconstruct a video from per-pixel 1D SPAD observations via UWB1D."
    )

    # I/O
    ap.add_argument("--input", type=str, required=True,
                    help="Path to log_flux .pt file of shape [T, H, W]")
    ap.add_argument("--output", type=str, default="reconstructed_uwb1d.mp4",
                    help="Output MP4 path (default: reconstructed_uwb1d.mp4)")
    ap.add_argument("--uwb_root", type=str, default=None,
                    help="Path to add to sys.path so that `uwb3d` can be imported "
                         "(e.g. /path/to/photon-testing-grounds)")

    # Temporal
    ap.add_argument("--sequence_length", type=int, default=1024,
                    help="Output temporal resolution (Nt for UWB1D, and video frame count)")

    # SPAD simulation
    ap.add_argument("--t_total", type=float, default=1.0,
                    help="Total SPAD observation window in seconds")
    ap.add_argument("--dark_count", type=float, default=7.74e-4,
                    help="Spurious detection rate per SPAD frame")
    ap.add_argument("--n_spad_frames", type=int, default=100_000,
                    help="Number of binary SPAD frames to simulate per pixel")
    ap.add_argument("--target_ppp", type=float, default=0.05,
                    help="Target average photons per pixel per time-bin")

    # Normalisation
    ap.add_argument("--flux_peak", type=float, default=10000.0,
                    help="Peak value after per-pixel normalisation before SPAD sim")
    ap.add_argument("--no_normalize_flux", action="store_true",
                    help="Disable per-pixel flux normalisation")

    # UWB1D
    ap.add_argument("--alpha", type=float, default=None,
                    help="CFAR significance level for UWB1D (default: Bonferroni 1/num_freqs)")

    # Runtime
    ap.add_argument("--spatial_chunk", type=int, default=1024,
                    help="Number of pixels per spatial processing chunk "
                         "(trades memory vs. overhead; default: 1024)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps", type=int, default=None,
                    help="Output video frame rate. Defaults to sequence_length so that "
                         "1 video-second = 1 physical second of SPAD observation.")

    args = ap.parse_args()

    # Resolve fps: default to sequence_length so playback speed == physical speed.
    fps = args.fps if args.fps is not None else args.sequence_length

    maybe_add_to_syspath(args.uwb_root)
    set_seed(args.seed)
    normalize_flux = not args.no_normalize_flux

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
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

    if T_vid < seq_len:
        raise ValueError(
            f"Temporal dimension T={T_vid} is shorter than "
            f"sequence_length={seq_len}. Cannot proceed."
        )
    if T_vid > seq_len:
        print(f"Note   : T={T_vid} > sequence_length={seq_len}; "
              f"using first {seq_len} frames only.")
        raw = raw[:seq_len]

    # Reshape to [num_pixels, seq_len].
    log_flux_2d = raw.numpy().astype(np.float64).reshape(seq_len, -1).T  # [N, seq_len]

    # Convert log-flux -> linear flux.
    flux_all = np.exp(log_flux_2d)  # [N, seq_len]

    # Per-pixel normalisation to [0, flux_peak].
    if normalize_flux:
        mx = flux_all.max(axis=1, keepdims=True)
        mx = np.where(mx > 0, mx, 1.0)
        flux_all = (flux_all / mx) * float(args.flux_peak)

    print(f"Flux range (linear) : [{flux_all.min():.1f}, {flux_all.max():.1f}]")

    # ------------------------------------------------------------------
    # 2.  Simulate SPAD & run UWB1D  (one spatial chunk at a time)
    # ------------------------------------------------------------------
    bin_edges    = np.linspace(0, args.n_spad_frames, seq_len + 1, dtype=np.int64)
    bin_counts_all = np.empty((num_pixels, seq_len), dtype=np.float32)
    flux_hat_all   = np.empty((num_pixels, seq_len), dtype=np.float32)

    print(f"\nSimulating SPAD (PPP={args.target_ppp}, frames={args.n_spad_frames}) "
          f"and running UWB1D (Nt={args.n_spad_frames} → binned to {seq_len}) ...")
    print(f"Output video : {seq_len} frames @ {fps} FPS  "
          f"({seq_len / fps:.3f} s playback = {args.t_total:.3f} s physical)")
    print(f"Spatial chunk size : {args.spatial_chunk}  "
          f"({-(-num_pixels // args.spatial_chunk)} chunks total)")

    chunks = list(chunked_indices(num_pixels, args.spatial_chunk))
    for sl in tqdm(chunks, desc="SPAD + UWB1D"):
        flux_chunk = flux_all[sl]  # [chunk, seq_len]

        # --- SPAD simulation ---
        binary_chunk, _ = generate_spad_binary_batch(
            flux_chunk,
            target_ppp=args.target_ppp,
            n_spad_frames=args.n_spad_frames,
            dark_count=args.dark_count,
            T=args.t_total,
        )  # binary_chunk: [chunk, n_spad_frames]

        # Bin counts for the SPAD-observation video.
        bin_counts_all[sl] = bin_binary_batch(binary_chunk, bin_edges)

        # --- UWB1D reconstruction at full temporal resolution ---
        flux_full = reconstruct_uwb1d_chunk(
            binary_chunk,
            T_exp=args.t_total,
            Nt=args.n_spad_frames,   # reconstruct at full 100K-FPS resolution
            alpha=args.alpha,
            use_cuda=use_cuda,
        )  # [chunk, n_spad_frames]

        # --- Integrate (average-bin) down to seq_len output frames ---
        flux_hat_all[sl] = temporal_bin_chunk(flux_full, seq_len)

    # Clip negative values (UWB1D can produce small negatives near zero).
    flux_hat_all = np.maximum(flux_hat_all, 0.0)

    # ------------------------------------------------------------------
    # 3.  Save SPAD-observation video  (binned counts)
    # ------------------------------------------------------------------
    spad_video  = bin_counts_all.T.reshape(seq_len, H, W)   # [seq_len, H, W]
    spad_frames = flux_to_video_frames(spad_video)
    spad_path   = os.path.splitext(args.output)[0] + "_spad.mp4"
    os.makedirs(os.path.dirname(os.path.abspath(spad_path)), exist_ok=True)
    save_video_mp4(spad_frames, spad_path, fps=fps)

    # ------------------------------------------------------------------
    # 4.  Save reconstructed video
    # ------------------------------------------------------------------
    flux_video = flux_hat_all.T.reshape(seq_len, H, W)      # [seq_len, H, W]
    print(f"\nReconstructed flux range : [{flux_video.min():.2f}, {flux_video.max():.2f}]")

    recon_frames = flux_to_video_frames(flux_video)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_video_mp4(recon_frames, args.output, fps=fps)
    print("Done.")


if __name__ == "__main__":
    main()
