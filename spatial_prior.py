"""
Bridge between 1D log-flux space and a pretrained 2D (RGB, [-1,1]) Gaussian-
diffusion score network.

See docs/phase0_recon.md for the compatibility analysis that justifies the
design choices here. In particular:
  - The 2D checkpoint target is guided-diffusion `256x256_diffusion_uncond.pt`:
    linear schedule, T=1000, epsilon-prediction, learn_sigma=True (6-channel
    output; the second half is variance-interpolation logits and is discarded).
  - Channel bridge is grayscale-replicate in, mean over 3 channels out. The
    default "mean" projection is chosen over Rec.601 luminance weights so that
    the round-trip identity holds exactly in float precision
    (Rec.601 sums to 0.9999, breaking 1e-5 round-trip).
    Luminance projection is available via NormalizationParams.use_luminance.
  - Range bridge is an invertible affine defined by (source_min, source_max)
    mapped to (-1, +1).
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


_REC601_WEIGHTS: Tuple[float, float, float] = (0.2989, 0.5870, 0.1140)


@dataclass
class NormalizationParams:
    """Affine bridge from 1D log-flux space to 2D `[-1, 1]` RGB space.

    The forward affine is
        x_gray = 2 * (x_logflux - source_min) / (source_max - source_min) - 1
    so that `source_min -> -1` and `source_max -> +1`. Values of `x_logflux`
    outside `[source_min, source_max]` are *not* clipped here; callers can
    clip if they need hard range guarantees (see `to_2d_space(..., clip=True)`).

    `source_min` / `source_max` should be fixed from dataset statistics of
    whatever log-flux convention the 1D prior was trained in (Phase 0 found
    the current 1D checkpoints were trained with `--normalize False`, i.e. on
    raw log-flux, so set these from empirical log-flux percentiles of that
    same training set — e.g. the 0.1 and 99.9 percentiles).
    """

    source_min: float
    source_max: float

    # If True, use Rec.601 luminance weights to project RGB -> gray in
    # `from_2d_space`. If False (default), use a plain mean — this is what
    # makes the round-trip identity exact on replicated inputs, and is the
    # cleaner choice when the 3 input channels are identical (which they are
    # when produced by `to_2d_space`). Flip to True only if you are projecting
    # non-identical RGB model outputs back to grayscale and want luminance
    # consistency with the dataset generator's sRGB->gray conversion.
    use_luminance: bool = False
    lum_weights: Tuple[float, float, float] = field(default_factory=lambda: _REC601_WEIGHTS)

    def __post_init__(self):
        if not (self.source_max > self.source_min):
            raise ValueError(
                f"source_max ({self.source_max}) must be > source_min ({self.source_min})"
            )

    @property
    def scale(self) -> float:
        """Multiplicative factor: `x_gray = scale * (x_logflux - source_min) - 1`."""
        return 2.0 / (self.source_max - self.source_min)


def to_2d_space(
    x_logflux: torch.Tensor,
    p: NormalizationParams,
    clip: bool = False,
) -> torch.Tensor:
    """1D log-flux `[B, H, W]` -> 2D model input `[B, 3, H, W]` in `[-1, 1]`.

    Differentiable (no `.detach()` / no in-place ops on the input).
    """
    if x_logflux.dim() != 3:
        raise ValueError(f"Expected [B, H, W], got shape {tuple(x_logflux.shape)}")

    scale = p.scale
    x_gray = scale * (x_logflux - p.source_min) - 1.0  # [B, H, W]
    if clip:
        x_gray = torch.clamp(x_gray, -1.0, 1.0)

    # Replicate grayscale into 3 RGB channels. `.expand` avoids an allocation,
    # but a later `.sum` or arithmetic on the returned tensor is still correct.
    # Use `.unsqueeze(1).expand(...)` rather than `.repeat(...)` for efficiency.
    x_2d = x_gray.unsqueeze(1).expand(-1, 3, -1, -1)
    return x_2d


def from_2d_space(
    x_2d: torch.Tensor,
    p: NormalizationParams,
) -> torch.Tensor:
    """2D model space `[B, 3, H, W]` -> 1D log-flux `[B, H, W]`.

    Exact inverse of `to_2d_space` when the 3 channels are identical and
    `p.use_luminance=False`. For non-identical RGB (e.g. 2D-model epsilon
    output projected back to grayscale), channels are mean-reduced by default;
    set `p.use_luminance=True` to use Rec.601 weights instead.
    """
    if x_2d.dim() != 4 or x_2d.shape[1] != 3:
        raise ValueError(f"Expected [B, 3, H, W], got shape {tuple(x_2d.shape)}")

    if p.use_luminance:
        w = torch.tensor(
            p.lum_weights, device=x_2d.device, dtype=x_2d.dtype
        ).view(1, 3, 1, 1)
        x_gray = (x_2d * w).sum(dim=1)
    else:
        x_gray = x_2d.mean(dim=1)  # [B, H, W]

    x_logflux = (x_gray + 1.0) / p.scale + p.source_min
    return x_logflux


def t_prime_from_t(
    t: int,
    scale: float,
    alphas_cumprod_2d: np.ndarray,
    alphas_cumprod_1d: Optional[np.ndarray] = None,
) -> int:
    """Noise-aligned diffusion-step lookup for the affine 1D->2D bridge.

    The bridge `A(x) = scale*x + b` maps a 1D-space `x_t` (with noise std
    `sqrt(1 - alpha_bar_t^(1D))`) to a 2D-space sample whose noise std is
    `scale * sqrt(1 - alpha_bar_t^(1D))`. The 2D model expects, at step `t'`,
    noise std `sqrt(1 - alpha_bar_{t'}^(2D))`. Equating:

        1 - alpha_bar_{t'}^(2D) = scale^2 * (1 - alpha_bar_t^(1D))        (*)

    The 1D and 2D schedules can differ (e.g. 1D linear T=1000 vs 2D cosine
    T=4000): the `target` on the RHS of (*) is read off the 1D schedule at
    step `t`, and the `searchsorted` on the LHS walks the 2D schedule to
    find `t' ∈ [0, T_2D - 1]`. When the schedules are identical, pass
    `alphas_cumprod_1d=None` (or the same array twice) — the behaviour is
    identical to before.

    `1 - alpha_bar` is monotonically increasing for any standard schedule,
    so the inverse is always a single `searchsorted`. Result is clamped to
    `[0, T_2D - 1]`.

    The signal coefficient on `x_0` after the bridge is `scale * sqrt(alpha_bar_t^(1D))`,
    which the 2D model interprets as `sqrt(alpha_bar_{t'}^(2D))`. These match
    only when `scale == 1`; with `scale != 1` the residual signal-amplitude
    mismatch is the standard, accepted bias of noise-only t-shift bridges.
    """
    alphas_cumprod_2d = np.asarray(alphas_cumprod_2d)
    if alphas_cumprod_1d is None:
        alphas_cumprod_1d = alphas_cumprod_2d
    else:
        alphas_cumprod_1d = np.asarray(alphas_cumprod_1d)
    target = (scale ** 2) * (1.0 - float(alphas_cumprod_1d[int(t)]))
    one_minus_ab_2d = 1.0 - alphas_cumprod_2d  # monotone increasing
    t_prime = int(np.searchsorted(one_minus_ab_2d, target))
    return max(0, min(t_prime, len(alphas_cumprod_2d) - 1))


def denoise_frame_2d(
    x_t_frame: torch.Tensor,
    t: Union[int, torch.Tensor],
    model_2d,
    diffusion_2d,
    normalization_params: NormalizationParams,
    *,
    align_noise: bool = False,
    alphas_cumprod_1d: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """Tweedie `x_0` estimate in log-flux space from a noisy log-flux frame.

    `x_t_frame`: `[B, H, W]` noisy log-flux at diffusion step `t`, where
        `t in [0, diffusion_2d.num_timesteps - 1]`. The `[B, H, W]` is
        interpreted as already living in the 1D model's log-flux convention
        (same space `to_2d_space` consumes).
    `t`: scalar `int` (same step for all batch items) or `[B]` long tensor.
    `model_2d`: callable `(x_2d [B,3,H,W], t [B]) -> model_out`. For guided-
        diffusion's `learn_sigma=True` checkpoints (e.g. 256x256_diffusion_uncond)
        `model_out` is `[B, 6, H, W]`, the first 3 channels being epsilon and
        the last 3 being variance-interpolation logits (discarded here — this
        wrapper only consumes epsilon for the Tweedie estimate).
    `diffusion_2d`: a `GaussianDiffusion` (guided-diffusion or improved-
        diffusion) whose schedule matches the checkpoint. Used for the closed-
        form Tweedie coefficients via `_predict_xstart_from_eps`.

    `align_noise`: when `True`, computes a shifted step `t'` such that the
        noise variance the 2D UNet sees matches what it was trained for at `t'`,
        compensating for the bridge's `scale != 1` (see `t_prime_from_t`).
        Requires a scalar-int `t` and `alphas_cumprod_1d` (the *1D* diffusion's
        `alphas_cumprod` — used to compute the target variance at step t).
        The *2D* schedule is read off `diffusion_2d.alphas_cumprod` inside
        this function and used for the searchsorted, so 1D and 2D schedules
        may differ (e.g. 1D linear T=1000 with 2D cosine T=4000).

    Returns `[B, H, W]` estimated `x_0` in log-flux space.
    """
    if x_t_frame.dim() != 3:
        raise ValueError(f"Expected [B, H, W], got shape {tuple(x_t_frame.shape)}")

    B = x_t_frame.shape[0]
    x_2d = to_2d_space(x_t_frame, normalization_params)  # [B, 3, H, W]

    if align_noise:
        if not isinstance(t, int):
            raise ValueError("align_noise=True requires a scalar int t")
        if alphas_cumprod_1d is None:
            raise ValueError("align_noise=True requires alphas_cumprod_1d")
        # Search on the 2D schedule (which may differ from the 1D schedule —
        # e.g. 1D linear T=1000 vs 2D cosine T=4000).
        t_eff = t_prime_from_t(
            t,
            normalization_params.scale,
            alphas_cumprod_2d=diffusion_2d.alphas_cumprod,
            alphas_cumprod_1d=alphas_cumprod_1d,
        )
    else:
        t_eff = t

    if isinstance(t_eff, int):
        t_tensor = torch.full(
            (B,), t_eff, device=x_t_frame.device, dtype=torch.long
        )
    else:
        t_tensor = t_eff.to(device=x_t_frame.device, dtype=torch.long)
        if t_tensor.ndim == 0:
            t_tensor = t_tensor.expand(B)

    # `.expand` above returns a non-contiguous view; UNets with grouped norms
    # can be picky. Materialize the 3-channel tensor for the forward pass.
    x_2d_forward = x_2d.contiguous()
    model_out = model_2d(x_2d_forward, t_tensor)

    # Epsilon parameterization — drop the variance head if present.
    C_in = x_2d_forward.shape[1]
    if model_out.shape[1] == 2 * C_in:
        eps = model_out[:, :C_in]
    elif model_out.shape[1] == C_in:
        eps = model_out
    else:
        raise ValueError(
            f"Unexpected 2D model output channel count {model_out.shape[1]}; "
            f"expected {C_in} (fixed sigma) or {2 * C_in} (learned sigma)."
        )

    # Tweedie: x_0_hat = sqrt(1/alpha_bar_t) * x_t - sqrt(1/alpha_bar_t - 1) * eps
    x_0_2d = diffusion_2d._predict_xstart_from_eps(x_t=x_2d_forward, t=t_tensor, eps=eps)

    x_0_logflux = from_2d_space(x_0_2d, normalization_params)
    return x_0_logflux
