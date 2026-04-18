"""Tests for `spatial_prior.py` — the 1D-log-flux <-> 2D-RGB bridge.

The real 2D checkpoint (`256x256_diffusion_uncond.pt`) is not required for any
test here: `test_denoise_*` tests use either a mock UNet (to exercise the
Tweedie wiring without a trained prior) or are skipped with a clear message
if the real checkpoint is unavailable. The wrapper's correctness (shapes,
roundtrip, range, Tweedie identity at t=0) is checked end-to-end.
"""

import math
import os
import sys
from pathlib import Path

import pytest
import torch

# Make `spatial_prior` importable when running `pytest` from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from spatial_prior import (  # noqa: E402
    NormalizationParams,
    to_2d_space,
    from_2d_space,
    denoise_frame_2d,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def params():
    """Log-flux range representative of the phantom_simulation dataset.

    Raw flux empirically spans roughly `[0, 0.23]` per-frame
    (see /Users/tan583/Documents/3D Probing/.../wineglassfall*.npy), so
    log-flux = log(flux + 1e-6) ≈ [-13.8, -1.47]. We pick a slightly wider
    symmetric-ish range for robustness.
    """
    return NormalizationParams(source_min=-14.0, source_max=-1.0)


@pytest.fixture
def rng():
    g = torch.Generator()
    g.manual_seed(0)
    return g


# ---------------------------------------------------------------------------
# Helpers — minimal stand-ins for a real 2D diffusion
# ---------------------------------------------------------------------------


class _ZeroEpsUNet(torch.nn.Module):
    """Mock UNet that always predicts epsilon = 0 with `learn_sigma=True`.

    Output shape `[B, 6, H, W]` to match `256x256_diffusion_uncond`. Under
    Tweedie `x_0 = sqrt(1/ab)*x_t - sqrt(1/ab - 1)*eps`, eps=0 gives
    `x_0 = sqrt(1/ab) * x_t`. At t=0, `ab_0 = alpha_0 = 1 - beta_0 ≈ 0.9999`
    so `x_0 ≈ x_t` (not exactly equal, but very close). This is sufficient
    to exercise wrapper wiring without a trained model.
    """

    def forward(self, x, t):
        B, C, H, W = x.shape
        return torch.zeros(B, 2 * C, H, W, device=x.device, dtype=x.dtype)


def _make_toy_diffusion(T: int = 1000):
    """Construct a `GaussianDiffusion` with the same schedule as the target
    2D checkpoint (linear, T=1000). Uses improved-diffusion because it's
    already installed; guided-diffusion has the identical schedule math for
    the subset we need (`_predict_xstart_from_eps`).
    """
    from improved_diffusion import gaussian_diffusion as gd

    betas = gd.get_named_beta_schedule("linear", T)
    diffusion = gd.GaussianDiffusion(
        betas=betas,
        model_mean_type=gd.ModelMeanType.EPSILON,
        model_var_type=gd.ModelVarType.LEARNED_RANGE,
        loss_type=gd.LossType.MSE,
        rescale_timesteps=False,
    )
    return diffusion


# ---------------------------------------------------------------------------
# Shape / roundtrip / range
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("B", [1, 4])
def test_shapes(params, B, rng):
    H, W = 32, 48
    x = torch.randn(B, H, W, generator=rng) * 3.0 - 7.0  # within [-14, -1] mostly

    x_2d = to_2d_space(x, params)
    assert x_2d.shape == (B, 3, H, W), f"to_2d_space shape {x_2d.shape}"

    x_back = from_2d_space(x_2d, params)
    assert x_back.shape == (B, H, W), f"from_2d_space shape {x_back.shape}"


def test_roundtrip(params, rng):
    B, H, W = 2, 16, 24
    x = torch.empty(B, H, W).uniform_(
        params.source_min + 1e-3,
        params.source_max - 1e-3,
        generator=rng,
    )
    x_back = from_2d_space(to_2d_space(x, params), params)
    max_err = (x - x_back).abs().max().item()
    assert max_err < 1e-5, f"round-trip max abs error {max_err} exceeds 1e-5"


def test_roundtrip_luminance_tolerance(params, rng):
    """When `use_luminance=True`, Rec.601 weights sum to 0.9999 in float,
    so round-trip on identical channels should be accurate to ~1e-4 but
    not 1e-5. Document this explicitly.
    """
    p = NormalizationParams(
        source_min=params.source_min,
        source_max=params.source_max,
        use_luminance=True,
    )
    B, H, W = 2, 16, 24
    x = torch.empty(B, H, W).uniform_(p.source_min + 1e-3, p.source_max - 1e-3, generator=rng)
    x_back = from_2d_space(to_2d_space(x, p), p)
    max_err = (x - x_back).abs().max().item()
    assert max_err < 2e-3, f"luminance round-trip max abs error {max_err}"


def test_range(params, rng):
    """Random log-flux drawn inside [source_min, source_max] maps to [-1, 1]."""
    B, H, W = 3, 32, 32
    x = torch.empty(B, H, W).uniform_(params.source_min, params.source_max, generator=rng)
    x_2d = to_2d_space(x, params)
    eps = 1e-5
    assert x_2d.min().item() >= -1.0 - eps, f"below -1: {x_2d.min().item()}"
    assert x_2d.max().item() <= 1.0 + eps, f"above +1: {x_2d.max().item()}"


def test_range_extrema(params):
    """Exact endpoints of source range map to exactly -1 / +1 (within fp)."""
    x = torch.tensor([[[params.source_min, params.source_max]]])  # [1, 1, 2]
    x_2d = to_2d_space(x, params)
    assert abs(x_2d[0, 0, 0, 0].item() - (-1.0)) < 1e-6
    assert abs(x_2d[0, 0, 0, 1].item() - (+1.0)) < 1e-6


def test_to_2d_differentiable(params):
    x = (torch.randn(2, 8, 8) * 2 - 7).requires_grad_(True)
    x_2d = to_2d_space(x, params)
    # Any scalar function of x_2d should produce a grad on x.
    x_2d.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0


def test_roundtrip_differentiable(params):
    """Full round-trip should backprop and produce a finite gradient."""
    x = (torch.randn(2, 8, 8) * 2 - 7).requires_grad_(True)
    y = from_2d_space(to_2d_space(x, params), params)
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# Tweedie wiring — using a mock 2D model so these don't require the real
# 256x256_diffusion_uncond checkpoint.
# ---------------------------------------------------------------------------


def test_denoise_noop_at_t0(params):
    """At t=0, Tweedie with any eps should return ~x_t. The guided-/improved-
    diffusion linear schedule has beta_0 = 1e-4, so sqrt(1/ab_0) ≈ 1.00005
    and sqrt(1/ab_0 - 1) ≈ 0.01 — so x_0 ≈ x_t up to a ~1% perturbation
    from eps. With the zero-eps mock, the result is exact up to the
    1.00005 scaling.
    """
    torch.manual_seed(0)
    B, H, W = 1, 16, 16
    x_t = torch.empty(B, H, W).uniform_(params.source_min + 1, params.source_max - 1)
    diffusion = _make_toy_diffusion(T=1000)
    model = _ZeroEpsUNet()

    x_0 = denoise_frame_2d(x_t, t=0, model_2d=model, diffusion_2d=diffusion,
                            normalization_params=params)
    assert x_0.shape == x_t.shape
    # Output should be finite and within the configured log-flux range,
    # modulo a tiny Tweedie-at-t=0 scaling (~1e-4 relative).
    assert torch.isfinite(x_0).all()
    rel_err = ((x_0 - x_t).abs() / (x_t.abs() + 1e-6)).max().item()
    assert rel_err < 5e-3, (
        f"Tweedie at t=0 should be near-identity with eps=0; got rel err {rel_err}"
    )


def test_denoise_shape_batch(params):
    B, H, W = 4, 32, 32
    diffusion = _make_toy_diffusion(T=1000)
    model = _ZeroEpsUNet()
    x_t = torch.empty(B, H, W).uniform_(params.source_min, params.source_max)

    # Scalar t
    x0_a = denoise_frame_2d(x_t, t=500, model_2d=model, diffusion_2d=diffusion,
                             normalization_params=params)
    assert x0_a.shape == (B, H, W)

    # Per-batch t
    t_vec = torch.tensor([100, 300, 500, 700])
    x0_b = denoise_frame_2d(x_t, t=t_vec, model_2d=model, diffusion_2d=diffusion,
                             normalization_params=params)
    assert x0_b.shape == (B, H, W)


# ---------------------------------------------------------------------------
# Real denoising check — requires the actual 256x256_diffusion_uncond checkpoint.
# Gated on environment / filesystem.
# ---------------------------------------------------------------------------


def _find_2d_checkpoint():
    """Return a Path to the real 2D checkpoint, or None if not present."""
    candidates = [
        REPO_ROOT / "checkpoints" / "256x256_diffusion_uncond.pt",
        REPO_ROOT.parent / "guided-diffusion" / "models" / "256x256_diffusion_uncond.pt",
        REPO_ROOT.parent / "DiffPIR-main" / "model_zoo" / "256x256_diffusion_uncond.pt",
        Path(os.environ.get("IMAGENET256_UNCOND_CKPT", "")),
    ]
    for p in candidates:
        if p and p.is_file():
            return p
    return None


def _load_imagenet256_uncond(ckpt_path: Path):
    """Build guided-diffusion's unconditional ImageNet-256 UNet and load weights.

    Flags are taken from guided-diffusion/README.md lines 67-71.
    """
    gd_repo = REPO_ROOT.parent / "guided-diffusion"
    if str(gd_repo) not in sys.path:
        sys.path.insert(0, str(gd_repo))
    from guided_diffusion.script_util import (  # type: ignore
        create_model_and_diffusion,
        model_and_diffusion_defaults,
    )

    cfg = model_and_diffusion_defaults()
    cfg.update(dict(
        image_size=256,
        num_channels=256,
        num_head_channels=64,
        num_res_blocks=2,
        attention_resolutions="32,16,8",
        resblock_updown=True,
        use_scale_shift_norm=True,
        learn_sigma=True,
        noise_schedule="linear",
        diffusion_steps=1000,
        class_cond=False,
        use_fp16=False,  # fp32 for test stability on any device
    ))
    model, diffusion = create_model_and_diffusion(**cfg)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, diffusion


@pytest.mark.skipif(_find_2d_checkpoint() is None,
                    reason="256x256_diffusion_uncond.pt not on disk; "
                           "see docs/phase2_verify.py for download instructions.")
def test_denoise_reduces_noise():
    """With the real trained 2D prior, denoising a noisy frame should
    reduce MSE to the clean frame.
    """
    torch.manual_seed(0)
    params = NormalizationParams(source_min=-14.0, source_max=-1.0)
    ckpt = _find_2d_checkpoint()
    model, diffusion = _load_imagenet256_uncond(ckpt)

    # Synthesize a clean log-flux "frame" — smooth gradient + low-freq texture.
    H = W = 256
    yy, xx = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij")
    clean = (-14.0 + 13.0 * (0.5 + 0.4 * torch.sin(4 * math.pi * xx) * torch.cos(3 * math.pi * yy)))
    clean = clean.unsqueeze(0)  # [1, H, W]

    t = 300
    # q(x_t|x_0) in 2D [-1,1] space — we need to apply noise in the same space
    # the schedule is defined, i.e. post to_2d_space. Use the diffusion's own
    # q_sample helper via a manual construction: sample noise in log-flux
    # space with the right std so that after to_2d_space the noise has the
    # per-schedule variance.
    #
    # Simpler: forward clean to 2D space, q_sample in 2D, then pass the result
    # back through from_2d_space to produce a log-flux x_t that denoise_frame_2d
    # can consume.
    clean_2d = to_2d_space(clean, params)  # [1,3,H,W]
    noise = torch.randn_like(clean_2d)
    t_tensor = torch.tensor([t], dtype=torch.long)
    xt_2d = diffusion.q_sample(clean_2d, t_tensor, noise=noise)
    xt_logflux = from_2d_space(xt_2d, params)

    with torch.no_grad():
        x0_hat = denoise_frame_2d(
            xt_logflux, t=t, model_2d=model, diffusion_2d=diffusion,
            normalization_params=params,
        )

    mse_noisy = ((xt_logflux - clean) ** 2).mean().item()
    mse_denoised = ((x0_hat - clean) ** 2).mean().item()
    assert mse_denoised < mse_noisy, (
        f"Denoising did not reduce MSE: noisy={mse_noisy:.4f}, "
        f"denoised={mse_denoised:.4f}"
    )
