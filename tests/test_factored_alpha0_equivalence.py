"""Critical equivalence test for `factored_diffpir.factored_sample_flux`.

At `alpha_s = 0.0`, the factored loop must reduce EXACTLY to 1D-only DiffPIR:
no 2D forward pass, no 2D-derived contribution to `x0_combined`. This test
runs the sampler twice with the same random seed — once with alpha_s=0 (and
no 2D model) and once with alpha_s=0 but with a mock 2D model that, if wired
in, would return non-zero contributions. Both outputs must be bit-identical
to within 1e-4.

No trained checkpoint is required: we use small random-weight UNets for the
1D branch. The 2D path is only exercised via a mock that never runs (since
`alpha_s=0` short-circuits the 2D call); if the short-circuit were ever
broken, the mock's outputs would perturb the result and this test would fail.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from factored_diffpir import (  # noqa: E402
    FactoredConfig,
    factored_sample_flux,
)
from spatial_prior import NormalizationParams  # noqa: E402


# ---------------------------------------------------------------------------
# Tiny deterministic 1D UNet — NOT the full temporal UNet; we only need a
# callable that takes [N, 1, T] + [N] t and returns [N, 1, T] epsilon.
# ---------------------------------------------------------------------------


class _Tiny1DEpsNet(torch.nn.Module):
    """1-channel-in, 1-channel-out 1D conv with a tiny time embedding.

    Deterministic under `torch.manual_seed`; small enough to run on CPU.
    """

    def __init__(self, seq_length: int = 128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(1, 8, 3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv1d(8, 8, 3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv1d(8, 1, 3, padding=1),
        )
        self.t_emb = torch.nn.Embedding(1000, 1)

    def forward(self, x, t):
        b = self.t_emb(t).view(-1, 1, 1)
        return self.net(x) + 0.01 * b


class _FailOnCall2D(torch.nn.Module):
    """Mock 2D model that screams if forward() is ever invoked."""

    def forward(self, x, t):  # pragma: no cover - only reached on regression
        raise AssertionError(
            "2D model forward was called when alpha_s=0 — the factored loop "
            "must short-circuit the 2D branch at alpha_s=0."
        )


def _make_1d_diffusion(T: int = 1000):
    from improved_diffusion import gaussian_diffusion as gd
    betas = gd.get_named_beta_schedule("linear", T)
    return gd.GaussianDiffusion(
        betas=betas,
        model_mean_type=gd.ModelMeanType.EPSILON,
        model_var_type=gd.ModelVarType.FIXED_LARGE,
        loss_type=gd.LossType.MSE,
        rescale_timesteps=False,
    )


@pytest.fixture
def tiny_cube():
    """Tiny synthetic binary-counts cube: [B=1, H=8, W=8, T=64]."""
    rng = np.random.default_rng(7)
    counts = rng.integers(0, 3, size=(1, 8, 8, 64)).astype(np.float32)
    return torch.from_numpy(counts)


@pytest.fixture
def tiny_1d():
    torch.manual_seed(0)
    model = _Tiny1DEpsNet(seq_length=64).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    diffusion = _make_1d_diffusion(T=1000)
    return model, diffusion


def test_alpha0_equivalence_no_2d_vs_mock_2d(tiny_cube, tiny_1d):
    """Running with alpha_s=0 and model_2d=None vs. model_2d=_FailOnCall2D()
    must give *identical* output. If 2D is ever accidentally invoked when
    alpha_s=0, _FailOnCall2D.forward raises, so the second call failing at
    all is itself a regression signal.
    """
    model_1d, diffusion_1d = tiny_1d

    cfg = FactoredConfig(
        sequence_length=64,
        num_sampling_steps=8,
        eta=0.85,
        alpha_s=0.0,
        pp_iters=2,
        pp_lr_scale=0.25,
        chunk_1d=4096,
        chunk_2d=4,
    )

    out_no2d = factored_sample_flux(
        counts_binned=tiny_cube,
        model_1d=model_1d, diffusion_1d=diffusion_1d,
        model_2d=None, diffusion_2d=None, norm_params=None,
        cfg=cfg, bin_size=1, dt_spad=1e-5,
        seed=123, verbose=False,
    )

    out_with_fail2d = factored_sample_flux(
        counts_binned=tiny_cube,
        model_1d=model_1d, diffusion_1d=diffusion_1d,
        model_2d=_FailOnCall2D(), diffusion_2d=None, norm_params=None,
        cfg=cfg, bin_size=1, dt_spad=1e-5,
        seed=123, verbose=False,
    )

    max_diff = (out_no2d - out_with_fail2d).abs().max().item()
    assert max_diff < 1e-4, (
        f"alpha_s=0 path differs when a 2D mock is supplied: max|Δ|={max_diff:.3e}"
    )


def test_alpha0_equivalence_vs_handwritten_1d(tiny_cube, tiny_1d):
    """Belt-and-braces: compare the alpha_s=0 factored output against an
    inline 1D-only DiffPIR loop that does not know the factored module
    exists. Both use the same seed, cube, model, and data step.
    """
    from factored_diffpir import anscombe_data_step

    model_1d, diffusion_1d = tiny_1d
    B, H, W, T = tiny_cube.shape

    cfg = FactoredConfig(
        sequence_length=T, num_sampling_steps=8, eta=0.85,
        alpha_s=0.0, pp_iters=2, pp_lr_scale=0.25,
        chunk_1d=4096, chunk_2d=4,
    )

    out_factored = factored_sample_flux(
        counts_binned=tiny_cube,
        model_1d=model_1d, diffusion_1d=diffusion_1d,
        model_2d=None, diffusion_2d=None, norm_params=None,
        cfg=cfg, bin_size=1, dt_spad=1e-5,
        seed=42, verbose=False,
    )

    # --- Reference: inline 1D-only loop (no factored imports beyond the
    # shared data step, which is also used by the factored path) ----------
    # Must mirror factored_sample_flux's RNG protocol exactly: seed the CPU
    # Generator for init noise with `seed`, then seed the global torch RNG
    # with `seed + 1` so DDIM noise streams match.
    device = tiny_cube.device
    alphas_cumprod = torch.from_numpy(diffusion_1d.alphas_cumprod).float()
    gen = torch.Generator(device="cpu").manual_seed(42)
    x_t = torch.randn((B, H, W, T), generator=gen)
    torch.manual_seed(42 + 1)
    counts = tiny_cube

    timesteps = np.linspace(diffusion_1d.num_timesteps - 1, 0,
                            cfg.num_sampling_steps, dtype=int)
    for step_idx, t in enumerate(timesteps):
        t_int = int(t)
        ab = alphas_cumprod[t_int]
        sqrt_ab = torch.sqrt(ab)
        sqrt_1mab = torch.sqrt(1.0 - ab)

        x_1d = x_t.reshape(B * H * W, 1, T)
        with torch.no_grad():
            tt = torch.full((x_1d.shape[0],), t_int, dtype=torch.long, device=device)
            eps = model_1d(x_1d, tt)
            x0 = (x_1d - sqrt_1mab * eps) / (sqrt_ab + 1e-8)

        sigma_t_bar = float(sqrt_1mab.item())
        rho_t = torch.tensor(cfg.lambda_data / (sigma_t_bar ** 2 + 1e-8))
        counts_1d = counts.reshape(B * H * W, 1, T)
        x0_hat_1d = anscombe_data_step(
            x0, counts_1d, bin_size=1.0, dt_spad=1e-5,
            rho_t=rho_t, sigma_t_bar=sigma_t_bar,
            n_iter=cfg.pp_iters, lr_scale=cfg.pp_lr_scale,
        )
        x0_hat = x0_hat_1d.reshape(B, H, W, T)

        if step_idx < len(timesteps) - 1:
            t_prev = int(timesteps[step_idx + 1])
            ab_p = alphas_cumprod[t_prev]
            eps_hat = (x_t - sqrt_ab * x0_hat) / (sqrt_1mab + 1e-8)
            sigma_t = cfg.eta * torch.sqrt(
                (1.0 - ab_p) / (1.0 - ab) * (1.0 - ab / ab_p)
            )
            dir_xt = torch.sqrt(torch.clamp(1.0 - ab_p - sigma_t ** 2, min=0.0)) * eps_hat
            noise = torch.randn_like(x_t)
            x_t = torch.sqrt(ab_p) * x0_hat + dir_xt + sigma_t * noise
        else:
            x_t = x0_hat

    max_diff = (out_factored - x_t).abs().max().item()
    assert max_diff < 1e-4, (
        f"factored(alpha_s=0) vs. hand-written 1D-only: max|Δ|={max_diff:.3e}"
    )
