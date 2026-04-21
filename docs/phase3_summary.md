# Factored 3D Prior — Status Summary

Reverse-sampling a SPAD flux cube `[B, H, W, T]` by convex-combining a 1D
temporal Tweedie estimate and a 2D spatial Tweedie estimate at every DDIM
step. 1D model is an improved-diffusion temporal UNet trained on log1p-flux;
2D model is guided-diffusion `256x256_diffusion_uncond.pt`.

## Mathematical formulation

**DDPM forward (shared by both priors, linear β schedule, T=1000):**

    q(x_t | x_0) = N( sqrt(α̅_t) · x_0 ,  (1 − α̅_t) · I )

**Tweedie x₀ estimate, per branch, from ε-prediction:**

    x̂₀(x_t, t) = ( x_t − sqrt(1 − α̅_t) · ε_θ(x_t, t) ) / sqrt(α̅_t)

**Factored score composition — convex combination in Tweedie space:**

    x̂₀ᶜᵒᵐᵇⁱⁿᵉᵈ = α_s · x̂₀ˢᵖᵃᵗⁱᵃˡ  +  (1 − α_s) · x̂₀ᵗᵉᵐᵖᵒʳᵃˡ

**Data subproblem (Binomial / Poisson, linear flux in log1p parametrization):**

    phi(x) = expm1(x).clamp(min=0)      # x is log1p(flux); phi is linear flux
    N_k    = phi · Δt_spad + dark_count
    p_k    = 1 − exp(−N_k)               # per-SPAD-frame detection probability
    n_k ~ Binomial(bin_size, p_k)

    nll    = −Σ_k [ n_k · log p_k + (bin_size − n_k) · log(1 − p_k) ]
    x̂₀ʰᵃᵗ = argmin_x  nll  +  w·(∫phi dt − N_obs)²  +  ½ · ρ_t · ‖x − x̂₀ᶜᵒᵐᵇⁱⁿᵉᵈ‖²
    ρ_t   = λ / ( (1 − α̅_t)/α̅_t  +  ε )

(Anscombe-VST path was removed 2026-04-20; it used `phi = exp(x)`, which is
wrong for a `log1p`-parametrised 1D model in the low-PPP regime.)

**DDIM update (η-stochastic, η=0.85):**

    ε̂     = ( x_t − sqrt(α̅_t)·x̂₀ʰᵃᵗ ) / sqrt(1 − α̅_t)
    σ_t   = η · sqrt( (1 − α̅_{t-1}) / (1 − α̅_t) · (1 − α̅_t / α̅_{t-1}) )
    x_{t-1} = sqrt(α̅_{t-1})·x̂₀ʰᵃᵗ + sqrt(1 − α̅_{t-1} − σ_t²)·ε̂ + σ_t·z

## 1D ↔ 2D bridge (Phase 2)

Affine in log1p-flux space, pinned to the 1D training distribution
`log1p(clip(flux, 0, None) + 1e-6)` with raw flux ∈ [0, 10000]:

    source_min = 0,  source_max = log1p(10000) ≈ 9.2103
    scale = 2 / (source_max − source_min) ≈ 0.2171

    A(x) : x_logflux → x_2d = scale · (x − source_min) − 1        (grayscale, then replicated to 3 channels)
    A⁻¹  : x_2d     → x_logflux = (x_2d.mean(dim=channel) + 1) / scale + source_min

## Noise-aligned t-shift (the fix we just landed)

Applying `A` to `x_t` rescales the *noise* by `scale`:

    A(x_t) = scale · sqrt(α̅_t) · x_0  +  scale · sqrt(1 − α̅_t) · ε  +  bias

So the 2D UNet, if queried at step t, receives variance `scale²·(1 − α̅_t)`
— only `0.047` of what it expects. Remedy: query it at `t'` where

    1 − α̅_{t'}  =  scale² · (1 − α̅_t)                            (variance match)

`t_prime_from_t` solves this by `searchsorted` on the monotone array
`1 − α̅` and clamps to `[0, T−1]`. Example (our `scale`):

    t = 999 → t' = 65        t = 500 → t' = ~62        t = 10 → t' = 1

**Residual approximation error.** The signal coefficient on `x_0` after
`A` is `scale·sqrt(α̅_t)`, but the 2D model reads it as `sqrt(α̅_{t'})`.
These match exactly only when `scale = 1`. This is the standard, accepted
bias of noise-only cross-domain diffusion bridges.

## Problems — solved

| # | Problem | Resolution |
|---|---------|-----------|
| 1 | **Channel mismatch** (1D grayscale ↔ 2D RGB) | `to_2d_space` replicate-3; `from_2d_space` mean over channels (exact round-trip in fp32) |
| 2 | **`learn_sigma=True`** on 2D checkpoint (6-channel output) | Drop variance head, ε-only composition |
| 3 | **Range mismatch** (log-flux vs [-1, +1]) | Affine bridge pinned to log1p training range |
| 4 | **α_s = 0 exact equivalence to 1D-only** | 2D branch short-circuited; bit-identity tested (`< 1e-4`) via mock-2D-fails-on-call AND hand-written 1D reference |
| 5 | **Noise-variance misalignment at 2D call** (the issue visible as oversmoothed α_s = 0.5 frames) | t-shift: `1 − α̅_{t'} = scale²·(1 − α̅_t)`; default-on in `FactoredConfig.align_2d_noise` |
| 6 | **RNG reproducibility across α_s** | Seed CPU `Generator` for init noise + `torch.manual_seed` for DDIM noise with deterministic offset |

## Problems — open

| # | Problem | Impact | Suggested fix |
|---|---------|--------|---------------|
| 1 | **Residual signal-coefficient bias after t-shift.** `scale·sqrt(α̅_t) ≠ sqrt(α̅_{t'})` unless `scale = 1`, so the 2D model under-estimates signal magnitude. | Low–medium; acceptable for POC | Either retrain 1D with `--normalize True` so `source_min/max = ±1` (preferred long-term), or accept as bias |
| 2 | **Bridge mean offset is not corrected by the t-shift.** With `source_min = 0`, `A(x) = scale·(x − 0) − 1` maps log1p(flux) = 0 → −1 in 2D space. At PPP = 0.1 most pixels sit near log1p ≈ 0, so the 2D UNet sees frames centered near −1, outside its ImageNet-[-1, +1] distribution. `t_prime_from_t` aligns *noise variance only*. | Medium | Recenter the bridge on the dataset log1p mean, or retrain the 1D prior with `--normalize True` so scale = 1 (also eliminates Problem #1) |
| 3 | **Spatial domain gap.** ImageNet-uncond as the 2D prior is a generic natural-image prior — not photon-counting-specific. At extreme crop sizes or unusual content it can hallucinate. | Medium (research, not bug) | Train a small 2D prior on SPAD-like spatial crops once the pipeline is stable |
| 4 | **2D UNet resolution mismatch.** The 256² UNet is fed 64²/128² frames; attention-resolution maps, receptive field, and GroupNorm statistics all assume 256². | Medium | Pad / upsample to 256 around the 2D call, crop back; or accept degraded 2D prior |
| 5 | **Per-step 2D cost still a concern at scale.** Now mitigated by `frame_stride_2d` + `t_prime_skip_below` (defaults cut the 2D workload by ≈4× and drop zero-contribution calls), but full T=1024 × 100 steps is still heavy. | Mitigated, not eliminated | Push `frame_stride_2d` higher for low-α_s runs; add fp16 inference for the 2D UNet |
| 6 | **1D-checkpoint / bridge sanity.** Nothing verifies the loaded 1D checkpoint was trained on `log1p(flux + 1e-6)` with range [0, 9.2103]. A checkpoint trained with `--normalize True` (range [−1, +1]) silently produces garbage. | Blocker if wrong ckpt loaded | Record `source_min/max` + log1p flag in checkpoint metadata; assert at load time |
| 7 | **`region_maps` field assumed by the Phase-3 spec does not exist** in the dataset. | Low (qualitative plot only) | Derive synthetically as a spatial-gradient threshold on GT log-flux; already wired in `docs/phase3_qualitative.py` |

## Potential blockers before the full sweep

- **1D checkpoint / bridge sanity (Open #6).** `evaluate_factored.py --ckpt_1d` currently requires a path the user supplies. No automatic check that the checkpoint's training transform (`log1p` with flux ∈ [0, 10000]) matches the bridge (`source_min=0`, `source_max=9.2103`).
- **Memory.** `[B=1, H=128, W=128, T=1024]` at fp32 = 64 MB per cube tensor; with 4 copies (x_t, x₀_temporal, x₀_spatial, x₀_combined) and the 2D UNet activations on top, a 16 GB GPU is the practical floor.
- **Schedule equality — now asserted.** `factored_sample_flux` now asserts `diffusion_1d.alphas_cumprod == diffusion_2d.alphas_cumprod` at entry (when α_s > 0).

## Changelog

- **2026-04-20.** Data step switched from Anscombe VST (`phi = exp(x)`) to
  Binomial / Poisson likelihood with `phi = expm1(x).clamp(min=0)` to match
  the 1D model's `log1p(flux)` training transform. `anscombe_data_step`
  removed; replacement is `factored_diffpir.poisson_data_step`. α_s = 0
  bit-identity test still passes.
- **2026-04-20.** `SEQUENCE_LENGTH` locked at 1024 in the notebook (the 1D
  UNet is not length-variable).
- **2026-04-20.** 2D-branch speedups: `FactoredConfig.frame_stride_2d`
  (process every k-th frame, broadcast to neighbours) and
  `t_prime_skip_below` (skip the 2D call when the mapped step t' collapses
  near 0, which happens quickly because scale ≈ 0.217). Default `chunk_2d`
  raised from 8 to 32. Schedule-equality assert added at sampler entry.

## Files

- `spatial_prior.py` — bridge, Tweedie wrapper, `t_prime_from_t`
- `factored_diffpir.py` — `FactoredConfig`, `factored_sample_flux`, SPAD sim, Anscombe data step
- `evaluate_factored.py` — PPP × α_s sweep driver
- `tests/test_spatial_prior.py` (13 tests) — bridge, round-trip, t-shift
- `tests/test_factored_alpha0_equivalence.py` (2 tests) — α_s = 0 bit-identity
- `docs/phase2_verify.py` / `docs/phase3_qualitative.py` — visual verification scripts
- `DiffPIR_3D_Factored.ipynb` — interactive visual notebook (log1p bridge, t-shift diagnostic)
