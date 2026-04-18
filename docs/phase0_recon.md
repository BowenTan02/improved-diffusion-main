# Phase 0 — Score Composition Reconnaissance

Scope: survey the current 1D temporal diffusion prior and the available 2D checkpoints to determine whether a factored spatial+temporal score composition (1D temporal ⊕ 2D spatial) is viable inside a DiffPIR-style reverse loop.

Repo root: `/Users/tan583/Documents/Diffusion/improved-diffusion-main`

---

## 1. 1D Temporal Diffusion Prior — Current State

### 1.1 Noise schedule

- **Schedule**: `linear` (default and the value used in every training command in the project README).
  - Source: `temporal_model_and_diffusion_defaults` at [improved_diffusion/temporal_script_util.py:29](improved_diffusion/temporal_script_util.py:29) → `noise_schedule="linear"`.
  - README training invocations all pass `--noise_schedule linear` ([ReadMe.md:51](ReadMe.md:51), [ReadMe.md:55](ReadMe.md:55)).
  - Inference notebook `../DiffPIR_1D_Flux_Estimation.ipynb` sets `'noise_schedule': 'linear'`.
- **Beta range**: `beta_start = 0.0001 * scale`, `beta_end = 0.02 * scale`, where `scale = 1000 / num_diffusion_timesteps`. With `T = 1000` the scale is 1.0, giving the canonical Ho et al. values `(1e-4, 0.02)`.
  - Source: [improved_diffusion/gaussian_diffusion.py:27-35](improved_diffusion/gaussian_diffusion.py:27).
- **No s-offset** (s-offset is specific to the cosine schedule, which is not used here).

### 1.2 Number of diffusion timesteps

- `**T_diff = 1000`** (DDPM training). Confirmed in:
  - Default: `diffusion_steps=1000` at [improved_diffusion/temporal_script_util.py:28](improved_diffusion/temporal_script_util.py:28).
  - README training commands: `--diffusion_steps 1000` at [ReadMe.md:50](ReadMe.md:50), [ReadMe.md:55](ReadMe.md:55).
  - Inference notebook: `NUM_DIFFUSION_STEPS = 1000`.
- Sampling supports DDIM respacing (e.g. `--timestep_respacing "50"`), but the underlying trained prior is 1000-step.

### 1.3 Parameterization

- **Epsilon-prediction** (ε-pred).
  - Default `predict_xstart=False` at [improved_diffusion/temporal_script_util.py:32](improved_diffusion/temporal_script_util.py:32).
  - Creation maps this to `ModelMeanType.EPSILON` at [improved_diffusion/temporal_script_util.py:167-169](improved_diffusion/temporal_script_util.py:167).
  - Training target is the injected Gaussian noise, loss is MSE:
    ```
    target = {ModelMeanType.EPSILON: noise, ...}[self.model_mean_type]
    terms["mse"] = mean_flat((target - model_output) ** 2)
    ```
    at [improved_diffusion/gaussian_diffusion.py:734-742](improved_diffusion/gaussian_diffusion.py:734).
  - `learn_sigma=False` ⇒ single-channel output interpreted directly as ε(x_t, t); variance is fixed (`FIXED_LARGE` unless `sigma_small=True`).

### 1.4 Input normalization (log-flux)

The code has **two layers** of normalization, and they can disagree — flag this before composition.

1. Dataset layer — `TemporalDataset` at [improved_diffusion/temporal_datasets.py:107-115](improved_diffusion/temporal_datasets.py:107):
  - If `normalize=True` (default), rescales the full tensor via global min/max to `[-1, 1]`:
  - This is a **global** (dataset-wide) min-max, not per-sample — the learned prior lives in that fixed range.
2. Training commands in the README actually pass `**--normalize False`** ([ReadMe.md:51](ReadMe.md:51), [ReadMe.md:55](ReadMe.md:55)), which means the checkpoints in use were trained on **raw log-flux values**, not on `[-1, 1]`.
3. The DiffPIR notebook applies its own pre-scaling (`flux_normalized = flux / flux.max()`, then affine `a*x + d`) to match the photon-rate parameterization, independent of the dataset loader.

**Implication**: The trained 1D prior expects inputs in a distribution set by the log-flux statistics of the training set (approximately mean/variance of log-flux), not `[-1, 1]`. Any 2D prior composed with it must operate in a commensurate numerical range, or the relative score magnitudes will be miscalibrated at every `t`.

### 1.5 Tensor shapes

- **UNet input**: `[B, C=1, T]` — single-channel 1D signal. Confirmed at:
  - Dataset return: `sequence.unsqueeze(0)` → `[1, T]` per item, [improved_diffusion/temporal_datasets.py:134](improved_diffusion/temporal_datasets.py:134).
  - Sampler shape tuple `(batch_size, 1, args.sequence_length)` at [scripts/temporal_sample.py:67](scripts/temporal_sample.py:67).
  - `in_channels=1`, `dims=1` in `create_temporal_model` at [improved_diffusion/temporal_script_util.py:122-130](improved_diffusion/temporal_script_util.py:122).
- **UNet output**: `[B, 1, T]` — `out_channels = 1 if not learn_sigma else 2` at [improved_diffusion/temporal_script_util.py:124](improved_diffusion/temporal_script_util.py:124). Since `learn_sigma=False`, output channel count is 1.
- `T` varies by run: 1024 (default inference), 2000 (default config), or 10240 (10K dataset).

### 1.6 Key file paths


| Role                             | Path                                                                                            |
| -------------------------------- | ----------------------------------------------------------------------------------------------- |
| Model wrapper / diffusion config | [improved_diffusion/temporal_script_util.py](improved_diffusion/temporal_script_util.py)        |
| UNet definition (shared w/ 2D)   | [improved_diffusion/unet.py](improved_diffusion/unet.py) (`UNetModel`, called with `dims=1`)    |
| Diffusion math / schedule / loss | [improved_diffusion/gaussian_diffusion.py](improved_diffusion/gaussian_diffusion.py)            |
| Data loading / normalization     | [improved_diffusion/temporal_datasets.py](improved_diffusion/temporal_datasets.py)              |
| Training entry point             | [scripts/temporal_train.py](scripts/temporal_train.py)                                          |
| Sampling entry point             | [scripts/temporal_sample.py](scripts/temporal_sample.py)                                        |
| DDIM-respaced reverse process    | [improved_diffusion/respace.py](improved_diffusion/respace.py)                                  |
| DiffPIR 1D integration (current) | `../DiffPIR_1D_Flux_Estimation.ipynb`, `../DiffPIR_1D_Flux_Estimation_Anscombe.ipynb`, variants |


---

## 2. improved-diffusion Repo + Local 2D Checkpoints

### 2.1 Repo presence

- The project **is** a fork / adaptation of openai/improved-diffusion. The original 2D machinery is all still present and functional:
  - [scripts/image_train.py](scripts/image_train.py) — 2D training entry.
  - `improved_diffusion/script_util.py` — `model_and_diffusion_defaults()` for images, with `image_size=64`, `num_channels=128`, `learn_sigma=False`, `diffusion_steps=1000`, `noise_schedule="linear"` as defaults (but the public OpenAI checkpoints were trained with different flags).
  - `improved_diffusion/image_datasets.py` — image loader.
- There is a sibling DiffPIR repo at `../DiffPIR-main` with its own `guided_diffusion` (Dhariwal/Nichol) fork, configs (`deblur.yaml`, `inpaint.yaml`, `sisr.yaml`), and an **empty** `model_zoo/` (only a README listing *download URLs*, no weights).

### 2.2 Local pretrained checkpoints

Filesystem scan for `*.pt / *.ckpt / *.pth / *.bin` under `/Users/tan583/Documents/Diffusion` and `/Users/tan583`:


| Path                                                    | Notes                                                                                                                                                                                                                                                       |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/Users/tan583/Documents/Diffusion/SiT/SiT-XL-2-256.pt` | **SiT (Scalable Interpolant Transformer), not improved-diffusion.** Different architecture (DiT backbone), different formulation (stochastic interpolant / flow). Cannot be dropped into the UNet + Gaussian-diffusion DiffPIR loop without a full rewrite. |


**There are no local improved-diffusion or guided-diffusion 2D checkpoints.** The paths listed in `DiffPIR-main/model_zoo/README.md` (`256x256_diffusion_uncond.pt`, `ffhq_10m.pt`) are remote URLs — nothing has been downloaded.

### 2.3 Public checkpoints commonly used with this repo

Since nothing is on disk, the realistic candidate set is the publicly hosted ones. For each, the training configuration documented by OpenAI / Nichol & Dhariwal (and therefore what you'd need to match) is:


| Candidate (remote)                          | Dataset      | Res | Schedule | T    | Parameterization                   | Channels |
| ------------------------------------------- | ------------ | --- | -------- | ---- | ---------------------------------- | -------- |
| improved-diffusion CIFAR-10                 | CIFAR-10     | 32  | cosine   | 4000 | L_hybrid (ε + learned σ)           | 3 (RGB)  |
| improved-diffusion ImageNet-64              | ImageNet     | 64  | cosine   | 4000 | L_hybrid (ε + learned σ)           | 3 (RGB)  |
| guided-diffusion `256x256_diffusion_uncond` | ImageNet-256 | 256 | linear   | 1000 | ε-pred, `learn_sigma=True` (range) | 3 (RGB)  |
| DPS / DiffPIR `ffhq_10m`                    | FFHQ         | 256 | linear   | 1000 | ε-pred, `learn_sigma=True`         | 3 (RGB)  |
| guided-diffusion LSUN (bedroom/cat/horse)   | LSUN         | 256 | linear   | 1000 | ε-pred, `learn_sigma=True`         | 3 (RGB)  |


> These numbers come from the published configs; **verify against the actual `state_dict` keys and the config JSON shipped with each checkpoint at download time** (parameter shapes for `out_channels` reveal `learn_sigma` unambiguously: 3 vs 6 output channels).

---

## 3. Compatibility Matrix

Target to match (1D prior): **linear β∈(1e-4, 0.02), T=1000, ε-prediction, learn_sigma=False, single channel, input range ≈ raw log-flux (not [-1,1])**.


| 2D Candidate                            | Schedule  | T            | Parameterization              | Channels match   | Compatibility        | Notes                                                                                                                                                                                                                                   |
| --------------------------------------- | --------- | ------------ | ----------------------------- | ---------------- | -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CIFAR-10 (improved-diffusion)           | cosine ✗  | 4000 ✗       | ε + learned σ (hybrid) ~      | 3 vs 1 ✗         | **Blocker**          | Schedule AND step count mismatch. Requires rederivation of `ᾱ_t`; composition under mismatched `ᾱ_t(t)` is incoherent.                                                                                                                  |
| ImageNet-64 (improved-diffusion)        | cosine ✗  | 4000 ✗       | ε + learned σ (hybrid) ~      | 3 vs 1 ✗         | **Blocker**          | Same blockers as CIFAR. Would need respacing to 1000 steps on a schedule that still doesn't match 1D's linear.                                                                                                                          |
| ImageNet-256 `256x256_diffusion_uncond` | linear ✓  | 1000 ✓       | ε-pred (`learn_sigma=True`) ~ | 3 vs 1 ✗         | **Reconcilable-ish** | β range agrees with 1D. `learn_sigma=True` → just ignore the variance head and use the ε channels (first 3 of 6). Channel mismatch means you can't share a pixel grid — you'd apply the 2D score to each scan-line frame and broadcast. |
| FFHQ `ffhq_10m`                         | linear ✓  | 1000 ✓       | ε-pred (`learn_sigma=True`) ~ | 3 vs 1 ✗         | **Reconcilable-ish** | Same as above, but domain (faces) is far from SPAD flux — prior will pull toward face-like spatial structure, which is almost certainly wrong.                                                                                          |
| LSUN bedroom/cat/horse                  | linear ✓  | 1000 ✓       | ε-pred (`learn_sigma=True`) ~ | 3 vs 1 ✗         | **Reconcilable-ish** | Same schedule match; domain irrelevant for SPAD.                                                                                                                                                                                        |
| SiT-XL-2-256 (local)                    | flow/SI ✗ | continuous ✗ | velocity / interpolant ✗      | 4 (latent VAE) ✗ | **Blocker**          | Different diffusion framework entirely. Not score composable with DDPM ε-pred in one loop.                                                                                                                                              |


### 3.1 Classification

- **Exact match on {schedule, T, parameterization}**: **none**.
- **Reconcilable with simple transforms** (linear β at T=1000, ε-pred; drop learned-σ head; handle channel mismatch by applying 2D prior per-frame on a tiled/grayscale input): `256x256_diffusion_uncond`, `ffhq_10m`, LSUN variants — all via the guided-diffusion code path, not improved-diffusion's CIFAR/IN64 checkpoints.
  - Simple transforms required:
    1. Drop the variance head (`model_output[:, :3]` as ε; discard `model_output[:, 3:]`) — the guided-diffusion UNet emits 6 channels for learned σ.
    2. Grayscale→RGB replication (or apply spatial prior on each channel independently and average) to bridge C=3 vs C=1.
    3. Rescale log-flux to the checkpoint's expected `[-1, 1]` image range (guided-diffusion expects `[-1,1]`). This is a nontrivial domain-shift caveat, not just a scaling trick.
- **Blockers**: cosine-schedule or 4000-step checkpoints (CIFAR, ImageNet-64 from improved-diffusion) and SiT. Noise schedule mismatch cannot be made coherent by respacing because `ᾱ_t` is a different function of `t` — any composed score uses the **same `t`** argument in both branches, so `q(x_t | x_0)` must factor identically.

---

## 4. Blockers (explicit)

1. **No improved-diffusion 2D checkpoint is present on disk.** `DiffPIR-main/model_zoo/` and this repo's tree contain no 2D weights. The only local checkpoint is an unrelated SiT transformer.
2. **Domain mismatch is the hard blocker**, not schedule. Even the schedule-compatible guided-diffusion checkpoints (ImageNet-256 / FFHQ / LSUN) are trained on natural RGB images in `[-1, 1]`. SPAD flux 2D frames are a **different modality** — single-channel, different spatial statistics, different dynamic range. Using them as a spatial prior will bias reconstructions toward natural-image textures regardless of how clean the schedule bridge is.
3. **Learned-σ vs fixed-σ asymmetry.** The 1D prior has `learn_sigma=False` (fixed variance). The downloadable 2D checkpoints all have `learn_sigma=True`. For **ε composition** this is OK (drop the σ head). For any scheme that uses the full posterior mean/variance, it is a coherence issue worth flagging.
4. **Channel-count mismatch (3 vs 1).** Forces either (a) replicate grayscale SPAD → RGB and average 2D score back to 1, or (b) retrain a single-channel 2D model. (a) distorts the learned color-correlation structure of the 2D prior; it is a known-caveat workaround, not exact.
5. **Normalization inconsistency in the 1D pipeline itself.** The dataset loader's default `normalize=True` (global min-max to `[-1,1]`) is **not** what trained checkpoints were produced with (README uses `--normalize False`). This must be pinned down per-checkpoint before Phase 1, otherwise the 1D ε predictions will be miscalibrated even in isolation.

---

## 5. Recommendation

No pretrained 2D checkpoint is compatible **end-to-end** with the 1D prior for a clean factored-score DiffPIR loop. The closest practical option is the **guided-diffusion linear/1000-step family** (`256x256_diffusion_uncond`, FFHQ, LSUN) with the caveats in §3.1 and §4. Two forks in the road for Phase 1:

- **A. Retrain a small single-channel 2D SPAD-flux diffusion prior** using this repo's UNet with `dims=2, in_channels=1, learn_sigma=False, linear schedule, T=1000`. Maximum coherence with the 1D prior; cost is additional training. This is the recommended path if domain fidelity matters (which, for SPAD, it does).
- **B. Use a schedule-compatible guided-diffusion checkpoint as a rough spatial regularizer** with the ε-head-only, channel-replicated, range-remapped workaround. Acceptable as a sanity-check baseline, **not** as a publication-quality result — the natural-image prior will imprint wrong spatial structure.

Proceeding with **A** avoids compounding caveats. Proceeding with **B** first is only defensible as a quick-look to validate the composition loop machinery before committing GPU time to training.

---

## 6. Re-assessment of the §4 blockers (post-review)

After user pushback and a wider filesystem check — a `guided-diffusion` sibling repo exists at `/Users/tan583/Documents/Diffusion/guided-diffusion/` (not previously listed), containing the full guided-diffusion codebase plus [model-card.md](../../guided-diffusion/model-card.md) and [README.md](../../guided-diffusion/README.md) with working download links. Weights still need to be pulled, but the inference code path is in-repo.

Revisiting each blocker from §4 on this basis:

### 6.1 Pretrained 2D ImageNet-256 unconditional — yes, usable

From [guided-diffusion/README.md:70](../../guided-diffusion/README.md:70) the unconditional 256 flags are:

```
--diffusion_steps 1000  --noise_schedule linear  --image_size 256
--learn_sigma True  --num_channels 256  --num_head_channels 64
--num_res_blocks 2  --resblock_updown True  --use_scale_shift_norm True
--attention_resolutions 32,16,8
```

Script-util mapping ([guided-diffusion/guided_diffusion/script_util.py:410-420](../../guided-diffusion/guided_diffusion/script_util.py:410)):
`model_mean_type = EPSILON`, `model_var_type = LEARNED_RANGE`, `out_channels = 6` (3 ε + 3 σ-interpolation logits).

**Net**: schedule ✓, T ✓, ε-pred ✓. Only `learn_sigma` and channel count differ. No longer a blocker.

### 6.2 Domain mismatch — resolvable via affine (likelihood-side) and dataset pedigree

The user's point and the `phantom_simulation/generate_spad_dataset.py` forward model make this concrete. The data-generation pipeline is:

```
natural video → RIFE/linear temporal upsample → sRGB→linear RGB → luminance grayscale
              → flux  N(x,t) = a·I(x,t) + d         (calculate_flux_torch, line 355)
              → Poisson binary                      (line 641)
```

Two implications that flip the original assessment:

1. **The underlying x₀ distribution IS natural images.** Spatial frames in SPAD flux are luminance-projected natural video frames, up to an affine map. An ImageNet-256 prior is therefore an *appropriate* spatial prior for this experimental setup, not a foreign domain.
2. **The affine bridge λ = a·(x̂₀ + b)** (from the notation block the user supplied) goes on the **likelihood / measurement side**, not inside the score. The score network keeps operating in its native `[-1, 1]` frame; `a` and `b` enter only where `F(λ)` (the SPAD forward model) is evaluated against photon counts. This is exactly how DPS / DiffPIR already handle domain adaptation, and it is numerically well-conditioned as long as `a > 0` and `b` keeps `λ ≥ 0`.

**Net**: spatial-domain mismatch downgraded from "hard blocker" to "calibrate `(a, b)` from flux statistics at init time." Residual risk: ImageNet's sharp-texture / high-frequency bias may slightly over-sharpen SPAD reconstructions versus a prior trained on the specific scene class. Monitor, don't block.

### 6.3 Learned-σ vs fixed-σ — use ε-only; don't mix variance heads

How `learn_sigma=True` works in guided-diffusion (and improved-diffusion): the UNet outputs 6 channels for RGB — the first 3 are ε, the last 3 are raw logits `v` that interpolate between fixed-small and fixed-large variance via `Σ_θ = exp(v·log β + (1-v)·log β̃)` (Nichol & Dhariwal 2021, eq. 15).

For **score composition**, only ε is needed:

- Score is `s(x_t, t) = -ε_θ(x_t, t) / √(1-ᾱ_t)`. The variance head does not enter the score.
- The learned Σ only matters when **sampling** from p(x_{t-1}|x_t). In a DiffPIR loop, you can:
  - **Option 1 (recommended)**: after combining ε predictions, use the **fixed-variance** reverse step (either `FIXED_LARGE` or `FIXED_SMALL`) — i.e. ignore the 2D model's σ head and run the 1D schedule's fixed σ_t for both. This is consistent with how the 1D prior was trained and is the cleanest composition.
  - Option 2: use the 2D model's learned σ when only the 2D branch is stepping (e.g. in a cascaded rather than per-step-composed scheme). More complex; not worth the implementation cost for Phase 1.
- **Dropping the σ head**: `eps_2d = model_output[:, :3]`; discard `model_output[:, 3:]`. This is a three-line change in the forward hook.

**Net**: not a blocker. Use ε-only composition with fixed σ reverse step.

### 6.4 Channel-count mismatch (3 vs 1) — recommended scheme

Using the same luminance weights the dataset uses ([phantom_simulation/generate_spad_dataset.py:113-124](../../phantom_simulation/generate_spad_dataset.py:113), `rgb_to_grayscale_torch`), the cleanest bridge is **grayscale replicate in, luminance-project out**:

```
# Forward
x_rgb = x_gray.repeat(1, 3, 1, 1)                 # [B,1,H,W] -> [B,3,H,W]
out   = unet_2d(x_rgb, t)                         # [B,6,H,W]
eps_rgb = out[:, :3]                              # drop σ head
# Inverse projection — match the dataset's forward luminance
w = torch.tensor([0.2989, 0.5870, 0.1140], device=x.device).view(1,3,1,1)
eps_gray = (w * eps_rgb).sum(dim=1, keepdim=True) # [B,1,H,W]
```

Why this specifically:

- The dataset's grayscale → flux chain uses exactly these Rec.601 weights. For an input `x_gray`, the ImageNet model effectively sees a zero-chroma RGB, which is on-distribution for gray-world images; the luminance projection of ε inverts the same weighting the data generator applied.
- Simple replication + unweighted mean over channels is the **(a)** variant and will also work — the two differ only by channel weights. Luminance-weighted is the principled match to the forward model; plain mean is the lazy baseline.
- **Magnitude calibration caveat**: identical replicated channels mean the UNet's channel-correlation structure is effectively collapsed, so ε_rgb channels are strongly correlated rather than i.i.d. The luminance projection therefore has a magnitude somewhat below what a natively single-channel model would emit. In practice this shows up as a scalar rescaling factor in the score-composition weight (the `α` in `α·ε_1D + (1-α)·ε_2D`). Tune `α` on a small validation set; do not expect the theoretical `α=0.5` to be optimal.
- **Cleaner alternative (Phase-1 option)**: finetune the 2D model for 5–20k steps with in-channels adapted to 1 by **averaging the first-conv weight across input channels** (init: `w_new[:,0] = w_old.sum(dim=1)/3` — preserves the effective luminance response) and a single-channel output head (average the corresponding 3 ε-output channels). This removes the correlated-channel miscalibration at modest compute cost.

**Net**: workable without retraining via luminance in/out; better with a short finetune.

### 6.5 1D normalization inconsistency — normalize/denormalize around the 1D call, but with care

User is right that you can wrap the 1D model call with normalize/denormalize at inference. What matters is **that the normalized x_t fed to the 1D model matches the marginal distribution the 1D model was trained on at that same `t`**. Two sub-cases:

- **Current checkpoints (trained with `--normalize False`)**: the training marginal at step `t` is `N(√ᾱ_t · x₀_raw, (1-ᾱ_t)·I)` where `x₀_raw` has whatever empirical mean/std the log-flux dataset had — call it `(μ_raw, σ_raw)`. If σ_raw ≠ 1, sharing a single x_t in the [-1,1] frame with the 2D model feeds the 1D model an x_t whose signal-to-noise ratio *at a given `t`* differs from training. The ε prediction will be systematically miscalibrated. Applying an affine `(x_norm ↔ x_raw)` around the 1D call fixes the input range but **does not** fix the SNR mismatch at intermediate `t`, because the noise-schedule β_t is fixed; the only way to make the 1D marginal align with the 2D marginal is to have σ(x₀_raw) ≈ σ(x₀_norm) ≈ 1.
- **Preferred fix**: retrain the 1D prior with `--normalize True` so its x₀ lives in [-1, 1]. Cheap (it's a 1D model) and gives exact schedule-marginal alignment with the 2D prior. Composition is then theoretically clean — one noise schedule, one x_t, two compatible priors.
- **Acceptable workaround if retraining is off the table**: (i) compute `σ_raw` on the training set, (ii) apply `x_1d_input = (x_t - μ_raw·√ᾱ_t) / σ_raw` before the 1D UNet, (iii) rescale ε_1D output by `1/σ_raw` before composing. This is approximate — it corrects first-order scale but not the fact that the 1D UNet's internal normalization layers were trained with a different input std. Expect some residual bias, usable as a baseline.

**Net**: user's approach works at first order. For publication-grade composition, retrain the 1D prior with `normalize=True` — it is the cheapest single fix that eliminates the largest remaining source of incoherence.

---

## 7. Revised recommendation

The original §5 overstated the blocker status. Corrected assessment:

| Concern                              | Status                         | Action for Phase 1                                                                                                                    |
| ------------------------------------ | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| 2D checkpoint availability           | Resolved                       | `wget` `256x256_diffusion_uncond.pt` from guided-diffusion README, confirm 6-channel output head.                                     |
| Noise schedule / T / parameterization | Match (linear, 1000, ε)        | No action.                                                                                                                           |
| Domain mismatch                      | Resolvable                     | Apply affine `λ = a·(x̂₀ + b)` on the **likelihood side**; calibrate `(a, b)` from training-set flux statistics.                       |
| `learn_sigma` asymmetry              | Resolvable                     | Use ε-only composition with fixed-σ reverse step; discard 2D σ head.                                                                  |
| Channel count 3↔1                    | Resolvable, slight miscalibration | Luminance replicate/project; tune composition weight `α` on a held-out set. Short finetune to 1-channel if budget allows.          |
| 1D normalization                     | Resolvable, retrain preferred  | **Retrain 1D prior with `--normalize True`** so both priors share the [-1,1] x₀ frame. If retrain is off the table, use σ_raw affine rescale with known caveats. |

**Revised path forward**: proceed with pretrained `256x256_diffusion_uncond` as the 2D prior. The single action that most improves composition coherence is retraining (or finetuning from the existing checkpoint) the 1D prior with `normalize=True`. That plus the luminance-replicate/project bridge and a tuned composition weight `α` should give a coherent factored-score DiffPIR loop. Retraining a custom 2D SPAD prior (original §5 path A) is no longer necessary as a prerequisite — it remains a nice-to-have for later quality improvements.

### 7.1 Remaining real risks (not blockers, worth tracking)

- **ε magnitude miscalibration** from channel replication + different model capacities. Mitigate with per-`t` α(t) schedule rather than a single scalar α.
- **Cross-resolution mismatch**: ImageNet-256 operates at 256×256. SPAD frames from `generate_spad_dataset.py` have resolution W×H matching the source video (not necessarily 256). Need a crop/pad/resize strategy for the spatial prior call — probably tile-and-stitch, or resize-and-inverse-resize per reverse step.
- **Natural-image prior over-sharpening** in low-photon regimes. Monitor against ground-truth flux; if present, reduce α at high noise levels.