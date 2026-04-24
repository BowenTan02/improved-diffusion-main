# Posterior Sampling for 1D Photon Flux Estimation from SPAD Observations

This note derives the mathematical pipeline implemented in
`DiffPIR_1D_Flux_Estimation_10K.ipynb`. The goal is to recover a 1D
temporal photon flux $\phi:[0,T]\to\mathbb{R}_{>0}$ from binary SPAD
detections using a diffusion prior trained on *normalized log-flux*.

Unless stated otherwise, the main derivation assumes **no binning**:
each time index corresponds to exactly one SPAD frame. Binning is
treated as a minor extension in the appendix.

---

## 1. Signal parameterization

Discretize time into $L$ bins, $k=1,\ldots,L$, with $\Delta t = T/L$.

- **Flux:** $\phi_k>0$.
- **Log-flux:** $\ell_k = \log \phi_k$.
- **Normalized log-flux** (the space the diffusion model operates in):

$$
x_k \;=\; \frac{2\,\ell_k}{\ell_{\max}} - 1 \;\in\; [-1,1],
\qquad
\ell_{\max} = \log(10^{4}).
$$

The inverse mappings are

$$
\ell_k = \tfrac{\ell_{\max}}{2}(x_k+1),
\qquad
\phi_k = \exp\!\bigl(\tfrac{\ell_{\max}}{2}(x_k+1)\bigr).
$$

All diffusion-side quantities ($x_t$, $\hat x_0$, proximal terms) live
in the normalized domain $[-1,1]$. Denormalization to $\phi$ only
happens when evaluating the SPAD likelihood.

---

## 2. SPAD forward model

Let $I_k = \phi_k/\phi_{\max}$ denote intensity normalized to $[0,1]$.
The expected photon count at frame $k$ is

$$
N_k \;=\; a\,I_k + d
\;=\; s\,\phi_k + d,
\qquad
s \;=\; \frac{a}{\phi_{\max}},
\qquad
a \;=\; \frac{\text{PPP}}{\bar{I}},
$$

where $d$ is the dark-count rate and $s$ (`ppp_scale` in code) absorbs
both normalization and the photons-per-pixel target. A SPAD frame
reports a **binary** detection $y_k\in\{0,1\}$ with

$$
\Pr\{y_k=1\mid\phi_k\} \;=\; p_k \;=\; 1 - e^{-N_k}.
$$

Given the instantaneous flux, detections across frames are
conditionally independent.

---

## 3. Observation likelihood (Poisson)

The exact per-frame distribution is Bernoulli$(p_k)$ with
$p_k=1-e^{-N_k}$, which comes from the fact that a Poisson-distributed
number of photons $n_k^{\star}\sim\mathrm{Poisson}(N_k)$ is observed
through a binary detector $y_k=\mathbf{1}\{n_k^{\star}\ge 1\}$. In the
SPAD low-flux regime ($N_k\ll 1$) the probability of pile-up is
negligible, so working directly with the *underlying* Poisson count is
both accurate and numerically much better behaved. We therefore model
the observation as

$$
n_k \mid \phi_k \;\sim\; \mathrm{Poisson}(N_k),
\qquad
N_k \;=\; s\,\phi_k + d,
$$

with $n_k\in\{0,1\}$ in practice because the detector saturates at one
event per frame. The joint likelihood factorizes over time indices and
the negative log-likelihood (dropping the constant $\log n_k!$) is

$$
\boxed{\;
\mathcal{L}_{\text{data}}(x)
\;=\; \sum_{k=1}^{L}\Bigl[\,N_k \;-\; n_k\,\log N_k\,\Bigr],
\;}
$$

with $N_k = s\,\exp\!\bigl(\tfrac{\ell_{\max}}{2}(x_k+1)\bigr)+d$.

### Gradient w.r.t. normalized log-flux

By the chain rule, using $\partial N_k/\partial \ell_k = s\,\phi_k$ and
$\partial \ell_k/\partial x_k = \ell_{\max}/2$,

$$
\frac{\partial \mathcal{L}_{\text{data}}}{\partial x_k}
\;=\;
\Bigl(1 \;-\; \frac{n_k}{N_k}\Bigr)\,
s\,\phi_k\,\frac{\ell_{\max}}{2}.
$$

This is the gradient used inside the data subproblem below. Unlike the
Bernoulli NLL, this expression does **not** saturate as $N_k\to\infty$:
the $N_k$ term in $\mathcal{L}_{\text{data}}$ grows linearly with flux,
so the gradient keeps pulling $\phi$ back toward $n_k/s$ even when the
current estimate overshoots the top of the normalized range.

---

## 4. Diffusion prior

A 1D UNet $\epsilon_\theta$ is trained on clean normalized log-flux
samples $x_0\sim p_{\text{data}}$ with the standard DDPM forward
process

$$
q(x_t\mid x_0) \;=\; \mathcal{N}\!\bigl(x_t;\sqrt{\bar\alpha_t}\,x_0,\;(1-\bar\alpha_t)\,\mathbf{I}\bigr),
\qquad t=0,\ldots,T_{\!d}{-}1,
$$

where $\bar\alpha_t=\prod_{s\le t}\alpha_s$ is the cumulative noise
schedule ($T_d=1000$ in code).

**Tweedie (one-shot clean estimate):**

$$
\hat x_0(x_t,t) \;=\; \frac{x_t \;-\; \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}.
$$

---

## 5. Posterior sampling: DiffPIR

We seek samples from $p(x_0\mid y)\propto p(y\mid x_0)\,p(x_0)$. DiffPIR
splits each reverse step into a prior-side Tweedie estimate, a data
fidelity proximal update, and a DDIM renoising step.

Define the noise-to-signal ratio at diffusion time $t$

$$
\sigma_t^{2} \;=\; \frac{1-\bar\alpha_t}{\bar\alpha_t},
\qquad
\rho_t \;=\; \frac{\lambda}{\sigma_t^{2}},
$$

with $\lambda>0$ the data-fidelity weight (`LAMBDA_DATA`).

At each reverse step $t\to t'$ (with $t'<t$):

### Step A — Prior estimate

$$
\hat x_0^{(t)} \;=\;
\mathrm{clip}_{[-1,1]}\!\left(
\frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}
\right).
$$

Clipping keeps the iterate inside the training support of the prior.

### Step B — Data subproblem (proximal update)

$$
\boxed{\;
\tilde x_0^{(t)}
\;=\;
\arg\min_{x\in[-1,1]^L}
\;\; \mathcal{L}_{\text{data}}(x)
\;+\;\frac{\rho_t}{2}\,\bigl\|x-\hat x_0^{(t)}\bigr\|_2^{2}.
\;}
$$

This is the only place the SPAD model enters. Because
$\mathcal{L}_{\text{data}}$ has no closed-form proximal operator under
the Poisson link in $x$, the minimization is carried out with a few
steps of projected gradient descent on $x$, using the gradient of §3
and projecting onto $[-1,1]$. (The adaptive step size used in code is
a practical choice and is intentionally omitted here.)

### Step C — DDIM renoising to $x_{t'}$

Reconstruct an implied noise from $\tilde x_0^{(t)}$,

$$
\hat\epsilon \;=\; \frac{x_t - \sqrt{\bar\alpha_t}\,\tilde x_0^{(t)}}{\sqrt{1-\bar\alpha_t}},
$$

and update with DDIM stochasticity $\eta\in[0,1]$:

$$
\sigma_t^{\text{DDIM}}
\;=\;
\eta\sqrt{\;\frac{1-\bar\alpha_{t'}}{1-\bar\alpha_t}\,
\Bigl(1-\frac{\bar\alpha_t}{\bar\alpha_{t'}}\Bigr)\;},
$$

$$
\boxed{\;
x_{t'}
\;=\;
\sqrt{\bar\alpha_{t'}}\,\tilde x_0^{(t)}
\;+\;\sqrt{1-\bar\alpha_{t'}-(\sigma_t^{\text{DDIM}})^{2}}\;\hat\epsilon
\;+\;\sigma_t^{\text{DDIM}}\,z,
\quad z\sim\mathcal{N}(0,\mathbf{I}).
\;}
$$

At the final step ($t'=0$) we simply return $\tilde x_0^{(t)}$.

### Complete algorithm

Sampling schedule: $t_0=T_d-1 > t_1 > \cdots > t_{S-1}=0$ ($S$=200 in code).

```
x ~ N(0, I)                             # initialize in normalized space
for s = 0,...,S-1:
    t  = t_s
    # A. prior
    x0_pred = clip( (x - sqrt(1-abar_t) * eps_theta(x, t)) / sqrt(abar_t) )
    # B. data
    rho = lambda / ((1-abar_t)/abar_t)
    x0_hat  = argmin_x [ L_data(x) + rho/2 * ||x - x0_pred||^2 ]   # PGD, clip to [-1,1]
    # C. DDIM
    if s < S-1:
        x = DDIM_renoise(x0_hat, x, t, t_{s+1}, eta)
    else:
        x = x0_hat
return phi = exp( (x+1) * log_flux_max / 2 )
```

---

## 6. Why this is posterior sampling

DiffPIR's plug-and-play structure corresponds to a half-quadratic
splitting of the MAP / posterior objective

$$
\min_{x_0}\; \mathcal{L}_{\text{data}}(x_0)\;-\;\log p(x_0),
$$

where the diffusion score implicitly represents $\nabla\log p(x_0)$.
Step A uses the score (via Tweedie) to move toward the prior manifold;
Step B pulls the estimate toward the SPAD data; Step C renoises to the
next marginal $q(x_{t'}\mid \tilde x_0^{(t)})$ so the next iteration
sees a point in the diffusion model's training distribution. Stochastic
DDIM ($\eta>0$) injects noise that, combined with random
initialization, yields *samples* from the posterior rather than a
single mode.

---

## Appendix A — Bernoulli / Binomial alternative

The Poisson likelihood of §3 is a small-$N_k$ approximation to the
*exact* per-frame Bernoulli model. The exact (unbinned) likelihood is

$$
y_k\mid\phi_k \;\sim\; \mathrm{Bernoulli}(p_k),
\qquad
p_k \;=\; 1-e^{-N_k},
$$

with NLL

$$
\mathcal{L}_{\text{data}}^{\text{Bern}}(x)
\;=\; -\sum_{k=1}^{L}\Bigl[\,y_k\log p_k \;+\;(1-y_k)\log(1-p_k)\,\Bigr].
$$

Its gradient w.r.t. $x_k$ is

$$
\frac{\partial\mathcal{L}_{\text{data}}^{\text{Bern}}}{\partial x_k}
\;=\;
\Bigl[\,(1-y_k) \;-\; y_k\,\frac{e^{-N_k}}{p_k}\,\Bigr]
\cdot s\,\phi_k\cdot\frac{\ell_{\max}}{2}.
$$

Plugging $\mathcal{L}_{\text{data}}^{\text{Bern}}$ into the data
subproblem (Step B) is a drop-in replacement; the prior and DDIM update
are unchanged. Poisson and Bernoulli agree to leading order when
$N_k\ll 1$ (since $p_k\approx N_k$ and $\log(1-p_k)\approx -N_k$), which
is exactly the SPAD low-flux regime.

## Appendix B — Binning of SPAD frames

In practice the SPAD runs at a much higher frame rate than the model's
sequence length. Let $B_k$ be the number of frames falling in bin $k$
and assume the flux is approximately constant within a bin. Aggregating
the underlying Poisson counts gives

$$
c_k \;=\; \sum_{j\in\text{bin }k} n_j
\;\sim\; \mathrm{Poisson}\!\bigl(\mu_k\bigr),
\qquad
\mu_k \;=\; B_k\,N_k \;=\; B_k\,(s\,\phi_k+d),
$$

so the Poisson data term generalizes to

$$
\mathcal{L}_{\text{data}}^{\text{binned}}(x)
\;=\; \sum_{k=1}^{L}\Bigl[\,\mu_k \;-\; c_k\log\mu_k\,\Bigr].
$$

Setting $B_k\equiv 1$ recovers §3. Under the exact Bernoulli model, the
bin count is Binomial$(B_k,p_k)$; when $p_k\ll 1$, Binomial$\to$Poisson
with rate $B_k p_k \approx B_k N_k = \mu_k$, so both roads meet.

## Appendix C — Practical terms omitted from the derivation

The implementation additionally uses (a) an adaptive gradient step
size derived from the local Lipschitz bound
$L_x = (\ell_{\max}/2)^2\,\max_k \mu_k$ on the Poisson NLL,
and (b) a soft total-count constraint
$\bigl(\sum_k \phi_k\Delta t - N_{\text{obs}}\bigr)^2$ that stabilizes
early diffusion steps when the prior estimate is still far from the
data. These are optimization-side aids and do not change the
probabilistic model specified above.
