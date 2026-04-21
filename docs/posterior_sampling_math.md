# Posterior Sampling and Likelihood Guidance for 1D Photon-Flux Estimation

This note derives the mathematics behind the DiffPIR-style posterior sampler used
in `DiffPIR_1D_Flux_Estimation.ipynb`. The goal is to recover a continuous
photon-flux function $\phi(t)\ge 0$ from binary SPAD detections, using a
pre-trained diffusion prior on $x(t)=\log\phi(t)$.

---

## 1. Forward / Observation Model

### 1.1 Continuous photon flux

Photon arrivals are modeled as an inhomogeneous Poisson process with intensity
$\phi(t)$ over $t\in[0,T]$. The cumulative count process is

$$
N(t) \;=\; \int_0^t \phi(u)\,du \;+\; M(t),
$$

where $M(t)$ is a zero-mean martingale (the "Poisson noise"). We work in the
log-domain

$$
x(t) \;=\; \log\phi(t),
$$

so that positivity is enforced and the diffusion prior can be unconstrained.

### 1.2 Discrete SPAD detection model

The sensor integrates photons over short frames. Let the observation window
$[0,T]$ be divided into $F$ SPAD frames of duration $\Delta_f = T/F$. Within
frame $f$ centered at time $t_f$, the expected photon count per frame is

$$
N_f \;=\; a\,\tilde I(t_f) \;+\; d,
$$

where $\tilde I = \phi/\max(\phi)$ is the normalized flux, $a = \mathrm{ppp}/\bar{\tilde I}$ is the
scaling that enforces the target photons-per-pixel (PPP), and $d$ is the dark
count rate. Because the SPAD only reports whether at least one photon was
detected, the per-frame observation $b_f\in\{0,1\}$ satisfies

$$
\Pr\{b_f = 1 \mid \phi\} \;=\; p_f \;=\; 1 - \exp(-N_f).
$$

For computational efficiency we aggregate $F$ frames into $K$ bins of size
$s_k$ (number of frames in bin $k$), giving sufficient statistics

$$
y_k \;=\; \sum_{f\in \text{bin }k} b_f, \qquad y_k \sim \mathrm{Binomial}(s_k,\,p_k).
$$

This is the observation the solver actually consumes
(`bin_counts`, `bin_sizes`).

### 1.3 Likelihood

Treating bins as conditionally independent given $\phi$,

$$
\log p(y\mid x)
\;=\; \sum_{k=1}^{K}\Big[\,y_k\log p_k \;+\; (s_k - y_k)\log(1-p_k)\,\Big],
\qquad p_k = 1 - e^{-N_k},\ \ N_k = c\,e^{x_k}+d,
$$

with $c = a/\max\phi$ (the `ppp_scale` in code). This is the **exact data
likelihood** that replaces the Gaussian proximal term used in the original
DiffPIR for image restoration.

---

## 2. Bayesian Formulation

We want samples from the posterior

$$
p(x\mid y)\;\propto\;p(y\mid x)\,p(x),
$$

where $p(x)$ is the distribution of log-flux traces represented implicitly by a
pre-trained denoising diffusion probabilistic model (DDPM).

### 2.1 Diffusion prior

The forward (noising) process is

$$
q(x_t\mid x_0) \;=\; \mathcal N\!\left(x_t;\sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)\,I\right),
\qquad t=1,\dots,T_\text{diff},
$$

with $\bar\alpha_t = \prod_{s=1}^t (1-\beta_s)$. The score network
$\epsilon_\theta(x_t,t)$ approximates the noise added at step $t$, from which
Tweedie's formula gives the posterior mean of $x_0$:

$$
\hat x_0(x_t,t)
\;=\;
\frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}.
\tag{P}
$$

### 2.2 Posterior sampling target

The true reverse process is

$$
p_\theta(x_{t-1}\mid x_t,y) \;\propto\; p_\theta(x_{t-1}\mid x_t)\,p(y\mid x_{t-1}).
$$

Exact sampling is intractable because $p(y\mid x_{t-1})$ depends on the clean
signal. DiffPIR handles this via **plug-and-play half-quadratic splitting** on
the intermediate $\hat x_0$ estimate.

---

## 3. DiffPIR Half-Quadratic Splitting

At every reverse step, DiffPIR decomposes the MAP update into a prior step (the
denoiser) and a data step (the likelihood), coupled by a quadratic proximal
term.

### 3.1 Splitting

Introduce an auxiliary variable $z$ and write

$$
\min_{x,z}\;\;
\underbrace{\tfrac{1}{2\sigma_t^2}\,\|z-\hat x_0\|^2}_{\text{prior proximity}}
\;-\; \lambda\,\log p(y\mid z)
\;+\;\tfrac{\rho_t}{2}\|z-x\|^2,
$$

with $\sigma_t^2 = (1-\bar\alpha_t)/\bar\alpha_t$. Alternating minimization
gives the two subproblems.

### 3.2 Prior subproblem (denoiser)

Fix $z$ and minimize over $x_0$: the solution is precisely Tweedie's estimate
(P). In code:

```
x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * model(x_t, t)) / sqrt_alpha_bar_t
```

### 3.3 Data (likelihood) subproblem — the guidance equation

Fix $\hat x_0$ and solve

$$
\boxed{\;\;
\hat z \;=\; \arg\min_{z}\;\;
-\log p(y\mid z) \;+\; \frac{\rho_t}{2}\,\|z-\hat x_0\|^2
\;\;}
\tag{G}
$$

with guidance weight

$$
\rho_t \;=\; \frac{\lambda}{\sigma_t^2}
\;=\; \lambda\,\frac{\bar\alpha_t}{1-\bar\alpha_t}.
$$

$\rho_t$ starts small when $t$ is large (noisy, trust the prior more) and grows
as $t\to 0$ (trust the likelihood more). This is the **likelihood-guidance
equation**.

### 3.4 Explicit form for the SPAD Binomial likelihood

Substituting $\log p(y\mid z)$ and an integrated-flux regularizer used as a
Poisson-compensator anchor, the per-step objective minimized in
`spad_data_step_binomial` is

$$
\mathcal L(z)
\;=\;
-\sum_{k=1}^{K}\bigl[y_k\log p_k + (s_k-y_k)\log(1-p_k)\bigr]
\;+\;\frac{w}{N_\text{obs}}\Bigl(\sum_k e^{z_k}\,\Delta t - N_\text{obs}\Bigr)^2
\;+\;\frac{\rho_t}{2}\|z-\hat x_0\|^2,
$$

where

$$
p_k \;=\; 1-\exp(-N_k),\qquad N_k = c\,e^{z_k}+d,
$$

and $N_\text{obs}$ is the target flux integral obtained by inverting the mean
SPAD detection rate,

$$
N_\text{obs} \;=\; T\cdot\frac{\bar N - d}{c},
\qquad
\bar N \;=\; -\log\!\left(1-\tfrac{\sum_k y_k}{\sum_k s_k}\right).
$$

### 3.5 Gradients

Let $\phi_k = e^{z_k}$. The Binomial NLL gradient is

$$
\frac{\partial(-\log p)}{\partial z_k}
\;=\;
\underbrace{c\,\phi_k\,e^{-N_k}}_{=c\phi_k(1-p_k)}
\left[\frac{s_k-y_k}{1-p_k} - \frac{y_k}{p_k}\right],
$$

the quadratic-count penalty contributes
$\nabla = \frac{2w}{N_\text{obs}}\bigl(\sum_k \phi_k\Delta t - N_\text{obs}\bigr)\phi_k\,\Delta t$,
and the proximal term contributes $\rho_t(z-\hat x_0)$.

### 3.6 Adaptive step size

The Lipschitz constant of the Binomial NLL is dominated by

$$
L_\text{NLL}
\;\approx\;
\max_k s_k\,(c\phi_k)^2 e^{-N_k}.
$$

The solver uses

$$
\eta \;=\; \frac{\eta_0}{L_\text{NLL}+\rho_t+\varepsilon},
$$

a simple preconditioner that stabilizes gradient descent across the widely
varying scales of $\rho_t$ encountered during the reverse trajectory
(`PP_LR_SCALE / (L_nll + rho_t)`).

---

## 4. Full Sampling Loop (DDIM-style)

Given timesteps $t_0 > t_1 > \dots > t_{N-1}$ and $x_{t_0}\sim\mathcal N(0,I)$:

**For each $t \to t'$:**

1. **Denoising (prior):** predict $\hat x_0$ via (P).
2. **Guidance (likelihood):** solve (G) for $\hat z$ via $n$ steps of adaptive
   gradient descent on $\mathcal L$.
3. **DDIM update:** recompute the noise estimate from the guided $\hat z$,
   $$
   \hat\epsilon \;=\; \frac{x_t - \sqrt{\bar\alpha_t}\,\hat z}{\sqrt{1-\bar\alpha_t}},
   $$
   and take the DDIM step
   $$
   x_{t'}
   \;=\;
   \sqrt{\bar\alpha_{t'}}\,\hat z
   \;+\;
   \sqrt{1-\bar\alpha_{t'}-\sigma_t^2}\,\hat\epsilon
   \;+\;
   \sigma_t\,\varepsilon,\qquad \varepsilon\sim\mathcal N(0,I),
   $$
   with stochasticity controlled by
   $$
   \sigma_t
   \;=\;
   \eta\sqrt{\frac{1-\bar\alpha_{t'}}{1-\bar\alpha_t}\Bigl(1-\tfrac{\bar\alpha_t}{\bar\alpha_{t'}}\Bigr)}.
   $$

$\eta=0$ recovers deterministic DDIM; $\eta=1$ gives DDPM stochasticity. The
notebook uses $\eta=0.85$.

---

## 5. Why This Is "Posterior Sampling"

At each reverse step, injecting the guidance (G) into the DDIM update is
equivalent to approximating

$$
\nabla_{x_t}\log p(y\mid x_t)
\;\approx\;
\nabla_{x_t}\log p\!\bigl(y\mid \hat x_0(x_t)\bigr),
$$

a form of the **diffusion posterior sampling (DPS)** approximation
(Chung et al., 2023). DiffPIR differs from vanilla DPS in that it solves the
likelihood subproblem (G) to a higher accuracy via several inner iterations,
rather than taking a single gradient step. The stochastic DDIM update then
produces draws whose distribution targets $p(x\mid y)$ asymptotically under the
Tweedie / linearization assumptions.

---

## 6. Summary of Symbols

| Symbol | Meaning |
|---|---|
| $\phi(t)$ | photon flux |
| $x = \log\phi$ | log-flux (what the diffusion model generates) |
| $b_f\in\{0,1\}$ | per-frame SPAD detection |
| $y_k,\,s_k$ | bin-$k$ detection count / frame count |
| $c$ | `ppp_scale` mapping flux → expected photons/frame |
| $d$ | dark count |
| $p_k = 1-e^{-(c\phi_k+d)}$ | per-bin detection probability |
| $\bar\alpha_t$ | cumulative diffusion schedule |
| $\sigma_t^2$ | $(1-\bar\alpha_t)/\bar\alpha_t$ |
| $\rho_t = \lambda/\sigma_t^2$ | data-fidelity / guidance weight |
| $\hat x_0$ | Tweedie estimate (denoiser output) |
| $\hat z$ | guidance-corrected $x_0$ estimate |

---

## References

- Zhu et al., *Denoising Diffusion Models for Plug-and-Play Image Restoration* (DiffPIR), CVPR 2023.
- Chung et al., *Diffusion Posterior Sampling for General Noisy Inverse Problems* (DPS), ICLR 2023.
- Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020.
- Song et al., *Denoising Diffusion Implicit Models* (DDIM), ICLR 2021.
