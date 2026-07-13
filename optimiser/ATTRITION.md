# ATTRITION.md — correlated attrition + circuit conditioning

How the optimiser models **clustered retirements** and **venue-specific DNF
levels**. This is the joint-tail companion to the per-driver DNF *rate*
(`dnf_prior.py`) and the dispersion model (`DISPERSION.md`).

## Why the independent DNF layer wasn't enough

The DNF layer draws each driver's retirement independently:
`dnf_i = U_i < p_i`. That reproduces each driver's *marginal* DNF rate, but
it gets the **joint** distribution of who-retires-together wrong — and the
comp's underdog bonus lives entirely in the joint tail (a chaos race retires
several front-runners at once and promotes a *cluster* of underdogs into the
points).

An odds-free study of 79 races (OpenF1, 2023-26, see the repo's model-selection
notes) quantifies the gap:

| statistic | observed | independent Bernoulli | ratio |
|---|---:|---:|---:|
| DNF rate | 12.5% | — | — |
| Var(DNFs per race) | 3.09 | 2.18 | **1.41× overdispersed** |
| P(≥5 DNF) | 12.7% | 9.5% | 1.3× |
| P(≥7 DNF) | 3.8% | 0.8% | **4.6×** |
| P(≥8 DNF) | 1.3% | 0.2% | **6.9×** |

Seven-plus-car retirement races happen ~5× more often than independent draws
predict. A model-comparison on the DNF *count* distribution prefers a
**beta-binomial** (continuous shared shock, AIC 305) over both a binomial
(AIC 308) and a two-regime calm/chaos mixture (AIC 307) — so the structure is
a smooth race-level shock, not two discrete regimes. About 19% of all DNFs are
lap-≤2 retirements and 13% of races have a multi-car first-lap pileup, which is
a concrete mechanism for the shared shock.

## The model: one shared race shock

Each simulated race draws a single latent shock `Z ~ N(0, 1)` that shifts
**every** driver's DNF log-odds together:

```
d_i,race = logistic( b_i + λ · Z )
```

- `λ` (the *loading*) sets the strength of the correlation; `λ = 0` is exactly
  the old independent layer.
- `b_i` is chosen so the shock is **mean-preserving**: `E_Z[d_i,race] = p_i`,
  the driver's marginal DNF rate from `dnf_prior`/the market. So the fit and
  `validate` marginals are untouched — the shock only reshapes the joint tail.
  We use the probit approximation `E[σ(b+λZ)] ≈ σ(b/√(1+λ²π/8))`, inverted to
  `b_i = logit(p_i)·√(1+λ²π/8)` — closed-form and vectorised
  (`model/simulate.py::_shock_base_logit`).

Because the shift is in **log-odds**, a shared shock lifts moderate-reliability
midfielders more (in absolute probability) than ultra-reliable leaders —
matching who actually gets caught in first-lap chaos.

Implementation: `sample_finish_positions(..., shock_lambda=λ)` draws one `Z`
per race, computes `d_i,race`, then does the usual independent Bernoulli given
that race's probabilities. Everything downstream (demotion, argsort, scoring)
is unchanged.

## Calibrating λ to the tail

`λ` is fit to the **historical DNF-count tail**, not the variance — the tail is
what the underdog bonus is sensitive to, and a beta-binomial slightly under-fits
the extreme. `attrition.calibrate_lambda` grid-searches λ to minimise squared
error on the CCDF `P(K ≥ t)` for `t ∈ {4,…,8}`, simulating a homogeneous
20-car field at the grid rate. Matching the tail lands λ a touch higher than
matching the variance would (tail-match overdispersion ≈ 1.7× vs the 1.41×
variance figure) — deliberately, so the chaos races are frequent enough.

## Circuit conditioning

DNF rate varies **5%–24%** by venue (Monza/Spa low; Melbourne/Mexico/Monaco
high) — a bigger swing than the correlation effect. `attrition.circuit_rates`
computes a per-circuit rate, Beta-smoothed toward the grid average with
`circuit_prior_starts` pseudo-starts (a single track has few races). The
**circuit factor** = `circuit_rate / grid_rate` scales the DNF *level* for the
race being priced:

```
effective default_prob = default_prob × circuit_factor
```

This scales the season-prior re-centring anchor and the flat fallback (it does
**not** touch explicit `per_driver` config or market-implied DNFs, which are
already venue-specific). So on a low-attrition track the whole field's DNF
level drops. Note the live example: **Spa is the lowest-DNF circuit in the
data**, so the Belgian GP gets a factor < 1, pulling the level below the season
average.

## Config & CLI

```yaml
model.dnf.attrition:
  enabled: true
  shock: true                       # the shared race shock (correlated tail)
  circuit: true                     # venue DNF-level scaling
  calibration_years: [2023, 2024, 2025]
  circuit_prior_starts: 40
```

- `python main.py attrition` — prints the calibrated λ, the observed-vs-fitted
  tail, the per-circuit rates, and this race's factor. `--refresh-dnf` re-fetches.
- Calibration caches to `data/attrition_calibration.json` (raw counts to
  `data/attrition_races_<years>.json`); the fit reads it instantly and records
  `shock_lambda` into `model_fit_<race>.json`, so `validate`/`optimise` apply it.
- Disabled or OpenF1-unavailable → factor 1.0, λ 0.0 (exactly the old behaviour).

## Caveats & follow-ups

- **The fit stays shock-free.** Strength calibration runs on the independent
  marginals (which the shock preserves), so θ/σ are unaffected; the shock is
  applied only in the hard sims (`validate`/`optimise`). The second-order
  Jensen effect of the shock on top-k *marginals* is small but nonzero — watch
  the `validate` table.
- **Uniform loading.** Every driver shares one `λ`; a refinement would let the
  shock hit first-lap-exposed midfielders more than leaders, or condition the
  shock on wet/street circuits.
- **Bayesian path** (`--bayes`) does not yet apply the shock.
- The real test is whether the layer moves the **optimised ticket / underdog
  EV**, not the marginal MAD — the shock is invisible to the marginals by
  construction.
