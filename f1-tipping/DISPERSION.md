# Heterogeneous Dispersion: fixing rank-1 market incoherence

**Status: design / derivation.** This document works through the maths
of giving each driver a *center and a spread* — sampling race
performance from a per-driver distribution $X_i \sim F(\mu_i, \sigma_i)$
instead of the single-parameter Plackett–Luce (PL) of
[MATH.md](MATH.md). It stays on the **least-squares point-fit** path
(MATH.md §§1–6); the Bayesian path (§7) is touched only in a closing
aside. Nothing here is implemented yet — the doc is the specification
the code would follow, in the same "match the implementation, name the
approximations" style as MATH.md.

Notation follows MATH.md: $n$ drivers, positions **0-indexed** (position
$0$ wins, "top-$k$" means position $< k$), $\hat p$ for de-vigged market
probabilities, $\sigma(\cdot)$ for the logistic sigmoid, $\Phi(\cdot)$
for the standard normal CDF, $d_i$ for DNF probability.

---

## 1. The problem: what "market incoherence" is, precisely

MATH.md §3.3 names the load-bearing approximation:

> PL is rank-1: a single $\theta_i$ per driver cannot in general match
> win, top-3, top-6 and top-10 *simultaneously* … the fit returns a
> weighted-least-squares compromise, not an exact match.

Here is the mechanism, stated as a reachability fact. In the current
model each driver's latent race performance is

$$X_i = \theta_i + G_i, \qquad G_i \overset{\text{iid}}{\sim}
\text{Gumbel}(0,1),$$

and the finishing order is $X$ sorted descending (MATH.md §2.2). Every
driver shares the **same** noise law — unit Gumbel. So a driver's whole
finishing-position distribution, and hence its entire top-$k$ profile

$$\Pi_i \;=\; \big(\Pr(i\in\text{top }1),\ \Pr(i\in\text{top }3),\
\Pr(i\in\text{top }6),\ \Pr(i\in\text{top }10)\big),$$

is pinned by the single scalar $\theta_i$ (given the field). As $\theta_i$
varies, $\Pi_i$ traces a **one-dimensional curve** through the 4-cube of
possible profiles. The market hands us a target *point* in that cube. If
the point is off the curve — and generically it is — least squares lands
at the nearest point on the curve and leaves residuals. That gap is the
incoherence, and it is structural, not a fitting failure.

The direction of the miss is systematic and physical. Unit-Gumbel scale
fixes how *spread out* a driver's results are. Two kinds of driver break
it:

- a **volatile front-runner** — fast but fragile, or quick-but-erratic:
  the market gives it a high win probability *and* a real chance of
  finishing out of the points. It needs performance **more dispersed**
  than unit Gumbel: a ceiling high enough to win, a floor low enough to
  miss top-10.
- a **metronomic midfielder** — reliably P7–P9, never wins, never
  outside the points. It needs performance **less dispersed** than unit
  Gumbel.

Rank-1 PL can represent neither. The missing knob is a **per-driver
dispersion**.

### 1.1 A concrete reachability check

Take a fixed 19-driver reference field (unit-dispersion Gaussian
performances at spread-out locations) and a target driver the market
prices as a volatile favourite: $\Pr(\text{win}) = 30\%$ but
$\Pr(\text{top-10}) = 86\%$ (wins nearly a third of the time, yet misses
the points one race in seven). Sweeping the target's location only (the
rank-1 case, $\sigma_t = 1$) and matching each anchor in turn
(400k-race Monte Carlo):

| fit | $\mu_t$ | $\Pr(\text{win})$ | $\Pr(\text{top-10})$ |
|---|---|---|---|
| match win | $2.44$ | **30.0%** | 99.8% (target 86%, **+13.8pp**) |
| match top-10 | $0.47$ | 1.4% (target 30%, **−28.6pp**) | **86.0%** |

No single location is within 13 points of both. Now free the scale and
hold $\Pr(\text{win})=30\%$ pinned (choosing $\mu_t$ per $\sigma_t$):

| $\sigma_t$ | $\mu_t$ | $\Pr(\text{win})$ | $\Pr(\text{top-3})$ | $\Pr(\text{top-6})$ | $\Pr(\text{top-10})$ |
|---|---|---|---|---|---|
| 1.0 | 2.44 | 30.0% | 69.4% | 94.7% | 99.8% |
| 2.0 | 1.98 | 30.0% | 52.0% | 73.8% | 90.5% |
| 2.5 | 1.73 | 30.0% | 47.7% | 66.2% | 83.2% |
| 3.0 | 1.48 | 30.0% | 44.8% | 60.5% | 76.4% |

The spread knob slides $\Pr(\text{top-10})$ down the whole way while the
win probability stays nailed. Solving both anchors together gives
$\mu_t = 1.83,\ \sigma_t = 2.31$, reproducing $\Pr(\text{win}) = 30.0\%$
**and** $\Pr(\text{top-10}) = 86.0\%$ exactly, with top-3/top-6 falling
out at 49% / 69%. One extra parameter per driver turns an unreachable
target into an exact hit on its two dominant anchors.

> The demo field is Gaussian rather than Gumbel, because the fix below
> recommends Gaussian; the *qualitative* result — 1-D curve vs 2-D
> reachable region — is identical for any location–scale family,
> including the current Gumbel one.

---

## 2. The generalization: location–scale random utility

Give every driver a **center** and a **scale**:

$$\boxed{\,X_i = \mu_i + \sigma_i\,\varepsilon_i, \qquad
\varepsilon_i \overset{\text{iid}}{\sim} F,\quad \sigma_i > 0\,}$$

with $F$ a fixed standardised law (CDF $F$, PDF $f$, location 0, unit
scale). Each driver now carries two parameters: location $\mu_i$ (the old
$\theta_i$) and dispersion $\sigma_i$. The current model is the special
case $F = \text{Gumbel}(0,1)$, $\sigma_i \equiv 1$.

This is the **Thurstonian** random-utility family: performances are
independent draws, the ranking is their descending order. PL is its one
member with a closed-form winner (softmax) and closed-form pairwise
(logistic); the price of those closed forms is the fixed homogeneous
noise that causes the incoherence.

The engine barely changes. Simulation is still: draw $\varepsilon$, form
$X = \mu + \sigma\,\varepsilon$, apply the DNF demotion (MATH.md §2.3),
`argsort(-X)`. Same $O(n\log n)$ per race, same vectorisation, same
common random numbers. In `simulate.py::sample_finish_positions` the one
line

```python
scores = theta + rng.gumbel(size=(n_sims, n))
```

becomes (for the Gaussian choice of §4)

```python
scores = mu + sigma * rng.standard_normal(size=(n_sims, n))
```

with `sigma` an extra `(n,)` (or `(n_sims, n)`) array alongside `mu`.
Everything downstream of the `SimSet` — DNF layer, scoring, the
expected-points matrix, the Hungarian optimiser — consumes finishing
positions and is **completely untouched**.

---

## 3. Marginals under a location–scale model

What changes is the *analytics* used in fitting and validation. Redo
MATH.md §2.4 for general $(\mu, \sigma, F)$.

### 3.1 Winner

$$\Pr(i\text{ wins}) = \Pr(X_i \ge X_j\ \forall j\ne i)
= \int_{-\infty}^{\infty} \frac{1}{\sigma_i}
  f\!\Big(\frac{x-\mu_i}{\sigma_i}\Big)
  \prod_{j\ne i} F\!\Big(\frac{x-\mu_j}{\sigma_j}\Big)\, dx.$$

A single 1-D integral. With $F$ Gumbel and all $\sigma_i$ equal it
collapses to the softmax $e^{\mu_i}/\sum_j e^{\mu_j}$; with heterogeneous
$\sigma_i$ that collapse fails and there is **no closed form** — the very
degree of freedom we are adding is what breaks the softmax. (This is
fine: the winner marginal is estimated by simulation for validation, and
fitting never needed the closed form — see §5.)

### 3.2 Head-to-head (the one closed form that survives — and only for Gaussian)

$$\Pr(X_a > X_b) = \Pr\big(\mu_a - \mu_b > \sigma_b\varepsilon_b -
\sigma_a\varepsilon_a\big).$$

The right-hand side is a difference of two independent scaled noises; its
tractability depends entirely on $F$.

- **Gaussian.** $\sigma_b\varepsilon_b - \sigma_a\varepsilon_a \sim
  \mathcal N\!\big(0,\ \sigma_a^2 + \sigma_b^2\big)$, so

  $$\boxed{\ \Pr(a \succ b) = \Phi\!\Big(
    \frac{\mu_a - \mu_b}{\sqrt{\sigma_a^2 + \sigma_b^2}}\Big)\ }
    \qquad\text{(Thurstone Case III).}$$

  This is a clean, differentiable closed form — the probit pairwise
  model — and it *generalises* the current logistic H2H (which was
  $\sigma(\theta_a-\theta_b)$, the equal-scale limit up to the
  logistic-vs-probit link). It reduces to
  $\Phi\!\big((\mu_a-\mu_b)/(\sigma\sqrt2)\big)$ when scales are equal.

- **Heteroscedastic Gumbel.** A difference of two Gumbels of *different*
  scale has no elementary closed form (only *equal* scales give the
  logistic). So scaled-Gumbel gives up the H2H closed form that fitting
  currently leans on.

**DNF-adjusted H2H.** The four-case decomposition of MATH.md §2.4 carries
over verbatim; only the base pairwise term changes. With $u_a =
d_a(1-d_b)$, $u_b = d_b(1-d_a)$ (the "exactly one retires"
probabilities),

$$\Pr(a\succ b) = (1 - u_a - u_b)\,
  \Phi\!\Big(\frac{\mu_a-\mu_b}{\sqrt{\sigma_a^2+\sigma_b^2}}\Big)
  \;+\; u_b,$$

the drop-in replacement for `_analytic_h2h` in [fit.py](model/fit.py).
(The "both retire" row keeps the same $\Phi(\cdot)$ because both cars are
demoted equally, so their performance order stands — exactly as in the
Gumbel derivation.)

### 3.3 Top-$k$, $k > 1$

The Poisson-binomial structure of MATH.md §2.4 is unchanged, only
heteroscedastic: conditional on $X_i = x$, driver $i$ is top-$k$ iff at
most $k-1$ others exceed $x$, where now
$p_j(x) = \Pr(X_j > x) = 1 - F\!\big((x-\mu_j)/\sigma_j\big)$. There is
**no closed form** (there wasn't before either), so — exactly as today —
fitting uses the simulated soft-rank surrogate (§5) and validation uses
the hard Monte Carlo. Nothing new is lost here.

---

## 4. Which distribution? Gaussian, scaled-Gumbel, or Beta

The comparison axes: (a) which closed forms survive, (b) whether center
and spread are **independently** settable, (c) support, (d) code delta.

### 4.1 Gaussian — recommended

$X_i \sim \mathcal N(\mu_i, \sigma_i^2)$, order by descending draw.

- **Closed forms.** Keeps a clean H2H (probit, §3.2); loses only the
  winner softmax, which fitting never used and validation simulates.
- **Independence of center and spread.** $\mu \in \mathbb R$ and
  $\sigma > 0$ are fully decoupled: *any* (mean, variance) pair is legal.
  A dominant favourite can be given an arbitrarily fat downside tail —
  precisely the volatile-front-runner case that motivated this.
- **Support** $\mathbb R$: natural for a latent "pace" with no hard
  bounds, and the tails — where win/DNF action lives — are not squashed.
- **Code delta.** `rng.gumbel` → `rng.standard_normal`, times $\sigma$.
- **Cost.** Gumbel's asymmetric skew is gone (arguably a *feature*: a
  symmetric performance law is easier to reason about), and the winner
  softmax init must be replaced by a plain monotone location init (§5).

### 4.2 Scaled-Gumbel — the minimal deviation

$X_i = \theta_i + \sigma_i G_i$. With $\sigma_i \equiv 1$ this is
**exactly** today's model, so it is the smallest conceptual step and
preserves the "PL when homogeneous" story. But once scales differ it
surrenders **both** closed forms (winner *and* H2H), whereas Gaussian
hands one of them (H2H) back. So you keep PL's shape but lose its
conveniences anyway, and skewed noise makes $\sigma_i$ harder to
interpret. Choose this only if continuity with the existing PL
derivation is worth more than the H2H closed form.

### 4.3 Beta — tempting, but the mean–variance coupling fights us

The user floated Beta, and it *looks* right: parametrise
$\text{Beta}(\alpha,\beta)$ by mean $m = \alpha/(\alpha+\beta)$ and
concentration $\kappa = \alpha+\beta$, giving a center ($m$) and a spread
knob ($\kappa$), with

$$\operatorname{Var} = \frac{m(1-m)}{\kappa + 1}.$$

Three problems, one of them fatal:

1. **Fatal — the spread is not free.** Variance is *bounded by*
   $m(1-m)$ and collapses as $m \to 0$ or $m \to 1$. A driver the market
   makes a strong favourite ($m$ near 1) is **forced** to low variance:
   Beta structurally cannot give a dominant driver a fat downside. That
   is the exact *opposite* of the effect we are adding dispersion to
   capture — Beta's mean–variance link works against the objective.
2. **No closed forms.** Winner, top-$k$, and even H2H
   ($\Pr(\text{Beta}_a > \text{Beta}_b)$) are all non-elementary
   integrals — strictly worse analytics than Gaussian.
3. **Bounded support** $[0,1]$ compresses the strength scale for no
   benefit; ranking only cares about order, and the interesting mass is
   in the tails Beta squashes.

Beta is the right tool when the modelled quantity *is* a bounded rate you
are estimating (a classification probability, a batting average). Latent
racing pace, whose tails are the whole point, wants a real-line
location–scale law. **Use Gaussian.**

The rest of this doc assumes the Gaussian choice.

---

## 5. The revised least-squares fit

The objective keeps the shape of MATH.md §3.1, now over $(\mu, \sigma)$:

$$L(\mu, \sigma) = \sum_{k\in\mathcal M} \omega_k
  \sum_{i} \big(P^{\text{model}}_k(i;\mu,\sigma) - \hat p_k(i)\big)^2
  \;+\; \sum_{(a,b)\in\mathcal H} \omega_{ab}
  \big(\Pr(a\succ b;\mu,\sigma) - \hat p_{ab}\big)^2
  \;+\; \lambda_\mu \lVert\mu\rVert^2
  \;+\; \lambda_\sigma \lVert\log\sigma\rVert^2 .$$

Concretely, versus [fit.py](model/fit.py):

- **Parameter vector doubles** to $[\mu;\ \log\sigma] \in \mathbb
  R^{2n}$. Optimise on $\log\sigma$ so $\sigma > 0$ is automatic and the
  multiplicative gauge (§6) is a plain shift in $\log\sigma$-space.
- **The soft-rank surrogate is unchanged in form.** Fix $M$ draws of
  standard-normal noise $\varepsilon_{m,i}$ and DNF uniforms; per-draw
  scores become

  $$s_{m,i} = \mu_i + \sigma_i\,\varepsilon_{m,i} - D\,B_{m,i},$$

  then the identical soft rank
  $\widetilde{\text{rank}}_{m,i} = \sum_j \sigma\!\big((s_{m,j} -
  s_{m,i})/\tau\big) - \tfrac12$ and outer top-$k$ sigmoid of MATH.md
  §3.2 apply verbatim. The **only** change in the top-$k$ path is that
  the fixed noise is Gaussian and each column is scaled by $\sigma_i$.
- **H2H term** swaps the logistic `_analytic_h2h` for the probit
  DNF-adjusted form of §3.2.
- **Initialisation.** $\mu^{(0)}_i = \log\hat p_{\text{win}}(i)$
  (clipped), the same monotone location seed as today — it is no longer
  the exact winner optimum (no softmax) but is still a good start.
  $\sigma^{(0)}_i = 1$, or a coarse moment-match seed: a driver whose
  market top-10 sits low relative to its win price starts at
  $\sigma^{(0)}_i > 1$ (its results must be more spread), which the demo
  of §1.1 shows is the right direction.
- **Regularisation / gauge.** $\lambda_\sigma \lVert\log\sigma\rVert^2$
  pulls weakly-identified dispersions toward $\sigma_i = 1$ (unit,
  field-average spread) and, with mean-centring of $\mu$, pins both
  gauge freedoms (§6). Keep $\lambda_\mu = \lambda_\sigma = 10^{-4}$ as a
  starting point; $\sigma$'s regulariser is more load-bearing than
  $\mu$'s because a lone win price says nothing about spread.
- **Finite-difference discipline unchanged.** The float32 surrogate with
  the coarse `eps = 1e-3` step (MATH.md §3.3) still applies — indeed the
  Part-B staircase that stalled a naive solver in the prototype is the
  same effect the existing `eps` already handles.

**Degrees of freedom, honestly.** We now fit $2n - 2$ free numbers
(after the two gauges) against up to $4n$ top-$k$ constraints plus H2H —
still over-determined, still least squares. The second parameter absorbs
the *dominant* residual mode (the win-vs-bulk dispersion tension of §1),
so most of the systematic misfit should collapse; but two parameters per
driver still cannot exactly match four top-$k$ markets, so expect a
*smaller* residual, not zero. What is captured is the second moment of
each driver's finishing distribution; third-and-higher shape (e.g. the
top-3-vs-top-6 curvature) can still deviate. The validation table of
[validate.py](model/validate.py) — unchanged, since the hard engine is
already generic — is where that residual is read off, and the honest
success test is "did the flagged-marginal count and the fit loss drop
versus rank-1 on the same snapshot."

---

## 6. Identifiability: a second gauge appears

PL had one gauge: the additive shift $\theta_i \mapsto \theta_i + c$
(MATH.md §3.3). The location–scale model has **two**, because the ranking
is invariant under any common *affine* map $X_i \mapsto a X_i + b$ with
$a > 0$:

$$\mu_i \mapsto a\mu_i + b, \qquad \sigma_i \mapsto a\sigma_i.$$

- **Additive $b$** — the old shift. Fixed by **mean-centring $\mu$**
  (as now), which the objective does on every evaluation.
- **Multiplicative $a$** — **new**. Scaling all locations *and* all
  dispersions by the same positive constant changes no ranking, so only
  the *ratios* $\sigma_i/\sigma_j$ are identified — the exact analogue of
  $\theta$ being identified only up to a shift. Fix it by **centring
  $\log\sigma$** (geometric-mean $\sigma = 1$), i.e. renormalise
  $\log\sigma \mapsto \log\sigma - \overline{\log\sigma}$ each
  evaluation, mirroring the mean-centring of $\mu$.

With both constraints the fit is well-posed. Two further notes:

- **Weak identification of individual $\sigma_i$.** A driver with only a
  win price (or extreme, pinned longshot prices) has almost no leverage
  on its spread — the analogue of the weakly-constrained back-marker
  $\theta$. The $\lambda_\sigma$ regulariser toward $\sigma_i = 1$
  supplies the missing curvature, just as the $L_2$ anchor does for
  $\theta$.
- **Meaning of a unit.** After fixing the gauge, a $\mu$-unit is "one
  field-average dispersion," and $\sigma_i = 1.5$ reads as "50% more
  spread than the typical driver." That interpretability is a reason to
  normalise on $\sigma$ rather than, say, anchoring one driver.

---

## 7. Dispersion vs the DNF layer: a confound to watch

A low top-10 probability relative to a driver's win price can be
explained **two** ways: high dispersion $\sigma_i$ (fat lower tail, but
the car still classifies) *or* a high DNF probability $d_i$ (mass removed
to "unclassified" entirely). They are partially confounded.

Their **shapes differ**, which is what breaks the tie:

- DNF removes the driver outright — a spike of mass at the
  "unclassified, behind every finisher" outcome (MATH.md §2.3).
- Dispersion fattens the whole lower tail — more mass at P11–P15 while
  the car is still *classified*.

Where the **"To be Classified"** market prices $d_i$ directly (MATH.md
§§1.3, 2.3), the confound is broken at the source: $d_i$ is pinned by its
own market, and $\sigma_i$ is then left to explain only the residual
*classified* dispersion. Where $d_i$ falls back to the flat 0.10 season
prior, expect $\sigma_i$ and the (unpriced) attrition to trade off — a
driver's genuine retirement risk may get absorbed into an inflated
$\sigma_i$, or vice versa. This is a real approximation of the extended
model and belongs in the validation commentary; it is *lessened*, not
introduced, by adding dispersion (rank-1 PL has the same confound, with
even less freedom to resolve it).

---

## 8. What this buys, and what it does not

**Buys.** The dominant, systematic, physically-meaningful residual of the
current fit — the win-vs-top-10 dispersion tension — becomes
representable, so the fitted marginals track the markets far more
coherently (§1.1: an exactly-unreachable target becomes an exact hit on
its two anchors). Downstream, the expected-points matrix (MATH.md §5.3)
is computed from *better-calibrated* marginals, and since today's scoring
is additive, a more coherent marginal for each driver is exactly what the
Hungarian optimiser needs. The volatile-underdog structure the comp
rewards (a fast-but-fragile midfielder with genuine podium upside) is now
directly expressible as high $\sigma$, rather than being smeared into the
DNF layer.

**Does not buy.** It is still a rank-*one-per-moment* model: two numbers
per driver cannot match four top-$k$ markets exactly, so a residual
remains (smaller, and now second-order). It does not model
**cross-driver** correlation — teammate co-movement, track-specific pace,
common-cause attrition — which is a different extension (a low-rank or
team-factor structure on $\mu$, or correlated $\varepsilon$). And it adds
a genuine confound with the DNF layer where the classified market is
unpriced (§7).

**Relation to the existing $\sigma_{\text{obs}}$ (the one Bayesian
note).** MATH.md §7 infers a *global* observation-noise scale
$\sigma_{\text{obs}}$ that measures exactly this rank-1 misfit. The two
are complementary, not competing: per-driver $\sigma_i$ is a
**structural** fix inside the race model (it removes misfit), whereas
$\sigma_{\text{obs}}$ **measures** whatever misfit remains. If this
extension works, a Bayesian re-fit should show a *smaller* posterior
$\sigma_{\text{obs}}$ — the residual it has to absorb has shrunk. But
that is future work; the point-fit above stands on its own.

---

## 9. Implementation checklist (when we build it)

1. **`model/simulate.py`** — `sample_finish_positions` takes a `sigma`
   array; `scores = mu + sigma * rng.standard_normal(...)`. Keep the
   DNF demotion and `argsort` exactly as-is. `SimSet` is unchanged.
2. **`model/fit.py`** — parameter vector $[\mu;\log\sigma]$; fixed draws
   become `rng.standard_normal`; per-column $\times\,\sigma$; probit
   DNF-adjusted H2H replacing `_analytic_h2h`; centre $\mu$ **and**
   $\log\sigma$ each objective call; add $\lambda_\sigma\lVert\log
   \sigma\rVert^2$; seed $\sigma^{(0)} = 1$ (or moment-match). Persist
   `sigma` in the fit JSON alongside `theta`→`mu`.
3. **`model/validate.py`** — structurally unchanged (hard engine is
   generic); just thread `sigma` through the simulate call. The 2pp
   flag and the market-vs-sim table work as-is and are the success
   metric.
4. **Everything after the `SimSet`** — scoring, expected-points matrix,
   Hungarian optimiser, report — **untouched**.
5. **Tests** — extend `tests/test_simulate.py` with a Gaussian toy case
   whose H2H matches the probit closed form of §3.2, mirroring the
   existing 3-driver PL marginal test.

The change is deliberately surgical: one line in the simulator, a
doubled parameter vector and a swapped H2H term in the fit, and a
threaded-through `sigma` everywhere else.
