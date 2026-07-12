# Rank-1 PL vs Gaussian dispersion: market coherence & timing

This document reports a head-to-head of the two race models on a **real
odds snapshot**:

- **rank-1 PL** — the Plackett–Luce model of [MATH.md](MATH.md): one
  strength $\theta_i$ per driver, `model.dist: gumbel`.
- **Gaussian dispersion** — the Thurstonian model of
  [DISPERSION.md](DISPERSION.md): each driver gets a location $\mu_i$
  **and** a spread $\sigma_i$, `model.dist: gaussian`.

The question DISPERSION.md set out to answer: does a second per-driver
parameter actually buy **market coherence** (fitted marginals that
reproduce the market across win/top-3/top-6/top-10 at once), and what
does it **cost in time**? Both are measured here, not asserted.

## Method

Both models are fit on the **identical** de-vigged markets from one
snapshot, then each is hard-simulated (100k races, same seed) and its
simulated marginals are compared to the de-vigged market probabilities.
"Coherence" is the deviation of those simulated marginals from the market
— exactly what [validate.py](model/validate.py) prints, aggregated:

- **MAD** — mean absolute deviation, in percentage points (pp), over
  every (market, driver) observation.
- **RMSE** — root-mean-square deviation (pp); punishes big misses.
- **max dev** — the single worst (market, driver) miss (pp).
- **flagged >2pp** — count of observations past validate.py's 2pp flag.

Comparing *hard-simulated* marginals (not the fit's own surrogate loss)
is the fair test: it is apples-to-apples across the two models, whose
internal loss functions differ (probit vs logistic H2H, an extra
$\log\sigma$ regulariser).

**Snapshot.** `odds_belgian_20260712T063735Z` — combined
Betfair+Kalshi+Polymarket, 22 drivers. Markets used: **win** (n=22),
**top-3** (n=22), **top-5** (Kalshi, n=12), **top-10** (n=22); no H2H in
this snapshot. 78 (market, driver) observations in total.
Config: `fit_sims=4000`, `fit_tau=0.15`, `n_sims=100000`, `seed=20260703`.

> Single snapshot, single race — the numbers below are an existence
> proof and a cost measurement, not a season-wide average. The *size* of
> the coherence gain scales with how incoherent the markets are; these
> combined-exchange prices are already fairly coherent, so this is closer
> to a lower bound on the benefit than an upper one.

---

## Market coherence — the result

Mean absolute deviation of simulated marginals from the market, per
market and overall (**lower is better**):

| metric | rank-1 PL | Gaussian | Δ |
|---|--:|--:|--:|
| MAD, win market | **0.58** | 0.88 | +0.30 |
| MAD, top-3 | 1.14 | **0.96** | −0.18 |
| MAD, top-5 | 3.81 | **1.69** | **−2.12** |
| MAD, top-10 | 1.88 | **1.17** | −0.72 |
| **MAD, overall** | 1.60 | **1.11** | **−0.49** (−31%) |
| **RMSE, overall** | 2.55 | **1.60** | **−0.95** (−37%) |
| **max deviation** | 10.19 | **5.75** | **−4.44** |
| flagged >2pp (of 78) | 21 | **17** | −4 |

**Reading it.** The Gaussian model is more coherent on every metric that
aggregates the whole field: overall MAD falls 31%, RMSE 37%, and the
worst single miss nearly halves (10.2 → 5.8 pp). The gain concentrates
exactly where rank-1 PL is known to struggle — the **intermediate
top-5 market**, whose MAD more than halves (3.81 → 1.69 pp). This is the
incoherence of MATH.md §3.3 made concrete: a single strength that
reproduces win and top-10 cannot also hit top-5, and the second
parameter is what frees it.

**The honest cost inside the win column.** Win-market MAD gets *worse*
(0.58 → 0.88 pp). That is not a bug — it is the fit **redistributing**
error. Rank-1 PL fits the win market almost perfectly *because it has
nothing else it can do* with the one knob; the Gaussian fit trades a
fraction of a point on the (already tiny) win residual to remove two full
points on top-5. Total error drops; its allocation flattens across
markets. A sub-1pp win residual is far below the model's structural
noise, so this is a good trade.

### Where the spread went — the thesis, confirmed

The fitted dispersions are modest on this snapshot (σ ∈ [0.64, 1.31],
median 1.04) but point in exactly the direction DISPERSION.md predicted:
**favourites get the widest spreads, midfielders the tightest.**

| widest σ | | tightest σ | |
|---|--:|---|--:|
| RUS (win 21.1%) | **1.31** | NOR (win 2.9%) | 0.74 |
| ANT (win 34.0%) | **1.29** | LAW (win 0.8%) | 0.71 |
| HAM (win 14.3%) | **1.23** | GAS (win 0.9%) | **0.64** |
| LEC (win 11.3%) | 1.19 | | |
| VER (win 6.8%) | 1.17 | | |

The strong cars are exactly the "volatile front-runner" of
DISPERSION.md §1: the market gives them a high ceiling (win chance) *and*
a real floor (they occasionally finish down the order), which only extra
spread can represent. The midfielders are the "metronomic" case — reliably
in a narrow band, needing *less* spread than unit. Rank-1 PL cannot say
either thing; a σ per driver says both.

### Per-driver: where the misses were, and what σ did to them

The two markets that carry the incoherence are the intermediate **top-5**
and the **top-10** bulk (win and top-3 are fit well by both models). For
four representative drivers, market probability vs each model's
hard-simulated marginal (deviation from market in parentheses):

| driver | σ | top-5 mkt | top-5 PL | top-5 Gauss | top-10 mkt | top-10 PL | top-10 Gauss |
|---|--:|--:|--:|--:|--:|--:|--:|
| ANT (fav) | 1.29 | 82.7% | 88.1% (+5.4) | **84.4% (+1.7)** | 88.9% | 90.2% (+1.3) | 89.8% (+0.9) |
| RUS (fav) | 1.31 | 77.3% | 83.5% (+6.2) | **78.5% (+1.2)** | 88.9% | 90.1% (+1.2) | 89.2% (+0.3) |
| VER (volatile) | 1.17 | 65.1% | 62.7% (−2.4) | 62.6% (−2.5) | 83.6% | 89.9% (+6.3) | **87.3% (+3.7)** |
| NOR (metronome) | 0.74 | 58.7% | 48.5% (**−10.2**) | **56.5% (−2.2)** | 83.6% | 89.8% (+6.2) | 89.4% (+5.8) |

Three patterns, all predicted by DISPERSION.md:

- **Favourites (ANT, RUS).** Rank-1 PL, forced to match their large win
  probability with one knob, **overshoots** their top-5 by +5–6 pp — a
  driver strong enough to win *that* often is, under homogeneous noise,
  too reliable to fall out of the top 5. Widening σ (≈1.3) restores the
  downside and cuts the overshoot to +1–2 pp.
- **The metronome (NOR).** The field's **single worst miss** was NOR's
  top-5: the market implies 58.7% but PL simulates only 48.5% (**−10.2
  pp**) — a driver the market rates as a reliable top-5 finisher, which
  unit spread cannot reproduce for a low win probability. Tightening σ to
  0.74 lifts it to 56.5% (−2.2 pp), an 8-point repair.
- **What it doesn't fix (honest).** VER's top-5 miss is unchanged (both
  ≈ −2.5 pp), and NOR's top-10 stays +5.8 pp high. Two moments per driver
  cannot bend all four markets into line; the model removes the dominant
  (top-5) mode of the misfit, not every residual.


---

## Timing — the cost

| | rank-1 PL | Gaussian | ratio |
|---|--:|--:|--:|
| fit time | **84.6 s** | 377.0 s | **4.5×** |
| L-BFGS iterations | 42 | 71 | 1.7× |
| parameters | 22 (μ) | 44 (μ, log σ) | 2× |
| surrogate loss | 0.0528 | 0.0203 | — |

The Gaussian fit is ~4.5× slower. The cause is mechanical: the parameter
vector doubles (locations **and** log-dispersions), so each
finite-difference gradient costs ~2× the objective evaluations, and the
richer objective takes ~1.7× as many L-BFGS steps to converge. The
per-evaluation surrogate itself is barely more expensive (one extra
`exp` and a probit instead of a logistic).

**What is *not* slower.** Only the **fit** pays this cost. Everything
downstream is untouched:

- **Simulation** — identical engine (draw noise, scale, argsort); one
  extra multiply by σ. 100k races cost the same.
- **Validation, scoring, the Hungarian optimiser** — consume the
  `SimSet` and are byte-for-byte the same code path.

So on a full `fit → validate → optimise` run the extra minutes land
entirely in the one-off calibration, not the repeated Monte-Carlo work.

> **Absolute times are machine- and `fit_sims`-dependent.** 85 s for the
> PL fit is high because `fit_sims=4000` and this box is not fast; the
> ratio (~4.5×) is the portable number. If fit latency matters (odds
> moving near lock-in), `fit_sims=1500`–2000 cuts both proportionally,
> and analytic gradients (future work — the surrogate is smooth) would
> remove most of the finite-difference penalty that the doubled
> parameter count incurs.

---

## Verdict

| | rank-1 PL | Gaussian dispersion |
|---|---|---|
| market coherence | baseline | **−31% MAD, −37% RMSE, worst miss halved** |
| intermediate markets (top-5/6) | weakest point | **materially fixed** |
| fit time | fast (~85 s) | ~4.5× slower (~6 min) |
| sim / optimise time | — | **identical** |
| interpretability | one strength | strength **+ a spread you can read** |

The dispersion model does what it was designed to do: it removes most of
the rank-1 incoherence, and it does so in the physically meaningful way
(favourites carry more variance). The price is a slower one-off fit, paid
only at calibration. For an overnight or pre-weekend run the trade is
clearly worth it; only under tight last-minute odds movement does the
fit-latency argue for the PL model (or a reduced `fit_sims`).

**Caveats kept in view.** (1) One snapshot, one race — a broader sweep
(e.g. across the archived rounds) would firm up the average gain. (2) The
benefit is bounded by how incoherent the input markets are; combined
exchange prices are relatively coherent, so this understates the gain on
thinner single-source books. (3) Two parameters per driver still cannot
match four top-$k$ markets *exactly* — 17 of 78 observations remain past
2pp — so the output stays a well-calibrated prior, not ground truth
(MATH.md §8). (4) Where the "To be Classified" market is unpriced, σ and
the DNF prior are partially confounded (DISPERSION.md §7); this snapshot
had no classified market, so all attrition sat on the flat 0.10 prior.

---

## Reproducing this

```bash
cd optimiser
# fit both models on the latest snapshot and compare — set model.dist:
python main.py fit                 # config model.dist: gumbel  (rank-1 PL)
# edit config.yaml -> model.dist: gaussian, then:
python main.py fit                 # Gaussian dispersion
python main.py validate            # prints the market-vs-sim table per model
```

The head-to-head numbers above come from a harness that fits both on the
same de-vigged markets and tabulates the deviations
(`scratchpad/compare_models.py` / `per_driver.py` in the working
session). Switching models in the pipeline proper is the single
`model.dist` key in [config.yaml](config.yaml); everything else —
validate, optimise, report — picks up `sigma`/`model` from the saved fit
automatically.
