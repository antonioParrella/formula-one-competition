# CLAUDE.md — F1 Tipping Optimiser

## Project Overview

A simulation engine that finds the optimal top-10 pick for an F1 tipping competition. Pipeline:

1. Pull betting odds from the Betfair Exchange API, Kalshi, and Polymarket (combinable)
2. De-vig odds into implied probabilities
3. Fit a probabilistic race model (Plackett-Luce with a DNF layer)
4. Monte Carlo simulate full finishing orders (~100k races)
5. Score candidate top-10 tickets under the comp's scoring rules
6. Search ticket space for the pick with the highest expected points

Key wrinkle: the comp has an **underdog bonus** based on driver position, so pick value depends on correlated tail scenarios (attrition promotes several underdogs at once). This is why we need a full joint distribution over finishing orders, not independent per-driver probabilities.

## How to run

**The single script is `main.py`.** Run everything from inside `optimiser/`
(`_paths.py` bootstraps `../src` onto the path, so it must be the working dir).

```bash
# One-time setup (from the repo root). A venv already exists at ../venv:
#   Windows:  ..\venv\Scripts\activate
#   macOS/Linux:  source ../venv/bin/activate
cd optimiser
pip install -r requirements.txt
```

**1. Configure `config.yaml`** — before each race, set `race.name` (must match the
Betfair/Polymarket event text), `race.year`, and `race.round` (round drives the
underdog bonus). Refresh `underdogs.manual_top10` if using the manual fallback.

**2. Run the whole pipeline in one shot:**

```bash
python main.py all --source all          # fetch every source + combine, fit, optimise, report
# no Betfair access / geo-blocked? use the manual CSV path instead:
python main.py all --manual example_odds.csv
```

`all` prints the **optimal top-10 ticket** with its expected points and p10/p50/p90,
the runner-up tickets, and the best DNF picks.

**Or run the stages individually** (each reads the previous stage's saved output):

```bash
python main.py fetch --source all        # pull odds -> archives a snapshot in data/
python main.py combine                   # (optional) re-merge latest snapshot per source
python main.py fit                        # de-vig + calibrate the race model + validate
python main.py optimise                   # search ticket space -> prints + saves the pick
python main.py report                     # render the self-contained analysis HTML
python main.py snapshots                  # list the archived odds history
python main.py dnf                        # show per-driver season DNF priors (OpenF1)
python main.py attrition                  # show correlated-attrition calibration (shock λ + circuits)
```

**Common flags:** `--source betfair|kalshi|polymarket|all` (default `betfair`);
`--manual <csv>` (paste-in `driver,market,odds` fallback); `--snapshot <file>` to
fit/validate a specific archived snapshot; `--allow-stale` to use a snapshot older
than 24h; `--refresh-dnf` to re-fetch the season DNF rates from OpenF1 (else cached);
`--bayes` (+ `--bayes-method mcmc|is`) for the Bayesian path (slow — see
MATH.md §7); `--config <path>` for an alternate config.

**Outputs** (all under `data/`): `odds_<race>_<ts>.json` (snapshot),
`model_fit_<race>.json` (fit), `optimise_report_<race>.json` (the pick),
`analysis_<race>.html` (report). `report.py` can also run standalone.

**Historical validation** is a separate script:
`python backtest.py` (writes `data/backtest_<model>.json`).

**Tests:** `python -m pytest tests/ -q`.

## Architecture

```
optimiser/
├── CLAUDE.md
├── config.yaml              # race name, market IDs, sim count, credentials path
├── odds/
│   ├── betfair_client.py    # Betfair: auth + market data fetching
│   ├── kalshi_client.py     # Kalshi: public REST, no auth (win/top3/top5/top10)
│   ├── polymarket_client.py # Polymarket: public Gamma API, no auth (win/top3/h2h)
│   ├── predmarket.py        # shared prob-quote -> decimal-odds helpers + HTTP GET
│   ├── combine.py           # merge snapshots across sources (liquidity-weighted)
│   ├── manual_input.py      # paste-in CSV fallback (always keep working)
│   └── devig.py             # overround removal
├── model/
│   ├── fit.py               # Plackett-Luce strength calibration
│   ├── simulate.py          # Monte Carlo engine + DNF layer
│   └── validate.py          # check sim marginals vs market probs
├── scoring/
│   └── rules.py             # comp scoring incl. underdog bonus (user-editable)
├── optimise/
│   └── search.py            # ticket optimisation
├── data/                    # cached odds snapshots (JSON, timestamped)
├── report.py                # self-contained analysis HTML (no backtesting)
└── main.py                  # CLI entry point
```

## Betfair API Notes

- Use the **Betting API (JSON-RPC)** via `betfairlightweight` (pip). Requires an app key + session token; non-interactive login needs a self-signed cert registered with the account.
- Credentials live in `~/.betfair/credentials.yaml` (username, password, app_key, certs path). **Never commit credentials or certs. Never hardcode them.**
- Relevant F1 markets, in priority order:
  1. `Race Winner` (WIN market)
  2. `Points Finish` / `Top 10 Finish`
  3. `Top 6 Finish`
  4. `Podium Finish` / `Top 3`
  5. Driver head-to-head matchups (search event for `v` in market names)
  6. `Yes To be Classified` / `No To be Classified` (driver-runner markets) — de-vigged into per-driver DNF probabilities
- Use `listMarketCatalogue` filtered on eventTypeId `27` (Motor Sport) to discover markets for the race weekend, then `listMarketBook` with `EX_BEST_OFFERS` for prices.
- Use **last traded price** where liquidity exists, else back/lay midpoint. Record both in the snapshot.
- Cache every fetch to `data/` with a timestamp before doing anything else. All downstream steps read from snapshots, never live — makes runs reproducible and avoids hammering the API.
- Rate limits are generous but don't poll; a snapshot per session is enough.

## Kalshi & Polymarket Notes

- Both have **fully public, unauthenticated read APIs** — no account, key, or cert. Auth is only for trading, which this project never does. Clients use stdlib `urllib` (no new dependencies).
- Both quote binary markets as probabilities in (0, 1); clients convert to decimal odds (1/p) with back = 1/ask, lay = 1/bid (same ordering as Betfair), so snapshots are source-agnostic downstream.
- **Kalshi** (`https://api.elections.kalshi.com/trade-api/v2`): one event per race per series — `KXF1RACE` (winner), `KXF1RACEPODIUM` (top3), `KXF1TOP5`, `KXF1TOP10`. Event ticker = `<series>-<suffix>` (e.g. `KXF1RACE-BRIGP26`); the suffix is discovered from the win series by race-name match. Race events **open ~2 weeks pre-race** — earlier fetches fail loudly. No H2H or classified markets. `total_matched` is in contracts (a ~$1-scale proxy).
- **Polymarket** (`https://gamma-api.polymarket.com`): events per race — "<Race>: Driver Winner" / "Driver Podium Finish" / "Head-to-Head", discovered via `tag_slug=f1&closed=false` + title fragments. Winner books are live and tight weeks out; podium/H2H sit on **placeholder quotes (0.02/0.98)** until near the weekend. H2H outcome names are surnames; Sainz appears as "Jr." and is deliberately skipped when unmappable.
- **Spread guard**: two-sided quotes wider than `max_spread` (default 0.15 in prob space) are dropped entirely — an unseeded book's midpoint is garbage, not a price. One-sided quotes are kept only when a last trade anchors them: a lone never-traded resting order (e.g. a solitary 0.99 ask in an h2h) is an empty book wearing a price. A market left with fewer priced runners than its N is skipped by devig with a warning instead of dying.
- Placeholder runners ("Driver A", "another driver") are skipped silently; real reserve drivers (Perez, Bottas, Lindblad) are kept, same as Betfair's 22-runner fields.

## Combining Sources

- `odds/combine.py` merges the latest snapshot per source into a snapshot with `source: "combined"`. Per structural market: **de-vig each source separately first** (each has its own overround), then average per-driver probabilities weighted by `_market_weight(total_matched)`, and write back as decimal odds. Re-devigging the combined market downstream is a near no-op, so fit/validate/optimise are untouched.
- H2H markets are concatenated (tagged `[source]` in the market name) — same pair from two sources is two independent fit targets, each at its own liquidity weight. Classified sides merge per driver, most liquid source first.
- `python main.py fetch --source all` fetches every source in `sources.enabled` (failures warn and continue), then saves a combined snapshot if ≥ 2 succeeded — being newest, it is what `fit` picks up. `python main.py combine` re-combines the latest archived snapshot per source (respects `--allow-stale`).
- Don't combine snapshots fetched far apart in time — odds move with news; `combine` reports each input's age.

## De-vigging

- Exchange midpoints have low overround but still normalise.
- Win market: power method or simple proportional normalisation (configurable, default proportional).
- Top-N markets: probabilities must sum to N (e.g. Top 10 probs sum to 10). Normalise accordingly.
- H2H markets: two-outcome, normalise to 1.

## Model

- **Plackett-Luce**: each driver has strength θᵢ. Sample finishing order by repeatedly drawing without replacement, weight ∝ exp(θᵢ).
- **Fitting**: optimise θ (scipy, L-BFGS) to minimise squared error between simulated/analytic marginals and de-vigged market probs across all available markets: P(win), P(top3), P(top5) (Kalshi) or P(top6) (Betfair), P(top10), and H2H win rates. Weight markets by liquidity if available.
- **Dispersion model** (`model.dist: gaussian`, DISPERSION.md): the rank-1 flaw of PL is that one strength θᵢ can't match win/top-3/top-6/top-10 at once ("market incoherence", MATH.md §3.3). The Gaussian/Thurstonian option gives each driver a *center* μᵢ **and** a *spread* σᵢ — performance `Xᵢ = μᵢ + σᵢ·εᵢ`, εᵢ standard normal — fit via `fit_dispersion` (params `[μ; log σ]`, probit H2H). Default is `gumbel` (PL); switching is the single `model.dist` key, and validate/optimise/report pick up `sigma`/`model` from the saved fit automatically. On a real Belgian snapshot it cut overall market-deviation MAD 31% (RMSE 37%, worst miss halved) at ~4.5× the fit time — see COMPARISON.md.
- **DNF layer**: per-driver retirement probability, resolved highest-priority-first — explicit `dnf.per_driver` config override → 1 − P(classified) from the per-driver `To Be Classified?` Yes/No markets where priced → the driver's **season-to-date DNF rate** from OpenF1 (`dnf_prior.py`, see below) → flat `dnf.default_prob` (~10%, the fallback for round 1 or when OpenF1 is down). In each sim, drivers DNF and are removed to the back of the order before Plackett-Luce ranks the survivors — but **not independently**: a shared per-race shock correlates them (see Correlated attrition).
- **Correlated attrition + circuit conditioning** (`attrition.py`, `ATTRITION.md`, `dnf.attrition` config): observed races retire cars in *clusters* — the DNF-count variance is 1.4× and the "≥7 cars out" tail ~5× what independent Bernoulli predicts, and this heavy tail is exactly what promotes a cluster of underdogs into the points (the underdog bonus). So the sim draws one latent shock `Z ~ N(0,1)` per race that shifts every driver's DNF log-odds together, `d_i,race = logistic(b_i + λ·Z)`, with `b_i` set so the shock is **mean-preserving** (marginals, hence the fit and validate, unchanged). `λ` is calibrated to the historical DNF-count tail (2023-25). Separately, **circuit conditioning** scales the DNF *level* to the venue (`circuit_factor = circuit_rate/grid_rate`, e.g. Spa ≈ 0.66) by scaling `default_prob` before the priors are built. `python main.py attrition` prints the calibration; it's cached and recorded into the fit as `shock_lambda`, which `validate`/`optimise` apply. Disabled/OpenF1-down → λ=0, factor=1 (old behaviour).
- **Season DNF prior** (`dnf_prior.py`, `dnf.season` config): each driver's retirement rate this season = DNFs / starts over the completed race sessions, read from OpenF1 `session_result` (reusing `leaderboard._fetch_openf1`, the same source as `comp_context`). A DNS is excluded from both counts; a DSQ counts as a start but not a DNF. Raw rates over ~9 races are noisy, so they're Beta-smoothed toward the grid-wide average with a pseudo-count `prior_strength` (default 4): `rate = (dnf + k·grid) / (starts + k)`. The smoothed rates are then **re-centered** (`season.anchor_to_default`, default on) so their mean equals `default_prob`, keeping only the cross-driver *shape*. This matters because the DNF layer caps a driver's finish distribution at `P(top-k) ≤ 1 − P(DNF)`: the unanchored season level here is ~20% (≈2× `default_prob`), which drops strong drivers' points-finish ceiling below their market prices — the strength fit can't climb past it and market alignment collapses. Re-centering keeps expected attrition where the flat prior put it while still telling fragile drivers from reliable ones. Counts are cached under `data/dnf_season_<year>_r<round>.json` (keyed by round, so completed results never restale); `python main.py dnf` prints the table and `--refresh-dnf` forces a re-fetch. A fetch failure warns and degrades to `default_prob` rather than killing the fit. The Bayesian path (`--bayes`) uses each driver's season rate as the prior *centre* for its sampled DNF eta.
- **Validation**: after fitting, `validate.py` must print a table comparing market probs vs simulated marginals for every market used. The deviation flag is **liquidity-adjusted** — a 2pp gap is only meaningful when the market is liquid enough to price to that precision, so the threshold is 2pp at/above reference liquidity (and when liquidity is unknown, e.g. the manual CSV path) and widens for thin books via `_flag_threshold` (weight 0.5 → 4pp, 0.2 → 10pp), keyed off the same `_market_weight` liquidity proxy used in de-vig. The Bayesian path is already liquidity-aware (its predictive band scales as 1/√weight).
- **Bayesian path** (`fit/validate/optimise --bayes`, MATH.md §7): de-vigged market probs become noisy observations of latent θ; the posterior P(θ | odds) is sampled by `model/bayes.py` with two engines — `bayes.method: mcmc` (ensemble stretch move in `model/mcmc.py`, robust, ~10 min) or `is` (Laplace-t importance sampling in `model/importance.py`, ~2-4 min, guarded by Pareto k̂ ≤ 0.7). A global noise scale σ_obs is inferred and measures rank-1 misfit; DNF probs of classified-priced drivers are sampled jointly. Posterior-predictive races are packaged as a normal `SimSet` (draws × races), so scoring/optimise are unchanged. Observations live on the **probability scale** (`bayes.obs_scale`) — the logit scale was measured to fail (pinned longshot prices dominate). Posterior saved to `data/model_posterior_<race>.npz`.
- Default simulation count: 100,000 races. Vectorise with numpy; a full run should take seconds, not minutes.

## Scoring Rules

- `scoring/rules.py` exposes `score_ticket(ticket: list[str], result: list[str]) -> float`.
- The exact rules are **not yet finalised** — keep this module isolated and dependency-free so it can be rewritten without touching the model.
- Known so far: there is an **underdog bonus scaled by driver position**. Leave a clearly marked `underdog_bonus(driver, predicted_pos, actual_pos)` stub with a TODO.
- Scoring must be vectorisable or at least fast: it runs (tickets × simulations) times inside the optimiser.

## Optimisation

- Ticket space is 20P10 ≈ 6.7 × 10¹¹ — no brute force.
- Approach: start from the probability-greedy ticket (sort by expected finishing position), then local search: swap-within-ticket and substitute-from-outside moves, accept improvements in mean score over the sim set. Random restarts (default 20) to escape local optima.
- Use **common random numbers**: evaluate all candidate tickets against the same fixed simulation set so comparisons are low-variance.
- Report: optimal ticket, expected points, score distribution (p10/p50/p90), and the top 5 runner-up tickets with their EVs.
- Future extension (not now): optimise for P(beating opponents) rather than EV — these differ when trailing on the season ladder (trailing → pick higher variance).

## Conventions

- Python 3.11+, type hints throughout, `numpy` + `scipy` + `betfairlightweight` + `pyyaml`.
- Drivers identified by 3-letter code (VER, NOR, PIA, ...). Single source of truth mapping in `config.yaml`: Betfair runner name → code.
- All randomness seeded via config for reproducibility.
- CLI: `python main.py fetch [--source betfair|kalshi|polymarket|all]`, `python main.py combine`, `python main.py fit`, `python main.py optimise`, `python main.py report` (or standalone `python report.py`), `python main.py dnf` (season DNF priors), `python main.py attrition` (correlated-attrition calibration), or `python main.py all`.
- `report.py` renders `data/analysis_<race>.html` — a self-contained page (no external assets, light/dark aware) built from the latest snapshot + fit + optimise report for the configured race. Current-race analysis only; backtesting stays out of the HTML (backtest.py prints its own text output). It never touches `docs/index.html`, which belongs to the comp leaderboard site.
- Fail loudly if a market is missing or stale (snapshot > 24h old) rather than silently fitting on partial data — print which markets were used.
- Tests: pytest; at minimum, test de-vig maths, Plackett-Luce sampling marginals against analytic values on a 3-driver toy case, and scoring on hand-computed examples.

## Gotchas

- Betfair market names vary by race weekend; don't hardcode market IDs, discover them via catalogue search each time and confirm in the fetch output.
- Sprint weekends have separate markets for sprint vs grand prix — filter carefully.
- Grid penalties and late driver changes can make odds shift sharply; always fetch fresh odds close to your comp's lock-in deadline.
- Events vanish from the Betfair catalogue shortly after the race — there is no retro fetch. Anything you want archived (especially the thin `To be Classified` markets) must be snapshotted pre-race.
- Betfair is geo-restricted in some countries; if unavailable, `manual_input.py` (paste odds as CSV: `driver,market,odds`) must remain a fully working alternative path through the entire pipeline.

---

## Integration with the parent repo

This directory lives inside the `formula-one-competition` repo — the comp this optimiser is trying to win. The comp's scoring engine and utilities already exist under `../src/` and are **imported, never edited or copied**:

- `src/scorer.py` — `Scorer` is the canonical comp scorer. `scoring/rules.py::score_ticket` delegates to `Scorer._score_main_race`, and the vectorised batch scorer reads its point values from the `Scorer` class constants (`POINTS_EXACT`, `POINTS_CLOSE`, `POINTS_TOP10`, `MULTIPLIER_UNDERDOG`). If the comp rules change in `Scorer`, this project follows automatically. Because the comp rules turned out to be known (exact/close/top10 + a 2× underdog multiplier), `score_ticket` is fully implemented; the `underdog_bonus` stub remains as the extension point should the bonus become position-scaled.
- `src/race_utils.py` — `DRIVER_MAP` is the single source of truth for 3-letter driver codes; Betfair runner names are matched to it (config mapping first, surname fallback second). `clean_race_name` normalises race names for filenames.
- `src/leaderboard.py` — `get_race_calendar` / `_fetch_openf1` are reused to pull the championship top-10 from OpenF1 (the underdog set: drivers *outside* it score double), and by `dnf_prior.py` to pull each driver's season-to-date DNF rate from `session_result` for the DNF prior.

The comp's underdog rule (from `Scorer`): from round 2 onwards, any pick of a driver outside the championship top-10 before the race scores **double**. That set is resolved by `comp_context.py` (OpenF1, or a manual list in `config.yaml`).

Path bootstrap: `_paths.py` puts `../src` and this directory on `sys.path`; run everything from inside `optimiser/`:

```
cd optimiser
python main.py fetch --manual example_odds.csv   # or plain `fetch` for Betfair
python main.py fit
python main.py optimise
python main.py all --manual example_odds.csv
python -m pytest tests/ -q
```
