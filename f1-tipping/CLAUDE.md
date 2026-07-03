# CLAUDE.md — F1 Tipping Optimiser

## Project Overview

A simulation engine that finds the optimal top-10 pick for an F1 tipping competition. Pipeline:

1. Pull betting odds from the Betfair Exchange API
2. De-vig odds into implied probabilities
3. Fit a probabilistic race model (Plackett-Luce with a DNF layer)
4. Monte Carlo simulate full finishing orders (~100k races)
5. Score candidate top-10 tickets under the comp's scoring rules
6. Search ticket space for the pick with the highest expected points

Key wrinkle: the comp has an **underdog bonus** based on driver position, so pick value depends on correlated tail scenarios (attrition promotes several underdogs at once). This is why we need a full joint distribution over finishing orders, not independent per-driver probabilities.

## Architecture

```
f1-tipping/
├── CLAUDE.md
├── config.yaml              # race name, market IDs, sim count, credentials path
├── odds/
│   ├── betfair_client.py    # auth + market data fetching
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
- Use `listMarketCatalogue` filtered on eventTypeId `27` (Motor Sport) to discover markets for the race weekend, then `listMarketBook` with `EX_BEST_OFFERS` for prices.
- Use **last traded price** where liquidity exists, else back/lay midpoint. Record both in the snapshot.
- Cache every fetch to `data/` with a timestamp before doing anything else. All downstream steps read from snapshots, never live — makes runs reproducible and avoids hammering the API.
- Rate limits are generous but don't poll; a snapshot per session is enough.

## De-vigging

- Exchange midpoints have low overround but still normalise.
- Win market: power method or simple proportional normalisation (configurable, default proportional).
- Top-N markets: probabilities must sum to N (e.g. Top 10 probs sum to 10). Normalise accordingly.
- H2H markets: two-outcome, normalise to 1.

## Model

- **Plackett-Luce**: each driver has strength θᵢ. Sample finishing order by repeatedly drawing without replacement, weight ∝ exp(θᵢ).
- **Fitting**: optimise θ (scipy, L-BFGS) to minimise squared error between simulated/analytic marginals and de-vigged market probs across all available markets: P(win), P(top3), P(top6), P(top10), and H2H win rates. Weight markets by liquidity if available.
- **DNF layer**: per-driver retirement probability (from DNF markets if available, else a season-average prior ~10% adjustable per driver). In each sim, drivers DNF independently and are removed to the back of the order before Plackett-Luce ranks the survivors. This creates the correlated attrition scenarios that matter for the underdog bonus.
- **Validation**: after fitting, `validate.py` must print a table comparing market probs vs simulated marginals for every market used. Flag any deviation > 2 percentage points.
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
- CLI: `python main.py fetch`, `python main.py fit`, `python main.py optimise`, or `python main.py all`.
- Fail loudly if a market is missing or stale (snapshot > 24h old) rather than silently fitting on partial data — print which markets were used.
- Tests: pytest; at minimum, test de-vig maths, Plackett-Luce sampling marginals against analytic values on a 3-driver toy case, and scoring on hand-computed examples.

## Gotchas

- Betfair market names vary by race weekend; don't hardcode market IDs, discover them via catalogue search each time and confirm in the fetch output.
- Sprint weekends have separate markets for sprint vs grand prix — filter carefully.
- Grid penalties and late driver changes can make odds shift sharply; always fetch fresh odds close to your comp's lock-in deadline.
- Betfair is geo-restricted in some countries; if unavailable, `manual_input.py` (paste odds as CSV: `driver,market,odds`) must remain a fully working alternative path through the entire pipeline.

---

## Integration with the parent repo

This directory lives inside the `formula-one-competition` repo — the comp this optimiser is trying to win. The comp's scoring engine and utilities already exist under `../src/` and are **imported, never edited or copied**:

- `src/scorer.py` — `Scorer` is the canonical comp scorer. `scoring/rules.py::score_ticket` delegates to `Scorer._score_main_race`, and the vectorised batch scorer reads its point values from the `Scorer` class constants (`POINTS_EXACT`, `POINTS_CLOSE`, `POINTS_TOP10`, `MULTIPLIER_UNDERDOG`). If the comp rules change in `Scorer`, this project follows automatically. Because the comp rules turned out to be known (exact/close/top10 + a 2× underdog multiplier), `score_ticket` is fully implemented; the `underdog_bonus` stub remains as the extension point should the bonus become position-scaled.
- `src/race_utils.py` — `DRIVER_MAP` is the single source of truth for 3-letter driver codes; Betfair runner names are matched to it (config mapping first, surname fallback second). `clean_race_name` normalises race names for filenames.
- `src/leaderboard.py` — `get_race_calendar` / `_fetch_openf1` are reused to pull the championship top-10 from OpenF1 (the underdog set: drivers *outside* it score double).

The comp's underdog rule (from `Scorer`): from round 2 onwards, any pick of a driver outside the championship top-10 before the race scores **double**. That set is resolved by `comp_context.py` (OpenF1, or a manual list in `config.yaml`).

Path bootstrap: `_paths.py` puts `../src` and this directory on `sys.path`; run everything from inside `f1-tipping/`:

```
cd f1-tipping
python main.py fetch --manual example_odds.csv   # or plain `fetch` for Betfair
python main.py fit
python main.py optimise
python main.py all --manual example_odds.csv
python -m pytest tests/ -q
```
