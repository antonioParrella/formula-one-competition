# CLAUDE.md — repo map

This repo holds **two related but separate projects**. Work in one; only touch the
other when the boundary below says to.

## 1. The competition (repo root)

The F1 tipping competition itself — fetch race results, score everyone's tips,
build the public leaderboard site.

- `src/` — scoring engine (`scorer.py`), leaderboard, result/schedule fetchers,
  tips parsing, site builder. This is the canonical source of truth for scoring.
- `data/` — raw + processed race data (results, schedule, tips, standings).
- `docs/` — the published GitHub Pages leaderboard site (`index.html`).
- `.github/workflows/score.yml` — CI that scores rounds. Only touches this project.
- `scripts/pipeline.py`, `config/`, `notebooks/`, `tests/` — competition tooling.
- Docs: `README.md`, `ARCHITECTURE.md`, `AGENTS.md`, `PLAN.md`.

This project is **self-contained** — it never imports from `optimiser/`.

## 2. The optimiser (`optimiser/`)

A personal Bayesian betting model that finds the optimal top-10 pick for the
competition above. Odds clients (Betfair / Kalshi / Polymarket), de-vig,
Plackett-Luce / Gaussian race model, Monte Carlo, ticket search.

- Has its own `CLAUDE.md`, `MATH.md`, `DISPERSION.md`, `COMPARISON.md`,
  `config.yaml`, `data/`, `tests/`, and `requirements.txt`. **Start at
  [optimiser/CLAUDE.md](optimiser/CLAUDE.md)** for anything in this subtree.
- Run its commands from inside `optimiser/` (`cd optimiser`).

## The boundary (one-way)

The optimiser **reuses** the competition's scoring code; the dependency runs in
one direction only:

```
optimiser/  ──imports──▶  src/   (scorer, leaderboard, DRIVER_MAP)
competition                       never imports optimiser/
```

`optimiser/_paths.py` puts `../src` on `sys.path` so the optimiser can
`from scorer import Scorer` and reuse the canonical comp rules
(`optimiser/comp_context.py`, `optimiser/scoring/rules.py`) instead of
copying them. If you change scoring in `src/scorer.py`, the optimiser follows
automatically — do **not** fork the rules into `optimiser/`.

> Naming note: this folder used to be called `f1-tipping/`, which was confusing
> because the *whole repo* is the tipping competition. It is now `optimiser/`.
