---
name: run-optimiser
description: Run the F1 tipping optimiser from its required working directory, using Betfair and other market sources or the manual CSV fallback when exchange access is unavailable. Use when preparing an optimal top-10 ticket, refreshing odds, selecting a saved snapshot, or validating the race model.
---

# Running the optimiser

The optimiser’s `main.py` depends on `_paths.py` to add `../src`, so run it from
inside `optimiser/`. The repository venv lives one directory above it.

```bash
# From the repository root, once:
# Windows:  .\venv\Scripts\activate
# macOS/Linux: source ./venv/bin/activate
cd optimiser
pip install -r requirements.txt
```

Before each race, set `race.name`, `race.year`, and `race.round` in `config.yaml`.
The race name must match the Betfair or Polymarket event text; the round controls
the underdog bonus. Refresh `underdogs.manual_top10` when using manual odds.

```bash
python main.py all --source all       # fetch, combine, fit, optimise and report
python main.py all --manual example_odds.csv  # geo-blocked or no-Betfair fallback
```

`all` prints the optimal top-10 ticket, expected points, p10/p50/p90,
runner-up tickets, and the best DNF picks. Stages can also run independently:

```bash
python main.py fetch --source all
python main.py combine
python main.py fit
python main.py optimise
python main.py report
python main.py snapshots
python main.py dnf
python main.py attrition
```

Useful flags: `--source betfair|kalshi|polymarket|all`, `--manual <csv>`,
`--snapshot <file>`, `--allow-stale`, `--refresh-dnf`, `--bayes`,
`--bayes-method mcmc|is`, and `--config <path>`. Outputs are archived under
`optimiser/data/`; validate historical performance separately with
`python backtest.py`. Run `python -m pytest tests/ -q` before relying on a code
change.
