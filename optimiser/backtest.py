"""Backtest the optimiser against real past rounds.

For each completed round the parent repo stores the actual result
(``data/raw/results/``) and every player's score (``data/processed/``).
This harness runs the full simulate -> optimise -> score pipeline for a
round and compares the optimiser's ticket to what the human field
actually scored, using the comp's own ``Scorer`` (via
``scoring.rules.score_ticket``) so the points are directly comparable.

Model source, per round:

- ``odds``  — a real odds snapshot in ``data/`` (the true backtest).
  Needs historical pre-race odds, which the live Betfair API cannot
  retrieve; drop per-round CSVs through ``odds/manual_input.py`` first.
- ``standings`` — strengths derived from the pre-race championship
  order (stored in the result file, genuinely known before the race).
  No odds required and no look-ahead: it answers "given the same public
  info the players had, does the engine beat them?". Real odds would
  only add pace/quali signal on top, so this is a lower bound.

Usage:
    python backtest.py                 # standings baseline, rounds 2-7
    python backtest.py --rounds 3-7
    python backtest.py --model odds    # once historical snapshots exist
"""

import _paths  # noqa: F401

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from model.fit import build_dnf_probs, fit_strengths, load_fit
from model.simulate import simulate_races
from odds.devig import devig_snapshot
from optimise.search import optimise
from race_utils import DRIVER_MAP
from scoring.rules import ScoringContext, score_ticket, underdog_multipliers

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "data" / "raw" / "results"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

N_SIMS = 100_000
BASE_SEED = 4242
STANDINGS_DECAY = 0.72   # win-weight ratio between adjacent championship ranks


def _load_round(dir_: Path, round_num: int, suffix: str) -> dict | None:
    matches = list(dir_.glob(f"r{round_num:02d}_*_{suffix}.json"))
    return json.loads(matches[0].read_text()) if matches else None


def standings_model(entrants: list[str], champ_order: list[str]) -> tuple[dict, dict]:
    """Baseline strengths from championship rank (pre-race, no look-ahead).

    Ranked drivers get geometrically decaying win-weight; entrants
    outside the championship top-10 sit just below the field. DNF risk
    is the flat season prior — using the round's actual DNFs would be
    look-ahead.
    """
    rank = {code: i for i, code in enumerate(champ_order)}
    outside_weight = STANDINGS_DECAY ** len(champ_order) * 0.4
    theta = {
        code: float(np.log(STANDINGS_DECAY ** rank[code] if code in rank
                           else outside_weight))
        for code in entrants
    }
    dnf = build_dnf_probs(entrants, {"default_prob": 0.10})
    return theta, dnf


def _model_from_odds(race_name: str, round_num: int) -> tuple[dict, dict, list[str]]:
    """Fit strengths from a stored odds snapshot for a past round."""
    from odds.snapshot import load_latest_snapshot

    snapshot = load_latest_snapshot(race_name, allow_stale=True)
    markets = devig_snapshot(snapshot)
    dnf = build_dnf_probs(markets["drivers"], {"default_prob": 0.10})
    fit = fit_strengths(markets, dnf, seed=BASE_SEED + round_num)
    return fit["theta"], fit["dnf_probs"], markets["drivers"]


def backtest_round(round_num: int, model: str) -> dict | None:
    result = _load_round(RESULTS_DIR, round_num, "result")
    scored = _load_round(PROCESSED_DIR, round_num, "scored")
    if result is None or scored is None:
        return None

    top10 = result["top10"]
    dnfs = result.get("dnfs") or []
    champ = result.get("championship_top10_before_race") or []
    context = ScoringContext(
        round_num=round_num,
        championship_top10=frozenset(champ) if round_num > 1 else frozenset(),
    )

    if model == "odds":
        theta, dnf_probs, _ = _model_from_odds(result["race_name"], round_num)
    else:
        # Entrant list = everyone known to be involved that weekend
        # (grid + classified + retirements) — all pre-race knowledge.
        entrants = list(dict.fromkeys([*champ, *top10, *dnfs]))
        theta, dnf_probs = standings_model(entrants, champ)

    sims = simulate_races(theta, dnf_probs, n_sims=N_SIMS, seed=BASE_SEED + round_num)
    multipliers = underdog_multipliers(sims.drivers, context)
    report = optimise(sims, multipliers, n_restarts=12, max_iters=120,
                      seed=BASE_SEED + 100 + round_num, verbose=False)

    opt_ticket = report.best_ticket
    opt_score = score_ticket(opt_ticket, top10, context)

    # Reference: the naive "just copy the standings order" ticket.
    naive_ticket = champ[:10]
    naive_score = (score_ticket(naive_ticket, top10, context)
                   if len(naive_ticket) == 10 else None)

    human_scores = scored["scores"]
    values = sorted(human_scores.values())
    winner_name = max(human_scores, key=human_scores.get)
    beat = sum(1 for v in values if opt_score > v)

    return {
        "round": round_num,
        "race": result["race_name"],
        "optimal_ticket": opt_ticket,
        "optimal_score": opt_score,
        "naive_standings_score": naive_score,
        "human_min": values[0],
        "human_median": statistics.median(values),
        "human_max": values[-1],
        "human_winner": winner_name,
        "beat_count": beat,
        "field_size": len(values),
    }


def run(rounds: list[int], model: str) -> None:
    label = ("REAL ODDS" if model == "odds"
             else "STANDINGS BASELINE (no odds; lower bound on the approach)")
    print(f"\nBacktest — model: {label}")
    print(f"Scoring via the comp's own Scorer; {N_SIMS:,} sims/round.\n")

    header = (f"{'Rd':>2} {'Race':<12} {'Optimal':>8} {'Naive':>6} "
              f"{'H-min':>6} {'H-med':>6} {'H-win':>6} {'Beat':>7}")
    print(header)
    print("─" * len(header))

    rows = []
    tot_opt = tot_naive = tot_winner = 0
    tot_beat = tot_field = 0
    for rn in rounds:
        row = backtest_round(rn, model)
        if row is None:
            continue
        rows.append(row)
        naive = row["naive_standings_score"]
        print(f"{row['round']:>2} {row['race']:<12} {row['optimal_score']:>8.0f} "
              f"{('-' if naive is None else f'{naive:.0f}'):>6} "
              f"{row['human_min']:>6.0f} {row['human_median']:>6.1f} "
              f"{row['human_max']:>6.0f} {row['beat_count']:>3}/{row['field_size']:<3}")
        tot_opt += row["optimal_score"]
        tot_naive += naive or 0
        tot_winner += row["human_max"]
        tot_beat += row["beat_count"]
        tot_field += row["field_size"]

    if not rows:
        print("No rounds with both a result and a scored file were found.")
        return

    print("─" * len(header))
    print(f"{'':>2} {'TOTAL':<12} {tot_opt:>8.0f} {tot_naive:>6.0f} "
          f"{'':>6} {'':>6} {tot_winner:>6.0f} {tot_beat:>3}/{tot_field:<3}")

    n = len(rows)
    print(f"\nOver {n} rounds:")
    print(f"  Optimiser total:      {tot_opt:.0f} pts "
          f"({tot_opt / n:.1f}/round)")
    print(f"  Naive standings:      {tot_naive:.0f} pts")
    print(f"  Actual round winners: {tot_winner:.0f} pts "
          f"(best human each week — a hard bar, different player each time)")
    print(f"  Optimiser beat {tot_beat}/{tot_field} human round-entries "
          f"({100 * tot_beat / tot_field:.0f}%).")
    rounds_won = sum(1 for r in rows if r["optimal_score"] >= r["human_max"])
    print(f"  Optimiser would have won {rounds_won}/{n} rounds outright.")

    out = REPO_ROOT / "optimiser" / "data" / f"backtest_{model}.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nDetail saved -> {out}")
    print("\nNote: 'standings baseline' uses only pre-race championship order — "
          "the same info the players had, no betting odds and no look-ahead. "
          "Supplying historical pre-race odds (--model odds) would add "
          "pace/quali signal the standings miss.")


def _parse_rounds(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", default="2-7",
                        help="e.g. 2-7 or 3,5,7 (round 1 has no underdog rule)")
    parser.add_argument("--model", choices=["standings", "odds"],
                        default="standings")
    args = parser.parse_args()
    run(_parse_rounds(args.rounds), args.model)


if __name__ == "__main__":
    main()
