"""F1 Tipping Optimiser CLI.

Run from inside f1-tipping/:

    python main.py fetch                    # Betfair Exchange
    python main.py fetch --manual odds.csv  # paste-in CSV fallback
    python main.py fit                      # de-vig + calibrate + validate
    python main.py optimise                 # search ticket space
    python main.py all --manual odds.csv    # whole pipeline
"""

import _paths  # noqa: F401

import argparse
import json
from pathlib import Path

import yaml

from comp_context import resolve_context
from model.fit import build_dnf_probs, fit_strengths, load_fit, save_fit
from model.simulate import simulate_races
from model.validate import validate_fit
from odds.devig import devig_snapshot
from odds.snapshot import DATA_DIR, load_latest_snapshot, race_slug
from optimise.search import optimise
from race_utils import DRIVER_MAP          # existing code — single source of driver names
from scorer import Scorer                  # existing comp scorer — for DNF pick value
from scoring.rules import underdog_multipliers


def load_config(path: str) -> dict:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).resolve().parent / cfg_path
    return yaml.safe_load(cfg_path.read_text())


def cmd_fetch(cfg: dict, args: argparse.Namespace) -> None:
    if args.manual:
        from odds.manual_input import fetch_manual

        fetch_manual(args.manual, cfg["race"]["name"],
                     cfg["drivers"]["betfair_names"])
    else:
        from odds.betfair_client import fetch_betfair

        fetch_betfair(cfg)


def _devigged_markets(cfg: dict, args: argparse.Namespace) -> dict:
    snapshot = load_latest_snapshot(cfg["race"]["name"],
                                    allow_stale=args.allow_stale)
    markets = devig_snapshot(snapshot, cfg["devig"]["win_method"],
                             cfg["devig"].get("topn_method", "power"))
    print("Markets used: " + "; ".join(markets["markets_used"]))
    return markets


def cmd_fit(cfg: dict, args: argparse.Namespace) -> None:
    markets = _devigged_markets(cfg, args)
    model_cfg = cfg["model"]
    dnf_probs = build_dnf_probs(markets["drivers"], model_cfg["dnf"])

    fit = fit_strengths(
        markets,
        dnf_probs,
        fit_sims=int(model_cfg["fit_sims"]),
        tau=float(model_cfg["fit_tau"]),
        seed=int(model_cfg["seed"]),
    )
    save_fit(fit, cfg["race"]["name"])
    cmd_validate(cfg, args, fit=fit, markets=markets)


def cmd_validate(cfg: dict, args: argparse.Namespace,
                 fit: dict | None = None, markets: dict | None = None) -> None:
    fit = fit or load_fit(cfg["race"]["name"])
    markets = markets or _devigged_markets(cfg, args)
    sims = simulate_races(
        fit["theta"], fit["dnf_probs"],
        n_sims=int(cfg["model"]["n_sims"]),
        seed=int(cfg["model"]["seed"]) + 1,  # fresh draws, not the fit's
    )
    validate_fit(fit, markets, sims)


def cmd_optimise(cfg: dict, args: argparse.Namespace) -> None:
    fit = load_fit(cfg["race"]["name"])
    context = resolve_context(cfg)

    sims = simulate_races(
        fit["theta"], fit["dnf_probs"],
        n_sims=int(cfg["model"]["n_sims"]),
        seed=int(cfg["model"]["seed"]) + 1,
    )
    multipliers = underdog_multipliers(sims.drivers, context)

    opt_cfg = cfg["optimise"]
    print(f"\nOptimising over {sims.n_sims:,} simulated races, "
          f"{len(sims.drivers)} drivers, {opt_cfg['n_restarts']} restarts...")
    report = optimise(
        sims,
        multipliers,
        n_restarts=int(opt_cfg["n_restarts"]),
        max_iters=int(opt_cfg["max_iters"]),
        n_runner_ups=int(opt_cfg["n_runner_ups"]),
        assume_additive=bool(opt_cfg["assume_additive"]),
        seed=int(cfg["model"]["seed"]) + 2,
    )

    underdogs = {c for c in sims.drivers if context.multiplier(c) > 1}
    pcts = report.score_percentiles
    print(f"\nOptimal ticket (EV {report.best_ev:.2f} pts; "
          f"p10/p50/p90 = {pcts['p10']:.0f}/{pcts['p50']:.0f}/{pcts['p90']:.0f}):")
    for pos, code in enumerate(report.best_ticket, start=1):
        star = "  ⭐ underdog (2x)" if code in underdogs else ""
        print(f"  P{pos:<3} {code}  {DRIVER_MAP.get(code, '?'):<12}{star}")

    print("\nRunner-up tickets:")
    for ticket, ev in report.runner_ups:
        print(f"  EV {ev:6.2f}  {' '.join(ticket)}")

    dnf_ev = sorted(
        ((c, fit["dnf_probs"][c] * Scorer.POINTS_DNF) for c in sims.drivers),
        key=lambda x: x[1], reverse=True,
    )[:5]
    print("\nDNF pick value (P(dnf) x "
          f"{Scorer.POINTS_DNF} pts; 5-pick season budget):")
    for code, ev in dnf_ev:
        print(f"  {code}  {DRIVER_MAP.get(code, '?'):<12} EV {ev:.1f}")

    out = DATA_DIR / f"optimise_report_{race_slug(cfg['race']['name'])}.json"
    out.write_text(json.dumps({
        "race": cfg["race"],
        "best_ticket": report.best_ticket,
        "best_ev": report.best_ev,
        "score_percentiles": report.score_percentiles,
        "runner_ups": [{"ticket": t, "ev": ev} for t, ev in report.runner_ups],
        "underdog_top10": sorted(context.championship_top10),
        "n_sims": report.n_sims,
    }, indent=2))
    print(f"\nReport saved -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",
                        choices=["fetch", "fit", "validate", "optimise", "all"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--manual", metavar="CSV",
                        help="fetch odds from a driver,market,odds CSV "
                             "instead of Betfair")
    parser.add_argument("--allow-stale", action="store_true",
                        help="use a snapshot older than 24h")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.command in ("fetch", "all"):
        cmd_fetch(cfg, args)
    if args.command in ("fit", "all"):
        cmd_fit(cfg, args)
    if args.command == "validate":
        cmd_validate(cfg, args)
    if args.command in ("optimise", "all"):
        cmd_optimise(cfg, args)


if __name__ == "__main__":
    main()
