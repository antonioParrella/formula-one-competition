"""F1 Tipping Optimiser CLI.

Run from inside optimiser/:

    python main.py fetch                       # Betfair Exchange
    python main.py fetch --source kalshi       # Kalshi (public API, no auth)
    python main.py fetch --source polymarket   # Polymarket (public API, no auth)
    python main.py fetch --source all          # every enabled source + combined
    python main.py fetch --manual odds.csv     # paste-in CSV fallback
    python main.py combine                     # merge latest snapshot per source
    python main.py fit                         # de-vig + calibrate + validate
    python main.py optimise                    # search ticket space
    python main.py report                      # self-contained analysis HTML
    python main.py all --source all            # whole pipeline
"""

import _paths  # noqa: F401

import argparse
import json
from pathlib import Path

import yaml

from attrition import circuit_factor, load_calibration, race_circuit
from comp_context import resolve_context
from dnf_prior import season_dnf_rates
from model.fit import (
    build_dnf_probs,
    fit_dispersion,
    fit_strengths,
    load_fit,
    save_fit,
)
from model.simulate import simulate_races
from model.validate import validate_fit
from odds.devig import devig_snapshot
from odds.snapshot import (
    DATA_DIR,
    list_snapshots,
    load_latest_snapshot,
    load_snapshot,
    race_slug,
    save_snapshot,
)
from optimise.search import optimise
from race_utils import DRIVER_MAP          # existing code — single source of driver names
from scorer import Scorer                  # existing comp scorer — for DNF pick value
from scoring.rules import underdog_multipliers


def load_config(path: str) -> dict:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).resolve().parent / cfg_path
    return yaml.safe_load(cfg_path.read_text())


def _fetch_source(source: str, cfg: dict) -> Path:
    if source == "betfair":
        from odds.betfair_client import fetch_betfair

        return fetch_betfair(cfg)
    if source == "kalshi":
        from odds.kalshi_client import fetch_kalshi

        return fetch_kalshi(cfg)
    if source == "polymarket":
        from odds.polymarket_client import fetch_polymarket

        return fetch_polymarket(cfg)
    raise ValueError(f"Unknown odds source {source!r}")


def cmd_fetch(cfg: dict, args: argparse.Namespace) -> None:
    if args.manual:
        from odds.manual_input import fetch_manual

        fetch_manual(args.manual, cfg["race"]["name"],
                     cfg["drivers"]["betfair_names"])
        return
    if args.source != "all":
        _fetch_source(args.source, cfg)
        return

    enabled = (cfg.get("sources") or {}).get(
        "enabled", ["betfair", "kalshi", "polymarket"])
    paths: list[Path] = []
    for source in enabled:
        print(f"\n=== fetch: {source} ===")
        try:
            paths.append(_fetch_source(source, cfg))
        except Exception as exc:  # one dead source must not sink the others
            print(f"WARNING: {source} fetch failed: {exc}")
    if not paths:
        raise RuntimeError("All odds sources failed — try --manual <csv>.")
    if len(paths) == 1:
        print("\nOnly one source fetched — nothing to combine.")
        return

    from odds.combine import combine_snapshots

    print(f"\n=== combine: {len(paths)} sources ===")
    snapshots = [load_snapshot(p) for p in paths]
    save_snapshot(combine_snapshots(snapshots, cfg["devig"]["win_method"],
                                    cfg["devig"].get("topn_method", "power")))


def cmd_combine(cfg: dict, args: argparse.Namespace) -> None:
    """Merge the latest archived snapshot of each source into one."""
    from odds.combine import combine_latest

    combine_latest(cfg["race"]["name"], cfg["devig"]["win_method"],
                   cfg["devig"].get("topn_method", "power"),
                   allow_stale=args.allow_stale)


def _devigged_markets(cfg: dict, args: argparse.Namespace) -> dict:
    if getattr(args, "snapshot", None):
        snapshot = load_snapshot(args.snapshot, allow_stale=True)
    else:
        snapshot = load_latest_snapshot(cfg["race"]["name"],
                                        allow_stale=args.allow_stale)
    markets = devig_snapshot(snapshot, cfg["devig"]["win_method"],
                             cfg["devig"].get("topn_method", "power"))
    print("Markets used: " + "; ".join(markets["markets_used"]))
    return markets


def cmd_snapshots(cfg: dict, args: argparse.Namespace) -> None:
    """List the archived odds snapshots (the odds history)."""
    snaps = list_snapshots()
    if not snaps:
        print("No odds snapshots archived yet. Run `python main.py fetch`.")
        return
    print(f"{len(snaps)} archived odds snapshot(s) in {DATA_DIR}:")
    for p in snaps:
        snap = json.loads(p.read_text())
        markets = snap.get("markets", {})
        structural = [k for k in ("win", "top3", "top5", "top6", "top10")
                      if k in markets]
        cls = markets.get("classified") or {}
        n_cls = len({c for entry in cls.values() for c in entry.get("runners", {})})
        print(f"  {p.name:<44} {snap.get('source', '?'):<7} "
              f"{snap.get('fetched_at', '?')[:19]}  "
              f"markets={structural} h2h={len(markets.get('h2h', []))} "
              f"classified={n_cls}")


def cmd_dnf(cfg: dict, args: argparse.Namespace) -> None:
    """Show the per-driver season DNF priors pulled from OpenF1."""
    from dnf_prior import _load_or_fetch_counts, recenter_rates, smooth_rates

    dnf_cfg = cfg["model"]["dnf"]
    season_cfg = dnf_cfg.get("season") or {}
    year = int(cfg["race"]["year"])
    rnd = int(cfg["race"]["round"])
    default = float(dnf_cfg.get("default_prob", 0.10))
    if not season_cfg.get("enabled", True):
        print("dnf.season.enabled is false — all drivers use the flat "
              f"{default:.0%} prior (or their per_driver/market value).")
        return
    if rnd <= 1:
        print(f"Round {rnd}: no completed races this season yet — every "
              f"driver falls back to the flat {default:.0%} prior.")
        return
    prior_strength = float(season_cfg.get("prior_strength", 4.0))
    include_sprints = bool(season_cfg.get("include_sprints", False))
    anchor = bool(season_cfg.get("anchor_to_default", True))
    dnf_counts, start_counts = _load_or_fetch_counts(
        year, rnd, include_sprints, getattr(args, "refresh_dnf", False))
    smoothed, grid = smooth_rates(dnf_counts, start_counts, prior_strength)
    if not smoothed:
        print("OpenF1 returned no completed-race results for the season.")
        return
    rates = recenter_rates(smoothed, default) if anchor else smoothed
    total_starts = sum(start_counts.values())
    print(f"\nSeason DNF priors — {year}, through round {rnd - 1} "
          f"({'races+sprints' if include_sprints else 'races only'})")
    line = (f"grid avg {grid:.1%} over {total_starts} starts | "
            f"shrink k={prior_strength:g}")
    if anchor:
        line += f" | re-centered to mean {default:.0%}"
    print(line + f" | flat fallback {default:.0%}\n")
    print(f"  {'DRV':<4} {'DNF/STA':>8}  {'raw':>5}  {'shrunk':>7}  {'prior':>6}")
    for c in sorted(rates, key=lambda c: -rates[c]):
        d, s = dnf_counts.get(c, 0), start_counts[c]
        print(f"  {c:<4} {f'{d}/{s}':>8}  {d / s:>5.0%}  {smoothed[c]:>7.1%}  "
              f"{rates[c]:>6.1%}")


def cmd_attrition(cfg: dict, args: argparse.Namespace) -> None:
    """Show the correlated-attrition calibration (shock λ + circuit rates)."""
    att_cfg = (cfg["model"]["dnf"].get("attrition") or {})
    bundle = load_calibration(att_cfg, refresh=getattr(args, "refresh_dnf", False))
    if not bundle:
        print("Attrition layer disabled or OpenF1 unavailable.")
        return
    print(f"\nCorrelated attrition — calibrated on {bundle['n_races']} races "
          f"{bundle['years']}")
    print(f"grid DNF rate {bundle['grid_rate']:.1%} | shared-shock λ = "
          f"{bundle['lambda']:.2f}\n")
    print(f"  {'DNF/race tail':>16}  {'observed':>9} {'fitted':>7}")
    obs, fit = bundle["tail_observed"], bundle["tail_fitted"]
    for t in sorted(obs, key=int):
        print(f"  {'P(K>='+t+')':>16}  {obs[t]:>9.1%} {fit[t]:>7.1%}")
    circuit = race_circuit(int(cfg["race"]["year"]), int(cfg["race"]["round"]))
    fac = circuit_factor(bundle, circuit)
    print(f"\nthis race: '{circuit}' factor {fac:.2f} -> "
          f"level {bundle['grid_rate'] * fac:.1%}")
    rates = bundle["circuit_rates"]
    hi = sorted(rates.items(), key=lambda t: -t[1])
    print("circuits  high:", ", ".join(f"{c} {r:.0%}" for c, r in hi[:4]))
    print("          low :", ", ".join(f"{c} {r:.0%}" for c, r in hi[-4:]))


def _bayes_cfg(cfg: dict, args: argparse.Namespace) -> dict:
    bayes_cfg = dict(cfg.get("bayes") or {})
    if getattr(args, "bayes_method", None):
        bayes_cfg["method"] = args.bayes_method
    return bayes_cfg


def _attrition(cfg: dict, args: argparse.Namespace) -> tuple[float, float]:
    """(circuit DNF-level factor, shared-shock loading λ) for this race.

    Circuit conditioning scales the DNF *level* to the venue (Spa low,
    Melbourne high); λ drives the correlated "many cars out" tail. Both
    come from the cached OpenF1 calibration (``attrition.py``); a disabled
    or unavailable calibration yields (1.0, 0.0) — no change.
    """
    att_cfg = (cfg["model"]["dnf"].get("attrition") or {})
    bundle = load_calibration(att_cfg, refresh=getattr(args, "refresh_dnf", False))
    if not bundle:
        return 1.0, 0.0
    lam = float(bundle["lambda"]) if att_cfg.get("shock", True) else 0.0
    factor = 1.0
    if att_cfg.get("circuit", True):
        circuit = race_circuit(int(cfg["race"]["year"]), int(cfg["race"]["round"]))
        factor = circuit_factor(bundle, circuit)
        print(f"Attrition: circuit '{circuit}' DNF factor {factor:.2f} "
              f"(grid {bundle['grid_rate']:.1%}), shock λ={lam:.2f}")
    return factor, lam


def cmd_fit(cfg: dict, args: argparse.Namespace) -> None:
    markets = _devigged_markets(cfg, args)
    model_cfg = cfg["model"]
    factor, shock_lambda = _attrition(cfg, args)
    # Circuit conditioning: scale the DNF *level* (the re-centring anchor and
    # the flat fallback) to the venue; per-driver and market DNFs are untouched.
    dnf_cfg = dict(model_cfg["dnf"])
    dnf_cfg["default_prob"] = float(dnf_cfg.get("default_prob", 0.10)) * factor
    season_dnf = season_dnf_rates(
        dnf_cfg, int(cfg["race"]["year"]), int(cfg["race"]["round"]),
        refresh=getattr(args, "refresh_dnf", False))
    dnf_probs = build_dnf_probs(markets["drivers"], dnf_cfg,
                                markets.get("dnf"), season_dnf)

    dist = str(model_cfg.get("dist", "gumbel"))
    fit_fn = fit_dispersion if dist == "gaussian" else fit_strengths
    fit = fit_fn(
        markets,
        dnf_probs,
        fit_sims=int(model_cfg["fit_sims"]),
        tau=float(model_cfg["fit_tau"]),
        seed=int(model_cfg["seed"]),
    )
    fit["shock_lambda"] = shock_lambda   # correlated attrition, used by the sims
    save_fit(fit, cfg["race"]["name"])

    posterior = None
    if getattr(args, "bayes", False):
        from model.bayes import fit_posterior, save_posterior

        posterior = fit_posterior(markets, dnf_probs, model_cfg["dnf"], fit,
                                  _bayes_cfg(cfg, args), season_dnf=season_dnf)
        save_posterior(posterior, cfg["race"]["name"])
    cmd_validate(cfg, args, fit=fit, markets=markets, posterior=posterior)


def cmd_validate(cfg: dict, args: argparse.Namespace,
                 fit: dict | None = None, markets: dict | None = None,
                 posterior: dict | None = None) -> None:
    markets = markets or _devigged_markets(cfg, args)
    if getattr(args, "bayes", False):
        from model.bayes import (
            _cfg as bayes_defaults,
            load_posterior,
            posterior_marginal_draws,
            simulate_posterior,
        )
        from model.validate import validate_posterior

        posterior = posterior or load_posterior(cfg["race"]["name"])
        bayes_cfg = bayes_defaults(_bayes_cfg(cfg, args))
        seed = int(cfg["model"]["seed"])
        sims = simulate_posterior(posterior, int(cfg["model"]["n_sims"]),
                                  seed=seed + 1)
        marginal_draws, draw_indices = posterior_marginal_draws(
            posterior,
            ks=sorted(markets["topk"]),
            n_draws=int(bayes_cfg["ci_draws"]),
            sims_per_draw=int(bayes_cfg["ci_sims_per_draw"]),
            seed=seed + 3,
        )
        validate_posterior(posterior, markets, sims, marginal_draws,
                           draw_indices, ci=float(bayes_cfg["ci"]))
        return

    fit = fit or load_fit(cfg["race"]["name"])
    sims = simulate_races(
        fit["theta"], fit["dnf_probs"],
        n_sims=int(cfg["model"]["n_sims"]),
        seed=int(cfg["model"]["seed"]) + 1,  # fresh draws, not the fit's
        sigma_by_code=fit.get("sigma"),
        dist=fit.get("model", "gumbel"),
        shock_lambda=float(fit.get("shock_lambda", 0.0)),
    )
    validate_fit(fit, markets, sims)


def cmd_optimise(cfg: dict, args: argparse.Namespace) -> None:
    context = resolve_context(cfg)

    if getattr(args, "bayes", False):
        from model.bayes import load_posterior, simulate_posterior

        posterior = load_posterior(cfg["race"]["name"])
        sims = simulate_posterior(posterior, int(cfg["model"]["n_sims"]),
                                  seed=int(cfg["model"]["seed"]) + 1)
        dnf_probs = dict(zip(posterior["drivers"],
                             posterior["dnf"].mean(axis=0)))
        report_suffix = "_bayes"
    else:
        fit = load_fit(cfg["race"]["name"])
        sims = simulate_races(
            fit["theta"], fit["dnf_probs"],
            n_sims=int(cfg["model"]["n_sims"]),
            seed=int(cfg["model"]["seed"]) + 1,
            sigma_by_code=fit.get("sigma"),
            dist=fit.get("model", "gumbel"),
            shock_lambda=float(fit.get("shock_lambda", 0.0)),
        )
        dnf_probs = fit["dnf_probs"]
        report_suffix = ""
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
        ((c, dnf_probs[c] * Scorer.POINTS_DNF) for c in sims.drivers),
        key=lambda x: x[1], reverse=True,
    )[:5]
    print("\nDNF pick value (P(dnf) x "
          f"{Scorer.POINTS_DNF} pts; 5-pick season budget):")
    for code, ev in dnf_ev:
        print(f"  {code}  {DRIVER_MAP.get(code, '?'):<12} EV {ev:.1f}")

    out = (DATA_DIR /
           f"optimise_report_{race_slug(cfg['race']['name'])}{report_suffix}.json")
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


def cmd_report(cfg: dict, args: argparse.Namespace) -> None:
    """Render the analysis HTML from the latest snapshot/fit/optimise run."""
    from report import build_report

    build_report(cfg, snapshot_name=getattr(args, "snapshot", None))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",
                        choices=["fetch", "combine", "fit", "validate",
                                 "optimise", "report", "all", "snapshots",
                                 "dnf", "attrition"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--source", default="betfair",
                        choices=["betfair", "kalshi", "polymarket", "all"],
                        help="odds source to fetch; 'all' fetches every "
                             "source in sources.enabled and also saves a "
                             "combined snapshot")
    parser.add_argument("--manual", metavar="CSV",
                        help="fetch odds from a driver,market,odds CSV "
                             "instead of an API source")
    parser.add_argument("--snapshot", metavar="FILE",
                        help="fit/validate against a specific archived odds "
                             "snapshot (filename under data/) instead of the latest")
    parser.add_argument("--allow-stale", action="store_true",
                        help="use a snapshot older than 24h")
    parser.add_argument("--refresh-dnf", action="store_true",
                        help="re-fetch the season DNF rates from OpenF1 "
                             "instead of using the cached counts under data/")
    parser.add_argument("--bayes", action="store_true",
                        help="Bayesian path: fit samples P(theta | odds); "
                             "validate/optimise consume posterior-predictive "
                             "races (MATH.md Section 7)")
    parser.add_argument("--bayes-method", choices=["mcmc", "is"],
                        help="override bayes.method from config: ensemble "
                             "MCMC (robust) or importance sampling (fast)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.command == "snapshots":
        cmd_snapshots(cfg, args)
        return
    if args.command == "dnf":
        cmd_dnf(cfg, args)
        return
    if args.command == "attrition":
        cmd_attrition(cfg, args)
        return
    if args.command == "combine":
        cmd_combine(cfg, args)
        return
    if args.command in ("fetch", "all"):
        cmd_fetch(cfg, args)
    if args.command in ("fit", "all"):
        cmd_fit(cfg, args)
    if args.command == "validate":
        cmd_validate(cfg, args)
    if args.command in ("optimise", "all"):
        cmd_optimise(cfg, args)
    if args.command in ("report", "all"):
        cmd_report(cfg, args)


if __name__ == "__main__":
    main()
