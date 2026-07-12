"""Analysis report — one self-contained HTML page for the configured race.

Reads what the pipeline already produced (latest odds snapshot, model
fit, optimise report) and renders it as a single HTML file with no
external assets. Re-run it any time after `fit`/`optimise`:

    python report.py                     # data/analysis_<race>.html
    python report.py --out my_page.html
    python report.py --snapshot odds_belgian_20260707T085322Z.json
    python main.py report                # same thing via the main CLI

The page covers the current race only — market vs model calibration,
the optimal ticket, runner-ups and DNF probabilities. Backtesting is
deliberately out of scope (see backtest.py's own text output).
"""

import _paths  # noqa: F401

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path

from model.fit import load_fit
from model.simulate import simulate_races, top_k_probs
from odds.devig import devig_snapshot
from odds.snapshot import (
    DATA_DIR,
    MAX_AGE_HOURS,
    load_latest_snapshot,
    load_snapshot,
    race_slug,
)
from race_utils import DRIVER_MAP
from scorer import Scorer

FLAG_PP = 0.02  # validation flag threshold, matches model/validate.py

# Reference dataviz palette (light / dark), consumed unchanged.
_CSS = """
:root {
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --border: rgba(11,11,11,0.10);
  --bar: #2a78d6; --dot: #1baf7a; --warn-bg: #fff3e2; --flag: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root {
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --border: rgba(255,255,255,0.10);
    --bar: #3987e5; --dot: #199e70; --warn-bg: #3a2c14; --flag: #e66767;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
}
main { max-width: 900px; margin: 0 auto; padding: 32px 20px 64px; }
h1 { font-size: 26px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 0 0 12px; }
.sub { color: var(--ink-2); margin: 0 0 20px; }
.card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 20px; margin: 0 0 20px;
}
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0 0; }
.chip {
  font-size: 12px; color: var(--ink-2); border: 1px solid var(--border);
  border-radius: 999px; padding: 2px 10px; white-space: nowrap;
}
.banner {
  background: var(--warn-bg); border: 1px solid var(--border);
  border-radius: 10px; padding: 10px 14px; margin: 0 0 20px; color: var(--ink);
}
.tiles { display: flex; flex-wrap: wrap; gap: 12px; margin: 0 0 16px; }
.tile {
  flex: 1 1 120px; background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 12px 16px;
}
.tile .label { font-size: 12px; color: var(--muted); }
.tile .value { font-size: 28px; font-weight: 600; }
.tile.hero .value { font-size: 44px; }
table { border-collapse: collapse; width: 100%; }
th {
  text-align: left; font-size: 12px; color: var(--muted); font-weight: 600;
  padding: 6px 10px; border-bottom: 1px solid var(--grid);
}
td { padding: 6px 10px; border-bottom: 1px solid var(--grid); }
tr:last-child td { border-bottom: none; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.badge {
  font-size: 11px; font-weight: 600; color: var(--ink-2);
  border: 1px solid var(--border); border-radius: 999px; padding: 1px 8px;
  white-space: nowrap;
}
.flag { color: var(--flag); font-weight: 600; }
.dim { color: var(--muted); }
.barwrap { position: relative; height: 20px; min-width: 220px; }
.bar {
  position: absolute; top: 4px; height: 12px; background: var(--bar);
  border-radius: 0 4px 4px 0;
}
.dotmark {
  position: absolute; top: 6px; width: 8px; height: 8px; border-radius: 50%;
  background: var(--dot); box-shadow: 0 0 0 2px var(--surface);
  transform: translateX(-4px);
}
.legend {
  display: flex; gap: 18px; font-size: 12px; color: var(--ink-2);
  margin: 0 0 10px; align-items: center;
}
.key-bar {
  display: inline-block; width: 18px; height: 10px; background: var(--bar);
  border-radius: 0 3px 3px 0; margin-right: 6px; vertical-align: -1px;
}
.key-dot {
  display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  background: var(--dot); box-shadow: 0 0 0 2px var(--surface);
  margin-right: 6px; vertical-align: -1px;
}
footer { color: var(--muted); font-size: 12px; margin-top: 8px; }
"""


def _driver(code: str) -> str:
    return html.escape(DRIVER_MAP.get(code, code))


def _pct(p: float) -> str:
    return f"{p * 100:.1f}%"


def _load_optimise_report(race_name: str) -> dict:
    path = DATA_DIR / f"optimise_report_{race_slug(race_name)}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No optimise report at {path}. Run `python main.py optimise` first."
        )
    return json.loads(path.read_text())


def _snapshot_age_hours(snapshot: dict) -> float:
    fetched = datetime.fromisoformat(snapshot["fetched_at"])
    return (datetime.now(timezone.utc) - fetched).total_seconds() / 3600


def _header_section(cfg: dict, snapshot: dict, markets: dict) -> str:
    race = cfg["race"]
    age = _snapshot_age_hours(snapshot)
    chips = "".join(f'<span class="chip">{html.escape(m)}</span>'
                    for m in markets["markets_used"])
    banner = ""
    if age > MAX_AGE_HOURS:
        banner = (f'<div class="banner">⚠ Odds snapshot is {age:.0f}h old '
                  f'(&gt; {MAX_AGE_HOURS:.0f}h). Re-fetch close to the comp '
                  f'deadline and re-run fit / optimise / report.</div>')
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"<h1>{html.escape(race['name'])} — tipping analysis</h1>"
        f'<p class="sub">Round {race["round"]}, {race["year"]} &middot; '
        f'odds source: {html.escape(snapshot.get("source", "?"))} '
        f'({age:.1f}h old) &middot; generated {generated}</p>'
        f"{banner}"
        f'<div class="card"><h2>Markets in the snapshot</h2>'
        f'<div class="chips">{chips}</div></div>'
    )


def _ticket_section(opt: dict) -> str:
    underdog_top10 = set(opt.get("underdog_top10") or [])
    pcts = opt["score_percentiles"]
    tiles = (
        f'<div class="tiles">'
        f'<div class="tile hero"><div class="label">Expected points</div>'
        f'<div class="value">{opt["best_ev"]:.2f}</div></div>'
        + "".join(
            f'<div class="tile"><div class="label">{lab}</div>'
            f'<div class="value">{pcts[key]:.0f}</div></div>'
            for lab, key in (("p10 score", "p10"), ("median score", "p50"),
                             ("p90 score", "p90"))
        )
        + "</div>"
    )
    rows = []
    for pos, code in enumerate(opt["best_ticket"], start=1):
        badge = ('<span class="badge">underdog 2×</span>'
                 if underdog_top10 and code not in underdog_top10 else "")
        rows.append(f"<tr><td>P{pos}</td><td>{html.escape(code)}</td>"
                    f"<td>{_driver(code)}</td><td>{badge}</td></tr>")
    runner_rows = "".join(
        f'<tr><td class="num">{r["ev"]:.2f}</td>'
        f'<td>{html.escape(" ".join(r["ticket"]))}</td></tr>'
        for r in opt.get("runner_ups", [])
    )
    return (
        f'<div class="card"><h2>Optimal ticket '
        f'({opt["n_sims"]:,} simulated races)</h2>{tiles}'
        f"<table><thead><tr><th>Slot</th><th>Code</th><th>Driver</th>"
        f"<th></th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
        f'<div class="card"><h2>Runner-up tickets</h2>'
        f'<table><thead><tr><th class="num">EV</th><th>Ticket (P1 → P10)</th>'
        f"</tr></thead><tbody>{runner_rows}</tbody></table></div>"
    )


def _win_chart_section(markets: dict, sims, sim_win: dict[str, float]) -> str:
    market_win = markets["topk"][1]
    scale = max(max(market_win.values()), max(sim_win.values())) or 1.0
    rows = []
    for code in sorted(market_win, key=market_win.get, reverse=True):
        mkt, mod = market_win[code], sim_win.get(code, 0.0)
        bar_w = 100.0 * mkt / scale
        dot_x = 100.0 * mod / scale
        rows.append(
            f"<tr><td>{html.escape(code)}</td>"
            f'<td class="barwrap-cell"><div class="barwrap" '
            f'title="{_driver(code)}: market {_pct(mkt)}, model {_pct(mod)}">'
            f'<div class="bar" style="width:{bar_w:.1f}%"></div>'
            f'<div class="dotmark" style="left:{dot_x:.1f}%"></div></div></td>'
            f'<td class="num">{_pct(mkt)}</td>'
            f'<td class="num">{_pct(mod)}</td></tr>'
        )
    return (
        f'<div class="card"><h2>Win probability — market vs model</h2>'
        f'<div class="legend"><span><span class="key-bar"></span>'
        f'Market (de-vigged)</span><span><span class="key-dot"></span>'
        f"Model ({sims.n_sims:,} sims)</span></div>"
        f'<table><thead><tr><th>Driver</th><th></th><th class="num">Market</th>'
        f'<th class="num">Model</th></tr></thead><tbody>{"".join(rows)}'
        f"</tbody></table></div>"
    )


def _validation_section(markets: dict, sims) -> str:
    col = {c: i for i, c in enumerate(sims.drivers)}
    rows, n_flagged = [], 0
    for k, probs in sorted(markets["topk"].items()):
        if k == 1:
            continue  # the win market has its own chart above
        sim_p = top_k_probs(sims.finish_pos, k)
        for code in sorted(probs, key=probs.get, reverse=True):
            diff = sim_p[col[code]] - probs[code]
            flagged = abs(diff) > FLAG_PP
            n_flagged += flagged
            diff_cell = (f'<span class="flag">⚠ {diff * 100:+.1f}pp</span>'
                         if flagged else
                         f'<span class="dim">{diff * 100:+.1f}pp</span>')
            rows.append(
                f"<tr><td>top {k}</td><td>{html.escape(code)}</td>"
                f'<td class="num">{_pct(probs[code])}</td>'
                f'<td class="num">{_pct(sim_p[col[code]])}</td>'
                f'<td class="num">{diff_cell}</td></tr>'
            )
    if not rows:
        return (
            '<div class="card"><h2>Fit validation — intermediate markets</h2>'
            '<p class="dim">Only the win market is priced so far; there are '
            "no top-N markets to validate against. The mid-order ranking is "
            "model extrapolation from win odds — re-fetch closer to the race "
            "weekend.</p></div>"
        )
    verdict = (f'<p class="dim">⚠ {n_flagged} marginal(s) deviate by more '
               f"than {FLAG_PP:.0%}.</p>" if n_flagged else
               f'<p class="dim">All marginals within {FLAG_PP:.0%} of the '
               "de-vigged market probabilities.</p>")
    return (
        '<div class="card"><h2>Fit validation — intermediate markets</h2>'
        f"{verdict}<table><thead><tr><th>Market</th><th>Driver</th>"
        f'<th class="num">Market</th><th class="num">Model</th>'
        f'<th class="num">Diff</th></tr></thead><tbody>{"".join(rows)}'
        "</tbody></table></div>"
    )


def _dnf_section(cfg: dict, markets: dict, fit: dict) -> str:
    per_driver = (cfg["model"]["dnf"].get("per_driver") or {})
    market_dnf = markets.get("dnf") or {}
    dnf_probs = fit["dnf_probs"]
    scale = max(dnf_probs.values()) or 1.0
    rows = []
    for code in sorted(dnf_probs, key=dnf_probs.get, reverse=True):
        p = dnf_probs[code]
        source = ("config override" if code in per_driver
                  else "market (classified)" if code in market_dnf
                  else "season prior")
        rows.append(
            f"<tr><td>{html.escape(code)}</td>"
            f'<td class="barwrap-cell"><div class="barwrap" '
            f'title="{_driver(code)}: P(DNF) {_pct(p)}">'
            f'<div class="bar" style="width:{100.0 * p / scale:.1f}%"></div>'
            f'</div></td>'
            f'<td class="num">{_pct(p)}</td>'
            f'<td class="num">{p * Scorer.POINTS_DNF:.1f}</td>'
            f'<td class="dim">{source}</td></tr>'
        )
    note = ""
    if not market_dnf:
        note = ('<p class="dim">No "To be Classified" markets in this '
                "snapshot — probabilities fall back to the season prior. "
                "Betfair lists those markets close to the race weekend; "
                "re-fetch then for market-implied DNF risk.</p>")
    return (
        '<div class="card"><h2>DNF probabilities &amp; DNF-pick value</h2>'
        f"{note}<table><thead><tr><th>Driver</th><th></th>"
        f'<th class="num">P(DNF)</th>'
        f'<th class="num">Pick EV ({Scorer.POINTS_DNF} pts)</th>'
        f'<th>Source</th></tr></thead><tbody>{"".join(rows)}</tbody>'
        "</table></div>"
    )


def build_report(cfg: dict, snapshot_name: str | None = None,
                 out: str | Path | None = None) -> Path:
    """Assemble the analysis page for the configured race and write it."""
    race_name = cfg["race"]["name"]
    if snapshot_name:
        snapshot = load_snapshot(snapshot_name, allow_stale=True)
    else:
        # The page reports on whatever exists; staleness becomes a banner,
        # not an error.
        snapshot = load_latest_snapshot(race_name, allow_stale=True)
    markets = devig_snapshot(snapshot, cfg["devig"]["win_method"],
                             cfg["devig"].get("topn_method", "power"))
    fit = load_fit(race_name)
    opt = _load_optimise_report(race_name)

    sims = simulate_races(
        fit["theta"], fit["dnf_probs"],
        n_sims=int(cfg["model"]["n_sims"]),
        seed=int(cfg["model"]["seed"]) + 1,  # same set validate/optimise use
        sigma_by_code=fit.get("sigma"),
        dist=fit.get("model", "gumbel"),
    )
    win = top_k_probs(sims.finish_pos, 1)
    sim_win = {c: float(win[i]) for i, c in enumerate(sims.drivers)}

    body = (
        _header_section(cfg, snapshot, markets)
        + _ticket_section(opt)
        + _win_chart_section(markets, sims, sim_win)
        + _validation_section(markets, sims)
        + _dnf_section(cfg, markets, fit)
        + "<footer>Generated by report.py from the latest snapshot, model "
          "fit and optimise report. Re-run fetch / fit / optimise first to "
          "refresh the inputs.</footer>"
    )
    page = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{html.escape(race_name)} — tipping analysis</title>"
        f"<style>{_CSS}</style></head><body><main>{body}</main></body></html>"
    )

    out_path = Path(out) if out else DATA_DIR / f"analysis_{race_slug(race_name)}.html"
    out_path.write_text(page, encoding="utf-8")
    print(f"Analysis report saved -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--snapshot", metavar="FILE",
                        help="report on a specific archived snapshot instead "
                             "of the latest")
    parser.add_argument("--out", metavar="HTML",
                        help="output path (default data/analysis_<race>.html)")
    args = parser.parse_args()

    import yaml
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).resolve().parent / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text())
    build_report(cfg, snapshot_name=args.snapshot, out=args.out)


if __name__ == "__main__":
    main()
