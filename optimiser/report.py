"""Analysis report — one self-contained HTML page for the configured race.

Reads what the pipeline already produced (latest odds snapshot, model
fit, optimise report) and renders it as a single HTML file with no
external assets — every chart is inline SVG, light/dark aware. Re-run it
any time after `fit`/`optimise`:

    python report.py                     # data/analysis_<race>.html
    python report.py --out my_page.html
    python report.py --snapshot odds_belgian_20260707T085322Z.json
    python main.py report                # same thing via the main CLI

The page covers the current race only. It now reports, in order:

  * the optimal ticket with the per-slot expected-points breakdown;
  * an interactive score distribution — click the pick or any runner-up
    to see its histogram on shared axes and how far behind it sits;
  * fit diagnostics — loss, iterations, calibration error (MAD/RMSE),
    the shock loading, the markets used;
  * a market browser — click through every priced market for its
    liquidity, fit weight, model fit and priced runners;
  * the fitted model parameters (μ location + σ dispersion per driver)
    and a μ-vs-σ scatter;
  * a market-vs-model calibration scatter across every top-N market;
  * the win-probability bar/dot comparison and the intermediate-market
    validation table (worst misses first, filterable by market);
  * the full finishing-position distribution as a heatmap;
  * per-driver DNF probabilities and DNF-pick value.

Backtesting is deliberately out of scope (see backtest.py's own text
output).
"""

import _paths  # noqa: F401

import argparse
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from model.fit import load_fit
from model.simulate import mean_finish, simulate_races, top_k_probs
from odds.devig import (
    _market_weight,
    devig_h2h,
    devig_snapshot,
    select_price,
)
from odds.snapshot import (
    DATA_DIR,
    MAX_AGE_HOURS,
    load_latest_snapshot,
    load_snapshot,
    race_slug,
)
from race_utils import DRIVER_MAP
from scorer import Scorer
from scoring.rules import expected_points_matrix, ticket_scores

FLAG_PP = 0.02  # validation flag threshold, matches model/validate.py

# Reference dataviz palette (light / dark), consumed unchanged. The
# --series-* slots are the categorical order from the palette; charts
# reference roles, so light/dark swaps in one place.
_CSS = """
:root {
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7;
  --border: rgba(11,11,11,0.10);
  --bar: #2a78d6; --dot: #1baf7a; --warn-bg: #fff3e2; --flag: #d03b3b;
  --good: #006300;
  --series-1: #2a78d6; --series-2: #1baf7a; --series-3: #eda100;
  --series-4: #008300;
}
@media (prefers-color-scheme: dark) {
  :root {
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --axis: #383835;
    --border: rgba(255,255,255,0.10);
    --bar: #3987e5; --dot: #199e70; --warn-bg: #3a2c14; --flag: #e66767;
    --good: #0ca30c;
    --series-1: #3987e5; --series-2: #199e70; --series-3: #c98500;
    --series-4: #008300;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
}
main { max-width: 940px; margin: 0 auto; padding: 32px 20px 64px; }
h1 { font-size: 26px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 0 0 6px; }
.sub { color: var(--ink-2); margin: 0 0 20px; }
.note { color: var(--ink-2); font-size: 13px; margin: 0 0 14px; }
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
.tile .unit { font-size: 14px; color: var(--ink-2); font-weight: 400; }
.tile.hero .value { font-size: 44px; }
/* Compact diagnostics grid — many small numbers, not a chart. */
.statgrid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 1px; background: var(--border); border: 1px solid var(--border);
  border-radius: 10px; overflow: hidden;
}
.stat { background: var(--surface); padding: 10px 14px; }
.stat .k { font-size: 12px; color: var(--muted); }
.stat .v {
  font-size: 18px; font-weight: 600; font-variant-numeric: tabular-nums;
}
.stat .v.ok { color: var(--good); }
.stat .v.bad { color: var(--flag); }
table { border-collapse: collapse; width: 100%; }
th {
  text-align: left; font-size: 12px; color: var(--muted); font-weight: 600;
  padding: 6px 10px; border-bottom: 1px solid var(--grid);
}
td { padding: 6px 10px; border-bottom: 1px solid var(--grid); }
tr:last-child td { border-bottom: none; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.pick td { background: color-mix(in srgb, var(--bar) 7%, transparent); }
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
  display: flex; gap: 18px; flex-wrap: wrap; font-size: 12px;
  color: var(--ink-2); margin: 0 0 10px; align-items: center;
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
.key-sw {
  display: inline-block; width: 10px; height: 10px; border-radius: 3px;
  margin-right: 6px; vertical-align: -1px;
}
.figure { margin: 6px 0 0; }
.figure svg { display: block; width: 100%; height: auto; }
.figure [data-tip], .barwrap[data-tip] { cursor: crosshair; }
.scroll { overflow-x: auto; }
svg text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
/* Floating hover tooltip (inverted ink-on-page for contrast in both themes). */
.tip {
  position: fixed; z-index: 60; pointer-events: none; max-width: 260px;
  background: var(--ink); color: var(--page); font-size: 12px; line-height: 1.4;
  padding: 6px 9px; border-radius: 6px; box-shadow: 0 2px 12px rgba(0,0,0,0.28);
  font-variant-numeric: tabular-nums;
}
footer { color: var(--muted); font-size: 12px; margin-top: 8px; }
/* Clickable pill tabs (market/validation filters) + selectable table rows
   (ticket score-distribution picker). Both toggle an `.active` class from
   the tiny tab controller in _TABS_JS. */
.tabs { display: flex; flex-wrap: wrap; gap: 6px; margin: 2px 0 14px; }
.tab {
  font: inherit; font-size: 12px; cursor: pointer; color: var(--ink-2);
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 999px; padding: 4px 12px; white-space: nowrap;
}
.tab:hover { border-color: var(--axis); }
.tab.active { background: var(--bar); color: #fff; border-color: var(--bar); }
tr.selectable { cursor: pointer; }
tr.selectable:hover td { background: color-mix(in srgb, var(--bar) 6%, transparent); }
tr.selectable.active td { background: color-mix(in srgb, var(--bar) 13%, transparent); }
.mkt-meta {
  display: flex; flex-wrap: wrap; gap: 8px 18px; margin: 0 0 12px;
  font-size: 13px; color: var(--ink-2);
}
.mkt-meta b { color: var(--ink); font-weight: 600; font-variant-numeric: tabular-nums; }
"""

# Immediate, styled hover tooltip for any element carrying data-tip — a
# nicer replacement for the browser's delayed native <title> tooltip.
# Self-contained (no external assets); works on inline SVG marks too.
_TOOLTIP_JS = """
(function () {
  var tip = document.createElement('div');
  tip.className = 'tip';
  tip.style.display = 'none';
  document.body.appendChild(tip);
  function place(e) {
    var pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
    var x = e.clientX + pad, y = e.clientY + pad;
    if (x + w > window.innerWidth) x = e.clientX - w - pad;
    if (y + h > window.innerHeight) y = e.clientY - h - pad;
    tip.style.left = x + 'px';
    tip.style.top = y + 'px';
  }
  document.addEventListener('mouseover', function (e) {
    var el = e.target.closest && e.target.closest('[data-tip]');
    if (el) { tip.textContent = el.getAttribute('data-tip'); tip.style.display = 'block'; place(e); }
  });
  document.addEventListener('mousemove', function (e) {
    if (tip.style.display !== 'block') return;
    var el = e.target.closest && e.target.closest('[data-tip]');
    if (el) place(e); else tip.style.display = 'none';
  });
  document.addEventListener('mouseout', function (e) {
    if (e.target.closest && e.target.closest('[data-tip]')) tip.style.display = 'none';
  });
})();
"""

# A tiny declarative tab/filter controller shared by the interactive
# sections (ticket score-distribution picker, validation filter, market
# browser). Markup only, no per-section JS:
#   * a control carries  data-tab-group="G" data-tab-value="V"  (add
#     class "active" to the one selected by default);
#   * any content carries data-tab-group="G" data-tab-show="V1 V2 …" and
#     is shown iff the group's active value is in its space-separated list.
# Clicking a control activates its value across the whole group.
_TABS_JS = """
(function () {
  function apply(group, value) {
    document.querySelectorAll('[data-tab-group="' + group + '"]').forEach(function (el) {
      if (el.hasAttribute('data-tab-value')) {
        el.classList.toggle('active', el.getAttribute('data-tab-value') === value);
      } else if (el.hasAttribute('data-tab-show')) {
        var vals = el.getAttribute('data-tab-show').split(' ');
        el.style.display = vals.indexOf(value) >= 0 ? '' : 'none';
      }
    });
  }
  document.querySelectorAll('[data-tab-value]').forEach(function (tab) {
    tab.addEventListener('click', function () {
      apply(tab.getAttribute('data-tab-group'), tab.getAttribute('data-tab-value'));
    });
  });
  var seen = {};
  document.querySelectorAll('[data-tab-value]').forEach(function (tab) {
    var g = tab.getAttribute('data-tab-group');
    if (seen[g]) return;
    seen[g] = true;
    var active = document.querySelector('[data-tab-group="' + g + '"].active[data-tab-value]') || tab;
    apply(g, active.getAttribute('data-tab-value'));
  });
})();
"""


def _driver(code: str) -> str:
    return html.escape(DRIVER_MAP.get(code, code))


def _pct(p: float) -> str:
    return f"{p * 100:.1f}%"


def _esc(x) -> str:
    return html.escape(str(x))


def _tip(text: str) -> str:
    """A ``data-tip`` attribute the page-level tooltip script reads."""
    return f'data-tip="{html.escape(str(text), quote=True)}"'


# ──────────────────────────────────────────────────────────────────────
# Tiny inline-SVG chart toolkit (no JS, no external assets). Charts
# reference CSS custom properties so they theme with the page. Every mark
# carries a <title> for a native hover tooltip.
# ──────────────────────────────────────────────────────────────────────

def _nice_ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    """A handful of round tick values spanning [lo, hi]."""
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    norm = raw / mag
    step = (1 if norm < 1.5 else 2 if norm < 3 else 5 if norm < 7 else 10) * mag
    start = math.ceil(lo / step) * step
    ticks, t = [], start
    while t <= hi + step * 1e-6:
        ticks.append(round(t, 10))
        t += step
    return ticks


def _fmt_tick(v: float) -> str:
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v))}"
    return f"{v:.2f}".rstrip("0").rstrip(".")


class _Frame:
    """Axis frame with data→pixel mappings and pre-rendered chrome."""

    def __init__(self, xr, yr, xlabel, ylabel, w=560, h=380,
                 xticks=None, yticks=None, xfmt=_fmt_tick, yfmt=_fmt_tick):
        self.w, self.h = w, h
        self.L, self.R, self.T, self.B = 58, 20, 18, 46
        self.x0, self.x1 = xr
        self.y0, self.y1 = yr
        self._xticks = xticks if xticks is not None else _nice_ticks(*xr)
        self._yticks = yticks if yticks is not None else _nice_ticks(*yr)
        self.xlabel, self.ylabel = xlabel, ylabel
        self.xfmt, self.yfmt = xfmt, yfmt

    def sx(self, x: float) -> float:
        return self.L + (x - self.x0) / (self.x1 - self.x0) * (self.w - self.L - self.R)

    def sy(self, y: float) -> float:
        return self.h - self.B - (y - self.y0) / (self.y1 - self.y0) * (self.h - self.T - self.B)

    def chrome(self) -> str:
        parts = []
        for t in self._xticks:
            x = self.sx(t)
            parts.append(f'<line x1="{x:.1f}" y1="{self.T:.1f}" x2="{x:.1f}" '
                         f'y2="{self.h - self.B:.1f}" stroke="var(--grid)" stroke-width="1"/>')
            parts.append(f'<text x="{x:.1f}" y="{self.h - self.B + 15:.1f}" '
                         f'text-anchor="middle" font-size="11" fill="var(--muted)">'
                         f'{_esc(self.xfmt(t))}</text>')
        for t in self._yticks:
            y = self.sy(t)
            parts.append(f'<line x1="{self.L:.1f}" y1="{y:.1f}" x2="{self.w - self.R:.1f}" '
                         f'y2="{y:.1f}" stroke="var(--grid)" stroke-width="1"/>')
            parts.append(f'<text x="{self.L - 8:.1f}" y="{y + 4:.1f}" text-anchor="end" '
                         f'font-size="11" fill="var(--muted)">{_esc(self.yfmt(t))}</text>')
        # axis baselines
        parts.append(f'<line x1="{self.L:.1f}" y1="{self.h - self.B:.1f}" '
                     f'x2="{self.w - self.R:.1f}" y2="{self.h - self.B:.1f}" '
                     f'stroke="var(--axis)" stroke-width="1"/>')
        parts.append(f'<line x1="{self.L:.1f}" y1="{self.T:.1f}" x2="{self.L:.1f}" '
                     f'y2="{self.h - self.B:.1f}" stroke="var(--axis)" stroke-width="1"/>')
        # axis titles
        parts.append(f'<text x="{(self.L + self.w - self.R) / 2:.1f}" y="{self.h - 6:.1f}" '
                     f'text-anchor="middle" font-size="12" fill="var(--ink-2)">'
                     f'{_esc(self.xlabel)}</text>')
        cy = (self.T + self.h - self.B) / 2
        parts.append(f'<text x="14" y="{cy:.1f}" text-anchor="middle" font-size="12" '
                     f'fill="var(--ink-2)" transform="rotate(-90 14 {cy:.1f})">'
                     f'{_esc(self.ylabel)}</text>')
        return "".join(parts)


def _svg(frame: _Frame, body: str) -> str:
    return (f'<svg viewBox="0 0 {frame.w} {frame.h}" width="100%" role="img" '
            f'preserveAspectRatio="xMidYMid meet">{frame.chrome()}{body}</svg>')


# ──────────────────────────────────────────────────────────────────────
# Section builders
# ──────────────────────────────────────────────────────────────────────

def _snapshot_age_hours(snapshot: dict) -> float:
    fetched = datetime.fromisoformat(snapshot["fetched_at"])
    return (datetime.now(timezone.utc) - fetched).total_seconds() / 3600


def _load_optimise_report(race_name: str) -> dict:
    path = DATA_DIR / f"optimise_report_{race_slug(race_name)}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No optimise report at {path}. Run `python main.py optimise` first."
        )
    return json.loads(path.read_text())


def _header_section(cfg: dict, snapshot: dict, markets: dict) -> str:
    race = cfg["race"]
    age = _snapshot_age_hours(snapshot)
    chips = "".join(f'<span class="chip">{_esc(m)}</span>'
                    for m in markets["markets_used"])
    banner = ""
    if age > MAX_AGE_HOURS:
        banner = (f'<div class="banner">⚠ Odds snapshot is {age:.0f}h old '
                  f'(&gt; {MAX_AGE_HOURS:.0f}h). Re-fetch close to the comp '
                  f'deadline and re-run fit / optimise / report.</div>')
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"<h1>{_esc(race['name'])} — tipping analysis</h1>"
        f'<p class="sub">Round {race["round"]}, {race["year"]} &middot; '
        f'odds source: {_esc(snapshot.get("source", "?"))} '
        f'({age:.1f}h old) &middot; generated {generated}</p>'
        f"{banner}"
        f'<div class="card"><h2>Markets in the snapshot</h2>'
        f'<p class="note">De-vigged and fed to the fit. Each chip is one '
        f'priced market; the model is calibrated to reproduce all of them '
        f'at once.</p>'
        f'<div class="chips">{chips}</div></div>'
    )


def _ticket_section(opt: dict, model: dict) -> str:
    """Optimal ticket: hero tiles, per-slot breakdown, score histogram."""
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
        m = model[code]
        badge = ('<span class="badge">underdog 2×</span>'
                 if underdog_top10 and code not in underdog_top10 else "")
        rows.append(
            f'<tr><td>P{pos}</td><td>{_esc(code)}</td>'
            f"<td>{_driver(code)}</td>"
            f'<td class="num">{m["slot_ev"][pos - 1]:.2f}</td>'
            f'<td class="num">{_pct(m["win"])}</td>'
            f'<td class="num">{m["mean_finish"]:.1f}</td>'
            f"<td>{badge}</td></tr>"
        )
    return (
        f'<div class="card"><h2>Optimal ticket '
        f'({opt["n_sims"]:,} simulated races)</h2>{tiles}'
        f'<p class="note">Per-slot EV is the mean points this driver earns '
        f'in this predicted slot across every simulated race — the ticket EV '
        f'is their sum. Underdog picks (outside the championship top-10) score '
        f'double. The full score distribution — and how it compares to each '
        f'runner-up — is in the next card.</p>'
        f"<table><thead><tr><th>Slot</th><th>Code</th><th>Driver</th>"
        f'<th class="num">Slot EV</th><th class="num">P(win)</th>'
        f'<th class="num">E[finish]</th><th></th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _hist_edges(lo: int, hi: int, max_bins: int = 60) -> np.ndarray:
    """Integer bin edges so every bar spans the same whole number of
    integer scores.

    Ticket scores are integers, so an even split of ``[lo, hi]`` into
    ``max_bins`` non-integer-width bins lands a variable count of integer
    scores in each bar (one bar 3–4, the next only 5) and the histogram
    spikes. Fixing the bin *width* to an integer — widened past 1 only
    when the range is large — makes every bar hold the same number of
    scores, so the shape is the distribution and not a binning artefact.
    """
    span = max(hi - lo + 1, 1)
    bw = max(1, math.ceil(span / max_bins))
    n = math.ceil(span / bw)
    return lo + bw * np.arange(n + 1)


def _hist_frame(edges: np.ndarray, ymax: float) -> _Frame:
    """A shared score-histogram frame; several tickets reuse it so the
    axes stay fixed as you flip between their distributions."""
    lo, hi = int(edges[0]), int(edges[-1])
    return _Frame((lo, hi), (0, (ymax or 1.0) * 1.08), "ticket points",
                  "share of races", w=720, h=300, xticks=_nice_ticks(lo, hi, 6),
                  yfmt=lambda v: f"{v * 100:.0f}%")


def _hist_body(scores: np.ndarray, edges: np.ndarray, fr: _Frame,
               pcts: dict) -> str:
    """Bars + p10/median/p90 reference lines for one ticket on ``fr``."""
    counts, _ = np.histogram(scores, bins=edges)
    n = len(scores) or 1
    body = []
    for c, e0, e1 in zip(counts, edges[:-1], edges[1:]):
        if c == 0:
            continue
        frac = c / n
        x = fr.sx(e0)
        wpx = max(fr.sx(e1) - fr.sx(e0) - 2, 1)  # 2px surface gap between columns
        y = fr.sy(frac)
        hpx = fr.sy(0) - y
        lab = (f"{int(e0)}" if int(e1) - int(e0) == 1
               else f"{int(e0)}–{int(e1) - 1}")
        body.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{wpx:.1f}" height="{hpx:.1f}" '
            f'rx="3" fill="var(--bar)" '
            f'{_tip(f"{lab} pts: {_pct(frac)} of races")}/>'
        )
    marks = [("p10", pcts["p10"], "var(--muted)"),
             ("median", pcts["p50"], "var(--dot)"),
             ("p90", pcts["p90"], "var(--muted)")]
    for lab, val, col in marks:
        x = fr.sx(val)
        body.append(
            f'<line x1="{x:.1f}" y1="{fr.T:.1f}" x2="{x:.1f}" '
            f'y2="{fr.sy(0):.1f}" stroke="{col}" stroke-width="1.5" '
            f'stroke-dasharray="4 3"/>'
            f'<text x="{x:.1f}" y="{fr.T - 4:.1f}" text-anchor="middle" '
            f'font-size="10" fill="var(--ink-2)">{lab} {val:.0f}</text>'
        )
    return _svg(fr, "".join(body))


def _distribution_section(tickets: list[dict]) -> str:
    """Interactive score distributions: the pick plus every runner-up,
    on shared axes, selected by clicking a row.

    ``tickets[0]`` is the pick; each entry carries ``ev``, ``ticket``,
    ``scores`` (per-sim totals) and ``pcts`` (mean/p10/p50/p90).
    """
    lo = min(int(t["scores"].min()) for t in tickets)
    hi = max(int(t["scores"].max()) for t in tickets)
    edges = _hist_edges(lo, hi)
    ymax = 0.0
    for t in tickets:
        counts, _ = np.histogram(t["scores"], bins=edges)
        ymax = max(ymax, counts.max() / (len(t["scores"]) or 1))
    fr = _hist_frame(edges, ymax)

    best_ev = tickets[0]["ev"]
    rows, panels = [], []
    for i, t in enumerate(tickets):
        p = t["pcts"]
        name = "Pick" if i == 0 else f"Runner-up {i}"
        gap = "—" if i == 0 else f"−{best_ev - t['ev']:.2f}"
        rows.append(
            f'<tr class="selectable{" active" if i == 0 else ""}" '
            f'data-tab-group="dist" data-tab-value="{i}">'
            f'<td>{name}</td><td class="num">{t["ev"]:.2f}</td>'
            f'<td class="num dim">{gap}</td>'
            f'<td>{_esc(" ".join(t["ticket"]))}</td>'
            f'<td class="num">{p["p10"]:.0f}</td>'
            f'<td class="num">{p["p50"]:.0f}</td>'
            f'<td class="num">{p["p90"]:.0f}</td></tr>'
        )
        hide = "" if i == 0 else ' style="display:none"'
        panels.append(
            f'<div data-tab-group="dist" data-tab-show="{i}"{hide}>'
            f'<p class="note" style="margin-top:14px"><strong>{name}</strong> '
            f'({_esc(" ".join(t["ticket"]))}) — mean {p["mean"]:.2f} · '
            f'p10 {p["p10"]:.0f} · median {p["p50"]:.0f} · p90 {p["p90"]:.0f}</p>'
            f'<div class="figure">{_hist_body(t["scores"], edges, fr, p)}</div></div>'
        )
    return (
        f'<div class="card"><h2>Score distribution — pick vs runner-ups</h2>'
        f'<p class="note">The next-best tickets on the same simulation set '
        f'(common random numbers), with their EV gap to the pick — a tiny gap '
        f'means the choice is close to a toss-up. <strong>Click any row</strong> '
        f'to see how that ticket actually pays out across the '
        f'{len(tickets[0]["scores"]):,} simulated races (axes are shared, so '
        f'the shape is comparable). A long right tail is correlated attrition '
        f'promoting several underdogs at once.</p>'
        f'<div class="scroll"><table><thead><tr><th>Ticket</th>'
        f'<th class="num">EV</th><th class="num">vs best</th>'
        f'<th>Drivers (P1 → P10)</th><th class="num">p10</th>'
        f'<th class="num">p50</th><th class="num">p90</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
        f'{"".join(panels)}</div>'
    )


def _diagnostics_section(fit: dict, opt: dict, calib: dict, sig_range) -> str:
    fr = fit["fit_report"]
    conv = fr.get("converged", False)
    lam = fit.get("shock_lambda", 0.0)
    stats = [
        ("model family", fit.get("model", "gumbel"), ""),
        ("fit loss (SSE)", f'{fr["loss"]:.4f}', ""),
        ("iterations", f'{fr["iterations"]}', ""),
        ("converged", "yes" if conv else "no", "ok" if conv else "bad"),
        ("calibration MAD", f'{calib["mad"] * 100:.2f}pp', ""),
        ("calibration RMSE", f'{calib["rmse"] * 100:.2f}pp', ""),
        ("worst miss", f'{calib["worst"] * 100:.1f}pp', ""),
        ("σ range", f'{sig_range[0]:.2f}–{sig_range[1]:.2f}', ""),
        ("shock loading λ", f'{lam:.2f}', ""),
        ("fit draws", f'{fr["fit_sims"]:,}', ""),
        ("soft-rank τ", f'{fr["tau"]:g}', ""),
        ("MC races", f'{opt["n_sims"]:,}', ""),
        ("seed", f'{fr["seed"]}', ""),
    ]
    cells = "".join(
        f'<div class="stat"><div class="k">{_esc(k)}</div>'
        f'<div class="v {cls}">{_esc(v)}</div></div>'
        for k, v, cls in stats
    )
    used = "".join(f'<span class="chip">{_esc(m)}</span>'
                   for m in fr.get("markets_used", []))
    return (
        f'<div class="card"><h2>Fit diagnostics</h2>'
        f'<p class="note">The optimiser fits per-driver strengths (and '
        f'dispersions) so the model reproduces every de-vigged market at '
        f'once. Loss is the liquidity-weighted squared error at the optimum; '
        f'MAD/RMSE are the average and root-mean-square gap between model and '
        f'market marginals on the hard {opt["n_sims"]:,}-race check. λ is the '
        f'shared attrition shock loading (0 = independent DNFs).</p>'
        f'<div class="statgrid">{cells}</div>'
        f'<div class="chips" style="margin-top:14px">{used}</div></div>'
    )


def _parameters_section(fit: dict, model: dict) -> str:
    theta = fit["theta"]
    sigma = fit.get("sigma") or {}
    dnf = fit["dnf_probs"]
    gaussian = fit.get("model") == "gaussian"
    order = sorted(theta, key=theta.get, reverse=True)
    rows = []
    for code in order:
        m = model[code]
        sig_cell = (f'<td class="num">{sigma[code]:.2f}</td>' if gaussian
                    else "")
        rows.append(
            f'<tr><td>{_esc(code)}</td><td>{_driver(code)}</td>'
            f'<td class="num">{theta[code]:+.2f}</td>{sig_cell}'
            f'<td class="num">{_pct(m["win"])}</td>'
            f'<td class="num">{_pct(m["top3"])}</td>'
            f'<td class="num">{_pct(m["top10"])}</td>'
            f'<td class="num">{m["mean_finish"]:.1f}</td>'
            f'<td class="num">{_pct(dnf[code])}</td></tr>'
        )
    sig_head = '<th class="num">σ spread</th>' if gaussian else ""
    intro = (
        "Each driver has a performance <em>location</em> μ and, in the "
        "Gaussian dispersion model, a <em>spread</em> σ — performance is "
        "Xᵢ = μᵢ + σᵢ·εᵢ and the field is ranked on X. A high σ means a "
        "wide, streaky outcome range (rookies, unreliable cars); a low σ "
        "is a metronome. The columns to the right are the marginals those "
        "parameters produce in simulation."
        if gaussian else
        "Each driver has a single Plackett-Luce strength θ; the field is "
        "ranked by θ + Gumbel noise. The right-hand columns are the "
        "marginals it produces in simulation."
    )
    scatter = _param_scatter(theta, sigma, model) if gaussian else ""
    return (
        f'<div class="card"><h2>Fitted model parameters</h2>'
        f'<p class="note">{intro}</p>'
        f'<div class="scroll"><table><thead><tr><th>Code</th><th>Driver</th>'
        f'<th class="num">μ strength</th>{sig_head}'
        f'<th class="num">P(win)</th><th class="num">P(top3)</th>'
        f'<th class="num">P(top10)</th><th class="num">E[finish]</th>'
        f'<th class="num">P(DNF)</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>{scatter}</div>"
    )


def _param_scatter(theta: dict, sigma: dict, model: dict) -> str:
    codes = list(theta)
    xs = [theta[c] for c in codes]
    ys = [sigma[c] for c in codes]
    xr = (min(xs) - 0.3, max(xs) + 0.3)
    yr = (min(ys) - 0.1, max(ys) + 0.15)
    fr = _Frame(xr, yr, "μ  (strength — higher finishes further forward)",
                "σ  (spread)", w=720, h=380)
    body = []
    for c in codes:
        x, y = fr.sx(theta[c]), fr.sy(sigma[c])
        tip = _tip(f'{c} {DRIVER_MAP.get(c, c)}: μ={theta[c]:+.2f}, '
                   f'σ={sigma[c]:.2f}, P(win)={_pct(model[c]["win"])}')
        body.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="var(--bar)" '
            f'stroke="var(--surface)" stroke-width="2" {tip}/>'
            f'<text x="{x + 7:.1f}" y="{y + 3.5:.1f}" font-size="10" '
            f'fill="var(--ink-2)">{_esc(c)}</text>'
        )
    return (
        '<h2 style="margin-top:22px">Strength vs spread</h2>'
        '<p class="note">Top-right is a fast <em>and</em> volatile driver; '
        'bottom-right is fast and consistent. The spread is exactly what a '
        'single-strength model cannot represent.</p>'
        f'<div class="figure">{_svg(fr, "".join(body))}</div>'
    )


_SERIES = {1: ("var(--series-1)", "win"), 3: ("var(--series-2)", "top 3"),
           5: ("var(--series-3)", "top 5"), 6: ("var(--series-3)", "top 6"),
           10: ("var(--series-4)", "top 10")}


def _calibration_section(calib: dict) -> str:
    pts = calib["points"]
    fr = _Frame((0, 1), (0, 1), "market probability (de-vigged)",
                "model probability (simulated)", w=560, h=460,
                xticks=[0, .25, .5, .75, 1], yticks=[0, .25, .5, .75, 1],
                xfmt=lambda v: f"{v * 100:.0f}%", yfmt=lambda v: f"{v * 100:.0f}%")
    # y = x reference (perfect calibration)
    body = [f'<line x1="{fr.sx(0):.1f}" y1="{fr.sy(0):.1f}" '
            f'x2="{fr.sx(1):.1f}" y2="{fr.sy(1):.1f}" stroke="var(--axis)" '
            f'stroke-width="1.5" stroke-dasharray="5 4"/>']
    for k, code, mkt, mod in pts:
        col = _SERIES.get(k, ("var(--bar)", f"top {k}"))[0]
        lab = _SERIES.get(k, (0, f"top {k}"))[1]
        x, y = fr.sx(mkt), fr.sy(mod)
        tip = _tip(f'{lab} — {code} {DRIVER_MAP.get(code, code)}: '
                   f'market {_pct(mkt)}, model {_pct(mod)} '
                   f'({(mod - mkt) * 100:+.1f}pp)')
        body.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{col}" '
            f'fill-opacity="0.85" stroke="var(--surface)" stroke-width="1.5" '
            f'{tip}/>'
        )
    ks = sorted({k for k, *_ in pts})
    legend = "".join(
        f'<span><span class="key-sw" style="background:{_SERIES.get(k, ("var(--bar)",))[0]}">'
        f'</span>{_esc(_SERIES.get(k, (0, f"top {k}"))[1])}</span>'
        for k in ks
    )
    return (
        f'<div class="card"><h2>Market vs model calibration</h2>'
        f'<p class="note">Every priced (market, driver) probability against '
        f'the marginal the fitted model produces. Points on the dashed '
        f'diagonal are perfectly reproduced; the further off, the worse the '
        f'model matches that price — <strong>hover any point</strong> for its '
        f'exact market/model split and driver. This is the single best '
        f'picture of fit quality.</p>'
        f'<div class="legend">{legend}</div>'
        f'<div class="figure">{_svg(fr, "".join(body))}</div>'
        f'<p class="note" style="margin-top:12px">Overall MAD '
        f'{calib["mad"] * 100:.2f}pp · RMSE {calib["rmse"] * 100:.2f}pp · '
        f'worst {calib["worst"] * 100:.1f}pp across {calib["n"]} priced '
        f'points.</p></div>'
    )


def _win_chart_section(markets: dict, sims, sim_win: dict[str, float]) -> str:
    market_win = markets["topk"][1]
    scale = max(max(market_win.values()), max(sim_win.values())) or 1.0
    rows = []
    for code in sorted(market_win, key=market_win.get, reverse=True):
        mkt, mod = market_win[code], sim_win.get(code, 0.0)
        bar_w = 100.0 * mkt / scale
        dot_x = 100.0 * mod / scale
        tip = _tip(f'{DRIVER_MAP.get(code, code)}: market {_pct(mkt)}, '
                   f'model {_pct(mod)}')
        rows.append(
            f"<tr><td>{_esc(code)}</td>"
            f'<td class="barwrap-cell"><div class="barwrap" {tip}>'
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


_VALID_WORST = 15  # rows shown on the default "worst misses" tab


def _validation_section(markets: dict, sims) -> str:
    col = {c: i for i, c in enumerate(sims.drivers)}
    entries = []  # (abs_diff, k, code, market_p, model_p, diff)
    for k, probs in sorted(markets["topk"].items()):
        if k == 1:
            continue  # the win market has its own chart above
        sim_p = top_k_probs(sims.finish_pos, k)
        for code in probs:
            model_p = float(sim_p[col[code]])
            diff = model_p - probs[code]
            entries.append((abs(diff), k, code, probs[code], model_p, diff))
    if not entries:
        return (
            '<div class="card"><h2>Fit validation — intermediate markets</h2>'
            '<p class="note">Only the win market is priced so far; there are '
            "no top-N markets to validate against. The mid-order ranking is "
            "model extrapolation from win odds — re-fetch closer to the race "
            "weekend.</p></div>"
        )
    entries.sort(key=lambda e: -e[0])  # worst calibration first
    n_flagged = sum(1 for e in entries if e[0] > FLAG_PP)
    ks = sorted({e[1] for e in entries})

    tabs = (
        '<button class="tab active" data-tab-group="valid" '
        f'data-tab-value="worst">worst {min(_VALID_WORST, len(entries))}</button>'
        + "".join(
            f'<button class="tab" data-tab-group="valid" '
            f'data-tab-value="top{k}">top {k}</button>' for k in ks
        )
    )
    rows = []
    for rank, (adiff, k, code, mkt_p, model_p, diff) in enumerate(entries):
        in_worst = rank < _VALID_WORST
        show = f"top{k}" + (" worst" if in_worst else "")
        hide = "" if in_worst else ' style="display:none"'  # default = worst
        flagged = adiff > FLAG_PP
        diff_cell = (f'<span class="flag">⚠ {diff * 100:+.1f}pp</span>'
                     if flagged else
                     f'<span class="dim">{diff * 100:+.1f}pp</span>')
        rows.append(
            f'<tr data-tab-group="valid" data-tab-show="{show}"{hide}>'
            f'<td>top {k}</td><td>{_esc(code)}</td><td>{_driver(code)}</td>'
            f'<td class="num">{_pct(mkt_p)}</td>'
            f'<td class="num">{_pct(model_p)}</td>'
            f'<td class="num">{diff_cell}</td></tr>'
        )
    verdict = (f'⚠ {n_flagged} of {len(entries)} marginal(s) deviate by more '
               f"than {FLAG_PP:.0%}." if n_flagged else
               f'All {len(entries)} marginals within {FLAG_PP:.0%} of the '
               "de-vigged market probabilities.")
    return (
        '<div class="card"><h2>Fit validation — intermediate markets</h2>'
        f'<p class="note">{verdict} Model vs de-vigged market marginal for '
        f'every priced (top-N, driver) point. Showing the worst-calibrated '
        f'{min(_VALID_WORST, len(entries))} by default — <strong>click a '
        f'market</strong> to see all of its runners instead.</p>'
        f'<div class="tabs">{tabs}</div>'
        f'<table><thead><tr><th>Market</th><th>Code</th><th>Driver</th>'
        f'<th class="num">Market</th><th class="num">Model</th>'
        f'<th class="num">Diff</th></tr></thead><tbody>{"".join(rows)}'
        "</tbody></table></div>"
    )


# Structural markets in browse order, with their top-k and display label.
_MKT_ORDER = [("win", 1, "Win"), ("top3", 3, "Top 3"), ("top5", 5, "Top 5"),
              ("top6", 6, "Top 6"), ("top10", 10, "Top 10")]


def _mkt_liquidity(tm) -> str:
    return f"{tm:,.0f}" if tm else "—"


def _mkt_meta(n: int, tm, weight: float, mad: float | None = None) -> str:
    """The liquidity / trust strip shown above each market's runner table."""
    parts = [f"runners <b>{n}</b>",
             f"matched volume <b>{_mkt_liquidity(tm)}</b>",
             f"fit weight <b>{weight:.2f}</b>"]
    if mad is not None:
        parts.append(f"model MAD <b>{mad * 100:.2f}pp</b>")
    return ('<div class="mkt-meta">'
            + "".join(f"<span>{p}</span>" for p in parts) + "</div>")


def _markets_section(snapshot: dict, markets: dict, sims, calib: dict) -> str:
    """Browsable per-market inputs: liquidity, fit weight, how well the
    model reproduces each one, and its priced runners."""
    raw = snapshot.get("markets", {})
    col = {c: i for i, c in enumerate(sims.drivers)}
    fp = sims.finish_pos
    mad_by_k = {k: mean_abs for (k, _cnt, mean_abs, _mx) in calib["per_market"]}
    # (tab_id, tab_label, detail_html)
    entries: list[tuple[str, str, str]] = []

    for key, k, label in _MKT_ORDER:
        if k not in markets["topk"]:
            continue
        probs = markets["topk"][k]
        rawm = raw.get(key, {})
        weight = markets["weights"].get(k, 1.0)
        sim_p = top_k_probs(fp, k)
        drows = []
        for code in sorted(probs, key=probs.get, reverse=True):
            mp, modp = probs[code], float(sim_p[col[code]])
            d = modp - mp
            dcls = "flag" if abs(d) > FLAG_PP else "dim"
            drows.append(
                f"<tr><td>{_esc(code)}</td><td>{_driver(code)}</td>"
                f'<td class="num">{_pct(mp)}</td><td class="num">{_pct(modp)}</td>'
                f'<td class="num"><span class="{dcls}">{d * 100:+.1f}pp</span></td>'
                f"</tr>"
            )
        detail = (
            f'<p class="note">{_esc(rawm.get("market_name", "?"))} — '
            f'{"win market" if k == 1 else f"P(finish in the top {k})"}, '
            f'de-vigged.</p>'
            f'{_mkt_meta(len(probs), rawm.get("total_matched"), weight, mad_by_k.get(k))}'
            f'<div class="scroll"><table><thead><tr><th>Code</th><th>Driver</th>'
            f'<th class="num">Market</th><th class="num">Model</th>'
            f'<th class="num">Diff</th></tr></thead>'
            f'<tbody>{"".join(drows)}</tbody></table></div>'
        )
        entries.append((key, label, detail))

    # Head-to-head matchups — each its own fit target.
    for j, market in enumerate(raw.get("h2h", [])):
        runners = market.get("runners", {})
        if len(runners) != 2:
            continue
        (a, ra), (b, rb) = sorted(runners.items())
        pa, pb = select_price(ra), select_price(rb)
        if pa is None or pb is None:
            continue
        p_a = devig_h2h(pa, pb)
        tm = market.get("total_matched")
        ia, ib = col.get(a), col.get(b)
        mod_a = float(np.mean(fp[:, ia] < fp[:, ib])) if None not in (ia, ib) else None
        drows = []
        for code, mp in ((a, p_a), (b, 1 - p_a)):
            modp = (mod_a if code == a else (1 - mod_a)) if mod_a is not None else None
            mod_cell = f'<td class="num">{_pct(modp)}</td>' if modp is not None else '<td class="num dim">—</td>'
            if modp is not None:
                d = modp - mp
                dcls = "flag" if abs(d) > FLAG_PP else "dim"
                diff_cell = f'<td class="num"><span class="{dcls}">{d * 100:+.1f}pp</span></td>'
            else:
                diff_cell = '<td class="num dim">—</td>'
            drows.append(
                f"<tr><td>{_esc(code)}</td><td>{_driver(code)}</td>"
                f'<td class="num">{_pct(mp)}</td>{mod_cell}{diff_cell}</tr>'
            )
        detail = (
            f'<p class="note">{_esc(market.get("market_name", "?"))} — '
            f'P(each driver finishes ahead of the other).</p>'
            f'{_mkt_meta(2, tm, _market_weight(tm))}'
            f'<table><thead><tr><th>Code</th><th>Driver</th>'
            f'<th class="num">Market</th><th class="num">Model</th>'
            f'<th class="num">Diff</th></tr></thead>'
            f'<tbody>{"".join(drows)}</tbody></table>'
        )
        entries.append((f"h2h{j}", f"{a} v {b}", detail))

    # "To be Classified" → per-driver DNF (an input, not a fit target).
    dnf = markets.get("dnf") or {}
    if dnf:
        cls = raw.get("classified") or {}
        yes = cls.get("yes") or {}
        tm = yes.get("total_matched")
        weight = next(iter((markets.get("dnf_weights") or {}).values()), 1.0)
        drows = []
        for code in sorted(dnf, key=dnf.get, reverse=True):
            drows.append(
                f"<tr><td>{_esc(code)}</td><td>{_driver(code)}</td>"
                f'<td class="num">{_pct(dnf[code])}</td>'
                f'<td class="num dim">{_pct(1 - dnf[code])}</td></tr>'
            )
        detail = (
            f'<p class="note">Per-driver "To be Classified" market, de-vigged '
            f'into P(DNF). Feeds the DNF layer directly — it is not a strength '
            f'fit target.</p>'
            f'{_mkt_meta(len(dnf), tm, weight)}'
            f'<div class="scroll"><table><thead><tr><th>Code</th><th>Driver</th>'
            f'<th class="num">P(DNF)</th><th class="num">P(classified)</th>'
            f'</tr></thead><tbody>{"".join(drows)}</tbody></table></div>'
        )
        entries.append(("classified", "Classified / DNF", detail))

    if not entries:
        return ""
    tabs = "".join(
        f'<button class="tab{" active" if i == 0 else ""}" '
        f'data-tab-group="mkt" data-tab-value="{tid}">{_esc(lab)}</button>'
        for i, (tid, lab, _d) in enumerate(entries)
    )
    panels = "".join(
        f'<div data-tab-group="mkt" data-tab-show="{tid}"'
        f'{"" if i == 0 else " style=\"display:none\""}>{d}</div>'
        for i, (tid, _lab, d) in enumerate(entries)
    )
    return (
        f'<div class="card"><h2>Market browser — inputs &amp; liquidity</h2>'
        f'<p class="note">Every market the fit consumed. <strong>Click a '
        f'market</strong> for its priced runners. <em>Matched volume</em> is '
        f'the liquidity behind it (units differ by source — GBP / USD / '
        f'contracts — so compare within a source); <em>fit weight</em> is how '
        f'much that liquidity lets the fit trust it (0.2 thin → 1.5 deep); '
        f'<em>model MAD</em> is the average gap between the fitted model and '
        f'this market, i.e. how well it is reproduced.</p>'
        f'<div class="tabs">{tabs}</div>{panels}</div>'
    )


def _finish_dist_section(sims, best_ticket: list[str], model: dict) -> str:
    """Heatmap of P(driver finishes at each classified position)."""
    n = len(sims.drivers)
    fp = sims.finish_pos
    order = sorted(range(n), key=lambda i: model[sims.drivers[i]]["mean_finish"])
    # position distribution per driver (0-based position -> fraction)
    dist = np.zeros((n, n))
    for i in range(n):
        counts = np.bincount(fp[:, i], minlength=n)
        dist[i] = counts / fp.shape[0]
    vmax = dist.max() or 1.0
    picks = set(best_ticket)

    gutter, top, cell_w, cell_h = 46, 18, 26, 20
    w = gutter + n * cell_w + 6
    h = top + n * cell_h + 6
    body = []
    # column headers (finishing position, 1-based)
    for p in range(n):
        cx = gutter + p * cell_w + cell_w / 2
        body.append(f'<text x="{cx:.1f}" y="{top - 5:.1f}" text-anchor="middle" '
                    f'font-size="9" fill="var(--muted)">{p + 1}</text>')
    for r, i in enumerate(order):
        code = sims.drivers[i]
        ry = top + r * cell_h
        weight = "600" if code in picks else "400"
        body.append(f'<text x="{gutter - 6:.1f}" y="{ry + cell_h / 2 + 3.5:.1f}" '
                    f'text-anchor="end" font-size="10" font-weight="{weight}" '
                    f'fill="var(--ink-2)">{_esc(code)}</text>')
        for p in range(n):
            v = dist[i, p]
            if v <= 0:
                continue
            op = 0.06 + 0.94 * (v / vmax) ** 0.6
            x = gutter + p * cell_w
            body.append(
                f'<rect x="{x + 1:.1f}" y="{ry + 1:.1f}" width="{cell_w - 2}" '
                f'height="{cell_h - 2}" rx="2" fill="var(--bar)" '
                f'fill-opacity="{op:.3f}" '
                f'{_tip(f"{code} {DRIVER_MAP.get(code, code)} finishes P{p + 1}: {_pct(v)}")}/>'
            )
    svg = (f'<svg viewBox="0 0 {w} {h}" width="{w}" role="img" '
           f'preserveAspectRatio="xMinYMin meet" style="max-width:none">'
           f'{"".join(body)}</svg>')
    return (
        '<div class="card"><h2>Finishing-position distribution</h2>'
        '<p class="note">Each row is a driver (ordered by expected finish, '
        'picked drivers in bold); each column a classified position. Cell '
        'darkness is the probability of landing exactly there. A tight, dark '
        'band = a predictable driver; a smeared row = a wide σ. The tail on '
        'the right is DNF/back-marker risk.</p>'
        f'<div class="figure scroll">{svg}</div></div>'
    )


def _dnf_section(cfg: dict, markets: dict, fit: dict) -> str:
    per_driver = (cfg["model"]["dnf"].get("per_driver") or {})
    market_dnf = markets.get("dnf") or {}
    dnf_probs = fit["dnf_probs"]
    lam = fit.get("shock_lambda", 0.0)
    scale = max(dnf_probs.values()) or 1.0
    rows = []
    for code in sorted(dnf_probs, key=dnf_probs.get, reverse=True):
        p = dnf_probs[code]
        source = ("config override" if code in per_driver
                  else "market (classified)" if code in market_dnf
                  else "season prior")
        rows.append(
            f"<tr><td>{_esc(code)}</td>"
            f'<td class="barwrap-cell"><div class="barwrap" '
            f'{_tip(f"{DRIVER_MAP.get(code, code)}: P(DNF) {_pct(p)}")}>'
            f'<div class="bar" style="width:{100.0 * p / scale:.1f}%"></div>'
            f'</div></td>'
            f'<td class="num">{_pct(p)}</td>'
            f'<td class="num">{p * Scorer.POINTS_DNF:.1f}</td>'
            f'<td class="dim">{source}</td></tr>'
        )
    note = (
        f'<p class="note">Per-driver retirement probability, resolved '
        f'config → market → season prior. A shared per-race shock '
        f'(λ = {lam:.2f}) correlates them so cars cluster out — the heavy '
        f'tail that promotes underdogs together. Pick EV is P(DNF) × '
        f'{Scorer.POINTS_DNF} pts for the separate DNF pick.</p>'
    )
    if not market_dnf:
        note += ('<p class="note">No "To be Classified" markets in this '
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


# ──────────────────────────────────────────────────────────────────────
# Derived quantities
# ──────────────────────────────────────────────────────────────────────

def _model_marginals(sims) -> dict[str, dict]:
    """Per-driver simulated marginals for the tables/scatters."""
    fp = sims.finish_pos
    win = top_k_probs(fp, 1)
    t3 = top_k_probs(fp, 3)
    t5 = top_k_probs(fp, 5)
    t10 = top_k_probs(fp, 10)
    mf = mean_finish(fp) + 1.0  # 1-based expected finishing position
    return {
        c: {"win": float(win[i]), "top3": float(t3[i]), "top5": float(t5[i]),
            "top10": float(t10[i]), "mean_finish": float(mf[i])}
        for i, c in enumerate(sims.drivers)
    }


def _calibration(markets: dict, sims) -> dict:
    """Market-vs-model residuals across every top-N market."""
    col = {c: i for i, c in enumerate(sims.drivers)}
    points, resid, per_market = [], [], []
    for k, probs in sorted(markets["topk"].items()):
        sim_p = top_k_probs(sims.finish_pos, k)
        rs = []
        for code, mkt in probs.items():
            mod = float(sim_p[col[code]])
            points.append((k, code, mkt, mod))
            r = mod - mkt
            resid.append(r)
            rs.append(abs(r))
        per_market.append((k, len(rs), float(np.mean(rs)), float(np.max(rs))))
    resid = np.array(resid) if resid else np.array([0.0])
    worst_i = int(np.argmax(np.abs(resid)))
    return {
        "points": points,
        "per_market": per_market,
        "n": len(points),
        "mad": float(np.mean(np.abs(resid))),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "worst": float(np.abs(resid)[worst_i]),
    }


def _ticket_multipliers(opt: dict, drivers: list[str]) -> np.ndarray:
    """Reconstruct the underdog multiplier vector from the saved report.

    Mirrors ``scoring.rules.ScoringContext.multiplier`` without a network
    call: from round 2, drivers outside the championship top-10 double.
    """
    champ = set(opt.get("underdog_top10") or [])
    active = int(opt["race"]["round"]) > 1 and bool(champ)
    return np.array(
        [Scorer.MULTIPLIER_UNDERDOG if (active and c not in champ) else 1
         for c in drivers], dtype=np.int64
    )


def _ticket_distributions(opt: dict, finish_pos: np.ndarray,
                          mult: np.ndarray, idx: dict[str, int]) -> list[dict]:
    """Per-sim scores + summary percentiles for the pick and every
    runner-up, on the shared sim set. Feeds ``_distribution_section``."""
    entries = [(opt["best_ticket"], opt["best_ev"])]
    entries += [(r["ticket"], r["ev"]) for r in opt.get("runner_ups", [])]
    tickets = []
    for codes, ev in entries:
        if any(c not in idx for c in codes):
            continue  # a ticket referencing a driver not in this sim set
        ti = np.array([idx[c] for c in codes], dtype=np.intp)
        scores = ticket_scores(finish_pos, ti, mult)
        p10, p50, p90 = np.percentile(scores, [10, 50, 90])
        tickets.append({
            "ev": ev, "ticket": list(codes), "scores": scores,
            "pcts": {"mean": float(scores.mean()), "p10": float(p10),
                     "p50": float(p50), "p90": float(p90)},
        })
    return tickets


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
        shock_lambda=float(fit.get("shock_lambda", 0.0)),  # match optimise
    )
    model = _model_marginals(sims)
    sim_win = {c: model[c]["win"] for c in sims.drivers}
    calib = _calibration(markets, sims)

    # Per-slot EV for the optimal ticket + its full score distribution,
    # on the same sim set / multipliers the optimiser used.
    mult = _ticket_multipliers(opt, sims.drivers)
    exp_pts = expected_points_matrix(sims.finish_pos, mult)
    idx = {c: i for i, c in enumerate(sims.drivers)}
    for i, c in enumerate(sims.drivers):
        model[c]["slot_ev"] = [float(exp_pts[i, s]) for s in range(10)]
    tickets = _ticket_distributions(opt, sims.finish_pos, mult, idx)

    sigma = fit.get("sigma") or {}
    sig_range = (min(sigma.values()), max(sigma.values())) if sigma else (1.0, 1.0)

    body = (
        _header_section(cfg, snapshot, markets)
        + _ticket_section(opt, model)
        + _distribution_section(tickets)
        + _diagnostics_section(fit, opt, calib, sig_range)
        + _markets_section(snapshot, markets, sims, calib)
        + _parameters_section(fit, model)
        + _calibration_section(calib)
        + _win_chart_section(markets, sims, sim_win)
        + _validation_section(markets, sims)
        + _finish_dist_section(sims, opt["best_ticket"], model)
        + _dnf_section(cfg, markets, fit)
        + "<footer>Generated by report.py from the latest snapshot, model "
          "fit and optimise report. Re-run fetch / fit / optimise first to "
          "refresh the inputs.</footer>"
    )
    page = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{_esc(race_name)} — tipping analysis</title>"
        f"<style>{_CSS}</style></head><body><main>{body}</main>"
        f"<script>{_TOOLTIP_JS}</script><script>{_TABS_JS}</script></body></html>"
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
