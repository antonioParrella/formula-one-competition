"""Cross-source odds combination.

Merges single-source snapshots (betfair / kalshi / polymarket / manual)
into one snapshot with source "combined". Per structural market each
source is de-vigged separately (removing its own overround), then
per-driver probabilities are averaged across sources weighted by each
market's liquidity weight, and written back as decimal odds (1/p).
Downstream de-vigging of the combined market is then a near no-op, so
fit/validate/optimise run unchanged.

``total_matched`` units differ per source (GBP, USD, contracts) but the
liquidity weight only uses the order of magnitude, where they agree.

H2H matchup markets are concatenated — each is an independent fit
target, so two sources pricing the same pair simply count as two pieces
of evidence, each at its own liquidity weight. "To be Classified" sides
are merged per driver, most liquid source first.
"""

import _paths  # noqa: F401

import json
from pathlib import Path

from odds.devig import (
    TOPK_BY_MARKET,
    _market_weight,
    devig_topn,
    devig_win,
    select_price,
)
from odds.snapshot import list_snapshots, load_snapshot, save_snapshot


def combine_snapshots(
    snapshots: list[dict],
    win_method: str = "proportional",
    topn_method: str = "power",
) -> dict:
    """Merge >= 2 single-source snapshots into a combined snapshot dict."""
    if len(snapshots) < 2:
        raise ValueError("Need at least two snapshots to combine")
    races = {s.get("race_name") for s in snapshots}
    if len(races) > 1:
        raise ValueError(f"Snapshots are for different races: {sorted(races)}")
    sources = [s.get("source", "?") for s in snapshots]

    markets: dict = {"h2h": [], "classified": {}}

    for key, k in TOPK_BY_MARKET.items():
        weighted: dict[str, float] = {}
        weight_sum: dict[str, float] = {}
        contributors: list[str] = []
        total_matched = 0.0
        for snap in snapshots:
            market = (snap.get("markets") or {}).get(key)
            if not market:
                continue
            odds = {code: p for code, runner in market.get("runners", {}).items()
                    if (p := select_price(runner)) is not None}
            if not odds or (k > 1 and len(odds) < k):
                continue
            probs = (devig_win(odds, win_method) if k == 1
                     else devig_topn(odds, k, topn_method))
            weight = _market_weight(market.get("total_matched"))
            for code, p in probs.items():
                weighted[code] = weighted.get(code, 0.0) + weight * p
                weight_sum[code] = weight_sum.get(code, 0.0) + weight
            contributors.append(snap.get("source", "?"))
            total_matched += market.get("total_matched") or 0.0
        if not weighted:
            continue
        markets[key] = {
            "market_name": f"combined {key} ({'+'.join(contributors)})",
            "market_id": None,
            "total_matched": total_matched or None,
            "runners": {
                code: {"last_traded": weight_sum[code] / weighted[code],
                       "back": None, "lay": None}
                for code in weighted
            },
        }

    for snap in snapshots:
        for entry in (snap.get("markets") or {}).get("h2h") or []:
            markets["h2h"].append({
                **entry,
                "market_name": f"[{snap.get('source', '?')}] "
                               f"{entry.get('market_name')}",
            })

    for side in ("yes", "no"):
        offered = [
            (s.get("source", "?"),
             ((s.get("markets") or {}).get("classified") or {}).get(side))
            for s in snapshots
        ]
        offered = [(src, entry) for src, entry in offered if entry]
        if not offered:
            continue
        offered.sort(key=lambda t: t[1].get("total_matched") or 0.0, reverse=True)
        runners: dict[str, dict] = {}
        for _src, entry in offered:
            for code, runner in entry.get("runners", {}).items():
                runners.setdefault(code, runner)
        markets["classified"][side] = {
            "market_name": (f"combined classified {side} "
                            f"({'+'.join(src for src, _ in offered)})"),
            "market_id": None,
            "total_matched": offered[0][1].get("total_matched"),
            "runners": runners,
        }

    if "win" not in markets:
        raise RuntimeError(
            f"No usable win market across sources ({'+'.join(sources)}) — "
            "nothing to combine."
        )

    return {
        "race_name": snapshots[0]["race_name"],
        "source": "combined",
        "combined_from": [{"source": s.get("source", "?"),
                           "fetched_at": s.get("fetched_at")}
                          for s in snapshots],
        "markets": markets,
    }


def combine_latest(
    race_name: str,
    win_method: str = "proportional",
    topn_method: str = "power",
    allow_stale: bool = False,
) -> Path:
    """Combine the latest archived snapshot of each fetch source."""
    latest: dict[str, Path] = {}
    for path in list_snapshots(race_name):
        source = json.loads(path.read_text()).get("source")
        if source and source != "combined":
            latest[source] = path  # list is oldest-first, so last wins
    if len(latest) < 2:
        raise RuntimeError(
            f"Combining needs snapshots from >= 2 sources for {race_name!r}; "
            f"found {sorted(latest) or 'none'}. Fetch more sources first "
            "(python main.py fetch --source all)."
        )
    snapshots = [load_snapshot(path, allow_stale=allow_stale)
                 for _src, path in sorted(latest.items())]
    combined = combine_snapshots(snapshots, win_method, topn_method)
    print(f"Combined sources: {', '.join(sorted(latest))}")
    return save_snapshot(combined)
