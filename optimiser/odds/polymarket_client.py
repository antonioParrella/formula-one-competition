"""Polymarket fetch — public Gamma API, no account or auth needed.

Polymarket lists one event per race per market type ("<Race>: Driver
Winner", "<Race>: Driver Podium Finish", "<Race>: Head-to-Head"), each
holding binary markets priced 0-1 = implied probability; quotes are
converted to decimal odds (1/p) for the standard snapshot. Winner books
go live weeks out and are tight; podium/H2H books usually sit on
placeholder quotes (0.02/0.98) until near the weekend — the spread
guard drops those, so early snapshots may carry the win market only.
Top-5/6/10 and "To be Classified" markets are not offered.
``total_matched`` is USD traded volume.
"""

import _paths  # noqa: F401

import json
import re
from pathlib import Path

from odds.names import runner_to_code
from odds.predmarket import fnum, get_json, runner_from_quotes
from odds.snapshot import save_snapshot

DEFAULT_MAX_SPREAD = 0.15  # widest usable two-sided quote, prob space
# "Driver A" / "another driver" filler outcomes — skipped without warning.
_PLACEHOLDER = re.compile(r"^(driver [a-z]|another driver|other)$")


def parse_binary_event(event: dict, config_names: dict[str, str],
                       max_spread: float = DEFAULT_MAX_SPREAD) -> dict | None:
    """Snapshot entry from a per-driver Yes/No event (winner, podium)."""
    runners: dict[str, dict] = {}
    for market in event.get("markets", []):
        if market.get("closed") or market.get("active") is False:
            continue
        name = (market.get("groupItemTitle") or "").strip()
        if not name or _PLACEHOLDER.match(name.casefold()):
            continue
        code = runner_to_code(name, config_names)
        if code is None:
            print(f"WARNING: unmatched Polymarket runner {name!r} in "
                  f"{event.get('title')!r}, skipped")
            continue
        runner = runner_from_quotes(market.get("bestBid"), market.get("bestAsk"),
                                    market.get("lastTradePrice"), max_spread)
        if runner is not None:
            runners[code] = runner
    if not runners:
        return None
    return {
        "market_name": event.get("title"),
        "market_id": event.get("slug"),
        "total_matched": fnum(event.get("volume")),
        "runners": runners,
    }


def _complement(p: float | None) -> float | None:
    return None if p is None else 1.0 - p


def parse_h2h_market(market: dict, config_names: dict[str, str],
                     max_spread: float = DEFAULT_MAX_SPREAD) -> dict | None:
    """Snapshot h2h entry from a two-outcome "who finishes higher" market.

    Quotes reference the first outcome; the second is its complement.
    """
    if market.get("closed") or market.get("active") is False:
        return None
    try:
        outcomes = json.loads(market.get("outcomes") or "[]")
    except ValueError:
        return None
    if len(outcomes) != 2:
        return None
    code_a = runner_to_code(str(outcomes[0]), config_names)
    code_b = runner_to_code(str(outcomes[1]), config_names)
    if code_a is None or code_b is None or code_a == code_b:
        print(f"WARNING: unmatched Polymarket h2h outcomes {outcomes} in "
              f"{market.get('question')!r}, skipped")
        return None
    bid, ask = fnum(market.get("bestBid")), fnum(market.get("bestAsk"))
    last = fnum(market.get("lastTradePrice"))
    runner_a = runner_from_quotes(bid, ask, last, max_spread)
    runner_b = runner_from_quotes(_complement(ask), _complement(bid),
                                  _complement(last), max_spread)
    if runner_a is None or runner_b is None:
        return None
    return {
        "market_name": market.get("question"),
        "market_id": market.get("slug") or market.get("id"),
        "total_matched": fnum(market.get("volume")),
        "runners": {code_a: runner_a, code_b: runner_b},
    }


def fetch_polymarket(cfg: dict) -> Path:
    """Discover, price, and snapshot the race's Polymarket events."""
    race_name = cfg["race"]["name"]
    pm_cfg = cfg["polymarket"]
    base = pm_cfg["api_base"].rstrip("/")
    config_names = cfg["drivers"]["betfair_names"]
    max_spread = float(pm_cfg.get("max_spread", DEFAULT_MAX_SPREAD))
    include_sprint = cfg["race"].get("include_sprint_markets", False)

    events = get_json(f"{base}/events", {
        "tag_slug": pm_cfg.get("tag_slug", "f1"), "closed": "false",
        "limit": 100,
    })
    picked: dict[str, dict] = {}
    for event in events:
        title = (event.get("title") or "").casefold()
        if race_name.casefold() not in title:
            continue
        if "sprint" in title and not include_sprint:
            continue
        for key, fragments in pm_cfg["market_titles"].items():
            if key not in picked and any(f.casefold() in title for f in fragments):
                picked[key] = event

    if "win" not in picked:
        open_gp = sorted(e.get("title", "?") for e in events
                         if "grand prix" in (e.get("title") or "").casefold())
        raise RuntimeError(
            f"Polymarket has no open winner event matching {race_name!r} "
            f"(tag {pm_cfg.get('tag_slug', 'f1')!r}). Open GP events: "
            + (", ".join(open_gp[:8]) or "none")
        )

    markets: dict = {"h2h": [], "classified": {}}  # classified not offered
    for key, event in picked.items():
        # Re-fetch by slug: the tag listing may return trimmed market objects.
        full = get_json(f"{base}/events", {"slug": event["slug"]})
        full = full[0] if full else event
        if key == "h2h":
            entries = [parse_h2h_market(m, config_names, max_spread)
                       for m in full.get("markets", [])]
            markets["h2h"] = [e for e in entries if e]
            print(f"  h2h    {full.get('title')} ({len(markets['h2h'])}/"
                  f"{len(full.get('markets', []))} matchups priced)")
        else:
            entry = parse_binary_event(full, config_names, max_spread)
            if entry is None:
                print(f"  {key:<6} {full.get('title')}: no priced runners "
                      "(placeholder books), skipped")
                continue
            markets[key] = entry
            print(f"  {key:<6} {full.get('title')} "
                  f"({len(entry['runners'])} runners, "
                  f"${entry['total_matched'] or 0:,.0f} traded)")

    if "win" not in markets:
        raise RuntimeError(
            f"Polymarket winner event {picked['win'].get('slug')!r} has no "
            "priced runners — refusing to snapshot."
        )

    return save_snapshot(
        {"race_name": race_name, "source": "polymarket", "markets": markets}
    )
