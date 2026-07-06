"""Kalshi fetch — public REST market data, no account or auth needed.

Kalshi lists one event per race per series: race winner (KXF1RACE),
podium (KXF1RACEPODIUM), top 5 (KXF1TOP5) and top 10 (KXF1TOP10), each
composed of per-driver Yes/No binaries priced in dollars = implied
probability. Quotes are converted to decimal odds (1/p) so snapshots
are indistinguishable from Betfair ones downstream. H2H and
"To be Classified" markets are not offered.

Race events open ~2 weeks before lights out; before that, discovery
fails loudly. ``total_matched`` is recorded in contracts traded (a
$1-scale proxy — the devig liquidity weight only uses its order of
magnitude). Auth is only needed for trading, which this never does.
"""

import _paths  # noqa: F401

from pathlib import Path

from odds.names import runner_to_code
from odds.predmarket import fnum, get_json, runner_from_quotes
from odds.snapshot import save_snapshot

DEFAULT_MAX_SPREAD = 0.15  # widest usable two-sided quote, prob space


def _event_markets(base: str, event_ticker: str) -> list[dict]:
    data = get_json(f"{base}/markets", {"event_ticker": event_ticker, "limit": 100})
    return data.get("markets") or []


def _discover_suffix(base: str, win_series: str, race_name: str) -> str:
    """Find the open race event, return its ticker suffix (e.g. BRIGP26)."""
    data = get_json(f"{base}/events", {"series_ticker": win_series, "limit": 200})
    matches = [e for e in data.get("events", [])
               if race_name.casefold() in (e.get("title") or "").casefold()]
    if not matches:
        raise RuntimeError(
            f"No Kalshi event matches {race_name!r} in series {win_series} — "
            "Kalshi opens race markets ~2 weeks before the race. Check "
            "race.name in config.yaml, or fetch another source."
        )
    for event in matches:
        ticker = event["event_ticker"]
        if any(m.get("status") == "active" for m in _event_markets(base, ticker)):
            print(f"Kalshi event: {event.get('title')} ({ticker})")
            return ticker.split("-", 1)[1]
    raise RuntimeError(
        f"Kalshi events matching {race_name!r} are all closed/settled "
        f"({', '.join(e['event_ticker'] for e in matches[:4])}). Race "
        "markets open ~2 weeks pre-race — try again closer to the weekend."
    )


def parse_event_markets(raw_markets: list[dict], event_ticker: str,
                        config_names: dict[str, str],
                        max_spread: float = DEFAULT_MAX_SPREAD) -> dict | None:
    """Snapshot market entry from one event's per-driver binary markets."""
    runners: dict[str, dict] = {}
    volume = 0.0
    for market in raw_markets:
        if market.get("status") != "active":
            continue
        name = (market.get("yes_sub_title") or "").strip()
        code = runner_to_code(name, config_names)
        if code is None:
            print(f"WARNING: unmatched Kalshi runner {name!r} in "
                  f"{event_ticker}, skipped")
            continue
        volume += fnum(market.get("volume_fp")) or 0.0
        runner = runner_from_quotes(
            market.get("yes_bid_dollars"), market.get("yes_ask_dollars"),
            market.get("last_price_dollars"), max_spread,
        )
        if runner is not None:
            runners[code] = runner
    if not runners:
        return None
    return {
        "market_name": event_ticker,
        "market_id": event_ticker,
        "total_matched": volume or None,
        "runners": runners,
    }


def fetch_kalshi(cfg: dict) -> Path:
    """Discover, price, and snapshot the race's Kalshi market ladder."""
    race_name = cfg["race"]["name"]
    k_cfg = cfg["kalshi"]
    base = k_cfg["api_base"].rstrip("/")
    config_names = cfg["drivers"]["betfair_names"]
    max_spread = float(k_cfg.get("max_spread", DEFAULT_MAX_SPREAD))
    series: dict[str, str] = k_cfg["series"]

    suffix = _discover_suffix(base, series["win"], race_name)

    markets: dict = {"h2h": [], "classified": {}}  # neither offered by Kalshi
    for key, series_ticker in series.items():
        event_ticker = f"{series_ticker}-{suffix}"
        entry = parse_event_markets(_event_markets(base, event_ticker),
                                    event_ticker, config_names, max_spread)
        if entry is None:
            print(f"  {key:<6} {event_ticker}: no active priced runners, skipped")
            continue
        markets[key] = entry
        print(f"  {key:<6} {event_ticker} ({len(entry['runners'])} runners, "
              f"{entry['total_matched'] or 0:,.0f} contracts traded)")

    if "win" not in markets:
        raise RuntimeError(
            f"Kalshi win event {series['win']}-{suffix} has no active "
            "priced runners — refusing to snapshot."
        )

    return save_snapshot(
        {"race_name": race_name, "source": "kalshi", "markets": markets}
    )
