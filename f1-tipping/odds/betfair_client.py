"""Betfair Exchange fetch — auth + market discovery + prices.

Uses ``betfairlightweight`` (JSON-RPC Betting API). Credentials are read
from ``~/.betfair/credentials.yaml`` (see config.yaml); nothing is ever
hardcoded or committed. Markets are discovered fresh each run via
``listMarketCatalogue`` (IDs vary per race weekend), priced with
``listMarketBook`` / EX_BEST_OFFERS, and cached to ``data/`` as a
snapshot before anything else runs.

If ``betfairlightweight`` is not installed or Betfair is geo-blocked,
use ``manual_input.py`` — the rest of the pipeline is identical.
"""

import _paths  # noqa: F401

from pathlib import Path

import yaml

from odds.names import runner_to_code
from odds.snapshot import save_snapshot

BOOK_CHUNK = 25  # listMarketBook request weight limit headroom


def _connect(credentials_path: str):
    try:
        import betfairlightweight
    except ImportError:
        raise RuntimeError(
            "betfairlightweight is not installed — `pip install "
            "betfairlightweight`, or use `python main.py fetch --manual "
            "<csv>` instead."
        ) from None

    creds_file = Path(credentials_path).expanduser()
    if not creds_file.exists():
        raise FileNotFoundError(
            f"Betfair credentials not found at {creds_file}. Expected a YAML "
            "with username, password, app_key, certs_path."
        )
    creds = yaml.safe_load(creds_file.read_text())

    client = betfairlightweight.APIClient(
        username=creds["username"],
        password=creds["password"],
        app_key=creds["app_key"],
        certs=str(Path(creds["certs_path"]).expanduser()),
    )
    client.login()
    return client


def _find_event(client, event_type_id: str, race_name: str):
    from betfairlightweight import filters

    results = client.betting.list_events(
        filter=filters.market_filter(
            event_type_ids=[event_type_id], text_query=race_name
        )
    )
    candidates = [r.event for r in results]
    if not candidates:
        raise RuntimeError(
            f"No Betfair events matched {race_name!r} (eventTypeId "
            f"{event_type_id}). Check race.name in config.yaml."
        )
    # Prefer the grand prix event proper over support categories.
    candidates.sort(
        key=lambda e: ("grand prix" in e.name.lower() or "f1" in e.name.lower()),
        reverse=True,
    )
    event = candidates[0]
    print(f"Event: {event.name} (id {event.id}, open {event.open_date})")
    if len(candidates) > 1:
        print(f"  ({len(candidates) - 1} other candidate(s): "
              f"{', '.join(e.name for e in candidates[1:4])})")
    return event


def _classify(market_name: str, market_names_cfg: dict[str, list[str]],
              include_sprint: bool) -> str | None:
    """Map a catalogue market name to win/top3/top6/top10/classified/h2h,
    else None."""
    lowered = market_name.lower()
    if "sprint" in lowered and not include_sprint:
        return None
    for key, fragments in market_names_cfg.items():
        if any(frag.lower() in lowered for frag in fragments):
            return key
    if " v " in f" {lowered} ":
        return "h2h"
    return None


def fetch_betfair(cfg: dict) -> Path:
    """Discover, price, and snapshot all relevant markets for the race."""
    from betfairlightweight import filters

    race_name = cfg["race"]["name"]
    bf_cfg = cfg["betfair"]
    config_names = cfg["drivers"]["betfair_names"]

    client = _connect(bf_cfg["credentials_path"])
    try:
        event = _find_event(client, bf_cfg["event_type_id"], race_name)

        catalogues = client.betting.list_market_catalogue(
            filter=filters.market_filter(event_ids=[event.id]),
            max_results="200",
            market_projection=["RUNNER_DESCRIPTION", "MARKET_START_TIME"],
        )

        by_key: dict[str, list] = {}
        for cat in catalogues:
            key = _classify(
                cat.market_name,
                bf_cfg["market_names"],
                cfg["race"].get("include_sprint_markets", False),
            )
            if key:
                by_key.setdefault(key, []).append(cat)

        if "win" not in by_key:
            raise RuntimeError(
                "No win market found — markets discovered: "
                + ", ".join(sorted(c.market_name for c in catalogues))
            )

        # One market per structural key (most matched wins); all h2h and
        # per-driver classified markets.
        chosen: dict[str, list] = {}
        for key, cats in by_key.items():
            cats.sort(key=lambda c: c.total_matched or 0, reverse=True)
            chosen[key] = cats if key in ("h2h", "classified") else cats[:1]

        all_cats = [c for cats in chosen.values() for c in cats]
        books = {}
        for i in range(0, len(all_cats), BOOK_CHUNK):
            chunk = all_cats[i : i + BOOK_CHUNK]
            for book in client.betting.list_market_book(
                market_ids=[c.market_id for c in chunk],
                price_projection=filters.price_projection(
                    price_data=["EX_BEST_OFFERS"]
                ),
            ):
                books[book.market_id] = book

        markets: dict = {"h2h": [], "classified": {}}
        for key, cats in chosen.items():
            for cat in cats:
                label = key
                if key == "classified":
                    # "Yes To be Classified" / "No To be Classified" are
                    # driver-runner markets; keep the most-matched per side.
                    side = _classified_side(cat.market_name)
                    if side in markets["classified"]:
                        continue
                    label = f"classified[{side}]"
                entry = _market_entry(cat, books.get(cat.market_id), config_names)
                if entry is None:
                    continue
                if key == "h2h":
                    markets["h2h"].append(entry)
                elif key == "classified":
                    markets["classified"][side] = entry
                else:
                    markets[key] = entry
                print(f"  {label:<6} {cat.market_name} "
                      f"({len(entry['runners'])} runners, "
                      f"matched £{cat.total_matched or 0:,.0f})")
    finally:
        client.logout()

    snapshot = {"race_name": race_name, "source": "betfair", "markets": markets}
    return save_snapshot(snapshot)


def _market_entry(cat, book, config_names: dict[str, str]) -> dict | None:
    """Build a snapshot market entry from a catalogue + book pair."""
    if book is None:
        print(f"WARNING: no book returned for {cat.market_name}, skipped")
        return None

    name_by_selection = {r.selection_id: r.runner_name for r in cat.runners}
    runners: dict[str, dict] = {}
    for rb in book.runners:
        runner_name = name_by_selection.get(rb.selection_id, "")
        code = runner_to_code(runner_name, config_names)
        if code is None:
            print(f"WARNING: unmatched runner {runner_name!r} in "
                  f"{cat.market_name}, skipped")
            continue
        backs = rb.ex.available_to_back if rb.ex else []
        lays = rb.ex.available_to_lay if rb.ex else []
        runners[code] = {
            "last_traded": rb.last_price_traded,
            "back": backs[0].price if backs else None,
            "lay": lays[0].price if lays else None,
        }

    if not runners:
        return None
    return {
        "market_name": cat.market_name,
        "market_id": cat.market_id,
        "total_matched": cat.total_matched,
        "runners": runners,
    }


def _classified_side(market_name: str) -> str:
    """Which side of the classification event a market prices.

    Betfair lists "Yes To be Classified" and "No To be Classified" as
    separate driver-runner markets; backing a driver in the "No" market
    is backing the DNF. An unprefixed name defaults to the Yes side.
    """
    first = market_name.lower().split()[0] if market_name.split() else ""
    return "no" if first == "no" else "yes"
