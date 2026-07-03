"""Manual odds fallback — a fully working alternative to Betfair.

Paste odds into a CSV with header ``driver,market,odds``:

    driver,market,odds
    VER,win,3.5
    Max Verstappen,win,3.5        # full names resolve via config/DRIVER_MAP
    VER,top3,1.25
    VER,top10,1.02
    VER,h2h:VER:NOR,1.8           # h2h market label names both drivers
    NOR,h2h:VER:NOR,2.1

Markets: ``win``, ``top3``, ``top6``, ``top10``, and ``h2h:<A>:<B>``
(one row per driver, both required). Blank lines and ``#`` comments are
ignored. The parsed odds are saved as a normal snapshot, so everything
downstream (fit, validate, optimise) is identical to the Betfair path.
"""

import _paths  # noqa: F401

import csv
from pathlib import Path

from odds.names import runner_to_code
from odds.snapshot import save_snapshot

STRUCTURAL_MARKETS = ("win", "top3", "top6", "top10")


def parse_csv(path: Path, race_name: str, config_names: dict[str, str]) -> dict:
    """Parse a manual odds CSV into an (unsaved) snapshot dict."""
    structural: dict[str, dict[str, dict]] = {}
    h2h_markets: dict[str, dict[str, dict]] = {}
    skipped: list[str] = []

    with open(path, newline="", encoding="utf-8-sig") as fh:
        for line_no, row in enumerate(csv.DictReader(fh), start=2):
            driver_raw = (row.get("driver") or "").split("#")[0].strip()
            market = (row.get("market") or "").split("#")[0].strip().lower()
            odds_raw = (row.get("odds") or "").split("#")[0].strip()
            if not driver_raw and not market:
                continue

            code = runner_to_code(driver_raw, config_names)
            if code is None:
                skipped.append(f"line {line_no}: unknown driver {driver_raw!r}")
                continue
            try:
                odds = float(odds_raw)
            except ValueError:
                raise ValueError(f"line {line_no}: bad odds {odds_raw!r}") from None
            if odds <= 1.0:
                raise ValueError(f"line {line_no}: odds must be > 1.0, got {odds}")

            runner = {"last_traded": odds, "back": None, "lay": None}
            if market in STRUCTURAL_MARKETS:
                structural.setdefault(market, {})[code] = runner
            elif market.startswith("h2h:"):
                h2h_markets.setdefault(market, {})[code] = runner
            else:
                raise ValueError(
                    f"line {line_no}: unknown market {market!r} (expected one of "
                    f"{STRUCTURAL_MARKETS} or h2h:<A>:<B>)"
                )

    if skipped:
        print("WARNING: skipped rows with unmatched drivers:")
        for msg in skipped:
            print(f"  {msg}")

    markets: dict = {
        key: {
            "market_name": f"manual {key}",
            "market_id": None,
            "total_matched": None,
            "runners": runners,
        }
        for key, runners in structural.items()
    }
    markets["h2h"] = [
        {
            "market_name": f"manual {label}",
            "market_id": None,
            "total_matched": None,
            "runners": runners,
        }
        for label, runners in h2h_markets.items()
        if len(runners) == 2
    ]
    for label, runners in h2h_markets.items():
        if len(runners) != 2:
            print(f"WARNING: h2h market {label!r} has {len(runners)} priced runner(s), skipped")

    if "win" not in markets:
        raise RuntimeError(f"Manual CSV {path} contains no 'win' market rows")

    return {"race_name": race_name, "source": "manual", "markets": markets}


def fetch_manual(path: str | Path, race_name: str, config_names: dict[str, str]) -> Path:
    """Parse a manual CSV and cache it as a timestamped snapshot."""
    snapshot = parse_csv(Path(path), race_name, config_names)
    n_structural = [k for k in STRUCTURAL_MARKETS if k in snapshot["markets"]]
    print(f"Parsed manual odds: markets={n_structural}, h2h={len(snapshot['markets']['h2h'])}")
    return save_snapshot(snapshot)
