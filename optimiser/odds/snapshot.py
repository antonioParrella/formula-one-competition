"""Odds snapshot cache.

Every fetch (Betfair, Kalshi, Polymarket, or manual CSV) is written to
``data/`` as a timestamped JSON snapshot before anything else happens.
All downstream steps (fit, validate, optimise) read the latest
snapshot, never live prices, so runs are reproducible.

Snapshot shape::

    {
      "race_name": "Silverstone",
      "source": "betfair" | "kalshi" | "polymarket" | "manual" | "combined",
      "fetched_at": "2026-07-03T10:15:00+00:00",
      "markets": {
        "win":   {"market_name": ..., "market_id": ..., "total_matched": ...,
                  "runners": {"VER": {"last_traded": 3.5, "back": 3.45,
                                       "lay": 3.55}, ...}},
        "top3":  {...}, "top5": {...}, "top6": {...}, "top10": {...},
        "h2h":   [{"market_name": ..., "market_id": ..., "total_matched": ...,
                   "runners": {"VER": {...}, "NOR": {...}}}, ...],
        "classified": {"yes": {"market_name": "Yes To be Classified",
                               "market_id": ..., "total_matched": ...,
                               "runners": {"VER": {...}, ...}},
                       "no": {...}}   # each side only if listed
      }
    }
"""

import _paths  # noqa: F401

import json
from datetime import datetime, timezone
from pathlib import Path

from race_utils import clean_race_name

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAX_AGE_HOURS = 24.0


def race_slug(race_name: str) -> str:
    return clean_race_name(race_name).lower().replace(" ", "_")


def save_snapshot(snapshot: dict, data_dir: Path = DATA_DIR) -> Path:
    """Write a timestamped snapshot JSON and return its path.

    Every query is archived and none is ever overwritten: if a snapshot
    with the same second-resolution timestamp already exists (two fetches
    in one second), a counter suffix keeps both.
    """
    now = datetime.now(timezone.utc)
    snapshot = {**snapshot, "fetched_at": now.isoformat()}
    data_dir.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    slug = race_slug(snapshot["race_name"])
    path = data_dir / f"odds_{slug}_{stamp}.json"
    counter = 1
    while path.exists():
        path = data_dir / f"odds_{slug}_{stamp}_{counter}.json"
        counter += 1
    path.write_text(json.dumps(snapshot, indent=2))
    print(f"Snapshot saved -> {path}")
    return path


def list_snapshots(race_name: str | None = None, data_dir: Path = DATA_DIR) -> list[Path]:
    """All archived odds snapshots (optionally for one race), oldest first."""
    pattern = (f"odds_{race_slug(race_name)}_*.json" if race_name
               else "odds_*.json")
    return sorted(data_dir.glob(pattern))


def load_snapshot(path: str | Path, allow_stale: bool = True) -> dict:
    """Load a specific archived snapshot by path (name resolved under
    ``data/`` if relative). Staleness is allowed by default — reloading
    an old archived snapshot on purpose is the whole point."""
    path = Path(path)
    if not path.is_absolute():
        path = DATA_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"No snapshot at {path}")
    snapshot = json.loads(path.read_text())
    _report_age(path, snapshot, allow_stale)
    return snapshot


def load_latest_snapshot(
    race_name: str,
    data_dir: Path = DATA_DIR,
    allow_stale: bool = False,
) -> dict:
    """Load the most recent snapshot for a race.

    Fails loudly if none exists or the latest is older than
    ``MAX_AGE_HOURS`` (odds move sharply with grid penalties and driver
    changes) — pass ``allow_stale=True`` to override.
    """
    matches = list_snapshots(race_name, data_dir)
    if not matches:
        raise FileNotFoundError(
            f"No odds snapshot for {race_name!r} in {data_dir}. "
            "Run `python main.py fetch` (or fetch --manual <csv>) first."
        )
    path = matches[-1]
    snapshot = json.loads(path.read_text())
    _report_age(path, snapshot, allow_stale)
    return snapshot


def _report_age(path: Path, snapshot: dict, allow_stale: bool) -> None:
    fetched_at = datetime.fromisoformat(snapshot["fetched_at"])
    age_h = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 3600
    if age_h > MAX_AGE_HOURS and not allow_stale:
        raise RuntimeError(
            f"Snapshot {path.name} is {age_h:.1f}h old (> {MAX_AGE_HOURS:.0f}h). "
            "Re-fetch close to the comp deadline, or pass --allow-stale."
        )
    print(f"Using snapshot {path.name} ({age_h:.1f}h old)")
