"""Odds snapshot cache.

Every fetch (Betfair or manual CSV) is written to ``data/`` as a
timestamped JSON snapshot before anything else happens. All downstream
steps (fit, validate, optimise) read the latest snapshot, never live
prices, so runs are reproducible.

Snapshot shape::

    {
      "race_name": "Silverstone",
      "source": "betfair" | "manual",
      "fetched_at": "2026-07-03T10:15:00+00:00",
      "markets": {
        "win":   {"market_name": ..., "market_id": ..., "total_matched": ...,
                  "runners": {"VER": {"last_traded": 3.5, "back": 3.45,
                                       "lay": 3.55}, ...}},
        "top3":  {...}, "top6": {...}, "top10": {...},
        "h2h":   [{"market_name": ..., "market_id": ..., "total_matched": ...,
                   "runners": {"VER": {...}, "NOR": {...}}}, ...]
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
    """Write a timestamped snapshot JSON and return its path."""
    now = datetime.now(timezone.utc)
    snapshot = {**snapshot, "fetched_at": now.isoformat()}
    data_dir.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    path = data_dir / f"odds_{race_slug(snapshot['race_name'])}_{stamp}.json"
    path.write_text(json.dumps(snapshot, indent=2))
    print(f"Snapshot saved -> {path}")
    return path


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
    pattern = f"odds_{race_slug(race_name)}_*.json"
    matches = sorted(data_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No odds snapshot matching {pattern} in {data_dir}. "
            "Run `python main.py fetch` (or fetch --manual <csv>) first."
        )
    path = matches[-1]
    snapshot = json.loads(path.read_text())

    fetched_at = datetime.fromisoformat(snapshot["fetched_at"])
    age_h = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 3600
    if age_h > MAX_AGE_HOURS and not allow_stale:
        raise RuntimeError(
            f"Latest snapshot {path.name} is {age_h:.1f}h old (> "
            f"{MAX_AGE_HOURS:.0f}h). Re-fetch close to the comp deadline, "
            "or pass --allow-stale."
        )
    print(f"Using snapshot {path.name} ({age_h:.1f}h old)")
    return snapshot
