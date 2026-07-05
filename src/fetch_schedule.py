"""
fetch_schedule.py
─────────────────────────────────────────────────────────────
Fetches the full session schedule (Practice / Qualifying /
Sprint / Race start times) for a single round from OpenF1 and
saves raw JSON used by the scorer to detect late submissions.

Round numbers come from the same cancellation-aware renumbering
used everywhere else (get_race_calendar), so a schedule file
lines up with its tips/results/scored files by the r{NN}_ prefix.

Files are write-once: if the schedule for a round already exists
it is not re-fetched unless force=True.

Usage:
    from fetch_schedule import ScheduleFetcher

    ScheduleFetcher(round_num=1, year=2026).fetch_and_save()
"""

import json
from datetime import datetime
from pathlib import Path

from leaderboard import get_race_calendar, _fetch_openf1
from race_utils import clean_race_name

# Session order used to sort the saved schedule chronologically as a
# fallback when start times are missing.
_SESSION_ORDER = {
    "Practice 1": 0,
    "Practice 2": 1,
    "Practice 3": 2,
    "Sprint Qualifying": 3,
    "Sprint": 4,
    "Qualifying": 5,
    "Race": 6,
}


class ScheduleFetcher:
    """Fetches the OpenF1 session schedule for a single round.

    Parameters
    ----------
    round_num : int
        Internal (cancellation-aware) round number, 1-based.
    year : int
        Season year, defaults to 2026.
    output_dir : str
        Directory to write raw schedule JSON files to.
    """

    def __init__(
        self,
        round_num: int,
        year: int = 2026,
        output_dir: str = "data/raw/schedule",
    ):
        self.round_num = round_num
        self.year = year
        self.output_dir = Path(output_dir)

    # ─────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────

    def _find_existing(self) -> Path | None:
        pattern = f"r{self.round_num:02d}_*_schedule.json"
        matches = list(self.output_dir.glob(pattern))
        return matches[0] if matches else None

    def fetch_and_save(self, force: bool = False) -> Path:
        """Fetch the schedule and write it to raw JSON.

        Returns the path to the saved (or already-present) file.
        """
        existing = self._find_existing()
        if existing is not None and not force:
            print(f"Schedule already saved ({self.round_num}): {existing.name}")
            return existing

        schedule = self._build_schedule()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        slug = schedule["race_name"].lower().replace(" ", "_")
        out_path = self.output_dir / f"r{self.round_num:02d}_{slug}_schedule.json"
        out_path.write_text(json.dumps(schedule, indent=2))
        print(f"Saved schedule ({self.round_num}): {schedule['race_name']} -> {out_path}")

        return out_path

    # ─────────────────────────────────────────────────────────
    # Building the schedule
    # ─────────────────────────────────────────────────────────

    def _resolve_meeting(self) -> tuple[int, str]:
        """Resolve (meeting_key, race_name) for this round via the calendar."""
        calendar = get_race_calendar(self.year)
        race_row = calendar[
            (calendar["round_number"] == self.round_num) &
            (calendar["session_name"] == "Race")
        ]
        if race_row.empty:
            raise ValueError(
                f"No race found for round {self.round_num}, year {self.year}"
            )
        meeting_key = int(race_row.iloc[0]["meeting_key"])
        race_name = clean_race_name(race_row.iloc[0]["circuit_short_name"].strip())
        return meeting_key, race_name

    def _build_schedule(self) -> dict:
        meeting_key, race_name = self._resolve_meeting()

        # All sessions for this meeting (Practice/Qualifying/Sprint/Race).
        sessions_df = _fetch_openf1("sessions", year=self.year, meeting_key=meeting_key)

        sessions = []
        for _, row in sessions_df.iterrows():
            sessions.append({
                "session_name": row.get("session_name"),
                "session_key": _as_int(row.get("session_key")),
                "date_start": row.get("date_start"),
                "date_end": row.get("date_end"),
                "gmt_offset": row.get("gmt_offset"),
                "is_cancelled": bool(row.get("is_cancelled", False)),
            })

        sessions.sort(key=lambda s: (
            s["date_start"] or "",
            _SESSION_ORDER.get(s["session_name"], 99),
        ))

        is_sprint_weekend = any(s["session_name"] == "Sprint" for s in sessions)

        return {
            "round": self.round_num,
            "race_name": race_name,
            "year": self.year,
            "meeting_key": meeting_key,
            "is_sprint_weekend": is_sprint_weekend,
            "fetched_at": datetime.now().isoformat(),
            "sessions": sessions,
        }


def _as_int(value):
    """Best-effort int conversion that tolerates NaN / None."""
    try:
        if value is None:
            return None
        return int(value)
    except (ValueError, TypeError):
        return None
