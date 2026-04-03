"""
fetch_results.py
─────────────────────────────────────────────────────────────
Fetches race (and sprint) results from the OpenF1 API via
ResultAggregator and saves raw JSON for the scoring pipeline.

The race name is automatically resolved from the circuit
short name in the OpenF1 calendar — no manual name needed.

Usage:
    from fetch_results import ResultsFetcher

    fetcher = ResultsFetcher(round_num=1, year=2026)
    fetcher.fetch_and_save()
"""

import json
from pathlib import Path

from leaderboard import ResultAggregator
from race_utils import clean_race_name


class ResultsFetcher:
    """Fetches OpenF1 results for a single round and saves them
    in a format ready for the scoring pipeline.

    Parameters
    ----------
    round_num : int
        Race round number (1-based).
    year : int
        Season year, defaults to 2026.
    output_dir : str
        Directory to write raw results JSON files to.
    """

    def __init__(
        self,
        round_num: int,
        year: int = 2026,
        output_dir: str = "data/raw/results",
    ):
        self.round_num = round_num
        self.year = year
        self.output_dir = Path(output_dir)
        self.aggregator = ResultAggregator(round_number=round_num, year=year)
        self.race_name = clean_race_name(self._resolve_race_name())

    # ─────────────────────────────────────────────────────────
    # Resolution
    # ─────────────────────────────────────────────────────────

    def _resolve_race_name(self) -> str:
        """Look up the race name from the OpenF1 calendar by round number."""
        calendar = self.aggregator.race_calendar
        race_row = calendar[
            (calendar["round_number"] == self.round_num) &
            (calendar["session_name"] == "Race")
        ]
        if race_row.empty:
            raise ValueError(f"No race found for round {self.round_num}, year {self.year}")
        circuit = race_row.iloc[0]["circuit_short_name"]
        return circuit.strip()

    # ─────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────

    def fetch_and_save(self) -> Path:
        """Extract results from OpenF1 and write to raw JSON.

        Returns the path to the saved file.
        """
        results = self._build_results()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        slug = self.race_name.lower().replace(" ", "_")
        out_path = self.output_dir / f"r{self.round_num:02d}_{slug}_result.json"
        out_path.write_text(json.dumps(results, indent=2))
        print(f"Saved results ({self.round_num}): {self.race_name} → {out_path}")

        return out_path

    # ─────────────────────────────────────────────────────────
    # Building results
    # ─────────────────────────────────────────────────────────

    def _build_results(self) -> dict:
        """Assemble the complete results payload."""
        return {
            "round":                     self.round_num,
            "race_name":                 self.race_name,
            "year":                      self.year,
            "is_sprint_weekend":         self._is_sprint_weekend(),
            "top10":                     self._get_top10(),
            "dnfs":                      self._get_dnfs(),
            "sprint_top3":               self._get_sprint_top3(),
            "championship_top10_before_race": self._get_championship_top10(),
        }

    # ─────────────────────────────────────────────────────────
    # Extractors
    # ─────────────────────────────────────────────────────────

    def _get_top10(self) -> list[str]:
        """Top 10 finishers by position."""
        top10_df = self.aggregator._get_position_scores(
            self.aggregator.race_results, top_k=10
        )
        return top10_df["name_acronym"].tolist()

    def _get_dnfs(self) -> list[str]:
        """Drivers who started the race but didn't finish (dnf=true)."""
        race_results = self.aggregator.race_results
        dnf_numbers = {
            r["driver_number"]
            for r in race_results
            if r.get("dnf") is True
        }
        driver_codes = {r["driver_number"]: r["name_acronym"] for r in race_results}
        return [driver_codes[n] for n in dnf_numbers]

    def _get_sprint_top3(self) -> list[str] | None:
        """Top 3 from the sprint race, or None if no sprint this round."""
        if self.aggregator.sprint_results is None:
            return None
        sprint_df = self.aggregator._get_position_scores(
            self.aggregator.sprint_results, top_k=3
        )
        return sprint_df["name_acronym"].tolist()

    def _is_sprint_weekend(self) -> bool:
        return self.aggregator.sprint_results is not None

    def _get_championship_top10(self) -> list[str]:
        """Top 10 in championship standings before this round."""
        standings = self.aggregator.championship_standings.head(10)
        return standings["name_acronym"].tolist()
