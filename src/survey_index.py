"""
survey_index.py
─────────────────────────────────────────────────────────────
Fetches all surveys from SurveyMars, matches them to the
correct F1 round numbers by cross-referencing survey titles
against the OpenF1 race calendar (skipping cancelled rounds),
and falls back to publish-date ordering for unmatched surveys.

Usage:
    from surveymars_client import SurveyMarsClient
    from survey_index import SurveyIndex
    from tips_parser import TipsParser

    client = SurveyMarsClient()
    client.authenticate()

    index  = SurveyIndex(client).fetch()
    parser = TipsParser(client)

    index.print_summary()

    for survey in index.all():
        parser.fetch_and_save(
            survey_id = survey["survey_id"],
            round_num = survey["round_num"],
            race_name = survey["title"],
        )
"""

import json
from pathlib import Path

from race_utils import clean_race_name


class SurveyIndex:
    """
    Fetches and indexes all SurveyMars surveys, matching them
    to the correct F1 round numbers via the OpenF1 calendar.

    Each survey in the index looks like:
    {
        "round_num":      1,
        "survey_id":      "7UogehrrK",
        "title":          "Melbourne GP",
        "status":         2,              # 1 = active, 2 = closed
        "published_at":   "2026-03-03T21:48:25",
        "response_count": 11,
    }
    """

    STATUS_ACTIVE = 1
    STATUS_CLOSED = 2

    def __init__(self, client):
        self.client   = client
        self._surveys = []

    # ─────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────

    def fetch(self) -> "SurveyIndex":
        """
        Fetch all surveys from the API, match each one to the
        correct F1 round number via the OpenF1 calendar, and
        fall back to publish-date order for unmatched surveys.

        Returns self to allow chaining:
            index = SurveyIndex(client).fetch()
        """
        print("Fetching survey list from SurveyMars...")
        raw = self._fetch_all_surveys()["data"]

        published = [s for s in raw if s.get("time_published") is not None]
        published.sort(key=lambda s: s["time_published"])

        year = 2026
        circuit_map, country_map = self._build_calendar_maps(year)

        fallback_round = 1
        assigned = set()
        surveys = []

        for s in published:
            title = s.get("title", "").strip()
            cleaned = clean_race_name(title).lower()

            round_num = self._match_calendar_round(
                cleaned, circuit_map, country_map
            )

            if round_num is not None:
                if round_num in assigned:
                    print(f"  WARNING: multiple surveys match round "
                          f"{round_num} — using fallback for '{title}'")
                    round_num = None
                else:
                    assigned.add(round_num)
                    fallback_round = max(fallback_round, round_num + 1)

            if round_num is None:
                round_num = fallback_round
                fallback_round += 1

            surveys.append(self._to_survey(s, round_num=round_num))

        self._surveys = surveys

        unmatched = [s for s in self._surveys
                     if s["round_num"] not in assigned]
        if unmatched:
            names = ", ".join(s["title"] for s in unmatched)
            print(f"  Calendar-matched rounds: {names}")

        print(f"Found {len(self._surveys)} surveys "
              f"({len(self.active())} active, {len(self.closed())} closed)")

        return self

    def all(self) -> list[dict]:
        """All surveys, ordered by round number."""
        self._require_fetch()
        return sorted(self._surveys, key=lambda s: s["round_num"])

    def active(self) -> list[dict]:
        """Surveys that are currently open (status=1)."""
        self._require_fetch()
        return [s for s in self._surveys if s["status"] == self.STATUS_ACTIVE]

    def closed(self) -> list[dict]:
        """Surveys that are closed (status=2)."""
        self._require_fetch()
        return [s for s in self._surveys if s["status"] == self.STATUS_CLOSED]

    def print_summary(self) -> None:
        """Print a formatted table of all surveys."""
        self._require_fetch()
        print(f"\n{'Rnd':<5} {'Title':<25} {'ID':<12} {'Responses':<11} {'Status'}")
        print("─" * 60)
        for s in sorted(self._surveys, key=lambda s: s["round_num"]):
            status = "Active" if s["status"] == self.STATUS_ACTIVE else "Closed"
            print(f"{s['round_num']:<5} {s['title']:<25} {s['survey_id']:<12} "
                  f"{s['response_count']:<11} {status}")

    def save(self, path: str = "data/raw/survey_index.json") -> Path:
        """Write the full survey index (list of survey dicts) to a JSON file.

        Parameters
        ----------
        path : str
            Output file path.

        Returns
        -------
        Path
            Path to the saved file.
        """
        self._require_fetch()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self._surveys, indent=2))
        print(f"Saved {len(self._surveys)} surveys -> {out}")
        return out

    def save_all_tips(self, parser) -> list[Path]:
        """Loop through all surveys and fetch+save raw tips JSON for each.

        Parameters
        ----------
        parser : TipsParser
            An authenticated TipsParser instance.

        Returns
        -------
        list[Path]
            Paths to each saved tips file.
        """
        self._require_fetch()
        saved = []
        for s in self.all():
            path = parser.fetch_and_save(
                survey_id = s["survey_id"],
                round_num = s["round_num"],
                race_name = s["title"],
            )
            saved.append(path)
        return saved

    # ─────────────────────────────────────────────────────────
    # Calendar matching
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _build_calendar_maps(year: int) -> tuple[dict, dict]:
        """Build lookup dictionaries from OpenF1 calendar.

        Returns (circuit_map, country_map) where each maps a
        lowercased name to the renumbered round number.
        """
        try:
            from leaderboard import get_race_calendar
            calendar = get_race_calendar(year=year)
            races = calendar[calendar.session_name == "Race"]
            circuit_map = dict(zip(
                races["circuit_short_name"].str.lower(),
                races["round_number"],
            ))
            country_map = dict(zip(
                races["country_name"].str.lower(),
                races["round_number"],
            ))
            return circuit_map, country_map
        except Exception as e:
            print(f"  WARNING: Could not fetch OpenF1 calendar ({e}) "
                  f"— using sequential numbering")
            return {}, {}

    @staticmethod
    def _match_calendar_round(
        cleaned_title: str,
        circuit_map: dict,
        country_map: dict,
    ) -> int | None:
        """Try to match a survey title to a calendar round.

        Returns the round number if matched, None otherwise.
        """
        if cleaned_title in circuit_map:
            return circuit_map[cleaned_title]
        if cleaned_title in country_map:
            return country_map[cleaned_title]
        return None

    # ─────────────────────────────────────────────────────────
    # Fetching
    # ─────────────────────────────────────────────────────────

    def _fetch_all_surveys(self) -> list[dict]:
        data = self.client.make_request(
            "GET",
            "surveys",
            params={"page_size": 100},
        )

        if not data.get("success"):
            raise RuntimeError(f"Failed to fetch surveys: {data}")

        return data["data"]

    # ─────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────

    def _to_survey(self, raw: dict, round_num: int) -> dict:
        return {
            "round_num":      round_num,
            "survey_id":      raw.get("survey_id"),
            "title":          raw.get("title", "").strip(),
            "status":         raw.get("status"),
            "published_at":   raw.get("time_published"),
            "response_count": raw.get("num_vaild_responses", 0),
        }

    def _require_fetch(self) -> None:
        if not self._surveys:
            raise RuntimeError("No surveys loaded. Call fetch() first.")