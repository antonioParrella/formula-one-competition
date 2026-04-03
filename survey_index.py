"""
survey_index.py
─────────────────────────────────────────────────────────────
Fetches all surveys from SurveyMars in chronological order
(oldest first = round 1 first) and assigns round numbers
based on that order.

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

class SurveyIndex:
    """
    Fetches and indexes all SurveyMars surveys ordered by
    publish date, assigning round numbers 1, 2, 3... based
    on that order.

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
        Fetch all surveys from the API, sort oldest-first, and
        assign round numbers sequentially.

        Returns self to allow chaining:
            index = SurveyIndex(client).fetch()
        """
        print("Fetching survey list from SurveyMars...")
        raw = self._fetch_all_surveys()["data"]

        # Exclude surveys that haven't been published
        published = [s for s in raw if s.get("time_published") is not None]

        # Sort oldest published first — this is round order
        published.sort(key=lambda s: s["time_published"])

        # Assign round numbers 1, 2, 3...
        self._surveys = [
            self._to_survey(s, round_num=i + 1)
            for i, s in enumerate(published)
        ]

        print(f"Found {len(self._surveys)} surveys "
              f"({len(self.active())} active, {len(self.closed())} closed)")

        return self

    def all(self) -> list[dict]:
        """All surveys, ordered by round number."""
        self._require_fetch()
        return self._surveys

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
        for s in self._surveys:
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
        print(f"Saved {len(self._surveys)} surveys → {out}")
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
        for s in self._surveys:
            path = parser.fetch_and_save(
                survey_id = s["survey_id"],
                round_num = s["round_num"],
                race_name = s["title"],
            )
            saved.append(path)
        return saved

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