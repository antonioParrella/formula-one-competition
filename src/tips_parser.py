"""
tips_parser.py
─────────────────────────────────────────────────────────────
Fetches, cleans, and saves SurveyMars tip submissions for a
single race weekend.

Usage:
    from tips_parser import TipsParser
    from surveymars_client import SurveyMarsClient

    client = SurveyMarsClient(account_id="...", secret="...")
    client.authenticate()

    parser = TipsParser(client)
    parser.fetch_and_save(
        survey_id  = "m1BbiXKjA",
        round_num  = 4,
        race_name  = "Japan GP",
    )

Or run directly from the command line:
    python tips_parser.py \
        --survey-id m1BbiXKjA \
        --round 4 \
        --race "Japan GP" \
        [--force]

Environment variables required:
    SURVEYMARS_ACCOUNT_ID
    SURVEYMARS_SECRET
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

from survey_mars import SurveyMarsClient
from race_utils import clean_race_name


class TipsParser:
    """
    Fetches responses from a SurveyMars survey and parses them
    into a clean, consistent JSON format ready for scoring.

    Attributes
    ----------
    client : SurveyMarsClient
        An authenticated SurveyMars API client.
    output_dir : Path
        Directory where raw tip JSON files are saved.
        Defaults to data/raw/tips/.
    """

    # ── Question index constants ──────────────────────────────
    # These match the question_index keys in SurveyMars responses.
    # Update if you restructure the survey.
    Q_NAME       = 10000
    Q_RACE_START = 20001   # P1
    Q_RACE_END   = 20010   # P10
    Q_SPRINT_S1  = 60001
    Q_SPRINT_S2  = 60002
    Q_SPRINT_S3  = 60003
    Q_DNF        = 50000

    # Championship predictions (round 1 only)
    Q_DRIVERS_CHAMP      = 30000
    Q_CONSTRUCTORS_CHAMP = 40000

    def __init__(self, client: SurveyMarsClient, output_dir: str = "data/raw/tips"):
        self.client     = client
        self.output_dir = Path(output_dir)

    # ─────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────

    def fetch_and_save(
        self,
        survey_id: str,
        round_num: int,
        race_name: str,
        force: bool = True,
    ) -> Path:
        """
        Fetch all responses for a survey, parse them, and save
        to a JSON file.

        Sprint weekends are auto-detected by checking for the
        presence of sprint question indices (60001) in the
        response data.

        Parameters
        ----------
        survey_id : str
            SurveyMars survey ID e.g. "m1BbiXKjA".
        round_num : int
            Race round number e.g. 4.
        race_name : str
            Human-readable race name (e.g. "Japan GP" → "Japan").
        force : bool
            If True, re-fetch and overwrite existing output.

        Returns
        -------
        Path
            Path to the saved JSON file.
        """
        race_name = clean_race_name(race_name)
        out_path = self._output_path(round_num, race_name)

        if out_path.exists() and not force:
            print(f"Tips already saved at {out_path}")
            print("Pass force=True to re-fetch and overwrite.")
            return out_path

        print(f"\nFetching responses for {race_name} (survey: {survey_id})...")
        raw_responses = self._fetch_all_responses(survey_id)
        print(f"Total responses fetched: {len(raw_responses)}")

        is_sprint = self._is_sprint_weekend(raw_responses)
        if is_sprint:
            print("  Sprint weekend detected via survey questions")

        submissions = [self._parse_response(r, is_sprint) for r in raw_responses]
        submissions = self._deduplicate_submissions(submissions)
        self._warn_unknown_players(submissions)

        has_predictions = self._has_championship_predictions(raw_responses)
        if has_predictions:
            print("  Championship predictions detected")

        payload = {
            "round":             round_num,
            "race_name":         race_name,
            "survey_id":         survey_id,
            "is_sprint_weekend": is_sprint,
            "fetched_at":        datetime.utcnow().isoformat(),
            "total_responses":   len(submissions),
            "submissions":       submissions,
        }

        # Championship predictions (round 1 only) — saved to a separate file
        if has_predictions:
            predictions = [self._extract_prediction(r) for r in raw_responses]
            self._save_championship_predictions(predictions, survey_id, round_num, race_name)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Saved {len(submissions)} submissions -> {out_path}")

        self._print_preview(submissions)
        return out_path

    def load(self, round_num: int, race_name: str) -> dict:
        """
        Load a previously saved tips file without re-fetching.

        Parameters
        ----------
        round_num : int
            Race round number.
        race_name : str
            Race name matching the one used when saving.

        Returns
        -------
        dict
            The full parsed tips payload.
        """
        path = self._output_path(round_num, race_name)
        if not path.exists():
            raise FileNotFoundError(
                f"No tips file found at {path}. "
                "Run fetch_and_save() first."
            )
        return json.loads(path.read_text())

    # ─────────────────────────────────────────────────────────
    # Sprint detection
    # ─────────────────────────────────────────────────────────

    def _is_sprint_weekend(self, raw_responses: list[dict]) -> bool:
        """Check if any response contains sprint question indices.

        If the first response has question 60001, the survey is a sprint
        weekend survey. We only need one response to confirm.
        """
        if not raw_responses:
            return False
        items = raw_responses[0].get("items", {})
        return str(self.Q_SPRINT_S1) in items

    # ─────────────────────────────────────────────────────────
    # Championship prediction detection & saving
    # ─────────────────────────────────────────────────────────

    def _has_championship_predictions(self, raw_responses: list[dict]) -> bool:
        """Check if any response contains championship prediction questions."""
        if not raw_responses:
            return False
        items = raw_responses[0].get("items", {})
        return str(self.Q_DRIVERS_CHAMP) in items

    def _extract_prediction(self, raw: dict) -> dict:
        """Extract one player's championship prediction."""
        items = raw.get("items", {})

        drivers = items.get(str(self.Q_DRIVERS_CHAMP), {}).get("answer_text", "").strip().upper()
        constructors = items.get(str(self.Q_CONSTRUCTORS_CHAMP), {}).get("answer_text", "").strip().title()

        player_name = self._get_player_name(items)

        return {
            "player":          player_name,
            "drivers_pick":    drivers or None,
            "constructors_pick": constructors if constructors else None,
        }

    def _save_championship_predictions(
        self,
        predictions: list[dict],
        survey_id: str,
        round_num: int,
        race_name: str,
    ) -> Path:
        """Save championship predictions to a separate JSON file."""
        payload = {
            "round":       round_num,
            "race_name":   race_name,
            "survey_id":   survey_id,
            "predictions": [p for p in predictions if p.get("drivers_pick") or p.get("constructors_pick")],
        }

        out_path = self.output_dir / "championship_predictions.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Saved {len(payload['predictions'])} championship predictions -> {out_path}")
        return out_path

    # ─────────────────────────────────────────────────────────
    # Fetching
    # ─────────────────────────────────────────────────────────

    def _fetch_all_responses(self, survey_id: str) -> list[dict]:
        """Page through all responses and return a flat list."""
        all_responses = []
        page = 1

        while True:
            print(f"  Fetching page {page}...")
            data = self.client.make_request(
                "GET",
                f"surveys/{survey_id}/responses",
                params={"page_index": page, "page_size": 100},
            )

            if not data.get("success"):
                raise RuntimeError(f"Failed to fetch responses: {data}")

            # Responses come back as a list of {'key': response_id, 'value': {...}} pairs;
            # the actual response object lives under ["value"]
            answers = data["data"]["answers"]
            batch = [entry["value"] for entry in answers]
            all_responses.extend(batch)
            print(f"    Got {len(batch)} responses (total: {len(all_responses)})")

            if not data["data"].get("has_next", False):
                break
            page += 1

        return all_responses

    # ─────────────────────────────────────────────────────────
    # Parsing
    # ─────────────────────────────────────────────────────────

    def _parse_response(self, raw: dict, is_sprint: bool) -> dict:
        """
        Convert one raw SurveyMars response into our clean format.

        Output shape:
        {
            "player":       "Luca",
            "response_id":  7,
            "submitted_at": "2026-03-26T22:32:33",
            "time_spent_s": 154,
            "main_race":    ["Russell", "Leclerc", ...],   # P1–P10
            "sprint":       ["Norris", "Piastri", "Russell"] | null,
            "dnf_picks":    ["Albon"] | []
        }
        """
        items = raw.get("items", {})

        player    = self._get_player_name(items)
        main_race = self._get_main_race_picks(items)
        sprint    = self._get_sprint_picks(items) if is_sprint else None
        dnf_picks = self._get_dnf_picks(items)

        return {
            "player":       player,
            "response_id":  raw.get("response_id"),
            "submitted_at": raw.get("time_submitted"),
            "time_spent_s": raw.get("time_spent"),
            "main_race":    main_race,
            "sprint":       sprint,
            "dnf_picks":    dnf_picks,
        }

    def _get_player_name(self, items: dict) -> str:
        item = items.get(str(self.Q_NAME), {})
        name = item.get("answer_text", "").strip()
        return name.title() if name else "Unknown"

    def _get_main_race_picks(self, items: dict) -> list:
        picks = []
        for q_idx in range(self.Q_RACE_START, self.Q_RACE_END + 1):
            item = items.get(str(q_idx), {})
            code = item.get("answer_text", "").strip().upper()
            picks.append(code if code else None)
        return picks

    def _get_sprint_picks(self, items: dict) -> list | None:
        picks = []
        for q_idx in [self.Q_SPRINT_S1, self.Q_SPRINT_S2, self.Q_SPRINT_S3]:
            item = items.get(str(q_idx), {})
            code = item.get("answer_text", "").strip().upper()
            picks.append(code if code else None)
        return picks if any(picks) else None

    def _get_dnf_picks(self, items: dict) -> list:
        item        = items.get(str(self.Q_DNF), {})
        answer_text = item.get("answer_text", "")

        if not answer_text or answer_text.strip() in ("", "(Empty)"):
            return []

        # Split on comma or the visual separator ┋ (U+250B); store raw codes
        codes = [
            d.strip().upper()
            for d in answer_text.replace("\u250b", ",").split(",")
        ]
        return [c for c in codes if c and c not in ("(EMPTY)", "")]

    # ─────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────

    def _output_path(self, round_num: int, race_name: str) -> Path:
        name = clean_race_name(race_name)
        slug = name.lower().replace(" ", "_")
        return self.output_dir / f"r{round_num:02d}_{slug}_tips.json"

    def _warn_unknown_players(self, submissions: list[dict]) -> None:
        unknown = [s for s in submissions if s["player"] == "Unknown"]
        if unknown:
            print(f"WARNING: {len(unknown)} submission(s) had no player name")

    def _deduplicate_submissions(self, submissions: list[dict]) -> list[dict]:
        """Keep only the latest submission per player (by submitted_at timestamp)."""
        latest_by_player: dict[str, dict] = {}
        for s in submissions:
            player = s.get("player", "Unknown")
            existing = latest_by_player.get(player)
            if existing is None or s.get("submitted_at", "") > existing.get("submitted_at", ""):
                latest_by_player[player] = s

        deduped = list(latest_by_player.values())
        removed_count = len(submissions) - len(deduped)
        if removed_count > 0:
            print(f"  Removed {removed_count} duplicate submission(s) — kept latest")

        return deduped

    def _print_preview(self, submissions: list[dict]) -> None:
        print(f"\n{'Player':<12} {'Main race top 3':<38} {'Sprint':<22} {'DNFs'}")
        print("─" * 85)
        for s in submissions:
            top3   = " -> ".join(str(d) for d in s["main_race"][:3])
            sprint = " -> ".join(str(d) for d in s["sprint"]) if s["sprint"] else "-"
            dnfs   = ", ".join(s["dnf_picks"]) if s["dnf_picks"] else "—"
            print(f"{s['player']:<12} {top3:<38} {sprint:<22} {dnfs}")
