"""
scorer.py
─────────────────────────────────────────────────────────────
Reads raw tips, raw results, and overrides for a round,
applies scoring rules, and saves processed data.

Usage:
    from scorer import Scorer

    scorer = Scorer(round_num=1)
    scorer.score_and_save()
"""

import json
import os
from pathlib import Path
from datetime import datetime

from tips_parser import TipsParser
from race_utils import clean_race_name, DRIVER_MAP
from penalties import assess_lateness


class Scorer:
    """Scores a single round against OpenF1 results.

    Parameters
    ----------
    round_num : int
        Race round number (1-based).
    tips_dir : str
        Location of raw tips JSON files.
    results_dir : str
        Location of raw results JSON files.
    overrides_dir : str
        Location of override JSON files (may not exist for all rounds).
    output_dir : str
        Where to write the processed scored file.
    """

    # ── Points ────────────────────────────────────────────────
    POINTS_EXACT  = 5
    POINTS_CLOSE  = 3
    POINTS_TOP10  = 1
    POINTS_MISS   = 0
    POINTS_DNF    = 15
    MULTIPLIER_UNDERDOG = 2

    # ── Penalties ─────────────────────────────────────────────
    PENALTY_LATE_PER_SESSION = 5

    def __init__(
        self,
        round_num: int,
        tips_dir: str = "data/raw/tips",
        results_dir: str = "data/raw/results",
        overrides_dir: str = "data/overrides",
        schedule_dir: str = "data/raw/schedule",
        output_dir: str = "data/processed",
    ):
        self.round_num = round_num
        self.tips_dir = Path(tips_dir)
        self.results_dir = Path(results_dir)
        self.overrides_dir = Path(overrides_dir)
        self.schedule_dir = Path(schedule_dir)
        self.output_dir = Path(output_dir)

    # ─────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────

    def score_and_save(self) -> Path:
        """Score the round and write the processed file.

        Returns the path to the saved file.
        """
        tips = self._load_tips()
        results = self._load_results()
        overrides = self._load_overrides()
        schedule = self._load_schedule()
        scored = self._build_scored(tips, results, overrides, schedule)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        race_name = tips["race_name"]
        slug = race_name.lower().replace(" ", "_")
        out_path = self.output_dir / f"r{self.round_num:02d}_{slug}_scored.json"
        out_path.write_text(json.dumps(scored, indent=2))
        print(f"Saved scored results ({self.round_num}): {race_name} -> {out_path}")

        return out_path

    # ─────────────────────────────────────────────────────────
    # Loading
    # ─────────────────────────────────────────────────────────

    def _find_tips_file(self) -> Path | None:
        """Locate the tips file for this round."""
        pattern = f"r{self.round_num:02d}_*_tips.json"
        matches = list(self.tips_dir.glob(pattern))
        return matches[0] if matches else None

    def _find_results_file(self) -> Path | None:
        """Locate the results file for this round."""
        pattern = f"r{self.round_num:02d}_*_result.json"
        matches = list(self.results_dir.glob(pattern))
        return matches[0] if matches else None

    def _find_overrides_file(self) -> Path | None:
        """Locate the overrides file (may not exist)."""
        pattern = f"r{self.round_num:02d}_*_overrides.json"
        matches = list(self.overrides_dir.glob(pattern))
        return matches[0] if matches else None

    def _find_schedule_file(self) -> Path | None:
        """Locate the session schedule file for this round."""
        pattern = f"r{self.round_num:02d}_*_schedule.json"
        matches = list(self.schedule_dir.glob(pattern))
        return matches[0] if matches else None

    def _load_tips(self) -> dict:
        path = self._find_tips_file()
        if path is None:
            raise FileNotFoundError(
                f"No tips file found for round {self.round_num}"
            )
        return json.loads(path.read_text())

    def _load_results(self) -> dict:
        path = self._find_results_file()
        if path is None:
            raise FileNotFoundError(
                f"No results file found for round {self.round_num}"
            )
        return json.loads(path.read_text())

    def _load_overrides(self) -> dict:
        path = self._find_overrides_file()
        if path is None:
            return {}
        return json.loads(path.read_text())

    def _load_schedule(self) -> dict:
        path = self._find_schedule_file()
        if path is None:
            raise FileNotFoundError(
                f"No schedule file found for round {self.round_num} — "
                f"run the pipeline schedule stage (ScheduleFetcher) first"
            )
        return json.loads(path.read_text())

    # ─────────────────────────────────────────────────────────
    # Building scored file
    # ─────────────────────────────────────────────────────────

    def _build_scored(
        self, tips: dict, results: dict, overrides: dict, schedule: dict
    ) -> dict:
        """Assemble the complete scored payload."""
        top10_codes = results["top10"]
        actual_dnfs = results.get("dnfs") or []
        champ_top10 = results.get("championship_top10_before_race") or []
        is_sprint = results.get("is_sprint_weekend", False)

        # Build lookup structures
        position_map = self._position_map(top10_codes)
        champ_set = set(champ_top10)
        dnf_set = set(actual_dnfs)

        # Parse overrides by player name -> penalty
        penalty_map = self._parse_overrides(overrides)

        player_scores: dict = {}
        player_tips: list = []
        late_players: set = set()

        for submission in tips["submissions"]:
            picks = self._score_main_race(submission, position_map, champ_set)
            dnf_result = self._score_dnfs(submission, dnf_set)

            # Automatic late-submission penalty from the session schedule,
            # stacked on top of any manual override penalty.
            late_info = assess_lateness(
                submission.get("submitted_at"), schedule,
                self.PENALTY_LATE_PER_SESSION,
            )
            override_penalty = penalty_map.get(submission["player"], 0)
            penalty = late_info["auto_penalty"] + override_penalty
            if late_info["late"]:
                late_players.add(submission["player"])

            raw_total = sum(p["points"] for p in picks) + dnf_result["score"]
            # Submissions after the cutoff (Qualifying / Sprint) score zero;
            # otherwise deduct the penalty and floor the weekend total at 0.
            total = 0 if late_info["zeroed"] else max(0, raw_total - penalty)

            player_scores[submission["player"]] = total

            player_tips.append({
                "player": submission["player"],
                "response_id": submission.get("response_id"),
                "submitted_at": submission.get("submitted_at"),
                "score": total,
                "penalty": penalty,
                "auto_penalty": late_info["auto_penalty"],
                "override_penalty": override_penalty,
                "late_sessions": late_info["late_sessions"],
                "zeroed": late_info["zeroed"],
                "picks": picks,
                "dnf_picks": dnf_result["picks"],
            })

        # Sprint scoring (if applicable — scored separately for display).
        # Late submissions cannot earn sprint points.
        sprint_player_tips = None
        if is_sprint and results.get("sprint_top3"):
            sprint_player_tips = self._score_sprint(
                tips["submissions"], results["sprint_top3"], late_players
            )

        return {
            "round":            self.round_num,
            "race_name":        tips["race_name"],
            "scores":           player_scores,
            "player_tips":      player_tips,
            "sprint_tips":      sprint_player_tips,
        }

    # ─────────────────────────────────────────────────────────
    # Main race scoring
    # ─────────────────────────────────────────────────────────

    def _score_main_race(
        self,
        submission: dict,
        position_map: dict,
        champ_set: set,
    ) -> list[dict]:
        """Score a player's main race picks."""
        picks = []
        for idx, code in enumerate(submission["main_race"]):
            if not code:
                picks.append({
                    "driver": None,
                    "result": "miss",
                    "underdog": False,
                    "points": 0,
                })
                continue

            surname = DRIVER_MAP.get(code, code)
            actual_idx = position_map.get(code)

            if actual_idx is None:
                result = "miss"
            elif actual_idx == idx:
                result = "exact"
            elif actual_idx in (idx - 1, idx + 1):
                result = "close"
            else:
                result = "top10"

            base_pts = {
                "exact": self.POINTS_EXACT,
                "close": self.POINTS_CLOSE,
                "top10": self.POINTS_TOP10,
                "miss":  self.POINTS_MISS,
            }[result]

            is_underdog = (self.round_num > 1) and (code not in champ_set)
            points = base_pts * self.MULTIPLIER_UNDERDOG if is_underdog else base_pts

            picks.append({
                "driver": surname,
                "result": result,
                "underdog": is_underdog,
                "points": points,
            })

        return picks

    # ─────────────────────────────────────────────────────────
    # Sprint scoring
    # ─────────────────────────────────────────────────────────

    def _score_sprint(
        self,
        submissions: list[dict],
        sprint_top3: list[str],
        late_players: set | None = None,
    ) -> list[dict]:
        """Score sprint picks (5 pts each for P1, P2, P3 correct).

        Players who submitted late earn zero sprint points; their picks
        are still shown (with truthful results) but scored 0.
        """
        position_map = self._position_map(sprint_top3)
        late_players = late_players or set()

        results = []
        for submission in submissions:
            is_late = submission["player"] in late_players
            sprint_picks = submission.get("sprint") or []
            picks = []
            for idx, code in enumerate(sprint_picks):
                if not code:
                    continue
                surname = DRIVER_MAP.get(code, code)
                if position_map.get(code) == idx:
                    picks.append({
                        "driver": surname,
                        "result": "exact",
                        "points": 0 if is_late else 5,
                    })
                else:
                    picks.append({
                        "driver": surname,
                        "result": "miss",
                        "points": 0,
                    })
            results.append({
                "player": submission["player"],
                "picks": picks,
                "score": sum(p["points"] for p in picks),
                "zeroed": is_late,
            })

        return results

    # ─────────────────────────────────────────────────────────
    # DNF scoring
    # ─────────────────────────────────────────────────────────

    def _score_dnfs(self, submission: dict, dnf_set: set) -> dict:
        """Score DNF picks. 15 pts per pick that actually DNFed."""
        picks = []
        total = 0
        for code in submission.get("dnf_picks") or []:
            surname = DRIVER_MAP.get(code, code)
            dnfd = code in dnf_set
            points = self.POINTS_DNF if dnfd else 0
            total += points
            picks.append({
                "driver": surname,
                "dnfd": dnfd,
                "points": points,
            })

        return {"picks": picks, "score": total}

    # ─────────────────────────────────────────────────────────
    # Overrides
    # ─────────────────────────────────────────────────────────

    def _parse_overrides(self, overrides: dict) -> dict[str, int]:
        """Build a player_name -> penalty points map from overrides."""
        if "overrides" not in overrides:
            return {}
        return {o["player"]: o["penalty_points"] for o in overrides["overrides"]}

    # ─────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _position_map(top10_codes: list[str]) -> dict[str, int]:
        """Map driver code -> finishing position index (0-based)."""
        return {code: i for i, code in enumerate(top10_codes)}
