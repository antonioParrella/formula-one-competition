"""
build_site.py
─────────────────────────────────────────────────────────────
Reads all processed scored files and standings, assembles
the data structures expected by docs/index.html, and injects
them between the DATA_START / DATA_END markers.

Usage:
    from build_site import SiteBuilder

    builder = SiteBuilder()
    builder.build_and_save()
"""

import re
import json
from pathlib import Path
from datetime import datetime

from race_utils import DRIVER_MAP


def _resolve_dnfs(dnf_codes: list[str]) -> list[str]:
    """Convert raw DNF codes to driver surnames for the website."""
    return [DRIVER_MAP.get(code, code) for code in dnf_codes if code]


class SiteBuilder:
    """Builds the website data by injecting JSON into index.html.

    Parameters
    ----------
    processed_dir : str
        Location of processed scored JSON files.
    standings_file : str
        Path to the aggregated standings.json.
    html_file : str
        Path to index.html to inject data into.
    """

    def __init__(
        self,
        processed_dir: str = "data/processed",
        standings_file: str = "data/processed/standings.json",
        html_file: str = "docs/index.html",
    ):
        self.processed_dir = Path(processed_dir)
        self.standings_file = Path(standings_file)
        self.html_file = Path(html_file)

    # ─────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────

    def build_and_save(self) -> None:
        """Assemble data and inject into index.html."""
        html = self.html_file.read_text(encoding="utf-8")

        standings = json.loads(self.standings_file.read_text(encoding="utf-8"))
        scored_files = self._find_scored_files()
        scored_data = self._load_all_scored(scored_files)

        players = self._build_players(standings)
        stand_array = self._build_standings_array(standings, scored_files)
        rounds_array = self._build_rounds(scored_data)
        main_tips = self._build_main_tips(scored_data)
        sprint_tips = self._build_sprint_tips(scored_data)
        dnf_tips = self._build_dnf_tips(scored_data)

        data_block = (
            f"const PLAYERS = {json.dumps(players)};\n\n"
            f"const STANDINGS = {json.dumps(stand_array, indent=2)};\n\n"
            f"const ROUNDS = {json.dumps(rounds_array, indent=2)};\n\n"
            f"const MAIN_TIPS = {json.dumps(main_tips, indent=2)};\n\n"
            f"const SPRINT_TIPS = {json.dumps(sprint_tips, indent=2)};\n\n"
            f"const DNF_TIPS = {json.dumps(dnf_tips, indent=2)};"
        )

        new_html = re.sub(
            r"// DATA_START.*?// DATA_END",
            lambda m: f"// DATA_START\n{data_block}\n// DATA_END",
            html,
            flags=re.DOTALL,
        )

        self.html_file.write_text(new_html, encoding="utf-8")
        print(f"Updated {self.html_file} with {len(stand_array)} players, {len(rounds_array)} rounds")

    # ─────────────────────────────────────────────────────────
    # Building data blocks
    # ─────────────────────────────────────────────────────────

    def _build_players(self, standings: dict) -> list[str]:
        """Flat list of player names."""
        return [p["name"] for p in standings["players"]]

    def _build_standings_array(
        self, standings: dict, scored_files: list[Path]
    ) -> list[dict]:
        """STANDINGS array for the HTML."""
        total_rounds = len(scored_files)

        result = []
        for player in standings["players"]:
            scores = [0] * total_rounds
            for round_key, round_data in player["roundScores"].items():
                round_idx = int(round_key) - 1
                if 0 <= round_idx < total_rounds:
                    scores[round_idx] = round_data["points"]

            result.append({
                "name":       player["name"],
                "points":     player["points"],
                "lastRace":   player["lastRace"],
                "bestRound":  player["bestRound"],
                "games_played": player["games_played"],
                "roundScores": scores,
            })

        return result

    def _build_rounds(self, scored_data: list[dict]) -> list[dict]:
        """ROUNDS array for the HTML — includes score breakdown."""
        rounds = []
        for data in scored_data:
            # Main race scores (includes DNF points — they're part of player_tips score)
            # Extract per-player DNF scores from player_tips
            dnf_scores = {}
            for pt in data.get("player_tips", []):
                dnf_pts = sum(d.get("points", 0) for d in pt.get("dnf_picks", []))
                if dnf_pts > 0:
                    dnf_scores[pt["player"]] = dnf_pts

            # Sprint scores
            sprint_scores = {}
            sprint_tips = data.get("sprint_tips") or []
            for st in sprint_tips:
                sprint_scores[st["player"]] = st.get("score", 0)

            # Combined scores for sorting
            race_scores = dict(data.get("scores", {}))
            total_scores = dict(race_scores)
            for player, s_pt in sprint_scores.items():
                total_scores[player] = total_scores.get(player, 0) + s_pt

            # Per-player breakdown as structured data.
            # Derive raw pick/DNF points from player_tips so a penalty
            # (or a zeroed total) is shown explicitly rather than being
            # hidden inside a reduced "main" figure.
            tips_by_player = {
                pt["player"]: pt for pt in data.get("player_tips", [])
            }
            breakdown = {}
            for player in total_scores:
                pt = tips_by_player.get(player)
                dnf = dnf_scores.get(player, 0)
                spr = sprint_scores.get(player, 0)
                if pt is not None:
                    main = sum(p.get("points", 0) for p in pt.get("picks", []))
                    # Effective deduction: penalty capped by the non-negative
                    # floor and by any zeroing, so main + dnf - penalty == score.
                    shown_penalty = main + dnf - pt.get("score", 0)
                else:
                    main = race_scores.get(player, 0) - dnf
                    shown_penalty = 0
                bd = {"main": main, "dnf": dnf, "sprint": spr}
                if shown_penalty > 0:
                    bd["penalty"] = shown_penalty
                breakdown[player] = bd

            scores_sorted = dict(
                sorted(total_scores.items(), key=lambda x: x[1], reverse=True)
            )

            rounds.append({
                "round":     data["round"],
                "name":      data["race_name"],
                "date":      "—",
                "winner":    "—",
                "pole":      "—",
                "fastest":   "—",
                "scores":    scores_sorted,
                "breakdown": breakdown,
            })

        return rounds

    def _build_main_tips(self, scored_data: list[dict]) -> list[dict]:
        """MAIN_TIPS array for the HTML.

        Picks include underdog flag from the scorer output.
        """
        tips_list = []
        for data in scored_data:
            positions = data.get("race_positions") or []

            tips = []
            for pt in data["player_tips"]:
                picks = []
                for pick in pt["picks"]:
                    entry = {
                        "driver": pick["driver"],
                        "result": pick["result"],
                    }
                    if pick.get("underdog"):
                        entry["underdog"] = True
                    picks.append(entry)

                tips.append({
                    "player": pt["player"],
                    "score":  pt["score"],
                    "picks":  picks,
                })

            tips_list.append({
                "round":     data["round"],
                "raceName":  data["race_name"],
                "positions": positions,
                "tips":      tips,
            })

        return tips_list

    def _build_sprint_tips(self, scored_data: list[dict]) -> list[dict]:
        """SPRINT_TIPS array — only rounds with sprint data."""
        tips_list = []
        for data in scored_data:
            sprint = data.get("sprint_tips")
            if not sprint:
                continue

            sprint_positions = data.get("sprint_positions") or []

            tips = []
            for st in sprint:
                picks = []
                for pick in st["picks"]:
                    picks.append({
                        "driver": pick["driver"],
                        "result": pick["result"],
                    })

                tips.append({
                    "player": st["player"],
                    "score":  st["score"],
                    "picks":  picks,
                })

            tips_list.append({
                "round":     data["round"],
                "raceName":  data["race_name"],
                "positions": sprint_positions,
                "tips":      tips,
            })

        return tips_list

    def _build_dnf_tips(self, scored_data: list[dict]) -> list[dict]:
        """DNF_TIPS array — includes all rounds, even when no one made DNF picks."""
        tips_list = []
        for data in scored_data:
            actual_dnfs = _resolve_dnfs(data.get("race_dnfs") or [])
            if not actual_dnfs:
                continue

            tips = []
            for pt in data["player_tips"]:
                dnfs = pt.get("dnf_picks") or []
                if not dnfs:
                    continue
                tips.append({
                    "player":     pt["player"],
                    "picks":      [d["driver"] for d in dnfs],
                    "picksUsed":  len(dnfs),
                    "score":      sum(d["points"] for d in dnfs),
                })

            tips_list.append({
                "round":      data["round"],
                "raceName":   data["race_name"],
                "actualDnfs": actual_dnfs,
                "tips":       tips,
            })

        return tips_list

    def _find_scored_files(self) -> list[Path]:
        """Find all processed scored files, sorted by round."""
        pattern = "*_scored.json"
        files = list(self.processed_dir.glob(pattern))
        files.sort()
        return files

    def _load_all_scored(self, scored_files: list[Path]) -> list[dict]:
        """Load all scored JSON files and enrich with race data."""
        all_data = []
        for path in scored_files:
            data = json.loads(path.read_text(encoding="utf-8"))

            # Try to load matching results for positions/DNFs
            slug = data["race_name"].lower().replace(" ", "_")
            round_num = data["round"]
            result_pattern = f"r{round_num:02d}_*_result.json"
            result_files = list(self.processed_dir.parent.joinpath("raw/results").glob(result_pattern))
            if result_files:
                results = json.loads(result_files[0].read_text(encoding="utf-8"))
                data["race_positions"] = results.get("top10", [])
                data["race_dnfs"] = results.get("dnfs", [])
                data["sprint_positions"] = results.get("sprint_top3", [])
            else:
                data["race_positions"] = []
                data["race_dnfs"] = []
                data["sprint_positions"] = []

            all_data.append(data)

        return all_data
