"""
aggregator.py
─────────────────────────────────────────────────────────────
Reads all processed scored files and builds season standings.

Usage:
    from aggregator import Aggregator

    agg = Aggregator()
    agg.build_and_save()
"""

import json
from pathlib import Path


class Aggregator:
    """Builds season standings from all scored round files.

    Parameters
    ----------
    processed_dir : str
        Location of processed scored JSON files.
    output_file : str
        Path to write the aggregated standby.json file.
    """

    def __init__(
        self,
        processed_dir: str = "data/processed",
        output_file: str = "data/processed/standings.json",
    ):
        self.processed_dir = Path(processed_dir)
        self.output_file = Path(output_file)

    # ─────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────

    def build_and_save(self) -> Path:
        """Build standings from all scored files and save.

        Returns the path to the saved file.
        """
        scored_files = self._find_scored_files()
        standings = self._build_standings(scored_files)

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.output_file.write_text(json.dumps(standings, indent=2))
        print(f"Saved standings → {self.output_file}")

        return self.output_file

    # ─────────────────────────────────────────────────────────
    # Building standings
    # ─────────────────────────────────────────────────────────

    def _build_standings(self, scored_files: list[Path]) -> dict:
        """Build the complete standings payload."""
        players: dict = {}  # name -> accumulator

        for path in scored_files:
            data = json.loads(path.read_text())
            round_num = data["round"]
            race_name = data["race_name"]

            for player_tips in data["player_tips"]:
                name = player_tips["player"]
                score = player_tips["score"]

                if name not in players:
                    players[name] = {
                        "games_played": 0,
                        "points": 0,
                        "lastRace": None,
                        "bestRound": None,
                        "roundScores": {},
                    }

                p = players[name]
                # Main race score
                main_score = score
                existing = p["roundScores"].get(str(round_num))
                if existing:
                    # Shouldn't happen but guard against double-counting
                    main_score = score - existing["points"]
                else:
                    p["games_played"] += 1

                p["points"] += main_score
                p["lastRace"] = main_score
                if p["bestRound"] is None or main_score > p["bestRound"]:
                    p["bestRound"] = main_score
                p["roundScores"][str(round_num)] = {
                    "race": race_name,
                    "points": main_score,
                }

            # Process sprint tips — add to same round's score
            for sprint_tip in data.get("sprint_tips") or []:
                name = sprint_tip["player"]
                sprint_score = sprint_tip.get("score", 0)

                if name not in players:
                    continue

                p = players[name]
                round_key = str(round_num)
                if round_key in p["roundScores"]:
                    p["roundScores"][round_key]["points"] += sprint_score
                p["points"] += sprint_score

        # Sort by points descending
        sorted_players = sorted(
            players.items(),
            key=lambda x: x[1]["points"],
            reverse=True,
        )

        standings_list = []
        for rank, (name, pdata) in enumerate(sorted_players, start=1):
            entry = {
                "name": name,
                "rank": rank,
                "games_played": pdata["games_played"],
                "points": pdata["points"],
                "lastRace": pdata["lastRace"] or 0,
                "bestRound": pdata["bestRound"] or 0,
                "roundScores": pdata["roundScores"],
            }
            standings_list.append(entry)

        return {
            "players": standings_list,
            "total_rounds": len(scored_files),
        }

    # ─────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────

    def _find_scored_files(self) -> list[Path]:
        """Find all processed scored files, sorted by round."""
        pattern = "r*_scored.json"
        files = list(self.processed_dir.glob(pattern))
        files.sort()
        return files
