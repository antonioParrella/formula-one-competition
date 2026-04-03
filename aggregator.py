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
                p["games_played"] += 1
                p["points"] += score
                p["lastRace"] = score
                if p["bestRound"] is None or score > p["bestRound"]:
                    p["bestRound"] = score
                p["roundScores"][str(round_num)] = {
                    "race": race_name,
                    "points": score,
                }

        # Sort by points descending
        sorted_players = sorted(
            players.items(),
            key=lambda x: x[1]["points"],
            reverse=True,
        )

        standings_list = []
        for rank, (name, data) in enumerate(sorted_players, start=1):
            entry = {
                "name": name,
                "rank": rank,
                "games_played": data["games_played"],
                "points": data["points"],
                "lastRace": data["lastRace"],
                "bestRound": data["bestRound"],
                "roundScores": data["roundScores"],
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
