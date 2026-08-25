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
        print(f"Saved standings -> {self.output_file}")

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
            round_key = str(round_num)

            # A sprint is part of its round, not a round of its own. Collect
            # the sprint scores up front so the single pass below records one
            # round total everywhere — folding them in afterwards is what left
            # lastRace and bestRound excluding sprint points on sprint weekends.
            sprint_scores: dict = {}
            for sprint_tip in data.get("sprint_tips") or []:
                sprint_scores[sprint_tip["player"]] = (
                    sprint_scores.get(sprint_tip["player"], 0)
                    + sprint_tip.get("score", 0)
                )

            scored_players = [pt["player"] for pt in data["player_tips"]]
            # A sprint tip without a main-race tip still earned its points.
            sprint_only = [n for n in sprint_scores if n not in set(scored_players)]

            for player_tips in data["player_tips"]:
                self._record_round(
                    players,
                    name=player_tips["player"],
                    round_key=round_key,
                    race_name=race_name,
                    round_score=player_tips["score"] + sprint_scores.get(player_tips["player"], 0),
                )

            for name in sprint_only:
                self._record_round(
                    players,
                    name=name,
                    round_key=round_key,
                    race_name=race_name,
                    round_score=sprint_scores[name],
                )

        # Sort by points descending
        sorted_players = sorted(
            players.items(),
            key=lambda x: x[1]["points"],
            reverse=True,
        )

        standings_list = []
        for rank, (name, pdata) in enumerate(sorted_players, start=1):
            rounds = pdata["roundScores"]
            entry = {
                "name": name,
                "rank": rank,
                "games_played": pdata["games_played"],
                "points": pdata["points"],
                # Both derived from roundScores rather than tracked alongside
                # it, so they cannot drift out of step with the round totals.
                "lastRace": self._last_race(rounds),
                "bestRound": max((r["points"] for r in rounds.values()), default=0),
                "roundScores": rounds,
            }
            standings_list.append(entry)

        return {
            "players": standings_list,
            "total_rounds": len(scored_files),
        }

    # ─────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _record_round(
        players: dict,
        name: str,
        round_key: str,
        race_name: str,
        round_score: int,
    ) -> None:
        """Fold one player's total for one round into the accumulator.

        ``round_score`` is the whole round — main race plus sprint. Re-recording
        a round already seen replaces it rather than double-counting, which
        shouldn't happen but would silently inflate the season otherwise.
        """
        if name not in players:
            players[name] = {
                "games_played": 0,
                "points": 0,
                "roundScores": {},
            }

        p = players[name]
        previous = p["roundScores"].get(round_key)
        if previous is None:
            p["games_played"] += 1

        p["points"] += round_score - (previous["points"] if previous else 0)
        p["roundScores"][round_key] = {"race": race_name, "points": round_score}

    @staticmethod
    def _last_race(round_scores: dict) -> int:
        """Points from the most recent round played, by round number.

        Keyed off the round rather than iteration order so it doesn't depend
        on the order the scored files happen to be read in.
        """
        if not round_scores:
            return 0
        latest = max(round_scores, key=int)
        return round_scores[latest]["points"]

    def _find_scored_files(self) -> list[Path]:
        """Find all processed scored files, sorted by round."""
        pattern = "r*_scored.json"
        files = list(self.processed_dir.glob(pattern))
        files.sort()
        return files
