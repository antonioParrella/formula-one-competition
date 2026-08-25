"""Sprint points must reach every standings field, not just the totals.

Regression for KNOWN_ISSUES #1: `_build_standings` used to set `lastRace` and
`bestRound` in the main-race pass and fold the sprint score in afterwards, so on
a sprint weekend both fields kept the main-race-only value while `points` and
`roundScores` were right.
"""

import json
import sys

import pytest

sys.path.insert(0, "src")
from aggregator import Aggregator


def _scored(round_num, race_name, main, sprint=None):
    """A minimal scored-round file: {player: score} for race and sprint."""
    return {
        "round": round_num,
        "race_name": race_name,
        "player_tips": [{"player": p, "score": s} for p, s in main.items()],
        "sprint_tips": [{"player": p, "score": s} for p, s in (sprint or {}).items()],
    }


def _build(tmp_path, rounds):
    for data in rounds:
        path = tmp_path / f"r{data['round']:02d}_{data['race_name'].lower()}_scored.json"
        path.write_text(json.dumps(data))
    standings = Aggregator(
        processed_dir=str(tmp_path),
        output_file=str(tmp_path / "standings.json"),
    ).build_and_save()
    return {p["name"]: p for p in json.loads(standings.read_text())["players"]}


class TestSprintPointsInEveryField:
    def test_last_race_includes_sprint(self, tmp_path):
        players = _build(tmp_path, [
            _scored(1, "Melbourne", {"Barry": 20}),
            _scored(2, "Dutch", {"Barry": 26}, sprint={"Barry": 5}),
        ])
        assert players["Barry"]["lastRace"] == 31
        assert players["Barry"]["points"] == 51

    def test_best_round_includes_sprint(self, tmp_path):
        # The sprint weekend only becomes the best round once its 5 points count.
        players = _build(tmp_path, [
            _scored(1, "Melbourne", {"Rino": 28}),
            _scored(2, "Dutch", {"Rino": 26}, sprint={"Rino": 5}),
        ])
        assert players["Rino"]["bestRound"] == 31

    def test_round_scores_and_points_still_agree(self, tmp_path):
        players = _build(tmp_path, [
            _scored(1, "Melbourne", {"Jake": 19}),
            _scored(2, "China", {"Jake": 41}, sprint={"Jake": 5}),
            _scored(3, "Japan", {"Jake": 22}),
        ])
        jake = players["Jake"]
        assert jake["roundScores"]["2"]["points"] == 46
        assert jake["points"] == sum(r["points"] for r in jake["roundScores"].values())
        assert jake["bestRound"] == 46
        assert jake["lastRace"] == 22
        assert jake["games_played"] == 3

    def test_non_sprint_weekend_unaffected(self, tmp_path):
        players = _build(tmp_path, [
            _scored(1, "Melbourne", {"Luca": 17}),
            _scored(2, "Japan", {"Luca": 30}),
        ])
        assert players["Luca"] == {
            "name": "Luca",
            "rank": 1,
            "games_played": 2,
            "points": 47,
            "lastRace": 30,
            "bestRound": 30,
            "roundScores": {
                "1": {"race": "Melbourne", "points": 17},
                "2": {"race": "Japan", "points": 30},
            },
        }


class TestDerivedFields:
    def test_last_race_follows_round_number_not_file_order(self, tmp_path):
        # Round 10 sorts before round 9 lexicographically without zero padding;
        # lastRace is keyed off the round number, so it doesn't care.
        players = _build(tmp_path, [
            _scored(9, "Silverstone", {"Tara": 36}),
            _scored(10, "Belgium", {"Tara": 12}),
        ])
        assert players["Tara"]["lastRace"] == 12
        assert players["Tara"]["bestRound"] == 36

    def test_sprint_only_player_still_scores(self, tmp_path):
        # No main-race tip that round — the sprint points used to be added to
        # the season total with no round recorded, or dropped entirely.
        players = _build(tmp_path, [
            _scored(1, "Melbourne", {"Dean": 28}),
            _scored(2, "Dutch", {"Dean": 26}, sprint={"Dean": 5, "Josh": 5}),
        ])
        assert players["Josh"]["points"] == 5
        assert players["Josh"]["roundScores"]["2"]["points"] == 5
        assert players["Josh"]["games_played"] == 1

    def test_duplicate_round_file_does_not_double_count(self, tmp_path):
        rounds = [
            _scored(1, "Melbourne", {"Alex": 21}),
            _scored(2, "Dutch", {"Alex": 23}, sprint={"Alex": 5}),
        ]
        players = _build(tmp_path, rounds)
        assert players["Alex"]["points"] == 49

        # Same round re-read (e.g. a stray rescored file) replaces, not adds.
        (tmp_path / "r02_dutch_copy_scored.json").write_text(json.dumps(rounds[1]))
        again = _build(tmp_path, rounds)
        assert again["Alex"]["points"] == 49
        assert again["Alex"]["games_played"] == 2
        assert again["Alex"]["bestRound"] == 28


@pytest.fixture(scope="module")
def players():
    """The published standings, if this checkout has them."""
    from pathlib import Path

    path = Path("data/processed/standings.json")
    if not path.exists():
        pytest.skip("no processed standings in this checkout")
    return {p["name"]: p for p in json.loads(path.read_text())["players"]}


class TestRealSeason:
    """The published season — the numbers KNOWN_ISSUES #1 predicted."""

    @pytest.mark.parametrize("name,last_race", [("Barry", 31), ("Jake", 24), ("Josh", 27)])
    def test_last_race_after_round_12(self, players, name, last_race):
        assert players[name]["lastRace"] == last_race

    @pytest.mark.parametrize("name,best", [
        ("Rino", 54), ("Jake", 46), ("Dean", 58),
        ("Veljko", 38), ("Tara", 41), ("Alex", 28),
    ])
    def test_best_round_after_round_12(self, players, name, best):
        assert players[name]["bestRound"] == best

    def test_every_player_total_matches_their_rounds(self, players):
        for name, p in players.items():
            assert p["points"] == sum(r["points"] for r in p["roundScores"].values()), name
            assert p["bestRound"] == max(r["points"] for r in p["roundScores"].values()), name
