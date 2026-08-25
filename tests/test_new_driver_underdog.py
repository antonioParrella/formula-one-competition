"""A mid-season driver change must not disturb the underdog multiplier.

Round 12 (Zandvoort, 2026) was the first roster change of the season: Yuki
Tsunoda (#22, Racing Bulls) made his 2026 debut in place of Isack Hadjar (#6),
running both the sprint and the race and finishing P11.

The two sides of that swap pull in opposite directions and both have to work:

* **The arrival** has no championship points, so he is outside the top 10 and
  every main-race pick of him scores double.
* **The departure** keeps the points he already scored, so he stays *inside*
  the championship top 10 (Hadjar was P8) even though he never took the start.
  A pick of him must score single, not double.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "src")
from race_utils import DRIVER_MAP
from scorer import Scorer

R12_RESULT = Path("data/raw/results/r12_zandvoort_result.json")


@pytest.fixture(scope="module")
def r12():
    if not R12_RESULT.exists():
        pytest.skip("no round 12 result in this checkout")
    return json.loads(R12_RESULT.read_text())


@pytest.fixture(scope="module")
def score_r12(r12):
    """Score one hypothetical main-race ticket against the real round 12."""
    scorer = Scorer(round_num=12)
    position_map = scorer._position_map(r12["top10"])
    champ_set = set(r12["championship_top10_before_race"])

    def _score(main_race):
        return scorer._score_main_race({"main_race": main_race}, position_map, champ_set)

    return _score


class TestTheArrival:
    def test_debutant_is_outside_the_championship_top_ten(self, r12):
        assert "TSU" not in r12["championship_top10_before_race"]

    def test_debutant_has_a_display_name(self):
        # Without this the leaderboard shows a bare "TSU" where every other
        # driver shows a surname.
        assert DRIVER_MAP.get("TSU") == "Tsunoda"

    def test_pick_of_the_debutant_is_flagged_underdog(self, score_r12):
        pick = score_r12(["TSU"])[0]
        assert pick["driver"] == "Tsunoda"
        assert pick["underdog"] is True

    def test_debutant_scores_double_when_he_lands(self, score_r12):
        # He actually finished P11, so score him against a slot he did not take
        # by comparing like for like: a top-10 hit by an underdog pays double.
        underdog_hit = score_r12(["HUL"])[0]          # P8, outside the champ top 10
        favourite_hit = score_r12(["NOR"])[0]         # P1, inside it
        assert underdog_hit["underdog"] is True
        assert favourite_hit["underdog"] is False
        assert underdog_hit["points"] == 2 * Scorer.POINTS_TOP10
        assert favourite_hit["points"] == Scorer.POINTS_EXACT

    def test_debutant_finished_outside_the_points_so_pays_nothing(self, r12, score_r12):
        assert "TSU" not in r12["top10"]
        pick = score_r12(["TSU"])[0]
        assert pick["result"] == "miss"
        assert pick["points"] == 0  # 0 x 2 is still 0


class TestTheDeparture:
    def test_replaced_driver_stays_in_the_championship_top_ten(self, r12):
        champ = r12["championship_top10_before_race"]
        assert champ[7] == "HAD"  # P8 on points he had already scored

    def test_replaced_driver_did_not_start(self, r12):
        assert "HAD" not in r12["top10"]
        assert "HAD" not in (r12["dnfs"] or [])

    def test_pick_of_the_replaced_driver_is_not_an_underdog(self, score_r12):
        # He is still a top-10 championship driver, so no multiplier — the
        # NaN bug (KNOWN_ISSUES #2) would have made him one.
        pick = score_r12(["HAD"])[0]
        assert pick["driver"] == "Hadjar"
        assert pick["underdog"] is False
        assert pick["points"] == 0  # didn't start, so nothing to pay either way


@pytest.fixture(scope="module")
def scored():
    path = Path("data/processed/r12_dutch_scored.json")
    if not path.exists():
        pytest.skip("no scored round 12 in this checkout")
    return json.loads(path.read_text())


class TestRoundTwelveAsScored:
    """What the swap actually did to the published round."""

    def test_the_one_real_tsunoda_pick_was_a_dnf_pick(self, scored):
        found = [
            (pt["player"], pick)
            for pt in scored["player_tips"]
            for pick in pt["dnf_picks"]
            if pick["driver"] == "Tsunoda"
        ]
        assert len(found) == 1
        _, pick = found[0]
        assert pick["dnfd"] is False and pick["points"] == 0

    def test_nobody_picked_the_replaced_driver(self, scored):
        names = [
            pick["driver"]
            for pt in scored["player_tips"]
            for pick in pt["picks"] + pt["dnf_picks"]
        ]
        assert "Hadjar" not in names

    def test_every_underdog_pick_paid_exactly_double(self, scored, r12):
        champ = set(r12["championship_top10_before_race"])
        base = {"exact": Scorer.POINTS_EXACT, "close": Scorer.POINTS_CLOSE,
                "top10": Scorer.POINTS_TOP10, "miss": Scorer.POINTS_MISS}
        surname_to_code = {v: k for k, v in DRIVER_MAP.items()}
        for pt in scored["player_tips"]:
            for pick in pt["picks"]:
                if pick["driver"] is None:
                    continue
                code = surname_to_code[pick["driver"]]
                assert pick["underdog"] == (code not in champ), pick
                expected = base[pick["result"]] * (2 if pick["underdog"] else 1)
                assert pick["points"] == expected, pick


class TestEveryRound:
    """No round may pay a championship top-10 driver at the underdog rate."""

    def test_underdog_flags_match_the_championship_list(self):
        surname_to_code = {v: k for k, v in DRIVER_MAP.items()}
        results = {
            json.loads(p.read_text())["round"]: json.loads(p.read_text())
            for p in Path("data/raw/results").glob("r*_result.json")
        }
        scored_files = sorted(Path("data/processed").glob("r*_scored.json"))
        if not scored_files or not results:
            pytest.skip("no processed rounds in this checkout")

        for path in scored_files:
            data = json.loads(path.read_text())
            rnd = data["round"]
            champ = set(results[rnd].get("championship_top10_before_race") or [])
            for pt in data["player_tips"]:
                for pick in pt["picks"]:
                    if pick["driver"] is None:
                        continue
                    code = surname_to_code[pick["driver"]]
                    if rnd == 1:
                        assert pick["underdog"] is False, (rnd, pick)
                    else:
                        assert pick["underdog"] == (code not in champ), (rnd, pick)

    def test_every_pick_resolves_to_a_known_driver(self):
        surnames = set(DRIVER_MAP.values())
        for path in sorted(Path("data/processed").glob("r*_scored.json")):
            data = json.loads(path.read_text())
            for pt in data["player_tips"]:
                for pick in pt["picks"] + pt["dnf_picks"]:
                    if pick["driver"] is not None:
                        assert pick["driver"] in surnames, (path.name, pick)
            for st in data.get("sprint_tips") or []:
                for pick in st["picks"]:
                    assert pick["driver"] in surnames, (path.name, pick)
