"""The underdog multiplier keys off the ladder *before* the weekend.

Regression for KNOWN_ISSUES #4: `_get_championship_top10` did
`championship_standings.head(10)`, but OpenF1 returns `championship_drivers`
rows in `position_current` order — the ladder *after* the session. So the saved
"before race" ladder was really the after-the-weekend one, which is not what
anyone submitted tips against.

Ordering alone is harmless; it only mis-scores when it changes the *set*. It did
in three rounds: Shanghai, Suzuka and Monaco.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, "src")
from fetch_results import ResultsFetcher
from race_utils import DRIVER_MAP


# The real Suzuka standings, rows in the order the API actually sends them —
# sorted by position_current, which is what made `.head(10)` pick up the
# after-the-race ladder. Piastri climbed into the top 10 during the race
# (11th -> 6th); Lindblad dropped out of it (10th -> 11th).
SUZUKA = pd.DataFrame([
    # driver          before  after
    {"driver_number": 12, "position_start": 2,  "position_current": 1},   # ANT
    {"driver_number": 63, "position_start": 1,  "position_current": 2},   # RUS
    {"driver_number": 16, "position_start": 3,  "position_current": 3},   # LEC
    {"driver_number": 44, "position_start": 4,  "position_current": 4},   # HAM
    {"driver_number": 1,  "position_start": 6,  "position_current": 5},   # NOR
    {"driver_number": 81, "position_start": 11, "position_current": 6},   # PIA
    {"driver_number": 87, "position_start": 5,  "position_current": 7},   # BEA
    {"driver_number": 10, "position_start": 7,  "position_current": 8},   # GAS
    {"driver_number": 3,  "position_start": 8,  "position_current": 9},   # VER
    {"driver_number": 30, "position_start": 9,  "position_current": 10},  # LAW
    {"driver_number": 41, "position_start": 10, "position_current": 11},  # LIN
])


class TestPreRaceLadder:
    def test_orders_by_position_start_not_api_order(self):
        top = ResultsFetcher._pre_race_ladder(SUZUKA)
        assert list(top["driver_number"]) == [63, 12, 16, 44, 87, 1, 10, 3, 30, 41]

    def test_the_set_differs_from_the_after_the_race_ladder(self):
        # This is the part that actually mis-scored: #41 Lindblad was in the
        # top 10 when tips locked, #81 Piastri was not.
        before = set(ResultsFetcher._pre_race_ladder(SUZUKA)["driver_number"])
        after = set(SUZUKA.head(10)["driver_number"])   # what the old code took
        assert 41 in before and 41 not in after
        assert 81 in after and 81 not in before

    def test_returns_exactly_ten(self):
        assert len(ResultsFetcher._pre_race_ladder(SUZUKA)) == 10

    def test_missing_position_start_column_raises(self):
        bare = SUZUKA[["driver_number", "position_current"]]
        with pytest.raises(ValueError, match="no position_start column"):
            ResultsFetcher._pre_race_ladder(bare)

    def test_null_position_start_raises_naming_the_driver(self):
        holed = SUZUKA.copy()
        holed.loc[holed["driver_number"] == 16, "position_start"] = None
        with pytest.raises(ValueError, match=r"no position_start for driver number\(s\) \[16\]"):
            ResultsFetcher._pre_race_ladder(holed)

    def test_a_gap_in_the_top_ten_raises(self):
        gapped = SUZUKA.copy()
        gapped.loc[gapped["driver_number"] == 87, "position_start"] = 99
        with pytest.raises(ValueError, match="not a complete 1-10"):
            ResultsFetcher._pre_race_ladder(gapped)

    def test_short_standings_raise(self):
        with pytest.raises(ValueError, match="not a complete 1-10"):
            ResultsFetcher._pre_race_ladder(SUZUKA.head(5))


class TestSavedLadders:
    """Every stored ladder must be a complete, plausible top 10."""

    @staticmethod
    def _results():
        files = sorted(Path("data/raw/results").glob("r*_result.json"))
        if not files:
            pytest.skip("no raw results in this checkout")
        return {json.loads(p.read_text())["round"]: json.loads(p.read_text()) for p in files}

    def test_round_one_has_no_ladder(self):
        assert self._results()[1]["championship_top10_before_race"] is None

    def test_every_other_round_has_ten_distinct_known_drivers(self):
        for rnd, data in self._results().items():
            if rnd == 1:
                continue
            ladder = data["championship_top10_before_race"]
            assert len(ladder) == 10, rnd
            assert len(set(ladder)) == 10, rnd
            assert all(code in DRIVER_MAP for code in ladder), rnd

    def test_the_three_rounds_whose_set_was_wrong(self):
        """Pinned so the after-the-race ladder can't creep back in."""
        results = self._results()
        # Shanghai: Bortoleto and Gasly were in; Piastri was not yet.
        r02 = results[2]["championship_top10_before_race"]
        assert "GAS" in r02 and "PIA" not in r02
        # Suzuka: Lindblad in, Piastri out.
        r03 = results[3]["championship_top10_before_race"]
        assert "LIN" in r03 and "PIA" not in r03
        # Monaco: Bearman in, Hadjar out.
        r06 = results[6]["championship_top10_before_race"]
        assert "BEA" in r06 and "HAD" not in r06

    def test_a_ladder_never_contains_a_driver_who_had_not_raced(self):
        # A driver cannot be on the championship ladder before their debut.
        results = self._results()
        assert "TSU" not in (results[12]["championship_top10_before_race"] or [])
