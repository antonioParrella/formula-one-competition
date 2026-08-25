"""A driver missing from OpenF1's session `drivers` table must never become NaN.

Regression for KNOWN_ISSUES #2: `drivers?session_key=11348` (the 2026 Zandvoort
sprint) omitted car #6, Isack Hadjar, so the left merge in
`_get_championship_top10` produced

    ["ANT", "HAM", "RUS", "LEC", "NOR", "VER", "PIA", nan, "GAS", "LAW"]

which writes a bare `NaN` token no strict JSON parser accepts, and drops a
championship top-10 driver out of the set that decides the underdog multiplier —
anyone picking him would have been paid double.
"""

import json
import sys

import pandas as pd
import pytest

sys.path.insert(0, "src")
from fetch_results import ResultsFetcher
from leaderboard import ResultAggregator


# ── The real Zandvoort shapes, trimmed to what the merges touch ──
ZANDVOORT_MEETING = 1292
SPRINT_SESSION = 11348

# OpenF1 listed 22 drivers for the sprint session and left out #6.
SPRINT_DRIVERS = pd.DataFrame([
    {"driver_number": 12, "name_acronym": "ANT"},
    {"driver_number": 44, "name_acronym": "HAM"},
    {"driver_number": 63, "name_acronym": "RUS"},
    {"driver_number": 16, "name_acronym": "LEC"},
    {"driver_number": 4,  "name_acronym": "NOR"},
    {"driver_number": 1,  "name_acronym": "VER"},
    {"driver_number": 81, "name_acronym": "PIA"},
    {"driver_number": 10, "name_acronym": "GAS"},
    {"driver_number": 30, "name_acronym": "LAW"},
])

# ...but the championship standings reference him at P8.
STANDINGS = pd.DataFrame([
    {"driver_number": n, "position_start": i}
    for i, n in enumerate([12, 44, 63, 16, 4, 1, 81, 6, 10, 30], start=1)
])


# Rounds 10-12, the shape `_lookup_driver` walks back through.
CALENDAR = pd.DataFrame([
    {"round_number": 10, "session_name": "Race", "session_key": 11334},
    {"round_number": 11, "session_name": "Race", "session_key": 11342},
    {"round_number": 12, "session_name": "Race", "session_key": 11353},
    {"round_number": 12, "session_name": "Sprint", "session_key": SPRINT_SESSION},
])


def _bare_aggregator(drivers, standings, race_results=None, sprint_results=None):
    """A ResultAggregator with its fields set directly — __init__ needs network."""
    agg = object.__new__(ResultAggregator)
    agg.round_number = 12
    agg.meeting_key = ZANDVOORT_MEETING
    agg.sprint_session_key = SPRINT_SESSION
    agg.race_calendar = CALENDAR
    agg.drivers = drivers
    agg.championship_standings = standings
    agg.race_results = race_results if race_results is not None else pd.DataFrame()
    agg.sprint_results = sprint_results
    return agg


class TestBackfillMissingDrivers:
    def test_missing_driver_is_looked_up_by_meeting(self):
        agg = _bare_aggregator(SPRINT_DRIVERS.copy(), STANDINGS)
        calls = []

        def fake_fetch(endpoint, **params):
            calls.append((endpoint, params))
            # This is what OpenF1 really returns for #6 at this meeting.
            return pd.DataFrame([{"driver_number": 6, "name_acronym": "HAD"}])

        agg._fetch = fake_fetch
        agg._backfill_missing_drivers()

        assert calls == [("drivers", {"driver_number": 6, "meeting_key": ZANDVOORT_MEETING})]
        assert set(agg.drivers["driver_number"]) >= {6}
        assert agg.drivers.set_index("driver_number").loc[6, "name_acronym"] == "HAD"

    def test_complete_table_makes_no_extra_calls(self):
        complete = pd.concat(
            [SPRINT_DRIVERS, pd.DataFrame([{"driver_number": 6, "name_acronym": "HAD"}])],
            ignore_index=True,
        )
        agg = _bare_aggregator(complete, STANDINGS)
        agg._fetch = lambda *a, **k: pytest.fail("should not need a lookup")
        agg._backfill_missing_drivers()
        assert len(agg.drivers) == len(complete)

    def test_unnamed_placeholder_row_is_replaced(self):
        # A row that exists but carries no acronym is just as useless as no row.
        placeholder = pd.concat(
            [SPRINT_DRIVERS, pd.DataFrame([{"driver_number": 6, "name_acronym": None}])],
            ignore_index=True,
        )
        agg = _bare_aggregator(placeholder, STANDINGS)
        agg._fetch = lambda *a, **k: pd.DataFrame([{"driver_number": 6, "name_acronym": "HAD"}])
        agg._backfill_missing_drivers()

        rows = agg.drivers[agg.drivers["driver_number"] == 6]
        assert len(rows) == 1
        assert rows.iloc[0]["name_acronym"] == "HAD"

    def test_unresolvable_driver_warns_rather_than_crashing(self, capsys):
        agg = _bare_aggregator(SPRINT_DRIVERS.copy(), STANDINGS)
        agg._fetch = lambda *a, **k: pd.DataFrame()
        agg._backfill_missing_drivers()

        assert "could not resolve" in capsys.readouterr().out
        assert 6 not in set(agg.drivers["driver_number"])  # the guard below catches it


class TestDriverLookupScopes:
    """A driver replaced mid-season keeps their championship points — and so
    their top-10 place — while dropping out of the session driver tables."""

    def test_meeting_scope_is_tried_first_and_stops_there(self):
        agg = _bare_aggregator(SPRINT_DRIVERS.copy(), STANDINGS)
        calls = []

        def fake_fetch(endpoint, **params):
            calls.append(params)
            return pd.DataFrame([{"driver_number": 6, "name_acronym": "HAD"}])

        agg._fetch = fake_fetch
        assert agg._lookup_driver(6) is not None
        assert calls == [{"driver_number": 6, "meeting_key": ZANDVOORT_MEETING}]

    def test_falls_back_to_earlier_races_this_season(self):
        """OpenF1 can drop a departed driver from the meeting entirely; without
        this fallback `_require_acronyms` would raise and the round wouldn't
        score at all."""
        agg = _bare_aggregator(SPRINT_DRIVERS.copy(), STANDINGS)
        calls = []

        def fake_fetch(endpoint, **params):
            calls.append(params)
            if params.get("session_key") == 11342:      # round 11, where he last raced
                return pd.DataFrame([{"driver_number": 6, "name_acronym": "HAD"}])
            return pd.DataFrame()

        agg._fetch = fake_fetch
        row = agg._lookup_driver(6)
        assert row is not None and row.iloc[0]["name_acronym"] == "HAD"
        # Meeting first, then the most recent earlier race — and it stops on a hit.
        assert calls == [
            {"driver_number": 6, "meeting_key": ZANDVOORT_MEETING},
            {"driver_number": 6, "session_key": 11342},
        ]

    def test_never_falls_back_to_an_unscoped_lookup(self):
        """Car numbers are reused across seasons — an unscoped `driver_number=6`
        returns HAD *and* GOE, so it could name the wrong driver."""
        agg = _bare_aggregator(SPRINT_DRIVERS.copy(), STANDINGS)
        calls = []

        def fake_fetch(endpoint, **params):
            calls.append(params)
            return pd.DataFrame()

        agg._fetch = fake_fetch
        assert agg._lookup_driver(6) is None
        assert all("meeting_key" in c or "session_key" in c for c in calls)
        assert calls == [
            {"driver_number": 6, "meeting_key": ZANDVOORT_MEETING},
            {"driver_number": 6, "session_key": 11342},
            {"driver_number": 6, "session_key": 11334},
        ]

    def test_only_earlier_rounds_are_searched(self):
        agg = _bare_aggregator(SPRINT_DRIVERS.copy(), STANDINGS)
        calls = []
        agg._fetch = lambda endpoint, **p: (calls.append(p), pd.DataFrame())[1]
        agg._lookup_driver(6)
        assert 11353 not in [c.get("session_key") for c in calls]  # this round
        assert SPRINT_SESSION not in [c.get("session_key") for c in calls]  # not a Race

    def test_race_and_sprint_results_are_also_scanned(self):
        results = pd.DataFrame([{"driver_number": 6, "position": 8, "dnf": False}])
        agg = _bare_aggregator(SPRINT_DRIVERS.copy(), pd.DataFrame(), race_results=results)
        agg._fetch = lambda *a, **k: pd.DataFrame([{"driver_number": 6, "name_acronym": "HAD"}])
        agg._backfill_missing_drivers()
        assert agg.drivers.set_index("driver_number").loc[6, "name_acronym"] == "HAD"


class TestRequireAcronyms:
    def test_unresolved_driver_raises_naming_the_number(self):
        merged = STANDINGS.merge(SPRINT_DRIVERS, on="driver_number", how="left")
        with pytest.raises(ValueError, match=r"championship_top10_before_race.*\[6\]"):
            ResultsFetcher._require_acronyms(merged, "championship_top10_before_race")

    def test_complete_merge_returns_the_acronyms(self):
        drivers = pd.concat(
            [SPRINT_DRIVERS, pd.DataFrame([{"driver_number": 6, "name_acronym": "HAD"}])],
            ignore_index=True,
        )
        merged = STANDINGS.merge(drivers, on="driver_number", how="left")
        assert ResultsFetcher._require_acronyms(merged, "championship_top10_before_race") == [
            "ANT", "HAM", "RUS", "LEC", "NOR", "VER", "PIA", "HAD", "GAS", "LAW",
        ]

    def test_nan_can_no_longer_reach_the_results_file(self):
        merged = STANDINGS.merge(SPRINT_DRIVERS, on="driver_number", how="left")
        # What the old code serialised: json.dump happily writes a bare NaN token.
        raw = json.dumps({"championship_top10_before_race": merged["name_acronym"].tolist()})
        assert "NaN" in raw
        with pytest.raises(ValueError):
            json.loads(raw, parse_constant=_reject)


def _reject(token):
    raise ValueError(f"invalid JSON constant {token!r}")


class TestPublishedResults:
    """Every stored result file must be strict JSON with no missing driver."""

    def test_all_result_files_parse_strictly(self):
        from pathlib import Path

        files = sorted(Path("data/raw/results").glob("r*_result.json"))
        if not files:
            pytest.skip("no raw results in this checkout")
        for path in files:
            data = json.loads(path.read_text(), parse_constant=_reject)
            top10 = data.get("championship_top10_before_race")
            if top10 is None:
                continue  # round 1 has no pre-race standings
            assert all(isinstance(d, str) and d for d in top10), path.name
