"""Comp context: the championship top-10 that defines the underdog set.

From round 2 the comp doubles points for picks of drivers outside the
championship top-10 before the race (``Scorer.MULTIPLIER_UNDERDOG``).
This resolves that set either from OpenF1 — reusing the parent repo's
``leaderboard.get_race_calendar`` / ``_fetch_openf1``, the same source
``ResultsFetcher`` uses for the official comp scoring — or from a
manual list in config.yaml.
"""

import _paths  # noqa: F401

from leaderboard import _fetch_openf1, get_race_calendar
from scoring.rules import ScoringContext


def resolve_context(cfg: dict) -> ScoringContext:
    round_num = int(cfg["race"]["round"])
    if round_num <= 1:
        print("Round 1: no underdog bonus this round.")
        return ScoringContext(round_num=round_num)

    source = cfg["underdogs"].get("source", "manual")
    if source == "manual":
        top10 = [c.upper() for c in cfg["underdogs"]["manual_top10"]]
        print(f"Championship top-10 (manual list): {', '.join(top10)}")
    elif source == "openf1":
        top10 = _openf1_top10(int(cfg["race"]["year"]), round_num)
        print(f"Championship top-10 (OpenF1, before round {round_num}): "
              f"{', '.join(top10)}")
    else:
        raise ValueError(f"Unknown underdogs.source: {source!r}")

    if len(top10) != 10:
        raise RuntimeError(
            f"Championship top-10 has {len(top10)} drivers: {top10}"
        )
    return ScoringContext(round_num=round_num,
                          championship_top10=frozenset(top10))


def _openf1_top10(year: int, round_num: int) -> list[str]:
    """Standings entering the round, via the same OpenF1 endpoints the
    comp's own ``ResultsFetcher._get_championship_top10`` reads.

    OpenF1 only publishes a round's pre-race standings once its race
    weekend exists (fetching earlier 404s), so between weekends the
    standings entering round N are read off the *end* of round N-1's
    race session (``position_current``) instead.
    """
    calendar = get_race_calendar(year)
    try:
        return _top10_at_race(calendar, round_num, "position_start")
    except Exception as exc:
        if round_num < 2:
            raise
        print(f"OpenF1 has no pre-race standings for round {round_num} yet "
              f"({exc}); using end-of-race standings from round {round_num - 1}.")
        try:
            return _top10_at_race(calendar, round_num - 1, "position_current")
        except Exception:
            raise RuntimeError(
                f"OpenF1 has no usable standings around round {round_num}. "
                "Set underdogs.source: manual in config.yaml and fill "
                "underdogs.manual_top10 with the current top 10."
            ) from None


def _top10_at_race(calendar, round_num: int, position_col: str) -> list[str]:
    """Championship top-10 at one round's race session, ranked by
    ``position_start`` (entering) or ``position_current`` (leaving)."""
    row = calendar[
        (calendar["round_number"] == round_num)
        & (calendar["session_name"] == "Race")
    ]
    if row.empty:
        raise RuntimeError(f"OpenF1 calendar has no race for round {round_num}")
    session_key = row.iloc[0]["session_key"]

    standings = _fetch_openf1("championship_drivers", session_key=session_key)
    if standings.empty or position_col not in standings.columns:
        raise RuntimeError(
            f"OpenF1 championship standings missing for round {round_num}"
        )
    drivers = _fetch_openf1("drivers", session_key=session_key)

    top10 = (
        standings.sort_values(position_col)
        .head(10)
        .merge(drivers[["driver_number", "name_acronym"]],
               on="driver_number", how="left")
    )
    return top10["name_acronym"].tolist()
