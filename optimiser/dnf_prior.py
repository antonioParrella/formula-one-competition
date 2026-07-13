"""Season-to-date DNF priors from OpenF1.

Replaces the flat season-average DNF fallback in ``model.fit.build_dnf_probs``
with each driver's *actual* retirement rate this season. Reuses the parent
repo's OpenF1 helpers (``leaderboard``) — the same source the comp scorer and
``comp_context`` read.

Small samples early in the season are noisy (a driver with 1 DNF in 4 starts
is not a 25% retirement risk), so raw rates are Beta-smoothed toward the
grid-wide average with a pseudo-count ``prior_strength`` k:

    rate_i = (dnf_i + k · grid_avg) / (starts_i + k)

A driver with no starts (a mid-season rookie/reserve) therefore lands on the
grid average rather than a hardcoded number.

Priority in ``build_dnf_probs`` is unchanged; this slots in as the third tier:
config ``per_driver`` > market-implied 1 - P(classified) > **season rate
(this)** > flat ``default_prob``. ``default_prob`` remains the ultimate
fallback for drivers/rounds with no season data (round 1) or when OpenF1 is
unreachable — the prior is a refinement, not core data, so a fetch failure
warns and degrades rather than killing the fit.
"""

import _paths  # noqa: F401

import json
import time
from collections import defaultdict
from pathlib import Path

from leaderboard import _fetch_openf1, get_race_calendar

DATA_DIR = Path(__file__).resolve().parent / "data"


def smooth_rates(
    dnf_counts: dict[str, int],
    start_counts: dict[str, int],
    prior_strength: float,
) -> tuple[dict[str, float], float | None]:
    """Beta-smoothed per-driver DNF rate, shrunk toward the grid average.

    Returns ``(rates, grid_avg)``; ``({}, None)`` when there are no starts.
    """
    total_dnf = sum(dnf_counts.values())
    total_starts = sum(start_counts.values())
    if total_starts == 0:
        return {}, None
    grid = total_dnf / total_starts
    rates = {
        code: (dnf_counts.get(code, 0) + prior_strength * grid)
        / (starts + prior_strength)
        for code, starts in start_counts.items()
    }
    return rates, grid


def recenter_rates(
    rates: dict[str, float], target_mean: float
) -> dict[str, float]:
    """Rescale per-driver rates so their mean equals ``target_mean``, keeping
    the drivers' relative ratios (multiplicative, so ordering and ratios are
    preserved and nothing goes negative).

    This anchors the *level* of attrition to the configured prior while
    letting the season data set only the cross-driver *shape*. It matters
    because the DNF layer caps a driver's finish distribution at
    ``P(top-k) <= 1 - P(DNF)``: an unanchored season average (here ~20%,
    well above ``default_prob``) pushes strong drivers' ceilings below their
    market win/points prices, and the strength fit cannot climb past that
    ceiling, so market alignment collapses. Re-centring to ``default_prob``
    keeps expected attrition where the flat prior used to put it while still
    telling fragile drivers apart from reliable ones.
    """
    if not rates:
        return {}
    current_mean = sum(rates.values()) / len(rates)
    if current_mean <= 0:
        return dict(rates)  # all-zero season: nothing to rescale
    factor = target_mean / current_mean
    return {c: float(min(0.95, max(0.0, r * factor))) for c, r in rates.items()}


def _cache_path(year: int, upto_round: int, include_sprints: bool) -> Path:
    tag = "rs" if include_sprints else "r"
    return DATA_DIR / f"dnf_season_{year}_{tag}{upto_round}.json"


def _fetch_counts(
    year: int, upto_round: int, include_sprints: bool
) -> tuple[dict[str, int], dict[str, int]]:
    """Per-driver (dnf, starts) counts over the season's completed sessions.

    Counts race sessions before ``upto_round`` (the current round hasn't run),
    plus sprints when ``include_sprints``. A DNS is not a start — the driver
    never ran — and is excluded from both numerator and denominator. A DSQ is
    a start but not a DNF: disqualification is a stewards' call, not a
    reliability signal.
    """
    calendar = get_race_calendar(year)
    wanted = ["Race"] + (["Sprint"] if include_sprints else [])
    sessions = calendar[
        calendar["session_name"].isin(wanted)
        & (calendar["round_number"] < upto_round)
    ]
    dnf_counts: dict[str, int] = defaultdict(int)
    start_counts: dict[str, int] = defaultdict(int)
    for session_key in sorted(sessions["session_key"].unique()):
        results = _fetch_openf1("session_result", session_key=int(session_key))
        drivers = _fetch_openf1("drivers", session_key=int(session_key))
        code_by_num = dict(
            zip(drivers["driver_number"], drivers["name_acronym"])
        )
        for _, row in results.iterrows():
            if bool(row.get("dns")):
                continue
            code = code_by_num.get(row["driver_number"])
            if not code:
                continue
            start_counts[code] += 1
            if bool(row.get("dnf")):
                dnf_counts[code] += 1
    return dict(dnf_counts), dict(start_counts)


def _load_or_fetch_counts(
    year: int, upto_round: int, include_sprints: bool, refresh: bool
) -> tuple[dict[str, int], dict[str, int]]:
    """Cached ``(dnf, starts)`` counts. A completed round's results never
    change, so the cache is keyed by ``(year, upto_round, include_sprints)``
    and reused indefinitely; ``refresh`` forces a re-fetch."""
    path = _cache_path(year, upto_round, include_sprints)
    if path.exists() and not refresh:
        cached = json.loads(path.read_text())
        return cached["dnf"], cached["starts"]
    dnf_counts, start_counts = _fetch_counts(year, upto_round, include_sprints)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({
        "year": year,
        "upto_round": upto_round,
        "include_sprints": include_sprints,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dnf": dnf_counts,
        "starts": start_counts,
    }, indent=2))
    return dnf_counts, start_counts


def season_dnf_rates(
    dnf_cfg: dict,
    year: int,
    current_round: int,
    refresh: bool = False,
    verbose: bool = True,
) -> dict[str, float]:
    """Per-driver DNF prior from this season's OpenF1 results.

    Returns ``{}`` — so ``build_dnf_probs`` falls back to ``default_prob`` —
    when the season prior is disabled, no races have run yet (round 1), or
    OpenF1 is unreachable.
    """
    season_cfg = (dnf_cfg or {}).get("season") or {}
    if not season_cfg.get("enabled", True):
        return {}
    if current_round <= 1:
        return {}  # no completed races this season yet
    prior_strength = float(season_cfg.get("prior_strength", 4.0))
    include_sprints = bool(season_cfg.get("include_sprints", False))
    anchor = bool(season_cfg.get("anchor_to_default", True))
    default = float((dnf_cfg or {}).get("default_prob", 0.10))
    try:
        dnf_counts, start_counts = _load_or_fetch_counts(
            year, current_round, include_sprints, refresh)
    except Exception as exc:
        print(f"Season DNF prior: OpenF1 unavailable ({exc}); "
              f"using the flat {default:.0%} prior for all drivers.")
        return {}
    rates, grid = smooth_rates(dnf_counts, start_counts, prior_strength)
    if anchor:
        rates = recenter_rates(rates, default)
    if verbose and rates:
        total_starts = sum(start_counts.values())
        msg = (f"Season DNF prior (OpenF1 {year}, through round "
               f"{current_round - 1}): grid avg {grid:.1%} over {total_starts} "
               f"starts, shrink k={prior_strength:g}")
        if anchor:
            msg += f", re-centered to mean {default:.0%}"
        print(f"{msg}, {len(rates)} drivers")
    return rates
