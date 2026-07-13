"""Correlated-attrition calibration from OpenF1 history (ATTRITION.md).

Real races retire cars in clusters, so the per-race DNF count is
overdispersed (var ≈ 1.4× the independent-Bernoulli value over 2023-26)
and the "many cars out" tail is ~5× heavier than independent draws
predict. The simulator models this with one shared race shock Zᵣ ~ N(0,1)
of loading ``λ`` (``model.simulate``); this module calibrates ``λ`` to the
historical DNF-count tail and estimates per-circuit base rates.

Everything is derived from the same OpenF1 ``session_result`` data as the
per-driver season prior (``dnf_prior``), pooled over several seasons.
Results cache to ``data/attrition_calibration.json`` (raw counts to
``data/attrition_races_<years>.json``) so the fit reads them instantly.
"""

import _paths  # noqa: F401

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from leaderboard import _fetch_openf1, get_race_calendar

DATA_DIR = Path(__file__).resolve().parent / "data"
TAIL_THRESHOLDS = (4, 5, 6, 7, 8)   # DNF-count tail P(K>=t) we calibrate to


# --------------------------------------------------------------------------
# data collection: (n_starters, k_dnf, circuit) per completed race
# --------------------------------------------------------------------------
def _fetch_races(years: list[int]) -> list[dict]:
    races = []
    for year in years:
        calendar = get_race_calendar(year)
        rows = calendar[calendar["session_name"] == "Race"]
        for _, r in rows.iterrows():
            sk = int(r["session_key"])
            try:
                res = _fetch_openf1("session_result", session_key=sk)
                if res.empty or "position" not in res.columns:
                    continue
            except Exception:
                continue                        # future/unpublished race
            started = res[~res["dns"].fillna(False).astype(bool)]
            if len(started) < 10:
                continue                        # incomplete data
            races.append({
                "year": year,
                "circuit": r["circuit_short_name"],
                "starters": int(len(started)),
                "dnf": int(started["dnf"].fillna(False).astype(bool).sum()),
            })
    return races


def _load_races(years: list[int], refresh: bool) -> list[dict]:
    tag = f"{min(years)}_{max(years)}"
    path = DATA_DIR / f"attrition_races_{tag}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    races = _fetch_races(years)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(races, indent=1))
    return races


# --------------------------------------------------------------------------
# per-circuit base rates (Beta-smoothed toward the grid average)
# --------------------------------------------------------------------------
def circuit_rates(races: list[dict], prior_starts: float = 40.0
                  ) -> tuple[dict[str, float], float]:
    """Smoothed DNF rate per circuit and the grid-wide rate.

    Circuits vary 5%-24% (Spa/Monza low, Melbourne/Mexico high), but a
    single track has few races, so shrink toward the grid average with
    ``prior_starts`` pseudo-starts.
    """
    grid_dnf = sum(r["dnf"] for r in races)
    grid_starts = sum(r["starters"] for r in races)
    grid = grid_dnf / grid_starts
    by_circ = defaultdict(lambda: [0, 0])
    for r in races:
        by_circ[r["circuit"]][0] += r["dnf"]
        by_circ[r["circuit"]][1] += r["starters"]
    rates = {
        c: (dnf + prior_starts * grid) / (starts + prior_starts)
        for c, (dnf, starts) in by_circ.items()
    }
    return rates, grid


# --------------------------------------------------------------------------
# shock-loading calibration to the DNF-count tail
# --------------------------------------------------------------------------
def _sim_tail(n: int, base_rate: float, lam: float, thresholds, n_sims=200_000,
              seed=0) -> dict[int, float]:
    """P(K>=t) for a homogeneous n-car field under the shared-shock model."""
    rng = np.random.default_rng(seed)
    scale = np.sqrt(1.0 + lam * lam * (np.pi / 8.0))
    b = np.log(base_rate / (1.0 - base_rate)) * scale        # mean-preserving
    z = rng.standard_normal((n_sims, 1))
    d = 1.0 / (1.0 + np.exp(-(b + lam * z)))
    k = (rng.random((n_sims, n)) < d).sum(axis=1)
    return {t: float((k >= t).mean()) for t in thresholds}


def calibrate_lambda(races: list[dict], thresholds=TAIL_THRESHOLDS
                     ) -> tuple[float, dict, dict]:
    """Loading ``λ`` whose simulated DNF-count tail best matches history.

    Grid-search λ minimising squared error on the tail CCDF P(K>=t); the
    tail (not the variance) is what the underdog bonus is sensitive to.
    Returns ``(lambda, observed_tail, fitted_tail)``.
    """
    n_bar = int(round(np.mean([r["starters"] for r in races])))
    grid = sum(r["dnf"] for r in races) / sum(r["starters"] for r in races)
    obs = {t: float(np.mean([r["dnf"] >= t for r in races])) for t in thresholds}
    best_lam, best_err, best_fit = 0.0, np.inf, None
    for lam in np.linspace(0.0, 1.6, 65):
        fit = _sim_tail(n_bar, grid, lam, thresholds)
        err = sum((fit[t] - obs[t]) ** 2 for t in thresholds)
        if err < best_err:
            best_lam, best_err, best_fit = float(lam), err, fit
    return best_lam, obs, best_fit


# --------------------------------------------------------------------------
# public entry point: cached calibration bundle
# --------------------------------------------------------------------------
def load_calibration(att_cfg: dict, refresh: bool = False) -> dict:
    """``{grid_rate, lambda, circuit_rates, years, tail}`` — cached.

    ``att_cfg`` is ``config.model.dnf.attrition``. Returns ``{}`` (shock
    off, no circuit adjustment) if disabled or OpenF1 is unavailable — the
    correlated layer is an enhancement, so it degrades rather than fails.
    """
    if not att_cfg.get("enabled", True):
        return {}
    years = [int(y) for y in att_cfg.get("calibration_years", [2023, 2024, 2025])]
    path = DATA_DIR / "attrition_calibration.json"
    if path.exists() and not refresh:
        cached = json.loads(path.read_text())
        if cached.get("years") == years:
            return cached
    try:
        races = _load_races(years, refresh)
    except Exception as exc:
        print(f"Attrition: OpenF1 unavailable ({exc}); correlated layer off.")
        return {}
    rates, grid = circuit_rates(races, float(att_cfg.get("circuit_prior_starts", 40.0)))
    lam, obs, fit = calibrate_lambda(races)
    bundle = {
        "years": years,
        "n_races": len(races),
        "grid_rate": grid,
        "lambda": lam,
        "circuit_rates": rates,
        "tail_observed": {str(k): v for k, v in obs.items()},
        "tail_fitted": {str(k): v for k, v in fit.items()},
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(bundle, indent=1))
    return bundle


def race_circuit(year: int, round_num: int) -> str | None:
    """OpenF1 ``circuit_short_name`` for the configured race, or None."""
    try:
        cal = get_race_calendar(year)
        row = cal[(cal["round_number"] == round_num)
                  & (cal["session_name"] == "Race")]
        return None if row.empty else str(row.iloc[0]["circuit_short_name"])
    except Exception:
        return None


def circuit_factor(bundle: dict, circuit: str | None) -> float:
    """Multiplier on the DNF *level* for ``circuit`` = circuit_rate / grid.

    1.0 when circuit conditioning is off, the circuit is unknown, or there
    is no history for it. Spa ≈ 0.4 (low), Melbourne ≈ 1.9 (high).
    """
    if not bundle or not circuit:
        return 1.0
    rate = bundle.get("circuit_rates", {}).get(circuit)
    grid = bundle.get("grid_rate")
    if not rate or not grid:
        return 1.0
    return rate / grid
