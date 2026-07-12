import numpy as np
import pytest

from model.simulate import simulate_races
from optimise.search import greedy_ticket, optimise
from scoring.rules import (
    ScoringContext,
    expected_points_matrix,
    underdog_multipliers,
)

DRIVERS = ["VER", "NOR", "PIA", "LEC", "HAM", "RUS", "ANT", "HAD", "SAI",
           "ALO", "GAS", "STR", "HUL", "TSU"]


@pytest.fixture(scope="module")
def sims():
    rng = np.random.default_rng(1)
    theta = {c: float(t) for c, t in zip(DRIVERS, np.sort(rng.normal(0, 1.5, len(DRIVERS)))[::-1])}
    dnf = {c: 0.1 for c in DRIVERS}
    return simulate_races(theta, dnf, n_sims=20_000, seed=5)


@pytest.fixture(scope="module")
def multipliers(sims):
    ctx = ScoringContext(round_num=4, championship_top10=frozenset(DRIVERS[:10]))
    return underdog_multipliers(sims.drivers, ctx)


def test_optimise_beats_greedy_and_orders_runner_ups(sims, multipliers):
    report = optimise(sims, multipliers, n_restarts=6, max_iters=50, seed=2)

    assert len(report.best_ticket) == 10
    assert len(set(report.best_ticket)) == 10

    exp_points = expected_points_matrix(sims.finish_pos, multipliers)
    greedy = greedy_ticket(sims)
    greedy_ev = exp_points[greedy, np.arange(10)].sum()
    assert report.best_ev >= greedy_ev - 1e-9

    assert len(report.runner_ups) == 5
    for ticket, ev in report.runner_ups:
        assert ev <= report.best_ev + 1e-9
        assert len(set(ticket)) == 10

    pct = report.score_percentiles
    assert pct["p10"] <= pct["p50"] <= pct["p90"]


def test_non_additive_path_runs(sims, multipliers):
    report = optimise(sims, multipliers, n_restarts=2, max_iters=10,
                      assume_additive=False, seed=2)
    assert len(set(report.best_ticket)) == 10
