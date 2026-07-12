"""Plackett-Luce sampling marginals vs analytic values on a 3-driver toy."""

import numpy as np
import pytest

from model.simulate import h2h_prob, simulate_races, top_k_probs

# 3-driver toy: strengths chosen so win probs are exactly (0.5, 0.3, 0.2).
WEIGHTS = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}
THETA = {c: float(np.log(w)) for c, w in WEIGHTS.items()}
NO_DNF = {c: 0.0 for c in WEIGHTS}
N_SIMS = 200_000
TOL = 0.005  # MC noise at 200k sims is ~0.001; generous margin


@pytest.fixture(scope="module")
def sims():
    return simulate_races(THETA, NO_DNF, n_sims=N_SIMS, seed=7)


def test_win_marginals_match_analytic(sims):
    p_win = top_k_probs(sims.finish_pos, 1)
    for code, w in WEIGHTS.items():
        assert p_win[sims.index_of(code)] == pytest.approx(w, abs=TOL)


def test_top2_marginals_match_analytic(sims):
    # P(i in top 2) = p_i + sum_{j != i} p_j * p_i / (1 - p_j)
    p_top2 = top_k_probs(sims.finish_pos, 2)
    for code, w in WEIGHTS.items():
        analytic = w + sum(
            wj * w / (1 - wj) for cj, wj in WEIGHTS.items() if cj != code
        )
        assert p_top2[sims.index_of(code)] == pytest.approx(analytic, abs=TOL)


def test_pairwise_marginals_match_analytic(sims):
    # Exact PL pairwise marginal: P(i before j) = w_i / (w_i + w_j).
    for a in WEIGHTS:
        for b in WEIGHTS:
            if a >= b:
                continue
            analytic = WEIGHTS[a] / (WEIGHTS[a] + WEIGHTS[b])
            sim = h2h_prob(sims.finish_pos, sims.index_of(a), sims.index_of(b))
            assert sim == pytest.approx(analytic, abs=TOL)


def test_certain_dnf_always_finishes_last():
    dnf = {"AAA": 1.0, "BBB": 0.0, "CCC": 0.0}
    sims = simulate_races(THETA, dnf, n_sims=5_000, seed=11)
    a = sims.index_of("AAA")
    assert (sims.finish_pos[:, a] == 2).all()
    assert sims.dnf[:, a].all()


def test_dnf_rate_matches_probability():
    dnf = {"AAA": 0.25, "BBB": 0.0, "CCC": 0.0}
    sims = simulate_races(THETA, dnf, n_sims=100_000, seed=13)
    a = sims.index_of("AAA")
    assert sims.dnf[:, a].mean() == pytest.approx(0.25, abs=0.01)
    # Every DNF is classified behind both survivors.
    assert (sims.finish_pos[sims.dnf[:, a], a] == 2).all()


def test_seeded_runs_are_reproducible():
    s1 = simulate_races(THETA, NO_DNF, n_sims=1_000, seed=42)
    s2 = simulate_races(THETA, NO_DNF, n_sims=1_000, seed=42)
    assert (s1.finish_pos == s2.finish_pos).all()
    assert (s1.dnf == s2.dnf).all()
