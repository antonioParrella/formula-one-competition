"""Gaussian heterogeneous-dispersion model (DISPERSION.md).

Checks that the simulator's pairwise marginals match the Thurstonian
probit closed form, that the analytic H2H helper agrees with it, and
that the fit recovers a spread the market implies but rank-1 PL cannot.
"""

import numpy as np
import pytest
from scipy.special import ndtr

from model.fit import _analytic_h2h_gaussian, fit_dispersion, fit_strengths
from model.simulate import h2h_prob, simulate_races, top_k_probs

# 3-driver Gaussian toy with heterogeneous spreads.
MU = {"AAA": 1.0, "BBB": 0.0, "CCC": -0.6}
SIGMA = {"AAA": 1.5, "BBB": 1.0, "CCC": 2.2}
NO_DNF = {c: 0.0 for c in MU}
TOL = 0.005


@pytest.fixture(scope="module")
def sims():
    return simulate_races(MU, NO_DNF, n_sims=300_000, seed=3,
                          sigma_by_code=SIGMA, dist="gaussian")


def test_pairwise_marginals_match_probit(sims):
    # P(a before b) = Φ((μa − μb) / sqrt(σa² + σb²)) — Thurstone Case III.
    for a in MU:
        for b in MU:
            if a >= b:
                continue
            analytic = ndtr((MU[a] - MU[b]) /
                            np.sqrt(SIGMA[a] ** 2 + SIGMA[b] ** 2))
            sim = h2h_prob(sims.finish_pos, sims.index_of(a), sims.index_of(b))
            assert sim == pytest.approx(analytic, abs=TOL)


def test_analytic_h2h_gaussian_no_dnf_is_plain_probit():
    p = _analytic_h2h_gaussian(1.0, 0.0, 1.5, 1.0, 0.0, 0.0)
    assert p == pytest.approx(ndtr(1.0 / np.sqrt(1.5**2 + 1.0**2)))


def test_analytic_h2h_gaussian_one_certain_dnf():
    # If b certainly retires and a never does, a is ahead with prob 1.
    assert _analytic_h2h_gaussian(0.0, 5.0, 1.0, 1.0, 0.0, 1.0) == pytest.approx(1.0)
    # If a certainly retires and b never does, a is ahead with prob 0.
    assert _analytic_h2h_gaussian(5.0, 0.0, 1.0, 1.0, 1.0, 0.0) == pytest.approx(0.0)


def test_gaussian_reproducible():
    s1 = simulate_races(MU, NO_DNF, n_sims=1000, seed=9,
                        sigma_by_code=SIGMA, dist="gaussian")
    s2 = simulate_races(MU, NO_DNF, n_sims=1000, seed=9,
                        sigma_by_code=SIGMA, dist="gaussian")
    assert (s1.finish_pos == s2.finish_pos).all()


def test_dispersion_fit_reaches_incoherent_target():
    """A win/top-10 pair that a single strength cannot match (rank-1 PL
    forces top-10 too high for the given win prob). The dispersion fit
    should hit both far better than the PL fit by widening σ."""
    # One "volatile favourite" plus a spread field; craft top-1 and top-10
    # targets that are jointly unreachable at σ ≡ 1.
    drivers = [f"D{i:02d}" for i in range(12)]
    # Win market: favourite at 0.30, rest tapering.
    raw_win = np.array([0.30, 0.16, 0.12, 0.09, 0.07, 0.06,
                        0.05, 0.05, 0.04, 0.03, 0.02, 0.01])
    win = dict(zip(drivers, raw_win / raw_win.sum()))
    # Top-10 market (sums to 10): favourite only 0.86 despite winning 30% —
    # the incoherent signal. Others near-certain, tail low.
    raw_t10 = np.array([0.86, 0.985, 0.985, 0.98, 0.98, 0.97,
                        0.96, 0.95, 0.93, 0.90, 0.80, 0.60])
    t10 = dict(zip(drivers, raw_t10 * (10.0 / raw_t10.sum())))
    market = {
        "drivers": drivers,
        "topk": {1: win, 10: t10},
        "weights": {1: 1.0, 10: 1.0},
        "h2h": [],
        "markets_used": ["win", "top10"],
    }
    dnf = {c: 0.0 for c in drivers}

    pl = fit_strengths(market, dnf, fit_sims=3000, tau=0.15, seed=1)
    gz = fit_dispersion(market, dnf, fit_sims=3000, tau=0.15, seed=1)

    def top10_gap(fit):
        sims = simulate_races(
            fit["theta"], fit["dnf_probs"], n_sims=60_000, seed=2,
            sigma_by_code=fit.get("sigma"), dist=fit.get("model", "gumbel"))
        p1 = top_k_probs(sims.finish_pos, 1)[sims.index_of("D00")]
        p10 = top_k_probs(sims.finish_pos, 10)[sims.index_of("D00")]
        return abs(p1 - win["D00"]) + abs(p10 - t10["D00"])

    # The dispersion model gets materially closer on the favourite's
    # (win, top-10) pair than rank-1 PL.
    assert top10_gap(gz) < top10_gap(pl) - 0.03
    # And it does so by widening the favourite's spread above unit.
    assert gz["sigma"]["D00"] > 1.2
