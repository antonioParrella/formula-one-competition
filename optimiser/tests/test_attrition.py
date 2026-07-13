import numpy as np
import pytest

from attrition import circuit_factor, circuit_rates
from model.simulate import simulate_races


def _sim(dnf_by_code, lam, n_sims=40000, seed=1):
    codes = list(dnf_by_code)
    theta = {c: 0.0 for c in codes}
    return simulate_races(theta, dnf_by_code, n_sims=n_sims, seed=seed,
                          shock_lambda=lam)


def test_shock_preserves_per_driver_marginal():
    # Heterogeneous base rates must survive the shared shock unchanged.
    d = {"AAA": 0.05, "BBB": 0.15, "CCC": 0.30, "DDD": 0.10}
    codes = list(d)
    s = _sim(d, lam=0.9)
    marg = s.dnf.mean(axis=0)
    for i, c in enumerate(codes):
        assert marg[i] == pytest.approx(d[c], abs=0.012)


def test_shock_adds_overdispersion_and_tail():
    d = {f"D{i}": 0.125 for i in range(20)}
    k0 = _sim(d, lam=0.0).dnf.sum(axis=1)
    k1 = _sim(d, lam=0.8).dnf.sum(axis=1)
    assert k0.mean() == pytest.approx(k1.mean(), abs=0.15)   # mean unchanged
    assert k1.var() > 1.4 * k0.var()                         # fatter spread
    assert (k1 >= 7).mean() > 3 * (k0 >= 7).mean()           # much heavier tail


def test_lambda_zero_is_independent():
    d = {f"D{i}": 0.2 for i in range(15)}
    k = _sim(d, lam=0.0).dnf.sum(axis=1)
    indep_var = 15 * 0.2 * 0.8
    assert k.var() == pytest.approx(indep_var, rel=0.1)      # binomial variance


def test_circuit_rates_shrink_toward_grid():
    # One high circuit (few races) and one low; both pulled toward the grid.
    races = ([{"circuit": "HIGH", "starters": 20, "dnf": 6}] * 2
             + [{"circuit": "LOW", "starters": 20, "dnf": 1}] * 2
             + [{"circuit": "MID", "starters": 20, "dnf": 3}] * 20)
    rates, grid = circuit_rates(races, prior_starts=40.0)
    assert grid == pytest.approx((12 + 2 + 60) / (4 * 20 + 20 * 20))
    assert grid < rates["HIGH"] < 6 / 20     # raw 30% shrunk down
    assert 1 / 20 < rates["LOW"] < grid      # raw 5% shrunk up


def test_circuit_factor_edges():
    bundle = {"grid_rate": 0.125, "circuit_rates": {"Spa": 0.06, "Melbourne": 0.24}}
    assert circuit_factor(bundle, "Spa") == pytest.approx(0.06 / 0.125)
    assert circuit_factor(bundle, "Melbourne") == pytest.approx(0.24 / 0.125)
    assert circuit_factor(bundle, "Unknown") == 1.0     # no history -> neutral
    assert circuit_factor({}, "Spa") == 1.0             # disabled -> neutral
