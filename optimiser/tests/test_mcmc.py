"""Stretch-move sampler recovers a known Gaussian target."""

import numpy as np
import pytest

from model.mcmc import autocorr_time, flatten_chain, split_rhat, stretch_sample

# 4-D anisotropic Gaussian: scales spanning two orders of magnitude, the
# geometry the affine-invariant move is supposed to shrug off.
MEAN = np.array([1.0, -2.0, 0.5, 3.0])
SD = np.array([0.1, 1.0, 5.0, 0.5])


def log_prob(x: np.ndarray) -> np.ndarray:
    return -0.5 * (((x - MEAN) / SD) ** 2).sum(axis=1)


@pytest.fixture(scope="module")
def result():
    rng = np.random.default_rng(0)
    x0 = MEAN + SD * rng.standard_normal((32, 4))
    return stretch_sample(log_prob, x0, n_steps=2000, seed=1)


def test_recovers_mean_and_scale(result):
    draws = flatten_chain(result.chain, burn_in=500, thin=5)
    err = np.abs(draws.mean(axis=0) - MEAN) / SD
    assert (err < 0.1).all(), f"standardised mean error {err}"
    assert np.allclose(draws.std(axis=0), SD, rtol=0.2)


def test_diagnostics_healthy(result):
    assert 0.15 < result.acceptance < 0.7
    rhat = split_rhat(result.chain, burn_in=500)
    assert rhat.shape == (4,)
    assert np.nanmax(rhat) < 1.1
    tau = autocorr_time(result.chain, burn_in=500)
    assert tau.shape == (4,)
    assert (tau > 0).all()


def test_seeded_chains_are_reproducible():
    x0 = MEAN + SD * np.random.default_rng(3).standard_normal((16, 4))
    r1 = stretch_sample(log_prob, x0, n_steps=50, seed=9)
    r2 = stretch_sample(log_prob, x0, n_steps=50, seed=9)
    assert (r1.chain == r2.chain).all()
    assert r1.acceptance == r2.acceptance


def test_rejects_odd_or_tiny_ensembles():
    with pytest.raises(ValueError):
        stretch_sample(log_prob, np.zeros((5, 4)), n_steps=10, seed=0)
    with pytest.raises(ValueError):
        stretch_sample(log_prob, np.zeros((2, 4)), n_steps=10, seed=0)
