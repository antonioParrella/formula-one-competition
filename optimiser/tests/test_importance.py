"""Importance-sampling machinery on known targets."""

import numpy as np
import pytest

from model.importance import (
    ess,
    fd_hessian,
    mvt_logpdf,
    mvt_sample,
    psis,
    regularise_hessian,
    systematic_resample,
)

MEAN = np.array([1.0, -1.0, 0.0])
SD = np.array([0.5, 2.0, 1.0])


def log_prob(x: np.ndarray) -> np.ndarray:
    return -0.5 * (((x - MEAN) / SD) ** 2).sum(axis=1)


def test_is_recovers_gaussian_moments():
    rng = np.random.default_rng(0)
    cov_chol = np.diag(SD)                 # exact scale, t tails dominate
    xs = mvt_sample(rng, MEAN, cov_chol, nu=7.0, n=20_000)
    log_w, k_hat = psis(log_prob(xs) - mvt_logpdf(xs, MEAN, cov_chol, nu=7.0))
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    assert k_hat < 0.5                     # good proposal => light-tailed weights
    assert ess(log_w) > 5_000
    mean_est = (w[:, None] * xs).sum(axis=0)
    assert np.allclose(mean_est, MEAN, atol=0.05 * SD)


def test_too_narrow_proposal_is_caught_by_khat():
    rng = np.random.default_rng(1)
    cov_chol = np.diag(SD / 6.0)           # proposal far tighter than target
    xs = mvt_sample(rng, MEAN, cov_chol, nu=7.0, n=4_000)
    _, k_hat = psis(log_prob(xs) - mvt_logpdf(xs, MEAN, cov_chol, nu=7.0))
    assert k_hat > 0.7                     # the diagnostic must fire


def test_systematic_resample_preserves_weighted_mean():
    rng = np.random.default_rng(2)
    xs = rng.standard_normal((10_000, 2))
    log_w = -0.5 * ((xs - 1.0) ** 2).sum(axis=1) + 0.5 * (xs**2).sum(axis=1)
    idx = systematic_resample(np.random.default_rng(3), log_w, 5_000)
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    weighted_mean = (w[:, None] * xs).sum(axis=0)
    assert np.allclose(xs[idx].mean(axis=0), weighted_mean, atol=0.08)


def test_fd_hessian_matches_analytic_gaussian():
    H = fd_hessian(log_prob, MEAN.copy(), step=0.05)
    assert np.allclose(H, np.diag(1.0 / SD**2), atol=1e-4)


def test_regularise_hessian_floors_flat_directions():
    H = np.diag([4.0, 0.0, -1.0])          # one flat, one noise-negative
    H_reg = regularise_hessian(H, floor=0.25)
    eigval = np.linalg.eigvalsh(H_reg)
    assert eigval.min() == pytest.approx(0.25)
    assert eigval.max() == pytest.approx(4.0)
