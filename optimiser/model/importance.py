"""Importance sampling machinery: Laplace-t proposal, PSIS, resampling.

Generic pieces of the fast inference engine (MATH.md Section 7.5):
finite-difference curvature at the posterior mode, a multivariate
Student-t proposal, self-normalised importance weights with the
Pareto-k̂ diagnostic of Pareto-smoothed importance sampling (PSIS,
Vehtari et al.), and systematic resampling down to unweighted draws.
No F1 specifics — ``model/bayes.py`` supplies the log-posterior.
"""

from typing import Callable

import numpy as np


def fd_hessian(
    log_prob_batch: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    step: float = 0.05,
    batch: int = 512,
) -> np.ndarray:
    """Central finite-difference Hessian of −log p at ``x`` (dim, dim).

    The step is coarse by design: the log-density runs a float32
    surrogate whose noise floor swamps small steps (same reasoning as
    the fit's eps, model/fit.py). Off-diagonals use the four-point
    stencil; everything is evaluated through batched calls.
    """
    x = np.asarray(x, dtype=np.float64)
    dim = x.shape[0]
    points = [x]
    for i in range(dim):
        for s in (step, -step):
            p = x.copy(); p[i] += s
            points.append(p)
    pairs = [(i, j) for i in range(dim) for j in range(i + 1, dim)]
    for i, j in pairs:
        for si, sj in ((step, step), (step, -step), (-step, step), (-step, -step)):
            p = x.copy(); p[i] += si; p[j] += sj
            points.append(p)

    pts = np.array(points)
    vals = np.empty(len(pts))
    for lo in range(0, len(pts), batch):
        vals[lo:lo + batch] = log_prob_batch(pts[lo:lo + batch])

    f0 = vals[0]
    H = np.empty((dim, dim))
    for i in range(dim):
        f_plus, f_minus = vals[1 + 2 * i], vals[2 + 2 * i]
        H[i, i] = -(f_plus - 2.0 * f0 + f_minus) / step**2
    base = 1 + 2 * dim
    for idx, (i, j) in enumerate(pairs):
        fpp, fpm, fmp, fmm = vals[base + 4 * idx : base + 4 * idx + 4]
        H[i, j] = H[j, i] = -(fpp - fpm - fmp + fmm) / (4.0 * step**2)
    return H


def regularise_hessian(H: np.ndarray, floor: float) -> np.ndarray:
    """Symmetrise and floor the eigenvalues at ``floor`` (> 0).

    Directions the data leave flat (pinned longshots) get near-zero or
    even negative finite-difference curvature; flooring at the prior
    curvature means "propose from the prior's scale there".
    """
    H = 0.5 * (H + H.T)
    eigval, eigvec = np.linalg.eigh(H)
    n_floored = int(np.sum(eigval < floor))
    if n_floored:
        print(f"  Hessian: floored {n_floored}/{len(eigval)} eigenvalue(s) "
              f"at {floor:g} (flat/noisy directions)")
    eigval = np.maximum(eigval, floor)
    return (eigvec * eigval) @ eigvec.T


def mvt_sample(
    rng: np.random.Generator,
    loc: np.ndarray,
    cov_chol: np.ndarray,
    nu: float,
    n: int,
) -> np.ndarray:
    """Draw n samples from a multivariate Student-t(loc, cov, nu)."""
    dim = loc.shape[0]
    z = rng.standard_normal((n, dim))
    g = rng.chisquare(nu, size=n) / nu
    return loc[None, :] + (z @ cov_chol.T) / np.sqrt(g)[:, None]


def mvt_logpdf(
    x: np.ndarray,
    loc: np.ndarray,
    cov_chol: np.ndarray,
    nu: float,
) -> np.ndarray:
    """Log-density of a multivariate Student-t, up to shape constants
    that cancel in self-normalised weights (the log|Σ| term is kept —
    it does not cancel against the target)."""
    dim = loc.shape[0]
    diff = x - loc[None, :]
    y = np.linalg.solve(cov_chol, diff.T)                # (dim, n) Mahalanobis

    maha = (y**2).sum(axis=0)
    log_det = np.sum(np.log(np.diag(cov_chol)))
    return -0.5 * (nu + dim) * np.log1p(maha / nu) - log_det


def ess(log_w: np.ndarray) -> float:
    """Effective sample size of self-normalised weights."""
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    return float(1.0 / np.sum(w**2))


def _gpd_fit(exceedances: np.ndarray) -> tuple[float, float]:
    """Generalised-Pareto (k, sigma) via the Zhang–Stephens (2009)
    posterior-mean estimator, with the weak prior regularisation used
    by PSIS. ``exceedances`` must be sorted ascending and positive."""
    x = exceedances
    n = len(x)
    m = 30 + int(np.sqrt(n))
    b = 1.0 - np.sqrt(m / (np.arange(1, m + 1, dtype=np.float64) - 0.5))
    b /= 3.0 * x[max(int(n / 4 + 0.5) - 1, 0)]
    b += 1.0 / x[-1]
    k = np.mean(np.log1p(-b[:, None] * x[None, :]), axis=1)      # (m,) < 0
    log_lik = n * (np.log(-b / k) - k - 1.0)
    w = 1.0 / np.exp(log_lik - log_lik[:, None]).sum(axis=1)
    b_post = float(np.sum(b * w))
    k_post = float(np.mean(np.log1p(-b_post * x)))
    sigma = -k_post / b_post
    k_post = (n * k_post + 5.0) / (n + 10.0)     # prior nudge toward k = 0.5
    return k_post, sigma                         # heavy tail => k > 0


def psis(log_w: np.ndarray) -> tuple[np.ndarray, float]:
    """Pareto-smoothed importance weights.

    Fits a generalised Pareto to the largest min(0.2 N, 3 sqrt(N))
    raw weights, replaces them with the fitted quantiles (capped at
    the raw maximum), and returns (smoothed log-weights, k̂).
    k̂ < 0.5: reliable; 0.5–0.7: usable; > 0.7: proposal misses the
    posterior — the caller should warn and fall back to MCMC.
    """
    log_w = np.asarray(log_w, dtype=np.float64) - np.max(log_w)
    n = len(log_w)
    n_tail = min(int(0.2 * n), int(3.0 * np.sqrt(n)))
    if n_tail < 5:
        return log_w, np.nan

    order = np.argsort(log_w)
    tail_idx = order[-n_tail:]
    cutoff = np.exp(log_w[order[-n_tail - 1]])
    tail = np.exp(log_w[tail_idx])
    exceed = np.sort(tail) - cutoff
    if exceed[-1] <= 0:
        return log_w, np.nan

    k_hat, sigma = _gpd_fit(np.maximum(exceed, 1e-15))
    if np.isfinite(k_hat):
        q = (np.arange(1, n_tail + 1) - 0.5) / n_tail
        if abs(k_hat) < 1e-6:
            quantiles = -sigma * np.log1p(-q)
        else:
            quantiles = sigma / k_hat * ((1.0 - q) ** (-k_hat) - 1.0)
        smoothed = np.minimum(cutoff + quantiles, np.exp(log_w[order[-1]]))
        out = log_w.copy()
        out[tail_idx[np.argsort(log_w[tail_idx])]] = np.log(smoothed)
        return out, k_hat
    return log_w, k_hat


def systematic_resample(
    rng: np.random.Generator,
    log_w: np.ndarray,
    n: int,
) -> np.ndarray:
    """Systematic resampling: n indices drawn ∝ w with strictly lower
    variance than multinomial resampling."""
    w = np.exp(log_w - np.max(log_w))
    w /= w.sum()
    positions = (rng.random() + np.arange(n)) / n
    return np.searchsorted(np.cumsum(w), positions).clip(max=len(w) - 1)
