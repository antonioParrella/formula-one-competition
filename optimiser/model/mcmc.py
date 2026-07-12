"""Affine-invariant ensemble MCMC: the Goodman & Weare stretch move.

Generic gradient-free sampler (MATH.md Section 7.4). It needs only a
deterministic batched log-density and is invariant under affine
transformations of the state, so it self-adapts to posteriors whose
coordinates live on wildly different scales — no proposal covariance to
tune. Used by ``model/bayes.py``; kept free of any F1 specifics so the
Gaussian-target tests exercise it in isolation.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class SampleResult:
    """One ensemble run: the full chain plus bookkeeping."""

    chain: np.ndarray      # (n_steps, n_walkers, dim) float64
    log_prob: np.ndarray   # (n_steps, n_walkers)
    acceptance: float      # mean accept fraction over all proposals


def stretch_sample(
    log_prob_batch: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    n_steps: int,
    seed: int,
    a: float = 2.0,
    progress_every: int = 0,
) -> SampleResult:
    """Run the stretch move from initial walkers ``x0`` (n_walkers, dim).

    Each step updates the two half-ensembles alternately: walker x_j
    picks a partner x_k from the other half, draws z with density
    ∝ 1/sqrt(z) on [1/a, a], proposes y = x_k + z (x_j − x_k) and
    accepts with log-probability (dim−1)·log z + logp(y) − logp(x_j).
    One batched log-density call per half-step.
    """
    x = np.array(x0, dtype=np.float64)
    n_walkers, dim = x.shape
    if n_walkers < 4 or n_walkers % 2:
        raise ValueError(f"Need an even number of walkers >= 4, got {n_walkers}")

    rng = np.random.default_rng(seed)
    lp = np.asarray(log_prob_batch(x), dtype=np.float64)
    if not np.all(np.isfinite(lp)):
        bad = int(np.sum(~np.isfinite(lp)))
        raise ValueError(f"{bad} walker(s) start at non-finite log-probability")

    half = n_walkers // 2
    halves = (np.arange(half), np.arange(half, n_walkers))
    chain = np.empty((n_steps, n_walkers, dim))
    log_prob = np.empty((n_steps, n_walkers))
    n_accept = 0

    for step in range(n_steps):
        for own, other in (halves, halves[::-1]):
            partners = other[rng.integers(0, half, size=half)]
            z = ((a - 1.0) * rng.random(half) + 1.0) ** 2 / a
            y = x[partners] + z[:, None] * (x[own] - x[partners])
            lp_y = np.asarray(log_prob_batch(y), dtype=np.float64)
            log_accept = (dim - 1) * np.log(z) + lp_y - lp[own]
            accept = np.log(rng.random(half)) < log_accept
            x[own[accept]] = y[accept]
            lp[own[accept]] = lp_y[accept]
            n_accept += int(accept.sum())
        chain[step] = x
        log_prob[step] = lp
        if progress_every and (step + 1) % progress_every == 0:
            frac = n_accept / ((step + 1) * n_walkers)
            print(f"  step {step + 1:>5}/{n_steps}  accept {frac:.2f}  "
                  f"mean logp {lp.mean():.1f}")

    return SampleResult(chain=chain, log_prob=log_prob,
                        acceptance=n_accept / (n_steps * n_walkers))


def flatten_chain(chain: np.ndarray, burn_in: int, thin: int) -> np.ndarray:
    """Post-burn, thinned draws pooled over walkers: (S, dim)."""
    kept = chain[burn_in::thin]                      # (n_kept, n_walkers, dim)
    if kept.size == 0:
        raise ValueError(
            f"No draws survive burn_in={burn_in}, thin={thin} on a "
            f"{chain.shape[0]}-step chain"
        )
    return kept.reshape(-1, chain.shape[2])


def split_rhat(chain: np.ndarray, burn_in: int) -> np.ndarray:
    """Split-R̂ per dimension, treating each walker's post-burn chain as a
    chain and splitting it in half (2·n_walkers pseudo-chains).

    Caveat printed by callers: walkers interact, so this understates
    disagreement relative to truly independent chains.
    """
    kept = chain[burn_in:]                           # (T, K, dim)
    T = (kept.shape[0] // 2) * 2
    if T < 4:
        return np.full(chain.shape[2], np.nan)
    # (2K, T/2, dim): first halves then second halves of every walker.
    halves = np.concatenate([kept[: T // 2], kept[T // 2 : T]], axis=1)
    halves = halves.transpose(1, 0, 2)               # (2K, T/2, dim)
    m, t = halves.shape[0], halves.shape[1]
    chain_means = halves.mean(axis=1)                # (2K, dim)
    within = halves.var(axis=1, ddof=1).mean(axis=0)             # W
    between = t * chain_means.var(axis=0, ddof=1)                # B
    var_hat = (t - 1) / t * within + between / t
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(var_hat / within)


def autocorr_time(chain: np.ndarray, burn_in: int, c: float = 5.0) -> np.ndarray:
    """Integrated autocorrelation time per dimension (Sokal window,
    FFT-based, autocorrelations averaged over walkers)."""
    kept = chain[burn_in:]                           # (T, K, dim)
    T = kept.shape[0]
    x = kept - kept.mean(axis=0, keepdims=True)
    n_fft = 1 << (2 * T - 1).bit_length()
    f = np.fft.rfft(x, n=n_fft, axis=0)
    acf = np.fft.irfft(f * np.conjugate(f), n=n_fft, axis=0)[:T]
    acf = acf.mean(axis=1)                           # average walkers: (T, dim)
    with np.errstate(divide="ignore", invalid="ignore"):
        acf = acf / acf[0]
    taus = 2.0 * np.cumsum(acf, axis=0) - 1.0        # (T, dim)
    out = np.empty(chain.shape[2])
    for d in range(chain.shape[2]):
        window = np.argmax(np.arange(T) >= c * taus[:, d])
        out[d] = taus[window if window > 0 else T - 1, d]
    return out
