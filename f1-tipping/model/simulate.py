"""Monte Carlo race engine: Plackett-Luce + independent DNF layer.

Plackett-Luce sampling uses the Gumbel-max trick: adding i.i.d. Gumbel
noise to each strength θᵢ and sorting descending draws a full finishing
order with exactly the PL sequential-softmax distribution — one argsort
per race instead of 20 sequential draws, fully vectorised.

**Heterogeneous dispersion** (``dist="gaussian"``, see DISPERSION.md):
performances become ``Xᵢ = μᵢ + σᵢ·εᵢ`` with εᵢ standard normal and a
per-driver spread σᵢ. This is the Thurstonian generalisation of PL that
gives each driver a *center and a variance*, fixing the rank-1 market
incoherence (a single strength can't match win/top-3/top-6/top-10 at
once). The engine is otherwise identical — draw noise, scale, add,
argsort — so the DNF layer and everything downstream are untouched. The
default (``dist="gumbel"``, ``sigma=None``) is exactly the PL model.

DNF layer: each driver retires independently with probability pᵢ. DNF'd
drivers are demoted behind every survivor (relative order among DNFs
kept PL-consistent), which produces the correlated attrition scenarios
that promote several underdogs into the top 10 at once.
"""

from dataclasses import dataclass

import numpy as np

# Demotion applied to a DNF'd driver's Gumbel score. Gumbel noise spans
# a few units, so this puts every DNF behind every survivor.
DNF_DEMOTION = 1e6


@dataclass(frozen=True)
class SimSet:
    """A fixed set of simulated races (the common-random-numbers set)."""

    drivers: list[str]        # column order for the arrays below
    finish_pos: np.ndarray    # (n_sims, n) int8 — 0-based finishing position
    dnf: np.ndarray           # (n_sims, n) bool

    @property
    def n_sims(self) -> int:
        return self.finish_pos.shape[0]

    def index_of(self, code: str) -> int:
        return self.drivers.index(code)


def sample_finish_positions(
    theta: np.ndarray,
    dnf_probs: np.ndarray,
    n_sims: int,
    rng: np.random.Generator,
    sigma: np.ndarray | None = None,
    dist: str = "gumbel",
) -> tuple[np.ndarray, np.ndarray]:
    """Sample finishing positions for every driver in every race.

    ``theta``/``dnf_probs`` are ``(n,)`` for one parameter set shared by
    every race, or ``(n_sims, n)`` for per-race parameters (the
    posterior-predictive path mixes over posterior draws this way).

    ``dist`` selects the performance-noise law: ``"gumbel"`` (default) is
    the Plackett-Luce model; ``"gaussian"`` is the Thurstonian
    heterogeneous-dispersion model (DISPERSION.md). ``sigma`` is the
    per-driver spread σᵢ (``(n,)`` or ``(n_sims, n)``), applied as
    ``θ + σ·noise``; ``None`` means unit spread (σᵢ ≡ 1), which for
    ``"gumbel"`` recovers PL exactly.

    Returns ``(finish_pos, dnf)`` where ``finish_pos[s, d]`` is driver
    d's 0-based classified position in race s.
    """
    theta = np.atleast_2d(theta)
    dnf_probs = np.atleast_2d(dnf_probs)
    n = theta.shape[-1]
    noise = (rng.standard_normal(size=(n_sims, n)) if dist == "gaussian"
             else rng.gumbel(size=(n_sims, n)))
    if sigma is not None:
        noise = noise * np.atleast_2d(sigma)
    scores = theta + noise
    dnf = rng.random((n_sims, n)) < dnf_probs
    scores = np.where(dnf, scores - DNF_DEMOTION, scores)

    order = np.argsort(-scores, axis=1)              # order[s, pos] = driver
    finish_pos = np.empty((n_sims, n), dtype=np.int64)
    finish_pos[np.arange(n_sims)[:, None], order] = np.arange(n)[None, :]
    return finish_pos.astype(np.int8), dnf


def simulate_races(
    theta_by_code: dict[str, float],
    dnf_probs_by_code: dict[str, float],
    n_sims: int,
    seed: int,
    sigma_by_code: dict[str, float] | None = None,
    dist: str = "gumbel",
) -> SimSet:
    """Simulate ``n_sims`` races for the fitted driver set.

    Pass ``sigma_by_code`` and ``dist="gaussian"`` for the
    heterogeneous-dispersion model (DISPERSION.md); the defaults are the
    Plackett-Luce model. ``theta_by_code`` holds the driver *locations*
    μ (identical to PL strengths in the default path).
    """
    drivers = list(theta_by_code)
    theta = np.array([theta_by_code[c] for c in drivers], dtype=np.float64)
    dnf_probs = np.array([dnf_probs_by_code[c] for c in drivers], dtype=np.float64)
    sigma = None
    if sigma_by_code is not None:
        sigma = np.array([sigma_by_code[c] for c in drivers], dtype=np.float64)

    rng = np.random.default_rng(seed)
    finish_pos, dnf = sample_finish_positions(
        theta, dnf_probs, n_sims, rng, sigma=sigma, dist=dist)
    return SimSet(drivers=drivers, finish_pos=finish_pos, dnf=dnf)


def top_k_probs(finish_pos: np.ndarray, k: int) -> np.ndarray:
    """P(driver finishes in the top k), per driver column."""
    return (finish_pos < k).mean(axis=0)


def h2h_prob(finish_pos: np.ndarray, i: int, j: int) -> float:
    """P(driver column i finishes ahead of driver column j)."""
    return float((finish_pos[:, i] < finish_pos[:, j]).mean())


def mean_finish(finish_pos: np.ndarray) -> np.ndarray:
    """Expected 0-based finishing position, per driver column."""
    return finish_pos.mean(axis=0, dtype=np.float64)
