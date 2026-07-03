"""Plackett-Luce strength calibration.

Fits per-driver strengths θ so the model's marginals reproduce the
de-vigged market probabilities: P(win), P(top3), P(top6), P(top10) and
H2H win rates, weighted by market liquidity.

The top-k probabilities of a PL model with a DNF layer have no closed
form, so the objective uses a *soft-rank* relaxation over a fixed set of
Gumbel/DNF draws: with sim scores s = θ + G (DNF'd drivers demoted),

    rank_i  = Σ_j σ((s_j - s_i)/τ)            soft "drivers ahead of i"
    P(top k) ≈ mean_sims σ((k - ½ - rank_i)/τ)

which is smooth in θ (the draws are fixed), so scipy L-BFGS-B works.
H2H probabilities are exact under PL+DNF and use the analytic form.
``model/validate.py`` re-checks the fit with hard 100k-race sims, which
catches any bias from the relaxation.
"""

import _paths  # noqa: F401

import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from odds.snapshot import race_slug

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SOFT_DNF_DEMOTION = 60.0   # saturates the sigmoids just like the hard sim
THETA_CLIP = np.log(1e-4)  # floor for the log-win-prob initialisation
L2_PENALTY = 1e-4


def build_dnf_probs(drivers: list[str], dnf_cfg: dict) -> dict[str, float]:
    """Per-driver DNF probabilities: season-average prior + overrides."""
    default = float(dnf_cfg.get("default_prob", 0.10))
    per_driver = dnf_cfg.get("per_driver") or {}
    return {c: float(per_driver.get(c, default)) for c in drivers}


def _analytic_h2h(theta_a: float, theta_b: float, dnf_a: float, dnf_b: float) -> float:
    """P(a classified ahead of b) under PL + independent DNFs.

    If exactly one retires the survivor wins; otherwise (both running or
    both out, since DNFs keep their PL order among themselves) the exact
    PL pairwise marginal σ(θa − θb) applies.
    """
    p_pl = expit(theta_a - theta_b)
    one_dnf_a = dnf_a * (1.0 - dnf_b)
    one_dnf_b = dnf_b * (1.0 - dnf_a)
    return (1.0 - one_dnf_a - one_dnf_b) * p_pl + one_dnf_b


def fit_strengths(
    market_probs: dict,
    dnf_probs: dict[str, float],
    fit_sims: int = 2000,
    tau: float = 0.3,
    seed: int = 0,
) -> dict:
    """Calibrate θ to the de-vigged markets.

    ``market_probs`` is the output of ``odds.devig.devig_snapshot``.
    Returns ``{"drivers", "theta", "dnf_probs", "fit_report"}``.
    """
    drivers = market_probs["drivers"]
    n = len(drivers)
    idx = {c: i for i, c in enumerate(drivers)}
    dnf_vec = np.array([dnf_probs[c] for c in drivers], dtype=np.float32)

    # Fixed draws — the objective is deterministic and smooth in theta.
    rng = np.random.default_rng(seed)
    gumbel = rng.gumbel(size=(fit_sims, n)).astype(np.float32)
    dnf_mask = rng.random((fit_sims, n)).astype(np.float32) < dnf_vec[None, :]
    demotion = np.where(dnf_mask, np.float32(SOFT_DNF_DEMOTION), np.float32(0.0))

    # Targets: (k, driver_index_array, target_probs, weight) per market.
    topk_targets = []
    for k, probs in sorted(market_probs["topk"].items()):
        cols = np.array([idx[c] for c in probs], dtype=np.intp)
        target = np.array([probs[c] for c in probs], dtype=np.float64)
        topk_targets.append((k, cols, target, market_probs["weights"][k]))
    h2h_targets = [
        (idx[a], idx[b], p_a, w) for (a, b), p_a, w in market_probs["h2h"]
    ]

    inv_tau = np.float32(1.0 / tau)

    def objective(theta: np.ndarray) -> float:
        theta = theta - theta.mean()
        s = theta.astype(np.float32)[None, :] + gumbel - demotion
        # ahead[m, i, j] = soft indicator that j finishes ahead of i
        ahead = expit((s[:, None, :] - s[:, :, None]) * inv_tau)
        soft_rank = ahead.sum(axis=2) - np.float32(0.5)  # remove self term
        sse = 0.0
        for k, cols, target, weight in topk_targets:
            p_model = expit((np.float32(k - 0.5) - soft_rank) * inv_tau).mean(
                axis=0, dtype=np.float64
            )
            sse += weight * float(((p_model[cols] - target) ** 2).sum())
        for i, j, p_target, weight in h2h_targets:
            p_model = _analytic_h2h(theta[i], theta[j], dnf_vec[i], dnf_vec[j])
            sse += weight * (p_model - p_target) ** 2
        return sse + L2_PENALTY * float((theta**2).sum())

    win_probs = market_probs["topk"][1]
    floor = min(win_probs.values())
    theta0 = np.array(
        [np.log(max(win_probs.get(c, floor), 1e-4)) for c in drivers]
    )
    theta0 = np.clip(theta0, THETA_CLIP, None)
    theta0 -= theta0.mean()

    t0 = time.perf_counter()
    # eps: the objective runs in float32, where scipy's default 1e-8
    # finite-difference step vanishes below the dtype's resolution and
    # stalls L-BFGS with zero gradients. The soft objective is smooth,
    # so a coarse step is fine.
    result = minimize(objective, theta0, method="L-BFGS-B",
                      options={"maxiter": 150, "eps": 1e-3})
    elapsed = time.perf_counter() - t0

    theta = result.x - result.x.mean()
    print(f"Fit: loss {result.fun:.6f} after {result.nit} iterations "
          f"({elapsed:.1f}s), {'converged' if result.success else result.message}")

    return {
        "drivers": drivers,
        "theta": {c: float(theta[idx[c]]) for c in drivers},
        "dnf_probs": {c: float(dnf_probs[c]) for c in drivers},
        "fit_report": {
            "loss": float(result.fun),
            "iterations": int(result.nit),
            "converged": bool(result.success),
            "fit_sims": fit_sims,
            "tau": tau,
            "seed": seed,
            "markets_used": market_probs["markets_used"],
        },
    }


def save_fit(fit: dict, race_name: str, data_dir: Path = DATA_DIR) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"model_fit_{race_slug(race_name)}.json"
    path.write_text(json.dumps(fit, indent=2))
    print(f"Model fit saved -> {path}")
    return path


def load_fit(race_name: str, data_dir: Path = DATA_DIR) -> dict:
    path = data_dir / f"model_fit_{race_slug(race_name)}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No model fit at {path}. Run `python main.py fit` first."
        )
    return json.loads(path.read_text())
