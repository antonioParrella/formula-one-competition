"""Bayesian inference over the race model: P(theta | odds).

MATH.md Section 7. The de-vigged market probabilities are treated as
noisy logit-scale observations of the model's marginals; the latent
state is x = (theta, eta, gamma) — Plackett-Luce strengths, logit-scale
DNF parameters for classified-priced drivers, and the log of a global
observation-noise scale sigma_obs (which absorbs rank-1 model misfit).

Two engines share one log-posterior:
- ``method: mcmc`` — affine-invariant ensemble MCMC (model/mcmc.py),
  robust default, sequential (~minutes);
- ``method: is``   — Laplace-t importance sampling with PSIS + SIR
  (model/importance.py), one-shot and batchable (~a minute), guarded
  by the Pareto-k̂ diagnostic.

Posterior-predictive races are packaged as a plain SimSet (S retained
draws x n_sims/S races each), so validate/optimise consume them
unchanged.
"""

import _paths  # noqa: F401

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from model.fit import SOFT_DNF_DEMOTION, _analytic_h2h
from model.importance import (
    ess,
    fd_hessian,
    mvt_logpdf,
    mvt_sample,
    psis,
    regularise_hessian,
    systematic_resample,
)
from model.mcmc import autocorr_time, flatten_chain, split_rhat, stretch_sample
from model.simulate import SimSet, sample_finish_positions
from odds.snapshot import race_slug

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DEFAULTS = {
    "method": "mcmc",       # mcmc | is
    "fit_sims": 1000,       # CRN draws behind the surrogate likelihood
    "tau": 0.15,            # soft-rank temperature (matches model.fit_tau)
    "batch": 16,            # walkers per surrogate batch (memory bound)
    "tau_prior": 2.0,       # N(0, tau_prior^2) prior on raw theta
    # Observation scale. "probability": residuals in probability units,
    # matching the point fit's objective and the validation metric.
    # "logit" was measured to fail on real snapshots: pinned longshot
    # prices produce logit residuals ~2.5 that dominate the likelihood,
    # inflate sigma_obs and wash out the liquid favourites (MATH.md 7.2).
    "obs_scale": "probability",
    "sigma_scale": 0.05,    # HalfNormal prior scale for sigma_obs, in obs
                            # units (0.05 = 5pp on the probability scale;
                            # use ~0.5 if obs_scale is logit)
    "dnf_prior_sd": 0.7,    # logit-normal sd around the season DNF prior
    "seed": 20260703,
    # mcmc
    "walkers": 64,          # auto-raised to 2*D (even)
    "steps": 1200,
    "burn_in": 400,
    "thin": 50,
    # is
    "is_draws": 8000,
    "is_nu": 7.0,
    "is_scale": 1.3,
    "resample_draws": 1024,
    # validation credible intervals
    "ci": 0.90,
    "ci_draws": 200,
    "ci_sims_per_draw": 4000,
}


def _cfg(bayes_cfg: dict | None) -> dict:
    return {**DEFAULTS, **(bayes_cfg or {})}


def build_log_posterior(
    market_probs: dict,
    dnf_probs: dict[str, float],
    dnf_cfg: dict,
    bayes_cfg: dict | None = None,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict]:
    """Batched log-posterior over x = (theta, eta, gamma).

    Returns ``(log_prob_batch, meta)`` where ``log_prob_batch`` maps a
    (W, D) block of states to (W,) log-densities (chunked internally),
    and ``meta`` carries the parameter layout and initial state.
    """
    cfg = _cfg(bayes_cfg)
    drivers = market_probs["drivers"]
    n = len(drivers)
    idx = {c: i for i, c in enumerate(drivers)}

    M = int(cfg["fit_sims"])
    inv_tau = np.float32(1.0 / float(cfg["tau"]))
    chunk = max(1, int(cfg["batch"]))
    eps = max(1e-4, 1.0 / (2.0 * M))
    obs_scale = str(cfg["obs_scale"]).lower()
    if obs_scale not in ("probability", "logit"):
        raise ValueError(f"Unknown bayes.obs_scale {obs_scale!r}")

    def _obs(p):
        """Map a probability (scalar or array) to observation units."""
        p = np.clip(p, eps, 1.0 - eps)
        return logit(p) if obs_scale == "logit" else p

    # Fixed CRN draws: the log-posterior is deterministic in x.
    rng = np.random.default_rng(int(cfg["seed"]))
    gumbel = rng.gumbel(size=(M, n)).astype(np.float32)
    uniforms = rng.random((M, n)).astype(np.float32)

    # Which drivers get a sampled eta: classified-priced, not config-pinned.
    per_driver = (dnf_cfg or {}).get("per_driver") or {}
    market_dnf = market_probs.get("dnf") or {}
    eta_codes = [c for c in drivers if c in market_dnf and c not in per_driver]
    m = len(eta_codes)
    eta_cols = np.array([idx[c] for c in eta_codes], dtype=np.intp)
    d_fixed64 = np.array([dnf_probs[c] for c in drivers], dtype=np.float64)
    d_fixed = d_fixed64.astype(np.float32)
    fixed_demotion = np.where(uniforms < d_fixed[None, :],
                              np.float32(SOFT_DNF_DEMOTION), np.float32(0.0))

    # Observation targets, in observation units (probability or logit).
    topk_targets = []
    n_obs = 0
    for k, probs in sorted(market_probs["topk"].items()):
        cols = np.array([idx[c] for c in probs], dtype=np.intp)
        y = np.array([_obs(probs[c]) for c in probs], dtype=np.float64)
        topk_targets.append((k, cols, y, float(market_probs["weights"][k])))
        n_obs += len(cols)
    h2h_targets = [
        (idx[a], idx[b], float(_obs(p_a)), float(w))
        for (a, b), p_a, w in market_probs["h2h"]
    ]
    n_obs += len(h2h_targets)
    dnf_weights = market_probs.get("dnf_weights") or {}
    cls_targets = [
        (e, float(_obs(1.0 - market_dnf[c])), float(dnf_weights.get(c, 1.0)))
        for e, c in enumerate(eta_codes)
    ]
    n_obs += len(cls_targets)

    tau_p = float(cfg["tau_prior"])
    sigma_sc = float(cfg["sigma_scale"])
    s_d = float(cfg["dnf_prior_sd"])
    eta0 = float(logit(np.clip(float((dnf_cfg or {}).get("default_prob", 0.10)),
                               1e-4, 1 - 1e-4)))
    D = n + m + 1

    def log_prob_batch(X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        out = np.empty(X.shape[0])
        for lo in range(0, X.shape[0], chunk):
            xb = X[lo:lo + chunk]
            W = xb.shape[0]
            theta_raw = xb[:, :n]
            theta = theta_raw - theta_raw.mean(axis=1, keepdims=True)
            gamma = xb[:, -1]

            if m:
                eta = xb[:, n:n + m]
                d = np.tile(d_fixed, (W, 1))
                d[:, eta_cols] = expit(eta).astype(np.float32)
                demotion = np.where(uniforms[None, :, :] < d[:, None, :],
                                    np.float32(SOFT_DNF_DEMOTION),
                                    np.float32(0.0))
            else:
                d = np.tile(d_fixed, (W, 1))
                demotion = np.broadcast_to(fixed_demotion, (W, M, n))

            s = theta.astype(np.float32)[:, None, :] + gumbel[None] - demotion
            ahead = expit((s[:, :, None, :] - s[:, :, :, None]) * inv_tau)
            soft_rank = ahead.sum(axis=3) - np.float32(0.5)   # (W, M, n)

            wsse = np.zeros(W)
            for k, cols, y, weight in topk_targets:
                p_model = expit((np.float32(k - 0.5) - soft_rank) * inv_tau
                                ).mean(axis=1, dtype=np.float64)   # (W, n)
                mu = _obs(p_model[:, cols])
                wsse += weight * ((y[None, :] - mu) ** 2).sum(axis=1)
            for i, j, y, weight in h2h_targets:
                p_model = _analytic_h2h(theta[:, i], theta[:, j],
                                        d[:, i].astype(np.float64),
                                        d[:, j].astype(np.float64))
                wsse += weight * (y - _obs(p_model)) ** 2
            for e, y, weight in cls_targets:
                eta_e = xb[:, n + e]
                mu = (-eta_e if obs_scale == "logit"
                      else 1.0 - expit(eta_e))
                wsse += weight * (y - mu) ** 2

            ll = -0.5 * wsse * np.exp(-2.0 * gamma) - n_obs * gamma
            lp = (-0.5 * (theta_raw ** 2).sum(axis=1) / tau_p**2
                  - np.exp(2.0 * gamma) / (2.0 * sigma_sc**2) + gamma)
            if m:
                lp -= 0.5 * ((xb[:, n:n + m] - eta0) ** 2).sum(axis=1) / s_d**2
            out[lo:lo + chunk] = ll + lp
        return out

    meta = {
        "drivers": drivers,
        "n": n,
        "m": m,
        "D": D,
        "eta_codes": eta_codes,
        "eta_cols": eta_cols,
        "d_fixed": d_fixed64,
        "n_obs": n_obs,
        "eps": eps,
        "cfg": cfg,
        "markets_used": market_probs.get("markets_used", []),
    }
    return log_prob_batch, meta


def _initial_state(meta: dict, map_fit: dict, market_dnf: dict) -> np.ndarray:
    """(D,) initial state: MAP theta, market-implied eta, data-driven gamma."""
    drivers, n, m = meta["drivers"], meta["n"], meta["m"]
    x = np.empty(meta["D"])
    theta = np.array([map_fit["theta"][c] for c in drivers])
    x[:n] = theta - theta.mean()
    for e, code in enumerate(meta["eta_codes"]):
        x[n + e] = logit(np.clip(market_dnf[code], 1e-4, 1 - 1e-4))
    # gamma init: the MAP fit's weighted SSE is in probability units, so
    # sqrt(loss / n_obs) is the residual scale the data already implies;
    # start 1.5x wide of it. Logit scale has no such shortcut.
    loss = (map_fit.get("fit_report") or {}).get("loss")
    if meta["cfg"]["obs_scale"] == "probability" and loss:
        sigma0 = 1.5 * np.sqrt(max(float(loss), 1e-8) / meta["n_obs"])
        x[-1] = np.log(max(sigma0, 1e-3))
    else:
        x[-1] = np.log(0.15)
    return x


def fit_posterior(
    market_probs: dict,
    dnf_probs: dict[str, float],
    dnf_cfg: dict,
    map_fit: dict,
    bayes_cfg: dict | None = None,
) -> dict:
    """Sample P(theta, eta, gamma | odds) with the configured engine.

    Returns ``{"drivers", "theta" (S, n), "dnf" (S, n), "log_sigma" (S,),
    "log_prob" (S,), "method", "bayes_report"}`` — theta draws are
    mean-centred, dnf draws mix sampled and fixed columns.
    """
    cfg = _cfg(bayes_cfg)
    log_prob, meta = build_log_posterior(market_probs, dnf_probs, dnf_cfg, cfg)
    n, m, D = meta["n"], meta["m"], meta["D"]
    x_init = _initial_state(meta, map_fit, market_probs.get("dnf") or {})
    method = str(cfg["method"]).lower()
    print(f"\nBayes ({method}): D = {D} ({n} theta"
          + (f" + {m} eta" if m else "") + " + 1 log-sigma), "
          f"{meta['n_obs']} observations, M = {cfg['fit_sims']} CRN draws")

    t0 = time.perf_counter()
    if method == "mcmc":
        draws, log_probs, report = _run_mcmc(log_prob, x_init, cfg, D)
    elif method == "is":
        draws, log_probs, report = _run_is(log_prob, x_init, cfg, D)
    else:
        raise ValueError(f"Unknown bayes.method {method!r} (mcmc | is)")
    elapsed = time.perf_counter() - t0
    print(f"Bayes ({method}): {draws.shape[0]} retained draws in {elapsed:.1f}s")

    theta_draws = draws[:, :n]
    theta_draws = theta_draws - theta_draws.mean(axis=1, keepdims=True)
    dnf_draws = np.tile(meta["d_fixed"], (draws.shape[0], 1))
    if m:
        dnf_draws[:, meta["eta_cols"]] = expit(draws[:, n:n + m])

    sigma_med = float(np.median(np.exp(draws[:, -1])))
    print(f"Posterior sigma_obs median: {sigma_med:.3f} "
          f"({cfg['obs_scale']} scale) — the realised market-vs-model misfit")

    return {
        "drivers": meta["drivers"],
        "theta": theta_draws,
        "dnf": dnf_draws,
        "log_sigma": draws[:, -1],
        "log_prob": log_probs,
        "method": method,
        "bayes_report": {
            **report,
            "config": {k: cfg[k] for k in DEFAULTS},
            "n_obs": meta["n_obs"],
            "eta_codes": meta["eta_codes"],
            "runtime_s": round(elapsed, 1),
            "markets_used": meta["markets_used"],
        },
    }


def _run_mcmc(log_prob, x_init: np.ndarray, cfg: dict, D: int):
    walkers = max(int(cfg["walkers"]), 2 * D)
    walkers += walkers % 2
    steps, burn, thin = int(cfg["steps"]), int(cfg["burn_in"]), int(cfg["thin"])
    seed = int(cfg["seed"])

    jitter_rng = np.random.default_rng(seed + 1)
    x0 = x_init[None, :] + 0.02 * jitter_rng.standard_normal((walkers, D))
    x0[:, -1] = x_init[-1] + 0.1 * jitter_rng.standard_normal(walkers)

    print(f"Ensemble MCMC: {walkers} walkers x {steps} steps "
          f"(burn {burn}, thin {thin})")
    result = stretch_sample(log_prob, x0, steps, seed=seed + 2,
                            progress_every=100)

    rhat = split_rhat(result.chain, burn)
    tau_int = autocorr_time(result.chain, burn)
    max_rhat = float(np.nanmax(rhat))
    max_tau = float(np.nanmax(tau_int))
    print(f"Diagnostics: acceptance {result.acceptance:.2f}, "
          f"max split-Rhat {max_rhat:.3f}, max tau_int {max_tau:.1f}")
    if not 0.15 <= result.acceptance <= 0.6:
        print("WARNING: acceptance outside [0.15, 0.6] — check walker init "
              "or raise steps")
    if max_rhat > 1.05:
        print("WARNING: split-Rhat > 1.05 — chain not converged, raise steps")
    if steps - burn < 50 * max_tau:
        print(f"WARNING: post-burn run ({steps - burn}) < 50 x tau_int "
              f"({max_tau:.0f}) — draws are strongly correlated")

    draws = flatten_chain(result.chain, burn, thin)
    log_probs = result.log_prob[burn::thin].reshape(-1)
    report = {
        "engine": "mcmc",
        "walkers": walkers,
        "steps": steps,
        "burn_in": burn,
        "thin": thin,
        "acceptance": round(result.acceptance, 3),
        "max_split_rhat": round(max_rhat, 3),
        "max_tau_int": round(max_tau, 1),
    }
    return draws, log_probs, report


def _run_is(log_prob, x_init: np.ndarray, cfg: dict, D: int):
    seed = int(cfg["seed"])
    n_is = int(cfg["is_draws"])
    nu, scale = float(cfg["is_nu"]), float(cfg["is_scale"])

    print("Laplace mode search (L-BFGS-B on the log-posterior)...")
    neg = lambda v: -float(log_prob(v[None])[0])  # noqa: E731
    opt = minimize(neg, x_init, method="L-BFGS-B",
                   options={"maxiter": 200, "eps": 1e-3})
    mode = opt.x
    print(f"  mode logp {-opt.fun:.2f} after {opt.nit} iterations "
          f"({'converged' if opt.success else opt.message})")

    print("Finite-difference Hessian + Laplace-t proposal...")
    H = fd_hessian(log_prob, mode, step=0.05)
    H = regularise_hessian(H, floor=1.0 / float(cfg["tau_prior"]) ** 2)
    cov = scale**2 * np.linalg.inv(H)
    cov_chol = np.linalg.cholesky(0.5 * (cov + cov.T))

    def _round(rng, loc, chol):
        xs = mvt_sample(rng, loc, chol, nu, n_is)
        log_q = mvt_logpdf(xs, loc, chol, nu)
        log_p = log_prob(xs)
        log_w, k_hat = psis(log_p - log_q)
        return xs, log_p, log_w, k_hat

    rng = np.random.default_rng(seed + 2)
    xs, log_p, log_w, k_hat = _round(rng, mode, cov_chol)
    n_rounds = 1
    print(f"Importance sampling: {n_is} draws, ESS {ess(log_w):.0f}, "
          f"Pareto k-hat {k_hat:.2f}")

    if np.isfinite(k_hat) and k_hat > 0.4:
        # One adaptive refinement: moment-match the proposal to the
        # PSIS-weighted draws (re-centres past any mode-search slack and
        # widens along the directions the Laplace curvature missed).
        w = np.exp(log_w - log_w.max())
        w /= w.sum()
        loc2 = w @ xs
        diff = xs - loc2[None, :]
        cov2 = scale**2 * (diff.T * w) @ diff + 1e-8 * np.eye(D)
        try:
            chol2 = np.linalg.cholesky(0.5 * (cov2 + cov2.T))
        except np.linalg.LinAlgError:
            chol2 = None
        if chol2 is not None:
            xs2, log_p2, log_w2, k_hat2 = _round(rng, loc2, chol2)
            print(f"  refined proposal: ESS {ess(log_w2):.0f}, "
                  f"Pareto k-hat {k_hat2:.2f}")
            if not np.isfinite(k_hat) or (np.isfinite(k_hat2)
                                          and k_hat2 < k_hat):
                xs, log_p, log_w, k_hat = xs2, log_p2, log_w2, k_hat2
            n_rounds = 2

    ess_val = ess(log_w)
    if np.isfinite(k_hat) and k_hat > 0.7:
        print("WARNING: k-hat > 0.7 — the proposal misses the posterior; "
              "rerun with bayes.method: mcmc")
    elif np.isfinite(k_hat) and k_hat > 0.5:
        print("NOTE: k-hat in (0.5, 0.7] — usable but noisy; prefer mcmc "
              "for the final fit")

    n_keep = int(cfg["resample_draws"])
    sel = systematic_resample(np.random.default_rng(seed + 3), log_w, n_keep)
    draws = xs[sel]
    log_probs = log_p[sel]
    n_unique = int(len(np.unique(sel)))
    print(f"SIR: {n_keep} draws resampled ({n_unique} unique)")

    report = {
        "engine": "is",
        "is_draws": n_is,
        "nu": nu,
        "scale": scale,
        "rounds": n_rounds,
        "mode_logp": round(-float(opt.fun), 3),
        "mode_converged": bool(opt.success),
        "ess": round(ess_val, 1),
        "pareto_khat": round(float(k_hat), 3) if np.isfinite(k_hat) else None,
        "resample_draws": n_keep,
        "unique_draws": n_unique,
    }
    return draws, log_probs, report


def simulate_posterior(posterior: dict, n_sims: int, seed: int) -> SimSet:
    """Posterior-predictive SimSet: n_sims // S races from each retained
    draw, concatenated draw-major — one seeded rng, byte-deterministic."""
    theta = np.asarray(posterior["theta"], dtype=np.float64)
    dnf = np.asarray(posterior["dnf"], dtype=np.float64)
    S = theta.shape[0]
    R = max(1, n_sims // S)
    total = S * R
    if total != n_sims:
        print(f"Posterior-predictive sims: {S} draws x {R} races = {total:,} "
              f"(requested {n_sims:,})")

    rng = np.random.default_rng(seed)
    finish_pos, dnf_mask = sample_finish_positions(
        np.repeat(theta, R, axis=0), np.repeat(dnf, R, axis=0), total, rng)
    return SimSet(drivers=list(posterior["drivers"]),
                  finish_pos=finish_pos, dnf=dnf_mask)


def posterior_marginal_draws(
    posterior: dict,
    ks: list[int],
    n_draws: int,
    sims_per_draw: int,
    seed: int,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Per-draw hard-engine top-k marginals for credible intervals.

    Returns ``({k: (n_draws, n) array}, selected_draw_indices)`` of
    P(top k) computed separately for each of ``n_draws`` evenly-spaced
    posterior draws; the indices let callers align other per-draw
    quantities (e.g. sigma_obs) with the marginal draws.
    """
    theta = np.asarray(posterior["theta"], dtype=np.float64)
    dnf = np.asarray(posterior["dnf"], dtype=np.float64)
    S, n = theta.shape
    n_draws = min(n_draws, S)
    sel = np.unique(np.linspace(0, S - 1, n_draws).astype(int))
    n_draws = len(sel)

    rng = np.random.default_rng(seed)
    out = {k: np.empty((n_draws, n)) for k in ks}
    chunk = max(1, 50_000 // max(1, sims_per_draw))     # draws per batch
    for lo in range(0, n_draws, chunk):
        idx = sel[lo:lo + chunk]
        fp, _ = sample_finish_positions(
            np.repeat(theta[idx], sims_per_draw, axis=0),
            np.repeat(dnf[idx], sims_per_draw, axis=0),
            len(idx) * sims_per_draw, rng)
        fp = fp.reshape(len(idx), sims_per_draw, n)
        for k in ks:
            out[k][lo:lo + len(idx)] = (fp < k).mean(axis=1)
    return out, sel


def save_posterior(posterior: dict, race_name: str,
                   data_dir: Path = DATA_DIR) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"model_posterior_{race_slug(race_name)}.npz"
    np.savez_compressed(
        path,
        theta=posterior["theta"],
        dnf=posterior["dnf"],
        log_sigma=posterior["log_sigma"],
        log_prob=posterior["log_prob"],
        drivers=np.array(posterior["drivers"]),
        meta=json.dumps({"method": posterior["method"],
                         "bayes_report": posterior["bayes_report"]}),
    )
    print(f"Posterior saved -> {path}")
    return path


def load_posterior(race_name: str, data_dir: Path = DATA_DIR) -> dict:
    path = data_dir / f"model_posterior_{race_slug(race_name)}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"No posterior at {path}. Run `python main.py fit --bayes` first."
        )
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        return {
            "drivers": [str(c) for c in z["drivers"]],
            "theta": z["theta"],
            "dnf": z["dnf"],
            "log_sigma": z["log_sigma"],
            "log_prob": z["log_prob"],
            "method": meta["method"],
            "bayes_report": meta["bayes_report"],
        }
