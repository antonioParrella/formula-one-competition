"""Fit validation: market probabilities vs hard-simulated marginals.

Re-simulates the fitted model with the full Monte Carlo engine (no
soft-rank relaxation) and prints a table comparing every market
probability used in fitting against its simulated marginal. Any
deviation over 2 percentage points is flagged.

The Bayesian path (``validate_posterior``) upgrades the threshold to a
calibration test: each market probability is compared against the
posterior-predictive marginal and its credible interval, and flagged
only when the market falls *outside* the interval (MATH.md Section 7.6).
"""

import _paths  # noqa: F401

import numpy as np

from model.fit import _analytic_h2h
from model.simulate import SimSet, h2h_prob, top_k_probs

FLAG_THRESHOLD = 0.02  # 2 percentage points


def validate_fit(fit: dict, market_probs: dict, sims: SimSet) -> bool:
    """Print the market-vs-sim comparison table; True if nothing flagged."""
    col = {c: i for i, c in enumerate(sims.drivers)}
    header = f"{'market':<8} {'driver(s)':<12} {'market %':>9} {'sim %':>9} {'diff pp':>8}"
    print("\nValidation: de-vigged market probs vs simulated marginals "
          f"({sims.n_sims:,} races)")
    print(header)
    print("─" * len(header))

    flagged: list[str] = []
    for k, probs in sorted(market_probs["topk"].items()):
        sim_p = top_k_probs(sims.finish_pos, k)
        label = "win" if k == 1 else f"top{k}"
        for code in sorted(probs, key=probs.get, reverse=True):
            diff = sim_p[col[code]] - probs[code]
            flag = "  ⚠" if abs(diff) > FLAG_THRESHOLD else ""
            print(f"{label:<8} {code:<12} {probs[code]:>8.1%} "
                  f"{sim_p[col[code]]:>8.1%} {diff * 100:>+7.1f}{flag}")
            if flag:
                flagged.append(f"{label} {code} ({diff * 100:+.1f}pp)")

    for (a, b), p_a, _w in market_probs["h2h"]:
        sim_p = h2h_prob(sims.finish_pos, col[a], col[b])
        diff = sim_p - p_a
        flag = "  ⚠" if abs(diff) > FLAG_THRESHOLD else ""
        print(f"{'h2h':<8} {a + ' v ' + b:<12} {p_a:>8.1%} {sim_p:>8.1%} "
              f"{diff * 100:>+7.1f}{flag}")
        if flag:
            flagged.append(f"h2h {a} v {b} ({diff * 100:+.1f}pp)")

    if flagged:
        print(f"\n⚠ {len(flagged)} marginal(s) deviate by more than "
              f"{FLAG_THRESHOLD:.0%}: {', '.join(flagged)}")
        print("Consider more fit iterations, a lower fit_tau, or checking "
              "the snapshot for stale/illiquid prices.")
    else:
        print(f"\nAll marginals within {FLAG_THRESHOLD:.0%} of market probs.")
    return not flagged


def validate_posterior(
    posterior: dict,
    market_probs: dict,
    sims: SimSet,
    marginal_draws: dict[int, "np.ndarray"],
    draw_indices: "np.ndarray",
    ci: float = 0.90,
) -> bool:
    """Posterior-predictive calibration table.

    Two intervals per market probability:

    - **CI** — credible interval of the *model marginal* (parameter
      uncertainty only, from per-draw hard-engine marginals);
    - **PI** — the observation-predictive interval, CI convolved with
      the sigma_obs/sqrt(omega) noise band the likelihood itself claims.

    The flag fires when the market falls outside the **PI**: that is a
    price the model cannot explain even after admitting its own misfit
    level — a calibrated model should leave ~ci of markets unflagged.
    """
    col = {c: i for i, c in enumerate(sims.drivers)}
    lo_q, hi_q = (1.0 - ci) / 2.0, 1.0 - (1.0 - ci) / 2.0
    sigma_sub = np.exp(posterior["log_sigma"][draw_indices])
    noise = np.random.default_rng(0).standard_normal(len(draw_indices))
    header = (f"{'market':<8} {'driver(s)':<12} {'market %':>9} "
              f"{'postpred %':>11} {f'{ci:.0%} CI':>17} "
              f"{f'{ci:.0%} PI':>17} {'diff pp':>8}")
    print(f"\nValidation (Bayes, {posterior['method']}): market probs vs "
          f"posterior-predictive marginals ({sims.n_sims:,} races, "
          f"{posterior['theta'].shape[0]} draws)")
    print(header)
    print("─" * len(header))

    flagged: list[str] = []
    n_obs = 0
    for k, probs in sorted(market_probs["topk"].items()):
        sim_p = top_k_probs(sims.finish_pos, k)
        draws = marginal_draws[k]
        w = float(market_probs["weights"][k])
        label = "win" if k == 1 else f"top{k}"
        for code in sorted(probs, key=probs.get, reverse=True):
            c = col[code]
            p_mkt, p_sim = probs[code], sim_p[c]
            lo, hi = np.quantile(draws[:, c], [lo_q, hi_q])
            y_pred = draws[:, c] + sigma_sub * noise / np.sqrt(w)
            p_lo, p_hi = np.clip(np.quantile(y_pred, [lo_q, hi_q]), 0.0, 1.0)
            outside = not (p_lo <= p_mkt <= p_hi)
            flag = "  ⚠" if outside else ""
            n_obs += 1
            print(f"{label:<8} {code:<12} {p_mkt:>8.1%} {p_sim:>10.1%} "
                  f"[{lo:>6.1%}, {hi:>6.1%}] [{p_lo:>6.1%}, {p_hi:>6.1%}] "
                  f"{(p_sim - p_mkt) * 100:>+7.1f}{flag}")
            if outside:
                flagged.append(f"{label} {code}")

    theta, dnf = posterior["theta"], posterior["dnf"]
    sigma_all = np.exp(posterior["log_sigma"])
    noise_all = np.random.default_rng(1).standard_normal(len(sigma_all))
    for (a, b), p_a, w in market_probs["h2h"]:
        i, j = col[a], col[b]
        p_draws = _analytic_h2h(theta[:, i], theta[:, j], dnf[:, i], dnf[:, j])
        p_sim = h2h_prob(sims.finish_pos, i, j)
        lo, hi = np.quantile(p_draws, [lo_q, hi_q])
        y_pred = p_draws + sigma_all * noise_all / np.sqrt(float(w))
        p_lo, p_hi = np.clip(np.quantile(y_pred, [lo_q, hi_q]), 0.0, 1.0)
        outside = not (p_lo <= p_a <= p_hi)
        flag = "  ⚠" if outside else ""
        n_obs += 1
        print(f"{'h2h':<8} {a + ' v ' + b:<12} {p_a:>8.1%} {p_sim:>10.1%} "
              f"[{lo:>6.1%}, {hi:>6.1%}] [{p_lo:>6.1%}, {p_hi:>6.1%}] "
              f"{(p_sim - p_a) * 100:>+7.1f}{flag}")
        if outside:
            flagged.append(f"h2h {a} v {b}")

    sigma = np.exp(posterior["log_sigma"])
    s_lo, s_med, s_hi = np.quantile(sigma, [lo_q, 0.5, hi_q])
    scale = (posterior.get("bayes_report", {}).get("config", {})
             .get("obs_scale", "probability"))
    print(f"\nsigma_obs ({scale}-scale misfit): median {s_med:.3f}, "
          f"{ci:.0%} CI [{s_lo:.3f}, {s_hi:.3f}]")
    covered = 1.0 - len(flagged) / max(n_obs, 1)
    print(f"Calibration: {covered:.0%} of {n_obs} market probs inside their "
          f"{ci:.0%} predictive interval (target ~{ci:.0%})")
    if flagged:
        print(f"⚠ outside: {', '.join(flagged)}")
    return not flagged
