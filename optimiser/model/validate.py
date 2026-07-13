"""Fit validation: market probabilities vs hard-simulated marginals.

Re-simulates the fitted model with the full Monte Carlo engine (no
soft-rank relaxation) and prints a table comparing every market
probability used in fitting against its simulated marginal. A deviation
is flagged when it exceeds a **liquidity-adjusted** threshold: a 2pp gap
only means something when the market is liquid enough to price to that
precision, so the threshold widens for thin markets (see
``_flag_threshold``).

The Bayesian path (``validate_posterior``) upgrades the threshold to a
calibration test: each market probability is compared against the
posterior-predictive marginal and its credible interval, and flagged
only when the market falls *outside* the interval (MATH.md Section 7.6).
That interval already widens with illiquidity (the noise band scales as
1/sqrt(weight)), so it is liquidity-aware by construction.
"""

import _paths  # noqa: F401

import numpy as np

from model.fit import _analytic_h2h
from model.simulate import SimSet, h2h_prob, top_k_probs

FLAG_THRESHOLD = 0.02  # 2 percentage points, at/above reference liquidity
# _market_weight (odds/devig.py) is the liquidity proxy: unknown -> 1.0,
# ~£10k matched -> 1.0, rising to 1.5 at ~£1M, falling to 0.2 for a near
# empty book. We anchor the flat 2pp flag to this reference weight.
LIQUID_WEIGHT = 1.0


def _flag_threshold(weight: float) -> float:
    """Liquidity-adjusted deviation threshold for a market.

    A 2pp gap is only worth a warning when the market is liquid enough to
    trust its de-vigged price to that precision. At or above the
    reference liquidity — and when liquidity is unknown (weight 1.0, e.g.
    the manual CSV path) — the flat ``FLAG_THRESHOLD`` holds. Below it the
    price is noisier, so the threshold widens in proportion to how thin
    the book is (weight 0.5 -> 4pp, weight 0.2 -> 10pp). It never tightens
    below 2pp for a very liquid market — that would flag gaps the flat
    rule never did.
    """
    return FLAG_THRESHOLD * max(1.0, LIQUID_WEIGHT / max(weight, 1e-9))


def validate_fit(fit: dict, market_probs: dict, sims: SimSet) -> bool:
    """Print the market-vs-sim comparison table; True if nothing flagged."""
    col = {c: i for i, c in enumerate(sims.drivers)}
    header = (f"{'market':<8} {'driver(s)':<12} {'market %':>9} {'sim %':>9} "
              f"{'diff pp':>8} {'thr pp':>7}")
    print("\nValidation: de-vigged market probs vs simulated marginals "
          f"({sims.n_sims:,} races)")
    print("flag threshold scales with market liquidity — "
          f"{FLAG_THRESHOLD:.0%} when liquid, wider for thin books")
    print(header)
    print("─" * len(header))

    weights = market_probs.get("weights", {})
    flagged: list[str] = []
    for k, probs in sorted(market_probs["topk"].items()):
        sim_p = top_k_probs(sims.finish_pos, k)
        label = "win" if k == 1 else f"top{k}"
        thr = _flag_threshold(weights.get(k, LIQUID_WEIGHT))
        for code in sorted(probs, key=probs.get, reverse=True):
            diff = sim_p[col[code]] - probs[code]
            flag = "  ⚠" if abs(diff) > thr else ""
            print(f"{label:<8} {code:<12} {probs[code]:>8.1%} "
                  f"{sim_p[col[code]]:>8.1%} {diff * 100:>+7.1f} "
                  f"{thr * 100:>6.1f}{flag}")
            if flag:
                flagged.append(
                    f"{label} {code} ({diff * 100:+.1f}pp > {thr * 100:.1f}pp)")

    for (a, b), p_a, w in market_probs["h2h"]:
        sim_p = h2h_prob(sims.finish_pos, col[a], col[b])
        diff = sim_p - p_a
        thr = _flag_threshold(w)
        flag = "  ⚠" if abs(diff) > thr else ""
        print(f"{'h2h':<8} {a + ' v ' + b:<12} {p_a:>8.1%} {sim_p:>8.1%} "
              f"{diff * 100:>+7.1f} {thr * 100:>6.1f}{flag}")
        if flag:
            flagged.append(
                f"h2h {a} v {b} ({diff * 100:+.1f}pp > {thr * 100:.1f}pp)")

    if flagged:
        print(f"\n⚠ {len(flagged)} marginal(s) deviate beyond their "
              f"liquidity-adjusted threshold: {', '.join(flagged)}")
        print("These are liquid enough for the gap to be real — consider more "
              "fit iterations, a lower fit_tau, or checking the snapshot for "
              "stale prices.")
    else:
        print("\nAll marginals within their liquidity-adjusted threshold "
              f"({FLAG_THRESHOLD:.0%} at full liquidity, wider for thin books).")
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
