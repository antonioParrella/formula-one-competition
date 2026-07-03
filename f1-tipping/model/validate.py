"""Fit validation: market probabilities vs hard-simulated marginals.

Re-simulates the fitted model with the full Monte Carlo engine (no
soft-rank relaxation) and prints a table comparing every market
probability used in fitting against its simulated marginal. Any
deviation over 2 percentage points is flagged.
"""

import _paths  # noqa: F401

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
