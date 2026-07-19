"""Bayesian path: toy-posterior recovery, DNF handling, plumbing."""

import numpy as np
import pytest
from scipy.special import expit

from model.bayes import (
    build_log_posterior,
    fit_posterior,
    load_posterior,
    posterior_marginal_draws,
    save_posterior,
    simulate_posterior,
)

# Same 3-driver toy as test_simulate: win probs exactly (0.5, 0.3, 0.2).
WEIGHTS = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}
THETA = {c: float(np.log(w)) for c, w in WEIGHTS.items()}
THETA_CENTRED = {c: t - np.mean(list(THETA.values())) for c, t in THETA.items()}
MAP_FIT = {"theta": THETA}

MCMC_CFG = {"method": "mcmc", "fit_sims": 500, "tau": 0.15, "batch": 8,
            "walkers": 16, "steps": 400, "burn_in": 150, "thin": 10, "seed": 5}
IS_CFG = {"method": "is", "fit_sims": 500, "tau": 0.15, "batch": 8,
          "is_draws": 3000, "resample_draws": 400, "seed": 5}


def toy_market_probs() -> dict:
    """Exact no-DNF targets: win = softmax, h2h = pairwise logistic."""
    h2h = []
    codes = list(WEIGHTS)
    for i, a in enumerate(codes):
        for b in codes[i + 1:]:
            p = WEIGHTS[a] / (WEIGHTS[a] + WEIGHTS[b])
            h2h.append(((a, b), p, 1.0))
    return {"drivers": codes, "topk": {1: dict(WEIGHTS)}, "weights": {1: 1.0},
            "h2h": h2h, "dnf": {}, "dnf_weights": {}, "markets_used": []}


NO_DNF_CFG = {"default_prob": 0.0, "per_driver": {}}
NO_DNF = {c: 0.0 for c in WEIGHTS}


@pytest.fixture(scope="module")
def mcmc_posterior():
    return fit_posterior(toy_market_probs(), NO_DNF, NO_DNF_CFG, MAP_FIT,
                         MCMC_CFG)


@pytest.fixture(scope="module")
def is_posterior():
    return fit_posterior(toy_market_probs(), NO_DNF, NO_DNF_CFG, MAP_FIT,
                         IS_CFG)


def test_mcmc_posterior_concentrates_on_truth(mcmc_posterior):
    mean = mcmc_posterior["theta"].mean(axis=0)
    for i, code in enumerate(mcmc_posterior["drivers"]):
        assert mean[i] == pytest.approx(THETA_CENTRED[code], abs=0.15)


def test_is_posterior_concentrates_on_truth(is_posterior):
    assert is_posterior["bayes_report"]["pareto_khat"] < 0.7
    mean = is_posterior["theta"].mean(axis=0)
    for i, code in enumerate(is_posterior["drivers"]):
        assert mean[i] == pytest.approx(THETA_CENTRED[code], abs=0.15)


def test_engines_agree(mcmc_posterior, is_posterior):
    assert np.allclose(mcmc_posterior["theta"].mean(axis=0),
                       is_posterior["theta"].mean(axis=0), atol=0.15)


def test_no_classified_market_keeps_dnf_fixed(mcmc_posterior):
    # Fallback path: m = 0, so the state is (theta, gamma) and every dnf
    # column is the fixed input, constant across draws.
    _, meta = build_log_posterior(toy_market_probs(), NO_DNF, NO_DNF_CFG,
                                  MCMC_CFG)
    assert meta["m"] == 0 and meta["D"] == len(WEIGHTS) + 1
    assert (mcmc_posterior["dnf"] == 0.0).all()


def test_classified_market_informs_dnf():
    # AAA priced at P(classified) = 0.95 pulls its posterior d to ~0.05;
    # unpriced drivers stay pinned at the 0.10 season prior.
    market = toy_market_probs()
    market["dnf"] = {"AAA": 0.05}
    market["dnf_weights"] = {"AAA": 1.0}
    dnf_cfg = {"default_prob": 0.10, "per_driver": {}}
    dnf_probs = {"AAA": 0.05, "BBB": 0.10, "CCC": 0.10}

    _, meta = build_log_posterior(market, dnf_probs, dnf_cfg, MCMC_CFG)
    assert meta["m"] == 1 and meta["eta_codes"] == ["AAA"]

    posterior = fit_posterior(market, dnf_probs, dnf_cfg, MAP_FIT, MCMC_CFG)
    col = {c: i for i, c in enumerate(posterior["drivers"])}
    d_mean = posterior["dnf"].mean(axis=0)
    assert d_mean[col["AAA"]] == pytest.approx(0.05, abs=0.02)
    assert posterior["dnf"][:, col["AAA"]].std() > 0        # genuinely sampled
    assert (posterior["dnf"][:, col["BBB"]] == 0.10).all()  # fixed
    assert (posterior["dnf"][:, col["CCC"]] == 0.10).all()


def test_simulate_posterior_blocks_and_determinism():
    # Two extreme draws: draw 0 makes AAA near-certain winner, draw 1 BBB.
    posterior = {
        "drivers": ["AAA", "BBB", "CCC"],
        "theta": np.array([[6.0, -3.0, -3.0], [-3.0, 6.0, -3.0]]),
        "dnf": np.zeros((2, 3)),
        "log_sigma": np.zeros(2),
        "log_prob": np.zeros(2),
        "method": "mcmc",
        "bayes_report": {},
    }
    sims = simulate_posterior(posterior, n_sims=2000, seed=11)
    assert sims.finish_pos.shape == (2000, 3)
    first, second = sims.finish_pos[:1000], sims.finish_pos[1000:]
    assert (first[:, 0] == 0).mean() > 0.95    # AAA wins block 0
    assert (second[:, 1] == 0).mean() > 0.95   # BBB wins block 1

    again = simulate_posterior(posterior, n_sims=2000, seed=11)
    assert (sims.finish_pos == again.finish_pos).all()


def test_posterior_marginal_draws_shapes(mcmc_posterior):
    draws, sel = posterior_marginal_draws(mcmc_posterior, ks=[1], n_draws=20,
                                          sims_per_draw=500, seed=13)
    assert sel.shape == (20,)
    assert draws[1].shape == (20, 3)
    assert ((draws[1] >= 0) & (draws[1] <= 1)).all()
    # Every draw's win probs should be in the right neighbourhood.
    assert np.abs(draws[1].mean(axis=0) - [0.5, 0.3, 0.2]).max() < 0.1


def test_save_load_roundtrip(tmp_path, mcmc_posterior):
    save_posterior(mcmc_posterior, "Toy Grand Prix", data_dir=tmp_path)
    loaded = load_posterior("Toy Grand Prix", data_dir=tmp_path)
    assert loaded["drivers"] == mcmc_posterior["drivers"]
    assert np.allclose(loaded["theta"], mcmc_posterior["theta"])
    assert np.allclose(loaded["dnf"], mcmc_posterior["dnf"])
    assert loaded["method"] == "mcmc"
    assert (loaded["bayes_report"]["acceptance"]
            == mcmc_posterior["bayes_report"]["acceptance"])


def test_missing_posterior_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="fit --bayes"):
        load_posterior("Toy Grand Prix", data_dir=tmp_path)
