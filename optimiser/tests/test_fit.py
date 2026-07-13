import pytest

from model.fit import build_dnf_probs


def test_dnf_precedence_config_over_market_over_prior():
    drivers = ["AAA", "BBB", "CCC"]
    cfg = {"default_prob": 0.10, "per_driver": {"AAA": 0.25}}
    market = {"AAA": 0.05, "BBB": 0.07}  # AAA also priced — config wins
    probs = build_dnf_probs(drivers, cfg, market)
    assert probs["AAA"] == pytest.approx(0.25)  # explicit override
    assert probs["BBB"] == pytest.approx(0.07)  # market-implied
    assert probs["CCC"] == pytest.approx(0.10)  # flat prior


def test_dnf_season_rate_sits_between_market_and_default():
    drivers = ["AAA", "BBB", "CCC", "DDD"]
    cfg = {"default_prob": 0.10, "per_driver": {"AAA": 0.25}}
    market = {"AAA": 0.05, "BBB": 0.07}       # AAA config-pinned, BBB priced
    season = {"AAA": 0.40, "BBB": 0.40, "CCC": 0.18}
    probs = build_dnf_probs(drivers, cfg, market, season)
    assert probs["AAA"] == pytest.approx(0.25)  # config beats everything
    assert probs["BBB"] == pytest.approx(0.07)  # market beats season
    assert probs["CCC"] == pytest.approx(0.18)  # season beats flat default
    assert probs["DDD"] == pytest.approx(0.10)  # nothing -> flat default


def test_dnf_without_market_matches_old_behaviour():
    drivers = ["AAA", "BBB"]
    cfg = {"default_prob": 0.12, "per_driver": {"BBB": 0.2}}
    assert build_dnf_probs(drivers, cfg) == {"AAA": 0.12, "BBB": 0.2}
