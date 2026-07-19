import pytest

from dnf_prior import recenter_rates, season_dnf_rates, smooth_rates


def test_smoothing_shrinks_toward_grid_average():
    dnf = {"AAA": 4, "BBB": 0}
    starts = {"AAA": 8, "BBB": 8}
    rates, grid = smooth_rates(dnf, starts, prior_strength=4.0)
    assert grid == pytest.approx(4 / 16)                       # 25% grid avg
    # Raw AAA 0.5 pulled down, raw BBB 0.0 pulled up — both toward the grid.
    assert grid < rates["AAA"] < 0.5
    assert 0.0 < rates["BBB"] < grid
    # Exact Beta form: (dnf + k*grid) / (starts + k).
    assert rates["AAA"] == pytest.approx((4 + 4 * 0.25) / (8 + 4))
    assert rates["BBB"] == pytest.approx((0 + 4 * 0.25) / (8 + 4))


def test_no_smoothing_returns_raw_rates():
    rates, grid = smooth_rates({"AAA": 2}, {"AAA": 10}, prior_strength=0.0)
    assert rates["AAA"] == pytest.approx(0.2)
    assert grid == pytest.approx(0.2)


def test_missing_driver_with_zero_dnf_still_smoothed():
    # A driver with starts but no DNF entry in dnf_counts -> treated as 0 DNFs.
    rates, grid = smooth_rates({"AAA": 3}, {"AAA": 6, "BBB": 6}, 4.0)
    assert grid == pytest.approx(3 / 12)
    assert rates["BBB"] == pytest.approx((0 + 4 * 0.25) / (6 + 4))


def test_no_starts_returns_empty():
    rates, grid = smooth_rates({}, {}, 4.0)
    assert rates == {} and grid is None


def test_recenter_hits_target_mean_and_keeps_ratios():
    rates = {"AAA": 0.40, "BBB": 0.20, "CCC": 0.10}  # mean 0.2333, ratios 4:2:1
    out = recenter_rates(rates, 0.10)
    assert sum(out.values()) / len(out) == pytest.approx(0.10)   # new mean
    # Multiplicative rescale preserves ratios exactly.
    assert out["AAA"] / out["CCC"] == pytest.approx(4.0)
    assert out["BBB"] / out["CCC"] == pytest.approx(2.0)
    # A high-DNF driver is pulled well below the raw level.
    assert out["AAA"] < 0.40


def test_recenter_empty_and_all_zero_are_safe():
    assert recenter_rates({}, 0.10) == {}
    z = recenter_rates({"AAA": 0.0, "BBB": 0.0}, 0.10)   # no signal to rescale
    assert z == {"AAA": 0.0, "BBB": 0.0}


def test_season_disabled_returns_empty():
    cfg = {"default_prob": 0.10, "season": {"enabled": False}}
    assert season_dnf_rates(cfg, 2026, 10) == {}


def test_round_one_has_no_season_data():
    cfg = {"default_prob": 0.10, "season": {"enabled": True}}
    # No network call: round 1 short-circuits before any OpenF1 fetch.
    assert season_dnf_rates(cfg, 2026, 1) == {}
