import pytest

from odds.devig import devig_snapshot
from odds.manual_input import parse_csv


def test_parse_csv_classified_sides(tmp_path):
    csv = tmp_path / "odds.csv"
    csv.write_text(
        "driver,market,odds\n"
        "VER,win,2.0\n"
        "NOR,win,2.5\n"
        "VER,classified,1.10\n"   # Yes side
        "VER,dnf,8.0\n"           # No side
        "NOR,classified,1.25\n"   # Yes only
    )
    snapshot = parse_csv(csv, "Test", {})
    classified = snapshot["markets"]["classified"]
    assert classified["yes"]["runners"]["VER"]["last_traded"] == 1.10
    assert classified["no"]["runners"]["VER"]["last_traded"] == 8.0
    assert "NOR" not in classified["no"]["runners"] if "no" in classified else True

    out = devig_snapshot(snapshot)
    q_yes, q_no = 1 / 1.10, 1 / 8.0
    assert out["dnf"]["VER"] == pytest.approx(1 - q_yes / (q_yes + q_no))
    assert out["dnf"]["NOR"] == pytest.approx(0.2)


def test_parse_csv_unknown_market_still_rejected(tmp_path):
    csv = tmp_path / "odds.csv"
    csv.write_text("driver,market,odds\nVER,win,2.0\nVER,fastestlap,3.0\n")
    with pytest.raises(ValueError, match="unknown market"):
        parse_csv(csv, "Test", {})
