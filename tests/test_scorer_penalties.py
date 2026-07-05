import json
import sys

sys.path.insert(0, "src")
from scorer import Scorer


# Normal weekend schedule (Melbourne-style times, UTC).
NORMAL_SESSIONS = [
    {"session_name": "Practice 1", "date_start": "2026-03-06T01:30:00+00:00", "is_cancelled": False},
    {"session_name": "Practice 2", "date_start": "2026-03-06T05:00:00+00:00", "is_cancelled": False},
    {"session_name": "Practice 3", "date_start": "2026-03-07T01:30:00+00:00", "is_cancelled": False},
    {"session_name": "Qualifying", "date_start": "2026-03-07T05:00:00+00:00", "is_cancelled": False},
    {"session_name": "Race",       "date_start": "2026-03-08T04:00:00+00:00", "is_cancelled": False},
]

SPRINT_SESSIONS = [
    {"session_name": "Practice 1",        "date_start": "2026-07-03T11:30:00+00:00", "is_cancelled": False},
    {"session_name": "Sprint Qualifying", "date_start": "2026-07-03T15:30:00+00:00", "is_cancelled": False},
    {"session_name": "Sprint",            "date_start": "2026-07-04T11:00:00+00:00", "is_cancelled": False},
    {"session_name": "Qualifying",        "date_start": "2026-07-04T15:00:00+00:00", "is_cancelled": False},
    {"session_name": "Race",              "date_start": "2026-07-05T14:00:00+00:00", "is_cancelled": False},
]


def _write(dirpath, name, payload):
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / name).write_text(json.dumps(payload))


def _make_scorer(tmp_path, round_num, tips, results, schedule, overrides=None):
    tips_dir = tmp_path / "tips"
    results_dir = tmp_path / "results"
    schedule_dir = tmp_path / "schedule"
    overrides_dir = tmp_path / "overrides"
    out_dir = tmp_path / "processed"
    slug = "test"
    _write(tips_dir, f"r{round_num:02d}_{slug}_tips.json", tips)
    _write(results_dir, f"r{round_num:02d}_{slug}_result.json", results)
    _write(schedule_dir, f"r{round_num:02d}_{slug}_schedule.json", schedule)
    if overrides is not None:
        _write(overrides_dir, f"r{round_num:02d}_{slug}_overrides.json", overrides)
    return Scorer(
        round_num=round_num,
        tips_dir=str(tips_dir),
        results_dir=str(results_dir),
        schedule_dir=str(schedule_dir),
        overrides_dir=str(overrides_dir),
        output_dir=str(out_dir),
    )


def test_normal_weekend_penalty_floor_and_zero(tmp_path):
    # top10: VER is P1, HAM is P2.
    results = {
        "round": 1, "race_name": "Test", "is_sprint_weekend": False,
        "top10": ["VER", "HAM"], "dnfs": [], "sprint_top3": None,
        "championship_top10_before_race": None,
    }
    schedule = {"round": 1, "sessions": NORMAL_SESSIONS}
    tips = {
        "round": 1, "race_name": "Test", "is_sprint_weekend": False,
        "submissions": [
            # On time — picks VER P1 (exact, 5).
            {"player": "OnTime", "submitted_at": "2026-03-05T12:00:00", "main_race": ["VER"], "dnf_picks": []},
            # Late after FP1+FP2 (-10). Raw 5 -> floored to 0.
            {"player": "LateFloor", "submitted_at": "2026-03-06T19:00:00", "main_race": ["VER"], "dnf_picks": []},
            # After Qualifying start -> zeroed even though pick is exact.
            {"player": "Zeroed", "submitted_at": "2026-03-07T20:00:00", "main_race": ["VER"], "dnf_picks": []},
        ],
    }
    scorer = _make_scorer(tmp_path, 1, tips, results, schedule)
    scored = scorer._build_scored(tips, results, {}, schedule)
    by = {pt["player"]: pt for pt in scored["player_tips"]}

    assert by["OnTime"]["score"] == 5
    assert by["OnTime"]["penalty"] == 0
    assert by["OnTime"]["zeroed"] is False

    assert by["LateFloor"]["penalty"] == 10
    assert by["LateFloor"]["auto_penalty"] == 10
    assert by["LateFloor"]["score"] == 0  # floored, not negative

    assert by["Zeroed"]["zeroed"] is True
    assert by["Zeroed"]["score"] == 0


def test_override_stacks_on_auto_penalty(tmp_path):
    results = {
        "round": 1, "race_name": "Test", "is_sprint_weekend": False,
        "top10": ["VER", "HAM", "LEC", "NOR"], "dnfs": [], "sprint_top3": None,
        "championship_top10_before_race": None,
    }
    schedule = {"round": 1, "sessions": NORMAL_SESSIONS}
    tips = {
        "round": 1, "race_name": "Test", "is_sprint_weekend": False,
        "submissions": [
            # Exact on all 4 -> 20 raw. Late after FP1 (-5 auto) + override (-5) = -10.
            {"player": "Both", "submitted_at": "2026-03-06T13:00:00",
             "main_race": ["VER", "HAM", "LEC", "NOR"], "dnf_picks": []},
        ],
    }
    overrides = {"round": 1, "overrides": [{"player": "Both", "penalty_points": 5}]}
    scorer = _make_scorer(tmp_path, 1, tips, results, schedule, overrides)
    scored = scorer._build_scored(tips, results, overrides, schedule)
    pt = scored["player_tips"][0]
    assert pt["auto_penalty"] == 5
    assert pt["override_penalty"] == 5
    assert pt["penalty"] == 10
    assert pt["score"] == 10  # 20 - 10


def test_sprint_zeroed_for_late_player(tmp_path):
    results = {
        "round": 2, "race_name": "Test", "is_sprint_weekend": True,
        "top10": ["VER"], "dnfs": [], "sprint_top3": ["VER", "HAM", "LEC"],
        "championship_top10_before_race": [],
    }
    schedule = {"round": 2, "sessions": SPRINT_SESSIONS}
    tips = {
        "round": 2, "race_name": "Test", "is_sprint_weekend": True,
        "submissions": [
            # On time — sprint exact VER P1 -> 5.
            {"player": "OnTime", "submitted_at": "2026-07-03T12:00:00",
             "main_race": ["VER"], "sprint": ["VER", "HAM", "LEC"], "dnf_picks": []},
            # Late after FP1 — sprint zeroed.
            {"player": "Late", "submitted_at": "2026-07-03T23:00:00",
             "main_race": ["VER"], "sprint": ["VER", "HAM", "LEC"], "dnf_picks": []},
        ],
    }
    scorer = _make_scorer(tmp_path, 2, tips, results, schedule)
    scored = scorer._build_scored(tips, results, {}, schedule)
    sprint = {st["player"]: st for st in scored["sprint_tips"]}
    assert sprint["OnTime"]["score"] == 15  # 3 exact * 5
    assert sprint["OnTime"]["zeroed"] is False
    assert sprint["Late"]["score"] == 0
    assert sprint["Late"]["zeroed"] is True
