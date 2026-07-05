import pytest
import sys
sys.path.insert(0, "src")
from penalties import assess_lateness, submitted_at_utc


# ── Fixture schedules (real OpenF1 2026 times, UTC) ──────────────
# Melbourne, round 1 — normal weekend (during AEDT, UTC+11).
NORMAL_SCHEDULE = {
    "round": 1,
    "is_sprint_weekend": False,
    "sessions": [
        {"session_name": "Practice 1", "date_start": "2026-03-06T01:30:00+00:00", "is_cancelled": False},
        {"session_name": "Practice 2", "date_start": "2026-03-06T05:00:00+00:00", "is_cancelled": False},
        {"session_name": "Practice 3", "date_start": "2026-03-07T01:30:00+00:00", "is_cancelled": False},
        {"session_name": "Qualifying", "date_start": "2026-03-07T05:00:00+00:00", "is_cancelled": False},
        {"session_name": "Race",       "date_start": "2026-03-08T04:00:00+00:00", "is_cancelled": False},
    ],
}

# Silverstone, round 9 — sprint weekend (during AEST, UTC+10).
SPRINT_SCHEDULE = {
    "round": 9,
    "is_sprint_weekend": True,
    "sessions": [
        {"session_name": "Practice 1",        "date_start": "2026-07-03T11:30:00+00:00", "is_cancelled": False},
        {"session_name": "Sprint Qualifying", "date_start": "2026-07-03T15:30:00+00:00", "is_cancelled": False},
        {"session_name": "Sprint",            "date_start": "2026-07-04T11:00:00+00:00", "is_cancelled": False},
        {"session_name": "Qualifying",        "date_start": "2026-07-04T15:00:00+00:00", "is_cancelled": False},
        {"session_name": "Race",              "date_start": "2026-07-05T14:00:00+00:00", "is_cancelled": False},
    ],
}


class TestSubmittedAtUtc:
    def test_naive_melbourne_aedt(self):
        # Melbourne is UTC+11 in early March (AEDT).
        dt = submitted_at_utc("2026-03-06T12:30:00")
        assert dt.isoformat() == "2026-03-06T01:30:00+00:00"

    def test_naive_melbourne_aest(self):
        # Melbourne is UTC+10 in July (AEST) — daylight saving ended.
        dt = submitted_at_utc("2026-07-03T21:30:00")
        assert dt.isoformat() == "2026-07-03T11:30:00+00:00"

    def test_missing_raises(self):
        with pytest.raises(ValueError):
            submitted_at_utc("")


class TestNormalWeekend:
    def test_before_fp1_clean(self):
        # Well before FP1 (Melbourne local, day before).
        res = assess_lateness("2026-03-05T20:00:00", NORMAL_SCHEDULE)
        assert res == {"late": False, "late_sessions": [], "auto_penalty": 0, "zeroed": False}

    def test_at_fp1_start(self):
        # Exactly at FP1 start (01:30 UTC == 12:30 Melbourne AEDT).
        res = assess_lateness("2026-03-06T12:30:00", NORMAL_SCHEDULE)
        assert res["late_sessions"] == ["Practice 1"]
        assert res["auto_penalty"] == 5
        assert res["zeroed"] is False

    def test_after_fp2_real_veljko(self):
        # Real r01 Veljko submission — after FP1 and FP2, before FP3.
        res = assess_lateness("2026-03-06T19:00:17", NORMAL_SCHEDULE)
        assert res["late_sessions"] == ["Practice 1", "Practice 2"]
        assert res["auto_penalty"] == 10
        assert res["zeroed"] is False

    def test_after_fp3_before_quali(self):
        # After FP3 (03-07 01:30 UTC == 12:30 Melbourne), before Quali.
        res = assess_lateness("2026-03-07T13:00:00", NORMAL_SCHEDULE)
        assert res["auto_penalty"] == 15
        assert res["zeroed"] is False

    def test_at_quali_zeroed(self):
        # At/after Quali start (05:00 UTC == 16:00 Melbourne).
        res = assess_lateness("2026-03-07T16:00:00", NORMAL_SCHEDULE)
        assert res["zeroed"] is True
        assert res["auto_penalty"] == 15


class TestSprintWeekend:
    def test_after_fp1_before_sq_real_alex(self):
        # Real r09 Alex submission — after FP1, before Sprint Qualifying.
        res = assess_lateness("2026-07-03T22:18:44", SPRINT_SCHEDULE)
        assert res["late_sessions"] == ["Practice 1"]
        assert res["auto_penalty"] == 5
        assert res["late"] is True
        assert res["zeroed"] is False

    def test_after_sq_before_sprint_real_riki(self):
        # Real r09 Riki submission — after FP1 and Sprint Qualifying (AEST +10).
        res = assess_lateness("2026-07-04T09:44:53", SPRINT_SCHEDULE)
        assert res["late_sessions"] == ["Practice 1", "Sprint Qualifying"]
        assert res["auto_penalty"] == 10
        assert res["zeroed"] is False

    def test_at_sprint_zeroed(self):
        # At/after Sprint start (11:00 UTC == 21:00 Melbourne AEST).
        res = assess_lateness("2026-07-04T21:00:00", SPRINT_SCHEDULE)
        assert res["zeroed"] is True
        assert res["auto_penalty"] == 10

    def test_no_practice_2_3_no_error(self):
        # Sprint weekend has no FP2/FP3 — must not raise, only FP1+SQ count.
        res = assess_lateness("2026-07-03T22:18:44", SPRINT_SCHEDULE)
        assert set(res["late_sessions"]).issubset({"Practice 1", "Sprint Qualifying"})


class TestEdgeCases:
    def test_cancelled_session_ignored(self):
        schedule = {
            "sessions": [
                {"session_name": "Practice 1", "date_start": "2026-03-06T01:30:00+00:00", "is_cancelled": True},
                {"session_name": "Practice 2", "date_start": "2026-03-06T05:00:00+00:00", "is_cancelled": False},
                {"session_name": "Practice 3", "date_start": "2026-03-07T01:30:00+00:00", "is_cancelled": False},
                {"session_name": "Qualifying", "date_start": "2026-03-07T05:00:00+00:00", "is_cancelled": False},
            ],
        }
        # After FP1's original slot but FP1 is cancelled — only FP2 counts.
        res = assess_lateness("2026-03-06T16:30:00", schedule)
        assert "Practice 1" not in res["late_sessions"]
        assert res["late_sessions"] == ["Practice 2"]

    def test_missing_submitted_at_raises(self):
        with pytest.raises(ValueError):
            assess_lateness(None, NORMAL_SCHEDULE)
