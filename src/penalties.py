"""
penalties.py
─────────────────────────────────────────────────────────────
Pure logic for automatic late-submission penalties.

A submission is compared against the real session start times for
its race weekend (fetched from OpenF1 and stored in the schedule
file). The counting basis is the *session start*:

  - The deadline is the start of Practice 1 (FP1).
  - Each practice-type session whose start has passed at submission
    time costs PENALTY_LATE_PER_SESSION points.
  - Once the cutoff session has started the submission scores zero.

Normal weekend
    penalty sessions = Practice 1, Practice 2, Practice 3
    cutoff           = Qualifying

Sprint weekend (2026 format: FP1, Sprint Qualifying, Sprint, ...)
    penalty sessions = Practice 1, Sprint Qualifying
    cutoff           = Sprint   (the sprint takes the place of qualifying)
    Any late submission on a sprint weekend also earns zero sprint points.

Timestamps in the tips files (`submitted_at`) are timezone-naive
Australia/Melbourne local time; session `date_start` values are
UTC ISO strings. Everything is compared in UTC.

No I/O, no network — this module is pure so it can be unit tested
offline.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

# Timezone of the `submitted_at` timestamps in the raw tips files.
SUBMISSION_TZ_NAME = "Australia/Melbourne"

# Sessions that incur a per-session penalty, and the cutoff session
# after which a submission scores zero.
NORMAL_PENALTY_SESSIONS = ("Practice 1", "Practice 2", "Practice 3")
NORMAL_CUTOFF_SESSION = "Qualifying"

SPRINT_PENALTY_SESSIONS = ("Practice 1", "Sprint Qualifying")
SPRINT_CUTOFF_SESSION = "Sprint"


def submitted_at_utc(submitted_at: str, tz_name: str = SUBMISSION_TZ_NAME) -> datetime:
    """Convert a naive local-time submission string to an aware UTC datetime.

    Parameters
    ----------
    submitted_at : str
        Naive ISO timestamp as stored by SurveyMars, e.g.
        "2026-03-06T19:00:17" — interpreted in ``tz_name`` local time.
    tz_name : str
        IANA timezone name the timestamp is expressed in.
    """
    if not submitted_at:
        raise ValueError("submitted_at is missing or empty")
    local = datetime.fromisoformat(submitted_at)
    if local.tzinfo is not None:
        # Already aware — just normalise to UTC.
        return local.astimezone(ZoneInfo("UTC"))
    aware = local.replace(tzinfo=ZoneInfo(tz_name))
    return aware.astimezone(ZoneInfo("UTC"))


def _session_start_map(schedule: dict) -> dict[str, datetime]:
    """Map ``session_name -> aware start datetime`` for non-cancelled sessions."""
    starts: dict[str, datetime] = {}
    for session in schedule.get("sessions", []):
        if session.get("is_cancelled"):
            continue
        date_start = session.get("date_start")
        if not date_start:
            continue
        # OpenF1 date_start is UTC ISO with an offset (e.g. "+00:00").
        starts[session["session_name"]] = datetime.fromisoformat(date_start)
    return starts


def assess_lateness(
    submitted_at: str,
    schedule: dict,
    penalty_per_session: int = 5,
    tz_name: str = SUBMISSION_TZ_NAME,
) -> dict:
    """Assess how late a submission is against the weekend's session schedule.

    Parameters
    ----------
    submitted_at : str
        Naive local-time submission timestamp.
    schedule : dict
        Schedule file contents with a ``sessions`` list, each having
        ``session_name``, ``date_start`` (UTC ISO), and ``is_cancelled``.
    penalty_per_session : int
        Points deducted per penalty session that has started.
    tz_name : str
        Timezone the submission timestamp is expressed in.

    Returns
    -------
    dict with keys:
        late          : bool  — any penalty session has started
        late_sessions : list  — names of started penalty sessions
        auto_penalty  : int   — points to deduct (before flooring)
        zeroed        : bool   — cutoff session has started (score zero)
    """
    sub_utc = submitted_at_utc(submitted_at, tz_name)
    starts = _session_start_map(schedule)

    is_sprint = SPRINT_CUTOFF_SESSION in starts
    penalty_sessions = SPRINT_PENALTY_SESSIONS if is_sprint else NORMAL_PENALTY_SESSIONS
    cutoff_session = SPRINT_CUTOFF_SESSION if is_sprint else NORMAL_CUTOFF_SESSION

    late_sessions = [
        name
        for name in penalty_sessions
        if name in starts and sub_utc >= starts[name]
    ]

    cutoff_start = starts.get(cutoff_session)
    zeroed = cutoff_start is not None and sub_utc >= cutoff_start

    return {
        "late": bool(late_sessions),
        "late_sessions": late_sessions,
        "auto_penalty": penalty_per_session * len(late_sessions),
        "zeroed": zeroed,
    }
