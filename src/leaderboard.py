from urllib.request import urlopen
from urllib.error import URLError
import pandas as pd
import json
import time
from pathlib import Path

BASE_URL = "https://api.openf1.org/v1"


def _fetch_openf1(endpoint: str, **params) -> pd.DataFrame:
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{BASE_URL}/{endpoint}?{query}"
    print(f"Fetching data from: {url}")

    max_retries = 5
    retry_delay = 2
    for attempt in range(max_retries):
        try:
            response = urlopen(url, timeout=30)
            data = json.loads(response.read().decode("utf-8"))
            return pd.DataFrame(data)
        except (URLError, OSError) as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise


def _load_cancelled_rounds(year: int) -> set[int]:
    config_path = Path("config/cancelled_rounds.json")
    if not config_path.exists():
        return set()

    with open(config_path) as f:
        config = json.load(f)

    if config.get("year") != year:
        return set()

    return set(config.get("cancelled_rounds", []))


def get_race_calendar(year: int) -> pd.DataFrame:
    sessions = _fetch_openf1("sessions", year=year)
    competitive_sessions = sessions[
        (sessions.session_name == "Race") | (sessions.session_name == "Sprint")
    ]

    races = competitive_sessions[competitive_sessions.session_name == "Race"].copy()
    races = races[~races["is_cancelled"]]
    races = races.reset_index(drop=True)
    races["round_number"] = range(1, len(races) + 1)

    round_map = dict(zip(races["meeting_key"], races["round_number"]))

    meeting_keys = races["meeting_key"].tolist()
    sprints = competitive_sessions[
        (competitive_sessions.session_name == "Sprint") &
        (competitive_sessions.meeting_key.isin(meeting_keys))
    ].copy()
    sprints["round_number"] = sprints["meeting_key"].map(round_map)

    race_calendar = pd.concat([races, sprints], ignore_index=True)

    race_calendar = race_calendar[[
        "round_number",
        "circuit_short_name",
        "country_name",
        "session_name",
        "meeting_key",
        "session_key"
    ]]
    return race_calendar


class ResultAggregator:
    BASE_URL = BASE_URL

    def __init__(self, round_number: int, year: int = 2026):
        self.round_number = round_number
        self.year = year
        self.cancelled_rounds = _load_cancelled_rounds(year)
        self.race_calendar = get_race_calendar(year)
        self.meeting_key, self.race_session_key, self.sprint_session_key = self._find_meeting_keys()
        self.race_results = self._fetch("session_result", session_key=self.race_session_key)
        if self.sprint_session_key is None:
            self.championship_standings = self._fetch("championship_drivers", session_key=self.race_session_key)
            self.drivers = self._fetch("drivers", session_key=self.race_session_key)
            self.sprint_results = None
        else:
            self.championship_standings = self._fetch("championship_drivers", session_key=self.sprint_session_key)
            self.drivers = self._fetch("drivers", session_key=self.sprint_session_key)
            self.sprint_results = self._fetch("session_result", session_key=self.sprint_session_key)

    def _fetch(self, endpoint: str, **params) -> pd.DataFrame:
        return _fetch_openf1(endpoint, **params)
    
    def _find_meeting_keys(self):
        current_sessions = self.race_calendar[self.race_calendar["round_number"] == self.round_number]
        if current_sessions.empty:
            raise ValueError(f"No sessions found for round number {self.round_number}")
        meeting_key = current_sessions.meeting_key.values[0]
        race_session_key = current_sessions[current_sessions["session_name"] == "Race"]["session_key"].values[0]
        if len(current_sessions) == 2:
            sprint_session_key = current_sessions[current_sessions["session_name"] == "Sprint"]["session_key"].values[0]
        else:
            sprint_session_key = None
        return meeting_key, race_session_key, sprint_session_key


    def refresh(self):
        self.__init__(self.session_key, self.year)

    def _get_position_scores(self, session_results, top_k) -> pd.DataFrame:
        standings = self.championship_standings[["driver_number", "position_start"]]
        results = session_results[["driver_number", "position"]]

        race_output = (
            results
            .merge(self.drivers[["driver_number", "name_acronym"]], on="driver_number", how="left")
            .merge(standings, on="driver_number", how="left")
        )
        return race_output[race_output["position"] <= top_k].sort_values("position")
    
    def aggregate_results(self):
        if self.sprint_results is None:
            print("No Sprint Race")
            aggregated_sprint_results = None
        else:
            print("Sprint Race")
            aggregated_sprint_results = self._get_position_scores(self.sprint_results,3)
        aggregated_race_results = self._get_position_scores(self.race_results,10)
        return aggregated_race_results, aggregated_sprint_results