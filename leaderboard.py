from urllib.request import urlopen
import pandas as pd
import json


class ResultAggregator:
    BASE_URL = "https://api.openf1.org/v1"

    def __init__(self, round_number: int , year: int = 2026):
        self.round_number = round_number
        self.year = year
        self.race_calendar = self._build_race_calendar()
        self.meeting_key, self.race_session_key, self.sprint_session_key = self._find_meeting_keys()
        self.race_results = self._fetch("session_result", session_key=self.race_session_key)
        if self.sprint_session_key is None:
            self.championship_standings = self._fetch("championship_drivers", session_key=self.race_session_key)
            self.sprint_results = None
        else:
            self.championship_standings = self._fetch("championship_drivers", session_key=self.sprint_session_key)
            self.sprint_results = self._fetch("session_result", session_key=self.race_session_key)

    def _fetch(self, endpoint: str, **params) -> pd.DataFrame:
        query = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{self.BASE_URL}/{endpoint}?{query}"
        print(f"Fetching data from: {url}")
        response = urlopen(url)
        data = json.loads(response.read().decode("utf-8"))
        return pd.DataFrame(data)
    
    def _build_race_calendar(self) -> pd.DataFrame:
        sessions = self._fetch("sessions", year=self.year)
        competitive_sessions = sessions[
            (sessions.session_name == "Race") | (sessions.session_name == "Sprint")
        ]
        round_number_key = competitive_sessions[(competitive_sessions.session_name == "Race")].reset_index(drop=True).reset_index(names="round_number")[["meeting_key","round_number"]]

        round_number_key["round_number"] = round_number_key["round_number"] + 1
        race_calendar = competitive_sessions.merge(round_number_key, on="meeting_key", how="left")[[
            "round_number",
            "circuit_short_name",
            "session_name",
            "meeting_key",
            "session_key"
        ]]
        return race_calendar
    
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
        aggregated_sprint_results = self._get_position_scores(self.sprint_results,10)
        aggregated_race_results = self._get_position_scores(self.race_results,3)
        return aggregated_race_results, aggregated_sprint_results