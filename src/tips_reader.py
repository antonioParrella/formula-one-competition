import pandas as pd


class TipsReader:
    """Reads and organises the Melbourne GP predictions spreadsheet."""

    COLUMN_MAP = {
        "Index": "index",
        "Submitted Time": "submitted_time",
        "Time Spent": "time_spent",
        "1. What is your name": "name",
        "2. Top ten for the race": "top_ten",
        "3. Pick driver(s) to DNF (max 5 per season, can be used on any race)": "dnf_picks",
        "4. Who will win the drivers championship?": "drivers_champ",
        "5. Who will win the constructors championship?": "constructors_champ",
    }

    DROP_COLUMNS = ["Collector", "Collector Details", "IP Address"]

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.raw: pd.DataFrame = pd.DataFrame()
        self.df: pd.DataFrame = pd.DataFrame()
        self._load()

    # ------------------------------------------------------------------
    # Loading & cleaning
    # ------------------------------------------------------------------

    def _load(self) -> None:
        self.raw = pd.read_excel(self.filepath)
        self.df = (
            self.raw
            .drop(columns=self.DROP_COLUMNS, errors="ignore")
            .rename(columns=self.COLUMN_MAP)
            .pipe(self._clean_types)
            .pipe(self._parse_top_ten)
            .pipe(self._parse_dnf_picks)
        )

    def _clean_types(self, df: pd.DataFrame) -> pd.DataFrame:
        df["submitted_time"] = pd.to_datetime(
            df["submitted_time"].str.extract(r"^(.+?)\s+\(")[0],
            format="%b %d %Y %I:%M:%S %p",
        )
        df["time_spent_s"] = (
            df["time_spent"].str.replace("s", "", regex=False).astype(int)
        )
        df = df.drop(columns=["time_spent"])

        # Replace survey placeholder for empty answers
        for col in ["dnf_picks", "drivers_champ", "constructors_champ"]:
            df[col] = df[col].replace("(Empty)", pd.NA)

        return df

    def _parse_top_ten(self, df: pd.DataFrame) -> pd.DataFrame:
        """Expand the → separated top-ten string into an ordered list column."""
        df["top_ten"] = df["top_ten"].str.split("→")
        return df

    def _parse_dnf_picks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Expand the ┋ separated DNF picks into a list column."""
        df["dnf_picks"] = df["dnf_picks"].str.split("┋")
        return df

    # ------------------------------------------------------------------
    # Convenience views
    # ------------------------------------------------------------------

    def top_ten_grid(self) -> pd.DataFrame:
        """Wide table: one column per finishing position (P1–P10)."""
        positions = pd.DataFrame(
            self.df["top_ten"].tolist(),
            index=self.df["name"],
            columns=[f"P{i}" for i in range(1, 11)],
        )
        return positions

    def driver_championship_tally(self) -> pd.Series:
        """Count of picks for each drivers' champion."""
        return self.df["drivers_champ"].value_counts()

    def constructors_championship_tally(self) -> pd.Series:
        """Count of picks for each constructors' champion."""
        return self.df["constructors_champ"].value_counts()

    def dnf_tally(self) -> pd.Series:
        """Count of how many times each driver was tipped to DNF."""
        all_dnf = self.df["dnf_picks"].dropna().explode()
        return all_dnf.value_counts()

    def position_frequency(self) -> pd.DataFrame:
        """For every driver, how often they were tipped at each position."""
        grid = self.top_ten_grid()
        drivers = pd.unique(grid.values.ravel())
        freq = {
            driver: [(grid[col] == driver).sum() for col in grid.columns]
            for driver in sorted(drivers)
        }
        return pd.DataFrame(freq, index=grid.columns).T

    def summary(self) -> None:
        """Print a quick readable summary of all predictions."""
        print(f"{'='*55}")
        print(f"  Melbourne GP Predictions  ({len(self.df)} entries)")
        print(f"{'='*55}\n")

        print("Drivers' champion picks:")
        print(self.driver_championship_tally().to_string(), "\n")

        print("Constructors' champion picks:")
        print(self.constructors_championship_tally().to_string(), "\n")

        print("DNF picks:")
        dnf = self.dnf_tally()
        print(dnf.to_string() if not dnf.empty else "  None", "\n")

        print("Top-ten grid (P1 → P10):")
        print(self.top_ten_grid().to_string(), "\n")

