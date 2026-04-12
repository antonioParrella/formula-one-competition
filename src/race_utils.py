def clean_race_name(name: str) -> str:
    """Strip ' GP' or ' Grand Prix' suffixes from a race name.

    Ensures consistent naming across tips files, results files,
    and filenames — just the circuit name, no suffix.
    """
    return name.rstrip().removesuffix(" Grand Prix").removesuffix(" GP").strip()


# Driver code → surname.  Add new drivers here — all other modules import
# from race_utils.DRIVER_MAP instead of maintaining their own copies.
DRIVER_MAP = {
    "NOR": "Norris",
    "PIA": "Piastri",
    "LEC": "Leclerc",
    "HAM": "Hamilton",
    "RUS": "Russell",
    "VER": "Verstappen",
    "ANT": "Antonelli",
    "HAD": "Hadjar",
    "SAI": "Sainz",
    "ALO": "Alonso",
    "ALB": "Albon",
    "GAS": "Gasly",
    "OCO": "Ocon",
    "STR": "Stroll",
    "HUL": "Hulkenberg",
    "TSU": "Tsunoda",
    "LAW": "Lawson",
    "BEA": "Bearman",
    "BOR": "Bortoleto",
    "LIN": "Lindblad",
    "DOO": "Doohan",
    "COL": "Colapinto",
    "BOT": "Bottas",
    "PER": "Pérez",
}
