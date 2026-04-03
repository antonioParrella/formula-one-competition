def clean_race_name(name: str) -> str:
    """Strip ' GP' or ' Grand Prix' suffixes from a race name.

    Ensures consistent naming across tips files, results files,
    and filenames — just the circuit name, no suffix.
    """
    return name.rstrip().removesuffix(" Grand Prix").removesuffix(" GP").strip()
