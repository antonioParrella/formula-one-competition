"""Runner name -> 3-letter driver code resolution.

Priority:
1. explicit ``drivers.betfair_names`` mapping in config.yaml
2. already a valid 3-letter code (manual CSV convenience)
3. surname match against ``race_utils.DRIVER_MAP`` — the repo's single
   source of truth for driver codes

Unmatched runners return ``None``; callers must warn and skip rather
than guess.
"""

import _paths  # noqa: F401

import unicodedata

from race_utils import DRIVER_MAP


def _fold(name: str) -> str:
    """Casefold and strip accents so 'Pérez' matches 'Perez'."""
    decomposed = unicodedata.normalize("NFKD", name)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold().strip()


def runner_to_code(runner_name: str, config_names: dict[str, str]) -> str | None:
    name = runner_name.strip()
    folded = _fold(name)

    for cfg_name, code in config_names.items():
        if _fold(cfg_name) == folded:
            return code

    if name.upper() in DRIVER_MAP:
        return name.upper()

    tokens = folded.split()
    for code, surname in DRIVER_MAP.items():
        if _fold(surname) in tokens:
            return code
    return None
