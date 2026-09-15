"""Build a temporary, race-specific optimiser config without editing the base."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml


DRIVER_CODE = re.compile(r"^[A-Z]{3}$")


def prepare_config(
    base_path: Path,
    output_path: Path,
    race_name: str,
    year: int,
    round_num: int,
    top10: list[str],
) -> Path:
    """Copy the base config and replace only per-race context."""
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")
    if year < 2020 or round_num < 1:
        raise ValueError("Year and round must identify a modern positive F1 round")

    codes = [code.strip().upper() for code in top10]
    if len(codes) != 10 or len(set(codes)) != 10:
        raise ValueError("--top10 requires exactly ten unique driver codes")
    invalid = [code for code in codes if not DRIVER_CODE.fullmatch(code)]
    if invalid:
        raise ValueError(f"Invalid three-letter driver codes: {invalid}")

    config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("race"), dict):
        raise ValueError(f"Invalid optimiser config: {base_path}")
    if not isinstance(config.get("underdogs"), dict):
        raise ValueError(f"Config has no underdogs section: {base_path}")

    config["race"].update(
        {
            "name": race_name.strip(),
            "year": year,
            "round": round_num,
            "include_sprint_markets": False,
        }
    )
    config["underdogs"].update(
        {"source": "manual", "manual_top10": codes}
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--race-name", required=True)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--round", required=True, dest="round_num", type=int)
    parser.add_argument("--top10", required=True, nargs=10, metavar="CODE")
    args = parser.parse_args()

    path = prepare_config(
        args.base,
        args.out,
        args.race_name,
        args.year,
        args.round_num,
        args.top10,
    )
    print(f"Temporary optimiser config saved -> {path}")


if __name__ == "__main__":
    main()
