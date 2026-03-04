"""
join_tiers.py
=============
Concatenates the three normalised tier CSVs in order (S → A → B) and writes
the result to items_all_normalised.csv in the same folder as this script.
"""

import csv
from fractions import Fraction
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent

SOURCES = [
    SCRIPT_DIR / "S" / "s_tier_items_normalised.csv",
    SCRIPT_DIR / "A" / "a_tier_items_normalised.csv",
    SCRIPT_DIR / "B" / "b_tier_items_normalised.csv",
]

OUTPUT_FILE = SCRIPT_DIR / "items_all_normalised.csv"


def main():
    all_rows = []
    fieldnames = None

    for path in SOURCES:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            tier_rows = sorted(
                reader,
                key=lambda r: (
                    "Completion Reward" in r["id"],   # False(0) before True(1) → completions last
                    -Fraction(r["percentage_probability"]),
                ),
            )
            all_rows.extend(tier_rows)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Done — {len(all_rows)} rows written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()