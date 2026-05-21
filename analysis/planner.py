"""
Roll Planner
============
Interactive CLI for estimating expected points from rolls, or rolls needed
to reach a points target, broken down by luck tier.

USAGE:
    python analysis/planner.py

Requires:
    results/simulation_results.csv
"""

import os
import sys
import csv

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
_CSV_PATH = os.path.join(_RESULTS_DIR, "simulation_results.csv")

sys.path.insert(0, _SCRIPT_DIR)
from costs import roll_cost

TIERS = [
    ("Very unlucky", "p1",  1),
    ("",             "p5",  5),
    ("",             "p10", 10),
    ("Below avg",    "p25", 25),
    ("Average",      "p50", 50),
    ("Above avg",    "p75", 75),
    ("",             "p90", 90),
    ("",             "p95", 95),
    ("Very lucky",   "p99", 99),
]

# ── data loading ─────────────────────────────────────────────────────────────

def load_csv():
    if not os.path.exists(_CSV_PATH):
        print(f"ERROR: {_CSV_PATH} not found. Run expected_points.py first.")
        sys.exit(1)

    rows = {}
    with open(_CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            roll = int(row["roll"])
            rows[roll] = {k: float(v) for k, v in row.items() if k != "roll"}
            rows[roll]["roll"] = roll
    return rows


# ── helpers ───────────────────────────────────────────────────────────────────

def prompt(message, valid):
    valid_upper = [v.upper() for v in valid]
    while True:
        answer = input(message).strip().upper()
        if answer in valid_upper:
            return answer
        print(f"  Please enter one of: {', '.join(valid_upper)}")


def prompt_int(message, min_val=1, max_val=None):
    while True:
        try:
            val = int(input(message).strip())
            if val < min_val:
                print(f"  Must be at least {min_val}.")
                continue
            if max_val is not None and val > max_val:
                print(f"  Must be at most {max_val:,}.")
                continue
            return val
        except ValueError:
            print("  Please enter a whole number.")


# ── mode R: rolls → points ────────────────────────────────────────────────────

def mode_rolls(csv_data):
    max_roll = max(csv_data.keys())
    n_rolls = prompt_int(f"  How many rolls are you planning? (1–{max_roll:,}): ",
                         min_val=1, max_val=max_roll)

    row = csv_data[n_rolls]
    cost = roll_cost(n_rolls)

    print(f"\n  Expected cumulative points after {n_rolls:,} rolls"
          f"  (est. cost: £{cost:.2f}):\n")
    print(f"  {'Tier':<14}  {'Percentile':>10}  {'Points':>12}")
    print(f"  {'-'*14}  {'-'*10}  {'-'*12}")
    for label, col, pct in TIERS:
        pts = row[col]
        print(f"  {label:<14}  {'p' + str(pct):>10}  {pts:>12,.0f}")
    print(f"\n  Mean: {row['mean']:,.0f}  |  Std dev: {row['std']:,.0f}")


# ── mode P: points → rolls ────────────────────────────────────────────────────

def mode_points(csv_data):
    target = prompt_int("  How many points are you targeting? (e.g. 5000): ",
                        min_val=1)

    rolls_sorted = sorted(csv_data.keys())

    print(f"\n  Rolls needed to reach {target:,} points:\n")
    print(f"  {'Tier':<14}  {'Percentile':>10}  {'Rolls needed':>12}  {'Est. cost':>10}")
    print(f"  {'-'*14}  {'-'*10}  {'-'*12}  {'-'*10}")

    for label, col, pct in TIERS:
        reached = None
        for roll in rolls_sorted:
            if csv_data[roll][col] >= target:
                reached = roll
                break
        tier = label if label else ""
        if reached is None:
            print(f"  {tier:<14}  {'p' + str(pct):>10}  {'> ' + str(rolls_sorted[-1]):>12}  {'':>10}")
        else:
            cost = roll_cost(reached)
            print(f"  {tier:<14}  {'p' + str(pct):>10}  {reached:>12,}  £{cost:>9.2f}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n  Roll Planner")
    print("  ============")
    print("  What would you like to do?")
    print("    (R) I have a number of rolls — show expected points")
    print("    (P) I have a points target  — show rolls needed")
    print("    (Q) Quit\n")

    csv_data = load_csv()

    while True:
        choice = prompt("  Choice (R/P/Q): ", ["R", "P", "Q"])

        if choice == "Q":
            print("  Bye.")
            break
        elif choice == "R":
            mode_rolls(csv_data)
        elif choice == "P":
            mode_points(csv_data)

        print()
        again = prompt("  Run another query? (Y/N): ", ["Y", "N"])
        if again == "N":
            print("  Bye.")
            break
        print()


if __name__ == "__main__":
    main()
