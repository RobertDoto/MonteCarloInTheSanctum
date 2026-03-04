import csv
from fractions import Fraction

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(SCRIPT_DIR, 'a_tier_items_raw.csv')
OUTPUT = os.path.join(SCRIPT_DIR, 'a_tier_items_normalised.csv')

with open(INPUT, newline='') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    fieldnames = reader.fieldnames

# Get target sum from the permanent row's probability
permanent_rows = [r for r in rows if r['type'] == 'permanent']
if len(permanent_rows) != 1:
    raise ValueError(f"Expected exactly 1 permanent row, found {len(permanent_rows)}")
TARGET_SUM = Fraction(permanent_rows[0]['percentage_probability'])
print(f"Target sum (from permanent row): {TARGET_SUM}  ({float(TARGET_SUM):.6f})")

one_time_sum = sum(
    Fraction(r['percentage_probability']) for r in rows if r['type'] == 'one_time'
)
print(f"Current one_time sum: {one_time_sum}  ({float(one_time_sum):.6f})")

if one_time_sum == TARGET_SUM:
    print("Sum already matches target. No normalisation needed.")
else:
    factor = TARGET_SUM / one_time_sum
    print(f"Normalisation factor: {factor}  ({float(factor):.10f})")

    for r in rows:
        if r['type'] == 'one_time':
            original = Fraction(r['percentage_probability'])
            r['percentage_probability'] = str(factor * original)

    new_sum = sum(
        Fraction(r['percentage_probability']) for r in rows if r['type'] == 'one_time'
    )
    print(f"New one_time sum: {new_sum}  ({float(new_sum):.6f})")
    assert new_sum == TARGET_SUM, f"Verification failed: {new_sum} != {TARGET_SUM}"
    print("Verified: sum matches target exactly.")

with open(OUTPUT, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved to {OUTPUT}")