"""
Pity Cap Impact Analysis
=========================
Explores how reducing the S1 pity cap affects total point accumulation
and the number of rolls required to reach point targets.

The S1 pity system guarantees a transform within N rolls (default 80).
This script sweeps across pity caps from 10 to 80 in steps of 10 and
measures the impact on expected points, variance, and rolls-to-target.

USAGE:
  Requires expected_points.py in the analysis directory.
  Run: python pity_cap_analysis.py
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


class Tee:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, filepath, mode="w"):
        self.file = open(filepath, mode, encoding="utf-8")
        self.stdout = sys.stdout
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        self.file.close()


# Derive paths relative to this script's location.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

sys.path.insert(0, _SCRIPT_DIR)
from expected_points import parse_data, simulate, prompt_backend

# ==========================================================================
#  CONFIGURATION
# ==========================================================================

DATA_PATH = os.path.join(_PROJECT_ROOT, "tier data", "items_all_normalised.csv")
OUTPUT_PATH = os.path.join(_PROJECT_ROOT, "plots")
RESULTS_PATH = os.path.join(_PROJECT_ROOT, "results")
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

# Tee all console output to a text file.
_tee = Tee(os.path.join(RESULTS_PATH, "pity_cap_analysis.txt"))
sys.stdout = _tee

SIMULATIONS = 300_000
MAX_ROLLS = 1000
SEED = 42

# Pity caps to sweep (in rolls, not 0-indexed).
PITY_CAPS = [10, 20, 30, 40, 50, 60, 70, 80]

# Checkpoints for summary tables.
# Focused on early game: a typical player stops around 150 expected points
# (~30-35 rolls). Rolls beyond 80 are included for completeness but most
# players never reach them.
CHECKPOINTS = [10, 20, 30, 50, 80, 100, 250]

# Point targets for rolls-to-target analysis.
# Centred around the realistic ~150 point stopping point.
POINT_TARGETS = [50, 100, 150, 200, 300, 500]

# RP pricing (400 RP per roll).
RP_COST_PER_ROLL = 400
RP_PACKAGES = [
    (575,   4.49),
    (1450,  10.99),
    (2850,  20.99),
    (5000,  34.99),
    (7250,  49.99),
    (15000, 99.99),
]
BEST_GBP_PER_RP = RP_PACKAGES[-1][1] / RP_PACKAGES[-1][0]
BEST_GBP_PER_ROLL = BEST_GBP_PER_RP * RP_COST_PER_ROLL

# Percentile indices in the results (matching expected_points.py output).
# percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
#                0  1   2   3   4   5   6   7   8
P50_IDX = 4  # median


# ==========================================================================
#  RUN SIMULATIONS
# ==========================================================================

config = parse_data(DATA_PATH)
use_gpu = prompt_backend()

results = {}  # cap -> simulate() return dict

for cap in PITY_CAPS:
    pity_limit = cap - 1  # convert to 0-indexed

    print("=" * 70)
    print(f"Pity cap: {cap} rolls (s1_pity_limit={pity_limit})")
    print("=" * 70)

    t0 = time.perf_counter()
    res = simulate(config, MAX_ROLLS, simulations=SIMULATIONS, seed=SEED,
                   use_gpu=use_gpu, dist_checkpoints=CHECKPOINTS,
                   s1_pity_limit=pity_limit)
    elapsed = time.perf_counter() - t0
    print(f"  {elapsed:.2f}s\n")

    results[cap] = res


# ==========================================================================
#  EXTRACT DATA
# ==========================================================================

means = {}      # cap -> array of mean cumulative points per roll
stds = {}       # cap -> array of std per roll
medians = {}    # cap -> array of median (p50) per roll
s1_probs = {}   # cap -> array of S1 transform probability per roll

for cap in PITY_CAPS:
    res = results[cap]
    means[cap] = np.array(res["cumulative_mean"])
    stds[cap] = np.array(res["cumulative_std"])
    # percentile_values is a list of lists: [roll][percentile_idx]
    pv = np.array(res["percentile_values"])
    medians[cap] = pv[:, P50_IDX]
    s1_probs[cap] = np.array(res["s1_transform_prob"])

rolls = np.arange(1, MAX_ROLLS + 1)


# ==========================================================================
#  HELPER: find roll where median reaches a point target
# ==========================================================================

def find_target_roll(median_arr, target):
    """Find the first roll where the median cumulative points >= target."""
    indices = np.where(median_arr >= target)[0]
    if len(indices) > 0:
        return indices[0] + 1  # 1-indexed
    return None  # target not reached within MAX_ROLLS


# ==========================================================================
#  TABLE 1: Summary at checkpoints
# ==========================================================================

print(f"\n\n{'=' * 70}")
print("SUMMARY: Mean & Median Points at Key Checkpoints by Pity Cap")
print("=" * 70)

for cp in CHECKPOINTS:
    idx = cp - 1
    print(f"\n  Roll {cp}:")
    print(f"  {'Cap':>6} {'Mean':>10} {'Std':>10} {'Median':>10} {'S1 prob':>10}")
    print(f"  {'-'*48}")

    for cap in PITY_CAPS:
        print(f"  {cap:>6} {means[cap][idx]:>10.1f} {stds[cap][idx]:>10.1f}"
              f" {medians[cap][idx]:>10.0f} {s1_probs[cap][idx]:>9.1%}")


# ==========================================================================
#  TABLE 2: Rolls to reach point targets
# ==========================================================================

print(f"\n\n{'=' * 70}")
print("ROLLS TO TARGET: Median Roll to Reach Each Point Target by Pity Cap")
print("=" * 70)

header = f"  {'Cap':>6}"
for target in POINT_TARGETS:
    header += f" {target:>8} pts"
print(header)
print(f"  {'-' * (6 + 12 * len(POINT_TARGETS))}")

for cap in PITY_CAPS:
    row = f"  {cap:>6}"
    for target in POINT_TARGETS:
        roll = find_target_roll(medians[cap], target)
        if roll is not None:
            row += f" {roll:>11}"
        else:
            row += f" {'>' + str(MAX_ROLLS):>11}"
    print(row)

# Same table but in GBP (best-value pricing).
print(f"\n  Cost in GBP (best-value: {BEST_GBP_PER_ROLL:.2f}/roll):\n")

header = f"  {'Cap':>6}"
for target in POINT_TARGETS:
    header += f" {target:>8} pts"
print(header)
print(f"  {'-' * (6 + 12 * len(POINT_TARGETS))}")

for cap in PITY_CAPS:
    row = f"  {cap:>6}"
    for target in POINT_TARGETS:
        roll = find_target_roll(medians[cap], target)
        if roll is not None:
            cost = roll * BEST_GBP_PER_ROLL
            row += f" {cost:>10.2f}"
        else:
            row += f" {'>' + f'{MAX_ROLLS * BEST_GBP_PER_ROLL:.0f}':>10}"
    print(row)


# ==========================================================================
#  TABLE 3: Marginal impact of each 10-roll reduction
# ==========================================================================

print(f"\n\n{'=' * 70}")
print("MARGINAL IMPACT: Extra Mean Points from Each 10-Roll Pity Reduction")
print("=" * 70)

for cp in CHECKPOINTS:
    idx = cp - 1
    print(f"\n  Roll {cp}:")
    print(f"  {'Reduction':>12} {'Mean pts':>10} {'Delta':>10} {'% gain':>10}")
    print(f"  {'-'*44}")

    prev_mean = means[PITY_CAPS[-1]][idx]  # start from cap=80

    for cap in reversed(PITY_CAPS):
        curr_mean = means[cap][idx]
        delta = curr_mean - prev_mean
        pct = (delta / prev_mean * 100) if prev_mean > 0 else 0.0
        label = f"{cap+10}->{cap}" if cap < 80 else "baseline"
        print(f"  {label:>12} {curr_mean:>10.1f} {delta:>+10.1f} {pct:>+9.2f}%")
        prev_mean = curr_mean


# ==========================================================================
#  PLOT 1: Mean Cumulative Points vs Rolls (one line per cap)
# ==========================================================================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

cmap = plt.cm.viridis
colors = [cmap(i / (len(PITY_CAPS) - 1)) for i in range(len(PITY_CAPS))]

for cap, color in zip(PITY_CAPS, colors):
    ax1.plot(rolls, means[cap], color=color, linewidth=1.8,
             label=f"Cap {cap}")

# Reference line: typical player stopping point (~150 expected points).
ax1.axhline(y=150, color="#dc2626", linestyle="--", alpha=0.6, linewidth=1.5,
            label="~150 pts (typical stop)")
ax1.set_xlabel("Roll Number", fontsize=12)
ax1.set_ylabel("Mean Cumulative Points", fontsize=12)
ax1.set_title("Effect of S1 Pity Cap on Expected Point Accumulation",
              fontsize=14, fontweight="bold")
ax1.legend(loc="upper left", fontsize=9, ncol=2)
ax1.set_xlim(0, 150)  # zoom to realistic player range
ax1.set_ylim(0, 800)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

# --- Bottom: S1 transform probability curves ---
for cap, color in zip(PITY_CAPS, colors):
    ax2.plot(rolls[:150], s1_probs[cap][:150], color=color, linewidth=1.8,
             label=f"Cap {cap}")

ax2.set_xlabel("Roll Number", fontsize=12)
ax2.set_ylabel("Fraction with S1 Transformed", fontsize=12)
ax2.set_title("S1 Transform Probability by Pity Cap (First 150 Rolls)",
              fontsize=14, fontweight="bold")
ax2.legend(loc="lower right", fontsize=9, ncol=2)
ax2.set_xlim(0, 150)
ax2.set_ylim(0, 1.05)

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_PATH, "pity_cap_mean_points.png"), dpi=150)
print(f"\nSaved: pity_cap_mean_points.png")
plt.close()


# ==========================================================================
#  PLOT 2: Rolls to Target (grouped bar chart)
# ==========================================================================

fig, ax = plt.subplots(figsize=(14, 7))

# Compute rolls-to-target matrix
target_rolls = {}  # cap -> list of rolls (one per target)
for cap in PITY_CAPS:
    target_rolls[cap] = []
    for target in POINT_TARGETS:
        roll = find_target_roll(medians[cap], target)
        target_rolls[cap].append(roll if roll is not None else MAX_ROLLS)

x = np.arange(len(POINT_TARGETS))
n_caps = len(PITY_CAPS)
total_width = 0.7
bar_w = total_width / n_caps

for i, (cap, color) in enumerate(zip(PITY_CAPS, colors)):
    offset = (i - n_caps / 2 + 0.5) * bar_w
    bars = target_rolls[cap]
    ax.bar(x + offset, bars, bar_w, color=color, label=f"Cap {cap}")

ax.set_xlabel("Point Target", fontsize=12)
ax.set_ylabel("Rolls Required (Median)", fontsize=12)
ax.set_title("Rolls to Reach Point Targets by S1 Pity Cap",
             fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"{t:,}" for t in POINT_TARGETS])
ax.legend(loc="upper left", fontsize=9, ncol=2)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_PATH, "pity_cap_rolls_to_target.png"), dpi=150)
print(f"Saved: pity_cap_rolls_to_target.png")
plt.close()


# ==========================================================================
#  PLOT 3: Marginal Value of Pity Reduction
# ==========================================================================

fig, ax = plt.subplots(figsize=(14, 7))

# For each checkpoint, show the extra mean points gained at each cap vs cap=80
markers = ["o", "s", "^", "D", "v", "P", "X"]
for cp, marker in zip(CHECKPOINTS, markers):
    idx = cp - 1
    baseline_mean = means[80][idx]
    deltas = [means[cap][idx] - baseline_mean for cap in PITY_CAPS]
    ax.plot(PITY_CAPS, deltas, marker=marker, linewidth=2, markersize=7,
            label=f"Roll {cp}")

ax.set_xlabel("S1 Pity Cap (Rolls)", fontsize=12)
ax.set_ylabel("Extra Mean Points vs Cap=80", fontsize=12)
ax.set_title("Marginal Value of Reducing S1 Pity Cap",
             fontsize=14, fontweight="bold")
ax.legend(loc="upper right", fontsize=10)
ax.set_xticks(PITY_CAPS)
ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
ax.invert_xaxis()  # lower caps on the left (more aggressive)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_PATH, "pity_cap_marginal_value.png"), dpi=150)
print(f"Saved: pity_cap_marginal_value.png")
plt.close()


print("\nDone!")

# Restore stdout and close the log file.
sys.stdout = _tee.stdout
_tee.close()
print(f"Output saved to: {os.path.join(RESULTS_PATH, 'pity_cap_analysis.txt')}")
