"""
Pity Cap & Variance Decomposition Synthesis
=============================================
Connects two findings:
  1. Variance decomposition shows S1 timing dominates outcome uncertainty
     in the critical early-to-mid game (rolls 25-100).
  2. Pity cap analysis shows lower caps accelerate point accumulation.

This script synthesises both by measuring S1's variance contribution AT EACH
pity cap. The result quantifies the "uncertainty window" -- the roll range
where a player genuinely doesn't know whether they got lucky -- and shows how
subtle cap changes extend this window, creating additional rolls of spending
pressure.

Key context: a typical player stops around 150 expected points (~30-35
rolls). The S1 pity cap at 80 is FAR BEYOND this natural stopping point.
Most players never trigger the pity guarantee -- but its existence as a
distant anchor creates sunk-cost pressure to continue rolling.

The thesis: widening the pity cap extends the zone of peak outcome
uncertainty across the player's entire natural session. Each additional
anxious roll is a potential spending decision made under maximum
psychological pressure. Small 10-roll cap increments discretely widen
this window, nudging aggregate spending upward without any single change
feeling dramatic.

METHOD:
  For each pity cap (10, 20, ..., 80), run two simulations:
    - Baseline: normal randomness
    - Fix-S1: force S1 to trigger at its median roll for that cap

  S1's variance contribution = V(baseline) - V(fix-S1).
  As a percentage of total variance, this gives S1's share at each roll.

  The "uncertainty window" is the roll range where this share exceeds 30%.

USAGE:
  Requires expected_points.py in the analysis directory.
  Run: python pity_cap_synthesis.py
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


def find_percentile_roll(prob_curve, percentile=0.50):
    """
    Given an array where prob_curve[i] is the fraction of simulations that
    have triggered by roll i+1, find the first roll where >= percentile have
    triggered. `percentile` is a fraction in [0, 1].
    """
    for i, p in enumerate(prob_curve):
        if p >= percentile:
            return i + 1
    return len(prob_curve)

# ==========================================================================
#  CONFIGURATION
# ==========================================================================

DATA_PATH = os.path.join(_PROJECT_ROOT, "tier data", "items_all_normalised.csv")
OUTPUT_PATH = os.path.join(_PROJECT_ROOT, "plots")
RESULTS_PATH = os.path.join(_PROJECT_ROOT, "results")
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

# Tee all console output to a text file.
_tee = Tee(os.path.join(RESULTS_PATH, "pity_cap_synthesis.txt"))
sys.stdout = _tee

SIMULATIONS = 300_000
MAX_ROLLS = 1000
SEED = 42

PITY_CAPS = [10, 20, 30, 40, 50, 60, 70, 80]

# Focused on the realistic player range: most players stop around 150
# expected points (~30-35 rolls). The pity cap at 80 rolls is BEYOND the
# natural stopping point for nearly all players.
CHECKPOINTS = [10, 20, 30, 50, 80]
POINT_TARGETS = [50, 100, 150]

# RP pricing and roll costs.
RP_COST_PER_ROLL = 400
RP_PACKAGES = [
    (575,   4.49),
    (1450,  10.99),
    (2850,  20.99),
    (5000,  34.99),
    (7250,  49.99),
    (15000, 99.99),
]
# Best-value cost per roll (largest package).
BEST_GBP_PER_RP = RP_PACKAGES[-1][1] / RP_PACKAGES[-1][0]
BEST_GBP_PER_ROLL = BEST_GBP_PER_RP * RP_COST_PER_ROLL
# Worst-value (smallest package).
WORST_GBP_PER_RP = RP_PACKAGES[0][1] / RP_PACKAGES[0][0]
WORST_GBP_PER_ROLL = WORST_GBP_PER_RP * RP_COST_PER_ROLL

# Threshold for defining the "uncertainty window" (S1 variance share).
UNCERTAINTY_THRESHOLD = 0.30  # 30%

# Smoothing window for variance share curves.
SMOOTH_WINDOW = 10


# ==========================================================================
#  RUN SIMULATIONS
# ==========================================================================

config = parse_data(DATA_PATH)
use_gpu = prompt_backend()

# Phase 1: Baselines (need S1 transform curves to find median trigger rolls)
baselines = {}
median_s1_rolls = {}

for cap in PITY_CAPS:
    pity_limit = cap - 1

    print("=" * 70)
    print(f"[Baseline] Pity cap: {cap} rolls")
    print("=" * 70)

    t0 = time.perf_counter()
    res = simulate(config, MAX_ROLLS, simulations=SIMULATIONS, seed=SEED,
                   use_gpu=use_gpu, dist_checkpoints=CHECKPOINTS,
                   s1_pity_limit=pity_limit)
    print(f"  {time.perf_counter() - t0:.2f}s")

    baselines[cap] = res

    # Find median S1 trigger roll for this cap.
    s1_prob = np.array(res["s1_transform_prob"])
    med_roll = find_percentile_roll(s1_prob, 0.50)
    median_s1_rolls[cap] = med_roll
    print(f"  Median S1 trigger: roll {med_roll}\n")

# Phase 2: Fix-S1 variants
fix_s1_runs = {}

for cap in PITY_CAPS:
    pity_limit = cap - 1
    med_roll = median_s1_rolls[cap]

    print("=" * 70)
    print(f"[Fix-S1 @ r{med_roll}] Pity cap: {cap} rolls")
    print("=" * 70)

    t0 = time.perf_counter()
    res = simulate(config, MAX_ROLLS, simulations=SIMULATIONS, seed=SEED,
                   use_gpu=use_gpu, dist_checkpoints=CHECKPOINTS,
                   s1_pity_limit=pity_limit, fix_s1_at_roll=med_roll)
    print(f"  {time.perf_counter() - t0:.2f}s\n")

    fix_s1_runs[cap] = res


# ==========================================================================
#  COMPUTE S1 VARIANCE SHARE PER CAP
# ==========================================================================

rolls = np.arange(1, MAX_ROLLS + 1)

V_baseline = {}      # cap -> variance array (per roll)
V_fix_s1 = {}        # cap -> variance array with S1 fixed
s1_var_share = {}    # cap -> S1's fraction of total variance (per roll)
means = {}           # cap -> mean cumulative points per roll
medians_pts = {}     # cap -> median cumulative points per roll

P50_IDX = 4  # index of p50 in percentile list [1, 5, 10, 25, 50, 75, 90, 95, 99]

for cap in PITY_CAPS:
    vb = np.array(baselines[cap]["cumulative_std"]) ** 2
    vf = np.array(fix_s1_runs[cap]["cumulative_std"]) ** 2

    V_baseline[cap] = vb
    V_fix_s1[cap] = vf

    with np.errstate(divide='ignore', invalid='ignore'):
        share = np.where(vb > 0, (vb - vf) / vb, 0)
    s1_var_share[cap] = share

    means[cap] = np.array(baselines[cap]["cumulative_mean"])
    pv = np.array(baselines[cap]["percentile_values"])
    medians_pts[cap] = pv[:, P50_IDX]


# ==========================================================================
#  COMPUTE UNCERTAINTY WINDOWS
# ==========================================================================

def smooth(arr, w):
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w)/w, mode='valid')

uncertainty_windows = {}  # cap -> (start_roll, end_roll, peak_roll, peak_pct, width)

for cap in PITY_CAPS:
    share = smooth(s1_var_share[cap], SMOOTH_WINDOW)
    # Offset for smoothing (first valid index corresponds to roll = SMOOTH_WINDOW)
    offset = SMOOTH_WINDOW

    above = share >= UNCERTAINTY_THRESHOLD
    indices = np.where(above)[0]

    if len(indices) > 0:
        start = indices[0] + offset
        end = indices[-1] + offset
        peak_idx = np.argmax(share)
        peak_roll = peak_idx + offset
        peak_pct = share[peak_idx]
        width = end - start + 1
    else:
        start = end = peak_roll = 0
        peak_pct = np.max(share) if len(share) > 0 else 0
        width = 0

    uncertainty_windows[cap] = (start, end, peak_roll, peak_pct, width)


# ==========================================================================
#  HELPER: rolls where median reaches a target
# ==========================================================================

def find_target_roll(median_arr, target):
    indices = np.where(median_arr >= target)[0]
    return (indices[0] + 1) if len(indices) > 0 else None


# ==========================================================================
#  TABLE 1: Uncertainty Window per Cap
# ==========================================================================

print(f"\n\n{'=' * 70}")
print("UNCERTAINTY WINDOW: Roll Range Where S1 Timing Drives >30% of Variance")
print("=" * 70)
print(f"\n  {'Cap':>6} {'Med S1':>8} {'Start':>8} {'End':>8} {'Width':>8}"
      f" {'Peak roll':>10} {'Peak %':>8}")
print(f"  {'-'*60}")

for cap in PITY_CAPS:
    start, end, peak_roll, peak_pct, width = uncertainty_windows[cap]
    med = median_s1_rolls[cap]
    if width > 0:
        print(f"  {cap:>6} {med:>8} {start:>8} {end:>8} {width:>8}"
              f" {peak_roll:>10} {peak_pct:>7.1%}")
    else:
        print(f"  {cap:>6} {med:>8} {'--':>8} {'--':>8} {'0':>8}"
              f" {'--':>10} {peak_pct:>7.1%}")

print(f"\n  'Width' = number of rolls where S1 timing accounts for >{UNCERTAINTY_THRESHOLD:.0%}")
print(f"  of total outcome variance. This is the anxious zone.")


# ==========================================================================
#  TABLE 2: Spending Pressure (anxious rolls before reaching targets)
# ==========================================================================

print(f"\n\n{'=' * 70}")
print("SPENDING PRESSURE: Anxious Rolls Before Reaching Point Targets")
print("=" * 70)
print("How many rolls does a player spend inside the high-uncertainty zone")
print("before their median cumulative points reach each target?\n")

header = f"  {'Cap':>6} {'Window':>12}"
for target in POINT_TARGETS:
    header += f" {target:>6} pts"
print(header)
print(f"  {'-' * (20 + 10 * len(POINT_TARGETS))}")

for cap in PITY_CAPS:
    start, end, _, _, width = uncertainty_windows[cap]
    row = f"  {cap:>6} {'r'+str(start)+'-'+str(end) if width > 0 else '--':>12}"

    for target in POINT_TARGETS:
        target_roll = find_target_roll(medians_pts[cap], target)
        if width == 0:
            row += f" {'0':>9}"
        elif target_roll is None:
            # Target not reached; all window rolls are anxious
            row += f" {width:>9}"
        else:
            # Count window rolls that occur before reaching the target
            anxious = max(0, min(end, target_roll) - start + 1)
            if target_roll < start:
                anxious = 0
            row += f" {anxious:>9}"

    print(row)


# ==========================================================================
#  TABLE 3: Marginal Anxious Rolls per 10-Roll Cap Increase
# ==========================================================================

print(f"\n\n{'=' * 70}")
print("MARGINAL IMPACT: Extra Anxious Rolls per 10-Roll Cap Increase")
print("=" * 70)
print("Each row shows the additional uncertainty-window rolls gained by")
print("increasing the pity cap by 10 rolls.\n")

print(f"  {'Change':>12} {'Old width':>10} {'New width':>10}"
      f" {'Delta':>8} {'Extra mean pts lost':>20}")
print(f"  {'-'*64}")

for i in range(len(PITY_CAPS) - 1):
    cap_low = PITY_CAPS[i]
    cap_high = PITY_CAPS[i + 1]

    _, _, _, _, width_low = uncertainty_windows[cap_low]
    _, _, _, _, width_high = uncertainty_windows[cap_high]
    delta_width = width_high - width_low

    # Mean points difference at the end of the higher cap's window
    _, end_high, _, _, _ = uncertainty_windows[cap_high]
    if end_high > 0 and end_high <= MAX_ROLLS:
        idx = end_high - 1
        pts_diff = means[cap_low][idx] - means[cap_high][idx]
    else:
        pts_diff = 0

    print(f"  {str(cap_low)+'->'+str(cap_high):>12} {width_low:>10} {width_high:>10}"
          f" {delta_width:>+8} {pts_diff:>+19.1f}")


# ==========================================================================
#  TABLE 4: RP Pricing & Cost per Roll
# ==========================================================================

print(f"\n\n{'=' * 70}")
print("RP PRICING: Real-World Cost of Rolling")
print("=" * 70)
print(f"  Each roll costs {RP_COST_PER_ROLL} RP.\n")

print(f"  {'Package':>10} {'GBP':>8} {'GBP/RP':>10} {'Rolls':>8} {'GBP/roll':>10}")
print(f"  {'-'*50}")

for rp, gbp in RP_PACKAGES:
    gbp_per_rp = gbp / rp
    n_rolls = rp / RP_COST_PER_ROLL
    gbp_per_roll = gbp_per_rp * RP_COST_PER_ROLL
    print(f"  {rp:>8} RP {gbp:>7.2f} {gbp_per_rp:>9.4f} {n_rolls:>8.1f} {gbp_per_roll:>9.2f}")

print(f"\n  Best value:  {BEST_GBP_PER_ROLL:.2f}/roll (largest package)")
print(f"  Worst value: {WORST_GBP_PER_ROLL:.2f}/roll (smallest package)")


# ==========================================================================
#  TABLE 5: Cost to reach 150 points by pity cap
# ==========================================================================

print(f"\n\n{'=' * 70}")
print("COST TO TARGET: GBP to Reach Point Milestones by Pity Cap")
print("=" * 70)
print(f"  Based on best-value pricing ({BEST_GBP_PER_ROLL:.2f}/roll).\n")

print(f"  {'Cap':>6}", end="")
for target in POINT_TARGETS:
    print(f" {str(target)+' pts':>10}", end="")
print(f" {'Pity cost':>12}")
print(f"  {'-' * (6 + 10 * len(POINT_TARGETS) + 12)}")

for cap in PITY_CAPS:
    row = f"  {cap:>6}"
    for target in POINT_TARGETS:
        target_roll = find_target_roll(medians_pts[cap], target)
        if target_roll is not None:
            cost = target_roll * BEST_GBP_PER_ROLL
            row += f" {cost:>9.2f}"
        else:
            row += f" {'>' + f'{MAX_ROLLS * BEST_GBP_PER_ROLL:.0f}':>9}"
    # Cost to reach the pity cap itself
    pity_cost = cap * BEST_GBP_PER_ROLL
    row += f" {pity_cost:>11.2f}"
    print(row)


# ==========================================================================
#  TABLE 6: Marginal GBP cost per 10-roll cap increase
# ==========================================================================

print(f"\n\n{'=' * 70}")
print("MARGINAL SPEND: Extra GBP per 10-Roll Cap Increase")
print("=" * 70)
print("If a player is persuaded to roll through the entire uncertainty window,")
print("how much extra do they spend for each 10-roll cap increase?\n")

print(f"  {'Change':>12} {'Extra rolls':>12} {'Best case':>12} {'Worst case':>12}")
print(f"  {'-'*52}")

total_extra_rolls_low_to_high = 0
for i in range(len(PITY_CAPS) - 1):
    cap_lo = PITY_CAPS[i]
    cap_hi = PITY_CAPS[i + 1]

    _, _, _, _, w_lo = uncertainty_windows[cap_lo]
    _, _, _, _, w_hi = uncertainty_windows[cap_hi]
    delta = w_hi - w_lo
    total_extra_rolls_low_to_high += delta

    best_cost = delta * BEST_GBP_PER_ROLL
    worst_cost = delta * WORST_GBP_PER_ROLL

    print(f"  {str(cap_lo)+'->'+str(cap_hi):>12} {delta:>+12}"
          f" {best_cost:>11.2f} {worst_cost:>11.2f}")

print(f"\n  Total extra anxious rolls (cap {PITY_CAPS[0]}->{PITY_CAPS[-1]}):"
      f" {total_extra_rolls_low_to_high}")
print(f"  Total extra spend (best):  "
      f"{total_extra_rolls_low_to_high * BEST_GBP_PER_ROLL:.2f}")
print(f"  Total extra spend (worst): "
      f"{total_extra_rolls_low_to_high * WORST_GBP_PER_ROLL:.2f}")


# ==========================================================================
#  TABLE 7: Summary Narrative
# ==========================================================================

print(f"\n\n{'=' * 70}")
print("SYNTHESIS: THE DISCRETE GOALPOST WIDENING EFFECT")
print("=" * 70)

cap_min = PITY_CAPS[0]
cap_max = PITY_CAPS[-1]
w_min = uncertainty_windows[cap_min][4]
w_max = uncertainty_windows[cap_max][4]

# Compute the typical stopping roll (median roll to reach ~150 points) at cap=80.
stop_roll_80 = find_target_roll(medians_pts[cap_max], 150)
stop_roll_min = find_target_roll(medians_pts[cap_min], 150)

# What fraction of players have triggered S1 by the stopping roll?
s1_prob_at_stop = np.array(baselines[cap_max]["s1_transform_prob"])
s1_pct_at_stop = s1_prob_at_stop[stop_roll_80 - 1] if stop_roll_80 else 0

# Cost to reach 150 pts and cost to reach pity cap.
cost_to_150 = (stop_roll_80 or 0) * BEST_GBP_PER_ROLL
cost_to_pity = cap_max * BEST_GBP_PER_ROLL
extra_cost_to_pity = cost_to_pity - cost_to_150

print(f"""
  CRITICAL CONTEXT: A typical player stops around 150 expected points,
  which is reached at approximately roll {stop_roll_80} (cap={cap_max}) or roll {stop_roll_min}
  (cap={cap_min}). The S1 pity cap at roll 80 is FAR BEYOND this natural
  stopping point.

  THE MONETISATION REALITY:

  At {RP_COST_PER_ROLL} RP per roll and best-value pricing ({BEST_GBP_PER_ROLL:.2f}/roll):
    - Reaching 150 pts (roll ~{stop_roll_80}):  ~{cost_to_150:.2f}
    - Reaching the pity cap (roll {cap_max}):   ~{cost_to_pity:.2f}
    - Gap (the sunk-cost trap):       ~{extra_cost_to_pity:.2f}

  A player at 150 points has spent ~{cost_to_150:.2f}. The pity guarantee
  is another ~{extra_cost_to_pity:.2f} away. This gap is the monetisation sweet
  spot: large enough to extract significant additional spend, but framed as
  "finishing what you started" rather than a new purchase.

  At the typical stopping roll ({stop_roll_80}) with cap={cap_max}, only {s1_pct_at_stop:.1%} of
  players have triggered S1. The remaining {1 - s1_pct_at_stop:.1%} leave without ever
  experiencing the pity guarantee. The cap exists not as a practical
  mechanic but as a psychological anchor.

  THE UNCERTAINTY WINDOW AND SPENDING PRESSURE:

  At cap={cap_min}:  The uncertainty window is {w_min} rolls wide.
  At cap={cap_max}:  The uncertainty window is {w_max} rolls wide.

  This is a {w_max - w_min}-roll expansion of the anxious zone. But the key
  insight is that this window overlaps almost perfectly with the typical
  player's entire session (rolls 1-{stop_roll_80 or '??'}). The player spends their
  ENTIRE experience inside the zone of maximum outcome uncertainty.

  The pity cap at {cap_max} rolls serves three psychological functions:

  1. SUNK COST ANCHOR: "I'm already {stop_roll_80 or '??'} rolls in, and the guarantee is at
     {cap_max}. I'm {cap_max - (stop_roll_80 or 0)} rolls away from a sure thing." This reframes
     stopping as a loss rather than a natural endpoint. The gap represents
     ~{extra_cost_to_pity:.2f} of additional spend at best-value pricing,
     or ~{(cap_max - (stop_roll_80 or 0)) * WORST_GBP_PER_ROLL:.2f} at worst-value.

  2. NEAR-MISS FRAMING: With only {s1_pct_at_stop:.1%} triggering S1 by the typical
     stop, most players feel they "almost" got lucky. A lower cap would
     resolve this uncertainty earlier, reducing the emotional hook.

  3. DISCRETE INVISIBILITY: Each 10-roll cap increase adds only a modest
     number of anxious rolls (see marginal table above). No single change
     feels dramatic, but the cumulative effect from cap={cap_min} to cap={cap_max}
     is substantial -- {total_extra_rolls_low_to_high} extra anxious rolls worth
     {total_extra_rolls_low_to_high * BEST_GBP_PER_ROLL:.2f}-\
{total_extra_rolls_low_to_high * WORST_GBP_PER_ROLL:.2f} in potential spend.

  The interaction effects from the variance decomposition confirm that S1
  timing acts nearly independently of S2 and S3 (all interactions <1.5% of
  baseline variance). This means the pity cap is a clean, isolated lever --
  adjusting it does not create unpredictable knock-on effects. It is a
  precision tool for tuning spending pressure.

  Key checkpoints (cap={cap_max} vs cap={cap_min}):""")

for cp in CHECKPOINTS:
    idx = cp - 1
    diff = means[cap_min][idx] - means[cap_max][idx]
    pct = (diff / means[cap_max][idx] * 100) if means[cap_max][idx] > 0 else 0
    cost_diff = diff  # in points, not rolls -- show the roll equivalent too
    print(f"    Roll {cp:>4}: cap={cap_min} gives {diff:>+8.1f} more mean points"
          f" ({pct:>+5.1f}% vs cap={cap_max})")


# ==========================================================================
#  PLOT 1: S1 Variance Share by Pity Cap
# ==========================================================================

fig, ax = plt.subplots(figsize=(14, 7))

cmap = plt.cm.viridis
colors = [cmap(i / (len(PITY_CAPS) - 1)) for i in range(len(PITY_CAPS))]

rolls_smooth = rolls[SMOOTH_WINDOW - 1:]

for cap, color in zip(PITY_CAPS, colors):
    share_smooth = smooth(s1_var_share[cap], SMOOTH_WINDOW) * 100
    ax.plot(rolls_smooth, share_smooth, color=color, linewidth=1.8,
            label=f"Cap {cap}")

ax.axhline(y=UNCERTAINTY_THRESHOLD * 100, color="#dc2626", linestyle="--",
           alpha=0.6, linewidth=1.5, label=f"Uncertainty threshold ({UNCERTAINTY_THRESHOLD:.0%})")
ax.fill_between(rolls_smooth, 0, UNCERTAINTY_THRESHOLD * 100,
                alpha=0.05, color="#dc2626")

ax.set_xlabel("Roll Number", fontsize=12)
ax.set_ylabel("S1 Timing Share of Total Variance (%)", fontsize=12)
ax.set_title("How Pity Cap Controls the Uncertainty Window",
             fontsize=14, fontweight="bold")
ax.legend(loc="upper right", fontsize=9, ncol=2)
ax.set_xlim(0, 250)  # zoom to where the action is
ax.set_ylim(0, 100)

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_PATH, "synthesis_variance_share.png"), dpi=150)
print(f"\nSaved: synthesis_variance_share.png")
plt.close()


# ==========================================================================
#  PLOT 2: Combined Uncertainty + Point Accumulation (cap=80 vs cap=40)
# ==========================================================================

cap_high = 80
cap_low = 40

fig, ax1 = plt.subplots(figsize=(14, 7))
ax2 = ax1.twinx()

# Shade uncertainty windows
start_h, end_h, _, _, _ = uncertainty_windows[cap_high]
start_l, end_l, _, _, _ = uncertainty_windows[cap_low]

if end_h > 0:
    ax1.axvspan(start_h, end_h, alpha=0.15, color="#dc2626",
                label=f"Uncertainty zone (cap={cap_high})")
if end_l > 0:
    ax1.axvspan(start_l, end_l, alpha=0.15, color="#2563eb",
                label=f"Uncertainty zone (cap={cap_low})")

# S1 variance share curves
share_h = smooth(s1_var_share[cap_high], SMOOTH_WINDOW) * 100
share_l = smooth(s1_var_share[cap_low], SMOOTH_WINDOW) * 100

ax1.plot(rolls_smooth, share_h, color="#dc2626", linewidth=2.5,
         label=f"S1 var share (cap={cap_high})")
ax1.plot(rolls_smooth, share_l, color="#2563eb", linewidth=2.5,
         label=f"S1 var share (cap={cap_low})")
ax1.axhline(y=UNCERTAINTY_THRESHOLD * 100, color="gray", linestyle=":",
            alpha=0.5, linewidth=1)

ax1.set_xlabel("Roll Number", fontsize=12)
ax1.set_ylabel("S1 Variance Share (%)", fontsize=12, color="#555")
ax1.set_xlim(0, 150)  # zoom to realistic player range
ax1.set_ylim(0, 100)

# Mark the typical stopping point
if stop_roll_80:
    ax1.axvline(x=stop_roll_80, color="#6b7280", linestyle="-.", alpha=0.7,
                linewidth=1.5, label=f"Typical stop (~150 pts, roll {stop_roll_80})")

# Mean points overlay
ax2.plot(rolls, means[cap_high], color="#dc2626", linewidth=1.5,
         linestyle="--", alpha=0.6, label=f"Mean pts (cap={cap_high})")
ax2.plot(rolls, means[cap_low], color="#2563eb", linewidth=1.5,
         linestyle="--", alpha=0.6, label=f"Mean pts (cap={cap_low})")
ax2.axhline(y=150, color="#6b7280", linestyle=":", alpha=0.3)
ax2.set_ylabel("Mean Cumulative Points", fontsize=12, color="#555")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

ax1.set_title("The Uncertainty-Spending Nexus: Cap 80 vs Cap 40",
              fontsize=14, fontweight="bold")

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_PATH, "synthesis_combined.png"), dpi=150)
print(f"Saved: synthesis_combined.png")
plt.close()


# ==========================================================================
#  PLOT 3: Marginal Anxious Rolls per Cap Increase
# ==========================================================================

fig, ax = plt.subplots(figsize=(14, 7))

changes = []
delta_widths = []
delta_pts_at_stop = []

# Use roll 50 as representative of the typical player range (~150 pts).
REFERENCE_ROLL = 50

for i in range(len(PITY_CAPS) - 1):
    cap_lo = PITY_CAPS[i]
    cap_hi = PITY_CAPS[i + 1]

    _, _, _, _, w_lo = uncertainty_windows[cap_lo]
    _, _, _, _, w_hi = uncertainty_windows[cap_hi]

    changes.append(f"{cap_lo} → {cap_hi}")
    delta_widths.append(w_hi - w_lo)
    # Points lost at reference roll from this cap increase
    delta_pts_at_stop.append(means[cap_lo][REFERENCE_ROLL - 1] -
                             means[cap_hi][REFERENCE_ROLL - 1])

x = np.arange(len(changes))

bars = ax.bar(x, delta_widths, 0.5, color="#dc2626", alpha=0.7,
              label="Extra anxious rolls")
ax.bar_label(bars, fmt="%+d", fontsize=11, fontweight="bold")

# Overlay: points lost
ax2 = ax.twinx()
ax2.plot(x, delta_pts_at_stop, color="#2563eb", marker="o", linewidth=2,
         markersize=8, label=f"Extra mean pts lost (roll {REFERENCE_ROLL})")
ax2.set_ylabel(f"Mean Points Lost at Roll {REFERENCE_ROLL}", fontsize=12,
               color="#2563eb")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

ax.set_xlabel("Pity Cap Increase", fontsize=12)
ax.set_ylabel("Additional Uncertainty Window Rolls", fontsize=12, color="#dc2626")
ax.set_title("Discrete Goalpost Widening: Marginal Impact of Each Cap Increase",
             fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(changes, fontsize=10)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_PATH, "synthesis_marginal_anxious.png"), dpi=150)
print(f"Saved: synthesis_marginal_anxious.png")
plt.close()


print("\nDone!")

# Restore stdout and close the log file.
sys.stdout = _tee.stdout
_tee.close()
print(f"Output saved to: {os.path.join(RESULTS_PATH, 'pity_cap_synthesis.txt')}")
