"""
Variance Decomposition for Expected Points Simulation
======================================================
Measures how much of the total variance in cumulative points is attributable
to each source of randomness: S1 timing, S2 timing, and S3 category selection.

METHOD -- FULL COMBINATORIAL + SHAPLEY VALUES:
  With 3 sources of randomness, there are 2^3 = 8 possible combinations of
  fixed/unfixed. We run all 8 simulations:

    Run 0: Nothing fixed           (baseline)
    Run 1: Fix S1
    Run 2: Fix S2
    Run 3: Fix S3
    Run 4: Fix S1 + S2
    Run 5: Fix S1 + S3
    Run 6: Fix S2 + S3
    Run 7: Fix S1 + S2 + S3        (everything fixed)

  From these 8 variance measurements, we compute each source's "fair share"
  of the total variance using Shapley values from cooperative game theory.

  The Shapley value for a source is the average marginal contribution of that
  source across ALL possible contexts. For example, S1's marginal contribution
  depends on whether S2 and S3 are also fixed:

    Context: S2 free,  S3 free  ->  marginal = V(nothing) - V(fix S1)
    Context: S2 fixed, S3 free  ->  marginal = V(fix S2)  - V(fix S1+S2)
    Context: S2 free,  S3 fixed ->  marginal = V(fix S3)  - V(fix S1+S3)
    Context: S2 fixed, S3 fixed ->  marginal = V(fix S2+S3) - V(fix all)

  Shapley(S1) = average of these four marginals.

  KEY PROPERTY: Shapley(S1) + Shapley(S2) + Shapley(S3) = V(baseline) - V(all fixed)

  This sum is exact -- no approximation, no leftover interaction terms.
  Each source gets a single clean percentage of the total variance.

USAGE:
  Requires expected_points_gpu.py in the analysis directory.
  Run: python variance_decomposition.py
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Derive paths relative to this script's location.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

sys.path.insert(0, _SCRIPT_DIR)
from expected_points import parse_data, simulate, detect_backend, prompt_backend

# ==========================================================================
#  CONFIGURATION
# ==========================================================================

DATA_PATH = os.path.join(_PROJECT_ROOT, "tier data", "items_all_normalised.csv")
OUTPUT_PATH = os.path.join(_PROJECT_ROOT, "plots")
os.makedirs(OUTPUT_PATH, exist_ok=True)

SIMULATIONS = 300_000
MAX_ROLLS = 1000
SEED = 42

# Checkpoints where we'll report the decomposition in the summary table.
CHECKPOINTS = [10, 25, 50, 100, 250, 500, 1000]


# ==========================================================================
#  HELPER: find the median trigger roll from a probability curve
# ==========================================================================

def find_median_roll(prob_curve):
    """
    Given an array where prob_curve[i] is the fraction of simulations that
    have triggered by roll i+1, find the first roll where >= 50% have triggered.
    """
    for i, p in enumerate(prob_curve):
        if p >= 0.50:
            return i + 1
    return len(prob_curve)


# ==========================================================================
#  RUN ALL 8 SIMULATIONS
# ==========================================================================

config = parse_data(DATA_PATH)

# Detect backend once, use for all 8 runs.
use_gpu = prompt_backend()

# --- Run 0: Baseline (nothing fixed) ------------------------------------

print("=" * 70)
print("Run 0/7: Baseline (nothing fixed)")
print("=" * 70)
t0 = time.perf_counter()
baseline = simulate(config, MAX_ROLLS, simulations=SIMULATIONS, seed=SEED, use_gpu=use_gpu,
                    dist_checkpoints=CHECKPOINTS)
print(f"  {time.perf_counter() - t0:.2f}s\n")

# Detect median trigger rolls from baseline to use as the fixed rolls.
s1_probs = np.array(baseline["s1_transform_prob"])
s2_probs = np.array(baseline["s2_complete_prob"])
median_s1 = find_median_roll(s1_probs)
median_s2 = find_median_roll(s2_probs)
print(f"  Median S1 transform: roll {median_s1}")
print(f"  Median S2 completion: roll {median_s2}\n")

# --- Run 1: Fix S1 only -------------------------------------------------

print("=" * 70)
print(f"Run 1/7: Fix S1 at roll {median_s1}")
print("=" * 70)
t0 = time.perf_counter()
run_s1 = simulate(config, MAX_ROLLS, simulations=SIMULATIONS, seed=SEED, use_gpu=use_gpu,
                  dist_checkpoints=CHECKPOINTS,
                  fix_s1_at_roll=median_s1)
print(f"  {time.perf_counter() - t0:.2f}s\n")

# --- Run 2: Fix S2 only -------------------------------------------------

print("=" * 70)
print(f"Run 2/7: Fix S2 at roll {median_s2}")
print("=" * 70)
t0 = time.perf_counter()
run_s2 = simulate(config, MAX_ROLLS, simulations=SIMULATIONS, seed=SEED, use_gpu=use_gpu,
                  dist_checkpoints=CHECKPOINTS,
                  fix_s2_at_roll=median_s2)
print(f"  {time.perf_counter() - t0:.2f}s\n")

# --- Run 3: Fix S3 only -------------------------------------------------

print("=" * 70)
print("Run 3/7: Fix S3 (shared draws)")
print("=" * 70)
t0 = time.perf_counter()
run_s3 = simulate(config, MAX_ROLLS, simulations=SIMULATIONS, seed=SEED, use_gpu=use_gpu,
                  dist_checkpoints=CHECKPOINTS,
                  fix_s3=True)
print(f"  {time.perf_counter() - t0:.2f}s\n")

# --- Run 4: Fix S1 + S2 -------------------------------------------------

print("=" * 70)
print(f"Run 4/7: Fix S1 + S2")
print("=" * 70)
t0 = time.perf_counter()
run_s1s2 = simulate(config, MAX_ROLLS, simulations=SIMULATIONS, seed=SEED, use_gpu=use_gpu,
                    dist_checkpoints=CHECKPOINTS,
                    fix_s1_at_roll=median_s1, fix_s2_at_roll=median_s2)
print(f"  {time.perf_counter() - t0:.2f}s\n")

# --- Run 5: Fix S1 + S3 -------------------------------------------------

print("=" * 70)
print(f"Run 5/7: Fix S1 + S3")
print("=" * 70)
t0 = time.perf_counter()
run_s1s3 = simulate(config, MAX_ROLLS, simulations=SIMULATIONS, seed=SEED, use_gpu=use_gpu,
                    dist_checkpoints=CHECKPOINTS,
                    fix_s1_at_roll=median_s1, fix_s3=True)
print(f"  {time.perf_counter() - t0:.2f}s\n")

# --- Run 6: Fix S2 + S3 -------------------------------------------------

print("=" * 70)
print(f"Run 6/7: Fix S2 + S3")
print("=" * 70)
t0 = time.perf_counter()
run_s2s3 = simulate(config, MAX_ROLLS, simulations=SIMULATIONS, seed=SEED, use_gpu=use_gpu,
                    dist_checkpoints=CHECKPOINTS,
                    fix_s2_at_roll=median_s2, fix_s3=True)
print(f"  {time.perf_counter() - t0:.2f}s\n")

# --- Run 7: Fix S1 + S2 + S3 (everything) -------------------------------

print("=" * 70)
print("Run 7/7: Fix everything")
print("=" * 70)
t0 = time.perf_counter()
run_all = simulate(config, MAX_ROLLS, simulations=SIMULATIONS, seed=SEED, use_gpu=use_gpu,
                   dist_checkpoints=CHECKPOINTS,
                   fix_s1_at_roll=median_s1, fix_s2_at_roll=median_s2, fix_s3=True)
print(f"  {time.perf_counter() - t0:.2f}s\n")


# ==========================================================================
#  COMPUTE SHAPLEY VALUES
# ==========================================================================
#
# For each source, the Shapley value is the average marginal variance
# reduction from fixing that source, across all 4 contexts (the other 2
# sources being fixed or free in every combination).
#
# Notation: V_xyz where x=S1, y=S2, z=S3. 0=free, 1=fixed.

# Extract variance arrays (one value per roll)
V_000 = np.array(baseline["cumulative_std"]) ** 2
V_100 = np.array(run_s1["cumulative_std"]) ** 2
V_010 = np.array(run_s2["cumulative_std"]) ** 2
V_001 = np.array(run_s3["cumulative_std"]) ** 2
V_110 = np.array(run_s1s2["cumulative_std"]) ** 2
V_101 = np.array(run_s1s3["cumulative_std"]) ** 2
V_011 = np.array(run_s2s3["cumulative_std"]) ** 2
V_111 = np.array(run_all["cumulative_std"]) ** 2

# Shapley value for S1:
#   The weight for each context depends on how many OTHER sources are
#   already fixed in that context: w = |S|! * (n-|S|-1)! / n!
#   where |S| = number of others fixed, n = 3 total sources.
#
#   0 others fixed: 0!*2!/3! = 2/6 = 1/3
#   1 other fixed:  1!*1!/3! = 1/6
#   2 others fixed: 2!*0!/3! = 2/6 = 1/3
#
#   Context (S2 free,  S3 free):   V_000 - V_100   weight 1/3 (0 others fixed)
#   Context (S2 fixed, S3 free):   V_010 - V_110   weight 1/6 (1 other fixed)
#   Context (S2 free,  S3 fixed):  V_001 - V_101   weight 1/6 (1 other fixed)
#   Context (S2 fixed, S3 fixed):  V_011 - V_111   weight 1/3 (2 others fixed)
w0 = 1/3  # weight when 0 others are fixed
w1 = 1/6  # weight when 1 other is fixed
w2 = 1/3  # weight when 2 others are fixed

shapley_s1 = (w0 * (V_000 - V_100) + w1 * (V_010 - V_110) +
              w1 * (V_001 - V_101) + w2 * (V_011 - V_111))

# Shapley value for S2:
#   Context (S1 free,  S3 free):   V_000 - V_010   weight 1/3
#   Context (S1 fixed, S3 free):   V_100 - V_110   weight 1/6
#   Context (S1 free,  S3 fixed):  V_001 - V_011   weight 1/6
#   Context (S1 fixed, S3 fixed):  V_101 - V_111   weight 1/3
shapley_s2 = (w0 * (V_000 - V_010) + w1 * (V_100 - V_110) +
              w1 * (V_001 - V_011) + w2 * (V_101 - V_111))

# Shapley value for S3:
#   Context (S1 free,  S2 free):   V_000 - V_001   weight 1/3
#   Context (S1 fixed, S2 free):   V_100 - V_101   weight 1/6
#   Context (S1 free,  S2 fixed):  V_010 - V_011   weight 1/6
#   Context (S1 fixed, S2 fixed):  V_110 - V_111   weight 1/3
shapley_s3 = (w0 * (V_000 - V_001) + w1 * (V_100 - V_101) +
              w1 * (V_010 - V_011) + w2 * (V_110 - V_111))

# Total explainable variance
V_explained = V_000 - V_111


# ==========================================================================
#  VERIFY: Shapley values sum to total explained variance
# ==========================================================================

shapley_sum = shapley_s1 + shapley_s2 + shapley_s3
max_deviation = np.max(np.abs(shapley_sum - V_explained))

print("=" * 70)
print("VERIFICATION")
print("=" * 70)
print(f"  Max deviation of Shapley sum from V_explained: {max_deviation:.6f}")
if max_deviation < 1.0:
    print("  Decomposition sums correctly (deviation < 1 variance point).")
else:
    print("  WARNING: Large deviation. Check for numerical issues.")


# ==========================================================================
#  PRINT RESULTS TABLE
# ==========================================================================

print(f"\n\n{'=' * 70}")
print("SHAPLEY VARIANCE DECOMPOSITION")
print("=" * 70)
print(f"S1 fixed at roll {median_s1} (baseline median transform roll)")
print(f"S2 fixed at roll {median_s2} (baseline median completion roll)")
print(f"S3 fixed via shared random draws\n")

print(f"{'Roll':<7}{'V_total':>12}{'V_residual':>12}"
      f"{'S1 timing':>12}{'S2 timing':>12}{'S3 select':>12}"
      f"{'%S1':>7}{'%S2':>7}{'%S3':>7}{'%Sum':>7}")
print("-" * 99)

for cp in CHECKPOINTS:
    idx = cp - 1
    vt = V_000[idx]
    vr = V_111[idx]
    ve = V_explained[idx]
    s1 = shapley_s1[idx]
    s2 = shapley_s2[idx]
    s3 = shapley_s3[idx]

    if ve > 0:
        p1 = s1 / ve * 100
        p2 = s2 / ve * 100
        p3 = s3 / ve * 100
        ps = p1 + p2 + p3
    else:
        p1 = p2 = p3 = ps = 0.0

    print(f"{cp:<7}{vt:>12.0f}{vr:>12.0f}"
          f"{s1:>12.0f}{s2:>12.0f}{s3:>12.0f}"
          f"{p1:>6.1f}%{p2:>6.1f}%{p3:>6.1f}%{ps:>6.1f}%")

print(f"\nV_total = total variance (baseline). V_residual = variance with everything")
print(f"fixed (noise from subset selection randomness that cannot be attributed).")
print(f"Percentages are of explained variance (V_total - V_residual).")


# ==========================================================================
#  PRINT MARGINAL CONTRIBUTIONS TABLE (detailed view)
# ==========================================================================

print(f"\n\n{'=' * 70}")
print("DETAILED: S1 MARGINAL CONTRIBUTIONS BY CONTEXT")
print("=" * 70)
print(f"{'Roll':<7}{'S2f S3f':>10}{'S2x S3f':>10}{'S2f S3x':>10}{'S2x S3x':>10}{'Average':>10}")
print("-" * 57)

for cp in CHECKPOINTS:
    idx = cp - 1
    m1 = V_000[idx] - V_100[idx]
    m2 = V_010[idx] - V_110[idx]
    m3 = V_001[idx] - V_101[idx]
    m4 = V_011[idx] - V_111[idx]
    avg = (m1 + m2 + m3 + m4) / 4
    print(f"{cp:<7}{m1:>10.0f}{m2:>10.0f}{m3:>10.0f}{m4:>10.0f}{avg:>10.0f}")

print(f"\nf = free (random), x = fixed. Each column shows how much variance drops")
print(f"when S1 is fixed, in that specific context.")


# ==========================================================================
#  PLOT 1: Percentage Contributions Over All Rolls (stacked area)
# ==========================================================================

rolls = np.arange(1, MAX_ROLLS + 1)

# Compute percentages, handling division by zero at early rolls
with np.errstate(divide='ignore', invalid='ignore'):
    pct_s1 = np.where(V_explained > 0, shapley_s1 / V_explained * 100, 0)
    pct_s2 = np.where(V_explained > 0, shapley_s2 / V_explained * 100, 0)
    pct_s3 = np.where(V_explained > 0, shapley_s3 / V_explained * 100, 0)

# Smooth with rolling average for readability (raw is noisy at early rolls)
window = 10
def smooth(arr, w):
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w)/w, mode='valid')

rolls_smooth = rolls[window-1:]
pct_s1_s = smooth(pct_s1, window)
pct_s2_s = smooth(pct_s2, window)
pct_s3_s = smooth(pct_s3, window)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# --- Top: stacked area chart ---
ax1.stackplot(rolls_smooth, pct_s1_s, pct_s2_s, pct_s3_s,
              colors=["#f59e0b", "#10b981", "#8b5cf6"],
              labels=["S1 timing", "S2 timing", "S3 selection"],
              alpha=0.7)
ax1.axhline(y=100, color="gray", linestyle="--", alpha=0.5, linewidth=1)
ax1.set_xlabel("Roll Number", fontsize=12)
ax1.set_ylabel("% of Explained Variance", fontsize=12)
ax1.set_title("Shapley Variance Decomposition: Which Source of Randomness Matters Most?",
              fontsize=14, fontweight="bold")
ax1.legend(loc="upper right", fontsize=10)
ax1.set_xlim(0, MAX_ROLLS)
ax1.set_ylim(0, 110)

# --- Bottom: absolute variance ---
ax2.plot(rolls, V_000, color="#2563eb", linewidth=2, label="Total variance (baseline)")
ax2.plot(rolls, V_111, color="#dc2626", linewidth=1.5, linestyle=":",
         label="Residual variance (all fixed)")

# Stack the Shapley contributions on top of residual to show composition
ax2.fill_between(rolls, V_111, V_111 + shapley_s1,
                 alpha=0.5, color="#f59e0b", label="S1 timing")
ax2.fill_between(rolls, V_111 + shapley_s1, V_111 + shapley_s1 + shapley_s2,
                 alpha=0.5, color="#10b981", label="S2 timing")
ax2.fill_between(rolls, V_111 + shapley_s1 + shapley_s2,
                 V_111 + shapley_s1 + shapley_s2 + shapley_s3,
                 alpha=0.5, color="#8b5cf6", label="S3 selection")

ax2.set_xlabel("Roll Number", fontsize=12)
ax2.set_ylabel("Variance (points²)", fontsize=12)
ax2.set_title("Absolute Variance Decomposition", fontsize=14, fontweight="bold")
ax2.legend(loc="upper left", fontsize=10)
ax2.set_xlim(0, MAX_ROLLS)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_PATH, "variance_decomposition.png"), dpi=150)
print(f"\nSaved: variance_decomposition.png")
plt.close()


# ==========================================================================
#  PLOT 2: Standard Deviation Comparison (all 8 runs)
# ==========================================================================

fig, ax = plt.subplots(figsize=(14, 7))

std_runs = {
    "Baseline":         np.array(baseline["cumulative_std"]),
    f"Fix S1 (r{median_s1})":  np.array(run_s1["cumulative_std"]),
    f"Fix S2 (r{median_s2})":  np.array(run_s2["cumulative_std"]),
    "Fix S3":           np.array(run_s3["cumulative_std"]),
    "Fix S1+S2":        np.array(run_s1s2["cumulative_std"]),
    "Fix S1+S3":        np.array(run_s1s3["cumulative_std"]),
    "Fix S2+S3":        np.array(run_s2s3["cumulative_std"]),
    "Fix all":          np.array(run_all["cumulative_std"]),
}

colors = ["#2563eb", "#f59e0b", "#10b981", "#8b5cf6",
          "#ea580c", "#0891b2", "#be185d", "#dc2626"]
styles = ["-", "--", "--", "--", ":", ":", ":", "-."]
widths = [2.5, 1.8, 1.8, 1.8, 1.3, 1.3, 1.3, 2]

for (label, std_arr), color, style, width in zip(
        std_runs.items(), colors, styles, widths):
    ax.plot(rolls, std_arr, color=color, linewidth=width,
            linestyle=style, label=label)

ax.set_xlabel("Roll Number", fontsize=12)
ax.set_ylabel("Standard Deviation (points)", fontsize=12)
ax.set_title("Effect of Fixing Each Randomness Source on Spread",
             fontsize=14, fontweight="bold")
ax.legend(loc="upper left", fontsize=9, ncol=2)
ax.set_xlim(0, MAX_ROLLS)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_PATH, "variance_std_comparison.png"), dpi=150)
print(f"Saved: variance_std_comparison.png")
plt.close()


# ==========================================================================
#  PLOT 3: Early rolls detail (first 150 rolls)
# ==========================================================================

fig, ax = plt.subplots(figsize=(14, 7))

mask = rolls <= 150
r_early = rolls[mask]

with np.errstate(divide='ignore', invalid='ignore'):
    pct_s1_early = np.where(V_explained[mask] > 0,
                            shapley_s1[mask] / V_explained[mask] * 100, 0)
    pct_s2_early = np.where(V_explained[mask] > 0,
                            shapley_s2[mask] / V_explained[mask] * 100, 0)
    pct_s3_early = np.where(V_explained[mask] > 0,
                            shapley_s3[mask] / V_explained[mask] * 100, 0)

ax.plot(r_early, pct_s1_early, color="#f59e0b", linewidth=2, label="S1 timing")
ax.plot(r_early, pct_s2_early, color="#10b981", linewidth=2, label="S2 timing")
ax.plot(r_early, pct_s3_early, color="#8b5cf6", linewidth=2, label="S3 selection")

ax.axvline(x=80, color="#f59e0b", linestyle=":", alpha=0.5, label="S1 pity (roll 80)")
ax.axvline(x=median_s2, color="#10b981", linestyle=":", alpha=0.5,
           label=f"S2 median (roll {median_s2})")

ax.set_xlabel("Roll Number", fontsize=12)
ax.set_ylabel("% of Explained Variance", fontsize=12)
ax.set_title("Early Rolls: Variance Decomposition Detail (Rolls 1-150)",
             fontsize=14, fontweight="bold")
ax.legend(loc="upper right", fontsize=10)
ax.set_xlim(0, 150)
ax.set_ylim(0, 105)

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_PATH, "variance_early_detail.png"), dpi=150)
print(f"Saved: variance_early_detail.png")
plt.close()


print("\nDone!")
