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
median_s1 = find_percentile_roll(s1_probs, 0.50)
median_s2 = find_percentile_roll(s2_probs, 0.50)
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
#  INTERACTION EFFECTS (2^3 Factorial ANOVA Contrasts)
# ==========================================================================
#
# With 3 binary factors (fix/free) and all 8 combinations, we can compute
# the classic 2^3 factorial contrasts to isolate interaction effects.
#
# Main effect of fixing Si = mean(V when Si fixed) - mean(V when Si free)
#   Negative means fixing reduces variance (expected).
#
# Two-way interaction S_i x S_j measures whether the variance reduction from
# fixing S_i changes depending on whether S_j is also fixed:
#   INT_ij = 0.25 * sum of (sign) * V_xyz  using the standard ANOVA contrast.
#
# Three-way interaction captures residual non-additivity.
#
# Key property: all interaction contrasts sum to zero by construction in a
# balanced 2^3 factorial.

# Main effects (average variance reduction from fixing each source)
ME_S1 = 0.25 * ((V_100 - V_000) + (V_110 - V_010) + (V_101 - V_001) + (V_111 - V_011))
ME_S2 = 0.25 * ((V_010 - V_000) + (V_110 - V_100) + (V_011 - V_001) + (V_111 - V_101))
ME_S3 = 0.25 * ((V_001 - V_000) + (V_101 - V_100) + (V_011 - V_010) + (V_111 - V_110))

# Two-way interactions
# INT_S1S2: does fixing S1 reduce MORE variance when S2 is also fixed?
#   = 0.25 * [(V_110 - V_010) - (V_100 - V_000) + (V_111 - V_011) - (V_101 - V_001)]
INT_S1S2 = 0.25 * ((V_000 - V_100 - V_010 + V_110) + (V_001 - V_101 - V_011 + V_111))
INT_S1S3 = 0.25 * ((V_000 - V_100 - V_001 + V_101) + (V_010 - V_110 - V_011 + V_111))
INT_S2S3 = 0.25 * ((V_000 - V_010 - V_001 + V_011) + (V_100 - V_110 - V_101 + V_111))

# Three-way interaction
INT_S1S2S3 = 0.125 * (
    V_000 - V_100 - V_010 - V_001 + V_110 + V_101 + V_011 - V_111
)

# --- Print interaction table ---
print(f"\n\n{'=' * 70}")
print("INTERACTION EFFECTS (2^3 FACTORIAL ANOVA CONTRASTS)")
print("=" * 70)
print("Main effects are average variance change from fixing a source.")
print("Interactions show whether sources amplify (+) or are redundant (-)")
print("with each other.\n")

print(f"{'Roll':<7}{'ME(S1)':>10}{'ME(S2)':>10}{'ME(S3)':>10}"
      f"{'S1xS2':>10}{'S1xS3':>10}{'S2xS3':>10}{'S1xS2xS3':>10}")
print("-" * 77)

for cp in CHECKPOINTS:
    idx = cp - 1
    print(f"{cp:<7}{ME_S1[idx]:>10.0f}{ME_S2[idx]:>10.0f}{ME_S3[idx]:>10.0f}"
          f"{INT_S1S2[idx]:>10.0f}{INT_S1S3[idx]:>10.0f}{INT_S2S3[idx]:>10.0f}"
          f"{INT_S1S2S3[idx]:>10.0f}")

# Also show as % of total variance for interpretability
print(f"\nAs percentage of baseline variance:\n")
print(f"{'Roll':<7}{'ME(S1)':>10}{'ME(S2)':>10}{'ME(S3)':>10}"
      f"{'S1xS2':>10}{'S1xS3':>10}{'S2xS3':>10}{'S1xS2xS3':>10}")
print("-" * 77)

for cp in CHECKPOINTS:
    idx = cp - 1
    vt = V_000[idx]
    if vt > 0:
        print(f"{cp:<7}{ME_S1[idx]/vt*100:>9.1f}%{ME_S2[idx]/vt*100:>9.1f}%"
              f"{ME_S3[idx]/vt*100:>9.1f}%{INT_S1S2[idx]/vt*100:>9.1f}%"
              f"{INT_S1S3[idx]/vt*100:>9.1f}%{INT_S2S3[idx]/vt*100:>9.1f}%"
              f"{INT_S1S2S3[idx]/vt*100:>9.1f}%")
    else:
        print(f"{cp:<7}{'0.0%':>10}{'0.0%':>10}{'0.0%':>10}"
              f"{'0.0%':>10}{'0.0%':>10}{'0.0%':>10}{'0.0%':>10}")

print(f"\nNegative main effects = fixing that source reduces variance (expected).")
print(f"Positive interaction = sources amplify each other's variance.")
print(f"Negative interaction = sources are redundant (fixing both helps less")
print(f"than the sum of fixing each alone).")


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


# ==========================================================================
#  PLOT 4: Interaction Effects Over All Rolls
# ==========================================================================

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 10))

# --- Top: two-way interactions as % of baseline variance ---
with np.errstate(divide='ignore', invalid='ignore'):
    pct_int12 = np.where(V_000 > 0, INT_S1S2 / V_000 * 100, 0)
    pct_int13 = np.where(V_000 > 0, INT_S1S3 / V_000 * 100, 0)
    pct_int23 = np.where(V_000 > 0, INT_S2S3 / V_000 * 100, 0)
    pct_int123 = np.where(V_000 > 0, INT_S1S2S3 / V_000 * 100, 0)

# Smooth for readability
pct_int12_s = smooth(pct_int12, window)
pct_int13_s = smooth(pct_int13, window)
pct_int23_s = smooth(pct_int23, window)
pct_int123_s = smooth(pct_int123, window)

ax_top.plot(rolls_smooth, pct_int12_s, color="#ea580c", linewidth=2, label="S1 × S2")
ax_top.plot(rolls_smooth, pct_int13_s, color="#0891b2", linewidth=2, label="S1 × S3")
ax_top.plot(rolls_smooth, pct_int23_s, color="#be185d", linewidth=2, label="S2 × S3")
ax_top.plot(rolls_smooth, pct_int123_s, color="#6b7280", linewidth=1.5,
            linestyle="--", label="S1 × S2 × S3")
ax_top.axhline(y=0, color="gray", linestyle="-", alpha=0.3, linewidth=1)

ax_top.set_xlabel("Roll Number", fontsize=12)
ax_top.set_ylabel("% of Baseline Variance", fontsize=12)
ax_top.set_title("Interaction Effects: Do Sources Amplify or Cancel Each Other?",
                 fontsize=14, fontweight="bold")
ax_top.legend(loc="best", fontsize=10)
ax_top.set_xlim(0, MAX_ROLLS)

# --- Bottom: absolute interaction magnitudes ---
ax_bot.plot(rolls, np.abs(INT_S1S2), color="#ea580c", linewidth=1.5, alpha=0.7, label="|S1 × S2|")
ax_bot.plot(rolls, np.abs(INT_S1S3), color="#0891b2", linewidth=1.5, alpha=0.7, label="|S1 × S3|")
ax_bot.plot(rolls, np.abs(INT_S2S3), color="#be185d", linewidth=1.5, alpha=0.7, label="|S2 × S3|")
ax_bot.plot(rolls, V_explained, color="#2563eb", linewidth=2, linestyle=":", alpha=0.5,
            label="Total explained var")

ax_bot.set_xlabel("Roll Number", fontsize=12)
ax_bot.set_ylabel("Variance (points²)", fontsize=12)
ax_bot.set_title("Interaction Magnitudes vs Total Explained Variance",
                 fontsize=14, fontweight="bold")
ax_bot.legend(loc="upper left", fontsize=10)
ax_bot.set_xlim(0, MAX_ROLLS)
ax_bot.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_PATH, "variance_interactions.png"), dpi=150)
print(f"Saved: variance_interactions.png")
plt.close()


# ==========================================================================
#  SENSITIVITY ANALYSIS: Sweep S1/S2 fix-points across percentiles
# ==========================================================================
#
# The baseline decomposition fixes S1 and S2 at their median (p50) trigger
# rolls. Here we repeat the Shapley calculation with fix-points at the 25th
# and 75th percentiles to test robustness.
#
# Key optimisation: when varying only S1's fix-point, the 4 runs that do NOT
# fix S1 (runs 0, 2, 3, 6) are unchanged. Only the 4 runs involving S1
# (runs 1, 4, 5, 7) need re-running. Same logic applies for S2.

print(f"\n\n{'=' * 70}")
print("SENSITIVITY ANALYSIS: Varying Fix-Point Percentiles")
print("=" * 70)

# Detect p25 and p75 trigger rolls from baseline probability curves.
p25_s1 = find_percentile_roll(s1_probs, 0.25)
p75_s1 = find_percentile_roll(s1_probs, 0.75)
p25_s2 = find_percentile_roll(s2_probs, 0.25)
p75_s2 = find_percentile_roll(s2_probs, 0.75)

print(f"  S1 trigger rolls:  p25={p25_s1}  p50={median_s1}  p75={p75_s1}")
print(f"  S2 trigger rolls:  p25={p25_s2}  p50={median_s2}  p75={p75_s2}\n")

# Helper: compute Shapley values from 8 variance arrays
def compute_shapley(v000, v100, v010, v001, v110, v101, v011, v111):
    w0, w1, w2 = 1/3, 1/6, 1/3
    s1 = w0*(v000-v100) + w1*(v010-v110) + w1*(v001-v101) + w2*(v011-v111)
    s2 = w0*(v000-v010) + w1*(v100-v110) + w1*(v001-v011) + w2*(v101-v111)
    s3 = w0*(v000-v001) + w1*(v100-v101) + w1*(v010-v011) + w2*(v110-v111)
    return s1, s2, s3

# Helper: run a single simulation with given fix params and return variance array
def run_and_get_var(label, fix_s1=None, fix_s2=None, fix_s3_flag=False):
    print(f"  {label} ...", end=" ", flush=True)
    t = time.perf_counter()
    res = simulate(config, MAX_ROLLS, simulations=SIMULATIONS, seed=SEED,
                   use_gpu=use_gpu, dist_checkpoints=CHECKPOINTS,
                   fix_s1_at_roll=fix_s1, fix_s2_at_roll=fix_s2, fix_s3=fix_s3_flag)
    print(f"{time.perf_counter() - t:.2f}s")
    return np.array(res["cumulative_std"]) ** 2

# Build a cache of already-computed runs to avoid redundant simulations.
# Key: (s1_fix_roll or None, s2_fix_roll or None, fix_s3 bool)
run_cache = {
    (None,      None,      False): V_000,
    (median_s1, None,      False): V_100,
    (None,      median_s2, False): V_010,
    (None,      None,      True):  V_001,
    (median_s1, median_s2, False): V_110,
    (median_s1, None,      True):  V_101,
    (None,      median_s2, True):  V_011,
    (median_s1, median_s2, True):  V_111,
}

def get_or_run(s1_fix, s2_fix, s3_flag):
    key = (s1_fix, s2_fix, s3_flag)
    if key not in run_cache:
        parts = []
        if s1_fix is not None: parts.append(f"S1@{s1_fix}")
        if s2_fix is not None: parts.append(f"S2@{s2_fix}")
        if s3_flag: parts.append("S3")
        label = "Fix " + "+".join(parts) if parts else "Baseline"
        run_cache[key] = run_and_get_var(label, fix_s1=s1_fix, fix_s2=s2_fix,
                                         fix_s3_flag=s3_flag)
    return run_cache[key]

# Run the full Shapley decomposition for each (s1_pct, s2_pct) combo.
s1_fixpoints = [("p25", p25_s1), ("p50", median_s1), ("p75", p75_s1)]
s2_fixpoints = [("p25", p25_s2), ("p50", median_s2), ("p75", p75_s2)]

# Store results: dict mapping (s1_label, s2_label) -> (shap_s1, shap_s2, shap_s3)
sensitivity_results = {}

print("Running sensitivity sweep (reusing cached simulations where possible):\n")

for s1_label, s1_roll in s1_fixpoints:
    for s2_label, s2_roll in s2_fixpoints:
        v000 = get_or_run(None,    None,    False)
        v100 = get_or_run(s1_roll, None,    False)
        v010 = get_or_run(None,    s2_roll, False)
        v001 = get_or_run(None,    None,    True)
        v110 = get_or_run(s1_roll, s2_roll, False)
        v101 = get_or_run(s1_roll, None,    True)
        v011 = get_or_run(None,    s2_roll, True)
        v111 = get_or_run(s1_roll, s2_roll, True)

        sh1, sh2, sh3 = compute_shapley(v000, v100, v010, v001, v110, v101, v011, v111)
        sensitivity_results[(s1_label, s2_label)] = (sh1, sh2, sh3, v000 - v111)

print()

# --- Print sensitivity table at key checkpoints ---
print(f"{'=' * 70}")
print("SENSITIVITY: Shapley % at Checkpoints by Fix-Point Choice")
print("=" * 70)

for cp in CHECKPOINTS:
    idx = cp - 1
    print(f"\n  Roll {cp}:")
    print(f"  {'S1 fix':>8} {'S2 fix':>8} {'%S1':>8} {'%S2':>8} {'%S3':>8} {'V_expl':>10}")
    print(f"  {'-'*54}")

    for s1_label, s1_roll in s1_fixpoints:
        for s2_label, s2_roll in s2_fixpoints:
            sh1, sh2, sh3, ve = sensitivity_results[(s1_label, s2_label)]
            if ve[idx] > 0:
                p1 = sh1[idx] / ve[idx] * 100
                p2 = sh2[idx] / ve[idx] * 100
                p3 = sh3[idx] / ve[idx] * 100
            else:
                p1 = p2 = p3 = 0.0
            print(f"  {s1_label+' r'+str(s1_roll):>8} {s2_label+' r'+str(s2_roll):>8}"
                  f" {p1:>7.1f}% {p2:>7.1f}% {p3:>7.1f}% {ve[idx]:>10.0f}")


# ==========================================================================
#  PLOT 5: Sensitivity -- Shapley % across fix-point choices
# ==========================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# For each checkpoint in a reduced set, show grouped bars.
sens_checkpoints = [cp for cp in [50, 250, 1000] if cp <= MAX_ROLLS]
combo_labels = [f"S1={s1l}\nS2={s2l}"
                for s1l, _ in s1_fixpoints for s2l, _ in s2_fixpoints]
x = np.arange(len(combo_labels))
bar_w = 0.25

for ax_i, cp in enumerate(sens_checkpoints):
    idx = cp - 1
    vals_s1, vals_s2, vals_s3 = [], [], []

    for s1_label, _ in s1_fixpoints:
        for s2_label, _ in s2_fixpoints:
            sh1, sh2, sh3, ve = sensitivity_results[(s1_label, s2_label)]
            if ve[idx] > 0:
                vals_s1.append(sh1[idx] / ve[idx] * 100)
                vals_s2.append(sh2[idx] / ve[idx] * 100)
                vals_s3.append(sh3[idx] / ve[idx] * 100)
            else:
                vals_s1.append(0); vals_s2.append(0); vals_s3.append(0)

    axes[ax_i].bar(x - bar_w, vals_s1, bar_w, color="#f59e0b", label="S1 timing")
    axes[ax_i].bar(x,         vals_s2, bar_w, color="#10b981", label="S2 timing")
    axes[ax_i].bar(x + bar_w, vals_s3, bar_w, color="#8b5cf6", label="S3 selection")

    axes[ax_i].set_title(f"Roll {cp}", fontsize=13, fontweight="bold")
    axes[ax_i].set_xticks(x)
    axes[ax_i].set_xticklabels(combo_labels, fontsize=8)
    axes[ax_i].set_ylim(0, 105)
    if ax_i == 0:
        axes[ax_i].set_ylabel("% of Explained Variance", fontsize=12)
    if ax_i == 1:
        axes[ax_i].legend(loc="upper center", fontsize=10)

fig.suptitle("Sensitivity: Shapley Decomposition Across Fix-Point Choices",
             fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_PATH, "variance_sensitivity.png"), dpi=150)
print(f"\nSaved: variance_sensitivity.png")
plt.close()


print("\nDone!")
