"""
Visualization for Expected Points Simulation
=============================================
Generates graphs from pre-computed simulation results.

USAGE:
  1. Run expected_points.py first to generate the results files:
       simulation_results.csv   -- summary arrays (one row per roll)
       simulation_snapshots.npz -- raw 100k-value distributions at checkpoints
  2. Run this script to generate all plots from those files.

  No simulation is run here. If the results files don't exist, an error is
  raised with instructions to run expected_points.py first.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats as sp_stats

# Derive paths relative to this script's location.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

# Import only the load function -- no simulation dependency.
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "analysis"))
from expected_points import load_results

# ===================================================================
#  LOAD RESULTS
# ===================================================================

OUTPUT_PATH = _SCRIPT_DIR
os.makedirs(OUTPUT_PATH, exist_ok=True)

RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")

# Check that results exist before proceeding.
csv_path = os.path.join(RESULTS_DIR, "simulation_results.csv")
if not os.path.exists(csv_path):
    print(f"ERROR: No simulation results found at {RESULTS_DIR}")
    print(f"Run expected_points.py first to generate the data.")
    sys.exit(1)

results = load_results(RESULTS_DIR)

rolls = np.array(results["roll_numbers"])
cum_mean = np.array(results["cumulative_mean"])
cum_std = np.array(results["cumulative_std"])
marginal = np.array(results["marginal_mean"])
pct_vals = np.array(results["percentile_values"])
pct_labels = results["percentiles"]
s1_tf = np.array(results["s1_transform_prob"])
s2_comp = np.array(results["s2_complete_prob"])
s3_ot_rem = np.array(results["s3_ot_remaining_mean"])
n_ot = results["n_s3_ot_total"]
snapshots = results["dist_snapshots"]

# Color palette
C_MAIN = "#2563eb"
C_MEDIAN = "#dc2626"
C_BAND1 = "#93c5fd"  # p25-p75
C_BAND2 = "#dbeafe"  # p5-p95
C_BAND3 = "#eff6ff"  # p1-p99
C_S1 = "#f59e0b"
C_S2 = "#10b981"
C_S3 = "#8b5cf6"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})


# ===================================================================
#  PLOT 1: Cumulative Expected Points with Percentile Bands
# ===================================================================

fig, ax = plt.subplots(figsize=(14, 7))

# p1-p99 band
ax.fill_between(rolls, pct_vals[:, 0], pct_vals[:, 8],
                alpha=0.25, color=C_BAND3, label="p1 - p99")
# p5-p95 band
ax.fill_between(rolls, pct_vals[:, 1], pct_vals[:, 7],
                alpha=0.35, color=C_BAND2, label="p5 - p95")
# p25-p75 band
ax.fill_between(rolls, pct_vals[:, 3], pct_vals[:, 5],
                alpha=0.5, color=C_BAND1, label="p25 - p75 (typical)")
# Median
ax.plot(rolls, pct_vals[:, 4], color=C_MEDIAN, linewidth=1.5,
        linestyle="--", label="Median (p50)")
# Mean
ax.plot(rolls, cum_mean, color=C_MAIN, linewidth=2, label="Mean")

ax.set_xlabel("Roll Number", fontsize=12)
ax.set_ylabel("Cumulative Points", fontsize=12)
ax.set_title("Expected Cumulative Points with Percentile Bands", fontsize=14, fontweight="bold")
ax.legend(loc="upper left", fontsize=10)
ax.set_xlim(0, rolls[-1])
ax.set_ylim(0, None)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_PATH,"plot1_cumulative_with_bands.png"), dpi=150)
print("Saved: plot1_cumulative_with_bands.png")
plt.close()


# ===================================================================
#  PLOT 2: Early Rolls Detail (first 150 rolls)
# ===================================================================

fig, ax = plt.subplots(figsize=(14, 7))

mask = rolls <= 150
r = rolls[mask]

ax.fill_between(r, pct_vals[mask, 1], pct_vals[mask, 7],
                alpha=0.35, color=C_BAND2, label="p5 - p95")
ax.fill_between(r, pct_vals[mask, 3], pct_vals[mask, 5],
                alpha=0.5, color=C_BAND1, label="p25 - p75 (typical)")
ax.plot(r, pct_vals[mask, 4], color=C_MEDIAN, linewidth=1.5,
        linestyle="--", label="Median")
ax.plot(r, cum_mean[mask], color=C_MAIN, linewidth=2, label="Mean")

# Mark S1 and S2 completion thresholds
ax.axvline(x=80, color=C_S1, linestyle=":", alpha=0.7, label="S1 pity limit (80)")
ax.axvline(x=100, color=C_S2, linestyle=":", alpha=0.7, label="~S2 completion (~100)")

ax.set_xlabel("Roll Number", fontsize=12)
ax.set_ylabel("Cumulative Points", fontsize=12)
ax.set_title("Early Rolls Detail (1-150): Mean vs Median Divergence", fontsize=14, fontweight="bold")
ax.legend(loc="upper left", fontsize=10)
ax.set_xlim(0, 150)
ax.set_ylim(0, None)

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_PATH,"plot2_early_rolls_detail.png"), dpi=150)
print("Saved: plot2_early_rolls_detail.png")
plt.close()


# ===================================================================
#  PLOT 3: Marginal Expected Points Per Roll
# ===================================================================

fig, ax = plt.subplots(figsize=(14, 6))

# Smooth marginal with rolling average for readability
window = 20
marginal_smooth = np.convolve(marginal, np.ones(window)/window, mode="valid")
r_smooth = rolls[window-1:]

ax.plot(r_smooth, marginal_smooth, color=C_MAIN, linewidth=1.5)

# Mark phase transitions
ax.axvline(x=80, color=C_S1, linestyle="--", alpha=0.7, label="S1 transforms (~roll 80)")
ax.axvline(x=100, color=C_S2, linestyle="--", alpha=0.7, label="S2 completes (~roll 100)")

ax.set_xlabel("Roll Number", fontsize=12)
ax.set_ylabel("E[Points This Roll]", fontsize=12)
ax.set_title("Marginal Expected Points Per Roll (20-roll moving average)", fontsize=14, fontweight="bold")
ax.legend(loc="upper left", fontsize=10)
ax.set_xlim(0, rolls[-1])

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_PATH,"plot3_marginal_per_roll.png"), dpi=150)
print("Saved: plot3_marginal_per_roll.png")
plt.close()


# ===================================================================
#  PLOT 4: Subset Completion & S3 Drain
# ===================================================================

fig, ax1 = plt.subplots(figsize=(14, 6))

ax1.plot(rolls, s1_tf * 100, color=C_S1, linewidth=2, label="P(S1 transformed)")
ax1.plot(rolls, s2_comp * 100, color=C_S2, linewidth=2, label="P(S2 completed)")
ax1.set_xlabel("Roll Number", fontsize=12)
ax1.set_ylabel("Completion Probability (%)", fontsize=12, color="black")
ax1.set_ylim(-5, 105)

ax2 = ax1.twinx()
ax2.plot(rolls, (s3_ot_rem / n_ot) * 100, color=C_S3, linewidth=2,
         linestyle="--", label="S3 one-time remaining (%)")
ax2.set_ylabel("S3 One-Time Items Remaining (%)", fontsize=12, color=C_S3)
ax2.set_ylim(-5, 105)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=10)

ax1.set_title("Subset Milestones & Pool Drain Over Time", fontsize=14, fontweight="bold")
ax1.set_xlim(0, rolls[-1])

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_PATH,"plot4_subset_completion.png"), dpi=150)
print("Saved: plot4_subset_completion.png")
plt.close()


# ===================================================================
#  PLOT 5: Distribution Histograms at Key Checkpoints
# ===================================================================

if not snapshots:
    print("Skipping plots 5-7: no distribution snapshots found.")
    print("  (simulation_snapshots.npz is missing or empty)")
else:
    checkpoint_rolls = sorted(snapshots.keys())
    n_checks = len(checkpoint_rolls)
    cols = 3
    rows = (n_checks + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()

    for idx, roll_num in enumerate(checkpoint_rolls):
        ax = axes[idx]
        pts = snapshots[roll_num]

        mean = np.mean(pts)
        median = np.median(pts)
        std = np.std(pts, ddof=1)
        skew = sp_stats.skew(pts)

        # Histogram
        n_bins = min(80, max(30, int(np.sqrt(len(pts)))))
        ax.hist(pts, bins=n_bins, density=True, color=C_BAND1, edgecolor="white",
                linewidth=0.3, alpha=0.8, label="Actual")

        # Overlay normal curve
        x_range = np.linspace(pts.min(), pts.max(), 300)
        normal_pdf = sp_stats.norm.pdf(x_range, mean, std)
        ax.plot(x_range, normal_pdf, color=C_MEDIAN, linewidth=1.5,
                linestyle="--", label="Normal fit", alpha=0.8)

        # Mark mean and median
        ax.axvline(mean, color=C_MAIN, linewidth=1.5, linestyle="-", alpha=0.8, label=f"Mean: {mean:,.0f}")
        ax.axvline(median, color=C_S1, linewidth=1.5, linestyle="--", alpha=0.8, label=f"Median: {median:,.0f}")

        ax.set_title(f"Roll {roll_num}  (skew={skew:+.2f})", fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlabel("Cumulative Points")
        ax.set_ylabel("Density")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # Hide unused subplots
    for idx in range(n_checks, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Distribution of Cumulative Points at Key Rolls", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_PATH,"plot5_distribution_histograms.png"), dpi=150, bbox_inches="tight")
    print("Saved: plot5_distribution_histograms.png")
    plt.close()


    # ===================================================================
    #  PLOT 6: Q-Q Plots (Normal Probability Plots)
    # ===================================================================

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()

    for idx, roll_num in enumerate(checkpoint_rolls):
        ax = axes[idx]
        pts = snapshots[roll_num]

        # Subsample for cleaner Q-Q plot
        rng = np.random.default_rng(42)
        sample = pts[rng.choice(len(pts), min(2000, len(pts)), replace=False)]
        sample_sorted = np.sort(sample)

        # Theoretical quantiles
        n = len(sample_sorted)
        theoretical = sp_stats.norm.ppf(np.linspace(0.001, 0.999, n))

        # Standardize
        mean = np.mean(sample_sorted)
        std = np.std(sample_sorted, ddof=1)
        standardized = (sample_sorted - mean) / std

        ax.scatter(theoretical, standardized, s=1, alpha=0.4, color=C_MAIN)
        lim = max(abs(theoretical.min()), abs(theoretical.max())) * 1.1
        ax.plot([-lim, lim], [-lim, lim], color=C_MEDIAN, linewidth=1.5, linestyle="--", alpha=0.7)

        skew = sp_stats.skew(pts)
        ax.set_title(f"Roll {roll_num}  (skew={skew:+.2f})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Theoretical (Normal)")
        ax.set_ylabel("Observed (Standardized)")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")

    for idx in range(n_checks, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Q-Q Plots: How Close to Normal?", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_PATH,"plot6_qq_plots.png"), dpi=150, bbox_inches="tight")
    print("Saved: plot6_qq_plots.png")
    plt.close()


    # ===================================================================
    #  PLOT 7: Skewness & Kurtosis Over Checkpoints
    # ===================================================================

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    skews = []
    kurts = []
    for roll_num in checkpoint_rolls:
        pts = snapshots[roll_num]
        skews.append(sp_stats.skew(pts))
        kurts.append(sp_stats.kurtosis(pts))

    ax1.plot(checkpoint_rolls, skews, marker="o", color=C_MAIN, linewidth=2, markersize=6)
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Roll Number", fontsize=12)
    ax1.set_ylabel("Skewness", fontsize=12)
    ax1.set_title("Skewness Over Time", fontsize=13, fontweight="bold")
    ax1.annotate("0 = perfectly symmetric", xy=(0.5, 0.02), xycoords="axes fraction",
                 fontsize=9, color="gray", ha="center")

    ax2.plot(checkpoint_rolls, kurts, marker="s", color=C_S2, linewidth=2, markersize=6)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Roll Number", fontsize=12)
    ax2.set_ylabel("Excess Kurtosis", fontsize=12)
    ax2.set_title("Excess Kurtosis Over Time", fontsize=13, fontweight="bold")
    ax2.annotate("0 = normal-like tails", xy=(0.5, 0.02), xycoords="axes fraction",
                 fontsize=9, color="gray", ha="center")

    fig.suptitle("Convergence Toward Normality", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_PATH,"plot7_skewness_kurtosis.png"), dpi=150)
    print("Saved: plot7_skewness_kurtosis.png")
    plt.close()


print("\nAll plots saved!")
