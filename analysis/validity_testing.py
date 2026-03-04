import os
import numpy as np
from scipy import stats
import time

# Import BOTH implementations
# Adjust names if needed
from expected_points_cpu import parse_data, simulate as simulate_cpu
from expected_points_gpu import simulate as simulate_gpu


SIMULATIONS = 200_000
MAX_ROLLS = 200
SEED = 42


def run_cpu(config):
    print("Running CPU simulation...")
    start = time.time()
    results = simulate_cpu(config, MAX_ROLLS, simulations=SIMULATIONS, seed=SEED, dist_checkpoints=[MAX_ROLLS])
    end = time.time()
    print(f"CPU time: {end - start:.2f}s")

    # Final distribution snapshot
    final_points = results["dist_snapshots"][MAX_ROLLS]
    return np.array(final_points)


def run_gpu(config):
    print("\nRunning GPU simulation...")
    start = time.time()
    results = simulate_gpu(config, MAX_ROLLS, simulations=SIMULATIONS, seed=SEED, dist_checkpoints=[MAX_ROLLS])
    end = time.time()
    print(f"GPU time: {end - start:.2f}s")

    final_points = results["dist_snapshots"][MAX_ROLLS]
    return np.array(final_points)


def compare_distributions(cpu, gpu):
    print("\n--- STATISTICAL COMPARISON ---")

    mean_cpu = cpu.mean()
    mean_gpu = gpu.mean()

    std_cpu = cpu.std(ddof=1)
    std_gpu = gpu.std(ddof=1)

    print(f"\nMeans:")
    print(f"  CPU: {mean_cpu:.6f}")
    print(f"  GPU: {mean_gpu:.6f}")
    print(f"  Difference: {mean_cpu - mean_gpu:.6f}")

    # Two-sample t-test
    t_stat, t_p = stats.ttest_ind(cpu, gpu, equal_var=False)
    print(f"\nTwo-sample t-test:")
    print(f"  t = {t_stat:.6f}, p = {t_p:.6f}")

    # F-test for variance
    f_stat = std_cpu**2 / std_gpu**2
    dfn = len(cpu) - 1
    dfd = len(gpu) - 1
    f_p = 2 * min(stats.f.cdf(f_stat, dfn, dfd),
                  1 - stats.f.cdf(f_stat, dfn, dfd))

    print(f"\nVariance comparison (F-test):")
    print(f"  F = {f_stat:.6f}, p = {f_p:.6f}")

    # Kolmogorov-Smirnov test
    ks_stat, ks_p = stats.ks_2samp(cpu, gpu)
    print(f"\nKolmogorov-Smirnov test:")
    print(f"  KS = {ks_stat:.6f}, p = {ks_p:.6f}")

    # Percentile comparison
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("\nPercentiles:")
    for p in percentiles:
        cpu_p = np.percentile(cpu, p)
        gpu_p = np.percentile(gpu, p)
        print(f"  {p:2d}%  CPU={cpu_p:.3f}  GPU={gpu_p:.3f}")

    # Confidence intervals for mean (using t critical value)
    t_crit_cpu = stats.t.ppf(0.975, len(cpu) - 1)
    t_crit_gpu = stats.t.ppf(0.975, len(gpu) - 1)
    ci_cpu = t_crit_cpu * std_cpu / np.sqrt(len(cpu))
    ci_gpu = t_crit_gpu * std_gpu / np.sqrt(len(gpu))

    print("\n95% Confidence Intervals:")
    print(f"  CPU: [{mean_cpu - ci_cpu:.6f}, {mean_cpu + ci_cpu:.6f}]")
    print(f"  GPU: [{mean_gpu - ci_gpu:.6f}, {mean_gpu + ci_gpu:.6f}]")

    # Effect size: is the difference practically meaningful?
    # Cohen's d measures the difference in means relative to the pooled
    # standard deviation. Rules of thumb: <0.2 = negligible, 0.2 = small,
    # 0.5 = medium, 0.8 = large.
    pooled_std = np.sqrt((std_cpu**2 + std_gpu**2) / 2)
    cohens_d = abs(mean_cpu - mean_gpu) / pooled_std
    pct_diff = abs(mean_cpu - mean_gpu) / max(abs(mean_cpu), 1e-15) * 100

    print(f"\nEffect size:")
    print(f"  Cohen's d:          {cohens_d:.6f}")
    print(f"  Mean difference:    {pct_diff:.4f}% of CPU mean")

    print("\n--- INTERPRETATION ---")
    # With large samples (200k), hypothesis tests can flag trivially small
    # differences as "significant". The effect size check guards against this:
    # even if p < 0.05, a tiny Cohen's d means the difference doesn't matter.
    EFFECT_THRESHOLD = 0.01   # Cohen's d below this = negligible
    PCT_THRESHOLD = 1.0       # mean difference below 1% = negligible

    stat_sig = t_p <= 0.05 or ks_p <= 0.05
    practically_sig = cohens_d >= EFFECT_THRESHOLD or pct_diff >= PCT_THRESHOLD

    if not stat_sig and not practically_sig:
        print("No statistically significant difference detected.")
        print("CPU and GPU implementations are statistically equivalent.")
    elif stat_sig and not practically_sig:
        print("Statistically significant but practically negligible difference.")
        print("The tests detected a difference, but it is too small to matter.")
        print("CPU and GPU implementations are effectively equivalent.")
    else:
        print("Meaningful statistical difference detected.")
        print("Investigate potential bias in implementation.")


if __name__ == "__main__":
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
    config = parse_data(os.path.join(_PROJECT_ROOT, "tier data", "items_all_normalised.csv"))

    cpu_dist = run_cpu(config)
    gpu_dist = run_gpu(config)

    compare_distributions(cpu_dist, gpu_dist)
