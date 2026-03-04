"""
expected_points.py — Monte Carlo Simulator for League of Legends Loot System
=============================================================================

This script estimates how many points a player can expect to accumulate over
many rolls of the Hextech Crafting loot system, using Monte Carlo simulation.

Monte Carlo simulation means: instead of solving the math analytically (which
is intractable here due to 737 one-time items creating 2^737 possible states),
we simulate the process many times with random numbers and observe the average
outcomes. With 100,000 simulations, the law of large numbers guarantees our
estimates converge to the true expected values.

THE LOOT SYSTEM:
  Each roll randomly selects one item from a pool. The pool is divided into
  three subsets, each with a fixed total probability:

    Subset 1 (0.5%):   1 item -> collected once (0 pts), then transforms into
                        a permanent reward worth 270 pts. Pity at 80 rolls.
    Subset 2 (10.0%):  10 items -> collected one by one (0 pts each), then once
                        all 10 are collected, replaced by a permanent reward
                        worth 35 pts. Pity at 10 rolls.
    Subset 3 (89.5%):  5 permanent items (5/10/25/50/100 pts) that can be
                        rolled repeatedly, plus 263 one-time emotes and 474
                        one-time icons (2 pts each) that disappear after being
                        collected. No pity.

  "Pity" means: if you go N rolls without hitting a subset, the next roll is
  forced into that subset. This prevents extreme bad luck streaks.

  "Redistribution" means: when a one-time item is removed, its probability is
  split equally among the remaining items within the same subset. This keeps
  each subset's total probability constant.

KEY OPTIMISATION -- TWO-STAGE SAMPLING:
  Because redistribution only happens within each subset, each subset's total
  probability never changes (0.5% + 10.0% + 89.5% = 100% always). This means
  we can split each roll into two steps:
    1. Pick WHICH subset (using the fixed 0.5/10.0/89.5 split)
    2. Pick WHICH item within that subset

  This avoids building a single 742-item probability distribution every roll.

KEY OPTIMISATION -- TIER GROUPING:
  Within subset 3, all 263 emotes share the same probability, and all 474
  icons share the same probability. So instead of tracking 737 individual
  items, we track just 7 categories: 5 permanents + "all emotes" + "all icons".
  The emote group's probability = (number remaining) x (per-emote probability).

KEY OPTIMISATION -- GPU ACCELERATION (CuPy):
  All large simulation arrays (300k elements) live on the GPU. Operations like
  random number generation, boolean masking, CDF sampling, and redistribution
  math run as GPU kernels, processing all simulations in parallel across
  thousands of GPU cores instead of ~8 CPU cores. The per-roll loop still runs
  on the CPU (it's just a counter), but each iteration dispatches GPU work that
  processes all 300k simulations simultaneously. If CuPy is not installed or no
  GPU is available, the script falls back to NumPy automatically.

DATA FORMAT (items_all.csv):
  id, subset, type, percentage_probability, points

  subset: 1, 2, or 3
  type:   "one_time" (collected once then removed) or "permanent" (stays forever)
  percentage_probability: may be a decimal (e.g. "51.37") or a fraction
                          (e.g. "14.053/263") to preserve exact arithmetic.

USAGE:
  1. Place items_all.csv in the same directory (or update DATA_PATH below).
  2. Adjust max_rolls, simulations, and DIST_CHECKPOINTS as needed.
  3. Run: python expected_points.py

DEPENDENCIES:
  - numpy  (pip install numpy)   -- fast array math (CPU fallback)
  - scipy  (pip install scipy)   -- statistical functions for distribution analysis
  - cupy   (pip install cupy-cuda12x) -- GPU-accelerated array math (optional)
    ^ replace "12x" with your CUDA version, e.g. cupy-cuda11x for CUDA 11
"""

# --- IMPORTS ------------------------------------------------------------------

# "csv" provides tools for reading and writing CSV (comma-separated values)
# files. We use csv.DictReader to read each row as a dictionary keyed by the
# column header names, which is more readable than indexing by column number.
import csv
import os

# "time" lets us measure wall-clock execution time to compare CPU vs GPU.
import time

# "numpy" (imported as np by convention) provides fast array operations on CPU.
# Even with GPU acceleration, we still need numpy for: output arrays (small,
# stored on CPU), interfacing with scipy (which only accepts numpy arrays),
# and as a fallback when CuPy is not available.
#
# Key numpy concepts used in this file:
#   np.array([1, 2, 3])       -- creates a 1D array
#   np.zeros(100000)           -- array of 100,000 zeros
#   array + 5                  -- adds 5 to every element (vectorised)
#   array[bool_mask]           -- selects elements where mask is True
#   np.where(cond, a, b)       -- element-wise: a if True, b if False
#   np.searchsorted(cdf, r)    -- inverse CDF sampling (explained below)
import numpy as np

# "scipy.stats" provides statistical tests and measures. We use it for:
#   skew()      -- measures distribution asymmetry
#   kurtosis()  -- measures tail heaviness relative to normal
#   shapiro()   -- formal test of whether data is normally distributed
# scipy only works with numpy arrays, so GPU data must be copied back to CPU
# before passing to these functions.
from scipy import stats as sp_stats

# --- GPU DETECTION ------------------------------------------------------------
#
# CuPy is a drop-in replacement for numpy that runs on NVIDIA GPUs. Most numpy
# functions have identical CuPy equivalents (same name, same arguments), but
# they execute on the GPU using CUDA instead of the CPU. This means we can
# switch between CPU and GPU by changing which library we use to create and
# operate on arrays.
#
# We use the variable "xp" (short for "array library") throughout the simulate
# function: xp = cupy when GPU is available, xp = numpy otherwise. This is a
# common convention in code that supports both backends.
#
# The try/except pattern: if "import cupy" succeeds, we have GPU support. If
# it raises ImportError (cupy not installed) or any other error (e.g. no GPU
# driver), we fall back to numpy.
try:
    import cupy as cp

    # Verify that CuPy can actually access a GPU. cp.cuda.runtime.getDeviceCount()
    # returns 0 if there are no visible GPUs, which would cause errors later.
    if cp.cuda.runtime.getDeviceCount() == 0:
        raise RuntimeError("CuPy installed but no GPU detected")

    GPU_AVAILABLE = True

    # Print which GPU we're using so the user can verify.
    device = cp.cuda.Device(0)
    mem_gb = device.mem_info[1] / (1024 ** 3)  # total VRAM in GB
    print(f"GPU acceleration: ENABLED")
    print(f"  Device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"  VRAM:   {mem_gb:.1f} GB\n")

except Exception:
    GPU_AVAILABLE = False
    print("GPU acceleration: DISABLED (CuPy not found or no GPU detected)")
    print("  Falling back to NumPy (CPU). To enable GPU, install CuPy:")
    print("  pip install cupy-cuda12x  (replace 12x with your CUDA version)\n")


# --- HELPER: GPU <-> CPU TRANSFER ---------------------------------------------
#
# When arrays live on the GPU, we sometimes need to bring values back to the
# CPU -- for example, to store a single scalar in a CPU output array, or to
# pass data to scipy (which only understands numpy arrays). These helper
# functions handle the transfer cleanly.

def to_cpu(x):
    """
    Converts a CuPy array to a numpy array (GPU -> CPU copy). If x is already
    a numpy array or a plain python number, returns it unchanged.

    This is needed because:
      - scipy functions only accept numpy arrays
      - output arrays stored on CPU need numpy values
      - .tolist() only works on numpy arrays
    """
    if GPU_AVAILABLE and isinstance(x, cp.ndarray):
        # .get() copies the data from GPU memory (VRAM) to CPU memory (RAM)
        # and returns a numpy array. This triggers a GPU synchronisation:
        # python waits until all pending GPU operations finish before copying.
        return x.get()
    return x


def scalar(x):
    """
    Extracts a single number from a 0-dimensional array (GPU or CPU).

    Operations like array.mean() return a 0-d array (an array with no
    dimensions that holds a single value). We need to convert these to plain
    python floats before storing them in CPU output arrays. .item() does this
    extraction. For CuPy arrays, it also triggers a GPU -> CPU transfer.
    """
    if hasattr(x, 'item'):
        return x.item()
    return float(x)


# --- FUNCTIONS ----------------------------------------------------------------

def parse_data(filepath):
    """
    Reads the items_all.csv file and organises the data into a dictionary
    grouped by subset number and item type.

    The percentage_probability column may contain either:
      - A plain decimal like "51.37" (meaning 51.37%)
      - A fraction like "14.053/263" (meaning 14.053 / 263 %)

    Fractions are used for the emote and icon tiers because 14.053/263
    produces a repeating decimal that cannot be represented exactly. Storing
    the fraction preserves the exact value.

    All probabilities are converted from percentages to proper probabilities
    (i.e. divided by 100) so that 51.37% becomes 0.5137.

    Parameters:
        filepath: path to the CSV file (e.g. "items_all.csv")

    Returns:
        A dictionary with six keys:
            "s1_one_time":  [(id, prob, pts), ...]
            "s1_permanent": [(id, prob, pts), ...]
            "s2_one_time":  [(id, prob, pts), ...]
            "s2_permanent": [(id, prob, pts), ...]
            "s3_one_time":  [(id, prob, pts), ...]
            "s3_permanent": [(id, prob, pts), ...]
        Each value is a list of tuples: (item_name, probability, points).
    """
    # Initialise the dictionary with empty lists for each category.
    config = {
        "s1_one_time": [], "s1_permanent": [],
        "s2_one_time": [], "s2_permanent": [],
        "s3_one_time": [], "s3_permanent": [],
    }

    # Open the CSV file for reading. encoding="utf-8" handles special
    # characters like apostrophes in item names (e.g. "Got 'em").
    with open(filepath, 'r', encoding='utf-8') as f:
        # csv.DictReader reads each row as a dictionary where the keys are
        # the column headers from the first row of the file. This lets us
        # write row["id"] instead of row[0], which is more readable.
        reader = csv.DictReader(f)
        for row in reader:
            item_id = row["id"].strip()
            subset = int(row["subset"])
            item_type = row["type"].strip()
            prob_str = row["percentage_probability"].strip()

            # Handle fraction probabilities (e.g. "14.053/263").
            # If the string contains a "/" character, split it into numerator
            # and denominator, evaluate the division, then convert from
            # percentage to probability by dividing by 100.
            if "/" in prob_str:
                num, den = prob_str.split("/")
                prob = float(num) / float(den) / 100.0
            else:
                # Plain decimal (e.g. "51.37"). Just convert to float and
                # divide by 100 to go from percentage to probability.
                prob = float(prob_str) / 100.0

            pts = float(row["points"])

            # Build the dictionary key from subset number and type.
            # e.g. subset=2, type="one_time" -> key = "s2_one_time"
            # f-strings (the f"..." syntax) let us embed variables directly.
            key = f"s{subset}_{item_type}"
            config[key].append((item_id, prob, pts))

    return config


def simulate(config, max_rolls, simulations=100_000, seed=42, dist_checkpoints=None,
             fix_s1_at_roll=None, fix_s2_at_roll=None, fix_s3=False):
    """
    Runs the Monte Carlo simulation: max_rolls rolls x simulations parallel
    simulations. Returns a dictionary of results including cumulative means,
    percentiles, and distribution snapshots.

    When CuPy is available, all large arrays (one element per simulation) are
    created on the GPU. The per-roll loop runs on the CPU but dispatches GPU
    kernels for all the heavy math. Small output arrays (one element per roll)
    stay on the CPU since they're only max_rolls elements.

    Parameters:
        config:           dictionary from parse_data()
        max_rolls:        number of rolls to simulate (e.g. 3000)
        simulations:      number of parallel simulations (default 100,000).
                          more simulations = more precise estimates but slower.
        seed:             random number generator seed for reproducibility.
                          using the same seed produces identical results every run.
        dist_checkpoints: list of roll numbers at which to save full distribution
                          snapshots for later analysis (e.g. [10, 100, 1000]).
        fix_s1_at_roll:   (variance decomposition) if set, forces ALL simulations
                          to transform S1 at exactly this roll number. Before this
                          roll, no sim can transform; at this roll, all transform
                          simultaneously. Removes S1-timing variance.
        fix_s2_at_roll:   (variance decomposition) if set, forces ALL simulations
                          to complete S2 at exactly this roll number. Same logic
                          as fix_s1_at_roll but for S2. Removes S2-timing variance.
        fix_s3:           (variance decomposition) if True, all simulations receive
                          the same S3 category outcome each roll. A single random
                          draw is broadcast to every sim, removing S3-selection
                          variance while preserving redistribution differences.

    Returns:
        dictionary with keys:
            roll_numbers, cumulative_mean, cumulative_std, marginal_mean,
            s1_transform_prob, s2_complete_prob, s3_ot_remaining_mean,
            n_s3_ot_total, percentiles, percentile_values, dist_snapshots
    """
    # --- Select the array backend (GPU or CPU) --------------------------------
    #
    # "xp" is our array library: cupy for GPU, numpy for CPU. Every array
    # creation and operation in the simulation loop uses xp instead of np,
    # so switching between GPU and CPU is automatic.
    if GPU_AVAILABLE:
        xp = cp
        # CuPy's random seeding. This seeds the default random generator on
        # the current GPU device. The GPU uses a different RNG algorithm than
        # numpy's PCG64 (typically XORWOW or Philox), so results will differ
        # from the CPU version even with the same seed -- but the statistical
        # properties are identical.
        xp.random.seed(seed)
        print(f"  Backend: CuPy (GPU)")
    else:
        xp = np
        print(f"  Backend: NumPy (CPU)")

    # For numpy, we use the modern Generator API with a fixed seed for
    # reproducibility. For CuPy, we use xp.random module functions directly
    # (seeded above). To unify the interface, we define a helper:
    if not GPU_AVAILABLE:
        _rng = np.random.default_rng(seed)
        def rng_random(size):
            """Generate uniform random numbers in [0, 1) on CPU."""
            return _rng.random(size)
    else:
        def rng_random(size):
            """Generate uniform random numbers in [0, 1) on GPU."""
            return xp.random.random(size, dtype=xp.float64)

    # Shorthand -- we reference this number constantly throughout the function.
    S = simulations

    # Convert dist_checkpoints to a set for O(1) membership testing.
    # Checking "if x in some_set" is instant regardless of size, whereas
    # "if x in some_list" gets slower as the list grows.
    if dist_checkpoints is None:
        dist_checkpoints = []
    dist_checkpoint_set = set(dist_checkpoints)

    # -- Extract and organise data from config ---------------------------------
    #
    # This section runs on the CPU (plain python lists and floats). The data
    # is small (< 1000 items) so there's no benefit to GPU here. We'll move
    # the necessary arrays to GPU when creating state arrays below.

    # --- Subset 1 ---
    s1_ot = config["s1_one_time"]    # list of one-time items (just 1: Viego)
    s1_pm = config["s1_permanent"]   # list of permanent items (the 270-pt reward)

    # Total probability of hitting subset 1 on any roll.
    # "sum(p for _, p, _ in s1_ot)" is a generator expression: it iterates
    # through each (name, prob, pts) tuple, ignores name and pts (the
    # underscores _ mean "I don't need this value"), and sums the probs.
    # For subset 1, this is just 0.005 (0.5%).
    s1_total_p = sum(p for _, p, _ in s1_ot)

    # After 80 consecutive rolls without hitting subset 1, the next roll is
    # forced into subset 1 (the "pity" mechanic).
    s1_pity_limit = 79  # counter starts at 0, so >= 79 forces on the 80th roll

    # Points awarded when hitting the permanent (transformed) version of S1.
    # s1_pm[0] is the first (and only) permanent item tuple: (name, prob, pts).
    # [2] gets the third element (pts), which is 270.
    s1_tf_pts = s1_pm[0][2]

    # --- Subset 2 ---
    s2_ot = config["s2_one_time"]    # 10 one-time items
    s2_pm = config["s2_permanent"]   # the 35-pt completion reward

    # Convert S2 probabilities to an array for vectorised math later.
    # Using xp creates the array on GPU (if available) or CPU.
    # Result: [0.014970..., 0.013770..., 0.013770..., ...] -- 10 values.
    s2_base_p = xp.array([p for _, p, _ in s2_ot])
    n_s2 = len(s2_ot)                          # number of S2 items = 10
    s2_total_p = float(s2_base_p.sum())        # total S2 probability (~0.10)
    s2_pity_limit = 9                          # counter starts at 0, so >= 9 forces on the 10th roll
    s2_comp_pts = s2_pm[0][2]                  # completion reward = 35 pts

    # --- Subset 3 ---
    s3_ot = config["s3_one_time"]    # 737 one-time items (263 emotes + 474 icons)
    s3_pm = config["s3_permanent"]   # 5 permanent items (5/10/25/50/100 pts)

    # Arrays for the 5 permanent items' probabilities and point values.
    # s3_perm_p  = [0.5137, 0.09129, 0.006265, 0.00179, 0.000895]
    # s3_perm_pts = [5.0, 10.0, 25.0, 50.0, 100.0]
    s3_perm_p = xp.array([p for _, p, _ in s3_pm])
    s3_perm_pts = xp.array([pt for _, _, pt in s3_pm])
    n_s3_perm = len(s3_pm)  # = 5

    # --- Tier grouping for subset 3 one-time items ---
    #
    # All 263 emotes have the same base probability, and all 474 icons have
    # the same (different) base probability. Rather than tracking 737 items
    # individually, we discover the two tiers automatically:
    #
    # 1. Take the first one-time item's probability as "tier A" probability.
    # 2. Items within 1e-6 of tier A -> tier A (emotes). Everything else -> tier B (icons).
    #
    # We use abs(x - target) < 1e-6 instead of == because floating-point
    # equality is unreliable. Two numbers that should be equal might differ
    # in their 15th decimal place due to how computers represent fractions.
    tier_a_prob = s3_ot[0][1]
    tier_a_pts = s3_ot[0][2]
    tier_a_items = [x for x in s3_ot if abs(x[1] - tier_a_prob) < 1e-6]
    tier_b_items = [x for x in s3_ot if abs(x[1] - tier_a_prob) >= 1e-6]

    # Defensive: "if tier_b_items else 0.0" handles the edge case where all
    # one-time items happen to be the same tier (no tier B exists).
    tier_b_prob = tier_b_items[0][1] if tier_b_items else 0.0
    tier_b_pts = tier_b_items[0][2] if tier_b_items else 0
    n_tier_a_total = len(tier_a_items)   # 263 emotes
    n_tier_b_total = len(tier_b_items)   # 474 icons
    n_s3_ot_total = n_tier_a_total + n_tier_b_total  # 737 total one-time

    # Total subset 3 probability = sum of permanents + (count x per-item prob)
    # for each tier. We multiply count by per-item prob (instead of summing
    # individually) because all items within a tier have the same probability.
    s3_total_p = (float(s3_perm_p.sum())
                  + n_tier_a_total * tier_a_prob
                  + n_tier_b_total * tier_b_prob)

    # --- Build the subset selection CDF ---
    #
    # subset_probs = [0.005, 0.10, 0.895] (the three subset totals).
    # We normalise so they sum to exactly 1.0 (correcting for any tiny
    # floating-point imprecision in the source data).
    subset_probs = xp.array([s1_total_p, s2_total_p, s3_total_p])
    subset_probs /= subset_probs.sum()

    # xp.cumsum() builds the cumulative distribution function (CDF):
    # [0.005, 0.105, 1.0]. This is used for "inverse CDF sampling" -- given a
    # uniform random number between 0 and 1, we find which bin it falls into
    # by finding the first CDF value that exceeds it.
    #
    # Example: if random number = 0.03, it's between 0.005 and 0.105 -> subset 2.
    #          if random number = 0.50, it's between 0.105 and 1.0 -> subset 3.
    subset_cdf = xp.cumsum(subset_probs)

    # Print a summary of what was loaded so the user can verify correctness.
    print(f"  Subset 1: {len(s1_ot)} one-time + {len(s1_pm)} permanent,"
          f"  total p = {s1_total_p*100:.4f}%")
    print(f"  Subset 2: {n_s2} one-time + {len(s2_pm)} permanent,"
          f"  total p = {s2_total_p*100:.4f}%")
    print(f"  Subset 3: {n_s3_perm} permanent + {n_tier_a_total} emotes"
          f" + {n_tier_b_total} icons"
          f" = {n_s3_perm + n_s3_ot_total} items,"
          f"  total p = {s3_total_p*100:.4f}%")

    # -- State arrays ----------------------------------------------------------
    #
    # These arrays track the state of each simulation. Every array has one
    # element per simulation (300,000 elements). They live on the GPU when
    # CuPy is available, so all operations on them execute as GPU kernels
    # that process all simulations in parallel.

    # Has this simulation's Viego item transformed into the permanent 270-pt
    # version? Starts all False (no simulation has hit it yet).
    s1_transformed = xp.zeros(S, dtype=bool)

    # How many consecutive rolls has this simulation gone without hitting ANY
    # subset 1 outcome? When this reaches 79, pity forces a subset 1 hit.
    s1_pity = xp.zeros(S, dtype=xp.int32)

    # Which of the 10 subset 2 items has this simulation collected? A 2D
    # boolean array of shape (300000, 10). s2_achieved[i, j] = True means
    # simulation i has collected item j. Starts all False.
    s2_achieved = xp.zeros((S, n_s2), dtype=bool)

    # Has this simulation collected all 10 S2 items and unlocked the permanent
    # 35-pt reward? Starts all False.
    s2_completed = xp.zeros(S, dtype=bool)

    # Same as s1_pity but for subset 2. Pity triggers at 10 rolls (counter >= 9).
    s2_pity = xp.zeros(S, dtype=xp.int32)

    # How many emotes / icons does this simulation still have uncollected?
    # xp.full(S, 263) creates an array of 300,000 values all set to 263.
    # We only need counts (not which specific items) because all emotes are
    # identical in probability and points, and likewise for icons.
    s3_tier_a_remaining = xp.full(S, n_tier_a_total, dtype=xp.int32)
    s3_tier_b_remaining = xp.full(S, n_tier_b_total, dtype=xp.int32)

    # Running total of points earned so far for each simulation.
    # dtype=float64 gives 64-bit floating-point precision (~15 decimal digits).
    cumulative_pts = xp.zeros(S, dtype=xp.float64)

    # -- Output arrays ---------------------------------------------------------
    #
    # These arrays store one value per roll (max_rolls elements). They're
    # small and stay on the CPU (numpy) regardless of backend, since we
    # store one scalar per roll and they're used for printing at the end.

    # Which percentiles to compute every roll. Distribution-free -- no
    # assumptions about the shape of the distribution.
    PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    n_pct = len(PERCENTILES)

    # np.empty() allocates memory without initialising it (slightly faster
    # than np.zeros since we'll overwrite every value anyway).
    # These always use numpy (CPU) since they store one float per roll.
    cum_means = np.empty(max_rolls)          # mean cumulative points per roll
    cum_stds = np.empty(max_rolls)           # std dev of cumulative points
    marginal_means = np.empty(max_rolls)     # expected points earned THIS roll
    s1_tf_arr = np.empty(max_rolls)          # fraction of sims with S1 transformed
    s2_comp_arr = np.empty(max_rolls)        # fraction of sims with S2 completed
    s3_ot_remaining_mean = np.empty(max_rolls)  # mean one-time items remaining

    # pct_arr is 2D: one row per roll, 9 columns (one per percentile).
    pct_arr = np.empty((max_rolls, n_pct))

    # Used to compute marginal expected value:
    #   marginal[roll] = cumulative_mean[roll] - cumulative_mean[roll - 1]
    prev_mean = 0.0

    # Dictionary mapping roll numbers -> numpy arrays (on CPU) of all
    # simulation cumulative totals, saved at checkpoint rolls for
    # distribution analysis with scipy.
    dist_snapshots = {}

    # Convert PERCENTILES to a GPU array for xp.percentile() if on GPU.
    # This avoids repeatedly transferring the list to GPU every roll.
    percentiles_arr = xp.array(PERCENTILES, dtype=xp.float64)

    # ==========================================================================
    #  MAIN SIMULATION LOOP -- one iteration per roll
    # ==========================================================================
    #
    # Each iteration processes all 300,000 simulations in parallel using
    # vectorised operations. On GPU, each operation launches a CUDA kernel
    # that distributes the work across thousands of GPU cores. The steps are:
    #   1. Check pity counters and force subsets where needed
    #   2. For non-forced simulations, randomly pick a subset
    #   3. Within each subset, resolve what happens (points, state changes)
    #   4. Update accumulators (points, pity counters)
    #   5. Record statistics for this roll

    # ==========================================================================
    #  GPU PERFORMANCE NOTE -- BRANCHLESS DESIGN
    # ==========================================================================
    #
    # On a GPU, every .any() call forces a GPU -> CPU synchronisation: the GPU
    # must finish ALL pending work, transfer one boolean to the CPU, and wait
    # for python to decide which branch to take. With ~12 .any() checks per
    # roll x 200 rolls = 2,400 pipeline stalls, this overhead dominates.
    #
    # The fix: remove ALL .any() guards. Operations on empty index arrays
    # (e.g. pts_earned[empty_array] = 270) are harmless no-ops on both GPU
    # and CPU, so skipping the check has zero effect on correctness. The GPU
    # can now queue the entire roll's work without waiting for python.
    #
    # The subset selection is also rewritten to avoid int(normal.sum()) which
    # is another sync point. Instead, we generate random numbers for ALL S
    # simulations and overwrite the pity-forced ones afterward.
    #
    # The only remaining GPU -> CPU syncs are in Step 5 (recording stats),
    # which are unavoidable since we need to store the results. These are
    # batched into a single to_cpu() call using xp.stack() to minimise the
    # number of sync points per roll.

    for roll in range(max_rolls):

        # -- Step 1: Pity check (branchless) -----------------------------------
        #
        # Build forced_subset using xp.where instead of conditional boolean
        # indexing. xp.where is a single GPU kernel with no sync.
        #   -1 = not forced, 0 = forced into S1, 1 = forced into S2.
        forced_subset = xp.where(s1_pity >= s1_pity_limit,
                                 xp.int8(0), xp.int8(-1))

        # S2 pity: only applies where S1 hasn't already claimed the slot.
        s2_pity_mask = (s2_pity >= s2_pity_limit) & (forced_subset == -1)
        forced_subset = xp.where(s2_pity_mask, xp.int8(1), forced_subset)

        # -- Step 2: Subset selection (branchless) -----------------------------
        #
        # Generate random numbers for ALL S simulations (not just non-forced
        # ones). This wastes ~0.5% of randoms (the pity-forced fraction) but
        # avoids the GPU -> CPU sync that int(normal.sum()) would cause.
        r = rng_random(S)
        chosen_subset = xp.searchsorted(subset_cdf, r).astype(xp.int8)
        chosen_subset = xp.clip(chosen_subset, 0, 2)

        # Overwrite pity-forced simulations. For non-forced sims (forced_subset
        # == -1), keep the random choice. For forced sims, use the pity value.
        forced = forced_subset >= 0
        chosen_subset = xp.where(forced, forced_subset, chosen_subset)

        # Create three boolean masks splitting the 300k simulations by subset.
        is_s1 = chosen_subset == 0
        is_s2 = chosen_subset == 1
        is_s3 = chosen_subset == 2

        # Fresh array: each simulation's points earned THIS roll (starts at 0).
        pts_earned = xp.zeros(S, dtype=xp.float64)

        # -- Step 3a: Resolve subset 1 (no .any() guard) -----------------------
        #
        # xp.where() on an all-False mask returns an empty array. All
        # subsequent fancy indexing with that empty array is a no-op,
        # so this block is safe to run unconditionally.

        s1_sims = xp.where(is_s1)[0]
        tf_mask = s1_transformed[s1_sims]
        pts_earned[s1_sims[tf_mask]] = s1_tf_pts
        s1_transformed[s1_sims[~tf_mask]] = True

        # -- Variance decomposition: fix S1 timing ----------------------------
        # If fix_s1_at_roll is set, override the natural S1 mechanics:
        #   - Before the fixed roll: undo any natural transforms (force False)
        #   - At the fixed roll: force ALL simulations to transform
        #   - After the fixed roll: leave s1_transformed alone (already True)
        # The first S1 hit awards 0 pts (collecting the one-time item), so
        # zeroing out points before the fixed roll is correct.
        if fix_s1_at_roll is not None:
            if (roll + 1) < fix_s1_at_roll:
                # Undo any natural transforms that just happened this roll.
                # The one-time item awards 0 pts, so zero out any points.
                s1_transformed[:] = False
                pts_earned[s1_sims] = 0.0
            elif (roll + 1) == fix_s1_at_roll:
                # Force transform for every simulation simultaneously.
                # This is equivalent to every sim collecting the one-time item
                # now, which awards 0 pts. From next roll onwards, S1 hits
                # award 270 pts. Natural S1 hits this roll already got 0 pts
                # from the code above; we just ensure all sims are transformed.
                s1_transformed[:] = True

        # -- Step 3b: Resolve subset 2 (no .any() guard) -----------------------

        s2_sims = xp.where(is_s2)[0]
        comp_mask = s2_completed[s2_sims]
        pts_earned[s2_sims[comp_mask]] = s2_comp_pts
        not_comp_sims = s2_sims[~comp_mask]

        # Within-subset item selection for incomplete simulations.
        # All operations here are safe with empty not_comp_sims (shape-0
        # arrays produce shape-0 results, and indexing with them is a no-op).
        s2_ach = s2_achieved[not_comp_sims]

        # Redistribution: removed probability split equally among remaining.
        removed_sum = (s2_ach * s2_base_p[None, :]).sum(axis=1)
        n_rem = xp.maximum(n_s2 - s2_ach.sum(axis=1), 1)
        bonus = removed_sum / n_rem

        active = ~s2_ach
        item_p = (s2_base_p[None, :] + bonus[:, None]) * active
        item_p_norm = item_p / xp.maximum(item_p.sum(axis=1, keepdims=True), 1e-15)

        cdf = xp.cumsum(item_p_norm, axis=1)
        r = rng_random(len(not_comp_sims))[:, None]
        chosen_item = xp.clip((cdf <= r).sum(axis=1), 0, n_s2 - 1)

        s2_achieved[not_comp_sims, chosen_item] = True

        # Check for newly completed simulations (no .any() guard).
        newly_done = (s2_achieved[not_comp_sims].all(axis=1)
                      & ~s2_completed[not_comp_sims])
        s2_completed[not_comp_sims[newly_done]] = True

        # -- Variance decomposition: fix S2 timing ----------------------------
        # Same logic as S1 fix: force all sims to complete S2 at a specific
        # roll. Before that roll, no sim completes (even if they naturally
        # collected all 10 items). At the fixed roll, all sims complete.
        # Individual item collection still happens randomly, but items award
        # 0 pts so it doesn't affect point variance. Only the completion
        # timing (which enables the 35-pt permanent) matters for points.
        if fix_s2_at_roll is not None:
            if (roll + 1) < fix_s2_at_roll:
                # Undo any natural completions. Sims that hit S2 this roll
                # while "completed" got 35 pts from the code above -- zero
                # those out since we're preventing completion.
                pts_earned[s2_sims[comp_mask]] = 0.0
                s2_completed[:] = False
            elif (roll + 1) == fix_s2_at_roll:
                # Force completion for every simulation simultaneously.
                s2_completed[:] = True

        # -- Step 3c: Resolve subset 3 (no .any() guard) -----------------------

        s3_sims = xp.where(is_s3)[0]
        n_s3 = len(s3_sims)

        a_rem = s3_tier_a_remaining[s3_sims].astype(xp.float64)
        b_rem = s3_tier_b_remaining[s3_sims].astype(xp.float64)

        n_remaining = a_rem + b_rem + n_s3_perm
        total_removed = ((n_tier_a_total - a_rem) * tier_a_prob
                         + (n_tier_b_total - b_rem) * tier_b_prob)
        bonus = total_removed / n_remaining

        # Category grouping: 7 columns (5 permanents + emotes + icons).
        n_cats = n_s3_perm + 2
        cat_p = xp.zeros((n_s3, n_cats))

        for i in range(n_s3_perm):
            cat_p[:, i] = s3_perm_p[i] + bonus
        cat_p[:, n_s3_perm] = a_rem * (tier_a_prob + bonus)
        cat_p[:, n_s3_perm + 1] = b_rem * (tier_b_prob + bonus)

        cat_p_norm = cat_p / xp.maximum(cat_p.sum(axis=1, keepdims=True), 1e-15)

        cdf = xp.cumsum(cat_p_norm, axis=1)

        # -- Variance decomposition: fix S3 outcomes --------------------------
        # If fix_s3 is True, use a single random number broadcast to all
        # simulations instead of independent draws. This means every sim
        # gets the same "luck" for S3 category selection each roll, removing
        # the variance from which category is chosen. Each sim may still
        # map the same r to different categories (because their CDFs differ
        # due to different redistribution states), but this second-order
        # effect is small and diminishes as the shared r drives sims toward
        # similar collection states over time.
        if fix_s3:
            r_single = rng_random(1)
            r = xp.broadcast_to(r_single, (n_s3, 1))
        else:
            r = rng_random(n_s3)[:, None]
        chosen_cat = xp.clip((cdf <= r).sum(axis=1), 0, n_cats - 1)

        for i in range(n_s3_perm):
            pts_earned[s3_sims[chosen_cat == i]] = s3_perm_pts[i]

        # Emote hits: award points and decrement remaining (no .any() guard).
        tier_a_hit = chosen_cat == n_s3_perm
        hit_a = s3_sims[tier_a_hit]
        pts_earned[hit_a] = tier_a_pts
        s3_tier_a_remaining[hit_a] = xp.maximum(s3_tier_a_remaining[hit_a] - 1, 0)

        # Icon hits: same logic (no .any() guard).
        tier_b_hit = chosen_cat == n_s3_perm + 1
        hit_b = s3_sims[tier_b_hit]
        pts_earned[hit_b] = tier_b_pts
        s3_tier_b_remaining[hit_b] = xp.maximum(s3_tier_b_remaining[hit_b] - 1, 0)

        # -- Step 4: Update accumulators ---------------------------------------

        cumulative_pts += pts_earned
        s1_pity = xp.where(is_s1, 0, s1_pity + 1)
        s2_pity = xp.where(is_s2, 0, s2_pity + 1)

        # -- Step 5: Record statistics -----------------------------------------
        #
        # Each scalar() and to_cpu() call triggers a GPU -> CPU sync, but
        # these are unavoidable — we need the numbers on the CPU.

        cum_means[roll] = scalar(cumulative_pts.mean())
        cum_stds[roll] = scalar(cumulative_pts.std(ddof=1))

        marginal_means[roll] = cum_means[roll] - prev_mean
        prev_mean = cum_means[roll]

        s1_tf_arr[roll] = scalar(s1_transformed.mean())
        s2_comp_arr[roll] = scalar(s2_completed.mean())

        ot_rem = s3_tier_a_remaining + s3_tier_b_remaining
        s3_ot_remaining_mean[roll] = scalar(ot_rem.astype(xp.float64).mean())

        pct_arr[roll] = to_cpu(xp.percentile(cumulative_pts, percentiles_arr))

        if (roll + 1) in dist_checkpoint_set:
            dist_snapshots[roll + 1] = to_cpu(cumulative_pts.copy())

        if (roll + 1) % 500 == 0:
            print(f"  Roll {roll+1}/{max_rolls}...")

    # -- Return results --------------------------------------------------------
    #
    # .tolist() converts numpy arrays to plain python lists for cleaner
    # serialisation and compatibility with non-numpy code. All output arrays
    # are already on CPU (numpy), so no GPU transfer needed here.
    return {
        "roll_numbers": list(range(1, max_rolls + 1)),
        "cumulative_mean": cum_means.tolist(),
        "cumulative_std": cum_stds.tolist(),
        "marginal_mean": marginal_means.tolist(),
        "s1_transform_prob": s1_tf_arr.tolist(),
        "s2_complete_prob": s2_comp_arr.tolist(),
        "s3_ot_remaining_mean": s3_ot_remaining_mean.tolist(),
        "n_s3_ot_total": n_s3_ot_total,
        "percentiles": PERCENTILES,
        "percentile_values": pct_arr.tolist(),
        "dist_snapshots": dist_snapshots,
    }


# --- OUTPUT FORMATTING --------------------------------------------------------

def print_summary(results, every_n=100):
    """
    Pretty-prints three output tables from the simulation results:

    1. MAIN TABLE: cumulative mean, marginal E[pts/roll], S1/S2 completion
       probabilities, and S3 one-time items remaining.

    2. PERCENTILE TABLE: p1 through p99 at each checkpoint. These are
       distribution-free (no normality assumption).

    3. QUICK REFERENCE: "unlucky / typical / lucky" ranges using p5, p25-p75,
       and p95 -- the practical summary for a player.

    Parameters:
        results:  dictionary returned by simulate()
        every_n:  print every Nth roll (default 100). The first 10 rolls and
                  the last roll are always printed regardless.
    """
    # Unpack the results dictionary into named variables for readability.
    rolls = results["roll_numbers"]
    cum = results["cumulative_mean"]
    marg = results["marginal_mean"]
    s1p = results["s1_transform_prob"]
    s2p = results["s2_complete_prob"]
    s3r = results["s3_ot_remaining_mean"]
    n_ot = results["n_s3_ot_total"]
    pct_labels = results["percentiles"]
    pct_vals = results["percentile_values"]

    # -- Table 1: Main summary --
    #
    # f-string formatting: {value:<7} means left-aligned in a field 7 chars wide.
    # The < is the alignment specifier (< = left, > = right, ^ = center).
    header = (f"{'Roll':<7}{'E[Cumul]':<13}{'E[/Roll]':<11}"
              f"{'P(S1)':<8}{'P(S2)':<8}{'S3 OT Left':<14}")
    print(f"\n{header}")
    print("-" * len(header))

    # enumerate() gives both the index (i) and the value (r) at each position.
    # We use the index to check if we're in the first 10 rows.
    for i, r in enumerate(rolls):
        # Print the first 10 rolls (for early detail), every Nth roll, and
        # always the last roll.
        if i < 10 or r % every_n == 0 or r == rolls[-1]:
            pct = s3r[i] / n_ot * 100  # remaining as a percentage of original
            # :<13.2f means left-aligned, 13 chars wide, 2 decimal places.
            print(f"{r:<7}{cum[i]:<13.2f}{marg[i]:<11.4f}"
                  f"{s1p[i]:<8.3f}{s2p[i]:<8.3f}{s3r[i]:<5.0f} ({pct:.1f}%)")

    # -- Table 2: Percentile table --
    print(f"\n\nPERCENTILE TABLE (cumulative points)")
    pct_header = f"{'Roll':<7}"
    for p in pct_labels:
        pct_header += f"{'p' + str(p):<9}"
    print(pct_header)
    print("-" * len(pct_header))
    for i, r in enumerate(rolls):
        if i < 10 or r % every_n == 0 or r == rolls[-1]:
            row = f"{r:<7}"
            for j in range(len(pct_labels)):
                row += f"{pct_vals[i][j]:<9.0f}"
            print(row)

    # -- Table 3: Quick reference --
    #
    # The index mapping into pct_vals columns:
    #   index 0 = p1, 1 = p5, 2 = p10, 3 = p25, 4 = p50,
    #   5 = p75, 6 = p90, 7 = p95, 8 = p99
    print(f"\n\nQUICK REFERENCE: Expected ranges at key rolls")
    print(f"{'Roll':<7}{'Unlucky (p5)':<14}{'Typical (p25-p75)':<22}"
          f"{'Lucky (p95)':<14}{'Mean':<12}")
    print("-" * 69)
    for i, r in enumerate(rolls):
        if i < 10 or r % every_n == 0 or r == rolls[-1]:
            p5 = pct_vals[i][1]    # 5th percentile
            p25 = pct_vals[i][3]   # 25th percentile
            p75 = pct_vals[i][5]   # 75th percentile
            p95 = pct_vals[i][7]   # 95th percentile
            print(f"{r:<7}{p5:<14.0f}{p25:.0f} - {p75:<16.0f}"
                  f"{p95:<14.0f}{cum[i]:<12.2f}")


def print_distribution_analysis(results):
    """
    Analyses the shape of the cumulative points distribution at each checkpoint
    roll, comparing it to a normal (bell curve) distribution.

    For each checkpoint, prints:
      - Mean, median, and standard deviation
      - Skewness (asymmetry: 0 = symmetric, positive = right-skewed)
      - Excess kurtosis (tail weight: 0 = normal-like, positive = heavier tails)
      - Shapiro-Wilk normality test
      - Actual 95% range vs normal approximation, and the error between them

    The practical takeaway: when the normal approximation error is small relative
    to the mean, you can use mean +/- 1.96 x std as a quick confidence interval.
    When it's large (especially early rolls), use the actual percentiles instead.

    NOTE: This function always uses numpy/scipy (CPU), since scipy does not
    support GPU arrays. The dist_snapshots are already on CPU (copied during
    the simulation via to_cpu()).
    """
    dist_snapshots = results.get("dist_snapshots", {})
    if not dist_snapshots:
        print("\n  No distribution checkpoints recorded.")
        return

    print(f"\n\nDISTRIBUTION ANALYSIS")
    print("=" * 90)
    print("Skewness:  0 = symmetric, >0 = right-skewed (long right tail)")
    print("Kurtosis:  0 = normal-like tails, >0 = heavier tails than normal")
    print("Shapiro-Wilk: tests normality (p > 0.05 = consistent with normal)")
    print("=" * 90)

    for roll_num in sorted(dist_snapshots.keys()):
        pts = dist_snapshots[roll_num]

        # Basic descriptive statistics across all simulations.
        mean = np.mean(pts)
        median = np.median(pts)    # 50th percentile: half above, half below
        std = np.std(pts, ddof=1)  # sample standard deviation: measure of spread

        # SKEWNESS: the third standardised moment: E[((X - mean) / std)^3].
        # Measures asymmetry. 0 = symmetric, positive = right tail is longer
        # (some players are much luckier than average). In early rolls, the
        # Viego transform creates high skewness: most players haven't hit it
        # (clustered at low points), but the few who have are way out right.
        skew = sp_stats.skew(pts)

        # EXCESS KURTOSIS: the fourth standardised moment minus 3.
        # Measures tail heaviness relative to normal. 0 = normal-like tails,
        # positive = heavier tails (more extreme outcomes than a bell curve
        # would predict). The "minus 3" makes a normal distribution = 0.
        kurt = sp_stats.kurtosis(pts)

        # SHAPIRO-WILK TEST: formal hypothesis test where the null hypothesis
        # is "this data came from a normal distribution". Returns a p-value:
        #   p > 0.05 -> consistent with normal (cannot reject the hypothesis)
        #   p < 0.05 -> not normal (reject the hypothesis)
        #
        # scipy limits this to 5000 samples, so we take a random subsample.
        # replace=False means no duplicates in the sample.
        #
        # CAVEAT: with 100k+ simulations, even trivially small deviations from
        # normality produce p ~ 0. So Shapiro-Wilk will reject at almost every
        # checkpoint, even when the distribution is practically normal. The
        # actual vs normal comparison below is more informative.
        rng_local = np.random.default_rng(0)
        sample = pts[rng_local.choice(len(pts), min(5000, len(pts)), replace=False)]
        _, shapiro_p = sp_stats.shapiro(sample)

        # Actual 95% range from the empirical distribution.
        p2_5 = np.percentile(pts, 2.5)    # 2.5th percentile (lower bound)
        p97_5 = np.percentile(pts, 97.5)  # 97.5th percentile (upper bound)

        # What the 95% range WOULD be if the distribution were perfectly normal.
        # 1.96 comes from the standard normal distribution's 97.5th percentile
        # (since 2.5% is excluded from each tail, leaving 95% in the middle).
        norm_lo = mean - 1.96 * std
        norm_hi = mean + 1.96 * std

        if shapiro_p > 0.05:
            normality = "YES (consistent with normal)"
        else:
            normality = "NO (not normal)"

        print(f"\n--- Roll {roll_num} ---")
        print(f"  Mean: {mean:.1f}   Median: {median:.1f}   Std: {std:.1f}")
        # The "+" in {:+.3f} forces a sign to always be shown (+ or -).
        print(f"  Skewness:        {skew:+.3f}")
        print(f"  Excess Kurtosis: {kurt:+.3f}")
        print(f"  Shapiro-Wilk p:  {shapiro_p:.4f}  ->  Normal? {normality}")
        print(f"  Actual 95% range (p2.5 - p97.5):  [{p2_5:.0f},  {p97_5:.0f}]")
        print(f"  Normal approx 95% (mean +/- 1.96s): [{norm_lo:.0f},  {norm_hi:.0f}]")

        # How far off the normal approximation is from the actual percentiles.
        # When these errors are small relative to the mean, the normal
        # approximation is practically useful. When large, use percentiles.
        lo_err = abs(p2_5 - norm_lo)
        hi_err = abs(p97_5 - norm_hi)
        print(f"  Normal approx error:  lower bound off by {lo_err:.0f} pts,"
              f"  upper bound off by {hi_err:.0f} pts")


# --- MAIN ---------------------------------------------------------------------
#
# When you run a .py file directly (e.g. "python expected_points.py"), python
# sets the special variable __name__ to "__main__". When the file is imported
# by another script (e.g. "from expected_points import simulate" in plots.py),
# __name__ is set to "expected_points" instead. This if-statement means: "only
# run the code below if this file is being executed directly."

if __name__ == "__main__":
    # Derive paths relative to this script's location.
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

    DATA_PATH = os.path.join(_PROJECT_ROOT, "tier data", "items_all_normalised.csv")

    # Roll numbers at which to save full distribution snapshots for analysis.
    # These are the checkpoints shown in the distribution analysis output.
    DIST_CHECKPOINTS = [10, 25, 50, 100, 250, 500, 1000, 2000, 3000]

    # Step 1: Parse the CSV into a structured dictionary.
    config = parse_data(DATA_PATH)

    print("Running simulation...\n")

    # Step 2: Run 300,000 simulations of 1000 rolls each.
    # time.perf_counter() measures wall-clock time with high precision.
    t_start = time.perf_counter()

    results = simulate(
        config,
        max_rolls=1000,         # total number of rolls to simulate
        simulations=300_000,    # number of parallel simulations
        seed=42,                # fixed seed for reproducible results
        dist_checkpoints=DIST_CHECKPOINTS,
    )

    t_elapsed = time.perf_counter() - t_start
    backend = "GPU (CuPy)" if GPU_AVAILABLE else "CPU (NumPy)"
    print(f"\n  Simulation completed in {t_elapsed:.2f}s [{backend}]")

    # Step 3: Print the three summary tables and distribution analysis.
    print_summary(results, every_n=25)
    print_distribution_analysis(results)
