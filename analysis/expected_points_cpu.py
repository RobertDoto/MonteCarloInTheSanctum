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
  - numpy  (pip install numpy)   -- fast array math
  - scipy  (pip install scipy)   -- statistical functions for distribution analysis
"""

# --- IMPORTS ------------------------------------------------------------------

# "csv" provides tools for reading and writing CSV (comma-separated values)
# files. We use csv.DictReader to read each row as a dictionary keyed by the
# column header names, which is more readable than indexing by column number.
import csv

# "numpy" (imported as np by convention) provides fast array operations. When
# we have 100,000 simulations running in parallel, numpy lets us do math on
# all of them at once in compiled C code, rather than looping through them
# one at a time in python. This is roughly 100x faster.
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
from scipy import stats as sp_stats
import os


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


def simulate(config, max_rolls, simulations=100_000, seed=42, dist_checkpoints=None):
    """
    Runs the Monte Carlo simulation: max_rolls rolls x simulations parallel
    simulations. Returns a dictionary of results including cumulative means,
    percentiles, and distribution snapshots.

    Parameters:
        config:           dictionary from parse_data()
        max_rolls:        number of rolls to simulate (e.g. 3000)
        simulations:      number of parallel simulations (default 100,000).
                          more simulations = more precise estimates but slower.
        seed:             random number generator seed for reproducibility.
                          using the same seed produces identical results every run.
        dist_checkpoints: list of roll numbers at which to save full distribution
                          snapshots for later analysis (e.g. [10, 100, 1000]).

    Returns:
        dictionary with keys:
            roll_numbers, cumulative_mean, cumulative_std, marginal_mean,
            s1_transform_prob, s2_complete_prob, s3_ot_remaining_mean,
            n_s3_ot_total, percentiles, percentile_values, dist_snapshots
    """
    # Create a random number generator with a fixed seed. "default_rng" uses
    # the PCG64 algorithm, which is high-quality and fast. Using a fixed seed
    # means the same "random" sequence is produced every run, making results
    # reproducible. Changing the seed gives slightly different results, but
    # with 100k simulations the differences are negligible (~0.3%).
    rng = np.random.default_rng(seed)

    # Shorthand -- we reference this number constantly throughout the function.
    S = simulations

    # Convert dist_checkpoints to a set for O(1) membership testing.
    # Checking "if x in some_set" is instant regardless of size, whereas
    # "if x in some_list" gets slower as the list grows.
    if dist_checkpoints is None:
        dist_checkpoints = []
    dist_checkpoint_set = set(dist_checkpoints)

    # -- Extract and organise data from config ---------------------------------

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

    # Convert S2 probabilities to a numpy array for vectorised math later.
    # Result: [0.014970..., 0.013770..., 0.013770..., ...] -- 10 values.
    s2_base_p = np.array([p for _, p, _ in s2_ot])
    n_s2 = len(s2_ot)                # number of S2 items = 10
    s2_total_p = s2_base_p.sum()     # total S2 probability (~0.10)
    s2_pity_limit = 9                # counter starts at 0, so >= 9 forces on the 10th roll
    s2_comp_pts = s2_pm[0][2]        # completion reward = 35 pts

    # --- Subset 3 ---
    s3_ot = config["s3_one_time"]    # 737 one-time items (263 emotes + 474 icons)
    s3_pm = config["s3_permanent"]   # 5 permanent items (5/10/25/50/100 pts)

    # Arrays for the 5 permanent items' probabilities and point values.
    # s3_perm_p  = [0.5137, 0.09129, 0.006265, 0.00179, 0.000895]
    # s3_perm_pts = [5.0, 10.0, 25.0, 50.0, 100.0]
    s3_perm_p = np.array([p for _, p, _ in s3_pm])
    s3_perm_pts = np.array([pt for _, _, pt in s3_pm])
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
    s3_total_p = (s3_perm_p.sum()
                  + n_tier_a_total * tier_a_prob
                  + n_tier_b_total * tier_b_prob)

    # --- Build the subset selection CDF ---
    #
    # subset_probs = [0.005, 0.10, 0.895] (the three subset totals).
    # We normalise so they sum to exactly 1.0 (correcting for any tiny
    # floating-point imprecision in the source data).
    subset_probs = np.array([s1_total_p, s2_total_p, s3_total_p])
    subset_probs /= subset_probs.sum()

    # np.cumsum() builds the cumulative distribution function (CDF):
    # [0.005, 0.105, 1.0]. This is used for "inverse CDF sampling" -- given a
    # uniform random number between 0 and 1, we find which bin it falls into
    # by finding the first CDF value that exceeds it.
    #
    # Example: if random number = 0.03, it's between 0.005 and 0.105 -> subset 2.
    #          if random number = 0.50, it's between 0.105 and 1.0 -> subset 3.
    subset_cdf = np.cumsum(subset_probs)

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
    # element per simulation (100,000 elements). numpy operates on all of
    # them simultaneously -- no python for-loop over simulations.

    # Has this simulation's Viego item transformed into the permanent 270-pt
    # version? Starts all False (no simulation has hit it yet).
    s1_transformed = np.zeros(S, dtype=bool)

    # How many consecutive rolls has this simulation gone without hitting ANY
    # subset 1 outcome? When this reaches 79, pity forces a subset 1 hit.
    s1_pity = np.zeros(S, dtype=np.int32)

    # Which of the 10 subset 2 items has this simulation collected? A 2D
    # boolean array of shape (100000, 10). s2_achieved[i, j] = True means
    # simulation i has collected item j. Starts all False.
    s2_achieved = np.zeros((S, n_s2), dtype=bool)

    # Has this simulation collected all 10 S2 items and unlocked the permanent
    # 35-pt reward? Starts all False.
    s2_completed = np.zeros(S, dtype=bool)

    # Same as s1_pity but for subset 2. Pity triggers at 10 rolls (counter >= 9).
    s2_pity = np.zeros(S, dtype=np.int32)

    # How many emotes / icons does this simulation still have uncollected?
    # np.full(S, 263) creates an array of 100,000 values all set to 263.
    # We only need counts (not which specific items) because all emotes are
    # identical in probability and points, and likewise for icons.
    s3_tier_a_remaining = np.full(S, n_tier_a_total, dtype=np.int32)
    s3_tier_b_remaining = np.full(S, n_tier_b_total, dtype=np.int32)

    # Running total of points earned so far for each simulation.
    # dtype=np.float64 gives 64-bit floating-point precision (~15 decimal digits).
    cumulative_pts = np.zeros(S, dtype=np.float64)

    # -- Output arrays ---------------------------------------------------------
    #
    # These arrays store one value per roll (3000 elements). They're filled
    # in during the main loop and returned at the end.

    # Which percentiles to compute every roll. Distribution-free -- no
    # assumptions about the shape of the distribution.
    PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    n_pct = len(PERCENTILES)

    # np.empty() allocates memory without initialising it (slightly faster
    # than np.zeros since we'll overwrite every value anyway).
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

    # Dictionary mapping roll numbers -> full copies of the 100k cumulative_pts
    # array, saved at checkpoint rolls for distribution analysis.
    dist_snapshots = {}

    # ==========================================================================
    #  MAIN SIMULATION LOOP -- one iteration per roll
    # ==========================================================================
    #
    # Each iteration processes all 100,000 simulations in parallel using
    # numpy vectorisation. The steps are:
    #   1. Check pity counters and force subsets where needed
    #   2. For non-forced simulations, randomly pick a subset
    #   3. Within each subset, resolve what happens (points, state changes)
    #   4. Update accumulators (points, pity counters)
    #   5. Record statistics for this roll

    for roll in range(max_rolls):

        # -- Step 1: Pity check ------------------------------------------------
        #
        # forced_subset tracks which simulations are being forced by pity.
        # -1 = not forced (roll normally), 0 = forced into S1, 1 = forced into S2.
        #
        # dtype=np.int8 uses 1 byte per value instead of 8 (default int64),
        # saving memory when we have 100k values and only need -1, 0, 1, 2.
        forced_subset = np.full(S, -1, dtype=np.int8)

        # Check which simulations have reached the S1 pity limit.
        # This is an element-wise comparison: numpy compares all 100k values
        # at once and returns a boolean array of the same shape.
        s1_pity_hit = s1_pity >= s1_pity_limit

        # .any() returns True if at least one element is True. The "if" avoids
        # doing unnecessary work when no simulation has triggered pity.
        # Boolean indexing: forced_subset[s1_pity_hit] = 0 sets forced_subset
        # to 0 only at positions where s1_pity_hit is True.
        if s1_pity_hit.any():
            forced_subset[s1_pity_hit] = 0

        # S2 pity check -- but only for simulations not already claimed by S1.
        # The "& (forced_subset == -1)" clause handles SIMULTANEOUS PITY:
        # if both S1 and S2 pity trigger on the same roll, S1 takes priority.
        # The S2 counter will increment by 1 more (since no S2 hit occurred),
        # and S2 pity will trigger on the very next roll instead.
        s2_pity_hit = (s2_pity >= s2_pity_limit) & (forced_subset == -1)
        if s2_pity_hit.any():
            forced_subset[s2_pity_hit] = 1

        # -- Step 2: Subset selection ------------------------------------------

        # Array to store which subset each simulation enters this roll.
        chosen_subset = np.empty(S, dtype=np.int8)

        # Boolean mask: True for simulations rolling normally (not pity-forced).
        normal = forced_subset == -1

        if normal.any():
            # Generate one uniform random number in [0, 1) for each normal
            # simulation. normal.sum() counts the True values (since True=1).
            r = rng.random(normal.sum())

            # INVERSE CDF SAMPLING: np.searchsorted finds, for each random
            # number, the index of the first CDF value that is >= r.
            #
            # subset_cdf = [0.005, 0.105, 1.0]
            #   if r < 0.005          -> index 0 -> subset 1  (0.5% chance)
            #   if 0.005 <= r < 0.105 -> index 1 -> subset 2  (10% chance)
            #   if r >= 0.105         -> index 2 -> subset 3  (89.5% chance)
            #
            # This is equivalent to rolling a weighted die with three faces.
            cs = np.searchsorted(subset_cdf, r)

            # Safety: clamp to [0, 2] in case floating-point edge cases
            # (like r = 1.0 exactly) produce an out-of-range index.
            cs = np.clip(cs, 0, 2)

            # Write results back into the positions of non-forced simulations.
            # .astype(np.int8) converts from int64 to match the array's dtype.
            chosen_subset[normal] = cs.astype(np.int8)

        # For pity-forced simulations, just use whatever pity dictated.
        # ~normal is the bitwise NOT: True where normal is False.
        forced = ~normal
        if forced.any():
            chosen_subset[forced] = forced_subset[forced]

        # Create three boolean masks splitting the 100k simulations by subset.
        is_s1 = chosen_subset == 0
        is_s2 = chosen_subset == 1
        is_s3 = chosen_subset == 2

        # Fresh array: each simulation's points earned THIS roll (starts at 0).
        pts_earned = np.zeros(S, dtype=np.float64)

        # -- Step 3a: Resolve subset 1 -----------------------------------------

        if is_s1.any():
            # np.where(condition) returns a tuple of arrays of indices where
            # the condition is True. [0] gets the first (only) array.
            # e.g. s1_sims might be [47, 2301, 8899, ...] -- the indices of
            # simulations that hit subset 1 this roll.
            s1_sims = np.where(is_s1)[0]

            # "Fancy indexing": s1_transformed[s1_sims] grabs the transform
            # status of just these simulations. tf_mask is a boolean array.
            tf_mask = s1_transformed[s1_sims]

            # Chained fancy indexing: s1_sims[tf_mask] selects the simulation
            # indices that BOTH hit S1 AND have already transformed. Those
            # simulations earn 270 points.
            pts_earned[s1_sims[tf_mask]] = s1_tf_pts

            # ~tf_mask inverts the mask: simulations hitting S1 for the FIRST
            # time. They earn 0 points (the pre-transform item), but their
            # transform flag is set to True. From next roll onward, hitting S1
            # will give them 270 points.
            s1_transformed[s1_sims[~tf_mask]] = True

        # -- Step 3b: Resolve subset 2 -----------------------------------------

        if is_s2.any():
            s2_sims = np.where(is_s2)[0]

            # Which of these simulations have already completed S2?
            comp_mask = s2_completed[s2_sims]

            # Completed simulations earn 35 points (the permanent reward).
            pts_earned[s2_sims[comp_mask]] = s2_comp_pts

            # Not-yet-completed simulations need within-subset item selection.
            not_comp_sims = s2_sims[~comp_mask]

            if len(not_comp_sims) > 0:
                # s2_ach is a 2D boolean array of shape (n_not_completed, 10).
                # Row i shows which of the 10 S2 items this simulation has.
                s2_ach = s2_achieved[not_comp_sims]

                # -- Redistribution calculation --
                #
                # When items are removed, their probability is split equally
                # among the remaining items. The formula is:
                #   adjusted_prob[i] = base_prob[i] + bonus
                #   bonus = sum(base_probs of removed items) / n_remaining
                #
                # Step 1: Total removed probability per simulation.
                # s2_base_p[np.newaxis, :] reshapes (10,) to (1, 10) for
                # broadcasting against s2_ach of shape (n, 10).
                # Multiplying: True (=1) keeps the prob, False (=0) zeros it.
                # .sum(axis=1) sums across the 10 items -> shape (n,).
                removed_sum = (s2_ach * s2_base_p[np.newaxis, :]).sum(axis=1)

                # Step 2: Count of remaining items per simulation.
                # np.maximum(..., 1) prevents division by zero.
                n_rem = np.maximum(n_s2 - s2_ach.sum(axis=1), 1)

                # Step 3: Per-item bonus = total removed / count remaining.
                bonus = removed_sum / n_rem

                # Step 4: Build adjusted probability vector.
                # active is True for items NOT yet collected.
                active = ~s2_ach

                # s2_base_p[np.newaxis, :] is shape (1, 10) -- base probs.
                # bonus[:, np.newaxis] is shape (n, 1) -- per-sim bonus.
                # Adding broadcasts to (n, 10): every item gets base + bonus.
                # Multiplying by active zeros out collected items.
                item_p = (s2_base_p[np.newaxis, :] + bonus[:, np.newaxis]) * active

                # Normalise each row to sum to 1. keepdims=True keeps the
                # result as shape (n, 1) so division broadcasts correctly.
                item_p_norm = item_p / np.maximum(item_p.sum(axis=1, keepdims=True), 1e-15)

                # Build a CDF per simulation. axis=1 means cumulative sum
                # across columns: [p0, p1, p2, ...] -> [p0, p0+p1, p0+p1+p2, ...].
                cdf = np.cumsum(item_p_norm, axis=1)

                # Generate one random number per simulation. [:, np.newaxis]
                # reshapes from (n,) to (n, 1) for broadcasting against the
                # (n, 10) CDF.
                r = rng.random(len(not_comp_sims))[:, np.newaxis]

                # VECTORISED INVERSE CDF SAMPLING:
                # cdf <= r produces an (n, 10) boolean array: True at every
                # CDF entry at or below the random number.
                # .sum(axis=1) counts how many are at or below -> this IS the
                # index of the chosen item.
                #
                # Example: CDF = [0.15, 0.30, 0.50, 0.70, 1.0], r = 0.42
                #   cdf <= r = [True, True, False, False, False]
                #   sum = 2 -> item index 2 is chosen.
                chosen_item = np.clip((cdf <= r).sum(axis=1), 0, n_s2 - 1)

                # 2D fancy indexing: mark the chosen item as collected.
                # not_comp_sims[i] collected item chosen_item[i].
                s2_achieved[not_comp_sims, chosen_item] = True

                # Check which simulations just completed all 10 items.
                # .all(axis=1) checks if every column in the row is True.
                newly_done = (s2_achieved[not_comp_sims].all(axis=1)
                              & ~s2_completed[not_comp_sims])
                if newly_done.any():
                    s2_completed[not_comp_sims[newly_done]] = True

        # -- Step 3c: Resolve subset 3 -----------------------------------------

        if is_s3.any():
            s3_sims = np.where(is_s3)[0]
            n_s3 = len(s3_sims)  # typically ~89,500 out of 100,000

            # Get remaining emote and icon counts. Cast to float64 because
            # we'll do division (integer division would truncate).
            a_rem = s3_tier_a_remaining[s3_sims].astype(np.float64)
            b_rem = s3_tier_b_remaining[s3_sims].astype(np.float64)

            # Total items remaining = emotes + icons + 5 permanents.
            n_remaining = a_rem + b_rem + n_s3_perm

            # Total base probability that's been removed from the pool.
            # (263 - a_rem) = consumed emotes, each with base prob tier_a_prob.
            total_removed = ((n_tier_a_total - a_rem) * tier_a_prob
                             + (n_tier_b_total - b_rem) * tier_b_prob)

            # Flat-equal redistribution: removed probability is divided equally
            # among ALL remaining items (permanents, emotes, and icons alike).
            bonus = total_removed / n_remaining

            # -- Category grouping --
            #
            # Instead of 742 columns (one per item), we use 7 columns:
            #   columns 0-4: the 5 individual permanent items
            #   column 5:    "all remaining emotes" as a single group
            #   column 6:    "all remaining icons" as a single group
            n_cats = n_s3_perm + 2   # = 7
            cat_p = np.zeros((n_s3, n_cats))

            # Each permanent item's adjusted probability = base + bonus.
            # Every simulation gets a different bonus (different remaining
            # counts), so this is a column of per-simulation values.
            for i in range(n_s3_perm):
                cat_p[:, i] = s3_perm_p[i] + bonus

            # Emote group probability = (count remaining) x (per-emote adjusted prob).
            # Since all emotes have the same adjusted prob (tier_a_prob + bonus),
            # multiplying by count gives the group total. This is mathematically
            # equivalent to summing 263 individual columns.
            cat_p[:, n_s3_perm] = a_rem * (tier_a_prob + bonus)

            # Same for the icon group.
            cat_p[:, n_s3_perm + 1] = b_rem * (tier_b_prob + bonus)

            # Normalise to sum to 1 per row.
            cat_p_norm = cat_p / np.maximum(cat_p.sum(axis=1, keepdims=True), 1e-15)

            # Inverse CDF sampling over 7 categories (same technique as S2).
            cdf = np.cumsum(cat_p_norm, axis=1)
            r = rng.random(n_s3)[:, np.newaxis]
            chosen_cat = np.clip((cdf <= r).sum(axis=1), 0, n_cats - 1)

            # Assign points based on which category was chosen.
            # Categories 0-4 = permanent items.
            for i in range(n_s3_perm):
                pts_earned[s3_sims[chosen_cat == i]] = s3_perm_pts[i]

            # Category 5 = emote group: award 2 pts, decrement remaining count.
            tier_a_hit = chosen_cat == n_s3_perm
            if tier_a_hit.any():
                hit = s3_sims[tier_a_hit]
                pts_earned[hit] = tier_a_pts
                # np.maximum(..., 0) prevents going negative (defensive).
                s3_tier_a_remaining[hit] = np.maximum(s3_tier_a_remaining[hit] - 1, 0)

            # Category 6 = icon group: same logic.
            tier_b_hit = chosen_cat == n_s3_perm + 1
            if tier_b_hit.any():
                hit = s3_sims[tier_b_hit]
                pts_earned[hit] = tier_b_pts
                s3_tier_b_remaining[hit] = np.maximum(s3_tier_b_remaining[hit] - 1, 0)

        # -- Step 4: Update accumulators ---------------------------------------

        # Add this roll's points to each simulation's running total.
        # This is element-wise: cumulative_pts[i] += pts_earned[i].
        cumulative_pts += pts_earned

        # Update pity counters. np.where(condition, value_if_true, value_if_false)
        # operates element-wise on all 100k simulations at once.
        # If a simulation hit S1 this roll, reset its counter to 0.
        # Otherwise, increment by 1 (one more roll without hitting S1).
        s1_pity = np.where(is_s1, 0, s1_pity + 1)
        s2_pity = np.where(is_s2, 0, s2_pity + 1)

        # -- Step 5: Record statistics -----------------------------------------

        # .mean() averages across all 100k simulations.
        cum_means[roll] = cumulative_pts.mean()

        # .std(ddof=1) computes the sample standard deviation (measure of spread).
        cum_stds[roll] = cumulative_pts.std(ddof=1)

        # Marginal = how much the mean increased this roll.
        marginal_means[roll] = cum_means[roll] - prev_mean
        prev_mean = cum_means[roll]

        # Since these are boolean arrays, .mean() gives the fraction that is
        # True = probability of having transformed S1 / completed S2 by now.
        s1_tf_arr[roll] = s1_transformed.mean()
        s2_comp_arr[roll] = s2_completed.mean()

        # Average number of one-time S3 items remaining across all simulations.
        ot_rem = s3_tier_a_remaining + s3_tier_b_remaining
        s3_ot_remaining_mean[roll] = ot_rem.astype(float).mean()

        # Compute empirical percentiles directly from the 100k values.
        # This is distribution-free -- no assumptions about the shape.
        pct_arr[roll] = np.percentile(cumulative_pts, PERCENTILES)

        # At checkpoint rolls, save a full copy of all 100k cumulative totals.
        # .copy() is critical: without it we'd save a reference to the same
        # array, which gets modified on subsequent rolls. .copy() freezes
        # this moment in time.
        if (roll + 1) in dist_checkpoint_set:
            dist_snapshots[roll + 1] = cumulative_pts.copy()

        # Progress indicator every 500 rolls.
        if (roll + 1) % 500 == 0:
            print(f"  Roll {roll+1}/{max_rolls}...")

    # -- Return results --------------------------------------------------------
    #
    # .tolist() converts numpy arrays to plain python lists for cleaner
    # serialisation and compatibility with non-numpy code.
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

        # Basic descriptive statistics across all 100k simulations.
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
        # CAVEAT: with 100k simulations, even trivially small deviations from
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

    # Step 2: Run 100,000 simulations of 1000 rolls each.
    results = simulate(
        config,
        max_rolls=1000,         # total number of rolls to simulate
        simulations=300_000,    # number of parallel simulations
        seed=42,                # fixed seed for reproducible results
        dist_checkpoints=DIST_CHECKPOINTS,
    )

    # Step 3: Print the three summary tables and distribution analysis.
    print_summary(results, every_n=1)
    print_distribution_analysis(results)
