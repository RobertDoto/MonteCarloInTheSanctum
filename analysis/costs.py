"""
Shared RP cost utilities.

Provides cheapest_cost() and roll_cost() using an unbounded-knapsack DP
that finds the minimum spend to cover any RP total by mixing packages.
All arithmetic is done in integer pence to avoid floating-point drift.
"""

RP_PER_ROLL = 400

PACKAGES = [
    (575,   4.50),
    (1380,  10.25),
    (2800,  19.99),
    (4500,  31.50),
    (6500,  44.99),
    (13500, 88.99),
    (33500, 220.00),
    (60200, 385.00),
]

# Sorted ascending by RP for DP.
_PACKAGES_SORTED = sorted(PACKAGES, key=lambda p: p[0])
_PKG_CENTS = [(rp, round(cost * 100)) for rp, cost in _PACKAGES_SORTED]
_MAX_RP = _PACKAGES_SORTED[-1][0]


def cheapest_cost(rp_needed):
    """
    Return the minimum cost (£) to purchase at least rp_needed RP by
    mixing packages from PACKAGES. Uses unbounded knapsack DP in integer
    pence to avoid floating-point drift.
    """
    INF = 10 ** 9
    size = rp_needed + _MAX_RP + 1
    dp = [INF] * size
    dp[0] = 0

    for i in range(1, size):
        for rp, cents in _PKG_CENTS:
            if rp <= i and dp[i - rp] + cents < dp[i]:
                dp[i] = dp[i - rp] + cents

    best_cents = min(dp[rp_needed : rp_needed + _MAX_RP + 1])
    return best_cents / 100


def roll_cost(n_rolls):
    """Return the minimum cost (£) to purchase enough RP for n_rolls rolls."""
    return cheapest_cost(n_rolls * RP_PER_ROLL)
