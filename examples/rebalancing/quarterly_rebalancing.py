"""
Quarterly Rebalancing Example

Demonstrates portfolio rebalancing every 3 months (quarterly).
Compares with default monthly rebalancing.

Key concepts:
- Non-staggered rebalancing
- Specific rebalancing months
- Reduced turnover vs monthly
- Holding period vs rebalancing frequency

Author: Giulio Rossetti
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import PyBondLab as pbl

# =============================================================================
# Generate Synthetic Data
# =============================================================================

def generate_bond_data(n_bonds=100, n_periods=72, seed=42):
    """Generate synthetic bond data (6 years monthly)."""
    np.random.seed(seed)

    dates = pd.date_range('2018-01-31', periods=n_periods, freq='M')
    bond_ids = [f'BOND{i:04d}' for i in range(n_bonds)]

    data = []
    for bond_id in bond_ids:
        rating = np.random.randint(1, 23)
        rating_premium = (rating - 10) * 0.001

        for date in dates:
            ret = 0.005 + rating_premium + np.random.normal(0, 0.02)
            bond_value = 1_000_000 * np.exp(np.random.normal(0, 0.3))

            data.append({
                'date': date,
                'ID': bond_id,
                'ret': ret,
                'RATING_NUM': rating,
                'BOND_VALUE': bond_value
            })

    df = pd.DataFrame(data)
    df = df.sort_values(['ID', 'date']).reset_index(drop=True)

    return df

# =============================================================================
# Example 1: Basic Quarterly Rebalancing
# =============================================================================

print("=" * 80)
print("Example 1: Quarterly Rebalancing (March, June, September, December)")
print("=" * 80)
print()

data = generate_bond_data(n_bonds=100, n_periods=72)

print("Data summary:")
print(f"  Date range: {data['date'].min().date()} to {data['date'].max().date()}")
print(f"  Total periods: {data['date'].nunique()} months")
print()

# Define quarterly rebalancing strategy
strategy_quarterly = pbl.SingleSort(
    holding_period=6,
    sort_var='RATING_NUM',
    num_portfolios=5,
    rebalance_frequency='quarterly',    # Rebalance every 3 months
    rebalance_month=[3, 6, 9, 12]      # March, June, September, December
)

print("Strategy configuration:")
print(f"  Holding period: {strategy_quarterly.holding_period} months")
print(f"  Rebalancing frequency: {strategy_quarterly.rebalance_frequency}")
print(f"  Rebalancing months: {strategy_quarterly.rebalance_month}")
print()

print("Rebalancing schedule:")
print("  2018: March, June, September, December")
print("  2019: March, June, September, December")
print("  ... (repeats quarterly)")
print()

# Run formation
results_quarterly = pbl.StrategyFormation(
    data,
    strategy=strategy_quarterly,
    turnover=True,
    verbose=False
).fit()

print("Formation complete!")
print()

# =============================================================================
# Example 2: Monthly vs Quarterly Comparison
# =============================================================================

print("=" * 80)
print("Example 2: Monthly vs Quarterly Rebalancing")
print("=" * 80)
print()

# Monthly rebalancing (default)
strategy_monthly = pbl.SingleSort(
    holding_period=6,
    sort_var='RATING_NUM',
    num_portfolios=5,
    rebalance_frequency='monthly'      # Default: rebalance every month
)

results_monthly = pbl.StrategyFormation(
    data,
    strategy=strategy_monthly,
    turnover=True,
    verbose=False
).fit()

print("MONTHLY rebalancing:")
print(f"  Total rebalancing dates: {data['date'].nunique()}")
print(f"  HML mean: {results_monthly.hml_vw.mean():.4f}")
print(f"  HML std:  {results_monthly.hml_vw.std():.4f}")
print(f"  HML Sharpe: {results_monthly.hml_vw.mean() / results_monthly.hml_vw.std():.4f}")
if hasattr(results_monthly, 'turnover') and results_monthly.turnover is not None:
    print(f"  Avg turnover: {results_monthly.turnover:.4f}")
print()

print("QUARTERLY rebalancing:")
print(f"  Total rebalancing dates: ~{data['date'].nunique() // 3}")
print(f"  HML mean: {results_quarterly.hml_vw.mean():.4f}")
print(f"  HML std:  {results_quarterly.hml_vw.std():.4f}")
print(f"  HML Sharpe: {results_quarterly.hml_vw.mean() / results_quarterly.hml_vw.std():.4f}")
if hasattr(results_quarterly, 'turnover') and results_quarterly.turnover is not None:
    print(f"  Avg turnover: {results_quarterly.turnover:.4f}")
print()

# =============================================================================
# Example 3: Different Quarterly Schedules
# =============================================================================

print("=" * 80)
print("Example 3: Alternative Quarterly Schedules")
print("=" * 80)
print()

# Standard quarters (Mar, Jun, Sep, Dec)
print("Standard quarters (Mar, Jun, Sep, Dec):")
print("  Rebalance: End of Q1, Q2, Q3, Q4")
print()

# Offset quarters (Feb, May, Aug, Nov)
strategy_offset = pbl.SingleSort(
    holding_period=6,
    sort_var='RATING_NUM',
    num_portfolios=5,
    rebalance_frequency='quarterly',
    rebalance_month=[2, 5, 8, 11]      # Offset by 1 month
)

results_offset = pbl.StrategyFormation(
    data,
    strategy=strategy_offset,
    turnover=False,
    verbose=False
).fit()

print("Offset quarters (Feb, May, Aug, Nov):")
print(f"  HML mean: {results_offset.hml_vw.mean():.4f}")
print(f"  HML Sharpe: {results_offset.hml_vw.mean() / results_offset.hml_vw.std():.4f}")
print()

# =============================================================================
# Example 4: Holding Period vs Rebalancing Frequency
# =============================================================================

print("=" * 80)
print("Example 4: Holding Period vs Rebalancing Frequency")
print("=" * 80)
print()

print("Key distinction:")
print("  Holding period: How long to hold each portfolio")
print("  Rebalancing frequency: How often to form new portfolios")
print()

# Case 1: holding_period = 6, rebalance quarterly
print("Case 1: holding_period=6, rebalance_frequency='quarterly'")
print("  - Form portfolio every 3 months")
print("  - Hold each portfolio for 6 months")
print("  - Result: Overlapping portfolios (2 active at any time)")
print()

# Case 2: holding_period = 3, rebalance quarterly
strategy_case2 = pbl.SingleSort(
    holding_period=3,
    sort_var='RATING_NUM',
    num_portfolios=5,
    rebalance_frequency='quarterly',
    rebalance_month=[3, 6, 9, 12]
)

results_case2 = pbl.StrategyFormation(
    data,
    strategy=strategy_case2,
    turnover=False,
    verbose=False
).fit()

print("Case 2: holding_period=3, rebalance_frequency='quarterly'")
print("  - Form portfolio every 3 months")
print("  - Hold each portfolio for 3 months")
print("  - Result: Non-overlapping portfolios")
print(f"  - HML mean: {results_case2.hml_vw.mean():.4f}")
print()

# =============================================================================
# Example 5: Quarterly with Different Holding Periods
# =============================================================================

print("=" * 80)
print("Example 5: Quarterly Rebalancing with Different Holding Periods")
print("=" * 80)
print()

holding_periods = [3, 6, 9, 12]

print("Testing quarterly rebalancing with various holding periods:")
print()

for hp in holding_periods:
    strategy_temp = pbl.SingleSort(
        holding_period=hp,
        sort_var='RATING_NUM',
        num_portfolios=5,
        rebalance_frequency='quarterly',
        rebalance_month=[3, 6, 9, 12],
        verbose=False
    )

    results_temp = pbl.StrategyFormation(
        data,
        strategy=strategy_temp,
        turnover=False,
        verbose=False
    ).fit()

    print(f"  Holding period = {hp} months:")
    print(f"    HML mean: {results_temp.hml_vw.mean():.4f}")
    print(f"    HML Sharpe: {results_temp.hml_vw.mean() / results_temp.hml_vw.std():.4f}")

print()

# =============================================================================
# Summary
# =============================================================================

print("=" * 80)
print("Summary")
print("=" * 80)
print()
print("This example demonstrated:")
print("  1. Basic quarterly rebalancing (Mar, Jun, Sep, Dec)")
print("  2. Comparison with monthly rebalancing")
print("  3. Alternative quarterly schedules (offset months)")
print("  4. Holding period vs rebalancing frequency distinction")
print("  5. Different holding periods with quarterly rebalancing")
print()
print("Quarterly rebalancing:")
print("  ✓ Reduced turnover (4 rebalances/year vs 12)")
print("  ✓ Lower transaction costs")
print("  ✓ Easier to implement")
print("  ✓ Common in practice (end of fiscal quarters)")
print()
print("Trade-offs:")
print("  - Less frequent portfolio updates")
print("  - May miss short-term opportunities")
print("  - Performance often similar to monthly")
print()
print("Use cases:")
print("  - Institutional portfolios with transaction costs")
print("  - Academic research (robustness check)")
print("  - Matching fiscal quarter reporting")
print()
print("Next steps:")
print("  - See annual_rebalancing.py for annual rebalancing")
print("  - See custom_frequency.py for custom intervals")
print("  - See rebalancing_comparison.py for full comparison")
print()
