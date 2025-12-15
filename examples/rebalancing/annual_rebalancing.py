"""
Annual Rebalancing Example

Demonstrates portfolio rebalancing once per year (annually).
Minimal turnover strategy.

Key concepts:
- Annual rebalancing frequency
- Choice of rebalancing month
- Long holding period strategies
- Comparison with higher frequencies

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

def generate_bond_data(n_bonds=100, n_periods=84, seed=42):
    """Generate synthetic bond data (7 years monthly)."""
    np.random.seed(seed)

    dates = pd.date_range('2017-01-31', periods=n_periods, freq='M')
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
# Example 1: Basic Annual Rebalancing (June)
# =============================================================================

print("=" * 80)
print("Example 1: Annual Rebalancing in June")
print("=" * 80)
print()

data = generate_bond_data(n_bonds=100, n_periods=84)

print("Data summary:")
print(f"  Date range: {data['date'].min().date()} to {data['date'].max().date()}")
print(f"  Total periods: {data['date'].nunique()} months ({data['date'].nunique()//12} years)")
print()

# Define annual rebalancing strategy
strategy_annual = pbl.SingleSort(
    holding_period=12,
    sort_var='RATING_NUM',
    num_portfolios=5,
    rebalance_frequency='annual',      # Rebalance once per year
    rebalance_month=6                  # June
)

print("Strategy configuration:")
print(f"  Holding period: {strategy_annual.holding_period} months")
print(f"  Rebalancing frequency: {strategy_annual.rebalance_frequency}")
print(f"  Rebalancing month: {strategy_annual.rebalance_month} (June)")
print()

print("Rebalancing schedule:")
print("  June 2017, June 2018, June 2019, ... (once per year)")
print()

# Run formation
results_annual = pbl.StrategyFormation(
    data,
    strategy=strategy_annual,
    turnover=True,
    verbose=False
).fit()

print("Formation complete!")
print()

# =============================================================================
# Example 2: Different Rebalancing Months
# =============================================================================

print("=" * 80)
print("Example 2: Annual Rebalancing in Different Months")
print("=" * 80)
print()

months = [1, 6, 12]
month_names = ['January', 'June', 'December']

print("Testing annual rebalancing in different months:")
print()

for month, name in zip(months, month_names):
    strategy_temp = pbl.SingleSort(
        holding_period=12,
        sort_var='RATING_NUM',
        num_portfolios=5,
        rebalance_frequency='annual',
        rebalance_month=month,
        verbose=False
    )

    results_temp = pbl.StrategyFormation(
        data,
        strategy=strategy_temp,
        turnover=False,
        verbose=False
    ).fit()

    print(f"  Rebalance in {name}:")
    print(f"    HML mean: {results_temp.hml_vw.mean():.4f}")
    print(f"    HML Sharpe: {results_temp.hml_vw.mean() / results_temp.hml_vw.std():.4f}")

print()
print("Note: Results vary slightly due to different rebalancing dates")
print()

# =============================================================================
# Example 3: Frequency Comparison
# =============================================================================

print("=" * 80)
print("Example 3: Monthly vs Quarterly vs Annual")
print("=" * 80)
print()

# Monthly
strategy_monthly = pbl.SingleSort(
    holding_period=12,
    sort_var='RATING_NUM',
    num_portfolios=5,
    rebalance_frequency='monthly',
    verbose=False
)

results_monthly = pbl.StrategyFormation(
    data,
    strategy=strategy_monthly,
    turnover=True,
    verbose=False
).fit()

# Quarterly
strategy_quarterly = pbl.SingleSort(
    holding_period=12,
    sort_var='RATING_NUM',
    num_portfolios=5,
    rebalance_frequency='quarterly',
    rebalance_month=[3, 6, 9, 12],
    verbose=False
)

results_quarterly = pbl.StrategyFormation(
    data,
    strategy=strategy_quarterly,
    turnover=True,
    verbose=False
).fit()

print("Comparison (all with 12-month holding period):")
print()

print("MONTHLY rebalancing:")
print(f"  Rebalances per year: 12")
print(f"  HML mean: {results_monthly.hml_vw.mean():.4f}")
print(f"  HML Sharpe: {results_monthly.hml_vw.mean() / results_monthly.hml_vw.std():.4f}")
print()

print("QUARTERLY rebalancing:")
print(f"  Rebalances per year: 4")
print(f"  HML mean: {results_quarterly.hml_vw.mean():.4f}")
print(f"  HML Sharpe: {results_quarterly.hml_vw.mean() / results_quarterly.hml_vw.std():.4f}")
print()

print("ANNUAL rebalancing:")
print(f"  Rebalances per year: 1")
print(f"  HML mean: {results_annual.hml_vw.mean():.4f}")
print(f"  HML Sharpe: {results_annual.hml_vw.mean() / results_annual.hml_vw.std():.4f}")
print()

# =============================================================================
# Example 4: Annual with Different Holding Periods
# =============================================================================

print("=" * 80)
print("Example 4: Annual Rebalancing with Different Holding Periods")
print("=" * 80)
print()

print("Annual rebalancing can be combined with various holding periods:")
print()

holding_periods = [6, 12, 18, 24]

for hp in holding_periods:
    strategy_temp = pbl.SingleSort(
        holding_period=hp,
        sort_var='RATING_NUM',
        num_portfolios=5,
        rebalance_frequency='annual',
        rebalance_month=6,
        verbose=False
    )

    results_temp = pbl.StrategyFormation(
        data,
        strategy=strategy_temp,
        turnover=False,
        verbose=False
    ).fit()

    overlap = "overlapping" if hp > 12 else "non-overlapping" if hp == 12 else "with gaps"

    print(f"  Holding period = {hp} months ({overlap}):")
    print(f"    HML mean: {results_temp.hml_vw.mean():.4f}")
    print(f"    HML Sharpe: {results_temp.hml_vw.mean() / results_temp.hml_vw.std():.4f}")

print()

# =============================================================================
# Example 5: Practical Considerations
# =============================================================================

print("=" * 80)
print("Example 5: Practical Considerations")
print("=" * 80)
print()

print("Common annual rebalancing months:")
print()

print("JUNE (fiscal year end for many institutions):")
print("  - Aligns with mid-year performance review")
print("  - Common in academic calendar")
print("  - Used in many studies")
print()

print("DECEMBER (calendar year end):")
print("  - Natural year boundary")
print("  - Tax considerations")
print("  - Reporting alignment")
print()

print("JANUARY (start of year):")
print("  - Fresh start effect")
print("  - January effect considerations")
print("  - Budget cycle alignment")
print()

print("Choice of month typically has minimal impact on long-term performance")
print()

# =============================================================================
# Example 6: Transaction Cost Implications
# =============================================================================

print("=" * 80)
print("Example 6: Transaction Cost Implications")
print("=" * 80)
print()

print("Annual rebalancing minimizes transaction costs:")
print()

print("Turnover comparison (relative to annual = 1.0):")
print("  Annual:    1.0× (baseline)")
print("  Quarterly: ~4.0× (4 times per year)")
print("  Monthly:   ~12.0× (12 times per year)")
print()

print("For a strategy with:")
print("  - 20% turnover per rebalance")
print("  - 50 bps transaction cost")
print()
print("Annual costs:")
print("  Annual:    20% × 1 × 50 bps = 10 bps/year")
print("  Quarterly: 20% × 4 × 50 bps = 40 bps/year")
print("  Monthly:   20% × 12 × 50 bps = 120 bps/year")
print()

print("Annual rebalancing can preserve ~30-110 bps/year vs more frequent rebalancing")
print()

# =============================================================================
# Summary
# =============================================================================

print("=" * 80)
print("Summary")
print("=" * 80)
print()
print("This example demonstrated:")
print("  1. Basic annual rebalancing (June)")
print("  2. Different rebalancing months (January, June, December)")
print("  3. Comparison with quarterly and monthly frequencies")
print("  4. Different holding periods with annual rebalancing")
print("  5. Practical month selection considerations")
print("  6. Transaction cost implications")
print()
print("Annual rebalancing:")
print("  ✓ Minimal turnover (1 rebalance/year)")
print("  ✓ Lowest transaction costs")
print("  ✓ Simple to implement")
print("  ✓ Suitable for buy-and-hold strategies")
print()
print("Trade-offs:")
print("  - Portfolios may drift significantly before rebalancing")
print("  - Less responsive to changing market conditions")
print("  - May underperform more frequent rebalancing in trending markets")
print()
print("Use cases:")
print("  - Low-turnover factor strategies")
print("  - Institutional portfolios minimizing costs")
print("  - Long-term value/quality strategies")
print("  - Academic studies (simple benchmark)")
print()
print("Next steps:")
print("  - See custom_frequency.py for custom intervals (e.g., 4 months)")
print("  - See rebalancing_comparison.py for detailed comparison")
print()
