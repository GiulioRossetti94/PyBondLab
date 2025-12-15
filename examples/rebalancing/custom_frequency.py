"""
Custom Rebalancing Frequency Example

Demonstrates custom rebalancing intervals (e.g., every 4, 5, or 8 months).
Provides flexibility beyond standard frequencies.

Key concepts:
- Custom integer frequency
- Starting month selection
- Non-standard intervals
- Use cases for custom frequencies

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
    """Generate synthetic bond data."""
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
# Example 1: 4-Month Rebalancing
# =============================================================================

print("=" * 80)
print("Example 1: Rebalance Every 4 Months")
print("=" * 80)
print()

data = generate_bond_data(n_bonds=100, n_periods=72)

print("Data summary:")
print(f"  Date range: {data['date'].min().date()} to {data['date'].max().date()}")
print(f"  Total periods: {data['date'].nunique()} months")
print()

# Rebalance every 4 months starting in January
strategy_4m = pbl.SingleSort(
    holding_period=6,
    sort_var='RATING_NUM',
    num_portfolios=5,
    rebalance_frequency=4,             # Custom: every 4 months
    rebalance_month=1                  # Start in January
)

print("Strategy configuration:")
print(f"  Rebalancing frequency: Every {strategy_4m.rebalance_frequency} months")
print(f"  Starting month: {strategy_4m.rebalance_month} (January)")
print()

print("Rebalancing schedule:")
print("  Jan, May, Sep (Year 1)")
print("  Jan, May, Sep (Year 2)")
print("  ... (repeats every 4 months)")
print()

results_4m = pbl.StrategyFormation(
    data,
    strategy=strategy_4m,
    turnover=False,
    verbose=False
).fit()

print("Results:")
print(f"  HML mean: {results_4m.hml_vw.mean():.4f}")
print(f"  HML Sharpe: {results_4m.hml_vw.mean() / results_4m.hml_vw.std():.4f}")
print()

# =============================================================================
# Example 2: Different Starting Months
# =============================================================================

print("=" * 80)
print("Example 2: Same Frequency, Different Starting Months")
print("=" * 80)
print()

print("4-month rebalancing with different start months:")
print()

start_months = [1, 3, 6]
month_names = ['January', 'March', 'June']

for start, name in zip(start_months, month_names):
    strategy_temp = pbl.SingleSort(
        holding_period=6,
        sort_var='RATING_NUM',
        num_portfolios=5,
        rebalance_frequency=4,
        rebalance_month=start,
        verbose=False
    )

    results_temp = pbl.StrategyFormation(
        data,
        strategy=strategy_temp,
        turnover=False,
        verbose=False
    ).fit()

    print(f"  Start in {name}:")
    if start == 1:
        print(f"    Schedule: Jan, May, Sep, Jan, ...")
    elif start == 3:
        print(f"    Schedule: Mar, Jul, Nov, Mar, ...")
    else:
        print(f"    Schedule: Jun, Oct, Feb, Jun, ...")
    print(f"    HML Sharpe: {results_temp.hml_vw.mean() / results_temp.hml_vw.std():.4f}")

print()

# =============================================================================
# Example 3: Various Custom Frequencies
# =============================================================================

print("=" * 80)
print("Example 3: Comparing Custom Frequencies")
print("=" * 80)
print()

frequencies = [2, 4, 5, 8, 10]

print("Testing various custom frequencies:")
print()

for freq in frequencies:
    strategy_temp = pbl.SingleSort(
        holding_period=12,
        sort_var='RATING_NUM',
        num_portfolios=5,
        rebalance_frequency=freq,
        rebalance_month=1,
        verbose=False
    )

    results_temp = pbl.StrategyFormation(
        data,
        strategy=strategy_temp,
        turnover=False,
        verbose=False
    ).fit()

    rebalances_per_year = 12 / freq

    print(f"  Every {freq} months (~{rebalances_per_year:.1f} rebalances/year):")
    print(f"    HML mean: {results_temp.hml_vw.mean():.4f}")
    print(f"    HML Sharpe: {results_temp.hml_vw.mean() / results_temp.hml_vw.std():.4f}")

print()

# =============================================================================
# Example 4: Matching Holding Period and Rebalancing
# =============================================================================

print("=" * 80)
print("Example 4: Matching Holding Period to Rebalancing Frequency")
print("=" * 80)
print()

print("Non-overlapping portfolios (holding_period = rebalance_frequency):")
print()

configs = [
    (3, 3, "Every 3 months, hold 3 months"),
    (4, 4, "Every 4 months, hold 4 months"),
    (6, 6, "Every 6 months, hold 6 months"),
]

for hp, freq, desc in configs:
    strategy_temp = pbl.SingleSort(
        holding_period=hp,
        sort_var='RATING_NUM',
        num_portfolios=5,
        rebalance_frequency=freq,
        rebalance_month=1,
        verbose=False
    )

    results_temp = pbl.StrategyFormation(
        data,
        strategy=strategy_temp,
        turnover=False,
        verbose=False
    ).fit()

    print(f"  {desc}:")
    print(f"    HML Sharpe: {results_temp.hml_vw.mean() / results_temp.hml_vw.std():.4f}")

print()

# =============================================================================
# Example 5: Use Cases for Custom Frequencies
# =============================================================================

print("=" * 80)
print("Example 5: Use Cases for Custom Frequencies")
print("=" * 80)
print()

print("2-MONTH rebalancing:")
print("  - More frequent than quarterly")
print("  - Responsive to changing conditions")
print("  - Moderate transaction costs")
print()

print("4-MONTH rebalancing:")
print("  - Between quarterly and semi-annual")
print("  - 3 rebalances per year")
print("  - Good balance of responsiveness and costs")
print()

print("5-MONTH rebalancing:")
print("  - Non-standard, less common")
print("  - 2.4 rebalances per year")
print("  - Research-specific applications")
print()

print("8-MONTH rebalancing:")
print("  - Between semi-annual and annual")
print("  - 1.5 rebalances per year")
print("  - Low turnover strategy")
print()

# =============================================================================
# Example 6: Practical Considerations
# =============================================================================

print("=" * 80)
print("Example 6: Practical Considerations")
print("=" * 80)
print()

print("When to use custom frequencies:")
print()

print("✓ Non-standard reporting cycles:")
print("    - Align with company-specific schedules")
print("    - Match institutional review periods")
print()

print("✓ Research robustness:")
print("    - Test strategy sensitivity to rebalancing")
print("    - Avoid data-snooping on standard frequencies")
print()

print("✓ Cost-return trade-off:")
print("    - Fine-tune transaction costs vs performance")
print("    - Find optimal frequency for specific market")
print()

print("✓ Calendar considerations:")
print("    - Avoid crowded rebalancing dates")
print("    - Spread liquidity demand")
print()

# =============================================================================
# Example 7: Frequency Spectrum
# =============================================================================

print("=" * 80)
print("Example 7: Complete Frequency Spectrum")
print("=" * 80)
print()

print("Rebalancing frequency spectrum (rebalances per year):")
print()
print("  Monthly (12×):      Standard, highest turnover")
print("  Bi-monthly (6×):    Custom, moderate-high turnover")
print("  Quarterly (4×):     Standard, moderate turnover")
print("  4-month (3×):       Custom, moderate turnover")
print("  Semi-annual (2×):   Standard, low turnover")
print("  8-month (1.5×):     Custom, low turnover")
print("  Annual (1×):        Standard, minimal turnover")
print()

print("Choose based on:")
print("  - Transaction costs in your market")
print("  - Strategy persistence/decay")
print("  - Operational constraints")
print("  - Research question")
print()

# =============================================================================
# Summary
# =============================================================================

print("=" * 80)
print("Summary")
print("=" * 80)
print()
print("This example demonstrated:")
print("  1. 4-month custom rebalancing")
print("  2. Different starting months for same frequency")
print("  3. Various custom frequencies (2, 4, 5, 8, 10 months)")
print("  4. Matching holding period to rebalancing frequency")
print("  5. Use cases for custom frequencies")
print("  6. Practical considerations")
print("  7. Complete frequency spectrum")
print()
print("Custom frequency parameters:")
print("  rebalance_frequency: Integer (e.g., 4 for every 4 months)")
print("  rebalance_month: Starting month (1-12)")
print()
print("Advantages:")
print("  ✓ Flexibility beyond standard frequencies")
print("  ✓ Fine-tune cost-return trade-off")
print("  ✓ Avoid crowded rebalancing dates")
print("  ✓ Align with institutional schedules")
print()
print("Common custom frequencies:")
print("  - 2 months: Between monthly and quarterly")
print("  - 4 months: Between quarterly and semi-annual")
print("  - 5 months: Non-standard, research-specific")
print("  - 8 months: Low-turnover alternative")
print()
