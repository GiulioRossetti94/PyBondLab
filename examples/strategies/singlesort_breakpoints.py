"""
Single Sort with Custom Breakpoints

Demonstrates using custom breakpoints instead of equal-sized portfolios.
Creates three portfolios: Low (bottom 30%), Medium (30-70%), High (top 30%).

Key concepts:
- Custom breakpoint specification
- Unequal portfolio sizes
- Breakpoint universe filtering (e.g., use only investment-grade bonds)

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

def generate_bond_data(n_bonds=100, n_periods=60, seed=42):
    """Generate synthetic bond panel data."""
    np.random.seed(seed)

    dates = pd.date_range('2010-01-31', periods=n_periods, freq='M')
    bond_ids = [f'BOND{i:04d}' for i in range(n_bonds)]

    data = []
    for bond_id in bond_ids:
        rating = np.random.randint(1, 23)
        is_ig = 1 if rating <= 10 else 0  # Investment grade flag

        rating_premium = (rating - 10) * 0.001

        for date in dates:
            ret = 0.005 + rating_premium + np.random.normal(0, 0.02)
            bond_value = 1_000_000 * np.exp(np.random.normal(0, 0.5))
            oas = 200 + rating * 20 + np.random.normal(0, 50)  # OAS in basis points

            data.append({
                'date': date,
                'ID': bond_id,
                'ret': ret,
                'RATING_NUM': rating,
                'BOND_VALUE': bond_value,
                'OAS': oas,
                'IG': is_ig
            })

    df = pd.DataFrame(data)
    df = df.sort_values(['ID', 'date']).reset_index(drop=True)

    return df

# =============================================================================
# Example 1: Custom Breakpoints (30-40-30)
# =============================================================================

print("=" * 80)
print("Example 1: Custom Breakpoints - Three Unequal Portfolios")
print("=" * 80)
print()

data = generate_bond_data(n_bonds=100, n_periods=60)

print("Creating 30-40-30 portfolios (Low-Medium-High)...")
print()

strategy = pbl.SingleSort(
    holding_period=6,
    sort_var='OAS',
    breakpoints=[30, 70],       # Bottom 30%, middle 40%, top 30%
    skip=0
)

print(f"  Breakpoints: {strategy.breakpoints} percentiles")
print(f"  Number of portfolios: {strategy.num_portfolios}")
print()

results = pbl.StrategyFormation(
    data,
    strategy=strategy,
    turnover=False,
    verbose=False
).fit()

print("Results:")
print(f"  Portfolio labels: {results.ret_vw.columns.tolist()}")
print()

for i, col in enumerate(results.ret_vw.columns):
    portfolio_ret = results.ret_vw[col]
    print(f"  {col}: mean={portfolio_ret.mean():.4f}, std={portfolio_ret.std():.4f}")

print()
print(f"  High-Minus-Low: mean={results.hml_vw.mean():.4f}, "
      f"std={results.hml_vw.std():.4f}")
print()

# =============================================================================
# Example 2: Breakpoint Universe Filtering
# =============================================================================

print("=" * 80)
print("Example 2: Breakpoint Universe Filtering")
print("=" * 80)
print()

print("Scenario: Sort all bonds by OAS, but compute breakpoints using")
print("          only investment-grade bonds.")
print()

# Method 1: String-based filtering (column name)
strategy_filtered = pbl.SingleSort(
    holding_period=6,
    sort_var='OAS',
    num_portfolios=5,
    breakpoint_universe_func='IG',  # Use only IG=1 bonds for breakpoints
    skip=0
)

print("Method 1: String-based filter (breakpoint_universe_func='IG')")
print(f"  Breakpoints computed from: bonds where IG==1")
print(f"  All bonds assigned to portfolios based on these breakpoints")
print()

results_filtered = pbl.StrategyFormation(
    data,
    strategy=strategy_filtered,
    turnover=False,
    verbose=False
).fit()

print("Results with IG-only breakpoints:")
for i, col in enumerate(results_filtered.ret_vw.columns):
    portfolio_ret = results_filtered.ret_vw[col]
    print(f"  {col}: mean={portfolio_ret.mean():.4f}, std={portfolio_ret.std():.4f}")
print()

# Method 2: Lambda function filtering
strategy_lambda = pbl.SingleSort(
    holding_period=6,
    sort_var='OAS',
    num_portfolios=5,
    breakpoint_universe_func=lambda df: df['RATING_NUM'] <= 10,  # Custom filter
    skip=0
)

print("Method 2: Lambda function filter")
print(f"  Breakpoints computed from: bonds where RATING_NUM <= 10")
print()

results_lambda = pbl.StrategyFormation(
    data,
    strategy=strategy_lambda,
    turnover=False,
    verbose=False
).fit()

print("Results with lambda filter:")
for i, col in enumerate(results_lambda.ret_vw.columns):
    portfolio_ret = results_lambda.ret_vw[col]
    print(f"  {col}: mean={portfolio_ret.mean():.4f}, std={portfolio_ret.std():.4f}")
print()

# =============================================================================
# Example 3: Decile Portfolios with Custom Breakpoints
# =============================================================================

print("=" * 80)
print("Example 3: Decile Portfolios")
print("=" * 80)
print()

print("Creating 10 portfolios using decile breakpoints...")
print()

strategy_decile = pbl.SingleSort(
    holding_period=6,
    sort_var='OAS',
    breakpoints=[10, 20, 30, 40, 50, 60, 70, 80, 90],  # 10 portfolios
    skip=0
)

print(f"  Breakpoints: {strategy_decile.breakpoints}")
print(f"  Number of portfolios: {strategy_decile.num_portfolios}")
print()

results_decile = pbl.StrategyFormation(
    data,
    strategy=strategy_decile,
    turnover=False,
    verbose=False
).fit()

print("Results (showing first 3 and last 3 portfolios):")
for i in [0, 1, 2, -3, -2, -1]:
    col = results_decile.ret_vw.columns[i]
    portfolio_ret = results_decile.ret_vw[col]
    print(f"  {col}: mean={portfolio_ret.mean():.4f}, std={portfolio_ret.std():.4f}")

print()
print(f"  High-Minus-Low (D10-D1): mean={results_decile.hml_vw.mean():.4f}, "
      f"std={results_decile.hml_vw.std():.4f}")
print()

# =============================================================================
# Comparison: Equal vs Custom Breakpoints
# =============================================================================

print("=" * 80)
print("Comparison: Equal Quintiles vs 30-40-30 Split")
print("=" * 80)
print()

# Equal quintiles
strategy_equal = pbl.SingleSort(
    holding_period=6,
    sort_var='OAS',
    num_portfolios=5,
    skip=0
)

results_equal = pbl.StrategyFormation(
    data,
    strategy=strategy_equal,
    turnover=False,
    verbose=False
).fit()

print("Equal Quintiles (20-20-20-20-20):")
print(f"  HML mean: {results_equal.hml_vw.mean():.4f}")
print(f"  HML std:  {results_equal.hml_vw.std():.4f}")
print(f"  HML Sharpe: {results_equal.hml_vw.mean() / results_equal.hml_vw.std():.4f}")
print()

# Custom 30-40-30
print("Custom 30-40-30:")
print(f"  HML mean: {results.hml_vw.mean():.4f}")
print(f"  HML std:  {results.hml_vw.std():.4f}")
print(f"  HML Sharpe: {results.hml_vw.mean() / results.hml_vw.std():.4f}")
print()

# =============================================================================
# Summary
# =============================================================================

print("=" * 80)
print("Summary")
print("=" * 80)
print()
print("This example demonstrated:")
print("  1. Specifying custom breakpoints (e.g., 30-40-30 split)")
print("  2. Universe filtering for breakpoint calculation")
print("     - String-based: breakpoint_universe_func='column_name'")
print("     - Lambda function: breakpoint_universe_func=lambda df: condition")
print("  3. Creating decile portfolios with 9 breakpoints")
print("  4. Comparing equal vs custom breakpoint specifications")
print()
print("Use cases:")
print("  - Custom breakpoints: Focus on tails (e.g., 20-60-20)")
print("  - Universe filtering: NYSE breakpoints for all stocks")
print("  - Deciles: More granular portfolio formation")
print()
