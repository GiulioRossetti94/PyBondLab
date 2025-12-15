"""
Double Sort - Unconditional (Independent Sorts)

Demonstrates bivariate portfolio sorting with independent sorts.
Forms portfolios based on two variables independently.

Key concepts:
- DoubleSort with how='unconditional'
- Independent sorting on two characteristics
- 2D portfolio grid interpretation
- High-minus-low in each dimension

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

def generate_bond_data(n_bonds=150, n_periods=60, seed=42):
    """Generate synthetic bond panel with two sorting variables."""
    np.random.seed(seed)

    dates = pd.date_range('2010-01-31', periods=n_periods, freq='M')
    bond_ids = [f'BOND{i:04d}' for i in range(n_bonds)]

    data = []
    for bond_id in bond_ids:
        # Two characteristics: rating and size
        rating = np.random.randint(1, 23)
        size_percentile = np.random.uniform(0, 1)  # Size rank within universe

        # Returns depend on both characteristics
        rating_premium = (rating - 10) * 0.0008
        size_premium = (size_percentile - 0.5) * 0.002  # Small minus big

        for date in dates:
            ret = 0.005 + rating_premium + size_premium + np.random.normal(0, 0.02)
            bond_value = 100_000 * np.exp(size_percentile * 4 + np.random.normal(0, 0.3))

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
# Unconditional Double Sort: 3x3 Grid
# =============================================================================

print("=" * 80)
print("Unconditional Double Sort: Rating × Size")
print("=" * 80)
print()

data = generate_bond_data(n_bonds=150, n_periods=60)

print("Data summary:")
print(f"  Bonds: {data['ID'].nunique()}")
print(f"  Periods: {data['date'].nunique()}")
print(f"  Rating range: {data['RATING_NUM'].min()}-{data['RATING_NUM'].max()}")
print()

# Define double sort strategy
strategy = pbl.DoubleSort(
    holding_period=6,
    sort_var='RATING_NUM',      # First sort: Rating
    sort_var2='BOND_VALUE',     # Second sort: Size
    num_portfolios=3,           # 3 rating groups
    num_portfolios2=3,          # 3 size groups
    how='unconditional'         # Independent sorts
)

print("Strategy configuration:")
print(f"  Type: {strategy.__strategy_name__}")
print(f"  Sort method: {strategy.how}")
print(f"  Primary variable: {strategy.sort_var} ({strategy.num_portfolios} portfolios)")
print(f"  Secondary variable: {strategy.sort_var2} ({strategy.num_portfolios2} portfolios)")
print(f"  Total portfolios: {strategy.num_portfolios * strategy.num_portfolios2}")
print()

# Run formation
results = pbl.StrategyFormation(
    data,
    strategy=strategy,
    turnover=False,
    verbose=False
).fit()

print("Formation complete!")
print()

# =============================================================================
# Understanding the Portfolio Grid
# =============================================================================

print("=" * 80)
print("Portfolio Grid Structure")
print("=" * 80)
print()

print("Unconditional (Independent) sorting:")
print("  1. Sort all bonds by RATING_NUM into 3 groups")
print("  2. Sort all bonds by BOND_VALUE into 3 groups")
print("  3. Intersection creates 3×3=9 portfolios")
print()

print("Portfolio labels:")
labels = results.ret_vw.columns.tolist()
print(f"  {labels}")
print()

# Reshape to 2D grid for visualization
print("Grid layout (rows=Rating, cols=Size):")
print()
print("                Small    Medium    Large")
print("             ─────────────────────────────")

for i in range(3):
    row_portfolios = labels[i*3:(i+1)*3]
    rating_label = ['Low Rating', 'Mid Rating', 'High Rating'][i]
    print(f"  {rating_label:12s} │  {row_portfolios[0]:6s}  {row_portfolios[1]:6s}  {row_portfolios[2]:6s}")

print()

# =============================================================================
# Portfolio Returns
# =============================================================================

print("=" * 80)
print("Portfolio Returns")
print("=" * 80)
print()

# Compute mean returns for each portfolio
mean_returns = results.ret_vw.mean()

print("Mean returns by portfolio:")
for i in range(3):
    row_portfolios = labels[i*3:(i+1)*3]
    rating_label = ['Low', 'Mid', 'High'][i]
    print(f"  {rating_label} Rating: ", end="")
    for p in row_portfolios:
        print(f"{mean_returns[p]:7.4f}  ", end="")
    print()

print()

# =============================================================================
# High-Minus-Low Spread
# =============================================================================

print("=" * 80)
print("High-Minus-Low Spread")
print("=" * 80)
print()

print("Default HML interpretation depends on primary sort variable:")
print(f"  Primary sort: {strategy.sort_var}")
print(f"  HML = High {strategy.sort_var} - Low {strategy.sort_var}")
print()

print("HML Statistics:")
print(f"  Mean: {results.hml_vw.mean():.4f}")
print(f"  Std:  {results.hml_vw.std():.4f}")
print(f"  t-stat: {results.hml_vw.mean() / results.hml_vw.std() * np.sqrt(len(results.hml_vw)):.2f}")
print()

# Custom spreads
print("Custom spreads can be computed manually:")
print()

# Rating spread (high - low, averaged across size groups)
high_rating = results.ret_vw[labels[6:9]].mean(axis=1)  # P7, P8, P9
low_rating = results.ret_vw[labels[0:3]].mean(axis=1)   # P1, P2, P3
rating_spread = high_rating - low_rating

print("Rating spread (High - Low, size-averaged):")
print(f"  Mean: {rating_spread.mean():.4f}")
print(f"  Std:  {rating_spread.std():.4f}")
print()

# Size spread (large - small, averaged across rating groups)
large_size = results.ret_vw[[labels[2], labels[5], labels[8]]].mean(axis=1)
small_size = results.ret_vw[[labels[0], labels[3], labels[6]]].mean(axis=1)
size_spread = large_size - small_size

print("Size spread (Large - Small, rating-averaged):")
print(f"  Mean: {size_spread.mean():.4f}")
print(f"  Std:  {size_spread.std():.4f}")
print()

# =============================================================================
# Corner Portfolios
# =============================================================================

print("=" * 80)
print("Corner Portfolios")
print("=" * 80)
print()

print("Extreme combinations:")
print()

# Low rating, small size
corner_LS = results.ret_vw[labels[0]]  # P1
print(f"Low Rating, Small Size ({labels[0]}):")
print(f"  Mean: {corner_LS.mean():.4f}, Std: {corner_LS.std():.4f}")
print()

# High rating, large size
corner_HL = results.ret_vw[labels[8]]  # P9
print(f"High Rating, Large Size ({labels[8]}):")
print(f"  Mean: {corner_HL.mean():.4f}, Std: {corner_HL.std():.4f}")
print()

# Double sort spread
double_spread = corner_HL - corner_LS
print(f"Corner spread ({labels[8]} - {labels[0]}):")
print(f"  Mean: {double_spread.mean():.4f}")
print(f"  Std:  {double_spread.std():.4f}")
print()

# =============================================================================
# Summary
# =============================================================================

print("=" * 80)
print("Summary")
print("=" * 80)
print()
print("This example demonstrated:")
print("  1. Unconditional (independent) double sorting")
print("  2. Interpreting the 2D portfolio grid")
print("  3. Default HML spread based on primary sort")
print("  4. Computing custom spreads:")
print("     - Rating spread (size-averaged)")
print("     - Size spread (rating-averaged)")
print("     - Corner-to-corner spread")
print("  5. Accessing specific portfolio combinations")
print()
print("Unconditional sorts:")
print("  - Two independent univariate sorts")
print("  - Portfolio assignment based on intersection")
print("  - Useful when characteristics are uncorrelated")
print()
print("Next steps:")
print("  - See doublesort_conditional.py for conditional sorting")
print("  - Compare unconditional vs conditional for correlated variables")
print()
