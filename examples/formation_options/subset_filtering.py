"""
Subset Filtering Example

Demonstrates filtering data by rating, characteristics, or custom criteria
before portfolio formation. Different from return filtering (trimming/winsorizing).

Key concepts:
- Rating-based filtering (IG, NIG)
- Characteristic-based filtering (duration, size)
- Multiple simultaneous filters
- Custom filtering logic

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

def generate_bond_data_full(n_bonds=150, n_periods=60, seed=42):
    """Generate bond data spanning all ratings and characteristics."""
    np.random.seed(seed)

    dates = pd.date_range('2018-01-31', periods=n_periods, freq='M')
    bond_ids = [f'BOND{i:04d}' for i in range(n_bonds)]

    data = []
    for bond_id in bond_ids:
        # Full rating spectrum (1-22)
        rating = np.random.randint(1, 23)

        # Characteristics correlated with rating
        base_duration = 3 + (22 - rating) * 0.2 + np.random.uniform(-1, 1)
        base_size = np.exp(12 + (22 - rating) * 0.1 + np.random.normal(0, 1))

        for date in dates:
            duration = base_duration + np.random.normal(0, 0.3)
            duration = np.clip(duration, 0.5, 15)

            size = base_size * np.exp(np.random.normal(0, 0.1))

            ret = 0.005 + (rating - 10) * 0.001 + np.random.normal(0, 0.02)

            data.append({
                'date': date,
                'ID': bond_id,
                'ret': ret,
                'RATING_NUM': rating,
                'BOND_VALUE': size,
                'DURATION': duration
            })

    df = pd.DataFrame(data)
    df = df.sort_values(['ID', 'date']).reset_index(drop=True)

    return df

# =============================================================================
# Example 1: Rating-Based Filtering (Investment Grade)
# =============================================================================

print("=" * 80)
print("Example 1: Investment Grade Bonds Only")
print("=" * 80)
print()

data = generate_bond_data_full(n_bonds=150, n_periods=60)

print("Full data:")
print(f"  Total bonds: {data['ID'].nunique()}")
print(f"  Rating range: {data['RATING_NUM'].min()}-{data['RATING_NUM'].max()}")
print(f"  IG bonds (rating 1-10): {(data['RATING_NUM'] <= 10).sum()} obs")
print(f"  NIG bonds (rating 11-22): {(data['RATING_NUM'] > 10).sum()} obs")
print()

# Define strategy with IG filter
strategy = pbl.SingleSort(
    holding_period=6,
    sort_var='RATING_NUM',
    num_portfolios=5
)

results_ig = pbl.StrategyFormation(
    data,
    strategy=strategy,
    rating='IG',                       # Filter to investment grade
    turnover=False,
    verbose=False
).fit()

print("Results with IG filter:")
print(f"  HML mean: {results_ig.hml_vw.mean():.4f}")
print(f"  HML Sharpe: {results_ig.hml_vw.mean() / results_ig.hml_vw.std():.4f}")
print()

# =============================================================================
# Example 2: Non-Investment Grade Filtering
# =============================================================================

print("=" * 80)
print("Example 2: Non-Investment Grade (High Yield) Bonds")
print("=" * 80)
print()

results_nig = pbl.StrategyFormation(
    data,
    strategy=strategy,
    rating='NIG',                      # Filter to non-investment grade
    turnover=False,
    verbose=False
).fit()

print("Results with NIG filter:")
print(f"  HML mean: {results_nig.hml_vw.mean():.4f}")
print(f"  HML Sharpe: {results_nig.hml_vw.mean() / results_nig.hml_vw.std():.4f}")
print()

print("IG vs NIG comparison:")
print(f"  IG HML:  {results_ig.hml_vw.mean():.4f}")
print(f"  NIG HML: {results_nig.hml_vw.mean():.4f}")
print()

# =============================================================================
# Example 3: Custom Rating Range
# =============================================================================

print("=" * 80)
print("Example 3: Custom Rating Range")
print("=" * 80)
print()

# Filter to specific rating range (e.g., 5-15)
results_custom = pbl.StrategyFormation(
    data,
    strategy=strategy,
    rating=(5, 15),                    # Tuple: (min, max)
    turnover=False,
    verbose=False
).fit()

print("Custom rating range (5-15):")
print("  Includes: Lower IG through upper HY")
print("  Excludes: Highest quality (1-4) and most distressed (16-22)")
print(f"  HML mean: {results_custom.hml_vw.mean():.4f}")
print()

# =============================================================================
# Example 4: Characteristic-Based Filtering
# =============================================================================

print("=" * 80)
print("Example 4: Filter by Duration")
print("=" * 80)
print()

# Filter to bonds with duration between 3 and 8 years
subset_filter = {
    'DURATION': (3, 8)
}

results_duration = pbl.StrategyFormation(
    data,
    strategy=strategy,
    subset_filter=subset_filter,       # Characteristic filter
    turnover=False,
    verbose=False
).fit()

print("Duration filter (3-8 years):")
print("  Excludes short-duration and long-duration bonds")
print(f"  HML mean: {results_duration.hml_vw.mean():.4f}")
print()

# =============================================================================
# Example 5: Size-Based Filtering
# =============================================================================

print("=" * 80)
print("Example 5: Filter by Bond Size")
print("=" * 80)
print()

# Calculate size thresholds
size_25 = data['BOND_VALUE'].quantile(0.25)
size_75 = data['BOND_VALUE'].quantile(0.75)

subset_filter_size = {
    'BOND_VALUE': (size_25, size_75)
}

results_size = pbl.StrategyFormation(
    data,
    strategy=strategy,
    subset_filter=subset_filter_size,
    turnover=False,
    verbose=False
).fit()

print(f"Size filter (25th to 75th percentile):")
print(f"  Min size: ${size_25:,.0f}")
print(f"  Max size: ${size_75:,.0f}")
print(f"  HML mean: {results_size.hml_vw.mean():.4f}")
print()

# =============================================================================
# Example 6: Multiple Simultaneous Filters
# =============================================================================

print("=" * 80)
print("Example 6: Multiple Filters (Rating + Duration + Size)")
print("=" * 80)
print()

# Combine rating and characteristic filters
subset_filter_multi = {
    'DURATION': (3, 8),
    'BOND_VALUE': (size_25, size_75)
}

results_multi = pbl.StrategyFormation(
    data,
    strategy=strategy,
    rating='IG',                       # IG bonds only
    subset_filter=subset_filter_multi, # Plus duration and size
    turnover=False,
    verbose=False
).fit()

print("Combined filters:")
print("  Rating: Investment grade (1-10)")
print("  Duration: 3-8 years")
print(f"  Size: ${size_25:,.0f} - ${size_75:,.0f}")
print(f"  HML mean: {results_multi.hml_vw.mean():.4f}")
print()

# =============================================================================
# Example 7: Comparison Across Filters
# =============================================================================

print("=" * 80)
print("Example 7: Comparison Across Different Filters")
print("=" * 80)
print()

print("Strategy: Rating sort (5 portfolios, 6-month holding)")
print()

configs = [
    ('Full sample', None, None),
    ('IG only', 'IG', None),
    ('NIG only', 'NIG', None),
    ('Duration 3-8', None, {'DURATION': (3, 8)}),
    ('IG + Duration', 'IG', {'DURATION': (3, 8)}),
]

print(f"{'Filter':<20} {'HML Mean':>10} {'HML Sharpe':>12}")
print("-" * 44)

for name, rating_filt, subset_filt in configs:
    res = pbl.StrategyFormation(
        data,
        strategy=strategy,
        rating=rating_filt,
        subset_filter=subset_filt,
        turnover=False,
        verbose=False
    ).fit()

    sharpe = res.hml_vw.mean() / res.hml_vw.std()
    print(f"{name:<20} {res.hml_vw.mean():>10.4f} {sharpe:>12.4f}")

print()

# =============================================================================
# Example 8: Practical Use Cases
# =============================================================================

print("=" * 80)
print("Example 8: Practical Use Cases")
print("=" * 80)
print()

print("USE CASE 1: Focus on liquid segment")
print("  Problem: Illiquid bonds have stale prices")
print("  Solution: Filter to large, frequently traded bonds")
print("  Example: subset_filter={'BOND_VALUE': (1e6, np.inf)}")
print()

print("USE CASE 2: Control for duration")
print("  Problem: Duration exposure confounds credit effects")
print("  Solution: Restrict to narrow duration range")
print("  Example: subset_filter={'DURATION': (4, 6)}")
print()

print("USE CASE 3: Separate IG and HY analysis")
print("  Problem: IG and HY behave differently")
print("  Solution: Run strategies separately")
print("  Example: rating='IG' and rating='NIG'")
print()

print("USE CASE 4: Remove micro-cap bonds")
print("  Problem: Smallest bonds may be illiquid/mispriced")
print("  Solution: Filter out bottom decile")
print("  Example: subset_filter={'BOND_VALUE': (p10, np.inf)}")
print()

# =============================================================================
# Summary
# =============================================================================

print("=" * 80)
print("Summary")
print("=" * 80)
print()
print("This example demonstrated:")
print("  1. Rating filtering (IG, NIG, custom range)")
print("  2. Duration filtering")
print("  3. Size-based filtering")
print("  4. Multiple simultaneous filters")
print("  5. Comparison across filters")
print("  6. Practical use cases")
print()
print("Key parameters:")
print("  rating: 'IG', 'NIG', or (min, max) tuple")
print("  subset_filter: Dict of {column: (min, max)}")
print()
print("Example:")
print("  rating='IG'")
print("  subset_filter={'DURATION': (3, 8), 'BOND_VALUE': (1e6, 1e9)}")
print()
print("Difference from return filtering:")
print("  - Subset filtering: Remove bonds from universe")
print("  - Return filtering: Adjust/remove specific returns")
print()
print("Benefits:")
print("  ✓ Focus analysis on specific segment")
print("  ✓ Control for confounding factors")
print("  ✓ Improve liquidity/data quality")
print("  ✓ Match research design")
print()
print("Common filters:")
print("  - rating='IG': Investment grade analysis")
print("  - rating='NIG': High yield analysis")
print("  - DURATION: Control interest rate sensitivity")
print("  - BOND_VALUE: Focus on liquid bonds")
print("  - AGE: Exclude newly issued or very old bonds")
print()
