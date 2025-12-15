"""
Double Sort - Conditional (Sequential Sorts)

Demonstrates bivariate portfolio sorting with conditional (sequential) sorts.
First sorts by one variable, then sorts within each group by the second variable.

Key concepts:
- DoubleSort with how='conditional'
- Sequential sorting: control for first variable
- Isolating effect of second variable
- Comparison with unconditional sorting

Author: Giulio Rossetti
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import PyBondLab as pbl

# =============================================================================
# Generate Synthetic Data with Correlated Characteristics
# =============================================================================

def generate_bond_data(n_bonds=150, n_periods=60, seed=42):
    """
    Generate synthetic bond panel with correlated characteristics.
    Rating and size are positively correlated (larger bonds have better ratings).
    """
    np.random.seed(seed)

    dates = pd.date_range('2010-01-31', periods=n_periods, freq='M')
    bond_ids = [f'BOND{i:04d}' for i in range(n_bonds)]

    data = []
    for bond_id in bond_ids:
        # Create correlation: larger bonds tend to have better ratings
        base_quality = np.random.uniform(0, 1)

        # Rating improves with base quality
        rating = int(1 + base_quality * 15)  # 1-16 range
        rating = np.clip(rating, 1, 22)

        # Size also improves with base quality
        log_size = 11 + base_quality * 4 + np.random.normal(0, 0.5)
        size = np.exp(log_size)

        # Returns depend on both (with interaction)
        rating_effect = (rating - 10) * 0.0010
        size_effect = (base_quality - 0.5) * 0.0015

        for date in dates:
            ret = 0.005 + rating_effect + size_effect + np.random.normal(0, 0.02)

            data.append({
                'date': date,
                'ID': bond_id,
                'ret': ret,
                'RATING_NUM': rating,
                'BOND_VALUE': size
            })

    df = pd.DataFrame(data)
    df = df.sort_values(['ID', 'date']).reset_index(drop=True)

    return df

# =============================================================================
# Conditional Double Sort: 3x3 Grid
# =============================================================================

print("=" * 80)
print("Conditional Double Sort: Rating then Size")
print("=" * 80)
print()

data = generate_bond_data(n_bonds=150, n_periods=60)

# Check correlation
sample_date = data['date'].iloc[0]
sample_data = data[data['date'] == sample_date]
corr = sample_data[['RATING_NUM', 'BOND_VALUE']].corr().iloc[0, 1]

print("Data summary:")
print(f"  Bonds: {data['ID'].nunique()}")
print(f"  Periods: {data['date'].nunique()}")
print(f"  Rating-Size correlation: {corr:.3f}")
print()

# Define conditional double sort
strategy_cond = pbl.DoubleSort(
    holding_period=6,
    sort_var='RATING_NUM',      # First: Sort by rating into 3 groups
    sort_var2='BOND_VALUE',     # Second: Within each group, sort by size into 3 groups
    num_portfolios=3,
    num_portfolios2=3,
    how='conditional'           # Sequential sorting
)

print("Strategy configuration:")
print(f"  Type: {strategy_cond.__strategy_name__}")
print(f"  Sort method: {strategy_cond.how}")
print(f"  Step 1: Sort by {strategy_cond.sort_var} into {strategy_cond.num_portfolios} groups")
print(f"  Step 2: Within each group, sort by {strategy_cond.sort_var2} into {strategy_cond.num_portfolios2} groups")
print(f"  Total portfolios: {strategy_cond.num_portfolios * strategy_cond.num_portfolios2}")
print()

# Run formation
results_cond = pbl.StrategyFormation(
    data,
    strategy=strategy_cond,
    turnover=False,
    verbose=False
).fit()

print("Formation complete!")
print()

# =============================================================================
# Understanding Conditional vs Unconditional
# =============================================================================

print("=" * 80)
print("Conditional vs Unconditional Sorting")
print("=" * 80)
print()

print("CONDITIONAL (Sequential):")
print("  1. Sort all bonds by RATING_NUM into 3 groups")
print("  2. Within LOW rating group: sort by SIZE into 3 subgroups")
print("  3. Within MID rating group: sort by SIZE into 3 subgroups")
print("  4. Within HIGH rating group: sort by SIZE into 3 subgroups")
print("  Result: 3×3=9 portfolios, SIZE effect is within-rating")
print()

print("UNCONDITIONAL (Independent):")
print("  1. Sort all bonds by RATING_NUM into 3 groups (ignoring SIZE)")
print("  2. Sort all bonds by SIZE into 3 groups (ignoring RATING)")
print("  3. Take intersection of the two classifications")
print("  Result: 3×3=9 portfolios, but SIZE effect mixes across ratings")
print()

# =============================================================================
# Run Unconditional for Comparison
# =============================================================================

strategy_uncond = pbl.DoubleSort(
    holding_period=6,
    sort_var='RATING_NUM',
    sort_var2='BOND_VALUE',
    num_portfolios=3,
    num_portfolios2=3,
    how='unconditional'
)

results_uncond = pbl.StrategyFormation(
    data,
    strategy=strategy_uncond,
    turnover=False,
    verbose=False
).fit()

# =============================================================================
# Compare Results
# =============================================================================

print("=" * 80)
print("Comparing Size Effect: Conditional vs Unconditional")
print("=" * 80)
print()

print("The size effect is the difference between large and small bonds,")
print("averaged across rating groups.")
print()

labels = results_cond.ret_vw.columns.tolist()

# Conditional: size effect within rating groups
large_cond = results_cond.ret_vw[[labels[2], labels[5], labels[8]]].mean(axis=1)
small_cond = results_cond.ret_vw[[labels[0], labels[3], labels[6]]].mean(axis=1)
size_effect_cond = large_cond - small_cond

print("CONDITIONAL size effect (within-rating):")
print(f"  Mean: {size_effect_cond.mean():.4f}")
print(f"  Std:  {size_effect_cond.std():.4f}")
print(f"  t-stat: {size_effect_cond.mean() / size_effect_cond.std() * np.sqrt(len(size_effect_cond)):.2f}")
print()

# Unconditional: size effect (not controlling for rating)
large_uncond = results_uncond.ret_vw[[labels[2], labels[5], labels[8]]].mean(axis=1)
small_uncond = results_uncond.ret_vw[[labels[0], labels[3], labels[6]]].mean(axis=1)
size_effect_uncond = large_uncond - small_uncond

print("UNCONDITIONAL size effect (cross-rating):")
print(f"  Mean: {size_effect_uncond.mean():.4f}")
print(f"  Std:  {size_effect_uncond.std():.4f}")
print(f"  t-stat: {size_effect_uncond.mean() / size_effect_uncond.std() * np.sqrt(len(size_effect_uncond)):.2f}")
print()

print("Difference:")
diff = size_effect_cond.mean() - size_effect_uncond.mean()
print(f"  Conditional - Unconditional: {diff:.4f}")
print()

if abs(diff) > 0.001:
    print("  → Conditional and unconditional effects differ!")
    print("    This occurs when characteristics are correlated.")
    print("    Conditional sort isolates the within-group effect.")
else:
    print("  → Conditional and unconditional effects are similar.")
    print("    This occurs when characteristics are uncorrelated.")

print()

# =============================================================================
# Portfolio Composition Differences
# =============================================================================

print("=" * 80)
print("Portfolio Composition")
print("=" * 80)
print()

print("With correlated characteristics, conditional sorting ensures:")
print("  - Each rating group has small, medium, and large bonds")
print("  - Size portfolios control for rating differences")
print()

print("Unconditional sorting may result in:")
print("  - Some combinations having few bonds (e.g., small + high rating)")
print("  - Size and rating effects mixing together")
print()

# =============================================================================
# When to Use Each Method
# =============================================================================

print("=" * 80)
print("When to Use Conditional vs Unconditional")
print("=" * 80)
print()

print("Use CONDITIONAL sorting when:")
print("  ✓ Characteristics are correlated")
print("  ✓ Want to isolate effect of second variable")
print("  ✓ Controlling for first variable is important")
print("  ✓ Example: Size effect controlling for rating")
print()

print("Use UNCONDITIONAL sorting when:")
print("  ✓ Characteristics are independent")
print("  ✓ Want to capture joint effects")
print("  ✓ Both variables are equally important")
print("  ✓ Example: Duration and convexity (often uncorrelated)")
print()

# =============================================================================
# Practical Example: Isolating Size Premium
# =============================================================================

print("=" * 80)
print("Practical Use Case: Size Premium Conditional on Rating")
print("=" * 80)
print()

print("Research question: Do larger bonds outperform within rating class?")
print()

print("Conditional sort isolates this effect:")
print()

for i in range(3):
    rating_label = ['Low', 'Mid', 'High'][i]
    large = results_cond.ret_vw[labels[i*3 + 2]]
    small = results_cond.ret_vw[labels[i*3]]
    premium = large - small

    print(f"  {rating_label} Rating group:")
    print(f"    Large - Small = {premium.mean():.4f} (t={premium.mean()/premium.std()*np.sqrt(len(premium)):.2f})")

print()

# =============================================================================
# Summary
# =============================================================================

print("=" * 80)
print("Summary")
print("=" * 80)
print()
print("This example demonstrated:")
print("  1. Conditional (sequential) double sorting")
print("  2. Difference between conditional and unconditional methods")
print("  3. When correlation matters for sorting choice")
print("  4. Isolating within-group effects")
print("  5. Practical application: size premium within rating class")
print()
print("Key insight:")
print("  When characteristics are correlated, conditional sorting controls")
print("  for the first variable and isolates the effect of the second.")
print()
print("Next steps:")
print("  - Compare both methods with your data")
print("  - Check characteristic correlation before choosing")
print("  - Use conditional when controlling for confounds")
print()
