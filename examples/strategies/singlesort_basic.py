"""
Basic Single Sort Example

Demonstrates single-variable portfolio sorting using PyBondLab.
Forms 5 portfolios sorted by credit rating with a 6-month holding period.

Key concepts:
- SingleSort strategy creation
- Basic StrategyFormation usage
- Accessing equal-weighted and value-weighted returns
- Computing high-minus-low spreads

Author: Giulio Rossetti
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import PyBondLab as pbl

# =============================================================================
# Generate Synthetic Data
# =============================================================================

def generate_bond_data(n_bonds=100, n_periods=60, seed=42):
    """
    Generate synthetic bond panel data for examples.

    Returns DataFrame with columns: date, ID, ret, RATING_NUM, BOND_VALUE
    """
    np.random.seed(seed)

    dates = pd.date_range('2010-01-31', periods=n_periods, freq='M')
    bond_ids = [f'BOND{i:04d}' for i in range(n_bonds)]

    data = []
    for bond_id in bond_ids:
        # Assign random but persistent rating (1-22)
        rating = np.random.randint(1, 23)

        # Bonds with worse ratings have higher average returns
        rating_premium = (rating - 10) * 0.001

        for date in dates:
            # Monthly return: base + rating premium + noise
            ret = 0.005 + rating_premium + np.random.normal(0, 0.02)

            # Market value fluctuates around $1M
            bond_value = 1_000_000 * np.exp(np.random.normal(0, 0.5))

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
# Create and Run Strategy
# =============================================================================

print("=" * 80)
print("Single Sort Example: Rating-Based Portfolios")
print("=" * 80)
print()

# Generate data
print("Generating synthetic bond data...")
data = generate_bond_data(n_bonds=100, n_periods=60)

print(f"  Data shape: {data.shape}")
print(f"  Date range: {data['date'].min()} to {data['date'].max()}")
print(f"  Number of bonds: {data['ID'].nunique()}")
print(f"  Rating range: {data['RATING_NUM'].min()}-{data['RATING_NUM'].max()}")
print()

# Define strategy
print("Creating SingleSort strategy...")
strategy = pbl.SingleSort(
    holding_period=6,           # Hold portfolios for 6 months
    sort_var='RATING_NUM',      # Sort by credit rating
    num_portfolios=5,           # Create 5 portfolios
    skip=0                      # No skip period
)
print(f"  Strategy: {strategy.strategy_name}")
print(f"  Holding period: {strategy.holding_period} months")
print(f"  Number of portfolios: {strategy.num_portfolios}")
print(f"  Sorting variable: {strategy.sort_var}")
print()

# Run portfolio formation
print("Running portfolio formation...")
results = pbl.StrategyFormation(
    data,
    strategy=strategy,
    turnover=False,              # Disable turnover tracking for speed
    verbose=False
).fit()

print("  Portfolio formation complete!")
print()

# =============================================================================
# Examine Results
# =============================================================================

print("=" * 80)
print("Results")
print("=" * 80)
print()

# Equal-weighted returns
print("Equal-Weighted Portfolio Returns:")
print(results.ret_ew.describe())
print()

# Value-weighted returns
print("Value-Weighted Portfolio Returns:")
print(results.ret_vw.describe())
print()

# High-minus-low spreads
print("High-Minus-Low Spreads:")
print(f"  EW HML mean: {results.hml_ew.mean():.4f}")
print(f"  EW HML std:  {results.hml_ew.std():.4f}")
print(f"  EW HML t-stat: {results.hml_ew.mean() / results.hml_ew.std() * np.sqrt(len(results.hml_ew)):.2f}")
print()
print(f"  VW HML mean: {results.hml_vw.mean():.4f}")
print(f"  VW HML std:  {results.hml_vw.std():.4f}")
print(f"  VW HML t-stat: {results.hml_vw.mean() / results.hml_vw.std() * np.sqrt(len(results.hml_vw)):.2f}")
print()

# Portfolio labels
print("Portfolio Labels:")
print(f"  {results.ret_ew.columns.tolist()}")
print()

# Time series length
print(f"Time series length: {len(results.ret_ew)} periods")
print()

# =============================================================================
# Accessing Specific Portfolios
# =============================================================================

print("=" * 80)
print("Accessing Individual Portfolios")
print("=" * 80)
print()

# Low rating portfolio (Q1)
q1_returns = results.ret_vw.iloc[:, 0]
print(f"Portfolio Q1 (Low rating):")
print(f"  Mean return: {q1_returns.mean():.4f}")
print(f"  Volatility: {q1_returns.std():.4f}")
print()

# High rating portfolio (Q5)
q5_returns = results.ret_vw.iloc[:, -1]
print(f"Portfolio Q5 (High rating):")
print(f"  Mean return: {q5_returns.mean():.4f}")
print(f"  Volatility: {q5_returns.std():.4f}")
print()

# High-minus-low
print(f"High-Minus-Low (Q5 - Q1):")
print(f"  Mean: {(q5_returns - q1_returns).mean():.4f}")
print(f"  Volatility: {(q5_returns - q1_returns).std():.4f}")
print()

# =============================================================================
# Summary
# =============================================================================

print("=" * 80)
print("Summary")
print("=" * 80)
print()
print("This example demonstrated:")
print("  1. Creating synthetic bond panel data")
print("  2. Defining a SingleSort strategy")
print("  3. Running portfolio formation with StrategyFormation")
print("  4. Accessing equal-weighted and value-weighted returns")
print("  5. Computing high-minus-low spreads and statistics")
print()
print("Next steps:")
print("  - See singlesort_breakpoints.py for custom breakpoint usage")
print("  - See ../filtering/ for data filtering examples")
print("  - See ../rebalancing/ for different rebalancing frequencies")
print()
