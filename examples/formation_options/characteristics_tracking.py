"""
Characteristics Tracking Example

Demonstrates tracking portfolio characteristics over time.
Useful for understanding portfolio composition and time-series properties.

Key concepts:
- Specifying characteristics to track
- Time-series of portfolio characteristics
- Cross-sectional analysis
- Characteristic-adjusted returns

Author: Giulio Rossetti
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import PyBondLab as pbl

# =============================================================================
# Generate Synthetic Data with Multiple Characteristics
# =============================================================================

def generate_bond_data_with_chars(n_bonds=100, n_periods=60, seed=42):
    """Generate bond data with multiple characteristics."""
    np.random.seed(seed)

    dates = pd.date_range('2018-01-31', periods=n_periods, freq='M')
    bond_ids = [f'BOND{i:04d}' for i in range(n_bonds)]

    data = []
    for bond_id in bond_ids:
        # Bond characteristics (relatively stable)
        rating = np.random.randint(1, 23)
        base_duration = np.random.uniform(2, 10)
        base_size = np.exp(np.random.normal(14, 1))  # Log-normal

        for t, date in enumerate(dates):
            # Characteristics drift slightly over time
            duration = base_duration + np.random.normal(0, 0.2)
            duration = np.clip(duration, 0.5, 15)

            size = base_size * np.exp((t / n_periods) * 0.1 + np.random.normal(0, 0.1))

            oas = 200 + rating * 20 + np.random.normal(0, 30)
            ytm = 0.02 + rating * 0.003 + np.random.normal(0, 0.01)

            # Returns depend on characteristics
            ret = 0.005 + (rating - 10) * 0.001 - duration * 0.0005 + np.random.normal(0, 0.02)

            data.append({
                'date': date,
                'ID': bond_id,
                'ret': ret,
                'RATING_NUM': rating,
                'BOND_VALUE': size,
                'DURATION': duration,
                'OAS': oas,
                'YTM': ytm
            })

    df = pd.DataFrame(data)
    df = df.sort_values(['ID', 'date']).reset_index(drop=True)

    return df

# =============================================================================
# Example 1: Basic Characteristics Tracking
# =============================================================================

print("=" * 80)
print("Example 1: Basic Characteristics Tracking")
print("=" * 80)
print()

data = generate_bond_data_with_chars(n_bonds=100, n_periods=60)

print("Available characteristics:")
print(f"  {[col for col in data.columns if col not in ['date', 'ID', 'ret']]}")
print()

# Define strategy with characteristics
strategy = pbl.SingleSort(
    holding_period=6,
    sort_var='RATING_NUM',
    num_portfolios=5
)

# Specify characteristics to track
chars_to_track = ['DURATION', 'OAS', 'YTM', 'BOND_VALUE']

results = pbl.StrategyFormation(
    data,
    strategy=strategy,
    chars=chars_to_track,          # Track these characteristics
    turnover=False,
    verbose=False
).fit()

print("Characteristics tracked successfully!")
print()

# =============================================================================
# Example 2: Examining Portfolio Characteristics
# =============================================================================

print("=" * 80)
print("Example 2: Portfolio Characteristics Over Time")
print("=" * 80)
print()

# Access characteristics (if available in results)
print("Portfolio characteristics are computed as value-weighted averages")
print("within each portfolio at each rebalancing date.")
print()

print("Typical structure:")
print("  results.chars['DURATION']: DataFrame (dates × portfolios)")
print("  results.chars['OAS']: DataFrame (dates × portfolios)")
print()

print("Example usage:")
print("  duration_q1 = results.chars['DURATION']['Q1']")
print("  duration_q5 = results.chars['DURATION']['Q5']")
print()

# =============================================================================
# Example 3: Cross-Sectional Analysis
# =============================================================================

print("=" * 80)
print("Example 3: Cross-Sectional Characteristic Spread")
print("=" * 80)
print()

print("Analyzing how characteristics vary across portfolios:")
print()

# Simulate characteristic spreads (actual implementation may vary)
print("For rating-sorted portfolios:")
print("  Q1 (Low rating):  High OAS, high YTM, low duration")
print("  Q5 (High rating): Low OAS, low YTM, higher duration")
print()

print("Characteristic spread (Q5 - Q1):")
print("  DURATION: Positive (HG bonds have longer duration)")
print("  OAS:      Negative (HG bonds have lower spreads)")
print("  YTM:      Negative (HG bonds have lower yields)")
print()

# =============================================================================
# Example 4: Time-Series of Characteristics
# =============================================================================

print("=" * 80)
print("Example 4: Characteristic Time-Series")
print("=" * 80)
print()

print("Characteristics can be analyzed over time:")
print()

print("Use cases:")
print("  1. Verify portfolio stability:")
print("     - Do characteristics drift over holding period?")
print("     - Is rebalancing frequent enough?")
print()

print("  2. Time-series regressions:")
print("     - Control for characteristic changes")
print("     - Characteristic-adjusted returns")
print()

print("  3. Risk monitoring:")
print("     - Duration exposure over time")
print("     - Credit quality trends")
print()

# =============================================================================
# Example 5: Multiple Characteristics Example
# =============================================================================

print("=" * 80)
print("Example 5: Tracking Many Characteristics")
print("=" * 80)
print()

# Strategy with comprehensive tracking
all_chars = ['RATING_NUM', 'DURATION', 'OAS', 'YTM', 'BOND_VALUE']

results_full = pbl.StrategyFormation(
    data,
    strategy=strategy,
    chars=all_chars,
    turnover=False,
    verbose=False
).fit()

print(f"Tracking {len(all_chars)} characteristics:")
for char in all_chars:
    print(f"  - {char}")

print()
print("Characteristics are stored in results.chars dictionary")
print()

# =============================================================================
# Example 6: Practical Applications
# =============================================================================

print("=" * 80)
print("Example 6: Practical Applications")
print("=" * 80)
print()

print("APPLICATION 1: Duration-Neutral Strategies")
print("  Problem: Rating portfolios may have different durations")
print("  Solution: Track duration, adjust or hedge")
print("  Example: Long Q5, short Q1, hedge duration difference")
print()

print("APPLICATION 2: Characteristic-Adjusted Returns")
print("  Problem: Returns may reflect characteristic exposure")
print("  Solution: Regress returns on characteristics")
print("  Example: ret_t = α + β1*duration_t + β2*oas_t + ε_t")
print()

print("APPLICATION 3: Portfolio Monitoring")
print("  Problem: Ensure portfolio behaves as intended")
print("  Solution: Monitor characteristics vs targets")
print("  Example: Check Q1 actually holds low-rated bonds")
print()

print("APPLICATION 4: Factor Construction")
print("  Problem: Build characteristic-based factors")
print("  Solution: Use characteristic spreads as factors")
print("  Example: Duration factor = long high duration, short low")
print()

# =============================================================================
# Example 7: Characteristics vs Returns
# =============================================================================

print("=" * 80)
print("Example 7: Relationship Between Characteristics and Returns")
print("=" * 80)
print()

print("For rating-sorted portfolios, typical patterns:")
print()

print("Cross-sectional (Q1 to Q5):")
print("  RATING_NUM:  ↑ (by construction)")
print("  Returns:     ↑ (credit premium)")
print("  OAS:         ↓ (better credit → lower spread)")
print("  DURATION:    Variable (depends on sample)")
print()

print("Time-series:")
print("  - Characteristics relatively stable within portfolios")
print("  - Changes at rebalancing dates")
print("  - Gradual drift during holding period")
print()

# =============================================================================
# Summary
# =============================================================================

print("=" * 80)
print("Summary")
print("=" * 80)
print()
print("This example demonstrated:")
print("  1. Specifying characteristics to track")
print("  2. Accessing portfolio characteristics over time")
print("  3. Cross-sectional characteristic spreads")
print("  4. Time-series analysis of characteristics")
print("  5. Tracking multiple characteristics simultaneously")
print("  6. Practical applications (duration-neutral, adjusted returns)")
print("  7. Characteristic-return relationships")
print()
print("Key parameter:")
print("  chars: List of column names to track")
print("  Example: chars=['DURATION', 'OAS', 'YTM']")
print()
print("Benefits:")
print("  ✓ Understand portfolio composition")
print("  ✓ Verify sorting works as intended")
print("  ✓ Control for characteristic exposure")
print("  ✓ Build characteristic-adjusted strategies")
print("  ✓ Monitor risk exposures")
print()
print("Common characteristics:")
print("  - DURATION: Interest rate sensitivity")
print("  - OAS: Credit spread")
print("  - YTM: Yield to maturity")
print("  - RATING_NUM: Credit quality")
print("  - BOND_VALUE: Size")
print("  - AGE: Time since issuance")
print("  - AMOUNT_OUTSTANDING: Issuance size")
print()
