"""
Diagnostic script to investigate VW EP NaN issue with wins filter.

This script helps identify WHY VW EP has NaN values when using aggressive
winsorization with NIG bonds and HP=3.

Run: python examples/diagnose_wins_nan.py
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '.')

from PyBondLab import DataUncertaintyAnalysis, Momentum
from PyBondLab.pbl_test import generate_synthetic_data_fast

# =============================================================================
# Generate synthetic data with realistic NIG-like characteristics
# =============================================================================
print("Generating synthetic data...")
np.random.seed(42)

# Generate base data
data = generate_synthetic_data_fast(
    n_dates=120,  # 10 years monthly
    n_bonds=800,
    seed=42,
    balanced_panel=False,  # More realistic with entry/exit
    pct_active_low=0.75,
    pct_active_high=0.95,
)

# Make ~40% NIG bonds (rating 11-22)
n_obs = len(data)
rating_values = np.random.choice([5, 8, 12, 15, 18], size=n_obs, p=[0.2, 0.2, 0.2, 0.2, 0.2])
data['RATING_NUM'] = rating_values

# Add more volatility to NIG bonds
nig_mask = data['RATING_NUM'] >= 11
data.loc[nig_mask, 'ret'] = data.loc[nig_mask, 'ret'] * 1.5  # More volatile

print(f"Data shape: {data.shape}")
print(f"NIG bonds: {nig_mask.sum()} / {len(data)} ({100*nig_mask.mean():.1f}%)")
print(f"Date range: {data['date'].min()} to {data['date'].max()}")

# =============================================================================
# Run analysis with various wins levels
# =============================================================================
wins_levels = [90, 70, 50, 30, 10]
wins_right = [(level, 'right') for level in wins_levels]

print("\n" + "="*60)
print("Running DataUncertaintyAnalysis with wins_right filters...")
print("="*60)

mom = Momentum(lookback_period=3, skip=1)
dua = DataUncertaintyAnalysis(
    data=data,
    strategy=mom,
    holding_periods=[3],  # HP=3
    num_portfolios=5,
    filters={'wins': wins_right},
    include_baseline=True,
    rating='NIG',  # Non-investment grade only
    dynamic_weights=True,
    verbose=True,
)

result = dua.fit()

# =============================================================================
# Analyze NaN patterns
# =============================================================================
print("\n" + "="*60)
print("NaN Analysis")
print("="*60)

# Check each component (using correct property names)
for col_type, df_attr in [('EW EA', 'ew_ex_ante'), ('VW EA', 'vw_ex_ante'),
                           ('EW EP', 'ew_ex_post'), ('VW EP', 'vw_ex_post')]:
    df = getattr(result, df_attr)
    print(f"\n{col_type}:")
    for col in df.columns:
        nan_count = df[col].isnull().sum()
        if nan_count > 0:
            print(f"  {col}: {nan_count} NaNs")

# =============================================================================
# Deep dive into VW EP NaN dates
# =============================================================================
print("\n" + "="*60)
print("VW EP NaN Deep Dive")
print("="*60)

vw_ep = result.vw_ex_post

# Find the most aggressive wins column
wins_10_cols = [c for c in vw_ep.columns if 'wins_10' in c]
if wins_10_cols:
    col = wins_10_cols[0]
    nan_dates = vw_ep[vw_ep[col].isnull()].index.tolist()
    print(f"\nNaN dates for {col}: {len(nan_dates)}")
    if nan_dates:
        print(f"First 5 NaN dates: {nan_dates[:5]}")
        print(f"Last 5 NaN dates: {nan_dates[-5:]}")

# =============================================================================
# Check long and short legs separately
# =============================================================================
print("\n" + "="*60)
print("Checking Long and Short Legs Separately")
print("="*60)

# Access leg data
vw_long = result.vw_long_ex_post
vw_short = result.vw_short_ex_post

for col in vw_ep.columns:
    if 'wins' in col:
        ls_nan = vw_ep[col].isnull().sum()
        long_nan = vw_long[col].isnull().sum() if col in vw_long.columns else 'N/A'
        short_nan = vw_short[col].isnull().sum() if col in vw_short.columns else 'N/A'
        print(f"{col}:")
        print(f"  L-S NaN: {ls_nan}, Long NaN: {long_nan}, Short NaN: {short_nan}")

# =============================================================================
# Compare with baseline (no wins filter)
# =============================================================================
print("\n" + "="*60)
print("Baseline Comparison")
print("="*60)

baseline_cols = [c for c in vw_ep.columns if 'baseline' in c]
if baseline_cols:
    col = baseline_cols[0]
    print(f"\nBaseline {col}:")
    print(f"  NaN count: {vw_ep[col].isnull().sum()}")
    print(f"  Mean: {vw_ep[col].mean():.4f}")
    print(f"  Std: {vw_ep[col].std():.4f}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*60)
print("DIAGNOSTIC SUMMARY")
print("="*60)
print("""
Possible causes for VW EP NaN with aggressive wins:

1. SIGNAL COMPRESSION
   - Aggressive wins (e.g., wins_10_right) clips ~90% of returns
   - Momentum signal becomes very compressed (little variation)
   - Ranking becomes unstable

2. THIN PORTFOLIOS
   - NIG universe is smaller than ALL bonds
   - Extreme portfolios (P1, P5) have fewest bonds
   - With compressed signals, portfolio assignment is less stable

3. VW MISMATCH
   - VW looked up from d-1 (dynamic_weights=True)
   - If bond exists at d but not at d-1, no valid VW
   - ALL bonds in P1 or P5 may lack valid VW on some dates

4. COHORT AGGREGATION
   - HP=3 requires valid data across 3 cohorts
   - If any cohort's P1 or P5 has no valid VW bonds, cohort VW = NaN
   - If ALL cohorts have NaN for a portfolio, averaged VW = NaN
   - L-S requires BOTH P1 and P5 to be non-NaN

RECOMMENDATIONS:
- Check if EW EP has fewer NaNs (should, since it doesn't require VW)
- Compare with dynamic_weights=False (uses formation date VW)
- Check portfolio sizes at NaN dates
- Consider less aggressive wins levels (e.g., 95, 99)
""")
