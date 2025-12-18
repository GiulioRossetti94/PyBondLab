#!/usr/bin/env python
"""Debug wins filter factor differences."""
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/user/PyBondLab-Dev')

import PyBondLab as pbl
from PyBondLab import DataUncertaintyAnalysis, Momentum
from PyBondLab.pbl_test import generate_synthetic_data

# Generate test data
np.random.seed(42)
data = generate_synthetic_data(n_dates=60, n_bonds=300, seed=42)
print(f"Data shape: {data.shape}")

# Create Momentum strategy
mom = Momentum(
    holding_period=1,
    num_portfolios=5,
    lookback_period=3,
    skip=1,
    verbose=False
)

# Test single wins filter (95)
filters = {
    'wins': [(95, 'both')],
}

# Run slow path
print("\n--- SLOW PATH ---")
results_slow = DataUncertaintyAnalysis(
    data=data,
    strategy=mom,
    holding_periods=[1],
    num_portfolios=5,
    filters=filters,
    include_baseline=True,
    use_fast_path=False,
    verbose=True
).fit()

# Run fast path
print("\n--- FAST PATH ---")
results_fast = DataUncertaintyAnalysis(
    data=data,
    strategy=mom,
    holding_periods=[1],
    num_portfolios=5,
    filters=filters,
    include_baseline=True,
    use_fast_path=True,
    verbose=True
).fit()

# Compare
print("\n--- Comparison ---")
print("Slow path:")
print(results_slow.summary()[['signal', 'hp', 'filter_type', 'level', 'ew_ea_mean', 'vw_ea_mean']])

print("\nFast path:")
print(results_fast.summary()[['signal', 'hp', 'filter_type', 'level', 'ew_ea_mean', 'vw_ea_mean']])

# Detailed comparison
for col in results_slow.ew_ea.columns:
    slow_vals = results_slow.ew_ea[col].dropna().values
    # Find matching fast column
    for fast_col in results_fast.ew_ea.columns:
        if 'wins' in col and 'wins' in fast_col:
            fast_vals = results_fast.ew_ea[fast_col].dropna().values
            min_len = min(len(slow_vals), len(fast_vals))
            if min_len > 0:
                diff = np.abs(slow_vals[:min_len] - fast_vals[:min_len])
                print(f"\n{col} vs {fast_col}:")
                print(f"  Max diff: {diff.max():.6f}")
                print(f"  Mean diff: {diff.mean():.6f}")
                print(f"  Slow mean: {slow_vals.mean():.6f}")
                print(f"  Fast mean: {fast_vals.mean():.6f}")
