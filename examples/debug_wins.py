#!/usr/bin/env python
"""Debug wins filter differences."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/user/PyBondLab-Dev')

from PyBondLab.pbl_test import generate_synthetic_data

# Generate test data
data = generate_synthetic_data(n_dates=20, n_bonds=100, seed=42)
print(f"Data shape: {data.shape}")

# Get dates and sort
dates = data['date'].unique()
dates = np.sort(dates)
print(f"Dates: {len(dates)}")

# Build date_idx mapping
date_to_idx = {d: i for i, d in enumerate(dates)}
date_idx = data['date'].map(date_to_idx).values.astype(np.int64)
ret = data['ret'].values.astype(np.float64)

level = 95
location = 'both'

# Method 1: My fast path
print("\n--- Fast Path Ex-Ante Thresholds ---")
for d in range(min(5, len(dates))):
    hist_mask = date_idx < d
    if not np.any(hist_mask):
        print(f"Date {d}: No historical data")
        continue
    hist_ret = ret[hist_mask]
    hist_ret = hist_ret[~np.isnan(hist_ret)]
    if len(hist_ret) == 0:
        continue
    lb = np.nanpercentile(hist_ret, 100 - level)
    ub = np.nanpercentile(hist_ret, level)
    print(f"Date {d}: n_hist={len(hist_ret)}, lb={lb:.6f}, ub={ub:.6f}")

# Method 2: Slow path simulation
print("\n--- Slow Path Ex-Ante Thresholds ---")
data_sorted = data.sort_values(by='date').copy()
for i, current_date in enumerate(dates[:5]):
    pooled_data = data_sorted[data_sorted['date'] < current_date]['ret']
    pooled_data = pooled_data.dropna()
    if len(pooled_data) == 0:
        print(f"Date {i}: No historical data")
        continue
    lb = np.nanpercentile(pooled_data, 100 - level)
    ub = np.nanpercentile(pooled_data, level)
    print(f"Date {i}: n_hist={len(pooled_data)}, lb={lb:.6f}, ub={ub:.6f}")
