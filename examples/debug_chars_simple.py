#!/usr/bin/env python
"""
Simple debug script for characteristics comparison.
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '.')

import PyBondLab as pbl
from PyBondLab.pbl_test import generate_synthetic_data_fast


def main():
    # Match validation script exactly
    data = generate_synthetic_data_fast(
        n_dates=120,
        n_bonds=300,
        seed=42,
        balanced_panel=False,
        n_chars=3,
    )

    print(f"Data shape: {data.shape}")

    # Create strategy and config
    strategy = pbl.SingleSort(
        holding_period=12,
        sort_var='signal1',
        num_portfolios=5,
        rebalance_frequency='annual',
        rebalance_month=6,
    )

    from PyBondLab.config import StrategyFormationConfig, DataConfig, FormationConfig

    config = StrategyFormationConfig(
        data=DataConfig(chars=['char1', 'char2']),
        formation=FormationConfig(
            compute_turnover=False,
            verbose=False,
        )
    )

    # Run slow path
    print("\nRunning slow path...")
    sf_slow = pbl.StrategyFormation(
        data=data,
        strategy=strategy,
        config=config,
        verbose=False,
    )
    sf_slow._can_use_nonstaggered_fast_path = lambda: False
    slow_result = sf_slow.fit()

    # Run fast path
    print("Running fast path...")
    sf_fast = pbl.StrategyFormation(
        data=data,
        strategy=strategy,
        config=config,
        verbose=False,
    )
    fast_result = sf_fast.fit()

    # Compare
    slow_chars = slow_result.get_characteristics()
    fast_chars = fast_result.get_characteristics()

    slow_ew_c, slow_vw_c = slow_chars
    fast_ew_c, fast_vw_c = fast_chars

    # Find where they differ most
    c = 'char1'
    slow_ew = slow_ew_c[c].values
    fast_ew = fast_ew_c[c].values

    diff = np.abs(slow_ew - fast_ew)
    max_idx = np.unravel_index(np.nanargmax(diff), diff.shape)
    max_diff = np.nanmax(diff)

    dates = slow_ew_c[c].index

    print(f"\nMax EW diff: {max_diff:.6f} at date={dates[max_idx[0]]}, portfolio={max_idx[1]+1}")

    # Print a few rows around the max diff
    print(f"\nValues around max diff (date index {max_idx[0]}):")
    print(f"{'Date':<15} {'Slow EW':<12} {'Fast EW':<12} {'Diff':<12}")
    print("-" * 60)

    for i in range(max(0, max_idx[0]-2), min(len(dates), max_idx[0]+3)):
        date = dates[i]
        slow_val = slow_ew[i, max_idx[1]]
        fast_val = fast_ew[i, max_idx[1]]
        d = slow_val - fast_val if not (np.isnan(slow_val) or np.isnan(fast_val)) else np.nan

        print(f"{str(date)[:10]:<15} {slow_val:.6f} {fast_val:.6f} {d:.6f}" if not np.isnan(slow_val) and not np.isnan(fast_val) else f"{str(date)[:10]:<15} NaN")

    # Check if the difference is related to ID intersection
    print("\n--- Checking at specific dates ---")
    for d_idx in [10, 20, 30, 40]:
        if d_idx < len(dates):
            slow_vals = slow_ew[d_idx, :]
            fast_vals = fast_ew[d_idx, :]
            d = np.abs(slow_vals - fast_vals)
            print(f"Date {dates[d_idx]}: max_diff={np.nanmax(d):.4e}")


if __name__ == '__main__':
    main()
