#!/usr/bin/env python
"""
Debug script for Phase 15b characteristics computation.
Compares slow path vs fast path at each step.
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '.')

import PyBondLab as pbl
from PyBondLab.pbl_test import generate_synthetic_data_fast


def main():
    print("=" * 80)
    print("DEBUG: Phase 15b Characteristics Computation")
    print("=" * 80)

    # Generate small test data with balanced panel for easier debugging
    data = generate_synthetic_data_fast(
        n_dates=24,  # 2 years
        n_bonds=50,
        seed=42,
        balanced_panel=True,  # Full panel
        n_chars=2,
    )
    print(f"\nData shape: {data.shape}")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"Unique dates: {data['date'].nunique()}")

    # Print date index mapping
    dates = sorted(data['date'].unique())
    print("\nDate index mapping:")
    for i, d in enumerate(dates[:8]):
        print(f"  {i}: {d}")

    # Test annual rebalancing in June
    hp = 12
    rebal_freq = 'annual'
    rebal_month = 6
    chars = ['char1', 'char2']

    print(f"\nConfig: hp={hp}, freq={rebal_freq}, month={rebal_month}, chars={chars}")

    # Find rebalancing dates
    rebal_dates = [d for d in dates if d.month == rebal_month]
    print(f"\nRebalancing dates: {rebal_dates}")

    # Check characteristic values at key dates
    if len(rebal_dates) > 0:
        rebal_date = rebal_dates[0]  # June 2018
        rebal_idx = dates.index(rebal_date)
        return_idx = rebal_idx + 1  # July 2018
        char_source_idx = return_idx - 1  # June (d-1 of July)

        print(f"\nFormation date: {rebal_date} (index {rebal_idx})")
        print(f"First return date: {dates[return_idx]} (index {return_idx})")
        print(f"Char source for first return (d-1): {dates[char_source_idx]} (index {char_source_idx})")

        # Get char values at formation (June) - this is what should be used for July return
        char_at_formation = data[data['date'] == dates[char_source_idx]][['ID', 'char1']].sort_values('ID')
        print(f"\nChar1 at June (first 5 bonds):")
        print(char_at_formation.head())

        # Get char values at July - this should NOT be used for July return
        char_at_return = data[data['date'] == dates[return_idx]][['ID', 'char1']].sort_values('ID')
        print(f"\nChar1 at July (first 5 bonds):")
        print(char_at_return.head())

    # Create strategy and config
    strategy = pbl.SingleSort(
        holding_period=hp,
        sort_var='signal1',
        num_portfolios=5,
        rebalance_frequency=rebal_freq,
        rebalance_month=rebal_month,
    )

    from PyBondLab.config import StrategyFormationConfig, DataConfig, FormationConfig

    config = StrategyFormationConfig(
        data=DataConfig(chars=chars),
        formation=FormationConfig(
            compute_turnover=False,
            verbose=False,
        )
    )

    # Run slow path
    print("\n=== Running SLOW path ===")
    sf_slow = pbl.StrategyFormation(
        data=data,
        strategy=strategy,
        config=config,
        verbose=False,
    )
    sf_slow._can_use_nonstaggered_fast_path = lambda: False
    slow_result = sf_slow.fit()

    # Run fast path
    print("=== Running FAST path ===")
    sf_fast = pbl.StrategyFormation(
        data=data,
        strategy=strategy,
        config=config,
        verbose=False,
    )
    fast_result = sf_fast.fit()

    # Compare characteristics
    slow_chars = slow_result.get_characteristics()
    fast_chars = fast_result.get_characteristics()

    slow_ew_c, slow_vw_c = slow_chars
    fast_ew_c, fast_vw_c = fast_chars

    print("\n" + "-" * 80)
    print("Characteristics Comparison (char1, Portfolio 1)")
    print("-" * 80)

    c = 'char1'
    slow_ew_vals = slow_ew_c[c].iloc[:, 0].values
    fast_ew_vals = fast_ew_c[c].iloc[:, 0].values
    dates_out = slow_ew_c[c].index

    print(f"{'Date':<15} {'Slow EW':<12} {'Fast EW':<12} {'Diff EW':<12}")
    print("-" * 55)

    for i in range(min(15, len(dates_out))):
        date = dates_out[i]
        slow_ew = slow_ew_vals[i]
        fast_ew = fast_ew_vals[i]
        diff_ew = slow_ew - fast_ew if not (np.isnan(slow_ew) or np.isnan(fast_ew)) else np.nan

        date_str = str(date)[:10]
        slow_ew_str = f"{slow_ew:.4f}" if not np.isnan(slow_ew) else "NaN"
        fast_ew_str = f"{fast_ew:.4f}" if not np.isnan(fast_ew) else "NaN"
        diff_ew_str = f"{diff_ew:.4f}" if not np.isnan(diff_ew) else "NaN"

        print(f"{date_str:<15} {slow_ew_str:<12} {fast_ew_str:<12} {diff_ew_str:<12}")

    # Check if fast path values are shifted by 1
    print("\n" + "-" * 80)
    print("Checking for 1-month shift...")
    print("-" * 80)

    # Compare slow[i] with fast[i+1]
    shift_matches = 0
    for i in range(len(slow_ew_vals) - 1):
        if not np.isnan(slow_ew_vals[i]) and not np.isnan(fast_ew_vals[i+1]):
            if abs(slow_ew_vals[i] - fast_ew_vals[i+1]) < 1e-10:
                shift_matches += 1
                print(f"  Match: slow[{i}]={slow_ew_vals[i]:.4f} == fast[{i+1}]={fast_ew_vals[i+1]:.4f}")

    print(f"\nTotal matches with 1-month shift: {shift_matches}")

    # Summary
    print("\n" + "-" * 80)
    print("Summary Statistics")
    print("-" * 80)

    for c in chars:
        for p in range(5):
            slow_ew_vals = slow_ew_c[c].iloc[:, p].values
            fast_ew_vals = fast_ew_c[c].iloc[:, p].values

            valid_ew = ~np.isnan(slow_ew_vals) & ~np.isnan(fast_ew_vals)

            if valid_ew.sum() > 0:
                diff_ew = np.abs(slow_ew_vals[valid_ew] - fast_ew_vals[valid_ew])
                print(f"{c} Portfolio {p+1}: EW max_diff={diff_ew.max():.4e}")


if __name__ == '__main__':
    main()
