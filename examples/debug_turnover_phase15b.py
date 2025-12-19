#!/usr/bin/env python
"""
Debug script for Phase 15b turnover computation.
Compares slow path vs fast path at each step.
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '.')

import PyBondLab as pbl
from PyBondLab.pbl_test import generate_synthetic_data_fast


def run_slow_path_with_debug(data, hp, rebal_freq, rebal_month):
    """Run slow path and capture turnover at each date."""
    strategy = pbl.SingleSort(
        holding_period=hp,
        sort_var='signal1',
        num_portfolios=5,
        rebalance_frequency=rebal_freq,
        rebalance_month=rebal_month,
    )

    from PyBondLab.config import StrategyFormationConfig, DataConfig, FormationConfig

    config = StrategyFormationConfig(
        data=DataConfig(),
        formation=FormationConfig(
            compute_turnover=True,
            verbose=False,
        )
    )

    sf = pbl.StrategyFormation(
        data=data,
        strategy=strategy,
        turnover=True,
        config=config,
        verbose=False,
    )

    # Force slow path
    original_can_use = sf._can_use_nonstaggered_fast_path
    sf._can_use_nonstaggered_fast_path = lambda: False

    result = sf.fit()

    # Get turnover from result (consistent with fast path)
    ew_turn_result, vw_turn_result = result.get_turnover()

    TM = len(sf.datelist)
    nport = 5

    # Convert to array format (dates x portfolios)
    ew_turn = np.full((TM, nport), np.nan)
    vw_turn = np.full((TM, nport), np.nan)

    if ew_turn_result is not None:
        for d_idx, date in enumerate(sf.datelist):
            if date in ew_turn_result.index:
                ew_turn[d_idx, :] = ew_turn_result.loc[date].values
                vw_turn[d_idx, :] = vw_turn_result.loc[date].values

    return result, ew_turn, vw_turn, sf.datelist


def run_fast_path_with_debug(data, hp, rebal_freq, rebal_month):
    """Run fast path and capture turnover at each date."""
    strategy = pbl.SingleSort(
        holding_period=hp,
        sort_var='signal1',
        num_portfolios=5,
        rebalance_frequency=rebal_freq,
        rebalance_month=rebal_month,
    )

    from PyBondLab.config import StrategyFormationConfig, DataConfig, FormationConfig

    config = StrategyFormationConfig(
        data=DataConfig(),
        formation=FormationConfig(
            compute_turnover=True,
            verbose=False,
        )
    )

    sf = pbl.StrategyFormation(
        data=data,
        strategy=strategy,
        turnover=True,
        config=config,
        verbose=False,
    )

    result = sf.fit()

    # Fast path uses result turnover directly, not turnover_state
    # Create turnover arrays from the result
    ew_turn_result, vw_turn_result = result.get_turnover()

    TM = len(sf.datelist)
    nport = 5

    # Convert to array format (dates x portfolios)
    ew_turn = np.full((TM, nport), np.nan)
    vw_turn = np.full((TM, nport), np.nan)

    if ew_turn_result is not None:
        for d_idx, date in enumerate(sf.datelist):
            if date in ew_turn_result.index:
                ew_turn[d_idx, :] = ew_turn_result.loc[date].values
                vw_turn[d_idx, :] = vw_turn_result.loc[date].values

    return result, ew_turn, vw_turn, sf.datelist


def main():
    print("=" * 80)
    print("DEBUG: Phase 15b Turnover Computation")
    print("=" * 80)

    # Generate small test data
    data = generate_synthetic_data_fast(
        n_dates=24,  # 2 years
        n_bonds=50,
        seed=42,
        balanced_panel=True,  # Full panel for easier debugging
    )
    print(f"\nData shape: {data.shape}")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")

    # Test annual rebalancing in June
    hp = 12
    rebal_freq = 'annual'
    rebal_month = 6

    print(f"\nConfig: hp={hp}, freq={rebal_freq}, month={rebal_month}")

    # Run slow path
    print("\nRunning slow path...")
    slow_result, slow_ew_turn, slow_vw_turn, datelist = run_slow_path_with_debug(
        data, hp, rebal_freq, rebal_month
    )

    # Run fast path
    print("Running fast path...")
    fast_result, fast_ew_turn, fast_vw_turn, _ = run_fast_path_with_debug(
        data, hp, rebal_freq, rebal_month
    )

    # Compare turnover at each date/portfolio
    print("\n" + "-" * 80)
    print("Turnover Comparison (EW, Portfolio 1)")
    print("-" * 80)
    print(f"{'Date':<15} {'Slow':<15} {'Fast':<15} {'Diff':<15}")
    print("-" * 60)

    for t_idx in range(len(datelist)):
        slow_val = slow_ew_turn[t_idx, 0]
        fast_val = fast_ew_turn[t_idx, 0]
        diff = slow_val - fast_val if not (np.isnan(slow_val) or np.isnan(fast_val)) else np.nan

        date_str = str(datelist[t_idx])[:10]
        slow_str = f"{slow_val:.6f}" if not np.isnan(slow_val) else "NaN"
        fast_str = f"{fast_val:.6f}" if not np.isnan(fast_val) else "NaN"
        diff_str = f"{diff:.6f}" if not np.isnan(diff) else "NaN"

        print(f"{date_str:<15} {slow_str:<15} {fast_str:<15} {diff_str:<15}")

    # Summary statistics
    print("\n" + "-" * 80)
    print("Summary Statistics")
    print("-" * 80)

    # Compare non-NaN values
    for ptf in range(5):
        slow_vals = slow_ew_turn[:, ptf]
        fast_vals = fast_ew_turn[:, ptf]

        valid = ~np.isnan(slow_vals) & ~np.isnan(fast_vals)
        if valid.sum() > 0:
            diff = np.abs(slow_vals[valid] - fast_vals[valid])
            print(f"Portfolio {ptf+1}: max_diff={diff.max():.6e}, mean_diff={diff.mean():.6e}, n={valid.sum()}")

    # Get long-short turnover for comparison
    slow_turn = slow_result.get_turnover()
    fast_turn = fast_result.get_turnover()

    if slow_turn is not None and fast_turn is not None:
        slow_ew_t, slow_vw_t = slow_turn
        fast_ew_t, fast_vw_t = fast_turn

        print(f"\nLong-Short Turnover (EW):")
        print(f"  Slow mean: {slow_ew_t.mean():.6f}")
        print(f"  Fast mean: {fast_ew_t.mean():.6f}")

        # Align and compare
        common_idx = slow_ew_t.index.intersection(fast_ew_t.index)
        if len(common_idx) > 0:
            diff = np.abs(slow_ew_t.loc[common_idx].values - fast_ew_t.loc[common_idx].values)
            print(f"  Max diff: {diff.max():.6e}")


if __name__ == '__main__':
    main()
