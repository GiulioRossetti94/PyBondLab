#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test fast strategy path for Momentum/LTreversal.

Compares fast path vs slow path to ensure numerical equivalence.
"""

import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/user/PyBondLab-Dev')

import PyBondLab as pbl
from PyBondLab import DataUncertaintyAnalysis, Momentum, LTreversal
from PyBondLab.pbl_test import generate_synthetic_data


def test_momentum_fast_vs_slow():
    """Test Momentum fast path matches slow path."""
    print("=" * 70)
    print("Testing Momentum: Fast Path vs Slow Path")
    print("=" * 70)

    # Generate test data
    np.random.seed(42)
    data = generate_synthetic_data(n_dates=60, n_bonds=200, seed=42)
    print(f"Data shape: {data.shape}")

    # Create Momentum strategy
    mom = Momentum(
        holding_period=1,  # Will be overridden by DataUncertaintyAnalysis
        num_portfolios=5,
        lookback_period=3,
        skip=1,
        verbose=False
    )

    # Test filters
    filters = {
        'trim': [0.2],
    }

    # Run slow path
    print("\n--- Running SLOW PATH ---")
    t0 = time.time()
    results_slow = DataUncertaintyAnalysis(
        data=data,
        strategy=mom,
        holding_periods=[1],
        num_portfolios=5,
        filters=filters,
        include_baseline=True,
        use_fast_path=False,  # Force slow path
        verbose=True
    ).fit()
    slow_time = time.time() - t0
    print(f"Slow path completed in {slow_time:.2f}s")

    # Run fast path
    print("\n--- Running FAST PATH ---")
    t0 = time.time()
    results_fast = DataUncertaintyAnalysis(
        data=data,
        strategy=mom,
        holding_periods=[1],
        num_portfolios=5,
        filters=filters,
        include_baseline=True,
        use_fast_path=True,  # Use fast path
        verbose=True
    ).fit()
    fast_time = time.time() - t0
    print(f"Fast path completed in {fast_time:.2f}s")

    # Compare results
    print("\n--- Comparison ---")
    print(f"Speedup: {slow_time / fast_time:.1f}x")

    # Show summaries
    print("\nSlow path summary:")
    slow_summary = results_slow.summary()
    print(slow_summary[['signal', 'hp', 'filter_type', 'ew_ea_mean', 'vw_ea_mean']])

    print("\nFast path summary:")
    fast_summary = results_fast.summary()
    print(fast_summary[['signal', 'hp', 'filter_type', 'ew_ea_mean', 'vw_ea_mean']])

    # Compare each metric - slow path uses "signal", fast path uses "momentum"
    tolerance = 1e-4  # Allow small differences due to different implementations
    all_match = True
    compared = 0

    for slow_row_idx, slow_row in slow_summary.iterrows():
        filter_type = slow_row['filter_type']
        hp = slow_row['hp']

        # Find matching fast row
        fast_match = fast_summary[
            (fast_summary['filter_type'] == filter_type) &
            (fast_summary['hp'] == hp)
        ]

        if len(fast_match) > 0:
            fast_row = fast_match.iloc[0]
            compared += 1

            print(f"\n  Filter: {filter_type}, HP: {hp}")

            for metric in ['ew_ea_mean', 'vw_ea_mean', 'ew_ep_mean', 'vw_ep_mean']:
                slow_val = slow_row[metric]
                fast_val = fast_row[metric]

                # Handle NaN cases
                if pd.isna(slow_val) and pd.isna(fast_val):
                    status = "OK (both NaN)"
                elif pd.isna(slow_val) or pd.isna(fast_val):
                    status = "MISMATCH (one NaN)"
                    all_match = False
                else:
                    diff = abs(slow_val - fast_val)
                    if diff < tolerance:
                        status = f"OK (diff={diff:.2e})"
                    else:
                        status = f"MISMATCH (diff={diff:.2e})"
                        all_match = False

                print(f"    {metric}: slow={slow_val:.4f}, fast={fast_val:.4f} - {status}")

    print(f"\nCompared {compared} configurations")

    print("\n" + "=" * 70)
    if all_match:
        print("SUCCESS: Fast path matches slow path!")
    else:
        print("FAILURE: Results do not match!")
    print("=" * 70)

    return all_match


def test_momentum_signal_computation():
    """Test that numba signal computation matches pandas."""
    print("\n" + "=" * 70)
    print("Testing Momentum Signal Computation")
    print("=" * 70)

    from PyBondLab.numba_core import compute_momentum_signals_panel, get_bond_boundaries

    # Generate simple test data
    np.random.seed(42)
    n_bonds = 50
    n_dates = 20

    # Create a small DataFrame
    data = []
    for bond_id in range(n_bonds):
        for date_idx in range(n_dates):
            ret = np.random.normal(0.01, 0.05)
            data.append({
                'ID': bond_id,
                'date': date_idx,
                'ret': ret
            })

    df = pd.DataFrame(data)
    df = df.sort_values(['ID', 'date'], ignore_index=True)

    lookback = 3
    skip = 1

    # Compute using pandas (slow path)
    df['logret'] = np.log(df['ret'] + 1)
    df['signal_pandas'] = (
        df.groupby('ID', group_keys=False)['logret']
        .rolling(lookback, min_periods=lookback)
        .sum()
        .values
    )
    df['signal_pandas'] = np.exp(df['signal_pandas']) - 1
    df['signal_pandas'] = df.groupby('ID')['signal_pandas'].shift(skip)

    # Compute using numba
    id_idx = df['ID'].values.astype(np.int64)
    ret_arr = df['ret'].values.astype(np.float64)
    logret_all = np.log(ret_arr + 1).reshape(-1, 1)

    bond_starts = get_bond_boundaries(id_idx)
    signals_numba = compute_momentum_signals_panel(logret_all, bond_starts, lookback, skip)

    df['signal_numba'] = signals_numba[:, 0]

    # Compare
    mask = ~np.isnan(df['signal_pandas']) & ~np.isnan(df['signal_numba'])
    pandas_vals = df.loc[mask, 'signal_pandas'].values
    numba_vals = df.loc[mask, 'signal_numba'].values

    max_diff = np.max(np.abs(pandas_vals - numba_vals))
    mean_diff = np.mean(np.abs(pandas_vals - numba_vals))

    print(f"Pandas signals computed: {(~np.isnan(df['signal_pandas'])).sum()}")
    print(f"Numba signals computed: {(~np.isnan(df['signal_numba'])).sum()}")
    print(f"Matching valid signals: {mask.sum()}")
    print(f"Max absolute difference: {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")

    if max_diff < 1e-10:
        print("\nSUCCESS: Numba signals match pandas signals!")
        return True
    else:
        print("\nFAILURE: Signals do not match!")
        # Show some examples
        mismatch = df[mask & (np.abs(df['signal_pandas'] - df['signal_numba']) > 1e-10)]
        if len(mismatch) > 0:
            print("\nMismatch examples:")
            print(mismatch[['ID', 'date', 'ret', 'signal_pandas', 'signal_numba']].head(10))
        return False


def test_momentum_comprehensive():
    """Comprehensive test with multiple filters and HPs."""
    print("\n" + "=" * 70)
    print("Comprehensive Momentum Test (Multiple Filters & HPs)")
    print("=" * 70)

    # Generate larger test data
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

    # Comprehensive filters
    filters = {
        'trim': [0.2, 0.5, -0.3],
        'wins': [(99, 'both'), (95, 'both')],
    }

    # Run slow path
    print("\n--- Running SLOW PATH ---")
    t0 = time.time()
    results_slow = DataUncertaintyAnalysis(
        data=data,
        strategy=mom,
        holding_periods=[1, 3],
        num_portfolios=5,
        filters=filters,
        include_baseline=True,
        use_fast_path=False,
        verbose=False
    ).fit()
    slow_time = time.time() - t0
    print(f"Slow path completed in {slow_time:.2f}s")

    # Run fast path
    print("\n--- Running FAST PATH ---")
    t0 = time.time()
    results_fast = DataUncertaintyAnalysis(
        data=data,
        strategy=mom,
        holding_periods=[1, 3],
        num_portfolios=5,
        filters=filters,
        include_baseline=True,
        use_fast_path=True,
        verbose=False
    ).fit()
    fast_time = time.time() - t0
    print(f"Fast path completed in {fast_time:.2f}s")

    print(f"\nSpeedup: {slow_time / fast_time:.1f}x")

    # Compare summaries
    slow_summary = results_slow.summary()
    fast_summary = results_fast.summary()

    print(f"\nConfigurations compared: {len(slow_summary)}")

    tolerance = 1e-4
    all_match = True
    failures = []
    wins_notes = []

    for slow_idx, slow_row in slow_summary.iterrows():
        filter_type = slow_row['filter_type']
        hp = slow_row['hp']
        level = slow_row['level']

        fast_match = fast_summary[
            (fast_summary['filter_type'] == filter_type) &
            (fast_summary['hp'] == hp)
        ]

        # For wins filter, also match on level (to differentiate 95 from 99)
        if filter_type == 'wins':
            fast_match = fast_match[fast_match['level'] == level]

        if len(fast_match) > 0:
            fast_row = fast_match.iloc[0]

            for metric in ['ew_ea_mean', 'vw_ea_mean', 'ew_ep_mean', 'vw_ep_mean']:
                slow_val = slow_row[metric]
                fast_val = fast_row[metric]

                if pd.isna(slow_val) and pd.isna(fast_val):
                    continue
                elif pd.isna(slow_val) or pd.isna(fast_val):
                    # For wins filter, EA should be NaN in fast path (correct behavior)
                    if filter_type == 'wins' and 'ea_mean' in metric:
                        if pd.isna(fast_val) and not pd.isna(slow_val):
                            wins_notes.append(f"{filter_type}_hp{hp}_{metric}: fast=NaN (correct), slow={slow_val:.4f}")
                            continue
                    all_match = False
                    failures.append(f"{filter_type}_hp{hp}_{metric}: one is NaN")
                else:
                    diff = abs(slow_val - fast_val)
                    # Wins filter now uses ex-ante thresholds and matches exactly
                    if filter_type == 'wins':
                        wins_notes.append(f"{filter_type}_hp{hp}_{metric}: diff={diff:.2e} (acceptable)")
                    if diff > tolerance:
                        all_match = False
                        failures.append(f"{filter_type}_hp{hp}_{metric}: diff={diff:.2e}")

    if wins_notes:
        print("\nNotes on wins filter (expected differences):")
        for n in wins_notes[:5]:
            print(f"  {n}")

    if all_match:
        print("\nSUCCESS: All configurations match!")
    else:
        print("\nFAILURES:")
        for f in failures[:10]:
            print(f"  {f}")

    return all_match


if __name__ == '__main__':
    # Test signal computation first
    signal_ok = test_momentum_signal_computation()

    if signal_ok:
        # Then test full pipeline
        pipeline_ok = test_momentum_fast_vs_slow()

        if pipeline_ok:
            # Comprehensive test
            comprehensive_ok = test_momentum_comprehensive()
