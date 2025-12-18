#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validate Fast Batch Path for BatchStrategyFormation.

Compares results from fast batch path (numba kernels) vs slow 1-by-1 path
to ensure numerical accuracy.

Usage:
    python examples/validate_fast_batch.py
    python examples/validate_fast_batch.py --n_signals 20
    python examples/validate_fast_batch.py --hp 3
"""

import sys
import time
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, '.')

from PyBondLab.pbl_test import generate_synthetic_data
from PyBondLab import BatchStrategyFormation, StrategyFormation, SingleSort
from PyBondLab.config import StrategyFormationConfig, DataConfig, FormationConfig


def generate_test_data(n_dates: int = 60, n_bonds: int = 500, n_signals: int = 10, seed: int = 42):
    """
    Generate test data with multiple signal columns.
    """
    # Use the existing synthetic data generator
    data = generate_synthetic_data(n_dates=n_dates, n_bonds=n_bonds, seed=seed)

    # The base data has 'signal1', 'signal2' columns, use signal1 as base
    np.random.seed(seed + 100)  # Different seed for additional signals

    # Add multiple signal columns with different characteristics
    signal_names = ['signal1', 'signal2']  # Start with existing signals

    base_signal = data['signal1'].values.copy()

    for i in range(2, n_signals):
        signal_name = f'signal_{i}'
        signal_names.append(signal_name)

        # Create signals with different noise patterns
        noise = np.random.randn(len(data)) * 0.1 * (i + 1)
        data[signal_name] = base_signal + noise

        # Add some NaN values (~2%)
        nan_mask = np.random.rand(len(data)) < 0.02
        data.loc[nan_mask, signal_name] = np.nan

    return data, signal_names


def run_slow_path(data, signals, holding_period, num_portfolios):
    """
    Run slow path: process each signal 1-by-1 with turnover=False.
    """
    results = {}

    for signal in signals:
        strategy = SingleSort(
            holding_period=holding_period,
            sort_var=signal,
            num_portfolios=num_portfolios,
            verbose=False
        )

        sf_config = StrategyFormationConfig(
            data=DataConfig(),
            formation=FormationConfig(
                dynamic_weights=True,
                compute_turnover=False,
                banding_threshold=None,
                verbose=False,
            )
        )

        sf = StrategyFormation(data=data, strategy=strategy, config=sf_config)
        result = sf.fit()
        ew_ls, vw_ls = result.get_long_short(strategy='ea')
        results[signal] = {'ew_ls': ew_ls, 'vw_ls': vw_ls}

    return results


def run_fast_batch_path(data, signals, holding_period, num_portfolios):
    """
    Run fast batch path using BatchStrategyFormation.
    """
    batch = BatchStrategyFormation(
        data=data,
        signals=signals,
        holding_period=holding_period,
        num_portfolios=num_portfolios,
        turnover=False,  # Required for fast path
        chars=None,      # Required for fast path
        banding=None,    # Required for fast path
        rating=None,     # Required for fast path
        n_jobs=1,
        verbose=True
    )

    results_batch = batch.fit()

    results = {}
    for signal in signals:
        ew_ls, vw_ls = results_batch[signal].get_long_short()
        results[signal] = {'ew_ls': ew_ls, 'vw_ls': vw_ls}

    return results


def compare_results(slow_results, fast_results, signals, tolerance=1e-10):
    """
    Compare slow path vs fast batch path results.
    """
    all_pass = True

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    for signal in signals:
        slow = slow_results[signal]
        fast = fast_results[signal]

        # Align indices
        common_idx = slow['ew_ls'].index.intersection(fast['ew_ls'].index)

        ew_slow = slow['ew_ls'].loc[common_idx]
        ew_fast = fast['ew_ls'].loc[common_idx]
        vw_slow = slow['vw_ls'].loc[common_idx]
        vw_fast = fast['vw_ls'].loc[common_idx]

        # Compute differences
        ew_diff = np.abs(ew_slow - ew_fast).max()
        vw_diff = np.abs(vw_slow - vw_fast).max()

        ew_pass = ew_diff < tolerance
        vw_pass = vw_diff < tolerance

        status = "PASS" if (ew_pass and vw_pass) else "FAIL"
        if not (ew_pass and vw_pass):
            all_pass = False

        print(f"{signal}: {status}  EW diff={ew_diff:.2e}  VW diff={vw_diff:.2e}  n_dates={len(common_idx)}")

    print("=" * 70)

    return all_pass


def main():
    parser = argparse.ArgumentParser(description='Validate fast batch path')
    parser.add_argument('--n_signals', type=int, default=10, help='Number of signals to test')
    parser.add_argument('--hp', type=int, default=1, help='Holding period')
    parser.add_argument('--n_dates', type=int, default=60, help='Number of dates')
    parser.add_argument('--n_bonds', type=int, default=500, help='Number of bonds')
    parser.add_argument('--num_portfolios', type=int, default=5, help='Number of portfolios')
    args = parser.parse_args()

    print("=" * 70)
    print("FAST BATCH PATH VALIDATION")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  n_signals:      {args.n_signals}")
    print(f"  holding_period: {args.hp}")
    print(f"  n_dates:        {args.n_dates}")
    print(f"  n_bonds:        {args.n_bonds}")
    print(f"  num_portfolios: {args.num_portfolios}")
    print("=" * 70)

    # Generate test data
    print("\n1. Generating test data...")
    t0 = time.time()
    data, signals = generate_test_data(
        n_dates=args.n_dates,
        n_bonds=args.n_bonds,
        n_signals=args.n_signals,
        seed=42
    )
    print(f"   Data shape: {data.shape}")
    print(f"   Signals: {signals}")
    print(f"   Time: {time.time() - t0:.2f}s")

    # Run slow path
    print(f"\n2. Running SLOW path (1-by-1)...")
    t0 = time.time()
    slow_results = run_slow_path(data, signals, args.hp, args.num_portfolios)
    slow_time = time.time() - t0
    print(f"   Time: {slow_time:.2f}s")

    # Run fast batch path
    print(f"\n3. Running FAST batch path...")
    t0 = time.time()
    fast_results = run_fast_batch_path(data, signals, args.hp, args.num_portfolios)
    fast_time = time.time() - t0
    print(f"   Time: {fast_time:.2f}s")

    # Compare results
    all_pass = compare_results(slow_results, fast_results, signals)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Slow path time:  {slow_time:.2f}s")
    print(f"Fast path time:  {fast_time:.2f}s")
    print(f"Speedup:         {slow_time / fast_time:.1f}x")
    print(f"All tests pass:  {all_pass}")
    print("=" * 70)

    if not all_pass:
        sys.exit(1)

    return 0


if __name__ == '__main__':
    main()
