# -*- coding: utf-8 -*-
"""
BatchStrategyFormation Example
==============================

This example demonstrates how to use BatchStrategyFormation to efficiently
process multiple signals in parallel, achieving significant speedups over
sequential processing.

Usage:
    python examples/batch_strategy_formation.py

Performance:
    With 4 workers, expect ~2x speedup over sequential processing.
"""

import time
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyBondLab import StrategyFormation, BatchStrategyFormation
from PyBondLab.StrategyClass import SingleSort
from PyBondLab.pbl_test import generate_synthetic_data


def main():
    print("=" * 70)
    print("BatchStrategyFormation Example")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Step 1: Generate synthetic data
    # -------------------------------------------------------------------------
    print("\n1. Generating synthetic data...")

    # Generate base data with required columns
    data = generate_synthetic_data(n_dates=60, n_bonds=500, seed=42)

    # Add multiple signal columns for testing
    np.random.seed(42)
    n_signals = 10
    for i in range(n_signals):
        data[f'signal_{i}'] = np.random.randn(len(data))

    signals = [f'signal_{i}' for i in range(n_signals)]

    print(f"   Data shape: {data.shape}")
    print(f"   Unique bonds: {data['ID'].nunique()}")
    print(f"   Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"   Signals to process: {len(signals)}")

    # -------------------------------------------------------------------------
    # Step 2: Run batch processing with different worker counts
    # -------------------------------------------------------------------------
    print("\n2. Running BatchStrategyFormation with different worker counts...")

    results_summary = []

    for n_jobs in [1, 2, 4]:
        print(f"\n   n_jobs={n_jobs}:")

        batch = BatchStrategyFormation(
            data=data,
            signals=signals,
            holding_period=1,      # Monthly rebalancing
            num_portfolios=5,      # Quintile portfolios
            turnover=True,         # Compute turnover
            n_jobs=n_jobs,         # Number of parallel workers
            verbose=False,         # Suppress progress output for timing
        )

        t0 = time.perf_counter()
        results = batch.fit()
        elapsed = time.perf_counter() - t0

        n_success = len([s for s in signals if s not in results.errors])
        print(f"      Time: {elapsed:.2f}s ({elapsed/len(signals)*1000:.0f}ms per signal)")
        print(f"      Success: {n_success}/{len(signals)} signals")

        results_summary.append({
            'n_jobs': n_jobs,
            'time': elapsed,
            'ms_per_signal': elapsed/len(signals)*1000
        })

    # -------------------------------------------------------------------------
    # Step 3: Compare with sequential processing
    # -------------------------------------------------------------------------
    print("\n3. Comparing with sequential StrategyFormation...")

    t0 = time.perf_counter()
    sequential_results = {}
    for signal in signals:
        strategy = SingleSort(
            holding_period=1,
            sort_var=signal,
            num_portfolios=5,
            verbose=False
        )
        sf = StrategyFormation(data=data, strategy=strategy)
        sf.config.formation.verbose = False
        sf.config.formation.compute_turnover = True
        sequential_results[signal] = sf.fit()
    t_sequential = time.perf_counter() - t0

    print(f"   Sequential time: {t_sequential:.2f}s ({t_sequential/len(signals)*1000:.0f}ms per signal)")

    # -------------------------------------------------------------------------
    # Step 4: Show speedup summary
    # -------------------------------------------------------------------------
    print("\n4. Speedup Summary:")
    print("-" * 50)
    print(f"{'Method':<25} {'Time (s)':<12} {'Speedup':<10}")
    print("-" * 50)
    print(f"{'Sequential':<25} {t_sequential:<12.2f} {'1.0x':<10}")

    for r in results_summary:
        speedup = t_sequential / r['time']
        print(f"{'Batch (n_jobs=' + str(r['n_jobs']) + ')':<25} {r['time']:<12.2f} {speedup:.1f}x")
    print("-" * 50)

    # -------------------------------------------------------------------------
    # Step 5: Access results
    # -------------------------------------------------------------------------
    print("\n5. Accessing batch results...")

    # Run batch again with verbose=True to show progress
    batch = BatchStrategyFormation(
        data=data,
        signals=signals[:3],  # Just first 3 for demo
        holding_period=1,
        num_portfolios=5,
        turnover=True,
        n_jobs=2,
        verbose=True,  # Show progress
    )
    results = batch.fit()

    # Access individual signal results (same API as StrategyFormation.fit())
    print("\n   Example results for 'signal_0':")
    signal_result = results['signal_0']

    # Get long-short returns
    ew_ls, vw_ls = signal_result.get_long_short()
    print(f"      EW L-S mean return: {ew_ls.mean():.6f}")
    print(f"      VW L-S mean return: {vw_ls.mean():.6f}")

    # Get turnover
    ew_turn, vw_turn = signal_result.get_turnover()
    print(f"      EW mean turnover: {ew_turn.mean().mean():.4f}")
    print(f"      VW mean turnover: {vw_turn.mean().mean():.4f}")

    # Get portfolio returns
    ew_returns = signal_result.get_returns(weight_type='ew')
    print(f"      Portfolio return shape: {ew_returns.shape}")

    # -------------------------------------------------------------------------
    # Step 6: Show available result attributes
    # -------------------------------------------------------------------------
    print("\n6. Available result methods:")
    print("   - results['signal_name'].get_long_short()   # EW and VW L-S returns")
    print("   - results['signal_name'].get_turnover()     # EW and VW turnover")
    print("   - results['signal_name'].get_returns()      # Portfolio returns")
    print("   - results['signal_name'].get_ptf()          # Portfolio assignments")
    print("   - results['signal_name'].summary()          # Summary statistics")

    # List all signals
    print(f"\n   Processed signals: {list(results.signals)}")

    # Check for any errors
    if results.errors:
        print(f"\n   Signals with errors: {list(results.errors.keys())}")
    else:
        print("\n   No errors encountered.")

    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
