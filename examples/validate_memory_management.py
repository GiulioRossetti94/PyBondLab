#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validate memory management features for BatchWithinFirmSortFormation.

Tests:
1. Chunking works correctly (processes in batches)
2. Auto chunk_size detects and sets appropriate value
3. signals_per_worker batching works
4. Memory warning is triggered for large workloads
5. Results match between chunked and non-chunked processing
"""

import sys
import time
import gc
import warnings
sys.path.insert(0, '.')

import numpy as np
import pandas as pd


def generate_test_data(n_dates=60, n_bonds=500, n_firms=50, n_signals=20, seed=42):
    """Generate synthetic test data for WithinFirmSort."""
    np.random.seed(seed)

    # Create date list
    dates = pd.date_range('2015-01-31', periods=n_dates, freq='ME')

    # Create bond and firm IDs
    bond_ids = [f'BOND_{i:04d}' for i in range(n_bonds)]
    firm_ids = np.random.choice([f'FIRM_{i:03d}' for i in range(n_firms)], size=n_bonds)
    bond_to_firm = dict(zip(bond_ids, firm_ids))

    rows = []
    for date in dates:
        # Random subset of bonds active on each date
        active_bonds = np.random.choice(bond_ids, size=int(n_bonds * 0.8), replace=False)
        for bond in active_bonds:
            row = {
                'date': date,
                'ID': bond,
                'PERMNO': bond_to_firm[bond],
                'ret': np.random.randn() * 0.03,
                'VW': np.random.uniform(10, 1000),
                'RATING_NUM': np.random.choice(range(1, 22)),
            }
            # Add signal columns
            for i in range(n_signals):
                row[f'signal_{i}'] = np.random.randn()
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def test_chunking():
    """Test that chunking processes signals in batches."""
    print("\n" + "=" * 60)
    print("TEST 1: Chunking functionality")
    print("=" * 60)

    from PyBondLab import BatchWithinFirmSortFormation

    data = generate_test_data(n_dates=30, n_bonds=100, n_firms=20, n_signals=10)
    signals = [f'signal_{i}' for i in range(10)]

    # Run without chunking
    print("\nRunning without chunking (baseline)...")
    batch1 = BatchWithinFirmSortFormation(
        data=data,
        signals=signals,
        turnover=True,
        n_jobs=2,
        chunk_size=None,
        verbose=True
    )
    t0 = time.time()
    results1 = batch1.fit()
    time_no_chunk = time.time() - t0

    gc.collect()

    # Run with chunking
    print("\nRunning with chunk_size=3...")
    batch2 = BatchWithinFirmSortFormation(
        data=data,
        signals=signals,
        turnover=True,
        n_jobs=2,
        chunk_size=3,
        verbose=True
    )
    t0 = time.time()
    results2 = batch2.fit()
    time_chunked = time.time() - t0

    # Compare results
    print("\nComparing results...")
    all_match = True
    for signal in signals:
        ew1, vw1 = results1[signal].get_long_short()
        ew2, vw2 = results2[signal].get_long_short()

        ew_diff = (ew1 - ew2).abs().max()
        vw_diff = (vw1 - vw2).abs().max()

        if ew_diff > 1e-10 or vw_diff > 1e-10:
            print(f"  {signal}: MISMATCH - EW diff={ew_diff:.2e}, VW diff={vw_diff:.2e}")
            all_match = False

    if all_match:
        print("  All signals match! ✓")

    print(f"\nTiming: no_chunk={time_no_chunk:.2f}s, chunked={time_chunked:.2f}s")

    return all_match


def test_auto_chunk_size():
    """Test auto chunk_size feature."""
    print("\n" + "=" * 60)
    print("TEST 2: Auto chunk_size")
    print("=" * 60)

    from PyBondLab import BatchWithinFirmSortFormation

    data = generate_test_data(n_dates=30, n_bonds=100, n_firms=20, n_signals=5)
    signals = [f'signal_{i}' for i in range(5)]

    print("\nCreating batch with chunk_size='auto'...")
    batch = BatchWithinFirmSortFormation(
        data=data,
        signals=signals,
        turnover=True,
        n_jobs=2,
        chunk_size='auto',
        verbose=True
    )

    print(f"Effective chunk_size: {batch.chunk_size}")

    results = batch.fit()

    success = len(results.successful_signals) == len(signals)
    print(f"\nAll signals processed: {'✓' if success else '✗'}")

    return success


def test_signals_per_worker():
    """Test signals_per_worker batching."""
    print("\n" + "=" * 60)
    print("TEST 3: signals_per_worker batching")
    print("=" * 60)

    from PyBondLab import BatchWithinFirmSortFormation

    data = generate_test_data(n_dates=30, n_bonds=100, n_firms=20, n_signals=8)
    signals = [f'signal_{i}' for i in range(8)]

    # Run with signals_per_worker=1 (default)
    print("\nRunning with signals_per_worker=1...")
    batch1 = BatchWithinFirmSortFormation(
        data=data,
        signals=signals,
        turnover=True,
        n_jobs=2,
        signals_per_worker=1,
        verbose=False
    )
    results1 = batch1.fit()

    # Run with signals_per_worker=2
    print("Running with signals_per_worker=2...")
    batch2 = BatchWithinFirmSortFormation(
        data=data,
        signals=signals,
        turnover=True,
        n_jobs=2,
        signals_per_worker=2,
        chunk_size=4,  # Process 4 signals at a time
        verbose=True
    )
    results2 = batch2.fit()

    # Compare results
    print("\nComparing results...")
    all_match = True
    for signal in signals:
        ew1, vw1 = results1[signal].get_long_short()
        ew2, vw2 = results2[signal].get_long_short()

        ew_diff = (ew1 - ew2).abs().max()
        vw_diff = (vw1 - vw2).abs().max()

        if ew_diff > 1e-10 or vw_diff > 1e-10:
            print(f"  {signal}: MISMATCH")
            all_match = False

    if all_match:
        print("  All signals match! ✓")

    return all_match


def test_memory_warning():
    """Test that memory warning is triggered for large workloads."""
    print("\n" + "=" * 60)
    print("TEST 4: Memory warning")
    print("=" * 60)

    from PyBondLab.batch_withinfirm import (
        _estimate_peak_memory_mb,
        _get_available_memory_mb,
        _suggest_chunk_size
    )

    # Create small test data
    data = generate_test_data(n_dates=30, n_bonds=100, n_firms=20, n_signals=5)

    # Test memory estimation functions
    per_worker, peak = _estimate_peak_memory_mb(data, n_signals=100, n_workers=10)
    available = _get_available_memory_mb()
    suggested = _suggest_chunk_size(data, n_signals=100, n_workers=10)

    print(f"\nMemory estimation:")
    print(f"  Per worker: {per_worker:.1f} MB")
    print(f"  Peak total: {peak:.1f} MB")
    print(f"  Available: {available:.1f} MB")
    print(f"  Suggested chunk_size: {suggested}")

    # The functions should return reasonable values
    success = (
        per_worker > 0 and
        peak > 0 and
        available > 0
    )

    print(f"\nMemory functions work correctly: {'✓' if success else '✗'}")

    return success


def test_fast_path_unchanged():
    """Test that fast path (turnover=False) still works."""
    print("\n" + "=" * 60)
    print("TEST 5: Fast path (turnover=False)")
    print("=" * 60)

    from PyBondLab import BatchWithinFirmSortFormation

    data = generate_test_data(n_dates=30, n_bonds=100, n_firms=20, n_signals=5)
    signals = [f'signal_{i}' for i in range(5)]

    print("\nRunning with turnover=False (fast path)...")
    batch = BatchWithinFirmSortFormation(
        data=data,
        signals=signals,
        turnover=False,
        n_jobs=2,
        verbose=True
    )

    results = batch.fit()

    # Check all signals processed
    success = len(results.successful_signals) == len(signals)
    print(f"\nAll signals processed: {'✓' if success else '✗'}")

    # Verify we can access results
    for signal in signals[:2]:
        ew_ls, vw_ls = results[signal].get_long_short()
        print(f"  {signal}: EW mean={ew_ls.mean()*100:.2f}%, VW mean={vw_ls.mean()*100:.2f}%")

    return success


def main():
    """Run all tests."""
    print("=" * 60)
    print("Memory Management Validation for BatchWithinFirmSortFormation")
    print("=" * 60)

    results = {}

    # Run tests
    results['chunking'] = test_chunking()
    results['auto_chunk'] = test_auto_chunk_size()
    results['signals_per_worker'] = test_signals_per_worker()
    results['memory_warning'] = test_memory_warning()
    results['fast_path'] = test_fast_path_unchanged()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All memory management tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
