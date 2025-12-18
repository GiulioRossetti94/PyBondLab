"""
Validation script for Phase 14: Filter support in fast batch path.

Compares fast batch path with filters against slow SingleSort with the same filters
to ensure numerical correctness and no look-ahead bias.

Tests:
1. rating=(1, 10) vs rating='IG' (should be identical)
2. Fast batch with rating vs slow SingleSort with rating
3. Fast batch with subset_filter vs slow SingleSort with subset_filter
4. HP=1 and HP=3 for all configurations
"""
import sys
import os
import time
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from PyBondLab import BatchStrategyFormation, StrategyFormation, SingleSort
from PyBondLab.config import StrategyFormationConfig, FormationConfig, DataConfig
from PyBondLab.pbl_test import generate_synthetic_data


def compare_results(fast_ew, fast_vw, slow_ew, slow_vw, tolerance=1e-10):
    """Compare fast vs slow results and return max differences."""
    # Align indices
    common_idx = fast_ew.index.intersection(slow_ew.index)

    ew_diff = (fast_ew.loc[common_idx] - slow_ew.loc[common_idx]).abs().max()
    vw_diff = (fast_vw.loc[common_idx] - slow_vw.loc[common_idx]).abs().max()

    return ew_diff, vw_diff


def run_slow_singlesort(data, signal, hp, rating=None, subset_filter=None):
    """Run slow SingleSort for ground truth."""
    strategy = SingleSort(
        holding_period=hp,
        sort_var=signal,
        num_portfolios=5,
        verbose=False
    )

    sf_config = StrategyFormationConfig(
        data=DataConfig(
            rating=rating,
            subset_filter=subset_filter,
        ),
        formation=FormationConfig(
            dynamic_weights=True,
            compute_turnover=False,
            verbose=False,
        )
    )

    sf = StrategyFormation(
        data=data,
        strategy=strategy,
        config=sf_config
    )

    result = sf.fit()
    ew_ls, vw_ls = result.get_long_short()

    return ew_ls, vw_ls


def validate_rating_filter(data, signals, hp, rating, tolerance=1e-10, verbose=True):
    """Validate fast batch with rating filter against slow SingleSort."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing rating={rating}, HP={hp}")
        print(f"{'='*60}")

    # Fast batch path
    t0 = time.time()
    batch = BatchStrategyFormation(
        data=data,
        signals=signals,
        holding_period=hp,
        num_portfolios=5,
        turnover=False,
        rating=rating,
        verbose=verbose
    )
    batch_results = batch.fit()
    fast_time = time.time() - t0

    # Slow SingleSort path
    slow_results = {}
    t0 = time.time()
    for signal in signals:
        ew_ls, vw_ls = run_slow_singlesort(data, signal, hp, rating=rating)
        slow_results[signal] = (ew_ls, vw_ls)
    slow_time = time.time() - t0

    # Compare results
    all_pass = True
    for signal in signals:
        fast_ew, fast_vw = batch_results[signal].get_long_short()
        slow_ew, slow_vw = slow_results[signal]

        ew_diff, vw_diff = compare_results(fast_ew, fast_vw, slow_ew, slow_vw, tolerance)

        passed = ew_diff < tolerance and vw_diff < tolerance
        status = "PASS" if passed else "FAIL"

        if verbose:
            print(f"  {signal}: EW diff={ew_diff:.2e}, VW diff={vw_diff:.2e} [{status}]")

        if not passed:
            all_pass = False

    if verbose:
        print(f"  Fast path: {fast_time:.2f}s, Slow path: {slow_time:.2f}s")
        print(f"  Speedup: {slow_time/fast_time:.1f}x")

    return all_pass


def validate_subset_filter(data, signals, hp, subset_filter, tolerance=1e-10, verbose=True):
    """Validate fast batch with subset_filter against slow SingleSort."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing subset_filter={subset_filter}, HP={hp}")
        print(f"{'='*60}")

    # Fast batch path
    t0 = time.time()
    batch = BatchStrategyFormation(
        data=data,
        signals=signals,
        holding_period=hp,
        num_portfolios=5,
        turnover=False,
        subset_filter=subset_filter,
        verbose=verbose
    )
    batch_results = batch.fit()
    fast_time = time.time() - t0

    # Slow SingleSort path
    slow_results = {}
    t0 = time.time()
    for signal in signals:
        ew_ls, vw_ls = run_slow_singlesort(data, signal, hp, subset_filter=subset_filter)
        slow_results[signal] = (ew_ls, vw_ls)
    slow_time = time.time() - t0

    # Compare results
    all_pass = True
    for signal in signals:
        fast_ew, fast_vw = batch_results[signal].get_long_short()
        slow_ew, slow_vw = slow_results[signal]

        ew_diff, vw_diff = compare_results(fast_ew, fast_vw, slow_ew, slow_vw, tolerance)

        passed = ew_diff < tolerance and vw_diff < tolerance
        status = "PASS" if passed else "FAIL"

        if verbose:
            print(f"  {signal}: EW diff={ew_diff:.2e}, VW diff={vw_diff:.2e} [{status}]")

        if not passed:
            all_pass = False

    if verbose:
        print(f"  Fast path: {fast_time:.2f}s, Slow path: {slow_time:.2f}s")
        print(f"  Speedup: {slow_time/fast_time:.1f}x")

    return all_pass


def validate_combined_filters(data, signals, hp, rating, subset_filter, tolerance=1e-10, verbose=True):
    """Validate fast batch with both rating and subset_filter."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing rating={rating}, subset_filter={subset_filter}, HP={hp}")
        print(f"{'='*60}")

    # Fast batch path
    t0 = time.time()
    batch = BatchStrategyFormation(
        data=data,
        signals=signals,
        holding_period=hp,
        num_portfolios=5,
        turnover=False,
        rating=rating,
        subset_filter=subset_filter,
        verbose=verbose
    )
    batch_results = batch.fit()
    fast_time = time.time() - t0

    # Slow SingleSort path
    slow_results = {}
    t0 = time.time()
    for signal in signals:
        ew_ls, vw_ls = run_slow_singlesort(data, signal, hp, rating=rating, subset_filter=subset_filter)
        slow_results[signal] = (ew_ls, vw_ls)
    slow_time = time.time() - t0

    # Compare results
    all_pass = True
    for signal in signals:
        fast_ew, fast_vw = batch_results[signal].get_long_short()
        slow_ew, slow_vw = slow_results[signal]

        ew_diff, vw_diff = compare_results(fast_ew, fast_vw, slow_ew, slow_vw, tolerance)

        passed = ew_diff < tolerance and vw_diff < tolerance
        status = "PASS" if passed else "FAIL"

        if verbose:
            print(f"  {signal}: EW diff={ew_diff:.2e}, VW diff={vw_diff:.2e} [{status}]")

        if not passed:
            all_pass = False

    if verbose:
        print(f"  Fast path: {fast_time:.2f}s, Slow path: {slow_time:.2f}s")
        print(f"  Speedup: {slow_time/fast_time:.1f}x")

    return all_pass


def main():
    parser = argparse.ArgumentParser(description='Validate fast batch path with filters')
    parser.add_argument('--hp', type=int, default=None, help='Test specific holding period (1 or 3)')
    parser.add_argument('--tolerance', type=float, default=1e-10, help='Numerical tolerance')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer signals')
    args = parser.parse_args()

    print("="*70)
    print("Phase 14 Validation: Fast Batch Path with Filters")
    print("="*70)

    # Generate synthetic data
    print("\nGenerating synthetic data...")
    data = generate_synthetic_data(n_dates=60, n_bonds=500, seed=42)

    # Add some characteristic columns for subset_filter testing
    # char1 already exists in synthetic data, let's use it

    signals = ['signal1', 'signal2'] if args.quick else ['signal1', 'signal2', 'signal3']
    hps = [args.hp] if args.hp else [1, 3]
    tolerance = args.tolerance

    all_passed = True
    test_count = 0
    pass_count = 0

    # Test 1: rating='IG' (string)
    for hp in hps:
        test_count += 1
        if validate_rating_filter(data, signals, hp, rating='IG', tolerance=tolerance):
            pass_count += 1
        else:
            all_passed = False

    # Test 2: rating=(1, 10) (tuple, equivalent to 'IG')
    for hp in hps:
        test_count += 1
        if validate_rating_filter(data, signals, hp, rating=(1, 10), tolerance=tolerance):
            pass_count += 1
        else:
            all_passed = False

    # Test 3: rating='NIG'
    for hp in hps:
        test_count += 1
        if validate_rating_filter(data, signals, hp, rating='NIG', tolerance=tolerance):
            pass_count += 1
        else:
            all_passed = False

    # Test 4: rating=(11, 22) (tuple, equivalent to 'NIG')
    for hp in hps:
        test_count += 1
        if validate_rating_filter(data, signals, hp, rating=(11, 22), tolerance=tolerance):
            pass_count += 1
        else:
            all_passed = False

    # Test 5: Custom rating range (BBB only: 7-10)
    for hp in hps:
        test_count += 1
        if validate_rating_filter(data, signals, hp, rating=(7, 10), tolerance=tolerance):
            pass_count += 1
        else:
            all_passed = False

    # Test 6: subset_filter with char1
    # char1 is generated with np.random.randn(), so filter to positive values
    for hp in hps:
        test_count += 1
        if validate_subset_filter(data, signals, hp,
                                   subset_filter={'char1': (-1.0, 1.0)},
                                   tolerance=tolerance):
            pass_count += 1
        else:
            all_passed = False

    # Test 7: Combined rating + subset_filter
    for hp in hps:
        test_count += 1
        if validate_combined_filters(data, signals, hp,
                                      rating='IG',
                                      subset_filter={'char1': (-0.5, 0.5)},
                                      tolerance=tolerance):
            pass_count += 1
        else:
            all_passed = False

    # Summary
    print("\n" + "="*70)
    print(f"SUMMARY: {pass_count}/{test_count} tests passed")
    print("="*70)

    if all_passed:
        print("\n*** ALL TESTS PASSED ***")
        print("Fast batch path with filters produces numerically identical results")
        print("to slow SingleSort path (no look-ahead bias).")
        return 0
    else:
        print("\n*** SOME TESTS FAILED ***")
        print("Check the output above for details.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
