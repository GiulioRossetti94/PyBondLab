#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validation script for DataUncertaintyAnalysis rating tuple and subset_filter support.

Tests:
1. rating as tuple (e.g., (1, 10)) vs rating='IG'
2. subset_filter support
3. Combined rating + subset_filter
4. HP=1 and HP=3 for all configurations
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '.')

import PyBondLab as pbl
from PyBondLab import DataUncertaintyAnalysis, StrategyFormation, SingleSort
from PyBondLab.pbl_test import generate_synthetic_data
from PyBondLab.config import StrategyFormationConfig, FormationConfig, DataConfig


def run_slow_singlesort(data, signal, hp, rating=None, subset_filter=None):
    """Run slow path SingleSort for comparison."""
    strategy = SingleSort(
        holding_period=hp,
        sort_var=signal,
        num_portfolios=5,
    )

    sf_config = StrategyFormationConfig(
        data=DataConfig(rating=rating, subset_filter=subset_filter),
        formation=FormationConfig(
            dynamic_weights=True,
            compute_turnover=False,
            verbose=False,
        )
    )

    sf = StrategyFormation(data=data, strategy=strategy, config=sf_config)
    result = sf.fit()

    ew_ls, vw_ls = result.get_long_short()
    return ew_ls, vw_ls


def validate_rating_tuple(data, signal, hp, tolerance=1e-10):
    """Validate that rating=(1, 10) produces same results as rating='IG'."""
    print(f"\n{'='*60}")
    print(f"Testing: rating=(1,10) vs rating='IG', HP={hp}")
    print(f"{'='*60}")

    # Run with rating='IG'
    results_ig = DataUncertaintyAnalysis(
        data=data,
        signals=[signal],
        holding_periods=[hp],
        rating='IG',
        include_baseline=True,
        verbose=False,
    ).fit()

    # Run with rating=(1, 10)
    results_tuple = DataUncertaintyAnalysis(
        data=data,
        signals=[signal],
        holding_periods=[hp],
        rating=(1, 10),
        include_baseline=True,
        verbose=False,
    ).fit()

    # Compare
    ew_ig = results_ig.ew_ea.iloc[:, 0]
    ew_tuple = results_tuple.ew_ea.iloc[:, 0]

    # Align indices
    common_idx = ew_ig.index.intersection(ew_tuple.index)
    ew_ig = ew_ig.loc[common_idx]
    ew_tuple = ew_tuple.loc[common_idx]

    ew_diff = (ew_ig - ew_tuple).abs().max()

    passed = ew_diff < tolerance
    status = "PASS" if passed else "FAIL"
    print(f"  EW diff (IG vs (1,10)): {ew_diff:.2e} [{status}]")

    return passed


def validate_subset_filter(data, signal, hp, subset_filter, tolerance=1e-10):
    """Validate DataUncertaintyAnalysis subset_filter against slow SingleSort."""
    print(f"\n{'='*60}")
    print(f"Testing: subset_filter={subset_filter}, HP={hp}")
    print(f"{'='*60}")

    # Run DataUncertaintyAnalysis with subset_filter
    results = DataUncertaintyAnalysis(
        data=data,
        signals=[signal],
        holding_periods=[hp],
        subset_filter=subset_filter,
        include_baseline=True,
        verbose=False,
    ).fit()

    # Run slow SingleSort with same subset_filter
    slow_ew, slow_vw = run_slow_singlesort(
        data, signal, hp, rating=None, subset_filter=subset_filter
    )

    # Compare
    fast_ew = results.ew_ea.iloc[:, 0]

    # Align indices
    common_idx = fast_ew.dropna().index.intersection(slow_ew.dropna().index)
    fast_ew = fast_ew.loc[common_idx]
    slow_ew_aligned = slow_ew.loc[common_idx]

    ew_diff = (fast_ew - slow_ew_aligned).abs().max()

    passed = ew_diff < tolerance
    status = "PASS" if passed else "FAIL"
    print(f"  EW diff (fast vs slow): {ew_diff:.2e} [{status}]")

    return passed


def validate_combined_filter(data, signal, hp, rating, subset_filter, tolerance=1e-10):
    """Validate combined rating + subset_filter."""
    print(f"\n{'='*60}")
    print(f"Testing: rating={rating}, subset_filter={subset_filter}, HP={hp}")
    print(f"{'='*60}")

    # Run DataUncertaintyAnalysis
    results = DataUncertaintyAnalysis(
        data=data,
        signals=[signal],
        holding_periods=[hp],
        rating=rating,
        subset_filter=subset_filter,
        include_baseline=True,
        verbose=False,
    ).fit()

    # Run slow SingleSort
    slow_ew, slow_vw = run_slow_singlesort(
        data, signal, hp, rating=rating, subset_filter=subset_filter
    )

    # Compare
    fast_ew = results.ew_ea.iloc[:, 0]

    # Align indices
    common_idx = fast_ew.dropna().index.intersection(slow_ew.dropna().index)
    if len(common_idx) == 0:
        print(f"  WARNING: No common dates found")
        return False

    fast_ew = fast_ew.loc[common_idx]
    slow_ew_aligned = slow_ew.loc[common_idx]

    ew_diff = (fast_ew - slow_ew_aligned).abs().max()

    passed = ew_diff < tolerance
    status = "PASS" if passed else "FAIL"
    print(f"  EW diff (fast vs slow): {ew_diff:.2e} [{status}]")

    # Count observations passing filter
    n_total = len(data)
    rating_mask = np.ones(n_total, dtype=bool)
    if rating == 'IG' or rating == (1, 10):
        rating_mask = (data['RATING_NUM'] >= 1) & (data['RATING_NUM'] <= 10)
    elif isinstance(rating, tuple):
        rating_mask = (data['RATING_NUM'] >= rating[0]) & (data['RATING_NUM'] <= rating[1])

    filter_mask = rating_mask.copy()
    if subset_filter:
        for col, (min_val, max_val) in subset_filter.items():
            filter_mask &= (data[col] >= min_val) & (data[col] <= max_val)

    n_pass = filter_mask.sum()
    print(f"  Observations passing filter: {n_pass}/{n_total} ({100*n_pass/n_total:.1f}%)")

    return passed


def main():
    print("="*70)
    print("DataUncertaintyAnalysis Filter Validation")
    print("="*70)

    # Generate test data
    print("\nGenerating synthetic data...")
    data = generate_synthetic_data(n_dates=60, n_bonds=500, seed=42)
    signal = 'signal1'

    tolerance = 1e-10
    all_passed = True
    results = []

    # Test 1: rating tuple vs string
    for hp in [1, 3]:
        passed = validate_rating_tuple(data, signal, hp, tolerance)
        results.append(('rating tuple vs IG', hp, passed))
        all_passed &= passed

    # Test 2: subset_filter only
    subset_filter = {'char1': (-1.0, 1.0)}
    for hp in [1, 3]:
        passed = validate_subset_filter(data, signal, hp, subset_filter, tolerance)
        results.append(('subset_filter', hp, passed))
        all_passed &= passed

    # Test 3: rating=(7, 10) - subset of IG
    for hp in [1, 3]:
        passed = validate_combined_filter(
            data, signal, hp, rating=(7, 10), subset_filter=None, tolerance=tolerance
        )
        results.append(('rating=(7,10)', hp, passed))
        all_passed &= passed

    # Test 4: Combined rating + subset_filter
    for hp in [1, 3]:
        passed = validate_combined_filter(
            data, signal, hp,
            rating='IG',
            subset_filter={'char1': (-0.5, 0.5)},
            tolerance=tolerance
        )
        results.append(('rating=IG + subset_filter', hp, passed))
        all_passed &= passed

    # Test 5: Combined rating tuple + subset_filter
    for hp in [1, 3]:
        passed = validate_combined_filter(
            data, signal, hp,
            rating=(1, 10),
            subset_filter={'char1': (-0.5, 0.5)},
            tolerance=tolerance
        )
        results.append(('rating=(1,10) + subset_filter', hp, passed))
        all_passed &= passed

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    n_passed = sum(1 for _, _, p in results if p)
    n_total = len(results)

    for test_name, hp, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}, HP={hp}: {status}")

    print(f"\n{n_passed}/{n_total} tests passed")

    if all_passed:
        print("\n*** ALL TESTS PASSED ***")
        return 0
    else:
        print("\n*** SOME TESTS FAILED ***")
        return 1


if __name__ == '__main__':
    sys.exit(main())
