# -*- coding: utf-8 -*-
"""
Validation script to demonstrate the non-staggered rebalancing bug.

BUG: When using rebalance_frequency != 'monthly' (e.g., quarterly, annual),
returns/turnover/chars are only computed at rebalancing dates instead of
EVERY month.

EXPECTED: Returns should be computed EVERY month. Portfolio composition stays
fixed between rebalancing dates, but returns are collected monthly with weights
renormalized when bonds drop out.

Authors: PyBondLab Team
Created: 2024
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd

import PyBondLab as pbl
from PyBondLab import StrategyFormation, SingleSort
from PyBondLab.pbl_test import generate_synthetic_data_fast


def create_balanced_test_data(n_dates=12, n_bonds=50, seed=42):
    """
    Create perfectly balanced panel data for testing.

    Every bond is present at every date.
    """
    np.random.seed(seed)

    dates = pd.date_range(start='2020-01-31', periods=n_dates, freq='M')
    bonds = [f'BOND_{i:03d}' for i in range(n_bonds)]

    # Create balanced panel
    rows = []
    for date in dates:
        for bond in bonds:
            rows.append({
                'date': date,
                'ID': bond,
                'ret': np.random.normal(0.005, 0.02),
                'VW': np.random.uniform(100, 1000),
                'RATING_NUM': np.random.randint(1, 22),
                'signal1': np.random.normal(0, 1),
            })

    data = pd.DataFrame(rows)
    return data


def create_unbalanced_test_data(n_dates=12, n_bonds=50, seed=42, dropout_rate=0.1):
    """
    Create unbalanced panel data with random bond dropouts.

    Some bonds randomly disappear at certain dates.
    """
    np.random.seed(seed)

    dates = pd.date_range(start='2020-01-31', periods=n_dates, freq='M')
    bonds = [f'BOND_{i:03d}' for i in range(n_bonds)]

    rows = []
    for date in dates:
        for bond in bonds:
            # Random dropout
            if np.random.random() < dropout_rate:
                continue
            rows.append({
                'date': date,
                'ID': bond,
                'ret': np.random.normal(0.005, 0.02),
                'VW': np.random.uniform(100, 1000),
                'RATING_NUM': np.random.randint(1, 22),
                'signal1': np.random.normal(0, 1),
            })

    data = pd.DataFrame(rows)
    return data


def test_nonstaggered_monthly_returns(data, rebalance_frequency='quarterly', verbose=True):
    """
    Test that non-staggered rebalancing produces returns EVERY month.

    Returns
    -------
    dict with:
    - n_return_dates: Number of dates with non-NaN returns
    - n_expected: Number of dates expected (all months after first rebal)
    - has_bug: True if bug is present (fewer return dates than expected)
    """
    # Pass rebalance_frequency to strategy, not StrategyFormation
    strategy = SingleSort(
        holding_period=1,
        sort_var='signal1',
        num_portfolios=5,
        rebalance_frequency=rebalance_frequency,
    )

    sf = StrategyFormation(
        data=data,
        strategy=strategy,
        turnover=False,
        chars=None,
        verbose=verbose,
    )

    result = sf.fit()

    # Get long-short returns
    ew_ls, vw_ls = result.get_long_short()

    # Count non-NaN dates
    n_return_dates = ew_ls.notna().sum()
    n_total_dates = len(ew_ls)

    # Calculate expected number of return dates based on actual rebalancing schedule
    # Expected = all dates after first rebalancing date
    from PyBondLab.utils_optimized import _get_rebalancing_dates
    datelist = list(ew_ls.index)
    rebal_idx = _get_rebalancing_dates(datelist, rebalance_frequency, 6)  # Default rebal_month=6
    first_rebal_idx = rebal_idx[0] if rebal_idx else 0
    n_expected = n_total_dates - first_rebal_idx - 1  # All dates after first rebal

    has_bug = n_return_dates < n_expected

    if verbose:
        print(f"\n{'='*60}")
        print(f"Non-Staggered Rebalancing Test: {rebalance_frequency}")
        print(f"{'='*60}")
        print(f"Total dates:           {n_total_dates}")
        print(f"Expected return dates: {n_expected}")
        print(f"Actual return dates:   {n_return_dates}")
        print(f"Missing dates:         {n_expected - n_return_dates}")
        print(f"Bug present:           {'YES!' if has_bug else 'No'}")

        # Show which dates have returns
        print(f"\nReturns by date:")
        for i, (date, val) in enumerate(ew_ls.items()):
            status = f"{val:.4f}" if pd.notna(val) else "NaN (missing!)"
            print(f"  {date.strftime('%Y-%m')}: {status}")

    return {
        'n_return_dates': n_return_dates,
        'n_expected': n_expected,
        'has_bug': has_bug,
        'ew_ls': ew_ls,
        'vw_ls': vw_ls,
    }


def test_with_turnover(data, rebalance_frequency='quarterly', verbose=True):
    """Test turnover computation with non-staggered rebalancing."""
    # Pass rebalance_frequency to strategy
    strategy = SingleSort(
        holding_period=1,
        sort_var='signal1',
        num_portfolios=5,
        rebalance_frequency=rebalance_frequency,
    )

    sf = StrategyFormation(
        data=data,
        strategy=strategy,
        turnover=True,
        chars=None,
        verbose=verbose,
    )

    result = sf.fit()

    # Get turnover
    turn_ew, turn_vw = result.get_turnover()

    # Count non-NaN dates
    # Turnover computed at all dates after first rebalancing, EXCEPT the first return date
    # (first return date has no previous period to compare with)
    from PyBondLab.utils_optimized import _get_rebalancing_dates
    datelist = list(turn_ew.index)
    # Note: turn_ew already skips first date, so indices are shifted
    full_datelist = pd.date_range(start=datelist[0] - pd.DateOffset(months=1), periods=len(datelist)+1, freq='ME')
    rebal_idx = _get_rebalancing_dates(list(full_datelist), rebalance_frequency, 6)
    first_rebal_idx = rebal_idx[0] if rebal_idx else 0
    # Turnover starts at SECOND return date (first has no previous to compare)
    n_expected = len(turn_ew) - first_rebal_idx - 1  # -1 for first return date

    n_turn_dates = turn_ew.iloc[:, 0].notna().sum()
    has_bug = n_turn_dates < n_expected

    if verbose:
        print(f"\n{'='*60}")
        print(f"Turnover Test: {rebalance_frequency}")
        print(f"{'='*60}")
        print(f"Expected turnover dates: {n_expected}")
        print(f"Actual turnover dates:   {n_turn_dates}")
        print(f"Bug present:             {'YES!' if has_bug else 'No'}")

        print(f"\nTurnover (portfolio 1) by date:")
        for date, val in turn_ew.iloc[:, 0].items():
            status = f"{val:.4f}" if pd.notna(val) else "NaN"
            print(f"  {date.strftime('%Y-%m')}: {status}")

    return {
        'n_turn_dates': n_turn_dates,
        'n_expected': n_expected,
        'has_bug': has_bug,
    }


def test_with_chars(data, rebalance_frequency='quarterly', verbose=True):
    """Test characteristics computation with non-staggered rebalancing."""
    # Add a characteristic column if not present
    if 'char1' not in data.columns:
        data = data.copy()
        data['char1'] = np.random.normal(0, 1, len(data))

    # Pass rebalance_frequency to strategy
    strategy = SingleSort(
        holding_period=1,
        sort_var='signal1',
        num_portfolios=5,
        rebalance_frequency=rebalance_frequency,
    )

    sf = StrategyFormation(
        data=data,
        strategy=strategy,
        turnover=False,
        chars=['char1'],
        verbose=verbose,
    )

    result = sf.fit()

    # Get characteristics
    chars_ew, chars_vw = result.get_characteristics()

    # Count non-NaN dates for first characteristic
    # Expected = all dates after first rebalancing date
    from PyBondLab.utils_optimized import _get_rebalancing_dates
    char_df = chars_ew['char1']
    datelist = list(char_df.index)
    rebal_idx = _get_rebalancing_dates(datelist, rebalance_frequency, 6)
    first_rebal_idx = rebal_idx[0] if rebal_idx else 0
    n_expected = len(char_df) - first_rebal_idx - 1  # All dates after first rebal

    n_char_dates = char_df.iloc[:, 0].notna().sum()
    has_bug = n_char_dates < n_expected

    if verbose:
        print(f"\n{'='*60}")
        print(f"Characteristics Test: {rebalance_frequency}")
        print(f"{'='*60}")
        print(f"Expected char dates: {n_expected}")
        print(f"Actual char dates:   {n_char_dates}")
        print(f"Bug present:         {'YES!' if has_bug else 'No'}")

        print(f"\nChar1 (portfolio 1) by date:")
        for date, val in char_df.iloc[:, 0].items():
            status = f"{val:.4f}" if pd.notna(val) else "NaN"
            print(f"  {date.strftime('%Y-%m')}: {status}")

    return {
        'n_char_dates': n_char_dates,
        'n_expected': n_expected,
        'has_bug': has_bug,
    }


def main():
    print("="*70)
    print("NON-STAGGERED REBALANCING BUG VALIDATION")
    print("="*70)
    print()
    print("Testing whether returns are computed EVERY month or only at")
    print("rebalancing dates (which is the bug).")
    print()

    # Create test data
    print("Creating balanced test data (12 months, 50 bonds)...")
    data = create_balanced_test_data(n_dates=12, n_bonds=50, seed=42)
    print(f"  Data shape: {data.shape}")
    print(f"  Dates: {data['date'].min().strftime('%Y-%m')} to {data['date'].max().strftime('%Y-%m')}")

    # Test quarterly rebalancing (without turnover/chars - uses fast path)
    print("\n" + "="*70)
    print("TEST 1: Quarterly rebalancing, returns only (fast path)")
    print("="*70)
    result = test_nonstaggered_monthly_returns(data, 'quarterly', verbose=True)

    # Test with turnover (slow path)
    print("\n" + "="*70)
    print("TEST 2: Quarterly rebalancing, with turnover (slow path)")
    print("="*70)
    result_turnover = test_with_turnover(data, 'quarterly', verbose=True)

    # Test with chars (slow path)
    print("\n" + "="*70)
    print("TEST 3: Quarterly rebalancing, with chars (slow path)")
    print("="*70)
    result_chars = test_with_chars(data, 'quarterly', verbose=True)

    # Test with unbalanced data (bonds dropping out)
    print("\n" + "="*70)
    print("TEST 4: Unbalanced data (10% random dropouts)")
    print("="*70)
    data_unbalanced = create_unbalanced_test_data(n_dates=12, n_bonds=50, seed=42, dropout_rate=0.1)
    print(f"  Data shape: {data_unbalanced.shape}")
    result_unbalanced = test_nonstaggered_monthly_returns(data_unbalanced, 'quarterly', verbose=True)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    all_bugs = [
        ("Fast path (returns only)", result['has_bug']),
        ("Slow path (turnover)", result_turnover['has_bug']),
        ("Slow path (chars)", result_chars['has_bug']),
        ("Unbalanced data", result_unbalanced['has_bug']),
    ]

    n_bugs = sum(1 for _, has_bug in all_bugs if has_bug)
    print(f"\nBugs detected: {n_bugs}/{len(all_bugs)}")

    for name, has_bug in all_bugs:
        status = "BUG!" if has_bug else "OK"
        print(f"  {name}: {status}")

    if n_bugs > 0:
        print("\n" + "!"*70)
        print("BUG CONFIRMED: Non-staggered rebalancing only returns values at")
        print("rebalancing dates instead of EVERY month!")
        print("!"*70)
        return 1
    else:
        print("\nNo bugs detected.")
        return 0


if __name__ == '__main__':
    sys.exit(main())
