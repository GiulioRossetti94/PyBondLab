#!/usr/bin/env python3
"""
Validation script for non-staggered rebalancing (quarterly, semi-annual, annual).

This script tests and documents the behavior of SingleSort and DoubleSort
with non-monthly rebalancing frequencies.

Test Matrix:
- rebalance_frequency: 3 (quarterly), 6 (semi-annual), 12 (annual)
- rebalance_month: 1 (January), 6 (June), 12 (December)
- holding_period: matches rebalance_frequency (3, 6, 12)
- turnover: True/False
- chars: None/['char1']
- banding: None/1

The goal is to understand the current implementation and validate correctness
before optimizing for speed.
"""

import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

sys.path.insert(0, '.')

import PyBondLab as pbl
from PyBondLab.pbl_test import generate_synthetic_data


def generate_test_data(n_dates: int = 120, n_bonds: int = 300, seed: int = 42):
    """Generate synthetic data for testing."""
    np.random.seed(seed)

    # Generate dates (10 years of monthly data)
    dates = pd.date_range('2010-01-31', periods=n_dates, freq='ME')

    # Generate bonds
    bond_ids = [f'BOND_{i:04d}' for i in range(n_bonds)]

    # Create panel
    data_list = []
    for date in dates:
        # Randomly sample bonds for this date (80-100% of bonds)
        n_active = int(n_bonds * (0.8 + 0.2 * np.random.rand()))
        active_bonds = np.random.choice(bond_ids, n_active, replace=False)

        for bond in active_bonds:
            data_list.append({
                'date': date,
                'ID': bond,
                'ret': np.random.randn() * 0.05,  # ~5% monthly vol
                'VW': np.exp(np.random.randn()) * 1e6,  # Log-normal market cap
                'RATING_NUM': np.random.randint(1, 23),
                'signal': np.random.randn(),  # Random signal
                'char1': np.random.randn(),
                'char2': np.random.randn(),
            })

    data = pd.DataFrame(data_list)
    data['date'] = pd.to_datetime(data['date'])

    return data


def run_single_test(
    data: pd.DataFrame,
    rebalance_frequency: int,
    rebalance_month: int,
    turnover: bool = False,
    chars: List[str] = None,
    banding: int = None,
    num_portfolios: int = 5,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run a single test configuration and return results."""

    # Use matching holding period for simplicity
    holding_period = rebalance_frequency

    strategy = pbl.SingleSort(
        holding_period=holding_period,
        sort_var='signal',
        num_portfolios=num_portfolios,
        rebalance_frequency=rebalance_frequency,
        rebalance_month=rebalance_month,
        verbose=verbose,
    )

    start_time = time.time()

    sf = pbl.StrategyFormation(
        data=data,
        strategy=strategy,
        turnover=turnover,
        chars=chars,
        banding_threshold=banding / num_portfolios if banding else None,
        verbose=verbose,
    )

    result = sf.fit()
    elapsed = time.time() - start_time

    # Extract results
    ew_ls, vw_ls = result.get_long_short()

    # Count rebalancing dates
    datelist = sorted(data['date'].unique())
    freq_map = {3: 'quarterly', 6: 'semi-annual', 12: 'annual'}
    freq_str = freq_map.get(rebalance_frequency, rebalance_frequency)

    from PyBondLab.utils_optimized import _get_rebalancing_dates
    rebal_dates_idx = _get_rebalancing_dates(datelist, freq_str, rebalance_month)

    output = {
        'rebalance_frequency': rebalance_frequency,
        'rebalance_month': rebalance_month,
        'holding_period': holding_period,
        'turnover': turnover,
        'chars': chars is not None,
        'banding': banding,
        'n_rebal_dates': len(rebal_dates_idx),
        'n_return_dates': ew_ls.notna().sum(),
        'ew_mean': ew_ls.mean(),
        'vw_mean': vw_ls.mean(),
        'ew_std': ew_ls.std(),
        'elapsed': elapsed,
    }

    # Get turnover if computed
    if turnover:
        try:
            ew_turn, vw_turn = result.get_turnover()
            if ew_turn is not None:
                output['turnover_ew_mean'] = ew_turn.mean().mean()
            else:
                output['turnover_ew_mean'] = np.nan
        except:
            output['turnover_ew_mean'] = np.nan

    # Get characteristics if computed
    if chars:
        try:
            ew_chars, vw_chars = result.get_characteristics()
            if ew_chars is not None:
                output['char1_ew_mean'] = ew_chars[chars[0]].mean().mean()
            else:
                output['char1_ew_mean'] = np.nan
        except:
            output['char1_ew_mean'] = np.nan

    return output, result, ew_ls, vw_ls


def analyze_rebalancing_dates(data: pd.DataFrame, rebalance_frequency: int, rebalance_month: int):
    """Analyze which dates are selected for rebalancing."""
    datelist = sorted(data['date'].unique())

    freq_map = {3: 'quarterly', 6: 'semi-annual', 12: 'annual'}
    freq_str = freq_map.get(rebalance_frequency, rebalance_frequency)

    from PyBondLab.utils_optimized import _get_rebalancing_dates
    rebal_dates_idx = _get_rebalancing_dates(datelist, freq_str, rebalance_month)

    print(f"\n{'='*60}")
    print(f"Rebalancing Analysis: freq={rebalance_frequency}, month={rebalance_month}")
    print(f"{'='*60}")
    print(f"Total dates in data: {len(datelist)}")
    print(f"Rebalancing dates: {len(rebal_dates_idx)}")
    print()

    print("Rebalancing dates (formation dates):")
    for i, idx in enumerate(rebal_dates_idx[:12]):  # Show first 12
        date = datelist[idx]
        print(f"  {i+1:2d}. {date.strftime('%Y-%m-%d')} (month={date.month})")

    if len(rebal_dates_idx) > 12:
        print(f"  ... and {len(rebal_dates_idx) - 12} more")

    return rebal_dates_idx


def analyze_return_coverage(ew_ls: pd.Series, rebalance_frequency: int, rebalance_month: int):
    """Analyze the return time series coverage."""
    print(f"\nReturn Series Coverage:")
    print(f"  Total dates: {len(ew_ls)}")
    print(f"  Non-NaN dates: {ew_ls.notna().sum()}")
    print(f"  First non-NaN: {ew_ls.first_valid_index()}")
    print(f"  Last non-NaN: {ew_ls.last_valid_index()}")

    # Analyze gaps
    valid_dates = ew_ls.dropna().index
    if len(valid_dates) > 1:
        gaps = (valid_dates[1:] - valid_dates[:-1]).days
        print(f"  Avg gap between returns: {np.mean(gaps):.1f} days")
        print(f"  Min gap: {np.min(gaps)} days, Max gap: {np.max(gaps)} days")


def test_timeline_for_annual_rebalancing(data: pd.DataFrame):
    """
    Detailed test to understand the timeline for annual rebalancing.

    For rebalance_frequency=12 (annual), rebalance_month=6 (June):
    - Portfolio is FORMED in June using signal from June
    - Portfolio is HELD for 12 months (July through next June)
    - Returns are collected for those 12 holding months
    """
    print("\n" + "="*80)
    print("DETAILED TIMELINE TEST: Annual Rebalancing (June)")
    print("="*80)

    strategy = pbl.SingleSort(
        holding_period=12,
        sort_var='signal',
        num_portfolios=5,
        rebalance_frequency='annual',
        rebalance_month=6,
        verbose=False,
    )

    sf = pbl.StrategyFormation(
        data=data,
        strategy=strategy,
        turnover=True,
        verbose=True,
    )

    result = sf.fit()
    ew_ls, vw_ls = result.get_long_short()

    print("\nReturn series (first 24 months):")
    print("-" * 50)
    for date, ret in ew_ls.head(24).items():
        status = "RETURN" if pd.notna(ret) else "NaN"
        print(f"  {date.strftime('%Y-%m')}: {status:8s} {ret if pd.notna(ret) else ''}")

    return result


def test_timeline_for_quarterly_rebalancing(data: pd.DataFrame):
    """
    Detailed test to understand the timeline for quarterly rebalancing.

    For rebalance_frequency=3 (quarterly), rebalance_month=6:
    - Portfolio is FORMED in June, September, December, March
    - Portfolio is HELD for 3 months after each formation
    - Returns are collected for those 3 holding months
    """
    print("\n" + "="*80)
    print("DETAILED TIMELINE TEST: Quarterly Rebalancing (starting June)")
    print("="*80)

    strategy = pbl.SingleSort(
        holding_period=3,
        sort_var='signal',
        num_portfolios=5,
        rebalance_frequency='quarterly',
        rebalance_month=6,  # Starting month
        verbose=False,
    )

    sf = pbl.StrategyFormation(
        data=data,
        strategy=strategy,
        turnover=True,
        verbose=True,
    )

    result = sf.fit()
    ew_ls, vw_ls = result.get_long_short()

    print("\nReturn series (first 24 months):")
    print("-" * 50)
    for date, ret in ew_ls.head(24).items():
        status = "RETURN" if pd.notna(ret) else "NaN"
        print(f"  {date.strftime('%Y-%m')}: {status:8s} {ret if pd.notna(ret) else ''}")

    return result


def run_full_validation():
    """Run the full validation matrix."""
    print("="*80)
    print("NON-STAGGERED REBALANCING VALIDATION")
    print("="*80)

    # Generate test data
    print("\nGenerating test data (120 dates, 300 bonds)...")
    data = generate_test_data(n_dates=120, n_bonds=300, seed=42)
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")

    # Test configurations
    configs = []

    # Core tests: different frequencies and months
    for freq in [3, 6, 12]:
        for month in [1, 6, 12]:
            configs.append({'rebalance_frequency': freq, 'rebalance_month': month})

    # Extended tests with turnover, chars, banding
    configs.append({'rebalance_frequency': 12, 'rebalance_month': 6, 'turnover': True})
    configs.append({'rebalance_frequency': 12, 'rebalance_month': 6, 'chars': ['char1']})
    configs.append({'rebalance_frequency': 12, 'rebalance_month': 6, 'banding': 1})
    configs.append({'rebalance_frequency': 6, 'rebalance_month': 6, 'turnover': True})
    configs.append({'rebalance_frequency': 3, 'rebalance_month': 6, 'turnover': True, 'chars': ['char1']})

    results = []

    print("\n" + "-"*80)
    print("Running validation tests...")
    print("-"*80)

    for i, config in enumerate(configs):
        freq = config.get('rebalance_frequency', 12)
        month = config.get('rebalance_month', 6)
        turnover = config.get('turnover', False)
        chars = config.get('chars', None)
        banding = config.get('banding', None)

        print(f"\n[{i+1}/{len(configs)}] freq={freq}, month={month}, " +
              f"turnover={turnover}, chars={chars is not None}, banding={banding}")

        try:
            output, result, ew_ls, vw_ls = run_single_test(
                data=data,
                rebalance_frequency=freq,
                rebalance_month=month,
                turnover=turnover,
                chars=chars,
                banding=banding,
                verbose=False,
            )

            print(f"  Rebal dates: {output['n_rebal_dates']}, Return dates: {output['n_return_dates']}")
            print(f"  EW mean: {output['ew_mean']:.4f}, VW mean: {output['vw_mean']:.4f}")
            print(f"  Time: {output['elapsed']:.3f}s")

            if turnover and 'turnover_ew_mean' in output:
                print(f"  Turnover EW mean: {output['turnover_ew_mean']:.4f}")

            results.append(output)

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({'error': str(e), **config})

    # Analyze rebalancing dates for key configurations
    print("\n" + "="*80)
    print("REBALANCING DATE ANALYSIS")
    print("="*80)

    analyze_rebalancing_dates(data, 12, 6)  # Annual, June
    analyze_rebalancing_dates(data, 6, 6)   # Semi-annual, June
    analyze_rebalancing_dates(data, 3, 6)   # Quarterly, June

    # Detailed timeline tests
    test_timeline_for_annual_rebalancing(data)
    test_timeline_for_quarterly_rebalancing(data)

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)

    df = pd.DataFrame(results)
    cols = ['rebalance_frequency', 'rebalance_month', 'holding_period',
            'n_rebal_dates', 'n_return_dates', 'ew_mean', 'elapsed']
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False))

    return results, data


if __name__ == '__main__':
    results, data = run_full_validation()
