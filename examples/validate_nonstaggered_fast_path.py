#!/usr/bin/env python3
"""
Validation script for Phase 15: Non-staggered rebalancing optimization.

Compares the fast numba path vs the slow pandas path for non-staggered rebalancing.

Test matrix:
- rebalance_frequency: 3 (quarterly), 6 (semi-annual), 12 (annual)
- rebalance_month: 1, 6, 12
"""

import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, '.')

import PyBondLab as pbl


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
            })

    data = pd.DataFrame(data_list)
    data['date'] = pd.to_datetime(data['date'])

    return data


def run_slow_path(data, rebalance_frequency, rebalance_month, holding_period, nport=5):
    """Run the slow (pandas) path by forcing turnover=True."""
    strategy = pbl.SingleSort(
        holding_period=holding_period,
        sort_var='signal',
        num_portfolios=nport,
        rebalance_frequency=rebalance_frequency,
        rebalance_month=rebalance_month,
        verbose=False,
    )

    # Force slow path by enabling turnover
    sf = pbl.StrategyFormation(
        data=data,
        strategy=strategy,
        turnover=True,  # Forces slow path
        verbose=False,
    )

    start_time = time.time()
    result = sf.fit()
    elapsed = time.time() - start_time

    ew_ls, vw_ls = result.get_long_short()
    return ew_ls, vw_ls, elapsed


def run_fast_path(data, rebalance_frequency, rebalance_month, holding_period, nport=5):
    """Run the fast (numba) path."""
    strategy = pbl.SingleSort(
        holding_period=holding_period,
        sort_var='signal',
        num_portfolios=nport,
        rebalance_frequency=rebalance_frequency,
        rebalance_month=rebalance_month,
        verbose=False,
    )

    # Fast path: turnover=False, chars=None, banding=None
    sf = pbl.StrategyFormation(
        data=data,
        strategy=strategy,
        turnover=False,
        verbose=False,
    )

    start_time = time.time()
    result = sf.fit()
    elapsed = time.time() - start_time

    ew_ls, vw_ls = result.get_long_short()
    return ew_ls, vw_ls, elapsed


def compare_results(ew_slow, vw_slow, ew_fast, vw_fast, tolerance=1e-10):
    """Compare slow vs fast path results."""
    # Align indices
    common_dates = ew_slow.index.intersection(ew_fast.index)

    ew_slow_aligned = ew_slow.loc[common_dates]
    ew_fast_aligned = ew_fast.loc[common_dates]
    vw_slow_aligned = vw_slow.loc[common_dates]
    vw_fast_aligned = vw_fast.loc[common_dates]

    # Only compare non-NaN values
    ew_mask = ew_slow_aligned.notna() & ew_fast_aligned.notna()
    vw_mask = vw_slow_aligned.notna() & vw_fast_aligned.notna()

    if ew_mask.sum() == 0:
        return {'ew_match': False, 'vw_match': False, 'ew_diff': np.nan, 'vw_diff': np.nan,
                'n_ew': 0, 'n_vw': 0}

    ew_diff = np.abs(ew_slow_aligned[ew_mask].values - ew_fast_aligned[ew_mask].values).max()
    vw_diff = np.abs(vw_slow_aligned[vw_mask].values - vw_fast_aligned[vw_mask].values).max()

    return {
        'ew_match': ew_diff < tolerance,
        'vw_match': vw_diff < tolerance,
        'ew_diff': ew_diff,
        'vw_diff': vw_diff,
        'n_ew': ew_mask.sum(),
        'n_vw': vw_mask.sum(),
    }


def run_validation():
    """Run the full validation."""
    print("="*80)
    print("PHASE 15 VALIDATION: Non-Staggered Rebalancing Fast Path")
    print("="*80)

    # Generate test data
    print("\nGenerating test data...")
    data = generate_test_data(n_dates=120, n_bonds=300, seed=42)
    print(f"  Data shape: {data.shape}")
    print(f"  Date range: {data['date'].min()} to {data['date'].max()}")

    # Test configurations
    configs = [
        # (rebalance_frequency, rebalance_month, holding_period)
        (12, 6, 12),   # Annual, June
        (12, 1, 12),   # Annual, January
        (12, 12, 12),  # Annual, December
        (6, 6, 6),     # Semi-annual, June
        (6, 1, 6),     # Semi-annual, January
        (3, 6, 3),     # Quarterly, June
        (3, 3, 3),     # Quarterly, March
    ]

    results = []

    print("\n" + "-"*80)
    print("Running validation tests...")
    print("-"*80)

    # Warmup JIT
    print("\nWarming up JIT...")
    warmup_data = generate_test_data(n_dates=24, n_bonds=50, seed=999)
    run_fast_path(warmup_data, 12, 6, 12)
    print("  JIT warmup complete.")

    for freq, month, hp in configs:
        freq_name = {3: 'quarterly', 6: 'semi-annual', 12: 'annual'}[freq]
        print(f"\n[{freq_name}] freq={freq}, month={month}, hp={hp}")

        # Run slow path
        ew_slow, vw_slow, slow_time = run_slow_path(data, freq, month, hp)

        # Run fast path
        ew_fast, vw_fast, fast_time = run_fast_path(data, freq, month, hp)

        # Compare results
        comparison = compare_results(ew_slow, vw_slow, ew_fast, vw_fast)

        speedup = slow_time / fast_time if fast_time > 0 else 0

        status = "PASS" if comparison['ew_match'] and comparison['vw_match'] else "FAIL"
        print(f"  Slow: {slow_time:.3f}s, Fast: {fast_time:.3f}s, Speedup: {speedup:.1f}x")
        print(f"  EW diff: {comparison['ew_diff']:.2e}, VW diff: {comparison['vw_diff']:.2e}")
        print(f"  N comparisons: EW={comparison['n_ew']}, VW={comparison['n_vw']}")
        print(f"  Status: {status}")

        results.append({
            'freq': freq,
            'month': month,
            'hp': hp,
            'slow_time': slow_time,
            'fast_time': fast_time,
            'speedup': speedup,
            'ew_diff': comparison['ew_diff'],
            'vw_diff': comparison['vw_diff'],
            'status': status,
        })

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    df = pd.DataFrame(results)
    print(df[['freq', 'month', 'hp', 'slow_time', 'fast_time', 'speedup', 'status']].to_string(index=False))

    n_pass = sum(1 for r in results if r['status'] == 'PASS')
    n_fail = len(results) - n_pass

    print(f"\n{n_pass}/{len(results)} tests PASSED")
    if n_fail > 0:
        print(f"  {n_fail} tests FAILED")

    avg_speedup = df['speedup'].mean()
    print(f"Average speedup: {avg_speedup:.1f}x")

    return results


if __name__ == '__main__':
    results = run_validation()
