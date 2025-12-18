#!/usr/bin/env python3
"""
Validation script for BatchStrategyFormation with non-staggered rebalancing.

Compares:
1. Fast batch path vs slow batch path for non-staggered rebalancing
2. BatchStrategyFormation vs StrategyFormation for consistency

Test matrix:
- rebalance_frequency: 3 (quarterly), 6 (semi-annual), 12 (annual)
- rebalance_month: 1, 6
- Multiple signals
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
                'signal1': np.random.randn(),  # Random signal
                'signal2': np.random.randn() * 2,  # Different scale
                'signal3': np.random.randn() - 0.5,  # Shifted
            })

    data = pd.DataFrame(data_list)
    data['date'] = pd.to_datetime(data['date'])

    return data


def run_batch_fast(data, signals, rebalance_frequency, rebalance_month, holding_period, nport=5):
    """Run batch with fast path (turnover=False)."""
    batch = pbl.BatchStrategyFormation(
        data=data,
        signals=signals,
        holding_period=holding_period,
        num_portfolios=nport,
        rebalance_frequency=rebalance_frequency,
        rebalance_month=rebalance_month,
        turnover=False,  # Fast path
        verbose=False,
    )

    start_time = time.time()
    results = batch.fit()
    elapsed = time.time() - start_time

    return results, elapsed


def run_batch_slow(data, signals, rebalance_frequency, rebalance_month, holding_period, nport=5):
    """Run batch with slow path (turnover=True)."""
    batch = pbl.BatchStrategyFormation(
        data=data,
        signals=signals,
        holding_period=holding_period,
        num_portfolios=nport,
        rebalance_frequency=rebalance_frequency,
        rebalance_month=rebalance_month,
        turnover=True,  # Forces slow path
        verbose=False,
    )

    start_time = time.time()
    results = batch.fit()
    elapsed = time.time() - start_time

    return results, elapsed


def run_single_strategy(data, signal, rebalance_frequency, rebalance_month, holding_period, nport=5):
    """Run single StrategyFormation for comparison."""
    strategy = pbl.SingleSort(
        holding_period=holding_period,
        sort_var=signal,
        num_portfolios=nport,
        rebalance_frequency=rebalance_frequency,
        rebalance_month=rebalance_month,
        verbose=False,
    )

    sf = pbl.StrategyFormation(
        data=data,
        strategy=strategy,
        turnover=False,
        verbose=False,
    )

    result = sf.fit()
    ew_ls, vw_ls = result.get_long_short()
    return ew_ls, vw_ls


def compare_results(fast_results, slow_results, signals, tolerance=1e-10):
    """Compare fast vs slow path results."""
    all_pass = True
    max_diff = 0.0

    for signal in signals:
        # Get fast path results
        fast = fast_results[signal]
        fast_ew, fast_vw = fast.get_long_short()

        # Get slow path results
        slow = slow_results[signal]
        slow_ew, slow_vw = slow.get_long_short()

        # Align indices
        common_dates = fast_ew.index.intersection(slow_ew.index)

        fast_ew_aligned = fast_ew.loc[common_dates]
        slow_ew_aligned = slow_ew.loc[common_dates]
        fast_vw_aligned = fast_vw.loc[common_dates]
        slow_vw_aligned = slow_vw.loc[common_dates]

        # Compare non-NaN values
        ew_mask = fast_ew_aligned.notna() & slow_ew_aligned.notna()
        vw_mask = fast_vw_aligned.notna() & slow_vw_aligned.notna()

        if ew_mask.sum() > 0:
            ew_diff = np.abs(fast_ew_aligned[ew_mask].values - slow_ew_aligned[ew_mask].values).max()
            max_diff = max(max_diff, ew_diff)
            if ew_diff > tolerance:
                all_pass = False

        if vw_mask.sum() > 0:
            vw_diff = np.abs(fast_vw_aligned[vw_mask].values - slow_vw_aligned[vw_mask].values).max()
            max_diff = max(max_diff, vw_diff)
            if vw_diff > tolerance:
                all_pass = False

    return all_pass, max_diff


def run_validation():
    """Run the full validation."""
    print("=" * 80)
    print("BATCH STRATEGY FORMATION: Non-Staggered Rebalancing Validation")
    print("=" * 80)

    # Generate test data
    print("\nGenerating test data...")
    data = generate_test_data(n_dates=120, n_bonds=300, seed=42)
    print(f"  Data shape: {data.shape}")
    print(f"  Date range: {data['date'].min()} to {data['date'].max()}")

    signals = ['signal1', 'signal2', 'signal3']

    # Test configurations
    configs = [
        # (rebalance_frequency, rebalance_month, holding_period)
        (12, 6, 12),   # Annual, June
        (12, 1, 12),   # Annual, January
        (6, 6, 6),     # Semi-annual, June
        (3, 6, 3),     # Quarterly, June
    ]

    results = []

    print("\n" + "-" * 80)
    print("Running validation tests...")
    print("-" * 80)

    # Warmup JIT
    print("\nWarming up JIT...")
    warmup_data = generate_test_data(n_dates=24, n_bonds=50, seed=999)
    run_batch_fast(warmup_data, ['signal1'], 12, 6, 12)
    print("  JIT warmup complete.")

    for freq, month, hp in configs:
        freq_name = {3: 'quarterly', 6: 'semi-annual', 12: 'annual'}[freq]
        print(f"\n[{freq_name}] freq={freq}, month={month}, hp={hp}")

        # Run fast batch path
        fast_results, fast_time = run_batch_fast(data, signals, freq, month, hp)

        # Run slow batch path
        slow_results, slow_time = run_batch_slow(data, signals, freq, month, hp)

        # Compare results
        passed, max_diff = compare_results(fast_results, slow_results, signals)

        speedup = slow_time / fast_time if fast_time > 0 else 0

        status = "PASS" if passed else "FAIL"
        print(f"  Fast: {fast_time:.3f}s, Slow: {slow_time:.3f}s, Speedup: {speedup:.1f}x")
        print(f"  Max diff: {max_diff:.2e}")
        print(f"  Status: {status}")

        # Also compare with single StrategyFormation for first signal
        single_ew, single_vw = run_single_strategy(data, signals[0], freq, month, hp)
        batch_ew, batch_vw = fast_results[signals[0]].get_long_short()

        common = single_ew.index.intersection(batch_ew.index)
        single_match = np.abs(single_ew.loc[common].dropna() - batch_ew.loc[common].dropna()).max()
        print(f"  Batch vs Single StrategyFormation diff: {single_match:.2e}")

        results.append({
            'freq': freq,
            'month': month,
            'hp': hp,
            'fast_time': fast_time,
            'slow_time': slow_time,
            'speedup': speedup,
            'max_diff': max_diff,
            'status': status,
        })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    df = pd.DataFrame(results)
    print(df[['freq', 'month', 'hp', 'fast_time', 'slow_time', 'speedup', 'status']].to_string(index=False))

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
