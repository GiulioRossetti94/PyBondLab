#!/usr/bin/env python3
"""
Validation script for DataUncertaintyAnalysis with non-staggered rebalancing.

Tests that DataUncertaintyAnalysis correctly handles non-monthly rebalancing
(quarterly, semi-annual, annual).
"""

import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, '.')

import PyBondLab as pbl


def generate_test_data(n_dates: int = 120, n_bonds: int = 200, seed: int = 42):
    """Generate synthetic data for testing."""
    np.random.seed(seed)

    dates = pd.date_range('2010-01-31', periods=n_dates, freq='ME')
    bond_ids = [f'BOND_{i:04d}' for i in range(n_bonds)]

    data_list = []
    for date in dates:
        n_active = int(n_bonds * (0.85 + 0.15 * np.random.rand()))
        active_bonds = np.random.choice(bond_ids, n_active, replace=False)

        for bond in active_bonds:
            data_list.append({
                'date': date,
                'ID': bond,
                'ret': np.random.randn() * 0.05,
                'VW': np.exp(np.random.randn()) * 1e6,
                'RATING_NUM': np.random.randint(1, 23),
                'signal1': np.random.randn(),
            })

    data = pd.DataFrame(data_list)
    data['date'] = pd.to_datetime(data['date'])
    return data


def run_dua(data, signal, rebalance_frequency, rebalance_month, holding_periods):
    """Run DataUncertaintyAnalysis."""
    dua = pbl.DataUncertaintyAnalysis(
        data=data,
        signals=[signal],
        holding_periods=holding_periods,
        num_portfolios=5,
        filters=None,  # Just baseline
        include_baseline=True,
        rebalance_frequency=rebalance_frequency,
        rebalance_month=rebalance_month,
        verbose=True,
    )

    start_time = time.time()
    results = dua.fit()
    elapsed = time.time() - start_time

    return results, elapsed


def run_validation():
    """Run the validation."""
    print("=" * 80)
    print("DATA UNCERTAINTY ANALYSIS: Non-Staggered Rebalancing Validation")
    print("=" * 80)

    # Generate test data
    print("\nGenerating test data...")
    data = generate_test_data(n_dates=120, n_bonds=200, seed=42)
    print(f"  Data shape: {data.shape}")
    print(f"  Date range: {data['date'].min()} to {data['date'].max()}")

    # Test configurations
    configs = [
        # (rebalance_frequency, rebalance_month, holding_periods)
        ('annual', 6, [12]),
        ('semi-annual', 6, [6]),
        ('quarterly', 6, [3]),
    ]

    results_summary = []

    print("\n" + "-" * 80)
    print("Running validation tests...")
    print("-" * 80)

    for freq, month, hps in configs:
        print(f"\n[{freq}] freq={freq}, month={month}, hp={hps}")

        results, elapsed = run_dua(data, 'signal1', freq, month, hps)

        # Check results
        summary = results.summary()
        n_configs = len(summary)

        hp = hps[0]
        col = f'signal1_hp{hp}_baseline'

        if col in results.ew_ea.columns:
            ew_mean = results.ew_ea[col].mean()
            n_obs = results.ew_ea[col].notna().sum()
            status = "PASS"
        else:
            ew_mean = np.nan
            n_obs = 0
            status = "FAIL"

        print(f"  Time: {elapsed:.3f}s")
        print(f"  N configs: {n_configs}")
        print(f"  EW mean: {ew_mean:.4f}")
        print(f"  N observations: {n_obs}")
        print(f"  Status: {status}")

        results_summary.append({
            'freq': freq,
            'month': month,
            'hp': hp,
            'elapsed': elapsed,
            'n_configs': n_configs,
            'ew_mean': ew_mean,
            'n_obs': n_obs,
            'status': status,
        })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    df = pd.DataFrame(results_summary)
    print(df.to_string(index=False))

    n_pass = sum(1 for r in results_summary if r['status'] == 'PASS')
    n_fail = len(results_summary) - n_pass

    print(f"\n{n_pass}/{len(results_summary)} tests PASSED")
    if n_fail > 0:
        print(f"  {n_fail} tests FAILED")

    return results_summary


if __name__ == '__main__':
    run_validation()
