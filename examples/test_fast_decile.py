# -*- coding: utf-8 -*-
"""
Test script to validate fast_decile_portfolios against PyBondLab.

Generates synthetic data and compares results.
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/user/PyBondLab-Dev')

from examples.fast_decile_portfolios import (
    form_portfolios_single_signal,
    form_all_portfolios,
    get_signal_columns,
    RETURN_COL, WEIGHT_COL, N_PORTFOLIOS
)


def generate_test_data(n_dates=120, n_bonds=500, n_signals=10, seed=42):
    """Generate synthetic bond panel data."""
    np.random.seed(seed)

    dates = pd.date_range('2010-01-31', periods=n_dates, freq='M')

    rows = []
    for t, date in enumerate(dates):
        # Random number of bonds each month
        n_active = int(n_bonds * np.random.uniform(0.7, 0.95))
        active_bonds = np.random.choice(n_bonds, n_active, replace=False)

        for bond_id in active_bonds:
            row = {
                'date': date,
                'cusip': f'BOND{bond_id:04d}',
                'permno': bond_id // 10,  # Group bonds into firms
                'r_1': np.random.normal(0.005, 0.03),  # ~0.5% monthly return
                'r_1_exc': np.random.normal(0.002, 0.025),
                'r_1_dur': np.random.normal(0.003, 0.02),
                'mv': np.random.uniform(10, 1000),  # Market value
            }

            # Add signal columns
            for s in range(n_signals):
                sig_name = f'signal_{s+1}'
                # Create some structure in signals
                row[sig_name] = np.random.normal(0, 1) + 0.1 * (bond_id % 10)

            rows.append(row)

    data = pd.DataFrame(rows)

    # Add some NaN values (~2%)
    for col in [c for c in data.columns if c.startswith('signal_')]:
        mask = np.random.random(len(data)) < 0.02
        data.loc[mask, col] = np.nan

    return data


def test_against_pybondlab(verbose=True):
    """Test fast method against PyBondLab."""
    import PyBondLab as pbl
    from PyBondLab import StrategyFormation, SingleSort
    from PyBondLab.config import StrategyFormationConfig, FormationConfig, DataConfig

    print("="*60)
    print("VALIDATION TEST: fast_decile_portfolios vs PyBondLab")
    print("="*60)

    # Generate test data
    print("\n1. Generating test data...")
    data = generate_test_data(n_dates=60, n_bonds=300, n_signals=5)
    print(f"   Data shape: {data.shape}")

    # Prepare data for PyBondLab
    pbl_data = data.copy()
    pbl_data = pbl_data.rename(columns={'cusip': 'ID', 'mv': 'VW', 'r_1': 'ret'})
    pbl_data['RATING_NUM'] = 5  # Dummy rating

    signal_cols = [f'signal_{i+1}' for i in range(5)]

    print("\n2. Testing each signal...")
    all_passed = True
    results = []

    for signal in signal_cols:
        # Fast method
        ew_fast, vw_fast = form_portfolios_single_signal(
            data, signal, 'r_1', 'mv', 10
        )

        # PyBondLab
        strategy = SingleSort(
            holding_period=1,
            sort_var=signal,
            num_portfolios=10
        )

        config = StrategyFormationConfig(
            data=DataConfig(),
            formation=FormationConfig(
                dynamic_weights=True,
                compute_turnover=False,
                verbose=False
            )
        )

        sf = StrategyFormation(data=pbl_data, strategy=strategy, config=config)
        result = sf.fit()
        ew_pbl, vw_pbl = result.get_long_short()

        # Align dates
        common_dates = ew_fast.index.intersection(ew_pbl.index)

        if len(common_dates) == 0:
            print(f"   {signal}: No overlapping dates!")
            all_passed = False
            continue

        ew_fast_aligned = ew_fast.loc[common_dates]
        vw_fast_aligned = vw_fast.loc[common_dates]
        ew_pbl_aligned = ew_pbl.loc[common_dates]
        vw_pbl_aligned = vw_pbl.loc[common_dates]

        # Compute differences
        ew_diff = (ew_fast_aligned - ew_pbl_aligned).abs()
        vw_diff = (vw_fast_aligned - vw_pbl_aligned).abs()

        ew_max_diff = ew_diff.max()
        vw_max_diff = vw_diff.max()

        tol = 1e-10
        ew_pass = ew_max_diff < tol
        vw_pass = vw_max_diff < tol

        status = "PASS" if (ew_pass and vw_pass) else "FAIL"

        if verbose:
            print(f"   {signal}: {status}")
            print(f"      EW max diff: {ew_max_diff:.2e} ({'OK' if ew_pass else 'MISMATCH'})")
            print(f"      VW max diff: {vw_max_diff:.2e} ({'OK' if vw_pass else 'MISMATCH'})")
            print(f"      EW mean: fast={ew_fast_aligned.mean()*100:.4f}%, pbl={ew_pbl_aligned.mean()*100:.4f}%")
            print(f"      VW mean: fast={vw_fast_aligned.mean()*100:.4f}%, pbl={vw_pbl_aligned.mean()*100:.4f}%")

        results.append({
            'signal': signal,
            'ew_max_diff': ew_max_diff,
            'vw_max_diff': vw_max_diff,
            'ew_pass': ew_pass,
            'vw_pass': vw_pass,
        })

        if not (ew_pass and vw_pass):
            all_passed = False

    print("\n" + "="*60)
    print(f"RESULT: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("="*60)

    return all_passed, results


def test_speed(n_signals=50, n_dates=120, n_bonds=500):
    """Test speed of the fast method."""
    print("\n" + "="*60)
    print("SPEED TEST")
    print("="*60)

    print(f"\nGenerating data: {n_dates} dates, {n_bonds} bonds, {n_signals} signals...")
    data = generate_test_data(n_dates=n_dates, n_bonds=n_bonds, n_signals=n_signals)
    print(f"Data shape: {data.shape}")

    signal_cols = [f'signal_{i+1}' for i in range(n_signals)]

    # Time the fast method
    print(f"\nProcessing {n_signals} signals...")
    t_start = time.time()

    ew_df, vw_df = form_all_portfolios(
        data, signal_cols,
        return_col='r_1',
        weight_col='mv',
        n_portfolios=10,
        sign_correct=True,
        verbose=False
    )

    elapsed = time.time() - t_start

    print(f"Completed in {elapsed:.2f}s ({n_signals / elapsed:.1f} signals/sec)")
    print(f"Output shapes: EW={ew_df.shape}, VW={vw_df.shape}")

    return elapsed


if __name__ == "__main__":
    import time

    # Run validation test
    passed, results = test_against_pybondlab(verbose=True)

    # Run speed test
    test_speed(n_signals=50, n_dates=120, n_bonds=500)

    print("\n" + "="*60)
    print("DONE")
    print("="*60)
