#!/usr/bin/env python3
"""
Validate that SingleSort and BatchStrategyFormation produce identical results
for non-staggered rebalancing after the dynamic_weights fix.

For non-staggered rebalancing:
- dynamic_weights should NOT apply
- VW always comes from formation date (true rebalancing date)
- Weights are renormalized within each portfolio on pseudo-rebalancing dates
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import PyBondLab as pbl
from PyBondLab.pbl_test import generate_synthetic_data

def validate_nonstaggered(data, rebalance_frequency, rebalance_month, signal_col='signal1'):
    """Compare SingleSort and BatchStrategyFormation for non-staggered rebalancing."""

    # 1. SingleSort with StrategyFormation
    strategy = pbl.SingleSort(
        holding_period=1,
        sort_var=signal_col,
        num_portfolios=5,
        rebalance_frequency=rebalance_frequency,
        rebalance_month=rebalance_month,
    )

    sf = pbl.StrategyFormation(
        data=data,
        strategy=strategy,
        turnover=False,
        verbose=False,
    )
    sf_result = sf.fit()
    sf_ew_ls, sf_vw_ls = sf_result.get_long_short()

    # 2. BatchStrategyFormation
    batch = pbl.BatchStrategyFormation(
        data=data,
        signals=[signal_col],
        holding_period=1,
        num_portfolios=5,
        rebalance_frequency=rebalance_frequency,
        rebalance_month=rebalance_month,
        turnover=False,
        n_jobs=1,
        verbose=False,
    )
    batch_results = batch.fit()
    batch_ew_ls, batch_vw_ls = batch_results.results[signal_col].get_long_short()

    # 3. Compare
    # Align indices (some dates may be NaN)
    common_dates = sf_ew_ls.dropna().index.intersection(batch_ew_ls.dropna().index)

    ew_diff = np.abs(sf_ew_ls.loc[common_dates] - batch_ew_ls.loc[common_dates]).max()
    vw_diff = np.abs(sf_vw_ls.loc[common_dates] - batch_vw_ls.loc[common_dates]).max()

    return {
        'n_dates': len(common_dates),
        'ew_diff': ew_diff,
        'vw_diff': vw_diff,
        'sf_ew_mean': sf_ew_ls.mean(),
        'batch_ew_mean': batch_ew_ls.mean(),
        'sf_vw_mean': sf_vw_ls.mean(),
        'batch_vw_mean': batch_vw_ls.mean(),
    }


def main():
    print("=" * 70)
    print("VALIDATING NON-STAGGERED REBALANCING: dynamic_weights FIX")
    print("=" * 70)
    print()
    print("After fix, both SingleSort and BatchStrategyFormation should:")
    print("  - Use VW from formation date (not d-1)")
    print("  - Produce identical results for all rebalancing options")
    print()

    # Generate unbalanced test data (with missing observations)
    np.random.seed(42)
    data = generate_synthetic_data(n_dates=24, n_bonds=200, seed=42)

    # Make it unbalanced by randomly dropping some observations
    drop_mask = np.random.random(len(data)) > 0.1  # Keep ~90%
    data = data[drop_mask].copy()
    print(f"Test data: {len(data)} rows ({len(data['date'].unique())} dates, "
          f"{len(data['ID'].unique())} bonds)")
    print()

    # Test configurations
    configs = [
        ('quarterly', 1),
        ('quarterly', 3),
        ('semi-annual', 1),
        ('semi-annual', 6),
        ('annual', 1),
        ('annual', 6),
        (3, 1),  # Integer rebalance_frequency
        (6, 1),
        (12, 1),
    ]

    print("Testing configurations...")
    print("-" * 70)

    all_pass = True
    for rebal_freq, rebal_month in configs:
        result = validate_nonstaggered(data, rebal_freq, rebal_month)

        is_match = result['ew_diff'] < 1e-10 and result['vw_diff'] < 1e-10
        status = "PASS" if is_match else "FAIL"

        if not is_match:
            all_pass = False

        freq_str = str(rebal_freq) if isinstance(rebal_freq, int) else f"'{rebal_freq}'"
        print(f"rebalance_frequency={freq_str:14}, rebalance_month={rebal_month}: "
              f"{status}")
        print(f"  Dates: {result['n_dates']}, EW diff: {result['ew_diff']:.2e}, "
              f"VW diff: {result['vw_diff']:.2e}")

        if not is_match:
            print(f"  SF EW mean:    {result['sf_ew_mean']:.8f}")
            print(f"  Batch EW mean: {result['batch_ew_mean']:.8f}")
            print(f"  SF VW mean:    {result['sf_vw_mean']:.8f}")
            print(f"  Batch VW mean: {result['batch_vw_mean']:.8f}")
        print()

    print("=" * 70)
    if all_pass:
        print("ALL TESTS PASSED!")
        print("SingleSort and BatchStrategyFormation produce identical results")
        print("for all non-staggered rebalancing configurations.")
    else:
        print("SOME TESTS FAILED!")
        print("Please investigate the discrepancies.")
    print("=" * 70)

    return all_pass


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
