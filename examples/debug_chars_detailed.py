#!/usr/bin/env python
"""
Detailed debug script for characteristics computation.
Computes manually and compares with both paths.
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '.')

import PyBondLab as pbl
from PyBondLab.pbl_test import generate_synthetic_data_fast


def main():
    # Generate test data - match validation script exactly
    data = generate_synthetic_data_fast(
        n_dates=120,
        n_bonds=300,
        seed=42,
        balanced_panel=False,
        n_chars=3,
    )

    print(f"Data shape: {data.shape}")

    dates = sorted(data['date'].unique())
    june_date = dates[5]
    july_date = dates[6]

    print(f"\nJune date: {june_date}")
    print(f"July date: {july_date}")

    # Compute ranks manually using same logic as slow path
    june_data = data[data['date'] == june_date].copy()
    june_data['rank'] = pd.qcut(june_data['signal1'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    june_data['rank'] = june_data['rank'].astype(int)

    print("\nRanks at June (first 10 bonds):")
    print(june_data[['ID', 'signal1', 'rank']].head(10))

    # Bonds in Portfolio 1
    ptf1_bonds = june_data[june_data['rank'] == 1]['ID'].values
    print(f"\nPortfolio 1 bonds: {len(ptf1_bonds)}")
    print(ptf1_bonds[:5])

    # Get characteristics at June for Portfolio 1 bonds
    june_chars_ptf1 = data[(data['date'] == june_date) & (data['ID'].isin(ptf1_bonds))][['ID', 'char1']]
    print(f"\nJune char1 for Portfolio 1 (first 5 bonds):")
    print(june_chars_ptf1.head())
    print(f"\nEW average of June char1 for Portfolio 1: {june_chars_ptf1['char1'].mean():.6f}")

    # Get characteristics at July for Portfolio 1 bonds (this is what fast path might be using wrongly)
    july_chars_ptf1 = data[(data['date'] == july_date) & (data['ID'].isin(ptf1_bonds))][['ID', 'char1']]
    print(f"\nJuly char1 for Portfolio 1 (first 5 bonds):")
    print(july_chars_ptf1.head())
    print(f"\nEW average of July char1 for Portfolio 1: {july_chars_ptf1['char1'].mean():.6f}")

    # Now run slow path and compare
    print("\n" + "=" * 60)
    print("Running slow path...")
    print("=" * 60)

    strategy = pbl.SingleSort(
        holding_period=12,
        sort_var='signal1',
        num_portfolios=5,
        rebalance_frequency='annual',
        rebalance_month=6,
    )

    from PyBondLab.config import StrategyFormationConfig, DataConfig, FormationConfig

    config = StrategyFormationConfig(
        data=DataConfig(chars=['char1']),
        formation=FormationConfig(
            compute_turnover=False,
            verbose=False,
        )
    )

    sf_slow = pbl.StrategyFormation(
        data=data,
        strategy=strategy,
        config=config,
        verbose=False,
    )
    sf_slow._can_use_nonstaggered_fast_path = lambda: False
    slow_result = sf_slow.fit()

    slow_chars = slow_result.get_characteristics()
    slow_ew_c, _ = slow_chars

    print(f"\nSlow path July (d=6) EW char1 Portfolio 1: {slow_ew_c['char1'].iloc[6, 0]:.6f}")
    print(f"Slow path Aug (d=7) EW char1 Portfolio 1: {slow_ew_c['char1'].iloc[7, 0]:.6f}")

    # Run fast path
    print("\n" + "=" * 60)
    print("Running fast path...")
    print("=" * 60)

    sf_fast = pbl.StrategyFormation(
        data=data,
        strategy=strategy,
        config=config,
        verbose=False,
    )
    fast_result = sf_fast.fit()

    fast_chars = fast_result.get_characteristics()
    fast_ew_c, _ = fast_chars

    print(f"\nFast path July (d=6) EW char1 Portfolio 1: {fast_ew_c['char1'].iloc[6, 0]:.6f}")
    print(f"Fast path Aug (d=7) EW char1 Portfolio 1: {fast_ew_c['char1'].iloc[7, 0]:.6f}")

    # Compare with manual calculation
    print("\n" + "=" * 60)
    print("Comparison:")
    print("=" * 60)
    print(f"Manual: June char1 for Ptf1 = {june_chars_ptf1['char1'].mean():.6f}")
    print(f"Manual: July char1 for Ptf1 = {july_chars_ptf1['char1'].mean():.6f}")
    print(f"Slow path July output = {slow_ew_c['char1'].iloc[6, 0]:.6f}")
    print(f"Fast path July output = {fast_ew_c['char1'].iloc[6, 0]:.6f}")

    # Check if slow path July matches manual June
    if abs(slow_ew_c['char1'].iloc[6, 0] - june_chars_ptf1['char1'].mean()) < 1e-6:
        print("\n✓ Slow path July uses June's chars (as expected)")
    else:
        print("\n✗ Slow path July does NOT match June's chars!")

    # Check if fast path July matches manual June or July
    if abs(fast_ew_c['char1'].iloc[6, 0] - june_chars_ptf1['char1'].mean()) < 1e-6:
        print("✓ Fast path July uses June's chars (correct)")
    elif abs(fast_ew_c['char1'].iloc[6, 0] - july_chars_ptf1['char1'].mean()) < 1e-6:
        print("✗ Fast path July uses July's chars (WRONG - should use June)")
    else:
        print("✗ Fast path July uses neither June nor July chars (unexpected)")

    # Check if fast path Aug matches slow path July
    if abs(fast_ew_c['char1'].iloc[7, 0] - slow_ew_c['char1'].iloc[6, 0]) < 1e-6:
        print("\n✗ Fast path Aug = Slow path July (1-month shift confirmed)")


if __name__ == '__main__':
    main()
