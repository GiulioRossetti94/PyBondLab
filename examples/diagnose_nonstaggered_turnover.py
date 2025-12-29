#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diagnostic script for non-staggered turnover computation.

This script investigates:
1. How turnover is computed between rebalancing dates
2. Whether "latent turnover" from bond dropouts is counted
3. How weight scaling works (single-period vs cumulative)
4. Comparison with staggered holding cohort behavior

Author: Claude
Date: December 2025
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/user/PyBondLab-Dev')

import PyBondLab as pbl
from PyBondLab import StrategyFormation, SingleSort
from PyBondLab.pbl_test import generate_synthetic_data


def create_balanced_panel(n_dates=12, n_bonds=30, seed=42):
    """Create a balanced panel where all bonds exist at all dates."""
    np.random.seed(seed)

    dates = pd.date_range('2020-01-31', periods=n_dates, freq='ME')

    rows = []
    for date in dates:
        for bond_id in range(1, n_bonds + 1):
            rows.append({
                'date': date,
                'ID': f'BOND_{bond_id:03d}',
                'ret': np.random.randn() * 0.02,  # ~2% monthly vol
                'VW': 1.0 + np.random.randn() * 0.1,  # VW around 1
                'RATING_NUM': 5,
                'signal1': np.random.randn(),
            })

    df = pd.DataFrame(rows)
    df['VW'] = df['VW'].clip(0.1, 10)
    return df


def create_unbalanced_panel(n_dates=12, n_bonds=30, dropout_rate=0.1, seed=42):
    """Create unbalanced panel with random bond dropouts."""
    np.random.seed(seed)

    dates = pd.date_range('2020-01-31', periods=n_dates, freq='ME')

    rows = []
    for d_idx, date in enumerate(dates):
        for bond_id in range(1, n_bonds + 1):
            # Random dropout after first few dates
            if d_idx > 2 and np.random.random() < dropout_rate:
                continue  # Skip this bond at this date

            rows.append({
                'date': date,
                'ID': f'BOND_{bond_id:03d}',
                'ret': np.random.randn() * 0.02,
                'VW': 1.0 + np.random.randn() * 0.1,
                'RATING_NUM': 5,
                'signal1': np.random.randn(),
            })

    df = pd.DataFrame(rows)
    df['VW'] = df['VW'].clip(0.1, 10)
    return df


def diagnose_quarterly_turnover(data, panel_type="balanced"):
    """Diagnose quarterly rebalancing turnover behavior."""
    print(f"\n{'='*70}")
    print(f"QUARTERLY REBALANCING TURNOVER DIAGNOSIS ({panel_type} panel)")
    print(f"{'='*70}")

    # Note: rebalance_frequency goes on the STRATEGY, not StrategyFormation
    strategy = SingleSort(
        holding_period=1,  # HP=1 for non-staggered
        sort_var='signal1',
        num_portfolios=5,
        rebalance_frequency='quarterly',  # Correct placement
        rebalance_month=1
    )

    sf = StrategyFormation(
        data=data,
        strategy=strategy,
        turnover=True,
        verbose=False
    )
    result = sf.fit()

    # Get turnover
    ew_turn, vw_turn = result.get_turnover()

    # Get returns
    ew_ls, vw_ls = result.get_long_short()

    print(f"\nData shape: {len(data)} rows")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"Unique dates: {data['date'].nunique()}")

    print(f"\n--- Returns ---")
    print(f"Returns shape: {ew_ls.shape}")
    print(f"Returns dates:\n{ew_ls.index.tolist()}")

    print(f"\n--- Turnover ---")
    print(f"Turnover shape: {ew_turn.shape}")
    print(f"Turnover columns: {ew_turn.columns.tolist()}")

    # Analyze turnover row by row
    print(f"\n--- Turnover Analysis (EW, first portfolio) ---")
    print(f"{'Date':<15} {'Turnover':>12} {'Is Rebal?':>12} {'Notes'}")
    print("-" * 55)

    # Determine rebalancing dates (quarterly = every 3 months starting from rebalance_month)
    rebal_dates = []
    for i, date in enumerate(ew_turn.index):
        if date.month in [1, 4, 7, 10]:  # Quarterly rebalancing months
            rebal_dates.append(date)

    for i, date in enumerate(ew_turn.index):
        turn_val = ew_turn.iloc[i, 0]  # First portfolio
        is_rebal = date.month in [1, 4, 7, 10]

        if pd.isna(turn_val):
            note = "NaN (warmup)"
        elif turn_val == 0.0:
            note = "Zero (no trading)"
        elif turn_val > 0:
            note = "Non-zero turnover!"
        else:
            note = ""

        print(f"{str(date.date()):<15} {turn_val:>12.4f} {str(is_rebal):>12} {note}")

    # Summary statistics
    print(f"\n--- Summary ---")
    non_nan = ew_turn.iloc[:, 0].dropna()
    if len(non_nan) > 0:
        print(f"Non-NaN turnover count: {len(non_nan)}")
        print(f"Zero turnover count: {(non_nan == 0.0).sum()}")
        print(f"Non-zero turnover count: {(non_nan != 0.0).sum()}")
        print(f"Mean turnover (when non-zero): {non_nan[non_nan != 0].mean():.4f}")

    return result


def diagnose_weight_scaling():
    """Diagnose how weights are scaled over holding period."""
    print(f"\n{'='*70}")
    print(f"WEIGHT SCALING ANALYSIS")
    print(f"{'='*70}")

    # Create a simple panel with known returns
    dates = pd.date_range('2020-01-31', periods=6, freq='ME')

    # Create a very simple panel: 2 bonds, 6 dates
    rows = []
    returns = {
        'BOND_A': [0.01, 0.05, -0.03, 0.02, 0.01, 0.04],  # Bond A returns
        'BOND_B': [-0.02, 0.03, 0.02, -0.01, 0.03, 0.02],  # Bond B returns
    }

    for i, date in enumerate(dates):
        for bond_id, bond_rets in returns.items():
            rows.append({
                'date': date,
                'ID': bond_id,
                'ret': bond_rets[i],
                'VW': 1.0,  # Equal VW
                'RATING_NUM': 5,
                'signal1': 1.0 if bond_id == 'BOND_A' else 0.0,  # A in portfolio 5, B in portfolio 1
            })

    data = pd.DataFrame(rows)

    print(f"\nTest data:")
    print(data.pivot(index='date', columns='ID', values='ret'))

    # Run quarterly strategy - rebalance_frequency goes on Strategy
    strategy = SingleSort(
        holding_period=1,
        sort_var='signal1',
        num_portfolios=2,  # Only 2 portfolios for simplicity
        rebalance_frequency='quarterly',
        rebalance_month=1
    )

    sf = StrategyFormation(
        data=data,
        strategy=strategy,
        turnover=True,
        verbose=True
    )
    result = sf.fit()

    # Get turnover
    ew_turn, vw_turn = result.get_turnover()

    print(f"\n--- Expected weight evolution (EW, BOND_A in Portfolio 2) ---")
    print("Starting weight = 1.0 (only bond in portfolio)")
    print("Month 1 (rebal): weight = 1.0")
    print("Month 2: weight = 1.0 * (1 + 0.05) / (1 + 0.05) = 1.0 (single bond)")
    print("Month 3: weight = 1.0 * (1 + -0.03) / (1 + -0.03) = 1.0 (single bond)")
    print("Month 4 (rebal): weight = 1.0 (fresh weights)")

    print(f"\n--- Actual turnover ---")
    print(ew_turn)

    print(f"\nKey insight: With a single bond per portfolio, weights should stay at 1.0")
    print("Turnover should be 0 between rebalancing dates (no trading occurs)")
    print("But if bonds drop out, weights get renormalized -> 'latent' turnover")


def compare_staggered_vs_nonstaggered():
    """Compare turnover behavior between staggered and non-staggered."""
    print(f"\n{'='*70}")
    print(f"STAGGERED vs NON-STAGGERED COMPARISON")
    print(f"{'='*70}")

    # Create unbalanced panel
    data = create_unbalanced_panel(n_dates=12, n_bonds=30, dropout_rate=0.15, seed=42)

    # Staggered (HP=3)
    strategy_stag = SingleSort(holding_period=3, sort_var='signal1', num_portfolios=5)
    sf_stag = StrategyFormation(data=data, strategy=strategy_stag, turnover=True, verbose=False)
    result_stag = sf_stag.fit()
    ew_turn_stag, _ = result_stag.get_turnover()

    # Non-staggered (quarterly) - rebalance_frequency goes on Strategy
    strategy_nonstag = SingleSort(
        holding_period=1, sort_var='signal1', num_portfolios=5,
        rebalance_frequency='quarterly', rebalance_month=1
    )
    sf_nonstag = StrategyFormation(
        data=data, strategy=strategy_nonstag, turnover=True, verbose=False
    )
    result_nonstag = sf_nonstag.fit()
    ew_turn_nonstag, _ = result_nonstag.get_turnover()

    print(f"\n--- Staggered HP=3 (holding cohorts set to 0) ---")
    print(f"Shape: {ew_turn_stag.shape}")
    stag_zeros = (ew_turn_stag.iloc[:, 0] == 0.0).sum()
    stag_nonzero = ((ew_turn_stag.iloc[:, 0] != 0.0) & (ew_turn_stag.iloc[:, 0].notna())).sum()
    print(f"Zero turnover count: {stag_zeros}")
    print(f"Non-zero turnover count: {stag_nonzero}")

    print(f"\n--- Non-staggered quarterly (current behavior) ---")
    print(f"Shape: {ew_turn_nonstag.shape}")
    nonstag_zeros = (ew_turn_nonstag.iloc[:, 0] == 0.0).sum()
    nonstag_nonzero = ((ew_turn_nonstag.iloc[:, 0] != 0.0) & (ew_turn_nonstag.iloc[:, 0].notna())).sum()
    print(f"Zero turnover count: {nonstag_zeros}")
    print(f"Non-zero turnover count: {nonstag_nonzero}")

    print(f"\n--- Analysis ---")
    print(f"Staggered HP=3: {stag_zeros} zeros, {stag_nonzero} non-zeros")
    print(f"  -> Holding cohorts correctly set to 0")
    print(f"")
    print(f"Non-staggered quarterly: {nonstag_zeros} zeros, {nonstag_nonzero} non-zeros")
    print(f"  -> Turnover computed every month (including between rebalancing)")
    print(f"")
    print(f"KEY FINDING: Non-staggered counts 'latent turnover' from bond dropouts")
    print(f"between rebalancing dates, while staggered sets holding cohorts to 0.")


def diagnose_shift_alignment():
    """Verify shift(1) is applied to non-staggered turnover."""
    print(f"\n{'='*70}")
    print(f"SHIFT(1) ALIGNMENT CHECK")
    print(f"{'='*70}")

    data = create_balanced_panel(n_dates=12, n_bonds=30, seed=42)

    # rebalance_frequency goes on Strategy
    strategy = SingleSort(
        holding_period=1, sort_var='signal1', num_portfolios=5,
        rebalance_frequency='quarterly', rebalance_month=1
    )
    sf = StrategyFormation(
        data=data, strategy=strategy, turnover=True, verbose=False
    )
    result = sf.fit()

    ew_turn, _ = result.get_turnover()
    ew_ls, _ = result.get_long_short()

    print(f"\nReturns first 5 rows:")
    print(ew_ls.head())

    print(f"\nTurnover first 5 rows:")
    print(ew_turn.head())

    print(f"\nFirst turnover row (should be NaN): {ew_turn.iloc[0].values}")

    if ew_turn.iloc[0].isna().all():
        print("PASS: First turnover row is NaN (shift(1) applied)")
    else:
        print("FAIL: First turnover row is NOT NaN")

    # Check indices match
    if ew_turn.index.equals(ew_ls.index):
        print("PASS: Turnover and returns have same index")
    else:
        print("FAIL: Turnover and returns have different indices")


def main():
    print("=" * 70)
    print("NON-STAGGERED TURNOVER DIAGNOSTIC")
    print("=" * 70)

    # Test 1: Balanced panel (no dropouts)
    print("\n\n" + "=" * 70)
    print("TEST 1: BALANCED PANEL (NO DROPOUTS)")
    print("=" * 70)
    balanced_data = create_balanced_panel()
    diagnose_quarterly_turnover(balanced_data, "balanced")

    # Test 2: Unbalanced panel (with dropouts)
    print("\n\n" + "=" * 70)
    print("TEST 2: UNBALANCED PANEL (WITH DROPOUTS)")
    print("=" * 70)
    unbalanced_data = create_unbalanced_panel(dropout_rate=0.15)
    diagnose_quarterly_turnover(unbalanced_data, "unbalanced")

    # Test 3: Weight scaling analysis
    diagnose_weight_scaling()

    # Test 4: Compare staggered vs non-staggered
    compare_staggered_vs_nonstaggered()

    # Test 5: Shift alignment
    diagnose_shift_alignment()

    print("\n\n" + "=" * 70)
    print("SUMMARY OF FINDINGS (AFTER FIX)")
    print("=" * 70)
    print("""
1. NON-STAGGERED TURNOVER BEHAVIOR (FIXED):
   - Turnover is now only computed at REBALANCING dates (first return after rebal)
   - Between rebalancing dates, turnover is set to 0.0 (holding period, no trading)
   - This matches staggered rebalancing behavior where holding cohorts get 0

2. WEIGHT SCALING:
   - Current implementation uses SINGLE-PERIOD scaling: w' = w * (1+r)/(1+R)
   - This is computed fresh each month from formation date weights
   - Weights are still updated every month for proper comparison at next rebalancing

3. SHIFT(1) ALIGNMENT:
   - shift(1) IS applied to non-staggered turnover
   - First row is NaN (warmup period)
   - Turnover[t] aligns with return[t]

4. FIX DETAILS:
   - Both slow path (utils_turnover.py) and fast path (numba_core.py) updated
   - Check if d == form_d + 1 (first return after rebalancing)
   - If yes: compute actual turnover (trading occurred)
   - If no: set turnover to 0.0 (holding period, no trading)
""")


if __name__ == "__main__":
    main()
