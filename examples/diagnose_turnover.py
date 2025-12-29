#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diagnostic script for turnover computation.

Creates a PERFECTLY BALANCED panel (all bonds present at all dates) with a
signal that MAXIMALLY CHANGES each month to achieve theoretical upper bound turnover.

Theoretical expectations:
- Complete turnover for rebalancing cohort = 2.0 (all bonds change portfolios)
- HP=1: turnover = 2.0 (no cohort averaging)
- HP=2: turnover = (2.0 + 0) / 2 = 1.0
- HP=3: turnover = (2.0 + 0 + 0) / 3 = 0.67

This script verifies the turnover computation is working correctly.
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/user/PyBondLab-Dev')

import PyBondLab as pbl
from PyBondLab import StrategyFormation, SingleSort


def create_balanced_panel(n_dates: int = 24, n_bonds: int = 100, seed: int = 42, hp: int = 1):
    """
    Create a perfectly balanced panel where ALL bonds are present at ALL dates.

    Signal changes every HP months to ensure complete turnover at each rebalancing.
    This ensures that for any HP, consecutive rebalancing dates see different signals.

    For HP=1: signal reverses every month
    For HP=2: signal reverses every 2 months
    For HP=3: signal reverses every 3 months
    """
    np.random.seed(seed)

    # Create date range
    dates = pd.date_range('2020-01-31', periods=n_dates, freq='ME')

    # Create all (date, bond) combinations
    records = []
    for t_idx, date in enumerate(dates):
        for bond_idx in range(n_bonds):
            bond_id = f"BOND_{bond_idx:04d}"

            # Signal that REVERSES every HP months
            # This ensures consecutive rebalancing dates see different signals
            # For HP=1: (0,1,0,1,0,1,...) - every month
            # For HP=2: (0,0,1,1,0,0,1,1,...) - every 2 months
            # For HP=3: (0,0,0,1,1,1,0,0,0,...) - every 3 months
            phase = (t_idx // hp) % 2

            if phase == 0:
                signal = bond_idx  # Ascending: 0, 1, 2, ..., N-1
            else:
                signal = n_bonds - 1 - bond_idx  # Descending: N-1, N-2, ..., 0

            # Small random noise to break ties (but preserve order)
            signal = signal + np.random.uniform(0, 0.1)

            # Return: small random value (doesn't affect turnover much)
            ret = np.random.uniform(-0.02, 0.02)

            # VW: all equal (so VW = EW effectively)
            vw = 1.0

            records.append({
                'date': date,
                'ID': bond_id,
                'signal': signal,
                'ret': ret,
                'VW': vw,
                'RATING_NUM': 5,  # All same rating
            })

    df = pd.DataFrame(records)
    print(f"Created balanced panel: {len(df)} rows, {n_bonds} bonds, {n_dates} dates (HP={hp})")
    print(f"Dates range: {dates[0]} to {dates[-1]}")

    return df


def run_turnover_test(data: pd.DataFrame, hp: int, verbose: bool = True):
    """
    Run StrategyFormation with specified HP and return turnover statistics.
    """
    strategy = SingleSort(
        holding_period=hp,
        sort_var='signal',
        num_portfolios=5
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

    if verbose:
        print(f"\n{'='*60}")
        print(f"HP = {hp}")
        print(f"{'='*60}")
        print(f"\nEW Turnover (mean per portfolio):")
        print(ew_turn.mean())
        print(f"\nOverall EW mean: {ew_turn.mean().mean():.4f}")
        print(f"Overall VW mean: {vw_turn.mean().mean():.4f}")

    return ew_turn, vw_turn, result


def inspect_raw_turnover_state(data: pd.DataFrame, hp: int):
    """
    Inspect the raw turnover arrays BEFORE averaging to understand what's happening.
    """
    strategy = SingleSort(
        holding_period=hp,
        sort_var='signal',
        num_portfolios=5
    )

    sf = StrategyFormation(
        data=data,
        strategy=strategy,
        turnover=True,
        verbose=False
    )

    # Call fit to compute everything
    result = sf.fit()

    # Access the turnover state
    state = sf.turnover_state

    print(f"\n{'='*60}")
    print(f"RAW TURNOVER STATE INSPECTION (HP={hp})")
    print(f"{'='*60}")

    print(f"\nTurnover array shape: {state.ew_turn_ea.shape}")
    print(f"  - Dimension 0 (time): {state.ew_turn_ea.shape[0]}")
    if len(state.ew_turn_ea.shape) == 3:
        print(f"  - Dimension 1 (cohort): {state.ew_turn_ea.shape[1]}")
        print(f"  - Dimension 2 (portfolio): {state.ew_turn_ea.shape[2]}")
    else:
        print(f"  - Dimension 1 (portfolio): {state.ew_turn_ea.shape[1]}")

    print(f"\nprev_seen_ew shape: {state.prev_seen_ew.shape}")
    print(f"prev_seen_ew:\n{state.prev_seen_ew}")

    # For staggered, show a few time slices
    if len(state.ew_turn_ea.shape) == 3:
        n_times = min(10, state.ew_turn_ea.shape[0])
        print(f"\nFirst {n_times} time slices of EW turnover (time x cohort x portfolio):")
        for t in range(n_times):
            print(f"\n  t={t}:")
            for c in range(state.ew_turn_ea.shape[1]):
                vals = state.ew_turn_ea[t, c, :]
                vals_str = ", ".join([f"{v:.4f}" if not np.isnan(v) else "NaN" for v in vals])
                print(f"    cohort {c}: [{vals_str}]")

        # Show statistics
        print(f"\n\nTurnover statistics by cohort (excluding first row):")
        for c in range(state.ew_turn_ea.shape[1]):
            cohort_vals = state.ew_turn_ea[1:, c, :]  # Exclude first row
            non_nan = cohort_vals[~np.isnan(cohort_vals)]
            non_zero = non_nan[non_nan > 0]
            zeros = non_nan[non_nan == 0]
            print(f"\n  Cohort {c}:")
            print(f"    Total values: {cohort_vals.size}")
            print(f"    NaN count: {np.isnan(cohort_vals).sum()}")
            print(f"    Zero count: {len(zeros)}")
            print(f"    Non-zero count: {len(non_zero)}")
            if len(non_zero) > 0:
                print(f"    Non-zero mean: {non_zero.mean():.4f}")
                print(f"    Non-zero min: {non_zero.min():.4f}")
                print(f"    Non-zero max: {non_zero.max():.4f}")

    return state


def test_h0_only_fix():
    """
    Test if the bug is indeed in the horizon loop.

    Now uses HP-specific data generation to ensure signal changes at rebalancing dates.
    """
    print("\n" + "=" * 70)
    print("TESTING HP-SPECIFIC TURNOVER (AFTER FIX)")
    print("=" * 70)

    print("\nWith the fix: only accumulate turnover for h=0")
    print("Using HP-specific signal patterns to ensure turnover at each rebalancing")
    print()

    for hp in [1, 2, 3]:
        # Create data with signal that changes at the right frequency for this HP
        data = create_balanced_panel(n_dates=24, n_bonds=50, seed=42, hp=hp)
        ew, vw, _ = run_turnover_test(data, hp=hp, verbose=False)
        mean_turn = ew.mean().mean()
        expected = 2.0 / hp  # Complete turnover (2.0) divided by number of cohorts
        ratio = mean_turn / expected

        print(f"HP={hp}: Mean={mean_turn:.4f}, Expected={expected:.4f}, Ratio={ratio:.4f}")


def main():
    print("=" * 70)
    print("TURNOVER DIAGNOSTIC SCRIPT")
    print("=" * 70)
    print("\nThis script creates a balanced panel with maximally-changing signal")
    print("to verify turnover computation matches theoretical expectations.")
    print("\nTheoretical expectations (complete turnover = 2.0 per cohort):")
    print("  HP=1: turnover = 2.0 (no averaging)")
    print("  HP=2: turnover = (2.0 + 0) / 2 = 1.0")
    print("  HP=3: turnover = (2.0 + 0 + 0) / 3 = 0.67")

    # Run tests for HP=1, 2, 3 with HP-specific data
    print("\n" + "=" * 70)
    print("RUNNING HP-SPECIFIC TESTS")
    print("=" * 70)
    print("\nEach HP uses data where signal changes at the correct frequency")
    print("to ensure complete turnover at each rebalancing date.")

    results = {}

    for hp in [1, 2, 3]:
        # Create data with signal that changes every HP months
        data = create_balanced_panel(n_dates=24, n_bonds=100, seed=42, hp=hp)
        ew_turn, vw_turn, result = run_turnover_test(data, hp)
        results[hp] = {
            'ew_turn': ew_turn,
            'vw_turn': vw_turn,
            'result': result
        }

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)

    print("\n{:>5} {:>12} {:>12} {:>12} {:>12}".format(
        "HP", "Expected", "Actual EW", "Actual VW", "Ratio"
    ))
    print("-" * 55)

    expected = {1: 2.0, 2: 1.0, 3: 2.0/3}

    for hp in [1, 2, 3]:
        actual_ew = results[hp]['ew_turn'].mean().mean()
        actual_vw = results[hp]['vw_turn'].mean().mean()
        exp = expected[hp]
        ratio = actual_ew / exp if exp > 0 else 0

        print(f"{hp:>5} {exp:>12.4f} {actual_ew:>12.4f} {actual_vw:>12.4f} {ratio:>12.4f}")

    # Inspect raw state for HP=3
    print("\n" + "=" * 70)
    print("DETAILED INSPECTION OF HP=3 RAW TURNOVER STATE")
    print("=" * 70)
    # Use HP=3 specific data for this inspection
    data_hp3 = create_balanced_panel(n_dates=24, n_bonds=100, seed=42, hp=3)
    state = inspect_raw_turnover_state(data_hp3, hp=3)

    # Check if the issue is in averaging
    print("\n" + "=" * 70)
    print("MANUAL AVERAGING CHECK (HP=3)")
    print("=" * 70)

    if len(state.ew_turn_ea.shape) == 3:
        # Manually compute average
        arr = state.ew_turn_ea[1:, :, :]  # Skip first row

        print(f"\nArray shape after skipping first row: {arr.shape}")

        # Check each time slice
        print("\nManual averaging for each time slice (first 10):")
        for t in range(min(10, arr.shape[0])):
            slice_vals = arr[t, :, :]  # (cohort, portfolio)

            # Average across cohorts for each portfolio
            avg_per_ptf = np.nanmean(slice_vals, axis=0)

            print(f"\n  t={t+1} (return date index):")
            for c in range(slice_vals.shape[0]):
                c_vals = slice_vals[c, :]
                c_str = ", ".join([f"{v:.3f}" if not np.isnan(v) else "NaN" for v in c_vals])
                print(f"    cohort {c}: [{c_str}]")
            avg_str = ", ".join([f"{v:.3f}" for v in avg_per_ptf])
            print(f"    AVERAGE: [{avg_str}]")


if __name__ == "__main__":
    main()

    # Test the hypothesis
    test_h0_only_fix()

    print("\n" + "=" * 70)
    print("BUG DIAGNOSIS COMPLETE")
    print("=" * 70)
    print("""
ROOT CAUSE IDENTIFIED:
======================
In PyBondLab/PyBondLab.py, the turnover accumulation is INSIDE the horizon loop:

    for h in range(self.hor):  # Lines 2090-2161
        ...
        if self.turnover:
            self.turnover_manager.accumulate(...)  # Called for EVERY h!

For HP=3, this means:
- h=0: First time → no turnover, set prev_seen=True, store scaled weights
- h=1: prev_seen=True now → computes small turnover (~0.01), overwrites h=0 weights
- h=2: prev_seen=True → computes small turnover (~0.01), overwrites again

The FINAL stored value is from comparing h=2 with h=1 weights (same formation month!).
Even at true rebalancing, the high turnover from h=0 gets overwritten by low values.

FIX REQUIRED:
=============
Move the turnover accumulation OUTSIDE the horizon loop, or only accumulate for h=0:

    for h in range(self.hor):
        ...
        # Only accumulate turnover for first horizon (h=0)
        if self.turnover and h == 0 and not result['weights_df'].empty:
            self.turnover_manager.accumulate(...)

This ensures turnover is computed once per formation date, comparing with the
previous rebalancing's weights, not the previous horizon's weights.
""")
