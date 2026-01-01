#!/usr/bin/env python
"""
Validate turnover alignment fix (Option B).

This script verifies that:
1. Returns and turnover have the same dates
2. Returns and turnover have the same number of observations
3. First row is NaN (formation date), second row is entry turnover (1.0)
4. This works for both HP=1 and HP>1
"""

import sys
sys.path.insert(0, '/home/user/PyBondLab-Dev')

import numpy as np
import pandas as pd
import PyBondLab as pbl
from PyBondLab.pbl_test import generate_synthetic_data

def test_alignment(hp, description):
    """Test turnover alignment for a given holding period."""
    print(f"\n{'='*60}")
    print(f"Testing HP={hp}: {description}")
    print('='*60)

    # Generate test data
    data = generate_synthetic_data(n_dates=60, n_bonds=100, seed=42)

    # Create strategy
    strategy = pbl.SingleSort(
        holding_period=hp,
        sort_var='signal1',
        num_portfolios=5
    )

    # Run with turnover
    sf = pbl.StrategyFormation(
        data=data,
        strategy=strategy,
        turnover=True,
        verbose=False
    )
    result = sf.fit()

    # Get returns and turnover using the correct API
    ew_ls, vw_ls = result.get_long_short()
    ew_turn, vw_turn = result.get_turnover()

    # Check alignment
    print(f"\n--- Return Dates ---")
    print(f"  First date: {ew_ls.index[0]}")
    print(f"  Last date:  {ew_ls.index[-1]}")
    print(f"  Count:      {len(ew_ls)}")
    print(f"  First 5: {list(ew_ls.index[:5])}")

    print(f"\n--- Turnover Dates ---")
    print(f"  First date: {ew_turn.index[0]}")
    print(f"  Last date:  {ew_turn.index[-1]}")
    print(f"  Count:      {len(ew_turn)}")
    print(f"  First 5: {list(ew_turn.index[:5])}")

    # Get factor turnover on common dates
    nport = ew_turn.shape[1]
    ew_factor_turn = (ew_turn.iloc[:, 0] + ew_turn.iloc[:, nport-1]) / 2
    vw_factor_turn = (vw_turn.iloc[:, 0] + vw_turn.iloc[:, nport-1]) / 2

    print(f"\n--- First 5 Return Values ---")
    print(f"  EW LS: {ew_ls.iloc[:5].values}")
    print(f"  VW LS: {vw_ls.iloc[:5].values}")

    print(f"\n--- First 5 Turnover Values ---")
    print(f"  EW factor: {ew_factor_turn.iloc[:5].values}")
    print(f"  VW factor: {vw_factor_turn.iloc[:5].values}")

    # Validate same dates and length
    dates_match = (ew_ls.index == ew_turn.index).all()
    lengths_match = len(ew_ls) == len(ew_turn)
    print(f"\n--- Alignment Check ---")
    print(f"  Length match: {lengths_match} (returns={len(ew_ls)}, turnover={len(ew_turn)})")
    print(f"  Dates match:  {dates_match}")

    # Check that first row is NaN for BOTH returns and turnover
    first_ret_nan = np.isnan(ew_ls.iloc[0]) and np.isnan(vw_ls.iloc[0])
    first_turn_nan = np.isnan(ew_factor_turn.iloc[0]) and np.isnan(vw_factor_turn.iloc[0])
    print(f"\n--- First Row Check (Formation Date) ---")
    print(f"  Returns[0] is NaN:  {first_ret_nan}")
    print(f"  Turnover[0] is NaN: {first_turn_nan}")

    # Check second row (first actual return date) has entry turnover = 1.0
    second_ew_turn = ew_factor_turn.iloc[1]
    second_vw_turn = vw_factor_turn.iloc[1]
    print(f"\n--- Second Row Check (Entry Turnover) ---")
    print(f"  EW factor turnover[1]: {second_ew_turn:.4f} (expected ~1.0)")
    print(f"  VW factor turnover[1]: {second_vw_turn:.4f} (expected ~1.0)")

    second_not_nan = not np.isnan(second_ew_turn) and not np.isnan(second_vw_turn)
    entry_turnover_correct = abs(second_ew_turn - 1.0) < 0.01 and abs(second_vw_turn - 1.0) < 0.01
    print(f"  Second is NOT NaN:  {second_not_nan}")
    print(f"  Entry ~1.0:         {entry_turnover_correct}")

    # Overall pass/fail
    all_pass = (dates_match and lengths_match and first_ret_nan and
                first_turn_nan and second_not_nan and entry_turnover_correct)
    status = "PASS" if all_pass else "FAIL"
    print(f"\n*** HP={hp} Result: {status} ***")

    return all_pass


def main():
    print("="*60)
    print("TURNOVER ALIGNMENT VALIDATION (Option B)")
    print("="*60)

    results = {}

    # Test HP=1 (monthly rebalancing)
    results['HP=1'] = test_alignment(1, "Monthly rebalancing")

    # Test HP=3 (quarterly staggered)
    results['HP=3'] = test_alignment(3, "Quarterly staggered")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        all_pass = all_pass and passed

    print(f"\n*** Overall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'} ***")

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
