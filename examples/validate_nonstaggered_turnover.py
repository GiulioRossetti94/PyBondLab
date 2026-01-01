#!/usr/bin/env python
"""
Validate turnover alignment for non-staggered rebalancing.

This script verifies that:
1. Returns and turnover have the same dates for quarterly/annual rebalancing
2. First non-NaN return and first non-NaN turnover are at the same date
3. First non-NaN turnover is entry turnover = 1.0
4. Holding period dates have turnover = 0.0
"""

import sys
sys.path.insert(0, '/home/user/PyBondLab-Dev')

import numpy as np
import pandas as pd
import PyBondLab as pbl
from PyBondLab.pbl_test import generate_synthetic_data

def test_nonstaggered_alignment(frequency, description):
    """Test turnover alignment for non-staggered rebalancing."""
    print(f"\n{'='*60}")
    print(f"Testing {frequency}: {description}")
    print('='*60)

    # Generate test data
    data = generate_synthetic_data(n_dates=60, n_bonds=100, seed=42)

    # Create strategy with non-staggered rebalancing
    strategy = pbl.SingleSort(
        holding_period=1,
        sort_var='signal1',
        num_portfolios=5,
        rebalance_frequency=frequency
    )

    # Run with turnover
    sf = pbl.StrategyFormation(
        data=data,
        strategy=strategy,
        turnover=True,
        verbose=False
    )
    result = sf.fit()

    # Get returns and turnover
    ew_ls, vw_ls = result.get_long_short()
    ew_turn, vw_turn = result.get_turnover()

    # Check alignment
    print(f"\n--- Return Dates ---")
    print(f"  First date: {ew_ls.index[0]}")
    print(f"  Last date:  {ew_ls.index[-1]}")
    print(f"  Count:      {len(ew_ls)}")

    print(f"\n--- Turnover Dates ---")
    print(f"  First date: {ew_turn.index[0]}")
    print(f"  Last date:  {ew_turn.index[-1]}")
    print(f"  Count:      {len(ew_turn)}")

    # Get factor turnover
    nport = ew_turn.shape[1]
    ew_factor_turn = (ew_turn.iloc[:, 0] + ew_turn.iloc[:, nport-1]) / 2
    vw_factor_turn = (vw_turn.iloc[:, 0] + vw_turn.iloc[:, nport-1]) / 2

    # Validate same dates and length
    dates_match = (ew_ls.index == ew_turn.index).all()
    lengths_match = len(ew_ls) == len(ew_turn)
    print(f"\n--- Alignment Check ---")
    print(f"  Length match: {lengths_match} (returns={len(ew_ls)}, turnover={len(ew_turn)})")
    print(f"  Dates match:  {dates_match}")

    # Find first non-NaN return and turnover indices
    first_ret_idx = ew_ls.first_valid_index()
    first_turn_idx = ew_factor_turn.first_valid_index()
    print(f"\n--- First Non-NaN Indices ---")
    print(f"  First non-NaN return:   {first_ret_idx}")
    print(f"  First non-NaN turnover: {first_turn_idx}")
    first_nonnan_match = first_ret_idx == first_turn_idx
    print(f"  Match: {first_nonnan_match}")

    # Check entry turnover = 1.0 at first non-NaN index
    if first_turn_idx is not None:
        entry_ew_turn = ew_factor_turn.loc[first_turn_idx]
        entry_vw_turn = vw_factor_turn.loc[first_turn_idx]
        print(f"\n--- Entry Turnover at {first_turn_idx} ---")
        print(f"  EW factor turnover: {entry_ew_turn:.4f} (expected ~1.0)")
        print(f"  VW factor turnover: {entry_vw_turn:.4f} (expected ~1.0)")
        entry_correct = abs(entry_ew_turn - 1.0) < 0.01 and abs(entry_vw_turn - 1.0) < 0.01
        print(f"  Entry ~1.0: {entry_correct}")
    else:
        entry_correct = False
        print(f"\n--- Entry Turnover ---")
        print(f"  No non-NaN turnover found!")

    # Check that there are holding period turnovers = 0.0
    # Find indices where turnover is 0.0
    holding_indices = ew_factor_turn[ew_factor_turn == 0.0].index
    print(f"\n--- Holding Period Check ---")
    print(f"  Number of dates with turnover=0.0: {len(holding_indices)}")
    if len(holding_indices) > 0:
        print(f"  First holding period date: {holding_indices[0]}")
        holding_exists = True
    else:
        holding_exists = False
    print(f"  Holding periods exist: {holding_exists}")

    # Print first 15 turnover values
    print(f"\n--- First 15 Turnover Values ---")
    print(f"  EW factor: {ew_factor_turn.iloc[:15].values}")
    print(f"  VW factor: {vw_factor_turn.iloc[:15].values}")

    # Overall pass/fail
    all_pass = (dates_match and lengths_match and first_nonnan_match and
                entry_correct and holding_exists)
    status = "PASS" if all_pass else "FAIL"
    print(f"\n*** {frequency} Result: {status} ***")

    return all_pass


def main():
    print("="*60)
    print("NON-STAGGERED TURNOVER ALIGNMENT VALIDATION")
    print("="*60)

    results = {}

    # Test quarterly rebalancing
    results['quarterly'] = test_nonstaggered_alignment('quarterly', "Quarterly rebalancing")

    # Test annual rebalancing
    results['annual'] = test_nonstaggered_alignment('annual', "Annual rebalancing")

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
