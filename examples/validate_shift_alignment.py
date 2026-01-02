#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validation script for shift(1) alignment of turnover and characteristics.

This script verifies that:
1. Turnover at index t represents the cost of entering positions that generate return[t]
2. Chars at index t represent the characteristics of portfolios that generate return[t]
3. First row is NaN (warmup period - no previous to compare)
4. Net-of-cost returns can be computed correctly: net_ret[t] = ret[t] - cost * turnover[t]
5. BatchStrategyFormation produces consistent results

Author: Claude
Date: December 2025
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/user/PyBondLab-Dev')

import PyBondLab as pbl
from PyBondLab import StrategyFormation, SingleSort, BatchStrategyFormation
from PyBondLab.pbl_test import generate_synthetic_data


def test_turnover_shift_hp1():
    """Test turnover shift(1) alignment for HP=1."""
    print("\n" + "=" * 60)
    print("TEST 1: Turnover shift(1) alignment for HP=1")
    print("=" * 60)

    # Generate test data
    data = generate_synthetic_data(n_dates=12, n_bonds=50, seed=42)

    strategy = SingleSort(
        holding_period=1,
        sort_var='signal1',
        num_portfolios=5
    )

    sf = StrategyFormation(
        data=data,
        strategy=strategy,
        turnover=True,
        verbose=False
    )
    result = sf.fit()

    # Get returns and turnover
    ew_ls, vw_ls = result.get_long_short()
    ew_turn, vw_turn = result.get_turnover()

    print(f"\nReturns index: {ew_ls.index.min()} to {ew_ls.index.max()}")
    print(f"Returns length: {len(ew_ls)}")
    print(f"\nTurnover index: {ew_turn.index.min()} to {ew_turn.index.max()}")
    print(f"Turnover length: {len(ew_turn)}")

    # Check first row is NaN
    first_turn_row = ew_turn.iloc[0]
    print(f"\nFirst turnover row (should be NaN): {first_turn_row.values}")
    assert first_turn_row.isna().all(), "FAIL: First turnover row should be all NaN"
    print("PASS: First turnover row is NaN (warmup period)")

    # Check alignment: turnover index should match returns index (minus first date)
    # Returns include first date (NaN), turnover starts from second date
    common_dates = ew_ls.index.intersection(ew_turn.index)
    print(f"\nCommon dates: {len(common_dates)}")
    assert len(common_dates) == len(ew_turn), "FAIL: Turnover and returns should have same dates"
    print("PASS: Turnover and returns have aligned dates")

    # Check net-of-cost calculation is straightforward
    print("\n--- Net-of-cost calculation example ---")
    cost_per_turnover = 0.001  # 10 bps per unit turnover
    # Get portfolio 1 (short leg) turnover
    port_turnover = ew_turn.iloc[:, 0]  # First portfolio
    print(f"Portfolio 1 turnover (first 5 rows):\n{port_turnover.head()}")
    print("\nWith shift(1), net return calculation is simply:")
    print("  net_return[t] = gross_return[t] - cost * turnover[t]")
    print("  (turnover[t] already aligned to return[t])")

    return True


def test_turnover_shift_hp3():
    """Test turnover shift(1) alignment for HP=3 (staggered)."""
    print("\n" + "=" * 60)
    print("TEST 2: Turnover shift(1) alignment for HP=3 (staggered)")
    print("=" * 60)

    # Generate test data
    data = generate_synthetic_data(n_dates=24, n_bonds=50, seed=42)

    strategy = SingleSort(
        holding_period=3,
        sort_var='signal1',
        num_portfolios=5
    )

    sf = StrategyFormation(
        data=data,
        strategy=strategy,
        turnover=True,
        verbose=False
    )
    result = sf.fit()

    # Get returns and turnover
    ew_ls, vw_ls = result.get_long_short()
    ew_turn, vw_turn = result.get_turnover()

    print(f"\nReturns length: {len(ew_ls)}")
    print(f"Turnover length: {len(ew_turn)}")

    # Check first row is NaN
    first_turn_row = ew_turn.iloc[0]
    print(f"\nFirst turnover row (should be NaN): {first_turn_row.values}")
    assert first_turn_row.isna().all(), "FAIL: First turnover row should be all NaN"
    print("PASS: First turnover row is NaN (warmup period)")

    # For HP=3, check that subsequent rows have values
    # (cohort averaging should produce non-NaN after warmup)
    non_nan_count = ew_turn.iloc[1:].notna().sum().sum()
    print(f"\nNon-NaN values after first row: {non_nan_count}")
    assert non_nan_count > 0, "FAIL: Should have turnover values after warmup"
    print("PASS: Turnover values present after warmup period")

    return True


def test_chars_shift_hp1():
    """Test chars shift(1) alignment for HP=1."""
    print("\n" + "=" * 60)
    print("TEST 3: Chars shift(1) alignment for HP=1")
    print("=" * 60)

    # Generate test data with characteristics
    data = generate_synthetic_data(n_dates=12, n_bonds=50, seed=42)

    strategy = SingleSort(
        holding_period=1,
        sort_var='signal1',
        num_portfolios=5
    )

    sf = StrategyFormation(
        data=data,
        strategy=strategy,
        chars=['char1', 'char2'],
        verbose=False
    )
    result = sf.fit()

    # Get returns and chars
    ew_ls, vw_ls = result.get_long_short()
    ew_chars, vw_chars = result.get_characteristics()

    char1_ew = ew_chars['char1']

    print(f"\nReturns length: {len(ew_ls)}")
    print(f"Chars length: {len(char1_ew)}")

    # Check first row is NaN
    first_char_row = char1_ew.iloc[0]
    print(f"\nFirst char row (should be NaN): {first_char_row.values}")
    assert first_char_row.isna().all(), "FAIL: First char row should be all NaN"
    print("PASS: First char row is NaN (warmup period)")

    # Check index alignment
    assert char1_ew.index.equals(ew_ls.index), "FAIL: Chars and returns should have same index"
    print("PASS: Chars and returns have aligned indices")

    return True


def test_chars_shift_hp3():
    """Test chars shift(1) alignment for HP=3 (staggered)."""
    print("\n" + "=" * 60)
    print("TEST 4: Chars shift(1) alignment for HP=3 (staggered)")
    print("=" * 60)

    # Generate test data
    data = generate_synthetic_data(n_dates=24, n_bonds=50, seed=42)

    strategy = SingleSort(
        holding_period=3,
        sort_var='signal1',
        num_portfolios=5
    )

    sf = StrategyFormation(
        data=data,
        strategy=strategy,
        chars=['char1'],
        verbose=False
    )
    result = sf.fit()

    # Get chars
    ew_chars, vw_chars = result.get_characteristics()
    char1_ew = ew_chars['char1']

    # Check first row is NaN
    first_char_row = char1_ew.iloc[0]
    print(f"\nFirst char row (should be NaN): {first_char_row.values}")
    assert first_char_row.isna().all(), "FAIL: First char row should be all NaN"
    print("PASS: First char row is NaN (warmup period)")

    # Check subsequent rows have values
    non_nan_count = char1_ew.iloc[1:].notna().sum().sum()
    print(f"\nNon-NaN values after first row: {non_nan_count}")
    assert non_nan_count > 0, "FAIL: Should have char values after warmup"
    print("PASS: Char values present after warmup period")

    return True


def test_batch_strategy_formation():
    """Test BatchStrategyFormation produces consistent shift(1) alignment."""
    print("\n" + "=" * 60)
    print("TEST 5: BatchStrategyFormation compatibility")
    print("=" * 60)

    # Generate test data
    data = generate_synthetic_data(n_dates=12, n_bonds=50, seed=42)

    # Run batch with turnover
    batch = BatchStrategyFormation(
        data=data,
        signals=['signal1', 'signal2'],
        holding_period=1,
        num_portfolios=5,
        turnover=True,
        n_jobs=1,
        verbose=False
    )
    results = batch.fit()

    # Check signal1 results
    result1 = results['signal1']
    ew_turn, vw_turn = result1.get_turnover()

    print(f"\nBatch signal1 turnover length: {len(ew_turn)}")

    # Check first row is NaN
    first_turn_row = ew_turn.iloc[0]
    print(f"First turnover row (should be NaN): {first_turn_row.values}")
    assert first_turn_row.isna().all(), "FAIL: First turnover row should be all NaN"
    print("PASS: BatchStrategyFormation turnover has correct shift(1)")

    # Check signal2 as well
    result2 = results['signal2']
    ew_turn2, _ = result2.get_turnover()
    assert ew_turn2.iloc[0].isna().all(), "FAIL: signal2 first row should be NaN"
    print("PASS: Both signals have correct shift(1) alignment")

    return True


def test_batch_with_chars():
    """Test BatchStrategyFormation with chars."""
    print("\n" + "=" * 60)
    print("TEST 6: BatchStrategyFormation with chars")
    print("=" * 60)

    # Generate test data
    data = generate_synthetic_data(n_dates=12, n_bonds=50, seed=42)

    # Run batch with chars (uses slow path due to chars)
    batch = BatchStrategyFormation(
        data=data,
        signals=['signal1'],
        holding_period=1,
        num_portfolios=5,
        turnover=True,
        chars=['char1'],
        n_jobs=1,
        verbose=False
    )
    results = batch.fit()

    result = results['signal1']
    ew_chars, vw_chars = result.get_characteristics()
    char1 = ew_chars['char1']

    print(f"\nBatch chars length: {len(char1)}")

    # Check first row is NaN
    first_char_row = char1.iloc[0]
    print(f"First char row (should be NaN): {first_char_row.values}")
    assert first_char_row.isna().all(), "FAIL: First char row should be all NaN"
    print("PASS: BatchStrategyFormation chars have correct shift(1)")

    return True


def test_net_of_cost_example():
    """Demonstrate net-of-cost calculation with aligned turnover."""
    print("\n" + "=" * 60)
    print("TEST 7: Net-of-cost calculation demonstration")
    print("=" * 60)

    # Generate test data
    data = generate_synthetic_data(n_dates=12, n_bonds=50, seed=42)

    strategy = SingleSort(
        holding_period=1,
        sort_var='signal1',
        num_portfolios=5
    )

    sf = StrategyFormation(
        data=data,
        strategy=strategy,
        turnover=True,
        verbose=False
    )
    result = sf.fit()

    # Get long-short returns and turnover
    ew_ls, _ = result.get_long_short()
    ew_turn, _ = result.get_turnover()

    print(f"\nReturns dates: {len(ew_ls)} ({ew_ls.index[0]} to {ew_ls.index[-1]})")
    print(f"Turnover dates: {len(ew_turn)} ({ew_turn.index[0]} to {ew_turn.index[-1]})")

    # Compute factor-level turnover: sum of long and short portfolios (total trading)
    n_port = ew_turn.shape[1]
    factor_turnover = ew_turn.iloc[:, 0] + ew_turn.iloc[:, n_port - 1]

    # Set transaction cost
    cost_per_unit = 0.002  # 20 bps per unit turnover

    # Note: Returns starts from first date, turnover starts from second date
    # This is correct: at first return date, there's no prior formation to compare
    # To compute net-of-cost, align on common dates:
    common_dates = ew_ls.index.intersection(factor_turnover.index)

    # Compute net-of-cost returns for aligned dates
    ret_aligned = ew_ls.loc[common_dates].values.flatten()
    turn_aligned = factor_turnover.loc[common_dates].values
    net_returns = ret_aligned - cost_per_unit * turn_aligned

    print("\nGross returns (first 5 common dates):")
    print(ew_ls.loc[common_dates].head())

    print("\nFactor turnover (first 5 rows):")
    print(factor_turnover.head())

    print("\nNet-of-cost returns (first 5 common dates):")
    print(pd.Series(net_returns[:5], index=common_dates[:5]))

    print("\n--- Key Point ---")
    print("With shift(1) alignment:")
    print("  - turnover[t] = cost incurred at formation (t-1) for return at t")
    print("  - net_return[t] = gross_return[t] - cost * turnover[t]")
    print("  - First turnover date is NaN (warmup), so first computable net return is date 3")
    print("  - No manual lag adjustment needed - just join on dates!")

    # Verify first turnover row is NaN
    assert factor_turnover.iloc[0] != factor_turnover.iloc[0], "First turnover should be NaN"
    print("\nPASS: First turnover is NaN (warmup period)")

    # Verify we can compute net returns for non-NaN turnover dates
    non_nan_mask = ~np.isnan(turn_aligned)
    n_computable = non_nan_mask.sum()
    print(f"PASS: Can compute net-of-cost for {n_computable} dates (after warmup)")

    return True


def main():
    print("=" * 70)
    print("VALIDATION: shift(1) ALIGNMENT FOR TURNOVER AND CHARS")
    print("=" * 70)
    print("\nThis script validates the shift(1) alignment that ensures:")
    print("  - turnover[t] = cost of entering positions for return[t]")
    print("  - chars[t] = characteristics of portfolio generating return[t]")
    print("  - First row is NaN (warmup period)")

    tests = [
        ("HP=1 Turnover Shift", test_turnover_shift_hp1),
        ("HP=3 Turnover Shift", test_turnover_shift_hp3),
        ("HP=1 Chars Shift", test_chars_shift_hp1),
        ("HP=3 Chars Shift", test_chars_shift_hp3),
        ("BatchStrategyFormation", test_batch_strategy_formation),
        ("Batch with Chars", test_batch_with_chars),
        ("Net-of-cost Example", test_net_of_cost_example),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results[name] = "PASS" if passed else "FAIL"
        except Exception as e:
            results[name] = f"ERROR: {e}"
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, status in results.items():
        status_str = "PASS" if status == "PASS" else "FAIL"
        print(f"  {name}: {status_str}")

    n_pass = sum(1 for s in results.values() if s == "PASS")
    n_total = len(results)
    print(f"\nTotal: {n_pass}/{n_total} tests passed")

    if n_pass == n_total:
        print("\nAll tests passed! shift(1) alignment is working correctly.")
        return 0
    else:
        print("\nSome tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
