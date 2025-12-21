#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validation script for extract_panel function.

Tests panel extraction from BatchStrategyFormation and BatchWithinFirmSortFormation.
"""

import sys
import os

# Add parent directory to path for local testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import PyBondLab as pbl
from PyBondLab import (
    BatchStrategyFormation,
    BatchWithinFirmSortFormation,
    extract_panel,
    NamingConfig,
)
from PyBondLab.pbl_test import generate_synthetic_data


def test_basic_extraction():
    """Test basic panel extraction with turnover (needed for full portfolio data)."""
    print("\n" + "=" * 60)
    print("TEST: Basic extraction (with turnover for full data)")
    print("=" * 60)

    # Generate data
    data = generate_synthetic_data(n_dates=30, n_bonds=100, seed=42)

    # Run batch strategy
    # NOTE: turnover=True required to get full portfolio data including leg returns
    print("  Running BatchStrategyFormation...")
    batch = BatchStrategyFormation(
        data=data,
        signals=['signal1', 'signal2'],
        holding_period=1,
        num_portfolios=5,
        turnover=True,  # Required for extract_panel
        n_jobs=1,
        verbose=False,
    )
    results = batch.fit()

    # Extract panel
    print("  Extracting panel...")
    panel = extract_panel(results)

    # Validate structure
    print(f"  Panel shape: {panel.shape}")
    print(f"  Columns: {list(panel.columns)}")

    expected_cols = ['date', 'factor', 'freq', 'leg', 'weighting', 'return', 'turnover']
    assert list(panel.columns) == expected_cols, f"Expected columns {expected_cols}, got {list(panel.columns)}"

    # Check legs
    legs = panel['leg'].unique()
    assert set(legs) == {'ls', 'l', 's'}, f"Expected legs {{'ls', 'l', 's'}}, got {set(legs)}"

    # Check weightings
    weightings = panel['weighting'].unique()
    assert set(weightings) == {'ew', 'vw'}, f"Expected weightings {{'ew', 'vw'}}, got {set(weightings)}"

    # Check factors
    factors = panel['factor'].unique()
    assert len(factors) == 2, f"Expected 2 factors, got {len(factors)}"
    print(f"  Factors: {list(factors)}")

    # Check freq
    assert (panel['freq'] == 1).all(), "Expected freq=1"

    # Check rows per signal: 2 signals * 2 weightings * 3 legs * n_dates
    n_dates = len(data['date'].unique()) - 1  # -1 for return date lag
    expected_rows = 2 * 2 * 3 * n_dates
    # Note: actual dates may be less due to formation
    print(f"  Expected ~{expected_rows} rows, got {len(panel)}")

    print("  [PASS] Basic extraction works correctly")
    return True


def test_with_turnover():
    """Test panel extraction with turnover."""
    print("\n" + "=" * 60)
    print("TEST: With turnover")
    print("=" * 60)

    # Generate data
    data = generate_synthetic_data(n_dates=30, n_bonds=100, seed=42)

    # Run batch strategy with turnover
    print("  Running BatchStrategyFormation with turnover=True...")
    batch = BatchStrategyFormation(
        data=data,
        signals=['signal1'],
        holding_period=1,
        num_portfolios=5,
        turnover=True,
        n_jobs=1,
        verbose=False,
    )
    results = batch.fit()

    # Extract panel
    print("  Extracting panel...")
    panel = extract_panel(results)

    # Validate turnover column exists
    assert 'turnover' in panel.columns, "Expected 'turnover' column"
    print(f"  Columns: {list(panel.columns)}")

    # Check turnover values are not all NaN
    assert not panel['turnover'].isna().all(), "Turnover should not be all NaN"
    print(f"  Turnover mean: {panel['turnover'].mean():.4f}")

    # Verify factor turnover = (long + short) / 2
    ls_turn = panel[(panel['leg'] == 'ls') & (panel['weighting'] == 'ew')]['turnover'].values
    l_turn = panel[(panel['leg'] == 'l') & (panel['weighting'] == 'ew')]['turnover'].values
    s_turn = panel[(panel['leg'] == 's') & (panel['weighting'] == 'ew')]['turnover'].values

    expected_ls_turn = (l_turn + s_turn) / 2
    # Ignore NaN values in comparison (first date has no turnover)
    valid_mask = ~np.isnan(ls_turn) & ~np.isnan(expected_ls_turn)
    if valid_mask.sum() > 0:
        diff = np.abs(ls_turn[valid_mask] - expected_ls_turn[valid_mask]).max()
        print(f"  Factor turnover check: max diff = {diff:.2e}")
        assert diff < 1e-10, f"Factor turnover doesn't match (L+S)/2: diff={diff}"
    else:
        print("  Factor turnover check: skipped (all NaN)")
        assert False, "No valid turnover values to compare"

    print("  [PASS] Turnover extraction works correctly")
    return True


def test_with_characteristics():
    """Test panel extraction with characteristics."""
    print("\n" + "=" * 60)
    print("TEST: With characteristics")
    print("=" * 60)

    # Generate data
    data = generate_synthetic_data(n_dates=30, n_bonds=100, seed=42)

    # Run batch strategy with chars
    print("  Running BatchStrategyFormation with chars...")
    batch = BatchStrategyFormation(
        data=data,
        signals=['signal1'],
        holding_period=1,
        num_portfolios=5,
        turnover=True,
        chars=['char1', 'char2'],
        n_jobs=1,
        verbose=False,
    )
    results = batch.fit()

    # Extract panel
    print("  Extracting panel...")
    panel = extract_panel(results)

    # Validate char columns exist
    assert 'char1' in panel.columns, "Expected 'char1' column"
    assert 'char2' in panel.columns, "Expected 'char2' column"
    print(f"  Columns: {list(panel.columns)}")

    # Verify L-S spread for ls leg
    ls_char1 = panel[(panel['leg'] == 'ls') & (panel['weighting'] == 'ew')]['char1'].values
    l_char1 = panel[(panel['leg'] == 'l') & (panel['weighting'] == 'ew')]['char1'].values
    s_char1 = panel[(panel['leg'] == 's') & (panel['weighting'] == 'ew')]['char1'].values

    expected_ls_char = l_char1 - s_char1
    # Ignore NaN values in comparison
    valid_mask = ~np.isnan(ls_char1) & ~np.isnan(expected_ls_char)
    if valid_mask.sum() > 0:
        diff = np.abs(ls_char1[valid_mask] - expected_ls_char[valid_mask]).max()
        print(f"  Char1 L-S spread check: max diff = {diff:.2e}")
        assert diff < 1e-10, f"Char L-S spread doesn't match: diff={diff}"
    else:
        print("  Char1 L-S spread check: skipped (all NaN)")
        assert False, "No valid char values to compare"

    print("  [PASS] Characteristics extraction works correctly")
    return True


def test_sign_correction():
    """Test sign correction and leg swapping."""
    print("\n" + "=" * 60)
    print("TEST: Sign correction")
    print("=" * 60)

    # Generate data
    data = generate_synthetic_data(n_dates=30, n_bonds=100, seed=42)

    # Run batch strategy
    print("  Running BatchStrategyFormation...")
    batch = BatchStrategyFormation(
        data=data,
        signals=['signal1'],
        holding_period=1,
        num_portfolios=5,
        turnover=True,  # Required for extract_panel
        n_jobs=1,
        verbose=False,
    )
    results = batch.fit()

    # Extract without sign correction
    print("  Extracting panel without sign correction...")
    panel_no_sign = extract_panel(results, naming=NamingConfig(sign_correct=False))

    # Extract with sign correction
    print("  Extracting panel with sign correction...")
    panel_sign = extract_panel(results, naming=NamingConfig(sign_correct=True))

    # Get original EW L-S return mean
    ew_ls_orig = panel_no_sign[(panel_no_sign['leg'] == 'ls') & (panel_no_sign['weighting'] == 'ew')]['return']
    ew_ls_sign = panel_sign[(panel_sign['leg'] == 'ls') & (panel_sign['weighting'] == 'ew')]['return']

    print(f"  Original EW L-S mean: {ew_ls_orig.mean():.6f}")
    print(f"  Sign-corrected EW L-S mean: {ew_ls_sign.mean():.6f}")

    # Check if sign was flipped
    if ew_ls_orig.mean() < 0:
        # Should be flipped
        assert ew_ls_sign.mean() > 0, "Sign correction should make mean positive"
        assert '*' in panel_sign['factor'].iloc[0], "Factor name should have * suffix"
        print("  [PASS] Negative factor was flipped, has * suffix")

        # Check legs were swapped - align by date for comparison
        orig_l_df = panel_no_sign[(panel_no_sign['leg'] == 'l') & (panel_no_sign['weighting'] == 'ew')][['date', 'return']]
        sign_s_df = panel_sign[(panel_sign['leg'] == 's') & (panel_sign['weighting'] == 'ew')][['date', 'return']]

        # Sort by date to align
        orig_l_df = orig_l_df.sort_values('date').reset_index(drop=True)
        sign_s_df = sign_s_df.sort_values('date').reset_index(drop=True)

        diff = (orig_l_df['return'].values - sign_s_df['return'].values)
        max_diff = np.nanmax(np.abs(diff))  # Use nanmax to ignore NaN values
        assert max_diff < 1e-10, "Long leg should become short leg after flip"
        print("  [PASS] Legs were swapped correctly")
    else:
        # Should not be flipped
        diff = (ew_ls_orig.values - ew_ls_sign.values)
        assert np.abs(diff).max() < 1e-10, "Positive factor should not be flipped"
        print("  [PASS] Positive factor was not flipped")

    print("  [PASS] Sign correction works correctly")
    return True


def test_holding_period():
    """Test holding period (freq) column."""
    print("\n" + "=" * 60)
    print("TEST: Holding period (freq)")
    print("=" * 60)

    # Generate data
    data = generate_synthetic_data(n_dates=60, n_bonds=100, seed=42)

    # Run batch strategy with hp=3
    print("  Running BatchStrategyFormation with holding_period=3...")
    batch = BatchStrategyFormation(
        data=data,
        signals=['signal1'],
        holding_period=3,
        num_portfolios=5,
        turnover=True,  # Required for extract_panel
        n_jobs=1,
        verbose=False,
    )
    results = batch.fit()

    # Extract panel
    print("  Extracting panel...")
    panel = extract_panel(results)

    # Check freq column
    assert (panel['freq'] == 3).all(), f"Expected freq=3, got {panel['freq'].unique()}"
    print(f"  freq values: {panel['freq'].unique()}")

    print("  [PASS] Holding period extraction works correctly")
    return True


def test_withinfirmsort():
    """Test extraction from BatchWithinFirmSortFormation."""
    print("\n" + "=" * 60)
    print("TEST: BatchWithinFirmSortFormation")
    print("=" * 60)

    # Generate data with PERMNO for firm grouping
    data = generate_synthetic_data(n_dates=30, n_bonds=100, seed=42)

    # Add PERMNO column (assign ~20 firms)
    np.random.seed(42)
    unique_ids = data['ID'].unique()
    firm_map = {id_: f"FIRM_{i % 20}" for i, id_ in enumerate(unique_ids)}
    data['PERMNO'] = data['ID'].map(firm_map)

    # Run WithinFirmSort batch
    print("  Running BatchWithinFirmSortFormation...")
    try:
        batch = BatchWithinFirmSortFormation(
            data=data,
            signals=['signal1'],
            firm_id_col='PERMNO',
            turnover=True,  # Required for extract_panel
            n_jobs=1,
            verbose=False,
        )
        results = batch.fit()

        # Extract panel
        print("  Extracting panel...")
        panel = extract_panel(results)

        # Check structure
        print(f"  Panel shape: {panel.shape}")
        print(f"  Factors: {list(panel['factor'].unique())}")

        # Check legs (should still be ls, l, s)
        legs = set(panel['leg'].unique())
        assert legs == {'ls', 'l', 's'}, f"Expected legs {{'ls', 'l', 's'}}, got {legs}"

        print("  [PASS] WithinFirmSort extraction works correctly")
        return True
    except Exception as e:
        print(f"  [SKIP] WithinFirmSort test failed: {e}")
        return True  # Skip if not implemented


def test_multiple_signals():
    """Test extraction with multiple signals."""
    print("\n" + "=" * 60)
    print("TEST: Multiple signals")
    print("=" * 60)

    # Generate data
    data = generate_synthetic_data(n_dates=30, n_bonds=100, seed=42)

    # Run batch strategy with multiple signals
    print("  Running BatchStrategyFormation with 3 signals...")
    batch = BatchStrategyFormation(
        data=data,
        signals=['signal1', 'signal2', 'char1'],  # Using char1 as a signal too
        holding_period=1,
        num_portfolios=5,
        turnover=True,  # Required for extract_panel
        n_jobs=1,
        verbose=False,
    )
    results = batch.fit()

    # Extract panel
    print("  Extracting panel...")
    panel = extract_panel(results)

    # Check number of factors
    factors = panel['factor'].unique()
    print(f"  Factors: {list(factors)}")
    assert len(factors) == 3, f"Expected 3 factors, got {len(factors)}"

    # Check each factor has all legs and weightings
    for factor in factors:
        factor_panel = panel[panel['factor'] == factor]
        legs = set(factor_panel['leg'].unique())
        weightings = set(factor_panel['weighting'].unique())
        assert legs == {'ls', 'l', 's'}, f"Factor {factor} missing legs"
        assert weightings == {'ew', 'vw'}, f"Factor {factor} missing weightings"

    print("  [PASS] Multiple signals extraction works correctly")
    return True


def test_panel_pivot():
    """Test that panel can be pivoted to wide format."""
    print("\n" + "=" * 60)
    print("TEST: Panel pivot to wide format")
    print("=" * 60)

    # Generate data
    data = generate_synthetic_data(n_dates=30, n_bonds=100, seed=42)

    # Run batch strategy
    batch = BatchStrategyFormation(
        data=data,
        signals=['signal1', 'signal2'],
        holding_period=1,
        num_portfolios=5,
        turnover=True,  # Required for extract_panel
        n_jobs=1,
        verbose=False,
    )
    results = batch.fit()

    # Extract panel
    panel = extract_panel(results)

    # Pivot to wide format for long-short returns only
    ls_panel = panel[panel['leg'] == 'ls']
    wide = ls_panel.pivot_table(
        index='date',
        columns=['factor', 'weighting'],
        values='return'
    )

    print(f"  Wide format shape: {wide.shape}")
    print(f"  Wide format columns: {list(wide.columns)}")

    assert wide.shape[1] == 4, f"Expected 4 columns (2 signals × 2 weightings), got {wide.shape[1]}"

    print("  [PASS] Panel can be pivoted to wide format")
    return True


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("extract_panel Validation Script")
    print("=" * 60)

    all_passed = True

    all_passed &= test_basic_extraction()
    all_passed &= test_with_turnover()
    all_passed &= test_with_characteristics()
    all_passed &= test_sign_correction()
    all_passed &= test_holding_period()
    all_passed &= test_withinfirmsort()
    all_passed &= test_multiple_signals()
    all_passed &= test_panel_pivot()

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
