"""
Synthetic Data Validation Tests for AnomalyAssayer Optimizations
================================================================

This script tests that:
1. turnover=False, save_idx=False produces identical RETURNS to full mode
2. TO and nbonds are correctly NaN when disabled
3. Results are deterministic across multiple runs

Run with: python test_synthetic_validation.py
"""

import numpy as np
import pandas as pd
import sys
import time
from pathlib import Path

# Add parent directories to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import PyBondLab as pbl


def create_synthetic_panel(n_bonds=200, n_dates=36, seed=42):
    """
    Create realistic synthetic bond panel data.

    Parameters
    ----------
    n_bonds : int
        Number of unique bonds
    n_dates : int
        Number of monthly dates
    seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Synthetic bond panel with all required columns
    """
    np.random.seed(seed)

    # Create date range
    dates = pd.date_range('2020-01-31', periods=n_dates, freq='ME')
    bond_ids = [f'CUSIP_{i:04d}' for i in range(n_bonds)]

    records = []
    for t, date in enumerate(dates):
        # Varying number of bonds per date (some enter/exit)
        n_available = int(n_bonds * np.random.uniform(0.7, 0.95))
        available_bonds = np.random.choice(bond_ids, size=n_available, replace=False)

        for bond_id in available_bonds:
            # Generate realistic characteristics
            bond_idx = int(bond_id.split('_')[1])

            # Stable characteristics (with some variation over time)
            base_maturity = 2 + (bond_idx % 20)  # 2-22 years
            maturity = base_maturity - t * 0.083  # Decreases over time
            maturity = max(0.5, maturity)

            # Rating can change over time
            base_rating = 5 + (bond_idx % 12)  # 5-16 (covers IG and NIG)
            rating_shock = np.random.choice([-1, 0, 0, 0, 0, 1], p=[0.05, 0.7, 0.1, 0.05, 0.05, 0.05])
            rating = np.clip(base_rating + rating_shock, 1, 21)

            # Market cap (time-varying)
            base_cap = 100 + (bond_idx % 900)
            cap_return = np.random.normal(0.01, 0.05)
            market_cap = base_cap * (1 + cap_return) ** t

            # Returns (cross-sectionally correlated + idiosyncratic)
            market_return = np.random.normal(0.005, 0.02)
            idio_return = np.random.normal(0, 0.03)
            ret = market_return + idio_return

            # Sort variable (e.g., VaR or volatility proxy)
            var_5pct = np.random.normal(0, 0.1)

            records.append({
                'date': date,
                'cusip_id': bond_id,
                'ret': ret,
                'sp_rating': float(rating),
                'bond_maturity': maturity,
                'MARKET_CAP': market_cap,
                'var_5pct': var_5pct,
            })

    df = pd.DataFrame(records)

    # Add some NaN values (realistic missing data)
    nan_mask_ret = np.random.random(len(df)) < 0.02
    nan_mask_var = np.random.random(len(df)) < 0.03

    df.loc[nan_mask_ret, 'ret'] = np.nan
    df.loc[nan_mask_var, 'var_5pct'] = np.nan

    # Set VW column
    df['VW'] = df['MARKET_CAP']

    # Sort by bond and date
    df = df.sort_values(['cusip_id', 'date']).reset_index(drop=True)

    return df


def compare_results(df_full, df_fast, test_name, check_to=True, check_nbonds=True):
    """
    Compare results from full and fast mode.

    Parameters
    ----------
    df_full : pd.DataFrame
        Results from full mode (turnover=True, save_idx=True)
    df_fast : pd.DataFrame
        Results from fast mode (turnover=False, save_idx=False)
    test_name : str
        Name of the test for reporting
    check_to : bool
        Whether to check that TO is NaN in fast mode
    check_nbonds : bool
        Whether to check that nbonds is NaN in fast mode

    Returns
    -------
    bool
        True if all checks pass
    """
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")

    all_passed = True

    # Check that returns are identical
    # Merge on all identifying columns
    merge_cols = ['Holding', 'Sort', 'Rating', 'Ret_string', 'Subset', 'weight', 'type']
    full_subset = df_full[merge_cols + ['ret']].copy()
    fast_subset = df_fast[merge_cols + ['ret']].copy()

    # Reset index for comparison
    full_subset = full_subset.reset_index()
    fast_subset = fast_subset.reset_index()

    # Merge
    merged = full_subset.merge(
        fast_subset,
        on=['index'] + merge_cols,
        suffixes=('_full', '_fast')
    )

    # Check returns match (with tolerance for floating point)
    ret_diff = np.abs(merged['ret_full'] - merged['ret_fast'])
    nan_match = merged['ret_full'].isna() == merged['ret_fast'].isna()

    if not nan_match.all():
        print(f"  ✗ FAIL: Return NaN patterns differ")
        n_mismatch = (~nan_match).sum()
        print(f"    {n_mismatch} rows have different NaN patterns")
        all_passed = False
    elif ret_diff[~merged['ret_full'].isna()].max() > 1e-10:
        print(f"  ✗ FAIL: Returns differ (max diff: {ret_diff.max():.2e})")
        all_passed = False
    else:
        print(f"  ✓ PASS: Returns are identical")

    # Check TO is NaN in fast mode
    if check_to:
        if df_fast['TO'].isna().all():
            print(f"  ✓ PASS: TO is all NaN in fast mode")
        else:
            print(f"  ✗ FAIL: TO should be all NaN in fast mode")
            all_passed = False

        # Check TO has values in full mode
        if df_full['TO'].notna().any():
            print(f"  ✓ PASS: TO has values in full mode")
        else:
            print(f"  ⚠ WARNING: TO is all NaN in full mode (may be expected)")

    # Check nbonds is NaN in fast mode
    if check_nbonds:
        if df_fast['nbonds'].isna().all():
            print(f"  ✓ PASS: nbonds is all NaN in fast mode")
        else:
            print(f"  ✗ FAIL: nbonds should be all NaN in fast mode")
            all_passed = False

        # Check nbonds has values in full mode
        if df_full['nbonds'].notna().any():
            print(f"  ✓ PASS: nbonds has values in full mode")
        else:
            print(f"  ⚠ WARNING: nbonds is all NaN in full mode (may be expected)")

    return all_passed


def test_basic_comparison():
    """Test 1: Basic comparison with simple parameters."""
    print("\n" + "="*70)
    print("TEST 1: Basic Comparison (Simple Parameters)")
    print("="*70)

    # Create synthetic data
    data = create_synthetic_panel(n_bonds=100, n_dates=24, seed=42)
    print(f"Created synthetic panel: {len(data)} rows, {data['cusip_id'].nunique()} bonds, {data['date'].nunique()} dates")

    sort_var = "var_5pct"

    # Run full mode
    print("\nRunning FULL mode (turnover=True, save_idx=True)...")
    t0 = time.time()
    result_full = pbl.AssayAnomaly(
        data=data,
        sort_var=sort_var,
        IDvar="cusip_id",
        RETvar="ret",
        RATINGvar="sp_rating",
        holding_periods=[1],
        nport=[3],
        ratings=[None],
        turnover=True,
        save_idx=True,
        verbose=False
    )
    t_full = time.time() - t0
    print(f"Full mode completed in {t_full:.2f}s")

    # Run fast mode
    print("\nRunning FAST mode (turnover=False, save_idx=False)...")
    t0 = time.time()
    result_fast = pbl.AssayAnomaly(
        data=data,
        sort_var=sort_var,
        IDvar="cusip_id",
        RETvar="ret",
        RATINGvar="sp_rating",
        holding_periods=[1],
        nport=[3],
        ratings=[None],
        turnover=False,
        save_idx=False,
        verbose=False
    )
    t_fast = time.time() - t0
    print(f"Fast mode completed in {t_fast:.2f}s")

    # Compare results
    return compare_results(result_full.df, result_fast.df, "Basic Comparison")


def test_multiple_ratings():
    """Test 2: Multiple rating categories."""
    print("\n" + "="*70)
    print("TEST 2: Multiple Rating Categories")
    print("="*70)

    data = create_synthetic_panel(n_bonds=150, n_dates=24, seed=123)
    sort_var = "var_5pct"

    # Run full mode
    print("\nRunning FULL mode...")
    t0 = time.time()
    result_full = pbl.AssayAnomaly(
        data=data,
        sort_var=sort_var,
        IDvar="cusip_id",
        RETvar="ret",
        RATINGvar="sp_rating",
        holding_periods=[1],
        nport=[5],
        ratings=[None, 'IG', 'NIG'],
        turnover=True,
        save_idx=True,
        verbose=False
    )
    t_full = time.time() - t0
    print(f"Full mode completed in {t_full:.2f}s")

    # Run fast mode
    print("\nRunning FAST mode...")
    t0 = time.time()
    result_fast = pbl.AssayAnomaly(
        data=data,
        sort_var=sort_var,
        IDvar="cusip_id",
        RETvar="ret",
        RATINGvar="sp_rating",
        holding_periods=[1],
        nport=[5],
        ratings=[None, 'IG', 'NIG'],
        turnover=False,
        save_idx=False,
        verbose=False
    )
    t_fast = time.time() - t0
    print(f"Fast mode completed in {t_fast:.2f}s")

    return compare_results(result_full.df, result_fast.df, "Multiple Ratings")


def test_subset_filter():
    """Test 3: Subset filtering by maturity."""
    print("\n" + "="*70)
    print("TEST 3: Subset Filtering (Maturity Buckets)")
    print("="*70)

    data = create_synthetic_panel(n_bonds=200, n_dates=24, seed=456)
    sort_var = "var_5pct"

    subset_filter = {
        "bond_maturity": [
            (0, 5),
            (5, 10),
            (10, np.inf)
        ]
    }

    # Run full mode
    print("\nRunning FULL mode...")
    t0 = time.time()
    result_full = pbl.AssayAnomaly(
        data=data,
        sort_var=sort_var,
        subset_filter=subset_filter,
        IDvar="cusip_id",
        RETvar="ret",
        RATINGvar="sp_rating",
        holding_periods=[1],
        nport=[3],
        ratings=[None],
        turnover=True,
        save_idx=True,
        verbose=False
    )
    t_full = time.time() - t0
    print(f"Full mode completed in {t_full:.2f}s")

    # Run fast mode
    print("\nRunning FAST mode...")
    t0 = time.time()
    result_fast = pbl.AssayAnomaly(
        data=data,
        sort_var=sort_var,
        subset_filter=subset_filter,
        IDvar="cusip_id",
        RETvar="ret",
        RATINGvar="sp_rating",
        holding_periods=[1],
        nport=[3],
        ratings=[None],
        turnover=False,
        save_idx=False,
        verbose=False
    )
    t_fast = time.time() - t0
    print(f"Fast mode completed in {t_fast:.2f}s")

    return compare_results(result_full.df, result_fast.df, "Subset Filter")


def test_complex_grid():
    """Test 4: Complex parameter grid."""
    print("\n" + "="*70)
    print("TEST 4: Complex Parameter Grid")
    print("="*70)

    data = create_synthetic_panel(n_bonds=200, n_dates=36, seed=789)
    sort_var = "var_5pct"

    subset_filter = {
        "bond_maturity": [
            (0, 5),
            (5, 10),
        ]
    }

    # Run full mode
    print("\nRunning FULL mode...")
    t0 = time.time()
    result_full = pbl.AssayAnomaly(
        data=data,
        sort_var=sort_var,
        subset_filter=subset_filter,
        IDvar="cusip_id",
        RETvar="ret",
        RATINGvar="sp_rating",
        holding_periods=[1, 3],
        nport=[3, 5],
        ratings=[None, 'IG'],
        turnover=True,
        save_idx=True,
        verbose=False
    )
    t_full = time.time() - t0
    print(f"Full mode completed in {t_full:.2f}s")

    # Run fast mode
    print("\nRunning FAST mode...")
    t0 = time.time()
    result_fast = pbl.AssayAnomaly(
        data=data,
        sort_var=sort_var,
        subset_filter=subset_filter,
        IDvar="cusip_id",
        RETvar="ret",
        RATINGvar="sp_rating",
        holding_periods=[1, 3],
        nport=[3, 5],
        ratings=[None, 'IG'],
        turnover=False,
        save_idx=False,
        verbose=False
    )
    t_fast = time.time() - t0
    print(f"Fast mode completed in {t_fast:.2f}s")

    return compare_results(result_full.df, result_fast.df, "Complex Grid")


def test_determinism():
    """Test 5: Verify determinism (multiple runs produce same results)."""
    print("\n" + "="*70)
    print("TEST 5: Determinism Check")
    print("="*70)

    data = create_synthetic_panel(n_bonds=100, n_dates=24, seed=42)
    sort_var = "var_5pct"

    print("\nRunning FAST mode twice...")

    # Run 1
    result1 = pbl.AssayAnomaly(
        data=data,
        sort_var=sort_var,
        IDvar="cusip_id",
        RETvar="ret",
        RATINGvar="sp_rating",
        holding_periods=[1],
        nport=[3],
        ratings=[None],
        turnover=False,
        save_idx=False,
        verbose=False
    )

    # Run 2
    result2 = pbl.AssayAnomaly(
        data=data,
        sort_var=sort_var,
        IDvar="cusip_id",
        RETvar="ret",
        RATINGvar="sp_rating",
        holding_periods=[1],
        nport=[3],
        ratings=[None],
        turnover=False,
        save_idx=False,
        verbose=False
    )

    # Compare
    df1 = result1.df.reset_index()
    df2 = result2.df.reset_index()

    if df1.equals(df2):
        print("  ✓ PASS: Two runs produce identical results (deterministic)")
        return True
    else:
        print("  ✗ FAIL: Two runs produce different results!")
        return False


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "#"*70)
    print("# SYNTHETIC DATA VALIDATION TESTS")
    print("# Testing that turnover=False, save_idx=False produces identical returns")
    print("#"*70)

    results = []

    results.append(("Basic Comparison", test_basic_comparison()))
    results.append(("Multiple Ratings", test_multiple_ratings()))
    results.append(("Subset Filter", test_subset_filter()))
    results.append(("Complex Grid", test_complex_grid()))
    results.append(("Determinism", test_determinism()))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        all_passed = all_passed and passed

    print("\n" + "="*70)
    if all_passed:
        print("ALL TESTS PASSED!")
        print("The turnover/save_idx optimization produces identical returns.")
    else:
        print("SOME TESTS FAILED!")
        print("Please investigate the failures above.")
    print("="*70)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
