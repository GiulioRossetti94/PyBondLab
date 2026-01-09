"""
Validation script for enforce_contiguous=True support in fast path.

This script validates that:
1. enforce_contiguous correctly identifies and handles gaps in monthly data
2. Signals are correctly computed on expanded (contiguous) data
3. Fast path produces correct results for non-contiguous data

Run: python examples/validate_enforce_contiguous.py
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '.')

from PyBondLab import DataUncertaintyAnalysis, Momentum, LTreversal
from PyBondLab.numba_core import expand_to_contiguous, get_bond_boundaries


def test_expand_to_contiguous_kernel():
    """Test the expand_to_contiguous numba kernel directly."""
    print("=" * 60)
    print("TEST 1: expand_to_contiguous Kernel Test")
    print("=" * 60)

    # Create test data with known gaps
    # Bond 0: months 0, 1, 2, 3, 4 (contiguous - 5 rows)
    # Bond 1: months 0, 1, 3, 4 (gap at month 2 - 4 rows -> 5 rows expanded)
    # Bond 2: months 0, 2, 5 (gaps at 1, 3, 4 - 3 rows -> 6 rows expanded)

    # Create sorted data (by ID, then by month)
    month_idx = np.array([0, 1, 2, 3, 4,  # Bond 0
                          0, 1, 3, 4,      # Bond 1
                          0, 2, 5],        # Bond 2
                         dtype=np.int64)

    ret = np.array([0.01, 0.02, 0.03, 0.04, 0.05,  # Bond 0
                    0.10, 0.11, 0.13, 0.14,         # Bond 1
                    0.20, 0.22, 0.25],              # Bond 2
                   dtype=np.float64)

    vw = np.array([100, 100, 100, 100, 100,  # Bond 0
                   200, 200, 200, 200,        # Bond 1
                   300, 300, 300],            # Bond 2
                  dtype=np.float64)

    # ID index (0=bond0, 1=bond1, 2=bond2)
    id_idx = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2], dtype=np.int64)

    # Get bond boundaries
    bond_starts = get_bond_boundaries(id_idx)
    print(f"Original data: {len(ret)} rows")
    print(f"Bond starts: {bond_starts}")

    # Expand to contiguous
    out_ret, out_vw, new_bond_starts = expand_to_contiguous(
        month_idx, ret, vw, bond_starts
    )

    print(f"\nExpanded data: {len(out_ret)} rows")
    print(f"New bond starts: {new_bond_starts}")

    # Expected:
    # Bond 0: 5 rows (was contiguous)
    # Bond 1: 5 rows (gap at month 2 filled with NaN)
    # Bond 2: 6 rows (gaps at months 1, 3, 4 filled with NaN)
    # Total: 5 + 5 + 6 = 16 rows
    assert len(out_ret) == 16, f"Expected 16 rows, got {len(out_ret)}"

    # Check Bond 0 (contiguous - no changes)
    bond0_ret = out_ret[new_bond_starts[0]:new_bond_starts[1]]
    assert len(bond0_ret) == 5, f"Bond 0 should have 5 rows"
    assert np.allclose(bond0_ret, [0.01, 0.02, 0.03, 0.04, 0.05]), "Bond 0 returns incorrect"

    # Check Bond 1 (gap at month 2)
    bond1_ret = out_ret[new_bond_starts[1]:new_bond_starts[2]]
    assert len(bond1_ret) == 5, f"Bond 1 should have 5 rows after expansion"
    assert bond1_ret[0] == 0.10, "Bond 1 month 0 incorrect"
    assert bond1_ret[1] == 0.11, "Bond 1 month 1 incorrect"
    assert np.isnan(bond1_ret[2]), "Bond 1 month 2 should be NaN (gap)"
    assert bond1_ret[3] == 0.13, "Bond 1 month 3 incorrect"
    assert bond1_ret[4] == 0.14, "Bond 1 month 4 incorrect"

    # Check Bond 2 (gaps at months 1, 3, 4)
    bond2_ret = out_ret[new_bond_starts[2]:new_bond_starts[3]]
    assert len(bond2_ret) == 6, f"Bond 2 should have 6 rows after expansion"
    assert bond2_ret[0] == 0.20, "Bond 2 month 0 incorrect"
    assert np.isnan(bond2_ret[1]), "Bond 2 month 1 should be NaN (gap)"
    assert bond2_ret[2] == 0.22, "Bond 2 month 2 incorrect"
    assert np.isnan(bond2_ret[3]), "Bond 2 month 3 should be NaN (gap)"
    assert np.isnan(bond2_ret[4]), "Bond 2 month 4 should be NaN (gap)"
    assert bond2_ret[5] == 0.25, "Bond 2 month 5 incorrect"

    print("\n✓ expand_to_contiguous kernel test PASSED")
    return True


def test_signal_with_gaps():
    """Test that enforce_contiguous correctly handles gaps in signal computation."""
    print("\n" + "=" * 60)
    print("TEST 2: Signal Computation with Gaps")
    print("=" * 60)

    # Create synthetic data with intentional gaps
    np.random.seed(42)

    # Create date range (monthly)
    dates = pd.date_range('2020-01-31', periods=12, freq='ME')

    # Create bonds with different gap patterns
    records = []

    # Bond A: All months present (contiguous)
    for i, d in enumerate(dates):
        records.append({'date': d, 'ID': 'A', 'ret': 0.01 * (i + 1), 'VW': 100, 'RATING_NUM': 5})

    # Bond B: Missing months 3, 4, 5 (Feb, Mar, Apr 2020 missing)
    for i, d in enumerate(dates):
        if i not in [2, 3, 4]:  # Skip months 3, 4, 5 (0-indexed)
            records.append({'date': d, 'ID': 'B', 'ret': 0.01 * (i + 1), 'VW': 200, 'RATING_NUM': 5})

    # Bond C: Only first and last 3 months (big gap in middle)
    for i, d in enumerate(dates):
        if i < 3 or i >= 9:
            records.append({'date': d, 'ID': 'C', 'ret': 0.01 * (i + 1), 'VW': 300, 'RATING_NUM': 5})

    data = pd.DataFrame(records)
    data = data.sort_values(['ID', 'date']).reset_index(drop=True)

    print(f"Data shape: {data.shape}")
    print(f"Bond A rows: {len(data[data['ID'] == 'A'])} (all 12 months)")
    print(f"Bond B rows: {len(data[data['ID'] == 'B'])} (9 months, gap in middle)")
    print(f"Bond C rows: {len(data[data['ID'] == 'C'])} (6 months, big gap)")

    # Test with lookback=3, skip=1
    lookback = 3
    skip = 1

    # WITHOUT enforce_contiguous: gap-blind (treats rows as consecutive)
    print("\n--- Without enforce_contiguous (default) ---")
    mom_default = Momentum(lookback_period=lookback, skip=skip)

    dua_default = DataUncertaintyAnalysis(
        data=data,
        strategy=mom_default,
        holding_periods=[1],
        num_portfolios=5,
        include_baseline=True,
        verbose=False,
    )
    result_default = dua_default.fit()
    ew_ea_default = result_default.ew_ex_ante
    nan_default = ew_ea_default.isna().sum().sum()

    # WITH enforce_contiguous: gap-aware (fills gaps with NaN)
    print("--- With enforce_contiguous=True ---")
    mom_contiguous = Momentum(lookback_period=lookback, skip=skip, enforce_contiguous=True)

    dua_contiguous = DataUncertaintyAnalysis(
        data=data,
        strategy=mom_contiguous,
        holding_periods=[1],
        num_portfolios=5,
        include_baseline=True,
        verbose=True,
    )
    result_contiguous = dua_contiguous.fit()
    ew_ea_contiguous = result_contiguous.ew_ex_ante
    nan_contiguous = ew_ea_contiguous.isna().sum().sum()

    print(f"\nNaN comparison:")
    print(f"  Default (gap-blind): {nan_default} NaN values")
    print(f"  enforce_contiguous:  {nan_contiguous} NaN values")

    # enforce_contiguous should have MORE NaNs because gaps cause signal = NaN
    # (any NaN in lookback window -> signal is NaN)
    assert nan_contiguous >= nan_default, \
        "enforce_contiguous should produce >= NaN values due to gap detection"

    print("\n✓ Signal computation with gaps test PASSED")
    return True


def test_contiguous_vs_non_contiguous():
    """Compare enforce_contiguous on contiguous data (should match default)."""
    print("\n" + "=" * 60)
    print("TEST 3: Contiguous Data (enforce_contiguous should match default)")
    print("=" * 60)

    # Create fully contiguous data (all bonds have all months)
    np.random.seed(42)
    dates = pd.date_range('2020-01-31', periods=24, freq='ME')
    n_bonds = 50

    records = []
    for bond in range(n_bonds):
        for d in dates:
            records.append({
                'date': d,
                'ID': f'BOND_{bond:03d}',
                'ret': np.random.randn() * 0.02,
                'VW': np.random.uniform(50, 200),
                'RATING_NUM': np.random.randint(1, 22)
            })

    data = pd.DataFrame(records)
    data = data.sort_values(['ID', 'date']).reset_index(drop=True)

    print(f"Data shape: {data.shape} (fully contiguous)")

    # Default (no expand)
    mom_default = Momentum(lookback_period=3, skip=1)
    dua_default = DataUncertaintyAnalysis(
        data=data,
        strategy=mom_default,
        holding_periods=[1],
        num_portfolios=5,
        include_baseline=True,
        verbose=False,
    )
    result_default = dua_default.fit()

    # With enforce_contiguous
    mom_contiguous = Momentum(lookback_period=3, skip=1, enforce_contiguous=True)
    dua_contiguous = DataUncertaintyAnalysis(
        data=data,
        strategy=mom_contiguous,
        holding_periods=[1],
        num_portfolios=5,
        include_baseline=True,
        verbose=False,
    )
    result_contiguous = dua_contiguous.fit()

    # Compare results - should be identical for contiguous data
    ew_default = result_default.ew_ex_ante
    ew_contiguous = result_contiguous.ew_ex_ante

    # Align indices
    common_dates = ew_default.index.intersection(ew_contiguous.index)
    common_cols = ew_default.columns.intersection(ew_contiguous.columns)

    diff = np.abs(ew_default.loc[common_dates, common_cols].values -
                  ew_contiguous.loc[common_dates, common_cols].values)
    max_diff = np.nanmax(diff)

    print(f"\nMax difference: {max_diff:.2e}")

    if max_diff < 1e-10:
        print("✓ Results match exactly (as expected for contiguous data)")
    else:
        print(f"⚠ Results differ by {max_diff:.2e}")

    assert max_diff < 1e-10, \
        f"enforce_contiguous should match default for contiguous data, but diff={max_diff}"

    print("\n✓ Contiguous data comparison test PASSED")
    return True


def test_mutual_exclusion():
    """Test that enforce_contiguous and drop_na cannot both be True."""
    print("\n" + "=" * 60)
    print("TEST 4: Mutual Exclusion (enforce_contiguous vs drop_na)")
    print("=" * 60)

    try:
        mom = Momentum(lookback_period=3, skip=1, enforce_contiguous=True, drop_na=True)
        print("✗ Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")

    try:
        ltr = LTreversal(lookback_period=12, skip=1, enforce_contiguous=True, drop_na=True)
        print("✗ Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")

    print("\n✓ Mutual exclusion test PASSED")
    return True


def test_ltreversal():
    """Test enforce_contiguous with LTreversal strategy."""
    print("\n" + "=" * 60)
    print("TEST 5: LTreversal with enforce_contiguous")
    print("=" * 60)

    np.random.seed(42)
    dates = pd.date_range('2018-01-31', periods=36, freq='ME')

    # Create data with gaps
    records = []
    for bond in range(20):
        # Randomly drop some months
        active_months = np.random.choice(range(36), size=30, replace=False)
        active_months = np.sort(active_months)
        for m in active_months:
            records.append({
                'date': dates[m],
                'ID': f'BOND_{bond:03d}',
                'ret': np.random.randn() * 0.02,
                'VW': np.random.uniform(50, 200),
                'RATING_NUM': np.random.randint(1, 22)
            })

    data = pd.DataFrame(records)
    data = data.sort_values(['ID', 'date']).reset_index(drop=True)

    print(f"Data shape: {data.shape}")
    print(f"Expected full panel: {20 * 36} rows")
    print(f"Actual rows: {len(data)} ({100*len(data)/(20*36):.1f}% of full)")

    # With enforce_contiguous
    ltr = LTreversal(lookback_period=12, skip=1, enforce_contiguous=True)
    dua = DataUncertaintyAnalysis(
        data=data,
        strategy=ltr,
        holding_periods=[1],
        num_portfolios=5,
        include_baseline=True,
        verbose=True,
    )
    result = dua.fit()

    ew_ea = result.ew_ex_ante
    n_valid = (~ew_ea.isna()).sum().sum()
    n_total = ew_ea.shape[0] * ew_ea.shape[1]

    print(f"\nEW EA valid values: {n_valid} / {n_total} ({100*n_valid/n_total:.1f}%)")

    print("\n✓ LTreversal with enforce_contiguous test PASSED")
    return True


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("VALIDATING enforce_contiguous=True SUPPORT IN FAST PATH")
    print("=" * 60)

    tests = [
        ("expand_to_contiguous Kernel", test_expand_to_contiguous_kernel),
        ("Signal Computation with Gaps", test_signal_with_gaps),
        ("Contiguous Data Comparison", test_contiguous_vs_non_contiguous),
        ("Mutual Exclusion", test_mutual_exclusion),
        ("LTreversal Strategy", test_ltreversal),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result, None))
        except Exception as e:
            results.append((name, False, str(e)))
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed, error in results:
        status = "✓ PASS" if passed else f"✗ FAIL: {error}"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
