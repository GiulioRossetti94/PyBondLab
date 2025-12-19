#!/usr/bin/env python3
"""
Test script to debug chars NaN issue with SingleSort.

This reproduces the issue where chars=['lib'] returns all NaN despite
'lib' having valid values (~84% non-NaN).
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '.')

import PyBondLab as pbl
from PyBondLab.pbl_test import generate_synthetic_data


def test_chars_with_synthetic_data():
    """Test chars computation with synthetic data."""
    print("=" * 60)
    print("Test 1: Synthetic data with chars")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    data = generate_synthetic_data(n_dates=60, n_bonds=500, seed=42)

    # Create 'lib' column with ~16% missing (matching user's data pattern)
    lib_values = np.random.randn(len(data))
    missing_mask = np.random.random(len(data)) < 0.16
    lib_values[missing_mask] = np.nan
    data['lib'] = lib_values

    print(f"Data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"'lib' non-null: {data['lib'].notna().sum()} ({data['lib'].notna().mean()*100:.1f}%)")
    print(f"'lib' null: {data['lib'].isna().sum()} ({data['lib'].isna().mean()*100:.1f}%)")

    # Run SingleSort with chars
    strategy = pbl.SingleSort(
        holding_period=1,
        sort_var='signal1',
        num_portfolios=5
    )

    sf = pbl.StrategyFormation(
        data=data,
        strategy=strategy,
        chars=['lib'],
        turnover=False,
        verbose=True
    )

    result = sf.fit()

    # Check chars output
    chars_ew, chars_vw = result.get_characteristics()

    print("\n" + "-" * 40)
    print("Chars output:")
    print("-" * 40)

    if chars_ew is None:
        print("ERROR: chars_ew is None!")
        return False

    lib_ew = chars_ew.get('lib')
    if lib_ew is None:
        print("ERROR: 'lib' not in chars_ew!")
        return False

    print(f"lib_ew shape: {lib_ew.shape}")
    print(f"lib_ew all NaN: {lib_ew.isnull().all().all()}")
    print(f"lib_ew non-NaN count: {lib_ew.notna().sum().sum()}")
    print(f"\nFirst 5 rows:\n{lib_ew.head()}")

    if lib_ew.isnull().all().all():
        print("\n*** ISSUE REPRODUCED: All chars are NaN! ***")
        return False
    else:
        print("\n*** OK: Chars have valid values ***")
        return True


def test_chars_debug_internals():
    """Debug the internal flow of chars computation."""
    print("\n" + "=" * 60)
    print("Test 2: Debug internal chars flow")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    data = generate_synthetic_data(n_dates=10, n_bonds=100, seed=42)

    # Create 'lib' column with ~16% missing
    lib_values = np.random.randn(len(data))
    missing_mask = np.random.random(len(data)) < 0.16
    lib_values[missing_mask] = np.nan
    data['lib'] = lib_values

    print(f"Data shape: {data.shape}")
    print(f"'lib' non-null: {data['lib'].notna().sum()}")

    # Create strategy
    strategy = pbl.SingleSort(
        holding_period=1,
        sort_var='signal1',
        num_portfolios=5
    )

    sf = pbl.StrategyFormation(
        data=data,
        strategy=strategy,
        chars=['lib'],
        turnover=False,
        verbose=False
    )

    # Check that self.chars is set
    print(f"sf.chars: {sf.chars}")

    # Run fit and capture intermediate data
    # Monkey-patch _form_single_period to debug
    original_form = sf._form_single_period
    debug_info = {'calls': 0, 'It1m_cols': None, 'char_values': None}

    def debug_form(*args, **kwargs):
        debug_info['calls'] += 1
        if debug_info['calls'] <= 1:  # Only capture first call
            It0, It1, It1m = args[0], args[1], args[2]
            debug_info['It0_cols'] = list(It0.columns) if not It0.empty else []
            debug_info['It1_cols'] = list(It1.columns) if not It1.empty else []
            debug_info['It1m_cols'] = list(It1m.columns) if not It1m.empty else []
            debug_info['It0_shape'] = It0.shape
            debug_info['It1_shape'] = It1.shape
            debug_info['It1m_shape'] = It1m.shape
            debug_info['It1m_lib_notna'] = It1m['lib'].notna().sum() if 'lib' in It1m.columns else 'NOT IN COLUMNS'

        result = original_form(*args, **kwargs)

        if debug_info['calls'] <= 1:
            if result.get('chars_ew') is not None:
                debug_info['result_chars_ew'] = result['chars_ew'].to_dict()
            else:
                debug_info['result_chars_ew'] = None

        return result

    sf._form_single_period = debug_form

    result = sf.fit()

    print(f"\nDebug info from first _form_single_period call:")
    print(f"  It0 columns: {debug_info.get('It0_cols')}")
    print(f"  It1 columns: {debug_info.get('It1_cols')}")
    print(f"  It1m columns: {debug_info.get('It1m_cols')}")
    print(f"  It0 shape: {debug_info.get('It0_shape')}")
    print(f"  It1 shape: {debug_info.get('It1_shape')}")
    print(f"  It1m shape: {debug_info.get('It1m_shape')}")
    print(f"  It1m 'lib' non-NaN: {debug_info.get('It1m_lib_notna')}")
    print(f"  Result chars_ew: {debug_info.get('result_chars_ew')}")

    # Restore original
    sf._form_single_period = original_form

    # Check required_cols after fit
    print(f"\nAfter fit:")
    print(f"  sf.required_cols: {sf.required_cols}")
    print(f"  'lib' in sf.required_cols: {'lib' in sf.required_cols}")

    # Check final result
    chars_ew, chars_vw = result.get_characteristics()
    if chars_ew is not None and 'lib' in chars_ew:
        lib_ew = chars_ew['lib']
        print(f"\nFinal lib_ew all NaN: {lib_ew.isnull().all().all()}")
        print(f"Final lib_ew non-NaN count: {lib_ew.notna().sum().sum()}")


def test_chars_with_column_mapping():
    """Test chars with column mapping (like user's case)."""
    print("\n" + "=" * 60)
    print("Test 3: Chars with column mapping")
    print("=" * 60)

    # Generate synthetic data with user-like column names
    np.random.seed(42)
    data = generate_synthetic_data(n_dates=30, n_bonds=200, seed=42)

    # Rename columns to match user's data
    data = data.rename(columns={
        'ID': 'cusip',
        'ret': 'ret_vw_bgn',
        'VW': 'mcap_e',
        'RATING_NUM': 'spc_rat'
    })

    # Create 'lib' column with ~16% missing
    lib_values = np.random.randn(len(data))
    missing_mask = np.random.random(len(data)) < 0.16
    lib_values[missing_mask] = np.nan
    data['lib'] = lib_values

    print(f"Data columns: {list(data.columns)}")
    print(f"'lib' non-null: {data['lib'].notna().sum()} ({data['lib'].notna().mean()*100:.1f}%)")

    # Create strategy
    strategy = pbl.SingleSort(
        holding_period=1,
        sort_var='signal1',
        num_portfolios=5
    )

    sf = pbl.StrategyFormation(
        data=data,
        strategy=strategy,
        chars=['lib'],
        turnover=False,
        verbose=True
    )

    # Use column mapping like user does
    result = sf.fit(
        IDvar='cusip',
        RETvar='ret_vw_bgn',
        VWvar='mcap_e',
        RATINGvar='spc_rat'
    )

    # Check chars output
    chars_ew, chars_vw = result.get_characteristics()

    print("\n" + "-" * 40)
    print("Chars output with column mapping:")
    print("-" * 40)

    if chars_ew is None:
        print("ERROR: chars_ew is None!")
        return False

    lib_ew = chars_ew.get('lib')
    if lib_ew is None:
        print("ERROR: 'lib' not in chars_ew!")
        return False

    print(f"lib_ew shape: {lib_ew.shape}")
    print(f"lib_ew all NaN: {lib_ew.isnull().all().all()}")
    print(f"lib_ew non-NaN count: {lib_ew.notna().sum().sum()}")
    print(f"\nFirst 5 rows:\n{lib_ew.head()}")

    if lib_ew.isnull().all().all():
        print("\n*** ISSUE REPRODUCED: All chars are NaN! ***")
        return False
    else:
        print("\n*** OK: Chars have valid values ***")
        return True


def test_chars_compare_multiple():
    """Compare chars behavior with multiple characteristics."""
    print("\n" + "=" * 60)
    print("Test 4: Compare multiple chars (lib vs char1)")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    data = generate_synthetic_data(n_dates=30, n_bonds=200, seed=42)

    # Create 'lib' with ~16% missing
    lib_values = np.random.randn(len(data))
    missing_mask = np.random.random(len(data)) < 0.16
    lib_values[missing_mask] = np.nan
    data['lib'] = lib_values

    print(f"'lib' non-null: {data['lib'].notna().sum()}")
    print(f"'char1' non-null: {data['char1'].notna().sum()}")

    # Create strategy
    strategy = pbl.SingleSort(
        holding_period=1,
        sort_var='signal1',
        num_portfolios=5
    )

    # Test with both chars
    sf = pbl.StrategyFormation(
        data=data,
        strategy=strategy,
        chars=['lib', 'char1'],
        turnover=False,
        verbose=False
    )

    result = sf.fit()

    chars_ew, chars_vw = result.get_characteristics()

    print("\nChars results:")
    for name in ['lib', 'char1']:
        if name in chars_ew:
            df = chars_ew[name]
            all_nan = df.isnull().all().all()
            non_nan = df.notna().sum().sum()
            print(f"  {name}: all_nan={all_nan}, non_nan_count={non_nan}")
        else:
            print(f"  {name}: NOT FOUND")


if __name__ == '__main__':
    # Run all tests
    test_chars_with_synthetic_data()
    test_chars_debug_internals()
    test_chars_with_column_mapping()
    test_chars_compare_multiple()

    print("\n" + "=" * 60)
    print("All tests complete!")
    print("=" * 60)
