#!/usr/bin/env python
"""
Test script to replicate column mapping corner-case errors.

Issue 1: chars using mapped column names
  - User specifies chars=['spc_rat'] and RATINGvar='spc_rat'
  - After mapping, spc_rat is renamed to RATING_NUM
  - chars still looks for spc_rat which no longer exists

Issue 2: Column already named with target name
  - User already has an 'ID' column
  - They specify IDvar='ID' (which is a no-op)
  - But other column mappings fail if those columns don't exist
"""

import sys
import pandas as pd
import numpy as np

sys.path.insert(0, '.')
import PyBondLab as pbl


def generate_test_data_with_custom_names():
    """Generate synthetic data with custom column names."""
    np.random.seed(42)
    n_dates = 30
    n_bonds = 100

    dates = pd.date_range('2020-01-31', periods=n_dates, freq='M')

    rows = []
    for date in dates:
        for bond_id in range(n_bonds):
            rows.append({
                'date': date,
                'cusip_id': f'BOND_{bond_id:04d}',  # Custom ID column name
                'ret_vw_bgn': np.random.randn() * 0.02,  # Custom return column name
                'mcap_e': np.random.uniform(100, 1000),  # Custom VW column name
                'spc_rat': np.random.randint(1, 23),  # Custom rating column name
                'cs': np.random.randn(),  # Signal column
                'duration': np.random.uniform(1, 10),  # Char column
            })

    data = pd.DataFrame(rows)
    return data


def test_issue1_chars_with_mapped_column():
    """
    Issue 1: chars referencing a column that gets renamed.

    User specifies:
      - RATINGvar='spc_rat' -> renames spc_rat to RATING_NUM
      - chars=['spc_rat'] -> tries to find spc_rat, but it's now RATING_NUM
    """
    print("\n" + "=" * 60)
    print("TEST: Issue 1 - chars using mapped column name")
    print("=" * 60)

    data = generate_test_data_with_custom_names()
    print(f"  Data columns: {list(data.columns)}")

    # Define strategy
    single_sort = pbl.SingleSort(
        holding_period=1,
        sort_var='cs',
        num_portfolios=5,
    )

    # This should fail because chars=['spc_rat'] but spc_rat gets renamed to RATING_NUM
    params = dict(
        strategy=single_sort,
        rating=None,
        turnover=True,
        chars=['spc_rat'],  # User wants average rating across sorted deciles
    )

    try:
        sf = pbl.StrategyFormation(data, **params)
        results = sf.fit(
            IDvar='cusip_id',
            RETvar='ret_vw_bgn',
            VWvar='mcap_e',
            RATINGvar='spc_rat',  # This renames spc_rat to RATING_NUM
        )
        print("  [UNEXPECTED] No error occurred!")
        return True
    except KeyError as e:
        print(f"  [EXPECTED ERROR] KeyError: {e}")
        print("  This is the expected error - chars looks for 'spc_rat' but it was renamed")
        return False
    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {e}")
        return False


def test_issue2_column_already_exists():
    """
    Issue 2: User already has standard column name but tries to map non-existent columns.

    User has:
      - 'ID' column (already correctly named)
      - 'RATING_NUM' column (already correctly named)
      - But they try to map other columns that don't exist
    """
    print("\n" + "=" * 60)
    print("TEST: Issue 2 - Column already exists + mapping non-existent")
    print("=" * 60)

    # Create data with some columns already named correctly
    data = generate_test_data_with_custom_names()

    # Rename some columns to standard names (simulating user already did this)
    data = data.rename(columns={
        'cusip_id': 'ID',
        'spc_rat': 'RATING_NUM',
    })
    print(f"  Data columns: {list(data.columns)}")

    # Define strategy
    single_sort = pbl.SingleSort(
        holding_period=1,
        sort_var='cs',
        num_portfolios=5,
    )

    # User tries to map columns that don't exist in their data
    # (they already renamed some manually, but try to use old names in fit())
    try:
        sf = pbl.StrategyFormation(data, strategy=single_sort, turnover=True)
        results = sf.fit(
            IDvar='ID',  # Correct - exists
            RETvar='ret_vw_bgn',  # Correct - exists
            VWvar='mcap_e',  # Correct - exists
            RATINGvar='spc_rat',  # WRONG - spc_rat doesn't exist, they renamed it to RATING_NUM
        )
        print("  [UNEXPECTED] No error occurred!")
        return True
    except ValueError as e:
        print(f"  [EXPECTED ERROR] ValueError: {e}")
        return False
    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {e}")
        return False


def test_issue2b_skip_mapping_for_existing():
    """
    Issue 2b: User specifies IDvar='ID' when data already has 'ID' column.

    This should work (mapping is skipped for already-correct names).
    """
    print("\n" + "=" * 60)
    print("TEST: Issue 2b - Skip mapping for already-correct names")
    print("=" * 60)

    data = generate_test_data_with_custom_names()

    # Rename ID column to standard name
    data = data.rename(columns={'cusip_id': 'ID'})
    print(f"  Data columns: {list(data.columns)}")

    # Define strategy
    single_sort = pbl.SingleSort(
        holding_period=1,
        sort_var='cs',
        num_portfolios=5,
    )

    # User specifies IDvar='ID' - this should work (mapping skipped)
    try:
        sf = pbl.StrategyFormation(data, strategy=single_sort, turnover=False)
        results = sf.fit(
            IDvar='ID',  # Already exists, mapping skipped
            RETvar='ret_vw_bgn',
            VWvar='mcap_e',
            RATINGvar='spc_rat',
        )
        print("  [PASS] Mapping worked correctly")
        return True
    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {e}")
        return False


def test_solution_chars_should_use_mapped_names():
    """
    Proposed solution: chars should use mapped column names.

    If user specifies:
      - RATINGvar='spc_rat' -> renamed to RATING_NUM
      - chars=['spc_rat'] -> should automatically use 'RATING_NUM'

    This test shows what SHOULD work after the fix.
    """
    print("\n" + "=" * 60)
    print("TEST: Proposed Solution - chars should accept mapped names")
    print("=" * 60)

    data = generate_test_data_with_custom_names()

    # Define strategy
    single_sort = pbl.SingleSort(
        holding_period=1,
        sort_var='cs',
        num_portfolios=5,
    )

    # User specifies chars using the MAPPED name (RATING_NUM) - this should work
    try:
        sf = pbl.StrategyFormation(
            data,
            strategy=single_sort,
            turnover=True,
            chars=['duration'],  # Use a column that won't be renamed
        )
        results = sf.fit(
            IDvar='cusip_id',
            RETvar='ret_vw_bgn',
            VWvar='mcap_e',
            RATINGvar='spc_rat',
        )
        print("  [PASS] Using non-renamed char works correctly")
        ew_chars, vw_chars = results.get_characteristics()
        print(f"  Available chars: {list(ew_chars.keys())}")
        return True
    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {e}")
        return False


def main():
    print("=" * 60)
    print("Column Mapping Corner-Case Error Replication")
    print("=" * 60)

    # Test Issue 1: chars with mapped column
    issue1_failed = not test_issue1_chars_with_mapped_column()

    # Test Issue 2: Column already exists
    issue2_failed = not test_issue2_column_already_exists()

    # Test Issue 2b: Skip mapping for existing names
    issue2b_passed = test_issue2b_skip_mapping_for_existing()

    # Test proposed solution
    solution_works = test_solution_chars_should_use_mapped_names()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Issue 1 (chars with mapped column): {'REPLICATED' if issue1_failed else 'NOT REPLICATED'}")
    print(f"  Issue 2 (non-existent column mapping): {'REPLICATED' if issue2_failed else 'NOT REPLICATED'}")
    print(f"  Issue 2b (skip mapping for existing): {'WORKS' if issue2b_passed else 'BROKEN'}")
    print(f"  Solution test (non-renamed char): {'WORKS' if solution_works else 'BROKEN'}")

    return issue1_failed and issue2_failed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
