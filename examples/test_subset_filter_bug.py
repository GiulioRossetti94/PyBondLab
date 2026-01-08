"""
Test script for Bug #2: subset_filter + custom breakpoints interaction

This test verifies that:
1. Equal breakpoints (num_portfolios) with subset_filter works correctly (~0.4% missing)
2. Custom breakpoints with subset_filter NOW works correctly after the fix
3. Different filter configurations all produce similar missing rates

Before the fix:
- Mid maturity (5-10y) + custom BP [10,90]: ~30% missing
- Long maturity (10y+) + custom BP [10,90]: ~75% missing
- HY rating + custom BP [10,90]: ~35% missing

After the fix:
- All configurations should have ~0.4% missing (same as equal BP baseline)
"""
import sys
sys.path.insert(0, '/Users/u1972481/Dropbox/1-research/C-coding/developments/PBL/PUBLIC/Opt/PyBondLab-Dev')

import pandas as pd
import numpy as np
import PyBondLab as pbl
from PyBondLab.config import DataConfig, FormationConfig
import warnings
warnings.filterwarnings('ignore')

# Load data
DATA_PATH = '/Users/u1972481/Dropbox/5-data/corporate_bonds/bond_data_monthly/main_panel_20251126.parquet'


def prepare_data(df):
    """Prepare data for PyBondLab."""
    df = df.copy()
    df['ID'] = df['cusip']
    df['ret'] = df['ret_vw']
    df['VW'] = df['mcap_e']
    df['RATING_NUM'] = df['spc_rat']
    df['tmt'] = df['tmat']  # time to maturity
    df['date'] = pd.to_datetime(df['date'])

    # Filter to valid data
    df = df[df['ret'].notna()].copy()
    df = df[df['ytm'].notna()].copy()
    df = df[df['VW'].notna()].copy()
    df = df[df['RATING_NUM'].notna()].copy()
    df = df[df['tmt'].notna()].copy()

    return df


def test_config(df, name, subset_filter=None, rating=None, breakpoints=None, num_portfolios=3):
    """Run a single test configuration and return missing count."""
    config = pbl.StrategyFormationConfig(
        data=DataConfig(rating=rating),
        formation=FormationConfig(compute_turnover=False, dynamic_weights=True),
    )

    if breakpoints is not None:
        strategy = pbl.SingleSort(
            sort_var='ytm',
            holding_period=1,
            breakpoints=breakpoints,
            verbose=False,
        )
    else:
        strategy = pbl.SingleSort(
            sort_var='ytm',
            holding_period=1,
            num_portfolios=num_portfolios,
            verbose=False,
        )

    sf = pbl.StrategyFormation(
        data=df,
        strategy=strategy,
        config=config,
        subset_filter=subset_filter,
    )
    res = sf.fit()
    ew_ls, _ = res.get_long_short()

    n_dates = len(ew_ls)
    n_missing = ew_ls.isna().sum()
    pct_missing = 100 * n_missing / n_dates

    return n_dates, n_missing, pct_missing


def main():
    print("="*70)
    print("Bug #2 Test: subset_filter + custom breakpoints interaction")
    print("="*70)

    # Load and prepare data
    print("\nLoading data...")
    df = pd.read_parquet(DATA_PATH)
    df = prepare_data(df)
    print(f"Data: {len(df):,} rows, {df['ID'].nunique():,} bonds, {df['date'].nunique()} dates")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Define test configurations
    # subset_filter format: {col: (min_val, max_val)}
    tests = [
        # Maturity filters
        ("short (0-5y)", {'tmt': (0, 5)}, None),
        ("mid (5-10y)", {'tmt': (5, 10)}, None),
        ("long (10y+)", {'tmt': (10, 100)}, None),

        # Rating filters (HY = rating > 10)
        ("HY all", None, (11, 22)),
        ("HY + short", {'tmt': (0, 5)}, (11, 22)),
        ("HY + long", {'tmt': (10, 100)}, (11, 22)),
    ]

    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    print(f"{'Configuration':<20} {'Equal BP (n=3)':<20} {'Custom BP [10,90]':<20} {'Custom BP [20,80]':<20}")
    print("-"*70)

    results = []
    all_passed = True
    THRESHOLD = 5.0  # Max acceptable missing % (allow some tolerance)

    for name, subset_filter, rating in tests:
        # Equal breakpoints baseline
        n_dates, n_miss_eq, pct_miss_eq = test_config(
            df, name, subset_filter=subset_filter, rating=rating,
            num_portfolios=3
        )

        # Custom breakpoints [10, 90]
        _, n_miss_10_90, pct_miss_10_90 = test_config(
            df, name, subset_filter=subset_filter, rating=rating,
            breakpoints=[10, 90]
        )

        # Custom breakpoints [20, 80]
        _, n_miss_20_80, pct_miss_20_80 = test_config(
            df, name, subset_filter=subset_filter, rating=rating,
            breakpoints=[20, 80]
        )

        # Check if fix worked (custom BP missing should be near baseline)
        passed_10_90 = pct_miss_10_90 < THRESHOLD
        passed_20_80 = pct_miss_20_80 < THRESHOLD

        status_10_90 = "PASS" if passed_10_90 else "FAIL"
        status_20_80 = "PASS" if passed_20_80 else "FAIL"

        print(f"{name:<20} {n_miss_eq:>3}/{n_dates} ({pct_miss_eq:>5.1f}%)    "
              f"{n_miss_10_90:>3}/{n_dates} ({pct_miss_10_90:>5.1f}%) {status_10_90}  "
              f"{n_miss_20_80:>3}/{n_dates} ({pct_miss_20_80:>5.1f}%) {status_20_80}")

        results.append({
            'name': name,
            'n_dates': n_dates,
            'equal_bp_missing': n_miss_eq,
            'equal_bp_pct': pct_miss_eq,
            'custom_10_90_missing': n_miss_10_90,
            'custom_10_90_pct': pct_miss_10_90,
            'custom_20_80_missing': n_miss_20_80,
            'custom_20_80_pct': pct_miss_20_80,
            'passed': passed_10_90 and passed_20_80,
        })

        all_passed = all_passed and passed_10_90 and passed_20_80

    print("-"*70)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    n_passed = sum(1 for r in results if r['passed'])
    n_total = len(results)

    print(f"Tests passed: {n_passed}/{n_total}")
    print(f"Threshold: < {THRESHOLD}% missing dates")

    if all_passed:
        print("\nBUG #2 FIX VERIFIED: All configurations have acceptable missing rates")
        print("Custom breakpoints now work correctly with subset_filter and rating filters")
    else:
        print("\nBUG #2 NOT FIXED: Some configurations still have excessive missing dates")
        for r in results:
            if not r['passed']:
                print(f"  - {r['name']}: {r['custom_10_90_pct']:.1f}% / {r['custom_20_80_pct']:.1f}% missing")

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
