"""
Validation Script: WithinFirmSort Strategy

This script validates that WithinFirmSort works correctly with both
turnover=True and turnover=False (after the fix to exclude it from fast path).

It also establishes baseline timing for Phase 16 optimization.

Author: Claude
Date: 2024
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import time
import PyBondLab as pbl

# =============================================================================
# Generate Synthetic Test Data
# =============================================================================

def generate_withinfirm_test_data(
    n_dates: int = 60,
    n_firms: int = 50,
    bonds_per_firm_range: tuple = (2, 6),
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic data suitable for WithinFirmSort testing.

    Creates a panel with:
    - Multiple firms (PERMNO)
    - Multiple bonds per firm
    - Rating variation
    - Signal variation within firms
    """
    np.random.seed(seed)

    dates = pd.date_range('2018-01-31', periods=n_dates, freq='ME')

    records = []
    cusip_counter = 0

    for firm_id in range(n_firms):
        # Each firm has a random number of bonds
        n_bonds = np.random.randint(bonds_per_firm_range[0], bonds_per_firm_range[1] + 1)

        # Firm-level characteristics
        firm_base_rating = np.random.randint(1, 18)  # Base rating for the firm
        firm_base_spread = np.random.uniform(0.02, 0.08)  # Base credit spread

        for bond_idx in range(n_bonds):
            cusip = f'CUSIP{cusip_counter:06d}'
            cusip_counter += 1

            # Bond-level characteristics (vary around firm base)
            bond_rating_offset = np.random.randint(-2, 3)  # Rating can vary ±2 from firm base
            bond_rating = max(1, min(21, firm_base_rating + bond_rating_offset))

            bond_spread_offset = np.random.uniform(-0.02, 0.02)  # Spread varies within firm

            for date in dates:
                # Time-varying characteristics
                spread = firm_base_spread + bond_spread_offset + np.random.uniform(-0.005, 0.005)
                ret = np.random.normal(0.005, 0.03)  # Monthly return
                vw = np.random.lognormal(10, 1)  # Market value
                price = 100 * np.random.uniform(0.85, 1.15)

                records.append({
                    'ID': cusip,
                    'PERMNO': firm_id,
                    'date': date,
                    'RATING_NUM': bond_rating,
                    'CS': max(0.001, spread),  # Credit spread (signal)
                    'ret': ret,
                    'VW': vw,
                    'PRICE': price,
                })

    df = pd.DataFrame(records)
    df = df.sort_values(['ID', 'date']).reset_index(drop=True)

    return df


# =============================================================================
# Test Functions
# =============================================================================

def test_withinfirmsort_basic(data: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Basic test: Run WithinFirmSort and verify it completes.
    """
    if verbose:
        print("\n" + "="*70)
        print("Test 1: Basic WithinFirmSort Execution")
        print("="*70)

    strategy = pbl.WithinFirmSort(
        holding_period=1,
        sort_var='CS',
        firm_id_col='PERMNO',
        min_bonds_per_firm=2,
        rating_bins=[-np.inf, 7, 10, np.inf],
        num_portfolios=2,
        verbose=False
    )

    t0 = time.time()
    result = pbl.StrategyFormation(
        data=data.copy(),
        strategy=strategy,
        turnover=False,
        verbose=verbose
    ).fit()
    elapsed = time.time() - t0

    ew_ls, vw_ls = result.get_long_short()

    if verbose:
        print(f"\nExecution time: {elapsed:.2f}s")
        print(f"EW L-S mean: {ew_ls.mean()*100:.4f}%")
        print(f"VW L-S mean: {vw_ls.mean()*100:.4f}%")
        print(f"Non-NaN observations: {(~np.isnan(ew_ls)).sum()}")

    return {
        'ew_ls': ew_ls,
        'vw_ls': vw_ls,
        'time': elapsed,
        'success': True
    }


def test_withinfirmsort_turnover_comparison(data: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Test: Compare turnover=True vs turnover=False.
    After the fix, both should produce identical returns.
    """
    if verbose:
        print("\n" + "="*70)
        print("Test 2: turnover=True vs turnover=False Comparison")
        print("="*70)

    strategy = pbl.WithinFirmSort(
        holding_period=1,
        sort_var='CS',
        firm_id_col='PERMNO',
        min_bonds_per_firm=2,
        rating_bins=[-np.inf, 7, 10, np.inf],
        num_portfolios=2,
        verbose=False
    )

    # Run with turnover=True
    t0 = time.time()
    result_with_turnover = pbl.StrategyFormation(
        data=data.copy(),
        strategy=strategy,
        turnover=True,
        verbose=False
    ).fit()
    time_with = time.time() - t0

    # Run with turnover=False
    t0 = time.time()
    result_without_turnover = pbl.StrategyFormation(
        data=data.copy(),
        strategy=strategy,
        turnover=False,
        verbose=False
    ).fit()
    time_without = time.time() - t0

    # Compare returns
    ew_with, vw_with = result_with_turnover.get_long_short()
    ew_without, vw_without = result_without_turnover.get_long_short()

    # Compute differences
    ew_diff = np.abs(np.array(ew_with) - np.array(ew_without))
    vw_diff = np.abs(np.array(vw_with) - np.array(vw_without))

    max_ew_diff = np.nanmax(ew_diff)
    max_vw_diff = np.nanmax(vw_diff)

    passed = max_ew_diff < 1e-10 and max_vw_diff < 1e-10

    if verbose:
        print(f"\nWith turnover=True:    {time_with:.2f}s, EW mean={ew_with.mean()*100:.4f}%")
        print(f"With turnover=False:   {time_without:.2f}s, EW mean={ew_without.mean()*100:.4f}%")
        print(f"\nMax EW difference: {max_ew_diff:.2e}")
        print(f"Max VW difference: {max_vw_diff:.2e}")
        print(f"\nResult: {'PASS' if passed else 'FAIL'}")

    return {
        'ew_with': ew_with,
        'vw_with': vw_with,
        'ew_without': ew_without,
        'vw_without': vw_without,
        'time_with': time_with,
        'time_without': time_without,
        'max_ew_diff': max_ew_diff,
        'max_vw_diff': max_vw_diff,
        'passed': passed
    }


def test_withinfirmsort_hp_gt1_disabled(data: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Test: Verify that WithinFirmSort with HP>1 raises an error.
    HP>1 is currently disabled due to known bugs in cohort averaging.
    """
    if verbose:
        print("\n" + "="*70)
        print("Test 3: Verify HP>1 is Disabled (should raise ValueError)")
        print("="*70)

    try:
        strategy = pbl.WithinFirmSort(
            holding_period=3,
            sort_var='CS',
            firm_id_col='PERMNO',
            min_bonds_per_firm=2,
            rating_bins=[-np.inf, 7, 10, np.inf],
            num_portfolios=2,
            verbose=False
        )
        passed = False  # Should have raised an error
        error_msg = "No error raised"
    except ValueError as e:
        passed = True
        error_msg = str(e)

    if verbose:
        print(f"\nResult: {'PASS' if passed else 'FAIL'}")
        if passed:
            print(f"ValueError raised as expected: {error_msg[:80]}...")
        else:
            print("ERROR: HP>1 should raise ValueError but didn't!")

    return {
        'passed': passed,
        'error_msg': error_msg if passed else None
    }


def test_withinfirmsort_vs_singlesort(data: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Test: Verify WithinFirmSort produces different results than SingleSort.
    This confirms the within-firm logic is actually being applied.
    """
    if verbose:
        print("\n" + "="*70)
        print("Test 4: WithinFirmSort vs SingleSort (should differ!)")
        print("="*70)

    # WithinFirmSort
    wfs_strategy = pbl.WithinFirmSort(
        holding_period=1,
        sort_var='CS',
        firm_id_col='PERMNO',
        min_bonds_per_firm=2,
        rating_bins=[-np.inf, 7, 10, np.inf],
        num_portfolios=2,
        verbose=False
    )

    result_wfs = pbl.StrategyFormation(
        data=data.copy(),
        strategy=wfs_strategy,
        turnover=False,
        verbose=False
    ).fit()

    # SingleSort (standard cross-sectional)
    ss_strategy = pbl.SingleSort(
        holding_period=1,
        sort_var='CS',
        num_portfolios=2,
        verbose=False
    )

    result_ss = pbl.StrategyFormation(
        data=data.copy(),
        strategy=ss_strategy,
        turnover=False,
        verbose=False
    ).fit()

    ew_wfs, vw_wfs = result_wfs.get_long_short()
    ew_ss, vw_ss = result_ss.get_long_short()

    # Correlation between the two approaches
    valid_mask = ~np.isnan(ew_wfs) & ~np.isnan(ew_ss)
    if valid_mask.sum() > 10:
        corr = np.corrcoef(np.array(ew_wfs)[valid_mask], np.array(ew_ss)[valid_mask])[0, 1]
    else:
        corr = np.nan

    # They should be different (not identical)
    diff = np.abs(np.array(ew_wfs) - np.array(ew_ss))
    max_diff = np.nanmax(diff)
    are_different = max_diff > 0.001  # Should have meaningful difference

    if verbose:
        print(f"\nWithinFirmSort EW mean: {ew_wfs.mean()*100:.4f}%")
        print(f"SingleSort EW mean:     {ew_ss.mean()*100:.4f}%")
        print(f"\nCorrelation: {corr:.4f}")
        print(f"Max absolute difference: {max_diff:.4f}")
        print(f"\nResult: {'PASS (different as expected)' if are_different else 'FAIL (too similar)'}")

    return {
        'ew_wfs': ew_wfs,
        'ew_ss': ew_ss,
        'correlation': corr,
        'max_diff': max_diff,
        'are_different': are_different
    }


def run_baseline_timing(data: pd.DataFrame, n_runs: int = 3, verbose: bool = True) -> dict:
    """
    Establish baseline timing for Phase 16 optimization.
    Note: Only HP=1 is tested since HP>1 is currently disabled.
    """
    if verbose:
        print("\n" + "="*70)
        print("Baseline Timing for Phase 16 Optimization")
        print("="*70)

    timings = {
        'hp1_no_turnover': [],
        'hp1_with_turnover': [],
    }

    # Only test HP=1 (HP>1 is disabled due to known bugs)
    for turnover in [False, True]:
        key = f"hp1_{'with' if turnover else 'no'}_turnover"

        strategy = pbl.WithinFirmSort(
            holding_period=1,
            sort_var='CS',
            firm_id_col='PERMNO',
            min_bonds_per_firm=2,
            rating_bins=[-np.inf, 7, 10, np.inf],
            num_portfolios=2,
            verbose=False
        )

        for run in range(n_runs):
            t0 = time.time()
            result = pbl.StrategyFormation(
                data=data.copy(),
                strategy=strategy,
                turnover=turnover,
                verbose=False
            ).fit()
            elapsed = time.time() - t0
            timings[key].append(elapsed)

    if verbose:
        print(f"\nData size: {len(data)} rows, {data['ID'].nunique()} bonds, "
              f"{data['PERMNO'].nunique()} firms, {data['date'].nunique()} dates")
        print(f"\nTiming Results (avg of {n_runs} runs):")
        print("-" * 50)
        for key, times in timings.items():
            avg_time = np.mean(times)
            print(f"  {key}: {avg_time:.3f}s")

    return timings


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*70)
    print("WithinFirmSort Validation Suite")
    print("="*70)

    # Generate test data
    print("\nGenerating test data...")
    data = generate_withinfirm_test_data(
        n_dates=60,
        n_firms=50,
        bonds_per_firm_range=(2, 6),
        seed=42
    )
    print(f"Data shape: {data.shape}")
    print(f"Unique bonds: {data['ID'].nunique()}")
    print(f"Unique firms: {data['PERMNO'].nunique()}")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")

    # Run tests
    results = {}

    # Test 1: Basic execution
    results['basic'] = test_withinfirmsort_basic(data)

    # Test 2: turnover comparison (the key fix validation)
    results['turnover_comparison'] = test_withinfirmsort_turnover_comparison(data)

    # Test 3: Verify HP>1 is disabled
    results['hp_gt1_disabled'] = test_withinfirmsort_hp_gt1_disabled(data)

    # Test 4: vs SingleSort
    results['vs_singlesort'] = test_withinfirmsort_vs_singlesort(data)

    # Baseline timing
    timings = run_baseline_timing(data, n_runs=3)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    all_passed = True

    test_names = [
        ('basic', 'Basic Execution', lambda r: r['success']),
        ('turnover_comparison', 'Turnover=True vs False', lambda r: r['passed']),
        ('hp_gt1_disabled', 'HP>1 Disabled Check', lambda r: r['passed']),
        ('vs_singlesort', 'vs SingleSort Different', lambda r: r['are_different']),
    ]

    for key, name, check_fn in test_names:
        passed = check_fn(results[key])
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return results, timings


if __name__ == '__main__':
    results, timings = main()
