"""
Validation Script: WithinFirmSort Fast Path (Phase 16d)

This script validates that the ultra-fast numba path matches the slow pandas path
for WithinFirmSort strategy.

Author: Claude
Date: 2024
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import time
import PyBondLab as pbl


def generate_withinfirm_test_data(
    n_dates: int = 60,
    n_firms: int = 50,
    avg_bonds_per_firm: int = 4,
    seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic data suitable for WithinFirmSort testing."""
    np.random.seed(seed)

    dates = pd.date_range('2018-01-31', periods=n_dates, freq='ME')

    records = []
    cusip_counter = 0

    for firm_id in range(n_firms):
        n_bonds = max(2, int(np.random.exponential(avg_bonds_per_firm)))
        n_bonds = min(n_bonds, 10)  # Cap at 10 bonds per firm

        firm_base_rating = np.random.randint(1, 18)
        firm_base_yield = np.random.uniform(0.02, 0.08)

        for bond_idx in range(n_bonds):
            cusip = f'CUSIP{cusip_counter:06d}'
            cusip_counter += 1

            bond_rating_offset = np.random.randint(-2, 3)
            bond_rating = max(1, min(21, firm_base_rating + bond_rating_offset))
            bond_yield_offset = np.random.uniform(-0.02, 0.02)

            # Not all bonds exist for all dates (realistic attrition)
            start_date_idx = np.random.randint(0, max(1, n_dates - 30))
            end_date_idx = min(n_dates, start_date_idx + np.random.randint(12, 60))

            for date_idx in range(start_date_idx, end_date_idx):
                date = dates[date_idx]
                ytm = max(0.01, firm_base_yield + bond_yield_offset + np.random.uniform(-0.005, 0.005))
                ret = np.random.normal(0.005, 0.03)
                mcap = np.random.lognormal(15, 2)

                records.append({
                    'ID': cusip,
                    'PERMNO': firm_id,
                    'date': date,
                    'RATING_NUM': bond_rating,
                    'ytm': ytm,
                    'ret': ret,
                    'VW': mcap,
                })

    df = pd.DataFrame(records)
    df = df.sort_values(['ID', 'date']).reset_index(drop=True)
    return df


def run_slow_path(data, hp=1):
    """Run WithinFirmSort using the slow path (turnover=True forces slow path)."""
    strategy = pbl.WithinFirmSort(
        holding_period=hp,
        sort_var='ytm',
        firm_id_col='PERMNO',
        min_bonds_per_firm=2,
        rating_bins=[-np.inf, 7, 10, np.inf],
        num_portfolios=2,
        verbose=False
    )

    # Use turnover=True to force slow path
    result = pbl.StrategyFormation(
        data=data.copy(),
        strategy=strategy,
        turnover=True,  # Forces slow path
        verbose=False
    ).fit()

    return result


def run_fast_path(data, hp=1):
    """Run WithinFirmSort using the fast path (turnover=False)."""
    strategy = pbl.WithinFirmSort(
        holding_period=hp,
        sort_var='ytm',
        firm_id_col='PERMNO',
        min_bonds_per_firm=2,
        rating_bins=[-np.inf, 7, 10, np.inf],
        num_portfolios=2,
        verbose=False
    )

    # Use turnover=False to enable fast path
    result = pbl.StrategyFormation(
        data=data.copy(),
        strategy=strategy,
        turnover=False,  # Enables fast path
        verbose=False
    ).fit()

    return result


def compare_results(slow_result, fast_result, tolerance=1e-10):
    """Compare fast path results to slow path."""
    slow_ew, slow_vw = slow_result.get_long_short()
    fast_ew, fast_vw = fast_result.get_long_short()

    # Align indices
    common_idx = slow_ew.index.intersection(fast_ew.index)

    slow_ew_aligned = slow_ew.reindex(common_idx)
    fast_ew_aligned = fast_ew.reindex(common_idx)
    slow_vw_aligned = slow_vw.reindex(common_idx)
    fast_vw_aligned = fast_vw.reindex(common_idx)

    # Compare EW
    ew_diff = np.abs(slow_ew_aligned.values - fast_ew_aligned.values)
    ew_max_diff = np.nanmax(ew_diff) if len(ew_diff) > 0 else 0

    # Compare VW
    vw_diff = np.abs(slow_vw_aligned.values - fast_vw_aligned.values)
    vw_max_diff = np.nanmax(vw_diff) if len(vw_diff) > 0 else 0

    return {
        'ew_max_diff': ew_max_diff,
        'vw_max_diff': vw_max_diff,
        'ew_mean_slow': slow_ew_aligned.mean(),
        'ew_mean_fast': fast_ew_aligned.mean(),
        'vw_mean_slow': slow_vw_aligned.mean(),
        'vw_mean_fast': fast_vw_aligned.mean(),
        'n_common_dates': len(common_idx),
        'passed': ew_max_diff < tolerance and vw_max_diff < tolerance
    }


def test_hp1_basic(data):
    """Test HP=1 basic case."""
    print("\n" + "="*70)
    print("Test 1: HP=1 Basic Case")
    print("="*70)

    # Warm up JIT
    print("Warming up JIT...")
    small_data = data[data['date'] <= data['date'].unique()[5]].copy()
    _ = run_fast_path(small_data, hp=1)

    # Run slow path
    print("\nRunning slow path (turnover=True)...")
    t0 = time.time()
    slow_result = run_slow_path(data, hp=1)
    slow_time = time.time() - t0

    # Run fast path
    print("Running fast path (turnover=False)...")
    t0 = time.time()
    fast_result = run_fast_path(data, hp=1)
    fast_time = time.time() - t0

    # Compare
    comparison = compare_results(slow_result, fast_result)

    print(f"\nResults:")
    print(f"  Slow path time: {slow_time:.3f}s")
    print(f"  Fast path time: {fast_time:.3f}s")
    print(f"  Speedup: {slow_time/fast_time:.1f}x")
    print(f"\n  EW max diff: {comparison['ew_max_diff']:.2e}")
    print(f"  VW max diff: {comparison['vw_max_diff']:.2e}")
    print(f"  EW mean (slow): {comparison['ew_mean_slow']*100:.4f}%")
    print(f"  EW mean (fast): {comparison['ew_mean_fast']*100:.4f}%")
    print(f"  VW mean (slow): {comparison['vw_mean_slow']*100:.4f}%")
    print(f"  VW mean (fast): {comparison['vw_mean_fast']*100:.4f}%")
    print(f"  Common dates: {comparison['n_common_dates']}")
    print(f"\n  RESULT: {'PASS' if comparison['passed'] else 'FAIL'}")

    return comparison


def test_hp3_staggered(data):
    """Test HP=3 staggered case."""
    print("\n" + "="*70)
    print("Test 2: HP=3 Staggered Case")
    print("="*70)

    # Run slow path
    print("\nRunning slow path (turnover=True)...")
    t0 = time.time()
    slow_result = run_slow_path(data, hp=3)
    slow_time = time.time() - t0

    # Run fast path
    print("Running fast path (turnover=False)...")
    t0 = time.time()
    fast_result = run_fast_path(data, hp=3)
    fast_time = time.time() - t0

    # Compare
    comparison = compare_results(slow_result, fast_result)

    print(f"\nResults:")
    print(f"  Slow path time: {slow_time:.3f}s")
    print(f"  Fast path time: {fast_time:.3f}s")
    print(f"  Speedup: {slow_time/fast_time:.1f}x")
    print(f"\n  EW max diff: {comparison['ew_max_diff']:.2e}")
    print(f"  VW max diff: {comparison['vw_max_diff']:.2e}")
    print(f"  EW mean (slow): {comparison['ew_mean_slow']*100:.4f}%")
    print(f"  EW mean (fast): {comparison['ew_mean_fast']*100:.4f}%")
    print(f"  VW mean (slow): {comparison['vw_mean_slow']*100:.4f}%")
    print(f"  VW mean (fast): {comparison['vw_mean_fast']*100:.4f}%")
    print(f"  Common dates: {comparison['n_common_dates']}")
    print(f"\n  RESULT: {'PASS' if comparison['passed'] else 'FAIL'}")

    return comparison


def test_ew_vs_vw_different(data):
    """Test that EW and VW are different (bug regression test)."""
    print("\n" + "="*70)
    print("Test 3: EW vs VW Different (Regression Test)")
    print("="*70)

    fast_result = run_fast_path(data, hp=1)
    ew, vw = fast_result.get_long_short()

    # They should NOT be identical
    diff = np.abs(ew.values - vw.values)
    max_diff = np.nanmax(diff)
    mean_diff = np.nanmean(diff)

    are_different = mean_diff > 1e-6  # EW and VW should have meaningful difference

    print(f"\nResults:")
    print(f"  EW mean: {ew.mean()*100:.4f}%")
    print(f"  VW mean: {vw.mean()*100:.4f}%")
    print(f"  Max |EW - VW|: {max_diff*100:.4f}%")
    print(f"  Mean |EW - VW|: {mean_diff*100:.4f}%")
    print(f"\n  RESULT: {'PASS (EW != VW)' if are_different else 'FAIL (EW = VW, bug!)'}")

    return are_different


def test_large_data(n_firms=500, n_dates=100):
    """Test with larger data for performance benchmarking."""
    print("\n" + "="*70)
    print(f"Test 4: Large Data Performance ({n_firms} firms, {n_dates} dates)")
    print("="*70)

    print("\nGenerating large test data...")
    data = generate_withinfirm_test_data(
        n_dates=n_dates,
        n_firms=n_firms,
        avg_bonds_per_firm=6,
        seed=42
    )
    print(f"Data shape: {data.shape}")
    print(f"Unique cusips: {data['ID'].nunique()}")
    print(f"Unique firms: {data['PERMNO'].nunique()}")

    # Warm up JIT
    print("\nWarming up JIT...")
    small_data = data[data['date'] <= data['date'].unique()[5]].copy()
    _ = run_fast_path(small_data, hp=1)

    # Run slow path
    print("\nRunning slow path...")
    t0 = time.time()
    slow_result = run_slow_path(data, hp=1)
    slow_time = time.time() - t0

    # Run fast path
    print("Running fast path...")
    t0 = time.time()
    fast_result = run_fast_path(data, hp=1)
    fast_time = time.time() - t0

    # Compare
    comparison = compare_results(slow_result, fast_result)

    print(f"\nResults:")
    print(f"  Slow path time: {slow_time:.3f}s")
    print(f"  Fast path time: {fast_time:.3f}s")
    print(f"  Speedup: {slow_time/fast_time:.1f}x")
    print(f"\n  EW max diff: {comparison['ew_max_diff']:.2e}")
    print(f"  VW max diff: {comparison['vw_max_diff']:.2e}")
    print(f"\n  RESULT: {'PASS' if comparison['passed'] else 'FAIL'}")

    return comparison, slow_time, fast_time


def main():
    print("="*70)
    print("WithinFirmSort Fast Path Validation (Phase 16d)")
    print("="*70)

    # Generate test data
    print("\nGenerating test data...")
    data = generate_withinfirm_test_data(
        n_dates=60,
        n_firms=50,
        avg_bonds_per_firm=4,
        seed=42
    )
    print(f"Data shape: {data.shape}")
    print(f"Unique cusips: {data['ID'].nunique()}")
    print(f"Unique firms: {data['PERMNO'].nunique()}")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")

    # Run tests
    results = {}

    results['hp1'] = test_hp1_basic(data)
    results['hp3'] = test_hp3_staggered(data)
    results['ew_vw_different'] = test_ew_vs_vw_different(data)

    # Large data test
    comparison, slow_time, fast_time = test_large_data(n_firms=500, n_dates=100)
    results['large_data'] = comparison
    results['large_data_speedup'] = slow_time / fast_time

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    all_passed = (
        results['hp1']['passed'] and
        results['hp3']['passed'] and
        results['ew_vw_different'] and
        results['large_data']['passed']
    )

    print(f"\nTest 1 (HP=1 basic):        {'PASS' if results['hp1']['passed'] else 'FAIL'}")
    print(f"Test 2 (HP=3 staggered):    {'PASS' if results['hp3']['passed'] else 'FAIL'}")
    print(f"Test 3 (EW != VW):          {'PASS' if results['ew_vw_different'] else 'FAIL'}")
    print(f"Test 4 (Large data):        {'PASS' if results['large_data']['passed'] else 'FAIL'}")
    print(f"        Speedup:            {results['large_data_speedup']:.1f}x")
    print(f"\nOVERALL: {'ALL PASS' if all_passed else 'SOME FAILED'}")

    return results


if __name__ == '__main__':
    results = main()
