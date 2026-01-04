"""
Audit: Compare SingleSort vs DataUncertaintyAnalysis baseline results.

This script verifies that SingleSort with 10 portfolios produces identical
long-short factor means as DataUncertaintyAnalysis with baseline filter.

Tests HP = 1, 3, 6

IMPORTANT: Both must use the same `dynamic_weights` setting!
- StrategyFormation default: dynamic_weights=False (from config)
- DataUncertaintyAnalysis default: dynamic_weights=True
This test explicitly sets dynamic_weights=True for both to match.
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/user/PyBondLab-Dev')

import PyBondLab as pbl
from PyBondLab import StrategyFormation, SingleSort, DataUncertaintyAnalysis


def generate_test_data(n_dates=60, n_bonds=200, seed=42):
    """Generate synthetic test data."""
    np.random.seed(seed)

    dates = pd.date_range('2018-01-31', periods=n_dates, freq='ME')
    bond_ids = [f'BOND_{i:03d}' for i in range(n_bonds)]

    rows = []
    for d in dates:
        for b in bond_ids:
            rows.append({
                'date': d,
                'ID': b,
                'ret': np.random.normal(0.005, 0.03),
                'VW': np.random.uniform(100, 1000),
                'RATING_NUM': np.random.randint(1, 23),
                'signal': np.random.uniform(-1, 1),
            })

    data = pd.DataFrame(rows)
    return data


def run_singlesort(data, signal, hp, num_portfolios=10, dynamic_weights=True):
    """Run SingleSort and return EW/VW long-short means."""
    from PyBondLab.config import StrategyFormationConfig, FormationConfig, DataConfig

    strategy = SingleSort(
        holding_period=hp,
        sort_var=signal,
        num_portfolios=num_portfolios
    )

    # Explicitly set dynamic_weights to match DUA
    config = StrategyFormationConfig(
        data=DataConfig(),
        formation=FormationConfig(dynamic_weights=dynamic_weights)
    )

    sf = StrategyFormation(
        data=data,
        strategy=strategy,
        turnover=False,
        verbose=False,
        config=config
    )

    result = sf.fit()
    ew_ls, vw_ls = result.get_long_short()

    return {
        'ew_mean': ew_ls.mean(),
        'vw_mean': vw_ls.mean(),
        'ew_series': ew_ls,
        'vw_series': vw_ls,
    }


def run_dua(data, signal, hp, num_portfolios=10):
    """Run DataUncertaintyAnalysis baseline and return EW/VW long-short means."""
    results = DataUncertaintyAnalysis(
        data=data,
        signals=[signal],
        holding_periods=[hp],
        filters={},  # No filters
        include_baseline=True,
        num_portfolios=num_portfolios,
        verbose=False,
    ).fit()

    # Get baseline results
    baseline = results.filter(filter_type='baseline')

    ew_ls = baseline.ew_ex_ante.iloc[:, 0]  # First (and only) column
    vw_ls = baseline.vw_ex_ante.iloc[:, 0]

    return {
        'ew_mean': ew_ls.mean(),
        'vw_mean': vw_ls.mean(),
        'ew_series': ew_ls,
        'vw_series': vw_ls,
    }


def compare_results(ss_result, dua_result, hp, tolerance=1e-10):
    """Compare SingleSort and DUA results."""
    ew_diff = abs(ss_result['ew_mean'] - dua_result['ew_mean'])
    vw_diff = abs(ss_result['vw_mean'] - dua_result['vw_mean'])

    # Also compare full series
    ew_series_diff = (ss_result['ew_series'] - dua_result['ew_series']).abs().max()
    vw_series_diff = (ss_result['vw_series'] - dua_result['vw_series']).abs().max()

    ew_pass = ew_diff < tolerance and ew_series_diff < tolerance
    vw_pass = vw_diff < tolerance and vw_series_diff < tolerance

    return {
        'hp': hp,
        'ew_ss_mean': ss_result['ew_mean'],
        'ew_dua_mean': dua_result['ew_mean'],
        'ew_diff': ew_diff,
        'ew_series_max_diff': ew_series_diff,
        'ew_pass': ew_pass,
        'vw_ss_mean': ss_result['vw_mean'],
        'vw_dua_mean': dua_result['vw_mean'],
        'vw_diff': vw_diff,
        'vw_series_max_diff': vw_series_diff,
        'vw_pass': vw_pass,
    }


def main():
    print("="*70)
    print("AUDIT: SingleSort vs DataUncertaintyAnalysis (Baseline)")
    print("="*70)

    # Generate test data
    print("\nGenerating test data...")
    data = generate_test_data(n_dates=60, n_bonds=200, seed=42)
    print(f"  Data shape: {data.shape}")
    print(f"  Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"  Bonds: {data['ID'].nunique()}")

    signal = 'signal'
    num_portfolios = 10
    holding_periods = [1, 3, 6]

    print(f"\nTest configuration:")
    print(f"  Signal: {signal}")
    print(f"  Num portfolios: {num_portfolios}")
    print(f"  Holding periods: {holding_periods}")

    results = []

    for hp in holding_periods:
        print(f"\n{'='*70}")
        print(f"Testing HP = {hp}")
        print("="*70)

        # Run SingleSort
        print("  Running SingleSort...", end=" ")
        ss_result = run_singlesort(data, signal, hp, num_portfolios)
        print(f"EW mean: {ss_result['ew_mean']:.6f}, VW mean: {ss_result['vw_mean']:.6f}")

        # Run DataUncertaintyAnalysis
        print("  Running DataUncertaintyAnalysis...", end=" ")
        dua_result = run_dua(data, signal, hp, num_portfolios)
        print(f"EW mean: {dua_result['ew_mean']:.6f}, VW mean: {dua_result['vw_mean']:.6f}")

        # Compare
        comparison = compare_results(ss_result, dua_result, hp)
        results.append(comparison)

        print(f"\n  Comparison:")
        print(f"    EW mean diff: {comparison['ew_diff']:.2e} | Series max diff: {comparison['ew_series_max_diff']:.2e} | {'PASS' if comparison['ew_pass'] else 'FAIL'}")
        print(f"    VW mean diff: {comparison['vw_diff']:.2e} | Series max diff: {comparison['vw_series_max_diff']:.2e} | {'PASS' if comparison['vw_pass'] else 'FAIL'}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    all_pass = True
    for r in results:
        status = "PASS" if (r['ew_pass'] and r['vw_pass']) else "FAIL"
        if not (r['ew_pass'] and r['vw_pass']):
            all_pass = False
        print(f"  HP={r['hp']}: EW diff={r['ew_diff']:.2e}, VW diff={r['vw_diff']:.2e} [{status}]")

    print("\n" + "="*70)
    if all_pass:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("="*70)

    return all_pass


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
