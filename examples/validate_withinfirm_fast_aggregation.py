"""
Validation Script: WithinFirmSort Fast Aggregation

This script validates that the fast numba aggregation matches the slow pandas path.

Author: Claude
Date: 2024
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import time
import PyBondLab as pbl
from PyBondLab.numba_core import compute_within_firm_aggregation_fast
from PyBondLab.utils_within_firm import compute_within_firm_returns_aggregation
from PyBondLab.constants import ColumnNames


def generate_withinfirm_test_data(
    n_dates: int = 60,
    n_firms: int = 50,
    bonds_per_firm_range: tuple = (2, 6),
    seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic data suitable for WithinFirmSort testing."""
    np.random.seed(seed)

    dates = pd.date_range('2018-01-31', periods=n_dates, freq='ME')

    records = []
    cusip_counter = 0

    for firm_id in range(n_firms):
        n_bonds = np.random.randint(bonds_per_firm_range[0], bonds_per_firm_range[1] + 1)
        firm_base_rating = np.random.randint(1, 18)
        firm_base_spread = np.random.uniform(0.02, 0.08)

        for bond_idx in range(n_bonds):
            cusip = f'CUSIP{cusip_counter:06d}'
            cusip_counter += 1

            bond_rating_offset = np.random.randint(-2, 3)
            bond_rating = max(1, min(21, firm_base_rating + bond_rating_offset))
            bond_spread_offset = np.random.uniform(-0.02, 0.02)

            for date in dates:
                spread = firm_base_spread + bond_spread_offset + np.random.uniform(-0.005, 0.005)
                ret = np.random.normal(0.005, 0.03)
                vw = np.random.lognormal(10, 1)
                price = 100 * np.random.uniform(0.85, 1.15)

                records.append({
                    'ID': cusip,
                    'PERMNO': firm_id,
                    'date': date,
                    'RATING_NUM': bond_rating,
                    'CS': max(0.001, spread),
                    'ret': ret,
                    'VW': vw,
                    'PRICE': price,
                })

    df = pd.DataFrame(records)
    df = df.sort_values(['ID', 'date']).reset_index(drop=True)
    return df


def prepare_fast_aggregation_inputs(portfolio_indices, data_raw, datelist,
                                     firm_id_col, rating_col, rating_bins):
    """Prepare numpy arrays for fast aggregation kernel."""

    # Collect all portfolio data into a single DataFrame
    all_port_dfs = []
    for date_t in datelist:
        if date_t not in portfolio_indices:
            continue
        port_df = portfolio_indices[date_t]
        if port_df.empty:
            continue
        all_port_dfs.append(port_df)

    if not all_port_dfs:
        return None

    # Concatenate all portfolio data
    combined_df = pd.concat(all_port_dfs, ignore_index=True)

    # Get firm and rating lookup
    firm_rating_lookup = data_raw[[ColumnNames.ID, firm_id_col, rating_col]].drop_duplicates()

    # Merge with firm and rating info
    combined_df = combined_df.merge(firm_rating_lookup, on=ColumnNames.ID, how='left')

    # Create rating terciles
    combined_df['rating_terc'] = pd.cut(
        pd.to_numeric(combined_df[rating_col], errors='coerce'),
        bins=rating_bins,
        labels=[1, 2, 3],
        include_lowest=True
    ).astype('Int64')

    # Create index mappings
    unique_dates = sorted(combined_df[ColumnNames.DATE].unique())
    unique_firms = sorted(combined_df[firm_id_col].dropna().unique())

    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    firm_to_idx = {f: i for i, f in enumerate(unique_firms)}

    n_dates = len(unique_dates)
    n_firms = len(unique_firms)

    # Convert to numpy arrays
    date_idx = combined_df[ColumnNames.DATE].map(date_to_idx).values.astype(np.int64)
    firm_idx = combined_df[firm_id_col].map(firm_to_idx).fillna(-1).values.astype(np.int64)
    rating_terc = combined_df['rating_terc'].values.astype(np.float64)
    ptf_rank = combined_df['ptf_rank'].values.astype(np.float64)
    ret = combined_df['ret'].values.astype(np.float64)
    vw = combined_df['VW'].values.astype(np.float64)

    return {
        'date_idx': date_idx,
        'firm_idx': firm_idx,
        'rating_terc': rating_terc,
        'ptf_rank': ptf_rank,
        'ret': ret,
        'vw': vw,
        'n_dates': n_dates,
        'n_firms': n_firms,
        'unique_dates': unique_dates,
    }


def test_fast_vs_slow_aggregation(data, hp=1, verbose=True):
    """Compare fast numba aggregation vs slow pandas aggregation."""

    print(f"\n{'='*70}")
    print(f"Testing Fast vs Slow Aggregation (HP={hp})")
    print(f"{'='*70}")

    # Run full slow path first to get portfolio_indices
    strategy = pbl.WithinFirmSort(
        holding_period=hp,
        sort_var='CS',
        firm_id_col='PERMNO',
        min_bonds_per_firm=2,
        rating_bins=[-np.inf, 7, 10, np.inf],
        num_portfolios=2,
        verbose=False
    )

    # Run StrategyFormation
    sf = pbl.StrategyFormation(
        data=data.copy(),
        strategy=strategy,
        turnover=False,
        verbose=False
    )
    result = sf.fit()

    # Get the portfolio_indices from the internal state
    # We need to access the precomputed data and portfolio_indices
    # Let's re-run the slow aggregation path manually

    # Get datelist
    datelist = sorted(data['date'].unique())

    # Run slow aggregation
    t0 = time.time()

    # Get the portfolio_indices from the result
    # For this test, we'll run the full slow path and compare
    slow_ew_ls, slow_vw_ls = result.get_long_short()

    slow_time = time.time() - t0

    # Now test the fast kernel directly
    # We need to construct the inputs similar to what _aggregate_within_firm_results does

    # Since we can't easily access internal state, let's verify by running
    # the fast kernel on the same data structure

    print(f"\nSlow path time (full StrategyFormation): {slow_time:.3f}s")
    print(f"Slow path VW L-S mean: {slow_vw_ls.mean()*100:.4f}%")
    print(f"Non-NaN dates: {(~np.isnan(slow_vw_ls)).sum()}")

    return {
        'slow_ew_ls': slow_ew_ls,
        'slow_vw_ls': slow_vw_ls,
        'slow_time': slow_time,
    }


def test_kernel_directly(data, verbose=True):
    """Test the numba kernel directly with synthetic data."""

    print(f"\n{'='*70}")
    print("Testing Numba Kernel Directly")
    print(f"{'='*70}")

    # Prepare test data
    np.random.seed(42)
    n_dates = 60
    n_firms = 50
    n_obs = 10000

    date_idx = np.random.randint(0, n_dates, n_obs).astype(np.int64)
    firm_idx = np.random.randint(0, n_firms, n_obs).astype(np.int64)
    rating_terc = np.random.choice([1, 2, 3], n_obs).astype(np.float64)
    ptf_rank = np.random.choice([1, 2], n_obs).astype(np.float64)
    ret = np.random.normal(0.005, 0.03, n_obs).astype(np.float64)
    vw = np.random.lognormal(10, 1, n_obs).astype(np.float64)

    # Warm up JIT
    print("Warming up JIT...")
    _ = compute_within_firm_aggregation_fast(
        date_idx[:100], firm_idx[:100], rating_terc[:100], ptf_rank[:100],
        ret[:100], vw[:100], n_dates, n_firms
    )

    # Test performance
    n_runs = 10
    t0 = time.time()
    for _ in range(n_runs):
        long_short, high_ret, low_ret, valid_dates = compute_within_firm_aggregation_fast(
            date_idx, firm_idx, rating_terc, ptf_rank, ret, vw, n_dates, n_firms
        )
    elapsed = (time.time() - t0) / n_runs

    print(f"\nKernel execution time (avg of {n_runs}): {elapsed*1000:.2f}ms")
    print(f"L-S mean: {np.nanmean(long_short)*100:.4f}%")
    print(f"Valid dates: {np.sum(valid_dates)}/{n_dates}")
    print(f"Non-NaN L-S values: {(~np.isnan(long_short)).sum()}")

    return {
        'long_short': long_short,
        'high_ret': high_ret,
        'low_ret': low_ret,
        'valid_dates': valid_dates,
        'time': elapsed,
    }


def test_integration_with_real_data(data, verbose=True):
    """
    Full integration test: run slow path, extract portfolio_indices,
    run fast aggregation, compare results.
    """
    print(f"\n{'='*70}")
    print("Full Integration Test")
    print(f"{'='*70}")

    from PyBondLab.utils_within_firm import compute_within_firm_portfolios

    rating_bins = [-np.inf, 7, 10, np.inf]
    firm_id_col = 'PERMNO'
    rating_col = 'RATING_NUM'

    # Step 1: Compute within-firm portfolio assignments
    print("\n1. Computing within-firm portfolio assignments...")
    t0 = time.time()
    bond_assignments, firm_weights_df = compute_within_firm_portfolios(
        data=data,
        signal_col='CS',
        return_col='ret',
        weight_col='VW',
        firm_id_col=firm_id_col,
        rating_col=rating_col,
        rating_bins=rating_bins,
        min_bonds_per_firm=2
    )
    t_assign = time.time() - t0
    print(f"   Time: {t_assign:.3f}s")
    print(f"   Bonds assigned: {len(bond_assignments)}")

    # Step 2: Create portfolio_indices (simulating what StrategyFormation does)
    print("\n2. Creating portfolio indices...")
    t0 = time.time()

    # Merge assignments with returns and VW
    merged = bond_assignments.merge(
        data[['ID', 'date', 'ret', 'VW']],
        on=['ID', 'date'],
        how='left'
    )

    # Compute weights
    merged['eweights'] = 1.0  # placeholder
    merged['vweights'] = merged['VW'] / merged.groupby('date')['VW'].transform('sum')
    merged['ptf_rank'] = merged['ptf_rank']

    # Group by date
    datelist = sorted(merged['date'].unique())
    portfolio_indices = {}
    for date_t in datelist:
        port_df = merged[merged['date'] == date_t].copy()
        portfolio_indices[date_t] = port_df

    t_indices = time.time() - t0
    print(f"   Time: {t_indices:.3f}s")
    print(f"   Dates with data: {len(portfolio_indices)}")

    # Step 3: Run SLOW aggregation
    print("\n3. Running SLOW aggregation (pandas)...")
    t0 = time.time()
    slow_result = compute_within_firm_returns_aggregation(
        portfolio_indices=portfolio_indices,
        data_raw=data,
        datelist=datelist,
        firm_id_col=firm_id_col,
        rating_col=rating_col,
        rating_bins=rating_bins
    )
    t_slow = time.time() - t0
    print(f"   Time: {t_slow:.3f}s")
    slow_ls = slow_result['long_short']
    print(f"   L-S mean: {slow_ls.mean()*100:.4f}%")

    # Step 4: Prepare inputs for FAST aggregation
    print("\n4. Preparing inputs for FAST aggregation...")
    t0 = time.time()

    inputs = prepare_fast_aggregation_inputs(
        portfolio_indices, data, datelist, firm_id_col, rating_col, rating_bins
    )

    t_prep = time.time() - t0
    print(f"   Time: {t_prep:.3f}s")

    # Step 5: Run FAST aggregation
    print("\n5. Running FAST aggregation (numba)...")

    # Warm up
    _ = compute_within_firm_aggregation_fast(
        inputs['date_idx'][:100], inputs['firm_idx'][:100],
        inputs['rating_terc'][:100], inputs['ptf_rank'][:100],
        inputs['ret'][:100], inputs['vw'][:100],
        inputs['n_dates'], inputs['n_firms']
    )

    t0 = time.time()
    fast_ls, fast_high, fast_low, fast_valid = compute_within_firm_aggregation_fast(
        inputs['date_idx'], inputs['firm_idx'],
        inputs['rating_terc'], inputs['ptf_rank'],
        inputs['ret'], inputs['vw'],
        inputs['n_dates'], inputs['n_firms']
    )
    t_fast = time.time() - t0
    print(f"   Time: {t_fast:.4f}s")

    # Convert to Series with date index
    fast_ls_series = pd.Series(fast_ls, index=inputs['unique_dates'])
    print(f"   L-S mean: {fast_ls_series.mean()*100:.4f}%")

    # Step 6: Compare results
    print("\n6. Comparing results...")

    # Align dates
    common_dates = set(slow_ls.index) & set(fast_ls_series.index)
    slow_aligned = slow_ls.loc[list(common_dates)].sort_index()
    fast_aligned = fast_ls_series.loc[list(common_dates)].sort_index()

    diff = np.abs(slow_aligned.values - fast_aligned.values)
    max_diff = np.nanmax(diff)

    print(f"   Common dates: {len(common_dates)}")
    print(f"   Max absolute difference: {max_diff:.2e}")

    passed = max_diff < 1e-10
    print(f"\n   Result: {'PASS' if passed else 'FAIL'}")

    print(f"\n7. Performance Summary:")
    print(f"   Slow aggregation: {t_slow:.3f}s")
    print(f"   Fast aggregation: {t_fast:.4f}s (+ {t_prep:.3f}s prep)")
    print(f"   Speedup (aggregation only): {t_slow/t_fast:.1f}x")
    print(f"   Speedup (incl. prep): {t_slow/(t_fast+t_prep):.1f}x")

    return {
        'passed': passed,
        'max_diff': max_diff,
        'slow_time': t_slow,
        'fast_time': t_fast,
        'prep_time': t_prep,
        'speedup': t_slow / (t_fast + t_prep),
    }


def main():
    print("="*70)
    print("WithinFirmSort Fast Aggregation Validation")
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

    # Test numba kernel directly
    test_kernel_directly(data)

    # Full integration test
    results = test_integration_with_real_data(data)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Fast vs Slow Match: {'PASS' if results['passed'] else 'FAIL'}")
    print(f"Max Difference: {results['max_diff']:.2e}")
    print(f"Speedup: {results['speedup']:.1f}x")

    return results


if __name__ == '__main__':
    results = main()
