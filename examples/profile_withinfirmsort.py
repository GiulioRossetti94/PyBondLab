"""
Profiling Script: WithinFirmSort Strategy

This script profiles the WithinFirmSort implementation to identify
bottlenecks for Phase 16 optimization.

Author: Claude
Date: 2024
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import time
import cProfile
import pstats
import io
from pstats import SortKey
import PyBondLab as pbl


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
                    'char1': np.random.uniform(0, 1),
                    'char2': np.random.uniform(0, 1),
                })

    df = pd.DataFrame(records)
    df = df.sort_values(['ID', 'date']).reset_index(drop=True)
    return df


def run_withinfirmsort(data, hp=1, turnover=False, verbose=False):
    """Run WithinFirmSort and return result."""
    strategy = pbl.WithinFirmSort(
        holding_period=hp,
        sort_var='CS',
        firm_id_col='PERMNO',
        min_bonds_per_firm=2,
        rating_bins=[-np.inf, 7, 10, np.inf],
        num_portfolios=2,
        verbose=verbose
    )

    result = pbl.StrategyFormation(
        data=data.copy(),
        strategy=strategy,
        turnover=turnover,
        verbose=verbose
    ).fit()

    return result


def profile_detailed(data, hp=1, turnover=False):
    """Run detailed profiling with cProfile."""
    print(f"\n{'='*70}")
    print(f"Detailed Profile: HP={hp}, turnover={turnover}")
    print(f"{'='*70}")

    # Create profiler
    pr = cProfile.Profile()
    pr.enable()

    # Run the code
    result = run_withinfirmsort(data, hp=hp, turnover=turnover, verbose=False)

    pr.disable()

    # Print stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(30)  # Top 30 functions
    print(s.getvalue())

    return result


def profile_by_section(data, hp=1, turnover=False):
    """Profile individual sections of WithinFirmSort."""
    print(f"\n{'='*70}")
    print(f"Section-by-Section Profile: HP={hp}, turnover={turnover}")
    print(f"{'='*70}")

    # Import necessary modules
    from PyBondLab.utils_within_firm import (
        compute_within_firm_portfolios,
        compute_within_firm_returns_aggregation
    )
    from PyBondLab.constants import ColumnNames

    strategy = pbl.WithinFirmSort(
        holding_period=hp,
        sort_var='CS',
        firm_id_col='PERMNO',
        min_bonds_per_firm=2,
        rating_bins=[-np.inf, 7, 10, np.inf],
        num_portfolios=2,
        verbose=False
    )

    # Section 1: Data preparation
    t0 = time.time()
    data_copy = data.copy()
    data_copy = data_copy.sort_values(['ID', 'date']).reset_index(drop=True)
    t_data_prep = time.time() - t0
    print(f"1. Data preparation: {t_data_prep:.3f}s")

    # Section 2: Within-firm portfolio assignment
    t0 = time.time()
    bond_assignments, firm_weights_df = compute_within_firm_portfolios(
        data=data_copy,
        signal_col='CS',
        return_col='ret',
        weight_col='VW',
        firm_id_col='PERMNO',
        rating_col='RATING_NUM',
        rating_bins=[-np.inf, 7, 10, np.inf],
        min_bonds_per_firm=2
    )
    t_portfolio_assign = time.time() - t0
    print(f"2. Portfolio assignment: {t_portfolio_assign:.3f}s")
    print(f"   - Bonds assigned: {len(bond_assignments)}")
    print(f"   - Firms with valid assignments: {len(firm_weights_df)}")

    # Section 3: Full StrategyFormation (includes aggregation)
    t0 = time.time()
    result = run_withinfirmsort(data, hp=hp, turnover=turnover, verbose=False)
    t_total = time.time() - t0
    print(f"3. Total StrategyFormation.fit(): {t_total:.3f}s")

    # Calculate overhead
    t_other = t_total - t_data_prep - t_portfolio_assign
    print(f"4. Other (aggregation, etc.): {t_other:.3f}s")

    print(f"\nBreakdown:")
    print(f"  Data prep:           {t_data_prep/t_total*100:5.1f}%")
    print(f"  Portfolio assignment: {t_portfolio_assign/t_total*100:5.1f}%")
    print(f"  Other:               {t_other/t_total*100:5.1f}%")

    return result


def profile_compute_within_firm_portfolios(data):
    """Deep dive into compute_within_firm_portfolios."""
    print(f"\n{'='*70}")
    print("Deep Dive: compute_within_firm_portfolios()")
    print(f"{'='*70}")

    from PyBondLab.utils_within_firm import compute_within_firm_assignments_numba
    from PyBondLab.constants import ColumnNames

    # Prepare data
    data_copy = data.copy()

    # Section 1: Rating tercile creation
    t0 = time.time()
    rating_terc = pd.cut(
        pd.to_numeric(data_copy['RATING_NUM'], errors='coerce'),
        bins=[-np.inf, 7, 10, np.inf],
        labels=[1, 2, 3],
        include_lowest=True
    ).astype('Int64')
    t_rating = time.time() - t0
    print(f"1. Rating tercile creation: {t_rating:.4f}s")

    # Section 2: Valid mask creation
    t0 = time.time()
    valid_mask = (
        rating_terc.notna() &
        data_copy['CS'].notna() &
        data_copy['VW'].notna() &
        data_copy['PERMNO'].notna()
    )
    sub = data_copy.loc[valid_mask].copy()
    sub['rating_terc'] = rating_terc[valid_mask].values
    sub['VW'] = sub['VW'].clip(lower=1e-10)
    t_mask = time.time() - t0
    print(f"2. Valid mask & subset: {t_mask:.4f}s")

    # Section 3: Grouping
    t0 = time.time()
    grp_cols = ['date', 'rating_terc', 'PERMNO']
    sub = sub.sort_values(grp_cols).reset_index(drop=True)
    sub['_grp'] = sub.groupby(grp_cols, sort=False).ngroup()
    t_grouping = time.time() - t0
    print(f"3. Grouping (sort + ngroup): {t_grouping:.4f}s")

    # Section 4: Find group boundaries
    t0 = time.time()
    group_changes = np.concatenate([
        [0],
        np.where(np.diff(sub['_grp']) != 0)[0] + 1,
        [len(sub)]
    ])
    group_starts = group_changes[:-1].astype(np.int64)
    group_ends = group_changes[1:].astype(np.int64)
    n_groups = len(group_starts)
    t_boundaries = time.time() - t0
    print(f"4. Group boundaries: {t_boundaries:.4f}s")
    print(f"   - Number of groups: {n_groups}")

    # Section 5: Convert to numpy
    t0 = time.time()
    sig_vals = sub['CS'].to_numpy(dtype=np.float64)
    w_vals = sub['VW'].to_numpy(dtype=np.float64)
    t_numpy = time.time() - t0
    print(f"5. Convert to numpy: {t_numpy:.4f}s")

    # Section 6: Group metadata
    t0 = time.time()
    group_meta = sub.groupby('_grp').agg({
        'date': 'first',
        'rating_terc': 'first',
        'PERMNO': 'first'
    }).reset_index()
    t_meta = time.time() - t0
    print(f"6. Group metadata: {t_meta:.4f}s")

    # Section 7: Numba kernel (FIRST RUN - includes JIT)
    t0 = time.time()
    portfolio_ranks, firm_weights_arr = compute_within_firm_assignments_numba(
        sig_vals, w_vals, group_starts, group_ends, 2
    )
    t_numba_first = time.time() - t0
    print(f"7a. Numba kernel (1st run, includes JIT): {t_numba_first:.4f}s")

    # Section 7b: Numba kernel (WARMED)
    t0 = time.time()
    for _ in range(3):
        portfolio_ranks, firm_weights_arr = compute_within_firm_assignments_numba(
            sig_vals, w_vals, group_starts, group_ends, 2
        )
    t_numba_warmed = (time.time() - t0) / 3
    print(f"7b. Numba kernel (warmed, avg of 3): {t_numba_warmed:.4f}s")

    # Section 8: Create output DataFrames
    t0 = time.time()
    bond_assignments = pd.DataFrame({
        'ID': sub['ID'].values,
        'date': sub['date'].values,
        'ptf_rank': portfolio_ranks.astype(int),
    })
    bond_assignments = bond_assignments[bond_assignments['ptf_rank'] > 0]

    firm_weights_df = pd.DataFrame({
        '_grp': np.arange(n_groups),
        'firm_weight': firm_weights_arr
    })
    firm_weights_df = firm_weights_df.merge(group_meta, on='_grp')
    firm_weights_df = firm_weights_df[firm_weights_df['firm_weight'] > 0]
    t_output = time.time() - t0
    print(f"8. Create output DataFrames: {t_output:.4f}s")

    total = t_rating + t_mask + t_grouping + t_boundaries + t_numpy + t_meta + t_numba_warmed + t_output
    print(f"\nTotal (with warmed numba): {total:.4f}s")
    print(f"\nBottleneck Analysis:")
    for name, t in [
        ("Rating tercile", t_rating),
        ("Valid mask", t_mask),
        ("Grouping", t_grouping),
        ("Boundaries", t_boundaries),
        ("Numpy conv", t_numpy),
        ("Metadata", t_meta),
        ("Numba kernel", t_numba_warmed),
        ("Output DFs", t_output)
    ]:
        print(f"  {name:20s}: {t:.4f}s ({t/total*100:5.1f}%)")


def main():
    print("="*70)
    print("WithinFirmSort Profiling Suite")
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

    # Warm up JIT
    print("\nWarming up JIT compilation...")
    _ = run_withinfirmsort(data.head(1000), hp=1, turnover=False, verbose=False)

    # Profile by section
    profile_by_section(data, hp=1, turnover=False)
    profile_by_section(data, hp=1, turnover=True)
    profile_by_section(data, hp=3, turnover=False)

    # Deep dive into portfolio assignment
    profile_compute_within_firm_portfolios(data)

    # Detailed cProfile
    print("\n\nRunning detailed cProfile (HP=1, turnover=False)...")
    profile_detailed(data, hp=1, turnover=False)


if __name__ == '__main__':
    main()
