"""
Profiling Script: WithinFirmSort Strategy with Large Data

This script profiles the WithinFirmSort implementation with data size
similar to the user's real data to identify bottlenecks.

User's data characteristics:
- unique permno: 2,885
- unique cusip : 50,433
- unique dates : 272  (2002-08-31 to 2025-03-31)

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


def generate_large_withinfirm_data(
    n_dates: int = 272,
    n_firms: int = 2885,
    avg_bonds_per_firm: int = 17,  # ~50k cusips / 2.9k firms
    seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic data matching user's data size."""
    np.random.seed(seed)

    dates = pd.date_range('2002-08-31', periods=n_dates, freq='ME')

    records = []
    cusip_counter = 0

    print(f"Generating {n_firms} firms with avg {avg_bonds_per_firm} bonds each...")

    for firm_id in range(n_firms):
        # Varying number of bonds per firm
        n_bonds = max(2, int(np.random.exponential(avg_bonds_per_firm)))
        n_bonds = min(n_bonds, 50)  # Cap at 50 bonds per firm

        firm_base_rating = np.random.randint(1, 18)
        firm_base_yield = np.random.uniform(0.02, 0.10)

        for bond_idx in range(n_bonds):
            cusip = f'CUSIP{cusip_counter:06d}'
            cusip_counter += 1

            bond_rating_offset = np.random.randint(-2, 3)
            bond_rating = max(1, min(21, firm_base_rating + bond_rating_offset))
            bond_yield_offset = np.random.uniform(-0.01, 0.01)

            # Not all bonds exist for all dates (realistic attrition)
            start_date_idx = np.random.randint(0, max(1, n_dates - 60))
            end_date_idx = min(n_dates, start_date_idx + np.random.randint(12, 120))

            for date_idx in range(start_date_idx, end_date_idx):
                date = dates[date_idx]
                ytm = max(0.01, firm_base_yield + bond_yield_offset + np.random.uniform(-0.005, 0.005))
                ret = np.random.normal(0.005, 0.03)
                mcap = np.random.lognormal(15, 2)

                records.append({
                    'cusip': cusip,
                    'permno': firm_id,
                    'date': date,
                    'spc_rat': bond_rating,
                    'ytm': ytm,
                    'ret_vw': ret,
                    'mcap_e': mcap,
                })

        if (firm_id + 1) % 500 == 0:
            print(f"  Generated {firm_id + 1}/{n_firms} firms, {len(records)} records so far...")

    df = pd.DataFrame(records)
    df = df.sort_values(['cusip', 'date']).reset_index(drop=True)
    return df


def profile_section_by_section(data, verbose=True):
    """Profile each section of WithinFirmSort."""
    print(f"\n{'='*70}")
    print("Section-by-Section Profile")
    print(f"{'='*70}")

    # Full run with timing
    strategy = pbl.WithinFirmSort(
        holding_period=1,
        sort_var='ytm',
        firm_id_col='permno',
        min_bonds_per_firm=2,
        rating_bins=[-np.inf, 7, 10, np.inf],
        num_portfolios=2,
        verbose=False
    )

    print("\nRunning full StrategyFormation.fit()...")
    t0 = time.time()
    result = pbl.StrategyFormation(
        data=data.copy(),
        strategy=strategy,
        turnover=False,
        verbose=False
    ).fit(
        IDvar='cusip',
        RETvar='ret_vw',
        VWvar='mcap_e',
        RATINGvar='spc_rat'
    )
    total_time = time.time() - t0

    ew_ls, vw_ls = result.get_long_short()
    print(f"\nTotal time: {total_time:.2f}s")
    print(f"EW L-S mean: {ew_ls.mean()*100:.4f}%")
    print(f"VW L-S mean: {vw_ls.mean()*100:.4f}%")
    print(f"Non-NaN observations: {(~np.isnan(vw_ls)).sum()}")

    return result, total_time


def profile_detailed(data):
    """Run detailed cProfile."""
    print(f"\n{'='*70}")
    print("Detailed cProfile")
    print(f"{'='*70}")

    strategy = pbl.WithinFirmSort(
        holding_period=1,
        sort_var='ytm',
        firm_id_col='permno',
        min_bonds_per_firm=2,
        rating_bins=[-np.inf, 7, 10, np.inf],
        num_portfolios=2,
        verbose=False
    )

    # Create profiler
    pr = cProfile.Profile()
    pr.enable()

    result = pbl.StrategyFormation(
        data=data.copy(),
        strategy=strategy,
        turnover=False,
        verbose=False
    ).fit(
        IDvar='cusip',
        RETvar='ret_vw',
        VWvar='mcap_e',
        RATINGvar='spc_rat'
    )

    pr.disable()

    # Print stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(40)  # Top 40 functions
    print(s.getvalue())

    return result


def main():
    print("="*70)
    print("WithinFirmSort Large Data Profiling")
    print("="*70)

    # Generate test data matching user's size
    print("\nGenerating large test data (matching user's data size)...")
    t0 = time.time()
    data = generate_large_withinfirm_data(
        n_dates=272,
        n_firms=2885,
        avg_bonds_per_firm=17,
        seed=42
    )
    gen_time = time.time() - t0
    print(f"\nData generation time: {gen_time:.2f}s")
    print(f"Data shape: {data.shape}")
    print(f"Unique cusips: {data['cusip'].nunique()}")
    print(f"Unique permno: {data['permno'].nunique()}")
    print(f"Unique dates: {data['date'].nunique()}")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")

    # Warm up JIT
    print("\nWarming up JIT compilation...")
    small_data = data[data['date'] <= data['date'].unique()[10]].copy()
    _ = profile_section_by_section(small_data, verbose=False)

    # Profile with full data
    result, total_time = profile_section_by_section(data)

    # Detailed profile
    profile_detailed(data)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Data size: {len(data):,} rows, {data['cusip'].nunique():,} cusips, {data['permno'].nunique():,} firms")
    print(f"Total time: {total_time:.2f}s")


if __name__ == '__main__':
    main()
