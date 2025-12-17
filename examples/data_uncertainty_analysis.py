# -*- coding: utf-8 -*-
"""
Data Uncertainty Analysis Example
=================================

This example demonstrates how to use the DataUncertaintyAnalysis wrapper
to run comprehensive data uncertainty analysis across multiple holding
periods and filter configurations.

Usage:
------
python examples/data_uncertainty_analysis.py

Options:
--------
--n-jobs N        Number of parallel workers (default: 1)
--no-parallel     Disable parallel processing
--quick           Run quick test with fewer configurations
"""

import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

import PyBondLab as pbl
from PyBondLab import DataUncertaintyAnalysis
from PyBondLab.pbl_test import generate_synthetic_data_fast


def run_signal_analysis(n_jobs: int = 1, quick: bool = False):
    """
    Run data uncertainty analysis on pre-computed signals.

    Parameters
    ----------
    n_jobs : int
        Number of parallel workers
    quick : bool
        If True, run fewer configurations for faster testing
    """
    print("=" * 70)
    print("DATA UNCERTAINTY ANALYSIS - Pre-computed Signals")
    print("=" * 70)

    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    t0 = time.time()
    data = generate_synthetic_data_fast(
        n_dates=60,
        n_bonds=500,
        seed=42,
        n_chars=3,
        balanced_panel=False,
        allow_nans=True,
    )
    print(f"   Shape: {data.shape}")
    print(f"   Time: {time.time()-t0:.2f}s")

    # Define filter configurations
    if quick:
        filters = {
            'trim': [0.2, -0.3],
            'wins': [(99, 'both')],
        }
        holding_periods = [1, 3]
    else:
        filters = {
            'trim': [0.2, 0.5, -0.3, [-0.3, 0.3]],
            'price': [50, 200, 500],
            'bounce': [0.05, -0.05],
            'wins': [(99, 'both'), (99, 'right'), (95, 'both')],
        }
        holding_periods = [1, 3, 6]

    # Run analysis
    print("\n2. Running data uncertainty analysis...")
    t0 = time.time()
    results = DataUncertaintyAnalysis(
        data=data,
        signals=['signal1'],
        holding_periods=holding_periods,
        filters=filters,
        num_portfolios=5,
        dynamic_weights=True,
        include_baseline=True,
        n_jobs=n_jobs,
        verbose=True,
    ).fit()
    print(f"   Total time: {time.time()-t0:.2f}s")

    # Display results
    print("\n3. Results Summary")
    print("=" * 70)
    print(results)

    # Display summary statistics
    print("\n4. Summary Statistics (means in %, NW t-stats)")
    print("-" * 100)
    summary = results.summary()
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(summary.to_string(index=False))

    # Show filter by hp
    print("\n5. Filter Results by Holding Period")
    print("-" * 70)
    for hp in holding_periods:
        hp_results = results.filter(hp=hp)
        hp_summary = hp_results.summary()
        print(f"\nHP = {hp}:")
        print(f"  Configs: {len(hp_results.configs)}")
        print(f"  Mean EW EA: {hp_summary['ew_ea_mean'].mean():.4f}%")
        print(f"  Mean VW EA: {hp_summary['vw_ea_mean'].mean():.4f}%")

    # Show filter by location
    print("\n6. Filter Results by Location")
    print("-" * 70)
    for loc in ['left', 'right', 'both']:
        loc_results = results.filter(location=loc)
        if len(loc_results.configs) > 0:
            loc_summary = loc_results.summary()
            print(f"\nLocation = {loc}:")
            print(f"  Configs: {len(loc_results.configs)}")
            print(f"  Mean EW EA: {loc_summary['ew_ea_mean'].mean():.4f}%")

    # Access factor panels
    print("\n7. Factor Panel Dimensions")
    print("-" * 70)
    print(f"  EW EA: {results.ew_ea.shape}")
    print(f"  VW EA: {results.vw_ea.shape}")
    print(f"  EW EP: {results.ew_ep.shape}")
    print(f"  VW EP: {results.vw_ep.shape}")

    return results


def run_strategy_analysis(n_jobs: int = 1):
    """
    Run data uncertainty analysis using a Strategy object (Momentum).

    Parameters
    ----------
    n_jobs : int
        Number of parallel workers
    """
    print("\n" + "=" * 70)
    print("DATA UNCERTAINTY ANALYSIS - Momentum Strategy")
    print("=" * 70)

    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    t0 = time.time()
    data = generate_synthetic_data_fast(
        n_dates=60,
        n_bonds=500,
        seed=42,
        n_chars=3,
        balanced_panel=False,
        allow_nans=True,
    )
    print(f"   Shape: {data.shape}")
    print(f"   Time: {time.time()-t0:.2f}s")

    # Create Momentum strategy
    mom = pbl.Momentum(
        holding_period=1,  # Will be overridden by analysis
        lookback_period=3,
        skip=1,
        num_portfolios=5,
        verbose=False
    )

    # Define filters
    filters = {
        'trim': [0.2, -0.3],
        'wins': [(99, 'both')],
    }

    # Run analysis
    print("\n2. Running data uncertainty analysis...")
    t0 = time.time()
    results = DataUncertaintyAnalysis(
        data=data,
        strategy=mom,
        holding_periods=[1, 3],
        filters=filters,
        num_portfolios=5,
        dynamic_weights=True,
        include_baseline=True,
        n_jobs=n_jobs,
        verbose=True,
    ).fit()
    print(f"   Total time: {time.time()-t0:.2f}s")

    # Display summary
    print("\n3. Summary Statistics")
    print("-" * 100)
    summary = results.summary()
    print(summary.to_string(index=False))

    return results


def run_multi_signal_analysis(n_jobs: int = 1):
    """
    Run data uncertainty analysis on multiple signals simultaneously.

    Parameters
    ----------
    n_jobs : int
        Number of parallel workers
    """
    print("\n" + "=" * 70)
    print("DATA UNCERTAINTY ANALYSIS - Multiple Signals")
    print("=" * 70)

    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    t0 = time.time()
    data = generate_synthetic_data_fast(
        n_dates=60,
        n_bonds=500,
        seed=42,
        n_chars=3,
        balanced_panel=False,
        allow_nans=True,
    )
    print(f"   Shape: {data.shape}")
    print(f"   Time: {time.time()-t0:.2f}s")

    # Define filters
    filters = {
        'trim': [0.2, -0.3],
    }

    # Run analysis on multiple signals
    print("\n2. Running data uncertainty analysis on 3 signals...")
    t0 = time.time()
    results = DataUncertaintyAnalysis(
        data=data,
        signals=['signal1', 'char1', 'char2'],  # Use multiple columns
        holding_periods=[1, 3],
        filters=filters,
        num_portfolios=5,
        dynamic_weights=True,
        include_baseline=True,
        n_jobs=n_jobs,
        verbose=True,
    ).fit()
    print(f"   Total time: {time.time()-t0:.2f}s")

    # Display results per signal
    print("\n3. Results by Signal")
    print("-" * 70)
    for signal in ['signal1', 'char1', 'char2']:
        sig_results = results.filter(signal=signal)
        sig_summary = sig_results.summary()
        print(f"\nSignal: {signal}")
        print(f"  Configs: {len(sig_results.configs)}")
        print(f"  Mean EW EA: {sig_summary['ew_ea_mean'].mean():.4f}%")
        print(f"  Mean VW EA: {sig_summary['vw_ea_mean'].mean():.4f}%")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run data uncertainty analysis examples"
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Number of parallel workers (default: 1)"
    )
    parser.add_argument(
        "--no-parallel", action="store_true",
        help="Disable parallel processing"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run quick test with fewer configurations"
    )
    parser.add_argument(
        "--example", choices=['signal', 'strategy', 'multi', 'all'],
        default='all',
        help="Which example to run (default: all)"
    )
    args = parser.parse_args()

    n_jobs = 1 if args.no_parallel else args.n_jobs

    results = {}

    if args.example in ['signal', 'all']:
        results['signal'] = run_signal_analysis(n_jobs=n_jobs, quick=args.quick)

    if args.example in ['strategy', 'all']:
        results['strategy'] = run_strategy_analysis(n_jobs=n_jobs)

    if args.example in ['multi', 'all']:
        results['multi'] = run_multi_signal_analysis(n_jobs=n_jobs)

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
