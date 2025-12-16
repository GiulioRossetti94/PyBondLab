# -*- coding: utf-8 -*-
"""
Data Uncertainty Test Script - SingleSort (Standard Signal)
============================================================

This script tests the fast path vs slow path using a STANDARD signal
(pre-computed column, not a derived signal like Momentum).

Unlike Momentum which requires lookback computation, SingleSort uses
an existing signal column directly. This isolates testing to the
portfolio formation logic without signal computation complexity.

Test Configurations:
--------------------
- Strategy: SingleSort with 'signal1' column
- Holding periods: hp=1 (simple) and hp=3 (staggered)
- dynamic_weights: True and False
- Filter types: trim, price, bounce, none

Usage:
------
python examples/data_uncertainty_singlesort.py
python examples/data_uncertainty_singlesort.py --no-validate  # Skip validation
python examples/data_uncertainty_singlesort.py --hp 1         # Test only hp=1

"""

import sys
import os
import time
import json
import pickle
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd

# Add parent directory to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

import PyBondLab as pbl
from PyBondLab.pbl_test import generate_synthetic_data_fast
from PyBondLab.config import StrategyFormationConfig, FormationConfig, DataConfig

# =============================================================================
# Configuration
# =============================================================================

RANDOM_SEED = 42
N_DATES = 60          # Number of monthly periods (5 years)
N_BONDS = 500         # Number of unique bonds
N_CHARS = 3           # Number of characteristics

# Strategy parameters
SIGNAL_VAR = 'signal1'    # Use pre-computed signal column
N_PORTFOLIOS = 5          # Number of portfolios

# Rating filter (None = all bonds)
RATING = None

# Output paths
RESULTS_DIR = Path(__file__).parent / "data_uncertainty_results"

# =============================================================================
# Filter Parameter Specifications
# =============================================================================

def get_filter_params(holding_period: int, dynamic_weights: bool) -> List[Dict]:
    """
    Generate filter parameter combinations for testing.

    Parameters
    ----------
    holding_period : int
        Holding period (1 or 3)
    dynamic_weights : bool
        Whether to use dynamic weights

    Returns
    -------
    List[Dict]
        List of parameter dictionaries for testing
    """
    # Create strategy
    strategy = pbl.SingleSort(
        holding_period=holding_period,
        sort_var=SIGNAL_VAR,
        num_portfolios=N_PORTFOLIOS,
        verbose=False
    )

    # Create config with dynamic_weights setting
    config = StrategyFormationConfig(
        data=DataConfig(rating=RATING),
        formation=FormationConfig(
            dynamic_weights=dynamic_weights,
            verbose=False
        )
    )

    params = []

    # ------------------------------------------------------------------
    # 1. TRIM filters (exclude extreme returns)
    # ------------------------------------------------------------------
    trim_levels = [0.20, -0.30]
    for level in trim_levels:
        params.append({
            'strategy': strategy,
            'config': config,
            'filters': {'adj': 'trim', 'level': level}
        })

    # ------------------------------------------------------------------
    # 2. PRICE filters (exclude bonds with extreme prices)
    # ------------------------------------------------------------------
    price_levels = [50, 200]
    for level in price_levels:
        params.append({
            'strategy': strategy,
            'config': config,
            'filters': {'adj': 'price', 'level': level}
        })

    # ------------------------------------------------------------------
    # 3. BOUNCE filters (exclude reversal returns)
    # ------------------------------------------------------------------
    bounce_levels = [0.05, -0.05]
    for level in bounce_levels:
        params.append({
            'strategy': strategy,
            'config': config,
            'filters': {'adj': 'bounce', 'level': level}
        })

    # ------------------------------------------------------------------
    # 4. NO FILTER baseline (for comparison)
    # ------------------------------------------------------------------
    params.append({
        'strategy': strategy,
        'config': config,
        'filters': None
    })

    return params


# =============================================================================
# Result Container
# =============================================================================

@dataclass
class SingleSortTestResult:
    """Container for a single test result."""
    test_name: str            # Unique test identifier
    holding_period: int       # hp=1 or hp=3
    dynamic_weights: bool     # True or False
    filter_type: str          # 'trim', 'price', 'bounce', 'none'
    filter_level: Any         # The filter level parameter

    # EA (Ex-Ante) results
    ew_ea_mean: float
    vw_ea_mean: float
    ew_ea_std: float
    vw_ea_std: float

    # EP (Ex-Post) results
    ew_ep_mean: float
    vw_ep_mean: float
    ew_ep_std: float
    vw_ep_std: float

    # Metadata
    n_dates: int
    runtime_seconds: float
    path_type: str            # 'fast' or 'slow'

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d['filter_level'] = str(self.filter_level)
        return d


# =============================================================================
# Test Runner
# =============================================================================

def run_single_test(
    data: pd.DataFrame,
    params: Dict,
    holding_period: int,
    dynamic_weights: bool,
    use_fast_path: bool = True,
    verbose: bool = True
) -> SingleSortTestResult:
    """
    Run a single filter test and capture EA and EP results.

    Parameters
    ----------
    data : pd.DataFrame
        Bond panel data
    params : dict
        Strategy parameters including filter specification
    holding_period : int
        Holding period (1 or 3)
    dynamic_weights : bool
        Whether to use dynamic weights
    use_fast_path : bool
        If True, use fast path. If False, force slow path.
    verbose : bool
        Whether to print progress

    Returns
    -------
    SingleSortTestResult
        Results container with EA and EP returns
    """
    t0 = time.time()

    # Make a copy of data for this test
    data_copy = data.copy()

    # Optionally disable fast path
    if not use_fast_path:
        original_method = pbl.StrategyFormation._can_use_fast_path
        pbl.StrategyFormation._can_use_fast_path = lambda self: False

    try:
        # Run strategy formation
        result = pbl.StrategyFormation(data_copy, **params).fit()

        # Get EA results
        ew_ea, vw_ea = result.get_long_short()

        # Get EP results (ex-post)
        try:
            ew_ep, vw_ep = result.get_long_short_ex_post()
        except (ValueError, AttributeError):
            # EP not available (no filter applied)
            ew_ep, vw_ep = ew_ea, vw_ea
    finally:
        if not use_fast_path:
            pbl.StrategyFormation._can_use_fast_path = original_method

    elapsed = time.time() - t0

    # Extract filter info
    filters = params.get('filters')
    if filters is None:
        filter_type = 'none'
        filter_level = None
    else:
        filter_type = filters.get('adj', 'none')
        filter_level = filters.get('level')

    # Generate test name
    dw_str = "dw_true" if dynamic_weights else "dw_false"
    filter_str = f"{filter_type}_{filter_level}" if filter_level else "no_filter"
    test_name = f"hp{holding_period}_{dw_str}_{filter_str}"

    path_type = "fast" if use_fast_path else "slow"

    test_result = SingleSortTestResult(
        test_name=test_name,
        holding_period=holding_period,
        dynamic_weights=dynamic_weights,
        filter_type=filter_type,
        filter_level=filter_level,
        ew_ea_mean=float(ew_ea.mean()),
        vw_ea_mean=float(vw_ea.mean()),
        ew_ea_std=float(ew_ea.std()),
        vw_ea_std=float(vw_ea.std()),
        ew_ep_mean=float(ew_ep.mean()),
        vw_ep_mean=float(vw_ep.mean()),
        ew_ep_std=float(ew_ep.std()),
        vw_ep_std=float(vw_ep.std()),
        n_dates=len(ew_ea),
        runtime_seconds=elapsed,
        path_type=path_type
    )

    if verbose:
        print(f"    [{path_type}] EA={ew_ea.mean():.6f}, EP={ew_ep.mean():.6f}, time={elapsed:.2f}s")

    return test_result


def run_test_with_validation(
    data: pd.DataFrame,
    params: Dict,
    holding_period: int,
    dynamic_weights: bool,
    tolerance: float = 1e-10,
    verbose: bool = True
) -> Tuple[SingleSortTestResult, SingleSortTestResult, bool, Dict]:
    """
    Run a test with BOTH slow and fast paths, and validate they match.

    Returns
    -------
    Tuple containing:
        - fast_result: from fast path
        - slow_result: from slow path (source of truth)
        - passed: bool indicating if results match within tolerance
        - diffs: dict with differences for each metric
    """
    if verbose:
        dw_str = "dw=True" if dynamic_weights else "dw=False"
        filters = params.get('filters')
        filter_str = f"{filters['adj']}={filters['level']}" if filters else "no_filter"
        print(f"  hp={holding_period}, {dw_str}, {filter_str}")

    # Run with slow path (source of truth)
    slow_result = run_single_test(
        data, params, holding_period, dynamic_weights,
        use_fast_path=False, verbose=verbose
    )

    # Run with fast path
    fast_result = run_single_test(
        data, params, holding_period, dynamic_weights,
        use_fast_path=True, verbose=verbose
    )

    # Compare results
    def safe_diff(a, b):
        if np.isnan(a) and np.isnan(b):
            return 0.0
        elif np.isnan(a) or np.isnan(b):
            return float('inf')
        return abs(a - b)

    diffs = {
        'ew_ea_mean': safe_diff(fast_result.ew_ea_mean, slow_result.ew_ea_mean),
        'vw_ea_mean': safe_diff(fast_result.vw_ea_mean, slow_result.vw_ea_mean),
        'ew_ep_mean': safe_diff(fast_result.ew_ep_mean, slow_result.ew_ep_mean),
        'vw_ep_mean': safe_diff(fast_result.vw_ep_mean, slow_result.vw_ep_mean),
    }

    passed = all(d <= tolerance for d in diffs.values())

    if verbose:
        max_diff = max(diffs.values())
        status = "PASS" if passed else "FAIL"
        print(f"    {status}: max_diff={max_diff:.2e}")

    return fast_result, slow_result, passed, diffs


def run_all_tests(
    data: pd.DataFrame,
    holding_periods: List[int] = [1, 3],
    dynamic_weights_options: List[bool] = [True, False],
    validate: bool = True,
    tolerance: float = 1e-10,
    verbose: bool = True
) -> Tuple[Dict[str, SingleSortTestResult], Dict[str, SingleSortTestResult], bool]:
    """
    Run all test configurations.

    Parameters
    ----------
    data : pd.DataFrame
        Bond panel data
    holding_periods : List[int]
        Holding periods to test
    dynamic_weights_options : List[bool]
        dynamic_weights settings to test
    validate : bool
        If True, run both slow and fast paths and compare
    tolerance : float
        Maximum allowed difference
    verbose : bool
        Whether to print progress

    Returns
    -------
    Tuple containing:
        - fast_results: dict of fast path results
        - slow_results: dict of slow path results (source of truth)
        - all_passed: bool
    """
    fast_results = {}
    slow_results = {}
    validation_results = []
    total_tests = 0
    total_slow_time = 0
    total_fast_time = 0

    for hp in holding_periods:
        for dw in dynamic_weights_options:
            if verbose:
                dw_str = "dynamic_weights=True" if dw else "dynamic_weights=False"
                print(f"\n{'='*60}")
                print(f"Testing: hp={hp}, {dw_str}")
                print(f"{'='*60}")

            params_list = get_filter_params(hp, dw)

            for params in params_list:
                total_tests += 1

                if validate:
                    fast_result, slow_result, passed, diffs = run_test_with_validation(
                        data, params, hp, dw, tolerance=tolerance, verbose=verbose
                    )
                    fast_results[fast_result.test_name] = fast_result
                    slow_results[slow_result.test_name] = slow_result
                    total_fast_time += fast_result.runtime_seconds
                    total_slow_time += slow_result.runtime_seconds
                    validation_results.append((fast_result.test_name, passed, max(diffs.values())))
                else:
                    fast_result = run_single_test(
                        data, params, hp, dw, use_fast_path=True, verbose=verbose
                    )
                    fast_results[fast_result.test_name] = fast_result
                    slow_results[fast_result.test_name] = fast_result
                    total_fast_time += fast_result.runtime_seconds
                    validation_results.append((fast_result.test_name, True, 0.0))

    all_passed = all(passed for _, passed, _ in validation_results)

    if verbose:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Total tests: {total_tests}")
        if validate:
            print(f"Slow path time: {total_slow_time:.2f}s")
            print(f"Fast path time: {total_fast_time:.2f}s")
            if total_fast_time > 0:
                print(f"Speedup: {total_slow_time/total_fast_time:.1f}x")
            n_passed = sum(1 for _, passed, _ in validation_results if passed)
            print(f"\nValidation: {n_passed}/{total_tests} PASSED")
            if not all_passed:
                print("\nFailed tests:")
                for name, passed, max_diff in validation_results:
                    if not passed:
                        print(f"  {name}: max_diff={max_diff:.2e}")
        else:
            print(f"Total time: {total_fast_time:.2f}s")

    return fast_results, slow_results, all_passed


# =============================================================================
# Save/Load Results
# =============================================================================

def save_results(results: Dict[str, SingleSortTestResult], output_dir: Path, filename: str):
    """Save results to JSON and pickle files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    json_path = output_dir / f"{filename}.json"
    json_results = {name: r.to_dict() for name, r in results.items()}
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved JSON results to: {json_path}")

    # Save as pickle
    pkl_path = output_dir / f"{filename}.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved pickle results to: {pkl_path}")


# =============================================================================
# Main
# =============================================================================

def main(
    holding_periods: List[int] = [1, 3],
    dynamic_weights_options: List[bool] = [True, False],
    validate: bool = True,
    tolerance: float = 1e-10
):
    """
    Run all SingleSort tests.

    Parameters
    ----------
    holding_periods : List[int]
        Holding periods to test (default: [1, 3])
    dynamic_weights_options : List[bool]
        dynamic_weights settings to test (default: [True, False])
    validate : bool
        If True, compare slow vs fast path
    tolerance : float
        Maximum allowed difference for validation
    """
    print("=" * 70)
    print("SINGLESORT DATA UNCERTAINTY TEST")
    print("Testing fast path vs slow path with standard signal")
    print("=" * 70)

    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    t0 = time.time()
    data = generate_synthetic_data_fast(
        n_dates=N_DATES,
        n_bonds=N_BONDS,
        seed=RANDOM_SEED,
        n_chars=N_CHARS,
        balanced_panel=False,
        allow_nans=True,
    )
    print(f"   Shape: {data.shape}")
    print(f"   Signal column: {SIGNAL_VAR}")
    print(f"   Signal range: {data[SIGNAL_VAR].min():.3f} - {data[SIGNAL_VAR].max():.3f}")
    print(f"   Time: {time.time()-t0:.2f}s")

    # Run tests
    print("\n2. Running tests...")
    fast_results, slow_results, all_passed = run_all_tests(
        data,
        holding_periods=holding_periods,
        dynamic_weights_options=dynamic_weights_options,
        validate=validate,
        tolerance=tolerance,
        verbose=True
    )

    # Print detailed results
    print("\n3. Detailed Results (Slow Path = Source of Truth)")
    print("=" * 90)
    print(f"{'Test Name':<40} {'EW EA':>10} {'VW EA':>10} {'EW EP':>10} {'VW EP':>10}")
    print("-" * 90)
    for name in sorted(slow_results.keys()):
        r = slow_results[name]
        print(f"{name:<40} {r.ew_ea_mean:>10.6f} {r.vw_ea_mean:>10.6f} {r.ew_ep_mean:>10.6f} {r.vw_ep_mean:>10.6f}")

    # Save results
    print("\n4. Saving results...")
    save_results(slow_results, RESULTS_DIR, "singlesort_baseline")

    # Final status
    print("\n" + "=" * 70)
    if validate:
        if all_passed:
            print("ALL TESTS PASSED")
            print("Fast path matches slow path within tolerance")
        else:
            print("SOME TESTS FAILED")
            print("Fast path does NOT match slow path - needs investigation")
    print("=" * 70)

    return fast_results, slow_results, all_passed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run SingleSort data uncertainty tests")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip slow vs fast path validation")
    parser.add_argument("--tolerance", type=float, default=1e-10,
                        help="Tolerance for validation (default: 1e-10)")
    parser.add_argument("--hp", type=int, choices=[1, 3], default=None,
                        help="Test only specific holding period")
    parser.add_argument("--dw", type=str, choices=['true', 'false'], default=None,
                        help="Test only specific dynamic_weights setting")
    args = parser.parse_args()

    # Determine what to test
    if args.hp:
        holding_periods = [args.hp]
    else:
        holding_periods = [1, 3]

    if args.dw:
        dynamic_weights_options = [args.dw.lower() == 'true']
    else:
        dynamic_weights_options = [True, False]

    fast_results, slow_results, all_passed = main(
        holding_periods=holding_periods,
        dynamic_weights_options=dynamic_weights_options,
        validate=not args.no_validate,
        tolerance=args.tolerance
    )
