# -*- coding: utf-8 -*-
"""
Data Uncertainty Baseline Test Script
=====================================

This script establishes SOURCE OF TRUTH results for data uncertainty analysis.
It tests various filter combinations with Momentum strategy and captures
both EA (Ex-Ante) and EP (Ex-Post) returns.

Based on: DataUncertainty_DRR.py (Dickerson, Robotti, Rossetti 2024)

Test Configurations:
--------------------
- Strategy: Momentum(3,3) with skip=1
- Filter types: trim, price, bounce
- Multiple levels per filter type
- Both EA and EP returns captured

Usage:
------
python examples/data_uncertainty_baseline.py

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

# =============================================================================
# Configuration
# =============================================================================

RANDOM_SEED = 42
N_DATES = 60          # Number of monthly periods (5 years)
N_BONDS = 500         # Number of unique bonds
N_CHARS = 3           # Number of characteristics

# Momentum strategy parameters
HOLDING_PERIOD = 3
LOOKBACK_PERIOD = 3
SKIP = 1
N_PORTFOLIOS = 5      # Use 5 for faster testing (vs 10 in original)

# Rating filter (None = all bonds, "NIG" = high-yield, "IG" = investment grade)
RATING = None  # Use all bonds for synthetic data

# Output paths
RESULTS_DIR = Path(__file__).parent / "data_uncertainty_results"

# =============================================================================
# Filter Parameter Specifications (Small subset for baseline)
# =============================================================================

def get_filter_params(strategy) -> List[Dict]:
    """
    Generate filter parameter combinations for baseline testing.

    This is a SMALL subset of the full DataUncertainty_DRR.py parameters,
    designed to run quickly while covering all filter types.
    """
    params = []

    # ------------------------------------------------------------------
    # 1. TRIM filters (exclude extreme returns)
    # ------------------------------------------------------------------
    trim_levels = [0.20, 0.50, -0.30, [-0.30, 0.30]]
    for level in trim_levels:
        params.append({
            'strategy': strategy,
            'rating': RATING,
            'filters': {'adj': 'trim', 'level': level}
        })

    # ------------------------------------------------------------------
    # 2. PRICE filters (exclude bonds with extreme prices)
    # ------------------------------------------------------------------
    price_levels = [50, 200, 500, [20, 500]]  # low, high, very high, range
    for level in price_levels:
        params.append({
            'strategy': strategy,
            'rating': RATING,
            'filters': {'adj': 'price', 'level': level}
        })

    # ------------------------------------------------------------------
    # 3. BOUNCE filters (exclude reversal returns)
    # ------------------------------------------------------------------
    bounce_levels = [0.05, -0.05, 0.10, [-0.05, 0.05]]
    for level in bounce_levels:
        params.append({
            'strategy': strategy,
            'rating': RATING,
            'filters': {'adj': 'bounce', 'level': level}
        })

    # ------------------------------------------------------------------
    # 4. NO FILTER baseline (for comparison)
    # ------------------------------------------------------------------
    params.append({
        'strategy': strategy,
        'rating': RATING,
        'filters': None
    })

    return params


# =============================================================================
# Result Container
# =============================================================================

@dataclass
class FilterTestResult:
    """Container for a single filter test result."""
    filter_type: str          # 'trim', 'price', 'bounce', 'none'
    filter_level: Any         # The filter level parameter
    strategy_name: str        # Full strategy name from result

    # EA (Ex-Ante) results
    ew_ea_mean: float         # EW long-short mean return (EA)
    vw_ea_mean: float         # VW long-short mean return (EA)
    ew_ea_std: float          # EW long-short std (EA)
    vw_ea_std: float          # VW long-short std (EA)

    # EP (Ex-Post) results
    ew_ep_mean: float         # EW long-short mean return (EP)
    vw_ep_mean: float         # VW long-short mean return (EP)
    ew_ep_std: float          # EW long-short std (EP)
    vw_ep_std: float          # VW long-short std (EP)

    # Metadata
    n_dates: int
    runtime_seconds: float

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert filter_level to string for JSON compatibility
        d['filter_level'] = str(self.filter_level)
        return d


# =============================================================================
# Test Runner
# =============================================================================

def run_single_filter_test(
    data: pd.DataFrame,
    params: Dict,
    verbose: bool = True,
    use_fast_path: bool = True
) -> FilterTestResult:
    """
    Run a single filter test and capture EA and EP results.

    Parameters
    ----------
    data : pd.DataFrame
        Bond panel data
    params : dict
        Strategy parameters including filter specification
    verbose : bool
        Whether to print progress
    use_fast_path : bool
        If True, use fast path. If False, force slow path.

    Returns
    -------
    FilterTestResult
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
        ew_ea, vw_ea = result.get_long_short()  # Default is EA

        # Get EP results (ex-post)
        try:
            ew_ep, vw_ep = result.get_long_short_ex_post()
        except (ValueError, AttributeError):
            # EP not available (no filter applied)
            ew_ep, vw_ep = ew_ea, vw_ea  # Use EA as fallback
    finally:
        # Restore fast path if we disabled it
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

    test_result = FilterTestResult(
        filter_type=filter_type,
        filter_level=filter_level,
        strategy_name=result.name,
        ew_ea_mean=float(ew_ea.mean()),
        vw_ea_mean=float(vw_ea.mean()),
        ew_ea_std=float(ew_ea.std()),
        vw_ea_std=float(vw_ea.std()),
        ew_ep_mean=float(ew_ep.mean()),
        vw_ep_mean=float(vw_ep.mean()),
        ew_ep_std=float(ew_ep.std()),
        vw_ep_std=float(vw_ep.std()),
        n_dates=len(ew_ea),
        runtime_seconds=elapsed
    )

    if verbose:
        path_type = "fast" if use_fast_path else "slow"
        print(f"  [{path_type}] {result.name}: EA={ew_ea.mean():.6f}, EP={ew_ep.mean():.6f}, time={elapsed:.2f}s")

    return test_result


def run_single_filter_test_with_validation(
    data: pd.DataFrame,
    params: Dict,
    tolerance: float = 1e-6,
    verbose: bool = True
) -> Tuple[FilterTestResult, FilterTestResult, bool, Dict]:
    """
    Run a single filter test with BOTH slow and fast paths, and validate they match.

    Parameters
    ----------
    data : pd.DataFrame
        Bond panel data
    params : dict
        Strategy parameters including filter specification
    tolerance : float
        Maximum allowed difference between slow and fast path
    verbose : bool
        Whether to print progress

    Returns
    -------
    Tuple containing:
        - fast_result: FilterTestResult from fast path
        - slow_result: FilterTestResult from slow path
        - passed: bool indicating if results match within tolerance
        - diffs: dict with differences for each metric
    """
    # Run with slow path (source of truth)
    slow_result = run_single_filter_test(data, params, verbose=verbose, use_fast_path=False)

    # Run with fast path
    fast_result = run_single_filter_test(data, params, verbose=verbose, use_fast_path=True)

    # Compare results (handle NaN gracefully)
    def safe_diff(a, b):
        if np.isnan(a) and np.isnan(b):
            return 0.0  # Both NaN = match
        elif np.isnan(a) or np.isnan(b):
            return float('inf')  # One NaN = fail
        return abs(a - b)

    diffs = {
        'ew_ea_mean': safe_diff(fast_result.ew_ea_mean, slow_result.ew_ea_mean),
        'vw_ea_mean': safe_diff(fast_result.vw_ea_mean, slow_result.vw_ea_mean),
        'ew_ep_mean': safe_diff(fast_result.ew_ep_mean, slow_result.ew_ep_mean),
        'vw_ep_mean': safe_diff(fast_result.vw_ep_mean, slow_result.vw_ep_mean),
    }

    # Check if all diffs are within tolerance (inf = fail)
    passed = all(d <= tolerance for d in diffs.values())

    return fast_result, slow_result, passed, diffs


def get_unique_key(params: Dict) -> str:
    """Generate a unique key for this filter configuration."""
    filters = params.get('filters')
    if filters is None:
        return "no_filter"

    adj = filters.get('adj', 'unknown')
    level = filters.get('level')

    if isinstance(level, list):
        level_str = f"{level[0]}_{level[1]}"
    else:
        level_str = str(level)

    return f"{adj}_{level_str}"


def run_all_baseline_tests(
    data: pd.DataFrame,
    verbose: bool = True,
    validate: bool = True,
    tolerance: float = 1e-6
) -> Tuple[Dict[str, FilterTestResult], Dict[str, FilterTestResult], bool]:
    """
    Run all baseline filter tests with BOTH slow and fast paths.

    Parameters
    ----------
    data : pd.DataFrame
        Bond panel data
    verbose : bool
        Whether to print progress
    validate : bool
        If True, run both slow and fast paths and compare
    tolerance : float
        Maximum allowed difference between slow and fast path

    Returns
    -------
    Tuple containing:
        - fast_results: dict mapping test name to FilterTestResult (fast path)
        - slow_results: dict mapping test name to FilterTestResult (slow path / source of truth)
        - all_passed: bool indicating if all tests passed validation
    """
    # Initialize Momentum strategy
    mom = pbl.Momentum(
        holding_period=HOLDING_PERIOD,
        lookback_period=LOOKBACK_PERIOD,
        skip=SKIP,
        num_portfolios=N_PORTFOLIOS,
        verbose=False
    )

    # Get filter parameters
    params_list = get_filter_params(mom)

    if verbose:
        print(f"\nRunning {len(params_list)} filter configurations...")
        if validate:
            print("Comparing SLOW path (source of truth) vs FAST path")
        print("=" * 70)

    fast_results = {}
    slow_results = {}
    validation_results = []
    total_slow_time = 0
    total_fast_time = 0

    for i, params in enumerate(params_list):
        filters = params.get('filters')
        if filters:
            filter_desc = f"{filters['adj']}={filters['level']}"
        else:
            filter_desc = "no filter"

        if verbose:
            print(f"\n[{i+1}/{len(params_list)}] {filter_desc}")

        unique_key = get_unique_key(params)

        if validate:
            # Run both paths and compare
            fast_result, slow_result, passed, diffs = run_single_filter_test_with_validation(
                data, params, tolerance=tolerance, verbose=verbose
            )
            fast_results[unique_key] = fast_result
            slow_results[unique_key] = slow_result
            total_fast_time += fast_result.runtime_seconds
            total_slow_time += slow_result.runtime_seconds

            # Report validation result
            max_diff = max(diffs.values())
            status = "✓ PASS" if passed else "✗ FAIL"
            if verbose:
                print(f"  {status} max_diff={max_diff:.2e} (tol={tolerance:.0e})")
                if not passed:
                    for metric, diff in diffs.items():
                        if diff > tolerance:
                            print(f"    {metric}: diff={diff:.2e}")
            validation_results.append((unique_key, passed, max_diff, diffs))
        else:
            # Just run fast path
            fast_result = run_single_filter_test(data, params, verbose=verbose, use_fast_path=True)
            fast_results[unique_key] = fast_result
            slow_results[unique_key] = fast_result  # Same as fast when not validating
            total_fast_time += fast_result.runtime_seconds
            validation_results.append((unique_key, True, 0.0, {}))

    # Summary
    all_passed = all(passed for _, passed, _, _ in validation_results)

    if verbose:
        print("\n" + "=" * 70)
        if validate:
            print(f"Total slow path time: {total_slow_time:.2f}s")
            print(f"Total fast path time: {total_fast_time:.2f}s")
            print(f"Speedup: {total_slow_time/total_fast_time:.1f}x")
        else:
            print(f"Total time: {total_fast_time:.2f}s")
        print(f"Average per filter: {total_fast_time/len(params_list):.2f}s")

        if validate:
            n_passed = sum(1 for _, passed, _, _ in validation_results if passed)
            n_failed = len(validation_results) - n_passed
            print(f"\nValidation: {n_passed}/{len(validation_results)} passed, {n_failed} failed")
            if not all_passed:
                print("\nFailed tests:")
                for name, passed, max_diff, diffs in validation_results:
                    if not passed:
                        print(f"  {name}: max_diff={max_diff:.2e}")

    return fast_results, slow_results, all_passed


# =============================================================================
# Save/Load Results
# =============================================================================

def save_baseline_results(results: Dict[str, FilterTestResult], output_dir: Path):
    """Save baseline results to JSON and pickle files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as JSON (human-readable)
    json_path = output_dir / "data_uncertainty_baseline.json"
    json_results = {name: r.to_dict() for name, r in results.items()}
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved JSON results to: {json_path}")

    # Save as pickle (preserves exact types)
    pkl_path = output_dir / "data_uncertainty_baseline.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved pickle results to: {pkl_path}")


def load_baseline_results(output_dir: Path) -> Dict[str, FilterTestResult]:
    """Load baseline results from pickle file."""
    pkl_path = output_dir / "data_uncertainty_baseline.pkl"
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


# =============================================================================
# Validation
# =============================================================================

def validate_against_baseline(
    current_results: Dict[str, FilterTestResult],
    baseline_results: Dict[str, FilterTestResult],
    tolerance: float = 1e-10
) -> bool:
    """
    Validate current results against baseline.

    Parameters
    ----------
    current_results : dict
        Current test results
    baseline_results : dict
        Baseline (source of truth) results
    tolerance : float
        Maximum allowed difference

    Returns
    -------
    bool
        True if all results match within tolerance
    """
    all_passed = True

    print("\nValidation Results:")
    print("=" * 70)

    for name, baseline in baseline_results.items():
        if name not in current_results:
            print(f"MISSING: {name}")
            all_passed = False
            continue

        current = current_results[name]

        # Check EA and EP means
        ea_diff = abs(current.ew_ea_mean - baseline.ew_ea_mean)
        ep_diff = abs(current.ew_ep_mean - baseline.ew_ep_mean)

        if ea_diff > tolerance or ep_diff > tolerance:
            print(f"FAIL: {name}")
            print(f"  EA diff: {ea_diff:.2e} (tol={tolerance:.0e})")
            print(f"  EP diff: {ep_diff:.2e} (tol={tolerance:.0e})")
            all_passed = False
        else:
            print(f"PASS: {name} (EA diff={ea_diff:.2e}, EP diff={ep_diff:.2e})")

    print("=" * 70)
    print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

    return all_passed


# =============================================================================
# Main
# =============================================================================

def main(validate: bool = True, tolerance: float = 1e-6):
    """
    Run baseline tests and save results.

    Parameters
    ----------
    validate : bool
        If True, run both slow and fast paths and compare
    tolerance : float
        Maximum allowed difference between slow and fast path
    """
    print("=" * 70)
    print("DATA UNCERTAINTY BASELINE TEST")
    print("=" * 70)

    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    t0 = time.time()
    data = generate_synthetic_data_fast(
        n_dates=N_DATES,
        n_bonds=N_BONDS,
        seed=RANDOM_SEED,
        n_chars=N_CHARS,
        balanced_panel=False,  # Realistic unbalanced panel
        allow_nans=True,       # Some missing data
    )
    print(f"   Shape: {data.shape}")
    print(f"   Time: {time.time()-t0:.2f}s")
    print(f"   Price range: {data['PRICE'].min():.1f} - {data['PRICE'].max():.1f}")
    print(f"   Return range: {data['ret'].min():.3f} - {data['ret'].max():.3f}")

    # Run baseline tests
    print("\n2. Running baseline tests...")
    fast_results, slow_results, all_passed = run_all_baseline_tests(
        data, verbose=True, validate=validate, tolerance=tolerance
    )

    # Print summary (using slow path as source of truth)
    print("\n3. Results Summary (SLOW PATH = Source of Truth)")
    print("=" * 70)
    print(f"{'Filter':<25} {'EA Mean':>12} {'EP Mean':>12} {'EA-EP Diff':>12} {'Time':>8}")
    print("-" * 70)
    for key, r in slow_results.items():
        ea_ep_diff = r.ew_ea_mean - r.ew_ep_mean
        print(f"{key:<25} {r.ew_ea_mean:>12.6f} {r.ew_ep_mean:>12.6f} {ea_ep_diff:>12.6f} {r.runtime_seconds:>7.2f}s")

    # Save slow path results as baseline (source of truth)
    print("\n4. Saving baseline results (slow path = source of truth)...")
    save_baseline_results(slow_results, RESULTS_DIR)

    # Final status
    print("\n" + "=" * 70)
    if validate:
        if all_passed:
            print("✓ ALL VALIDATION TESTS PASSED")
            print("Fast path matches slow path within tolerance")
        else:
            print("✗ SOME VALIDATION TESTS FAILED")
            print("Fast path does NOT match slow path - needs investigation")
    print("BASELINE TESTS COMPLETE")
    print("Slow path results saved as SOURCE OF TRUTH")
    print("=" * 70)

    return fast_results, slow_results, all_passed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run data uncertainty baseline tests")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip slow vs fast path validation (faster)")
    parser.add_argument("--tolerance", type=float, default=1e-6,
                        help="Tolerance for validation (default: 1e-6)")
    args = parser.parse_args()

    fast_results, slow_results, all_passed = main(
        validate=not args.no_validate,
        tolerance=args.tolerance
    )
