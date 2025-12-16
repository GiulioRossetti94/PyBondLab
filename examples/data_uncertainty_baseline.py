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
    verbose: bool = True
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

    Returns
    -------
    FilterTestResult
        Results container with EA and EP returns
    """
    t0 = time.time()

    # Make a copy of data for this test
    data_copy = data.copy()

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
        print(f"  {result.name}: EA={ew_ea.mean():.6f}, EP={ew_ep.mean():.6f}, time={elapsed:.2f}s")

    return test_result


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
    verbose: bool = True
) -> Dict[str, FilterTestResult]:
    """
    Run all baseline filter tests.

    Parameters
    ----------
    data : pd.DataFrame
        Bond panel data
    verbose : bool
        Whether to print progress

    Returns
    -------
    dict
        Dictionary mapping test name to FilterTestResult
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
        print("=" * 70)

    results = {}
    total_time = 0

    for i, params in enumerate(params_list):
        if verbose:
            filters = params.get('filters')
            if filters:
                filter_desc = f"{filters['adj']}={filters['level']}"
            else:
                filter_desc = "no filter"
            print(f"\n[{i+1}/{len(params_list)}] {filter_desc}")

        result = run_single_filter_test(data, params, verbose=verbose)

        # Use unique key based on filter config (not strategy name which is the same for all)
        unique_key = get_unique_key(params)
        results[unique_key] = result
        total_time += result.runtime_seconds

    if verbose:
        print("=" * 70)
        print(f"Total time: {total_time:.2f}s")
        print(f"Average per filter: {total_time/len(params_list):.2f}s")

    return results


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

def main():
    """Run baseline tests and save results."""
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
    results = run_all_baseline_tests(data, verbose=True)

    # Print summary
    print("\n3. Results Summary")
    print("=" * 70)
    print(f"{'Filter':<25} {'EA Mean':>12} {'EP Mean':>12} {'EA-EP Diff':>12} {'Time':>8}")
    print("-" * 70)
    for key, r in results.items():
        ea_ep_diff = r.ew_ea_mean - r.ew_ep_mean
        print(f"{key:<25} {r.ew_ea_mean:>12.6f} {r.ew_ep_mean:>12.6f} {ea_ep_diff:>12.6f} {r.runtime_seconds:>7.2f}s")

    # Save results
    print("\n4. Saving baseline results...")
    save_baseline_results(results, RESULTS_DIR)

    print("\n" + "=" * 70)
    print("BASELINE TESTS COMPLETE")
    print("These results are the SOURCE OF TRUTH for optimization validation")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = main()
