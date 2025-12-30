#!/usr/bin/env python3
"""
Diagnostic script for DataUncertaintyAnalysis.

Tests all features and evaluates UX:
1. All filter types (baseline, trim, price, bounce, wins)
2. Multiple signals
3. Multiple holding periods
4. Rating filtering (IG, NIG, tuple)
5. Summary output
6. Filter method for subsetting
7. Fast vs slow path comparison

Author: Claude
"""

import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, '.')

from PyBondLab import DataUncertaintyAnalysis, Momentum
from PyBondLab.pbl_test import generate_synthetic_data


def section(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def subsection(title: str):
    """Print a subsection header."""
    print(f"\n--- {title} ---")


def test_basic_usage():
    """Test 1: Basic usage with pre-computed signal."""
    section("TEST 1: Basic Usage with Pre-Computed Signal")

    # Generate synthetic data with a signal column
    data = generate_synthetic_data(n_dates=60, n_bonds=300, seed=42)
    data['signal1'] = np.random.randn(len(data))

    print(f"Data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")

    # Run analysis with default settings
    t0 = time.time()
    results = DataUncertaintyAnalysis(
        data=data,
        signals=['signal1'],
        holding_periods=[1, 3],
        filters={'trim': [0.2]},
        include_baseline=True,
        verbose=True
    ).fit()
    elapsed = time.time() - t0

    print(f"\nTotal time: {elapsed:.2f}s")

    # Check results structure
    subsection("Results Structure")
    print(f"ew_ea shape: {results.ew_ea.shape}")
    print(f"vw_ea shape: {results.vw_ea.shape}")
    print(f"ew_ep shape: {results.ew_ep.shape}")
    print(f"vw_ep shape: {results.vw_ep.shape}")
    print(f"configs shape: {results.configs.shape}")

    print(f"\nColumn names: {list(results.ew_ea.columns)}")
    print(f"\nConfigs:\n{results.configs}")

    return results


def test_all_filter_types():
    """Test 2: All filter types."""
    section("TEST 2: All Filter Types")

    data = generate_synthetic_data(n_dates=60, n_bonds=300, seed=42)
    data['signal1'] = np.random.randn(len(data))
    # Add PRICE column for price filter
    data['PRICE'] = np.random.uniform(50, 200, len(data))

    # Comprehensive filter specification
    filters = {
        'trim': [0.2, 0.5, -0.3, [-0.2, 0.3]],  # right, right, left, both
        'price': [[5, 10], [150, 200]],  # left levels, right levels
        'bounce': [0.05, -0.05],  # right, left
        'wins': [(99, 'both'), (95, 'both'), (99, 'right')],
    }

    t0 = time.time()
    results = DataUncertaintyAnalysis(
        data=data,
        signals=['signal1'],
        holding_periods=[1],
        filters=filters,
        include_baseline=True,
        verbose=True
    ).fit()
    elapsed = time.time() - t0

    print(f"\nTotal time: {elapsed:.2f}s")
    print(f"\nTotal configurations: {len(results.configs)}")

    # Show all filter configurations
    subsection("All Filter Configurations")
    print(results.configs[['column_name', 'filter_type', 'level', 'location']])

    return results


def test_summary_output():
    """Test 3: Summary output and statistics."""
    section("TEST 3: Summary Output")

    data = generate_synthetic_data(n_dates=120, n_bonds=300, seed=42)
    data['signal1'] = np.random.randn(len(data))
    data['signal2'] = np.random.randn(len(data))

    results = DataUncertaintyAnalysis(
        data=data,
        signals=['signal1', 'signal2'],
        holding_periods=[1, 3],
        filters={'trim': [0.2, 0.5]},
        include_baseline=True,
        verbose=False
    ).fit()

    # Test summary() method
    subsection("Summary Statistics")
    summary = results.summary()
    print(f"Summary shape: {summary.shape}")
    print(f"Summary columns: {list(summary.columns)}")
    print("\nSummary table:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(summary)

    return results


def test_filter_method():
    """Test 4: Filter method for subsetting results."""
    section("TEST 4: Filter Method")

    data = generate_synthetic_data(n_dates=60, n_bonds=300, seed=42)
    data['signal1'] = np.random.randn(len(data))
    data['signal2'] = np.random.randn(len(data))

    results = DataUncertaintyAnalysis(
        data=data,
        signals=['signal1', 'signal2'],
        holding_periods=[1, 3],
        filters={'trim': [0.2, 0.5], 'wins': [(99, 'both')]},
        include_baseline=True,
        verbose=False
    ).fit()

    print(f"Total configurations: {len(results.configs)}")

    # Test different filter criteria
    subsection("Filter by signal")
    signal1_results = results.filter(signal='signal1')
    print(f"signal1 configs: {len(signal1_results.configs)}")

    subsection("Filter by holding period")
    hp1_results = results.filter(hp=1)
    print(f"hp=1 configs: {len(hp1_results.configs)}")

    subsection("Filter by filter_type")
    trim_results = results.filter(filter_type='trim')
    print(f"trim configs: {len(trim_results.configs)}")
    print(trim_results.configs[['column_name', 'filter_type', 'level']])

    subsection("Filter by multiple criteria")
    specific = results.filter(signal='signal1', hp=1, filter_type='baseline')
    print(f"signal1, hp=1, baseline configs: {len(specific.configs)}")
    print(specific.configs)

    return results


def test_rating_filtering():
    """Test 5: Rating filtering."""
    section("TEST 5: Rating Filtering")

    data = generate_synthetic_data(n_dates=60, n_bonds=300, seed=42)
    data['signal1'] = np.random.randn(len(data))

    subsection("Single rating (IG)")
    results_ig = DataUncertaintyAnalysis(
        data=data,
        signals=['signal1'],
        holding_periods=[1],
        filters={'trim': [0.2]},
        rating='IG',
        verbose=False
    ).fit()
    print(f"IG configs: {len(results_ig.configs)}")
    print(results_ig.configs[['column_name', 'rating']])

    subsection("Multiple ratings (ratings parameter)")
    results_multi = DataUncertaintyAnalysis(
        data=data,
        signals=['signal1'],
        holding_periods=[1],
        filters={'trim': [0.2]},
        ratings=['IG', 'NIG', None],
        verbose=False
    ).fit()
    print(f"Multiple ratings configs: {len(results_multi.configs)}")
    print(results_multi.configs[['column_name', 'rating']])

    subsection("Rating tuple (custom range)")
    results_tuple = DataUncertaintyAnalysis(
        data=data,
        signals=['signal1'],
        holding_periods=[1],
        filters={'trim': [0.2]},
        rating=(1, 10),  # Same as IG
        verbose=False
    ).fit()
    print(f"Rating tuple configs: {len(results_tuple.configs)}")
    print(results_tuple.configs[['column_name', 'rating']])

    return results_multi


def test_strategy_mode():
    """Test 6: Strategy mode (Momentum)."""
    section("TEST 6: Strategy Mode (Momentum)")

    data = generate_synthetic_data(n_dates=60, n_bonds=300, seed=42)

    # Momentum strategy
    mom = Momentum(lookback_period=3, skip=1)

    t0 = time.time()
    results = DataUncertaintyAnalysis(
        data=data,
        strategy=mom,
        holding_periods=[1],
        filters={'trim': [0.2], 'wins': [(99, 'both')]},
        include_baseline=True,
        verbose=True
    ).fit()
    elapsed = time.time() - t0

    print(f"\nTotal time: {elapsed:.2f}s")
    print(f"Configs: {len(results.configs)}")
    print(results.configs[['column_name', 'signal', 'filter_type']])

    return results


def test_fast_vs_slow_path():
    """Test 7: Compare fast vs slow path."""
    section("TEST 7: Fast vs Slow Path Comparison")

    data = generate_synthetic_data(n_dates=60, n_bonds=300, seed=42)
    data['signal1'] = np.random.randn(len(data))
    data['PRICE'] = np.random.uniform(50, 200, len(data))

    filters = {
        'trim': [0.2],
        'price': [[5], [150]],
        'bounce': [0.05],
    }

    subsection("Fast Path")
    t0 = time.time()
    results_fast = DataUncertaintyAnalysis(
        data=data,
        signals=['signal1'],
        holding_periods=[1, 3],
        filters=filters,
        use_fast_path=True,
        verbose=True
    ).fit()
    fast_time = time.time() - t0

    subsection("Slow Path")
    t0 = time.time()
    results_slow = DataUncertaintyAnalysis(
        data=data,
        signals=['signal1'],
        holding_periods=[1, 3],
        filters=filters,
        use_fast_path=False,
        verbose=True
    ).fit()
    slow_time = time.time() - t0

    subsection("Comparison")
    print(f"Fast path: {fast_time:.2f}s")
    print(f"Slow path: {slow_time:.2f}s")
    print(f"Speedup: {slow_time/fast_time:.1f}x")

    # Compare results
    subsection("Numerical Comparison")
    common_cols = sorted(set(results_fast.ew_ea.columns) & set(results_slow.ew_ea.columns))

    for col in common_cols[:5]:  # First 5 configs
        fast_vals = results_fast.ew_ea[col].dropna()
        slow_vals = results_slow.ew_ea[col].dropna()

        # Align indices
        common_idx = fast_vals.index.intersection(slow_vals.index)
        if len(common_idx) > 0:
            diff = np.abs(fast_vals.loc[common_idx] - slow_vals.loc[common_idx]).max()
            print(f"{col}: max diff = {diff:.2e}")

    return results_fast, results_slow


def test_ux_issues():
    """Test 8: Identify UX issues."""
    section("TEST 8: UX Issues and Improvements")

    data = generate_synthetic_data(n_dates=60, n_bonds=300, seed=42)
    data['signal1'] = np.random.randn(len(data))

    issues = []

    # Issue 1: No average across filter types in summary
    subsection("Issue 1: Summary doesn't show averages across filter types")
    results = DataUncertaintyAnalysis(
        data=data,
        signals=['signal1'],
        holding_periods=[1],
        filters={'trim': [0.2, 0.3, 0.5]},
        verbose=False
    ).fit()

    summary = results.summary()
    print("Current summary output:")
    print(summary[['signal', 'hp', 'filter_type', 'level', 'ew_ea_mean', 'ew_ea_tstat']])
    print("\n=> Missing: Average across all trim levels, average by filter_type")
    issues.append("No summary aggregation by filter_type")

    # Issue 2: Column naming convention
    subsection("Issue 2: Column names are verbose")
    print(f"Column names: {list(results.ew_ea.columns)}")
    print("\n=> 'signal1_hp1_baseline' is clear but long")
    issues.append("Column names could be shorter/configurable")

    # Issue 3: No easy way to get all EA vs EP results
    subsection("Issue 3: Accessing EA vs EP")
    print("Current access pattern:")
    print("  results.ew_ea   # EW Ex-Ante")
    print("  results.vw_ea   # VW Ex-Ante")
    print("  results.ew_ep   # EW Ex-Post")
    print("  results.vw_ep   # VW Ex-Post")
    print("\n=> Would be nice to have results.get_panel('ew', 'ea') or similar")
    issues.append("No unified panel access method")

    # Issue 4: Summary doesn't show EA vs EP comparison
    subsection("Issue 4: EA vs EP comparison in summary")
    print("Summary has ew_ea_mean, vw_ea_mean, ew_ep_mean, vw_ep_mean")
    print("=> No EA-EP spread or ratio in summary")
    issues.append("Summary doesn't compute EA-EP spread")

    # Issue 5: No export to wide format
    subsection("Issue 5: Export format")
    print("to_excel() exports as-is")
    print("=> Would be nice to have to_excel(format='wide') or to_excel(format='long')")
    issues.append("Export format not configurable")

    # Issue 6: Verbose output during fit
    subsection("Issue 6: Verbose output formatting")
    print("Current verbose output is informative but could be cleaner")
    print("=> Progress bars with tqdm would be better for large runs")
    issues.append("No progress bars for large analyses")

    print("\n" + "=" * 50)
    print("IDENTIFIED UX ISSUES:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")

    return issues


def test_aggregated_summary():
    """Test 9: Evaluate aggregated summaries with NEW features."""
    section("TEST 9: Aggregated Summary (NEW FEATURES)")

    data = generate_synthetic_data(n_dates=120, n_bonds=300, seed=42)
    data['signal1'] = np.random.randn(len(data))
    data['signal2'] = np.random.randn(len(data))

    results = DataUncertaintyAnalysis(
        data=data,
        signals=['signal1', 'signal2'],
        holding_periods=[1, 3],
        filters={'trim': [0.2, 0.3, 0.5], 'wins': [(99, 'both')]},
        verbose=False
    ).fit()

    subsection("NEW: EA-EP diff columns in summary")
    summary = results.summary()
    print("New columns: ea_ep_diff_ew, ea_ep_diff_vw")
    print(summary[['filter_type', 'ew_ea_mean', 'ew_ep_mean', 'ea_ep_diff_ew']].head(4))

    subsection("NEW: aggregate_by parameter")
    by_filter = results.summary(aggregate_by='filter_type')
    print("summary(aggregate_by='filter_type'):")
    print(by_filter[['filter_type', 'ew_ea_mean', 'vw_ea_mean', 'sharpe']])

    subsection("NEW: average_by_filter() method")
    avg = results.average_by_filter()
    print("average_by_filter():")
    print(avg[['signal', 'hp', 'filter_type', 'ew_ea_mean', 'vw_ea_mean']])

    subsection("Baseline vs Filter comparison (by signal, hp)")
    # Compare baseline to filtered using NEW method
    baseline = summary[summary['filter_type'] == 'baseline'].set_index(['signal', 'hp'])
    filtered = summary[summary['filter_type'] != 'baseline']

    print("\nBaseline returns:")
    print(baseline[['ew_ea_mean', 'vw_ea_mean']])

    print("\nFiltered returns (mean across filter levels):")
    filtered_avg = filtered.groupby(['signal', 'hp', 'filter_type'])[['ew_ea_mean', 'vw_ea_mean']].mean()
    print(filtered_avg)

    return results


def main():
    """Run all diagnostic tests."""
    print("=" * 70)
    print(" DataUncertaintyAnalysis Diagnostic Tests")
    print("=" * 70)

    tests = [
        ("Basic Usage", test_basic_usage),
        ("All Filter Types", test_all_filter_types),
        ("Summary Output", test_summary_output),
        ("Filter Method", test_filter_method),
        ("Rating Filtering", test_rating_filtering),
        ("Strategy Mode", test_strategy_mode),
        ("Fast vs Slow Path", test_fast_vs_slow_path),
        ("UX Issues", test_ux_issues),
        ("Aggregated Summary", test_aggregated_summary),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
            print(f"\n[PASS] {name}")
        except Exception as e:
            print(f"\n[FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = None

    section("SUMMARY")
    for name in tests:
        status = "PASS" if results.get(name[0]) is not None else "FAIL"
        print(f"  [{status}] {name[0]}")

    return results


if __name__ == "__main__":
    main()
