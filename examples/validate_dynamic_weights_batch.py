#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validation script for dynamic_weights parameter in BatchStrategyFormation.

This script validates:
1. Test E: HP=1 - dynamic_weights=True and False produce IDENTICAL results
2. Test F: HP=3 - dynamic_weights=True and False produce DIFFERENT results (VW only)
3. Test A/B: Fast path matches slow path for dynamic_weights=True
4. Test C/D: Fast path matches slow path for dynamic_weights=False (when implemented)

Usage:
    python examples/validate_dynamic_weights_batch.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import time

from PyBondLab import BatchStrategyFormation
from PyBondLab.pbl_test import generate_synthetic_data


def compare_results(result1, result2, name1, name2, tolerance=1e-10):
    """Compare two batch results and return max difference."""
    signals = list(result1.results.keys())

    max_ew_diff = 0.0
    max_vw_diff = 0.0

    for signal in signals:
        r1 = result1.results[signal]
        r2 = result2.results[signal]

        ew1, vw1 = r1.get_long_short()
        ew2, vw2 = r2.get_long_short()

        # Align indices
        common_idx = ew1.index.intersection(ew2.index)

        # Use nanmax to handle NaN values properly
        ew_diff = np.nanmax(np.abs(ew1.loc[common_idx].values - ew2.loc[common_idx].values))
        vw_diff = np.nanmax(np.abs(vw1.loc[common_idx].values - vw2.loc[common_idx].values))

        # Handle case where all values are NaN
        if np.isnan(ew_diff):
            ew_diff = 0.0
        if np.isnan(vw_diff):
            vw_diff = 0.0

        max_ew_diff = max(max_ew_diff, ew_diff)
        max_vw_diff = max(max_vw_diff, vw_diff)

    return max_ew_diff, max_vw_diff


def run_validation():
    """Run all validation tests."""
    print("=" * 70)
    print("VALIDATION: dynamic_weights parameter in BatchStrategyFormation")
    print("=" * 70)
    print()

    # Generate synthetic data
    print("Generating synthetic data (60 dates, 300 bonds)...")
    data = generate_synthetic_data(n_dates=60, n_bonds=300, seed=42)
    signals = ['signal1', 'signal2']  # Use available signals

    print(f"  Data shape: {data.shape}")
    print(f"  Signals: {signals}")
    print()

    results = {}

    # =========================================================================
    # Test E & F: Compare True vs False for HP=1 and HP=3
    # =========================================================================
    print("-" * 70)
    print("TEST E & F: Compare dynamic_weights=True vs False")
    print("-" * 70)

    for hp in [1, 3]:
        print(f"\n  HP={hp}:")

        for dw in [True, False]:
            key = f"hp{hp}_dw{dw}"
            print(f"    Running dynamic_weights={dw}...", end=" ")

            # Force slow path by using turnover=True
            t0 = time.time()
            batch = BatchStrategyFormation(
                data=data,
                signals=signals,
                holding_period=hp,
                num_portfolios=5,
                turnover=True,  # Forces slow path
                dynamic_weights=dw,
                n_jobs=1,
                verbose=False
            )
            result = batch.fit()
            elapsed = time.time() - t0

            results[key] = result
            print(f"Done in {elapsed:.2f}s")

        # Compare True vs False
        key_true = f"hp{hp}_dwTrue"
        key_false = f"hp{hp}_dwFalse"

        ew_diff, vw_diff = compare_results(
            results[key_true], results[key_false],
            "True", "False"
        )

        if hp == 1:
            # HP=1: Should be IDENTICAL
            status = "PASS" if ew_diff < 1e-10 and vw_diff < 1e-10 else "FAIL"
            print(f"\n    Test E (HP=1 should be identical):")
            print(f"      EW diff: {ew_diff:.2e} (expected: 0)")
            print(f"      VW diff: {vw_diff:.2e} (expected: 0)")
            print(f"      Status: {status}")
        else:
            # HP=3: Should be DIFFERENT (at least VW)
            # EW might differ slightly due to ID intersection differences
            status = "PASS" if vw_diff > 1e-6 else "FAIL"
            print(f"\n    Test F (HP=3 should be different for VW):")
            print(f"      EW diff: {ew_diff:.2e}")
            print(f"      VW diff: {vw_diff:.2e} (expected: > 1e-6)")
            print(f"      Status: {status}")

    # =========================================================================
    # Test A & B: Fast path vs slow path for dynamic_weights=True
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST A & B: Fast path vs Slow path (dynamic_weights=True)")
    print("-" * 70)

    for hp in [1, 3]:
        print(f"\n  HP={hp}:")

        # Slow path (turnover=True)
        print("    Running slow path (turnover=True)...", end=" ")
        t0 = time.time()
        slow = BatchStrategyFormation(
            data=data,
            signals=signals,
            holding_period=hp,
            num_portfolios=5,
            turnover=True,  # Forces slow path
            dynamic_weights=True,
            n_jobs=1,
            verbose=False
        )
        slow_result = slow.fit()
        print(f"Done in {time.time() - t0:.2f}s")

        # Fast path (turnover=False)
        print("    Running fast path (turnover=False)...", end=" ")
        t0 = time.time()
        fast = BatchStrategyFormation(
            data=data,
            signals=signals,
            holding_period=hp,
            num_portfolios=5,
            turnover=False,  # Allows fast path
            dynamic_weights=True,
            n_jobs=1,
            verbose=False
        )
        fast_result = fast.fit()
        print(f"Done in {time.time() - t0:.2f}s")

        # Compare
        ew_diff, vw_diff = compare_results(slow_result, fast_result, "slow", "fast")

        status = "PASS" if ew_diff < 1e-10 and vw_diff < 1e-10 else "FAIL"
        test_name = "A" if hp == 1 else "B"
        print(f"\n    Test {test_name} (fast vs slow, HP={hp}):")
        print(f"      EW diff: {ew_diff:.2e}")
        print(f"      VW diff: {vw_diff:.2e}")
        print(f"      Status: {status}")

    # =========================================================================
    # Test C & D: Fast path vs slow path for dynamic_weights=False
    # =========================================================================
    print()
    print("-" * 70)
    print("TEST C & D: Fast path vs Slow path (dynamic_weights=False)")
    print("-" * 70)

    for hp in [1, 3]:
        print(f"\n  HP={hp}:")

        # Slow path (turnover=True)
        print("    Running slow path (turnover=True)...", end=" ")
        t0 = time.time()
        slow = BatchStrategyFormation(
            data=data,
            signals=signals,
            holding_period=hp,
            num_portfolios=5,
            turnover=True,  # Forces slow path
            dynamic_weights=False,
            n_jobs=1,
            verbose=False
        )
        slow_result = slow.fit()
        print(f"Done in {time.time() - t0:.2f}s")

        # Fast path (turnover=False)
        # Note: For HP>1 with dynamic_weights=False, fast path is currently disabled
        print("    Running fast path (turnover=False)...", end=" ")
        t0 = time.time()
        fast = BatchStrategyFormation(
            data=data,
            signals=signals,
            holding_period=hp,
            num_portfolios=5,
            turnover=False,  # Would use fast path if enabled
            dynamic_weights=False,
            n_jobs=1,
            verbose=False
        )
        fast_result = fast.fit()
        print(f"Done in {time.time() - t0:.2f}s")

        # Compare
        ew_diff, vw_diff = compare_results(slow_result, fast_result, "slow", "fast")

        status = "PASS" if ew_diff < 1e-10 and vw_diff < 1e-10 else "FAIL"
        test_name = "C" if hp == 1 else "D"

        # Now using fast path with v2 kernel for HP>1 with dynamic_weights=False
        print(f"\n    Test {test_name} (fast vs slow, HP={hp}, dynamic_weights=False):")
        print(f"      EW diff: {ew_diff:.2e}")
        print(f"      VW diff: {vw_diff:.2e}")
        print(f"      Status: {status}")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print()
    print("Expected results:")
    print("  Test E (HP=1, True vs False): IDENTICAL (both settings same for HP=1)")
    print("  Test F (HP=3, True vs False): DIFFERENT (VW differs for HP>1)")
    print("  Test A (HP=1, fast vs slow, dw=True): MATCH")
    print("  Test B (HP=3, fast vs slow, dw=True): MATCH")
    print("  Test C (HP=1, fast vs slow, dw=False): MATCH")
    print("  Test D (HP=3, fast vs slow, dw=False): MATCH (uses fast path v2 kernel)")
    print()


if __name__ == "__main__":
    run_validation()
