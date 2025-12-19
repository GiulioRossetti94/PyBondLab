#!/usr/bin/env python
"""
Validation script for Phase 15b: Non-staggered rebalancing with full features.

Tests:
1. Returns only (baseline from Phase 15a)
2. Returns + turnover
3. Returns + chars
4. Returns + banding
5. All features combined

Compares fast path vs slow path for numerical accuracy.
"""

import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, '.')

import PyBondLab as pbl
from PyBondLab.pbl_test import generate_synthetic_data_fast


def run_slow_path(data, signal, hp, rebal_freq, rebal_month, turnover, chars, banding):
    """Run slow path by disabling fast path."""
    strategy = pbl.SingleSort(
        holding_period=hp,
        sort_var=signal,
        num_portfolios=5,
        rebalance_frequency=rebal_freq,
        rebalance_month=rebal_month,
    )

    # Force slow path by using a filter (which disables fast path)
    # Alternative: we could add a flag to disable fast path explicitly
    # For now, we'll run with a dummy filter that doesn't change anything
    from PyBondLab.config import StrategyFormationConfig, DataConfig, FormationConfig

    config = StrategyFormationConfig(
        data=DataConfig(chars=chars),  # Pass chars through config
        formation=FormationConfig(
            compute_turnover=turnover,
            banding_threshold=banding,
            verbose=False,
        )
    )

    sf = pbl.StrategyFormation(
        data=data,
        strategy=strategy,
        config=config,
        verbose=False,
    )

    # Hack: temporarily disable fast path
    original_can_use = sf._can_use_nonstaggered_fast_path
    sf._can_use_nonstaggered_fast_path = lambda: False

    t0 = time.time()
    result = sf.fit()
    elapsed = time.time() - t0

    return result, elapsed


def run_fast_path(data, signal, hp, rebal_freq, rebal_month, turnover, chars, banding):
    """Run fast path."""
    strategy = pbl.SingleSort(
        holding_period=hp,
        sort_var=signal,
        num_portfolios=5,
        rebalance_frequency=rebal_freq,
        rebalance_month=rebal_month,
    )

    from PyBondLab.config import StrategyFormationConfig, DataConfig, FormationConfig

    config = StrategyFormationConfig(
        data=DataConfig(chars=chars),  # Pass chars through config
        formation=FormationConfig(
            compute_turnover=turnover,
            banding_threshold=banding,
            verbose=False,
        )
    )

    sf = pbl.StrategyFormation(
        data=data,
        strategy=strategy,
        config=config,
        verbose=False,
    )

    t0 = time.time()
    result = sf.fit()
    elapsed = time.time() - t0

    return result, elapsed


def compare_results(slow_result, fast_result, test_name, turnover=False, chars=None):
    """Compare slow vs fast path results."""
    issues = []

    # Compare long-short returns
    slow_ew, slow_vw = slow_result.get_long_short()
    fast_ew, fast_vw = fast_result.get_long_short()

    ew_diff = np.nanmax(np.abs(slow_ew.values - fast_ew.values))
    vw_diff = np.nanmax(np.abs(slow_vw.values - fast_vw.values))

    if ew_diff > 1e-10:
        issues.append(f"EW LS diff: {ew_diff:.2e}")
    if vw_diff > 1e-10:
        issues.append(f"VW LS diff: {vw_diff:.2e}")

    # Compare turnover if computed
    if turnover:
        slow_turn = slow_result.get_turnover()
        fast_turn = fast_result.get_turnover()

        if slow_turn is not None and fast_turn is not None:
            slow_ew_t, slow_vw_t = slow_turn
            fast_ew_t, fast_vw_t = fast_turn

            # Align indices
            common_idx = slow_ew_t.index.intersection(fast_ew_t.index)
            if len(common_idx) > 0:
                ew_t_diff = np.nanmax(np.abs(
                    slow_ew_t.loc[common_idx].values - fast_ew_t.loc[common_idx].values
                ))
                vw_t_diff = np.nanmax(np.abs(
                    slow_vw_t.loc[common_idx].values - fast_vw_t.loc[common_idx].values
                ))

                if ew_t_diff > 1e-8:
                    issues.append(f"EW turnover diff: {ew_t_diff:.2e}")
                if vw_t_diff > 1e-8:
                    issues.append(f"VW turnover diff: {vw_t_diff:.2e}")
        else:
            issues.append("Turnover result mismatch (None vs not None)")

    # Compare characteristics if computed
    if chars:
        slow_chars = slow_result.get_characteristics()
        fast_chars = fast_result.get_characteristics()

        if slow_chars is not None and fast_chars is not None:
            slow_ew_c, slow_vw_c = slow_chars
            fast_ew_c, fast_vw_c = fast_chars

            for c in chars:
                if c in slow_ew_c and c in fast_ew_c:
                    common_idx = slow_ew_c[c].index.intersection(fast_ew_c[c].index)
                    if len(common_idx) > 0:
                        ew_c_diff = np.nanmax(np.abs(
                            slow_ew_c[c].loc[common_idx].values - fast_ew_c[c].loc[common_idx].values
                        ))
                        vw_c_diff = np.nanmax(np.abs(
                            slow_vw_c[c].loc[common_idx].values - fast_vw_c[c].loc[common_idx].values
                        ))

                        if ew_c_diff > 1e-10:
                            issues.append(f"EW char {c} diff: {ew_c_diff:.2e}")
                        if vw_c_diff > 1e-10:
                            issues.append(f"VW char {c} diff: {vw_c_diff:.2e}")

    return issues


def main():
    print("=" * 80)
    print("PHASE 15b: Non-Staggered Rebalancing Full Features Validation")
    print("=" * 80)

    # Generate test data with characteristics
    print("\nGenerating test data...")
    data = generate_synthetic_data_fast(
        n_dates=120,
        n_bonds=300,
        seed=42,
        n_chars=3,
        balanced_panel=False,
    )
    print(f"  Data shape: {data.shape}")
    print(f"  Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"  Characteristics: char1, char2, char3")

    # Test configurations
    # Note: banding threshold is between 0 and 1 (e.g., 0.2 = 1/nport for 5 portfolios)
    test_configs = [
        # (name, hp, rebal_freq, rebal_month, turnover, chars, banding)
        ("returns_only", 12, 'annual', 6, False, None, None),
        ("with_turnover", 12, 'annual', 6, True, None, None),
        ("with_chars", 12, 'annual', 6, False, ['char1', 'char2'], None),
        ("with_banding", 12, 'annual', 6, False, None, 0.2),  # 1 portfolio = 0.2 for 5 portfolios
        ("turnover+chars", 6, 'semi-annual', 6, True, ['char1'], None),
        ("all_features", 3, 'quarterly', 6, True, ['char1', 'char2'], 0.2),
    ]

    results = []

    print("\n" + "-" * 80)
    print("Running validation tests...")
    print("-" * 80)

    for name, hp, rebal_freq, rebal_month, turnover, chars, banding in test_configs:
        print(f"\n[{name}] hp={hp}, freq={rebal_freq}, turnover={turnover}, chars={chars}, banding={banding}")

        try:
            # Run slow path
            slow_result, slow_time = run_slow_path(
                data, 'signal1', hp, rebal_freq, rebal_month, turnover, chars, banding
            )

            # Run fast path
            fast_result, fast_time = run_fast_path(
                data, 'signal1', hp, rebal_freq, rebal_month, turnover, chars, banding
            )

            # Compare results
            issues = compare_results(slow_result, fast_result, name, turnover, chars)

            speedup = slow_time / fast_time if fast_time > 0 else 0

            if issues:
                status = "FAIL"
                print(f"  FAIL: {', '.join(issues)}")
            else:
                status = "PASS"
                print(f"  PASS (slow={slow_time:.3f}s, fast={fast_time:.3f}s, speedup={speedup:.1f}x)")

            results.append({
                'name': name,
                'hp': hp,
                'freq': rebal_freq,
                'turnover': turnover,
                'chars': chars is not None,
                'banding': banding is not None,
                'slow_time': slow_time,
                'fast_time': fast_time,
                'speedup': speedup,
                'status': status,
                'issues': issues,
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'name': name,
                'hp': hp,
                'freq': rebal_freq,
                'turnover': turnover,
                'chars': chars is not None,
                'banding': banding is not None,
                'slow_time': 0,
                'fast_time': 0,
                'speedup': 0,
                'status': 'ERROR',
                'issues': [str(e)],
            })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    df = pd.DataFrame(results)
    print(df[['name', 'freq', 'turnover', 'chars', 'banding', 'slow_time', 'fast_time', 'speedup', 'status']].to_string(index=False))

    passed = sum(1 for r in results if r['status'] == 'PASS')
    total = len(results)
    print(f"\n{passed}/{total} tests PASSED")

    return results


if __name__ == '__main__':
    main()
