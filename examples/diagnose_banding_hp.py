#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diagnostic script for banding with HP > 1 (staggered rebalancing).

Key Question: Is banding reducing turnover as expected for HP > 1?

With 10 portfolios and banding_threshold=0.20:
- threshold_portfolios = 0.20 * 10 = 2 portfolios
- A bond must move by 2+ portfolio ranks to be reassigned
- This should REDUCE turnover compared to no banding

Test approach:
1. Create balanced panel with LOW-VOLATILITY signal (gradual changes)
2. Compare turnover WITH and WITHOUT banding
3. Turnover WITH banding MUST be <= turnover WITHOUT banding
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/user/PyBondLab-Dev')

import PyBondLab as pbl
from PyBondLab import StrategyFormation, SingleSort


def create_low_volatility_panel(n_dates: int = 24, n_bonds: int = 100, seed: int = 42):
    """
    Create a balanced panel with LOW-VOLATILITY signal.

    Signal changes GRADUALLY so that banding can take effect:
    - Most bonds move by 0-1 portfolio ranks per period
    - Some bonds move by 2+ ranks (should trigger reassignment with banding=0.2)
    """
    np.random.seed(seed)

    dates = pd.date_range('2020-01-31', periods=n_dates, freq='ME')

    records = []

    # Base signal: bond's "true" percentile in the signal distribution
    base_signal = np.linspace(0, 100, n_bonds)  # 0 to 100 for 100 bonds

    for t_idx, date in enumerate(dates):
        for bond_idx in range(n_bonds):
            bond_id = f"BOND_{bond_idx:04d}"

            # Signal = base + small random walk
            # Standard deviation of ~5 means most moves are within 1 portfolio (10% width)
            # But ~5% of moves will exceed 2 portfolios (20%)
            noise = np.random.normal(0, 5)  # Small noise
            signal = base_signal[bond_idx] + noise

            # Add a small drift over time (to ensure some changes)
            # This slowly shuffles the rankings
            drift = t_idx * 0.5 * np.random.choice([-1, 1])
            signal = signal + drift

            ret = np.random.uniform(-0.02, 0.02)
            vw = 1.0

            records.append({
                'date': date,
                'ID': bond_id,
                'signal': signal,
                'ret': ret,
                'VW': vw,
                'RATING_NUM': 5,
            })

    df = pd.DataFrame(records)
    print(f"Created low-volatility panel: {len(df)} rows, {n_bonds} bonds, {n_dates} dates")

    return df


def create_high_volatility_panel(n_dates: int = 24, n_bonds: int = 100, seed: int = 42, hp: int = 1):
    """
    Create a balanced panel with HIGH-VOLATILITY signal.

    Signal changes DRAMATICALLY each HP months to maximize turnover.
    This is used as a control to verify banding DOES reduce turnover.
    """
    np.random.seed(seed)

    dates = pd.date_range('2020-01-31', periods=n_dates, freq='ME')

    records = []

    for t_idx, date in enumerate(dates):
        for bond_idx in range(n_bonds):
            bond_id = f"BOND_{bond_idx:04d}"

            # High volatility: complete reversal every HP months
            phase = (t_idx // hp) % 2
            if phase == 0:
                signal = bond_idx
            else:
                signal = n_bonds - 1 - bond_idx

            signal = signal + np.random.uniform(0, 0.1)
            ret = np.random.uniform(-0.02, 0.02)
            vw = 1.0

            records.append({
                'date': date,
                'ID': bond_id,
                'signal': signal,
                'ret': ret,
                'VW': vw,
                'RATING_NUM': 5,
            })

    df = pd.DataFrame(records)
    print(f"Created high-volatility panel: {len(df)} rows, {n_bonds} bonds, {n_dates} dates")

    return df


def run_turnover_comparison(data: pd.DataFrame, hp: int, num_portfolios: int = 10,
                            banding_threshold: float = 0.20):
    """
    Run StrategyFormation with and without banding, compare turnover.
    """
    print(f"\n{'='*60}")
    print(f"HP={hp}, num_portfolios={num_portfolios}, banding_threshold={banding_threshold}")
    print(f"{'='*60}")

    # WITHOUT banding
    strategy_no_band = SingleSort(
        holding_period=hp,
        sort_var='signal',
        num_portfolios=num_portfolios
    )

    sf_no_band = StrategyFormation(
        data=data,
        strategy=strategy_no_band,
        turnover=True,
        verbose=False
    )

    result_no_band = sf_no_band.fit()
    ew_no_band, vw_no_band = result_no_band.get_turnover()

    # WITH banding
    strategy_band = SingleSort(
        holding_period=hp,
        sort_var='signal',
        num_portfolios=num_portfolios
    )

    sf_band = StrategyFormation(
        data=data,
        strategy=strategy_band,
        turnover=True,
        banding_threshold=banding_threshold,
        verbose=False
    )

    result_band = sf_band.fit()
    ew_band, vw_band = result_band.get_turnover()

    # Compare
    mean_no_band = ew_no_band.mean().mean()
    mean_band = ew_band.mean().mean()
    reduction = (1 - mean_band / mean_no_band) * 100 if mean_no_band > 0 else 0

    print(f"\nTurnover Comparison (EW mean):")
    print(f"  WITHOUT banding: {mean_no_band:.4f}")
    print(f"  WITH banding:    {mean_band:.4f}")
    print(f"  Reduction:       {reduction:.1f}%")

    # Check the invariant: banding should NOT increase turnover
    if mean_band > mean_no_band + 1e-10:
        print(f"\n  *** WARNING: Banding INCREASED turnover! This is a BUG! ***")
    else:
        print(f"\n  OK: Banding reduced turnover as expected.")

    return {
        'no_band': mean_no_band,
        'band': mean_band,
        'reduction': reduction,
        'ew_no_band': ew_no_band,
        'ew_band': ew_band,
        'sf_no_band': sf_no_band,
        'sf_band': sf_band,
    }


def inspect_banding_state(sf):
    """Inspect the lag_rank state to understand banding behavior."""
    print(f"\n{'='*60}")
    print("BANDING STATE INSPECTION")
    print(f"{'='*60}")

    if not hasattr(sf, 'lag_rank') or sf.lag_rank is None:
        print("No banding state found (banding not enabled or not used)")
        return

    print(f"\nlag_rank keys: {list(sf.lag_rank.keys())}")

    for key, df in sf.lag_rank.items():
        print(f"\n  Cohort {key}: {len(df)} bonds")
        if not df.empty:
            rank_dist = df['ptf_rank'].value_counts().sort_index()
            print(f"    Rank distribution: {dict(rank_dist)}")


def trace_banding_for_cohort(data: pd.DataFrame, hp: int, cohort: int = 0,
                             num_portfolios: int = 10, banding_threshold: float = 0.20):
    """
    Trace banding behavior for a specific cohort step by step.
    """
    print(f"\n{'='*60}")
    print(f"TRACING BANDING FOR COHORT {cohort} (HP={hp})")
    print(f"{'='*60}")

    # Rebalancing times for this cohort
    dates = sorted(data['date'].unique())
    rebal_times = [t for t in range(len(dates)) if t % hp == cohort]

    print(f"\nCohort {cohort} rebalances at t = {rebal_times[:8]}...")

    # Get bonds that are present at all times
    common_bonds = set(data[data['date'] == dates[0]]['ID'].unique())
    for date in dates:
        common_bonds &= set(data[data['date'] == date]['ID'].unique())

    # Pick a few bonds to trace
    trace_bonds = sorted(list(common_bonds))[:5]
    print(f"\nTracing bonds: {trace_bonds}")

    # Get signal and rank for these bonds at rebalancing times
    print(f"\nSignal and rank at rebalancing times:")
    print(f"{'Bond':<12} ", end="")
    for t in rebal_times[:6]:
        print(f"t={t:<6}", end="")
    print()

    # First, compute ranks without banding
    for bond in trace_bonds:
        bond_data = data[data['ID'] == bond].sort_values('date')

        signals = []
        for t in rebal_times[:6]:
            if t < len(dates):
                date = dates[t]
                row = bond_data[bond_data['date'] == date]
                if not row.empty:
                    signals.append(row['signal'].values[0])
                else:
                    signals.append(np.nan)
            else:
                signals.append(np.nan)

        print(f"{bond:<12} ", end="")
        for s in signals:
            if np.isnan(s):
                print(f"{'NaN':<7}", end="")
            else:
                print(f"{s:>6.1f} ", end="")
        print()


def test_banding_invariant():
    """
    Test the key invariant: Banding should NEVER increase turnover.

    Run multiple configurations and verify this holds.
    """
    print("\n" + "=" * 70)
    print("TESTING BANDING INVARIANT: Turnover(with banding) <= Turnover(no banding)")
    print("=" * 70)

    results = []

    # Test with low volatility signal (gradual changes)
    print("\n--- LOW VOLATILITY SIGNAL ---")
    data_low = create_low_volatility_panel(n_dates=24, n_bonds=100, seed=42)

    for hp in [1, 2, 3]:
        r = run_turnover_comparison(data_low, hp=hp, num_portfolios=10, banding_threshold=0.20)
        results.append({
            'signal': 'low_vol',
            'hp': hp,
            **r
        })

    # Test with high volatility signal (complete reversals)
    print("\n--- HIGH VOLATILITY SIGNAL ---")
    for hp in [1, 3]:
        data_high = create_high_volatility_panel(n_dates=24, n_bonds=100, seed=42, hp=hp)
        r = run_turnover_comparison(data_high, hp=hp, num_portfolios=10, banding_threshold=0.20)
        results.append({
            'signal': 'high_vol',
            'hp': hp,
            **r
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Signal':<10} {'HP':>4} {'No Band':>10} {'Band':>10} {'Reduction':>12} {'OK?':>6}")
    print("-" * 55)

    all_ok = True
    for r in results:
        ok = r['band'] <= r['no_band'] + 1e-10
        ok_str = "YES" if ok else "NO ***"
        all_ok = all_ok and ok
        print(f"{r['signal']:<10} {r['hp']:>4} {r['no_band']:>10.4f} {r['band']:>10.4f} "
              f"{r['reduction']:>10.1f}% {ok_str:>6}")

    print("-" * 55)
    if all_ok:
        print("ALL TESTS PASSED: Banding correctly reduces turnover.")
    else:
        print("*** SOME TESTS FAILED: Banding increased turnover - BUG! ***")

    return results


def debug_single_case():
    """
    Deep debug of a single case to understand exactly what's happening.
    """
    print("\n" + "=" * 70)
    print("DEEP DEBUG: HP=3, 10 portfolios, banding=0.20")
    print("=" * 70)

    # Create small panel for easy tracing
    data = create_low_volatility_panel(n_dates=12, n_bonds=50, seed=42)

    hp = 3
    num_portfolios = 10
    banding_threshold = 0.20

    # Run WITH banding
    strategy = SingleSort(
        holding_period=hp,
        sort_var='signal',
        num_portfolios=num_portfolios
    )

    sf = StrategyFormation(
        data=data,
        strategy=strategy,
        turnover=True,
        banding_threshold=banding_threshold,
        save_idx=True,  # Save portfolio indices
        verbose=True
    )

    result = sf.fit()

    # Inspect banding state
    inspect_banding_state(sf)

    # Trace specific cohort
    trace_banding_for_cohort(data, hp=hp, cohort=0,
                            num_portfolios=num_portfolios,
                            banding_threshold=banding_threshold)

    # Get turnover
    ew_turn, vw_turn = result.get_turnover()
    print(f"\nTurnover (EW, all portfolios):")
    print(ew_turn)

    print(f"\nMean EW turnover per portfolio:")
    print(ew_turn.mean())

    return sf, result


def check_banding_application_order():
    """
    Check if banding is being applied correctly in the horizon loop.

    The concern: For HP>1, _form_single_period is called multiple times (once per h).
    Each call applies banding and updates lag_rank.
    Is this causing issues?
    """
    print("\n" + "=" * 70)
    print("CHECKING BANDING APPLICATION ORDER IN HORIZON LOOP")
    print("=" * 70)

    print("""
For HP=3, at each formation time t_idx:
- self.cohort = t_idx % 3
- Loop over h = 0, 1, 2:
  - _form_single_period called for return date t_idx + h + 1
  - _apply_banding_to_period uses self.cohort as key
  - lag_rank[cohort] is compared and updated

Key question: Does calling _apply_banding_to_period for each h cause issues?

Expected behavior:
- h=0: Compare current ranks with previous rebalancing's ranks (correct)
- h=1: Compare current ranks with lag_rank (which was just updated by h=0)
       BUT: current ranks come from ranks_map[date_t] (same for all h)
       So comparison should give same result as h=0
- h=2: Same as h=1

The ranks are identical for all h (same formation date), so banding should
give the same result. BUT lag_rank is being updated each time (wasteful).

Let's verify this by checking if the banding result is consistent.
""")

    # Create data and run
    data = create_low_volatility_panel(n_dates=12, n_bonds=20, seed=42)

    strategy = SingleSort(
        holding_period=3,
        sort_var='signal',
        num_portfolios=10
    )

    sf = StrategyFormation(
        data=data,
        strategy=strategy,
        turnover=True,
        banding_threshold=0.20,
        save_idx=True,
        verbose=False
    )

    result = sf.fit()

    # Check port_idx - do ranks change across horizons for same formation?
    if hasattr(sf, 'port_idx'):
        dates = sorted(sf.port_idx.keys())
        print(f"\nport_idx has {len(dates)} dates")

        # Group by formation date (assuming formation = return - 1)
        # Check if ranks are consistent
        print("\nChecking rank consistency across horizons...")

        # Actually, port_idx is keyed by return date (date_t1), not formation date
        # So for HP=3, returns at t=1,2,3 all have formation at t=0
        # Let's check if the ranks in these are consistent

        for i in range(0, min(9, len(dates)), 3):
            if i + 2 < len(dates):
                d0, d1, d2 = dates[i], dates[i+1], dates[i+2]
                idx0 = sf.port_idx[d0]
                idx1 = sf.port_idx[d1]
                idx2 = sf.port_idx[d2]

                # Find common bonds
                common = set(idx0['ID']) & set(idx1['ID']) & set(idx2['ID'])

                if common:
                    sample_bond = list(common)[0]
                    r0 = idx0[idx0['ID'] == sample_bond]['ptf_rank'].values[0]
                    r1 = idx1[idx1['ID'] == sample_bond]['ptf_rank'].values[0]
                    r2 = idx2[idx2['ID'] == sample_bond]['ptf_rank'].values[0]

                    print(f"  Dates {d0.date()}, {d1.date()}, {d2.date()}: "
                          f"Bond {sample_bond} ranks = [{r0}, {r1}, {r2}]")

    return sf, result


def main():
    print("=" * 70)
    print("BANDING DIAGNOSTIC FOR HP > 1")
    print("=" * 70)
    print("""
This script investigates whether banding is working correctly with HP > 1.

With 10 portfolios and banding_threshold = 0.20:
- threshold_portfolios = 0.20 * 10 = 2
- A bond must move by >= 2 portfolio ranks to be reassigned
- Otherwise, it stays in its previous portfolio

Expected behavior:
- Turnover WITH banding <= Turnover WITHOUT banding (always!)
- The reduction depends on signal volatility
""")

    # Run invariant test
    results = test_banding_invariant()

    # Deep debug single case
    sf, result = debug_single_case()

    # Check banding application order
    check_banding_application_order()

    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
