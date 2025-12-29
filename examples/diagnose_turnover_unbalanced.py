#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diagnostic script for turnover computation with UNBALANCED panels.

This tests what happens when bonds drop in/out of the sample between rebalancing dates.

Key question: Is it "fair" to set holding cohort turnover to 0 when bonds can drop out?
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/user/PyBondLab-Dev')

import PyBondLab as pbl
from PyBondLab import StrategyFormation, SingleSort


def create_unbalanced_panel(
    n_dates: int = 24,
    n_bonds: int = 100,
    seed: int = 42,
    hp: int = 1,
    dropout_rate: float = 0.10,  # 10% of bonds drop out each period
    dropin_rate: float = 0.05,   # 5% new bonds enter each period
):
    """
    Create an UNBALANCED panel where bonds randomly drop in/out.

    Parameters:
    -----------
    dropout_rate : float
        Probability that a bond present at t will be absent at t+1
    dropin_rate : float
        Probability that a new bond ID enters at each period
    """
    np.random.seed(seed)

    # Create date range
    dates = pd.date_range('2020-01-31', periods=n_dates, freq='ME')

    # Track which bonds are "alive" at each date
    bond_pool = set(range(n_bonds))
    alive_bonds = set(range(n_bonds))  # All start alive
    next_bond_id = n_bonds

    records = []
    bond_presence = {}  # Track which dates each bond appears

    for t_idx, date in enumerate(dates):
        # Handle dropouts (except first date)
        if t_idx > 0:
            # Some bonds drop out
            for bond_idx in list(alive_bonds):
                if np.random.random() < dropout_rate:
                    alive_bonds.discard(bond_idx)

            # Some new bonds enter
            n_new = int(dropin_rate * n_bonds)
            for _ in range(n_new):
                alive_bonds.add(next_bond_id)
                next_bond_id += 1

        # Create records for alive bonds
        for bond_idx in alive_bonds:
            bond_id = f"BOND_{bond_idx:04d}"

            # Track presence
            if bond_id not in bond_presence:
                bond_presence[bond_id] = []
            bond_presence[bond_id].append(t_idx)

            # Signal that reverses every HP months
            phase = (t_idx // hp) % 2
            if phase == 0:
                signal = bond_idx % n_bonds  # Ascending (mod n_bonds for new bonds)
            else:
                signal = (n_bonds - 1) - (bond_idx % n_bonds)  # Descending

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

    # Statistics
    bonds_per_date = df.groupby('date').size()
    total_unique_bonds = df['ID'].nunique()

    print(f"\nCreated UNBALANCED panel:")
    print(f"  Total rows: {len(df)}")
    print(f"  Unique bonds: {total_unique_bonds}")
    print(f"  Dates: {n_dates}")
    print(f"  Bonds/date: min={bonds_per_date.min()}, max={bonds_per_date.max()}, mean={bonds_per_date.mean():.1f}")
    print(f"  Dropout rate: {dropout_rate*100:.1f}%")
    print(f"  Drop-in rate: {dropin_rate*100:.1f}%")

    return df, bond_presence


def run_turnover_test(data: pd.DataFrame, hp: int, verbose: bool = True):
    """
    Run StrategyFormation with specified HP and return turnover statistics.
    """
    strategy = SingleSort(
        holding_period=hp,
        sort_var='signal',
        num_portfolios=5
    )

    sf = StrategyFormation(
        data=data,
        strategy=strategy,
        turnover=True,
        verbose=False
    )

    result = sf.fit()
    ew_turn, vw_turn = result.get_turnover()

    if verbose:
        print(f"\n{'='*60}")
        print(f"HP = {hp}")
        print(f"{'='*60}")
        print(f"Overall EW mean: {ew_turn.mean().mean():.4f}")
        print(f"Overall VW mean: {vw_turn.mean().mean():.4f}")

    return ew_turn, vw_turn, result, sf


def inspect_holding_cohort_weights(sf, hp: int):
    """
    Inspect what happens to weights in holding cohorts when bonds drop out.
    """
    print(f"\n{'='*60}")
    print(f"HOLDING COHORT WEIGHT ANALYSIS (HP={hp})")
    print(f"{'='*60}")

    state = sf.turnover_state

    if hp == 1:
        print("\nHP=1: No holding cohorts, all months rebalance.")
        return

    # For HP>1, look at the raw turnover array
    if len(state.ew_turn_ea.shape) == 3:
        print(f"\nRaw turnover array shape: {state.ew_turn_ea.shape}")
        print(f"  - Time: {state.ew_turn_ea.shape[0]}")
        print(f"  - Cohorts: {state.ew_turn_ea.shape[1]}")
        print(f"  - Portfolios: {state.ew_turn_ea.shape[2]}")

        # Look at a few time slices
        print("\nFirst 12 time slices (turnover by cohort, portfolio 0):")
        for t in range(min(12, state.ew_turn_ea.shape[0])):
            rebal_cohort = t % hp
            vals = [state.ew_turn_ea[t, c, 0] for c in range(hp)]
            vals_str = ", ".join([f"{v:.4f}" if not np.isnan(v) else "NaN" for v in vals])
            marker = " <-- rebal" if t >= hp else ""
            print(f"  t={t:2d}: cohort values = [{vals_str}], rebal_cohort={rebal_cohort}{marker}")


def analyze_dropout_impact(data: pd.DataFrame, hp: int):
    """
    Analyze the actual weight changes when bonds drop out of holding cohorts.
    """
    print(f"\n{'='*60}")
    print(f"DROPOUT IMPACT ANALYSIS (HP={hp})")
    print(f"{'='*60}")

    dates = sorted(data['date'].unique())

    # For each holding period window, track which bonds drop out
    print("\nTracking bond dropouts between consecutive dates:")

    for i in range(1, min(13, len(dates))):
        prev_date = dates[i-1]
        curr_date = dates[i]

        prev_bonds = set(data[data['date'] == prev_date]['ID'].unique())
        curr_bonds = set(data[data['date'] == curr_date]['ID'].unique())

        dropped = prev_bonds - curr_bonds
        added = curr_bonds - prev_bonds

        cohort = i % hp
        is_rebal = (i % hp == 0) if hp > 1 else True
        status = "REBAL" if is_rebal else f"HOLD (cohort {cohort})"

        print(f"  {i-1} -> {i}: Dropped={len(dropped):3d}, Added={len(added):3d}  [{status}]")

    print(f"\nKey Question:")
    print(f"  When bonds drop out of a HOLDING cohort, weights must be renormalized.")
    print(f"  This changes the portfolio composition without explicit trading.")
    print(f"  Current implementation: Sets holding cohort turnover to 0 (no trading).")
    print(f"  Alternative: Could compute 'latent turnover' from weight renormalization.")


def main():
    print("=" * 70)
    print("UNBALANCED PANEL TURNOVER DIAGNOSTIC")
    print("=" * 70)
    print("\nThis script examines turnover computation when bonds drop in/out.")
    print("Key question: Should holding cohorts have non-zero turnover when bonds drop out?")

    # Test with different dropout rates
    print("\n" + "=" * 70)
    print("TEST 1: MODERATE DROPOUT (10% dropout, 5% drop-in)")
    print("=" * 70)

    results = {}
    for hp in [1, 2, 3]:
        data, presence = create_unbalanced_panel(
            n_dates=24, n_bonds=100, seed=42, hp=hp,
            dropout_rate=0.10, dropin_rate=0.05
        )
        ew, vw, result, sf = run_turnover_test(data, hp)
        results[hp] = {'ew': ew, 'vw': vw, 'sf': sf}

    # Summary
    print("\n" + "-" * 55)
    print("SUMMARY: Moderate Dropout")
    print("-" * 55)
    print(f"{'HP':>5} {'Expected':>12} {'Actual EW':>12} {'Actual VW':>12}")
    print("-" * 55)
    expected = {1: 2.0, 2: 1.0, 3: 2.0/3}
    for hp in [1, 2, 3]:
        actual_ew = results[hp]['ew'].mean().mean()
        actual_vw = results[hp]['vw'].mean().mean()
        print(f"{hp:>5} {expected[hp]:>12.4f} {actual_ew:>12.4f} {actual_vw:>12.4f}")

    # Analyze HP=3 in detail
    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS: HP=3 with Dropouts")
    print("=" * 70)

    data_hp3, _ = create_unbalanced_panel(
        n_dates=24, n_bonds=100, seed=42, hp=3,
        dropout_rate=0.10, dropin_rate=0.05
    )
    _, _, _, sf_hp3 = run_turnover_test(data_hp3, hp=3, verbose=False)

    inspect_holding_cohort_weights(sf_hp3, hp=3)
    analyze_dropout_impact(data_hp3, hp=3)

    # Heavy dropout scenario
    print("\n" + "=" * 70)
    print("TEST 2: HEAVY DROPOUT (25% dropout, 10% drop-in)")
    print("=" * 70)

    for hp in [1, 3]:
        data, _ = create_unbalanced_panel(
            n_dates=24, n_bonds=100, seed=42, hp=hp,
            dropout_rate=0.25, dropin_rate=0.10
        )
        ew, vw, _, _ = run_turnover_test(data, hp)
        exp = expected[hp]
        print(f"  HP={hp}: EW={ew.mean().mean():.4f} (expected {exp:.4f})")


def discuss_turnover_semantics():
    """
    Print discussion of the two perspectives on turnover.
    """
    print("\n" + "=" * 70)
    print("DISCUSSION: TWO PERSPECTIVES ON TURNOVER")
    print("=" * 70)

    print("""
CURRENT IMPLEMENTATION: "Trading Turnover"
==========================================
- Turnover measures ACTUAL TRADING activity
- Holding cohorts do NOT trade (by definition)
- Therefore: Holding cohort turnover = 0
- Renormalization when bonds drop out is NOT trading
  (It's just adjusting remaining weights to sum to 1)

ALTERNATIVE VIEW: "Weight Change Turnover"
==========================================
- Turnover measures CHANGE IN PORTFOLIO COMPOSITION
- When bonds drop out, remaining bond weights increase
- This IS a change in composition (even without trading)
- Example:
    Before dropout: Bond A = 10%, Bond B = 10%, Bond C = 10%
    After B drops:  Bond A = 15%, Bond C = 15%
    Weight change = |0.15 - 0.10| * 2 = 0.10 turnover

WHICH IS "CORRECT"?
===================
Both perspectives are valid - it depends on what you're measuring:

1. TRADING COSTS: Use "Trading Turnover" (current)
   - Only rebalancing cohort trades
   - Holding cohorts have zero trading costs
   - Dropouts don't incur transaction costs

2. FACTOR EXPOSURE CHANGES: Use "Weight Change Turnover"
   - Tracks how portfolio factor loadings change
   - Even holding cohorts can have exposure drift
   - Important for risk management

3. REAL-WORLD HYBRID:
   - When a bond drops out (defaults, matures, gets called):
     * Long position: Forced to sell → trading turnover!
     * Short position: Cover is automatic → still turnover
   - When a bond enters (new issuance):
     * Could trade to include → optional turnover
     * Could ignore until rebalancing → no turnover

CURRENT CHOICE RATIONALE:
=========================
PyBondLab uses "Trading Turnover" because:
1. It matches how most practitioners think about turnover
2. It's consistent with the staggered rebalancing literature
3. Dropouts are relatively rare in large, liquid bond universes
4. The alternative is complex (need to track every weight change)

TO IMPLEMENT "Weight Change Turnover":
======================================
Would require:
1. Track weights at EVERY time step, not just rebalancing
2. Compute weight changes due to:
   - Return drift (already captured in scaled weights)
   - Bond dropouts (would need new tracking)
   - VW changes (market cap movements)
3. Much more complex state management
""")


if __name__ == "__main__":
    main()
    discuss_turnover_semantics()
