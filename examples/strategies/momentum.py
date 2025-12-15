"""
Momentum Strategy

Demonstrates momentum-based portfolio formation.
Sorts bonds by past returns over a formation period.

Key concepts:
- Momentum strategy with formation and holding periods
- Skip period to avoid microstructure effects
- Signal computation from past returns
- Winner-minus-loser portfolio

Author: Giulio Rossetti
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import PyBondLab as pbl

# =============================================================================
# Generate Synthetic Data with Momentum
# =============================================================================

def generate_bond_data_momentum(n_bonds=100, n_periods=80, seed=42):
    """
    Generate synthetic bond data with momentum patterns.
    Past winners tend to continue winning (positive autocorrelation).
    """
    np.random.seed(seed)

    dates = pd.date_range('2008-01-31', periods=n_periods, freq='M')
    bond_ids = [f'BOND{i:04d}' for i in range(n_bonds)]

    data = []
    for bond_id in bond_ids:
        # Each bond has persistent component + noise
        persistent_quality = np.random.normal(0, 0.005)

        returns = []
        for t in range(n_periods):
            # Returns have momentum: AR(1) structure
            if t == 0:
                ret = persistent_quality + np.random.normal(0, 0.02)
            else:
                # Positive autocorrelation (momentum)
                ret = 0.3 * returns[-1] + persistent_quality + np.random.normal(0, 0.02)

            returns.append(ret)

        for t, date in enumerate(dates):
            data.append({
                'date': date,
                'ID': bond_id,
                'ret': returns[t],
                'BOND_VALUE': 1_000_000 * np.exp(np.random.normal(0, 0.3))
            })

    df = pd.DataFrame(data)
    df = df.sort_values(['ID', 'date']).reset_index(drop=True)

    return df

# =============================================================================
# Classic Momentum: 6-month formation, 1-month skip, 6-month holding
# =============================================================================

print("=" * 80)
print("Momentum Strategy: 6-1-6 Specification")
print("=" * 80)
print()

data = generate_bond_data_momentum(n_bonds=100, n_periods=80)

print("Data summary:")
print(f"  Bonds: {data['ID'].nunique()}")
print(f"  Periods: {data['date'].nunique()}")
print(f"  Mean return: {data['ret'].mean():.4f}")
print()

# Define momentum strategy
strategy = pbl.Momentum(
    holding_period=6,        # Hold portfolios for 6 months
    lookback_period=6,       # Formation period: past 6 months
    skip=1,                  # Skip most recent month
    num_portfolios=10        # Decile portfolios
)

print("Strategy configuration:")
print(f"  Formation period (lookback): {strategy.lookback_period} months")
print(f"  Skip period: {strategy.skip} month")
print(f"  Holding period: {strategy.holding_period} months")
print(f"  Number of portfolios: {strategy.num_portfolios}")
print()

print("Momentum signal:")
print(f"  Cumulative return from t-{strategy.lookback_period} to t-{strategy.skip}")
print(f"  Example: At June 2010, use returns from Dec 2009 to May 2010")
print()

# Run formation
results = pbl.StrategyFormation(
    data,
    strategy=strategy,
    turnover=False,
    verbose=False
).fit()

print("Formation complete!")
print()

# =============================================================================
# Examine Results
# =============================================================================

print("=" * 80)
print("Momentum Portfolio Returns")
print("=" * 80)
print()

# Portfolio returns
labels = results.ret_vw.columns.tolist()
mean_rets = results.ret_vw.mean()

print("Decile portfolios (D1=losers, D10=winners):")
for i, label in enumerate(labels):
    percentile = (i + 1) * 10
    print(f"  {label}: {mean_rets[label]:.4f}  ({percentile}th percentile)")

print()

# Winner-minus-loser
print("Winner-Minus-Loser (WML) Portfolio:")
print(f"  Mean: {results.hml_vw.mean():.4f}")
print(f"  Std:  {results.hml_vw.std():.4f}")
print(f"  Sharpe: {results.hml_vw.mean() / results.hml_vw.std():.4f}")
print(f"  t-stat: {results.hml_vw.mean() / results.hml_vw.std() * np.sqrt(len(results.hml_vw)):.2f}")
print()

# =============================================================================
# Alternative Specifications
# =============================================================================

print("=" * 80)
print("Alternative Momentum Specifications")
print("=" * 80)
print()

# Short-term momentum: 3-1-3
print("Short-term momentum (3-1-3):")
strategy_st = pbl.Momentum(
    holding_period=3,
    lookback_period=3,
    skip=1,
    num_portfolios=5,
    verbose=False
)

results_st = pbl.StrategyFormation(
    data,
    strategy=strategy_st,
    turnover=False,
    verbose=False
).fit()

print(f"  WML mean: {results_st.hml_vw.mean():.4f}")
print(f"  WML Sharpe: {results_st.hml_vw.mean() / results_st.hml_vw.std():.4f}")
print()

# Long-term momentum: 12-1-6
print("Long-term momentum (12-1-6):")
strategy_lt = pbl.Momentum(
    holding_period=6,
    lookback_period=12,
    skip=1,
    num_portfolios=5,
    verbose=False
)

results_lt = pbl.StrategyFormation(
    data,
    strategy=strategy_lt,
    turnover=False,
    verbose=False
).fit()

print(f"  WML mean: {results_lt.hml_vw.mean():.4f}")
print(f"  WML Sharpe: {results_lt.hml_vw.mean() / results_lt.hml_vw.std():.4f}")
print()

# No skip: 6-0-6
print("No skip period (6-0-6):")
strategy_ns = pbl.Momentum(
    holding_period=6,
    lookback_period=6,
    skip=0,
    num_portfolios=5,
    verbose=False
)

results_ns = pbl.StrategyFormation(
    data,
    strategy=strategy_ns,
    turnover=False,
    verbose=False
).fit()

print(f"  WML mean: {results_ns.hml_vw.mean():.4f}")
print(f"  WML Sharpe: {results_ns.hml_vw.mean() / results_ns.hml_vw.std():.4f}")
print()

print("Note: Skip period typically improves performance by avoiding")
print("      short-term reversal/microstructure effects.")
print()

# =============================================================================
# Cumulative Returns
# =============================================================================

print("=" * 80)
print("Cumulative Returns")
print("=" * 80)
print()

# Compute cumulative returns
cum_winners = (1 + results.ret_vw.iloc[:, -1]).cumprod()
cum_losers = (1 + results.ret_vw.iloc[:, 0]).cumprod()
cum_wml = (1 + results.hml_vw).cumprod()

print(f"Final cumulative returns:")
print(f"  Winners (D10): {cum_winners.iloc[-1]:.3f}x")
print(f"  Losers (D1):   {cum_losers.iloc[-1]:.3f}x")
print(f"  WML strategy:  {cum_wml.iloc[-1]:.3f}x")
print()

# =============================================================================
# Signal Interpretation
# =============================================================================

print("=" * 80)
print("Understanding the Momentum Signal")
print("=" * 80)
print()

print("The Momentum strategy computes:")
print(f"  signal_t = cumulative_return(t-{strategy.lookback_period-strategy.skip}, t-{strategy.skip})")
print()
print("Example timeline at March 2010:")
print("  Formation: Sep 2009 to Feb 2010 (6 months)")
print("  Skip: Feb 2010 (1 month)")
print("  Holding: Mar 2010 to Aug 2010 (6 months)")
print()
print("Portfolios formed: Winners (high signal) vs Losers (low signal)")
print()

# =============================================================================
# Summary
# =============================================================================

print("=" * 80)
print("Summary")
print("=" * 80)
print()
print("This example demonstrated:")
print("  1. Momentum strategy with formation, skip, and holding periods")
print("  2. Classic 6-1-6 specification (6mo formation, 1mo skip, 6mo holding)")
print("  3. Winner-minus-loser portfolio construction")
print("  4. Alternative specifications (short-term, long-term, no skip)")
print("  5. Signal computation from cumulative past returns")
print()
print("Common specifications in literature:")
print("  - JT (1993): 12-1-1 momentum")
print("  - FF (1996): 12-2-1 momentum")
print("  - Bonds: 6-1-6 or 12-1-6")
print()
print("Key parameters:")
print("  lookback_period: Formation window length")
print("  skip: Avoid short-term reversal (typically 1 month)")
print("  holding_period: How long to hold portfolios")
print()
