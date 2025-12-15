"""
Long-Term Reversal Strategy

Demonstrates long-term reversal (contrarian) portfolio formation.
Buys past losers and sells past winners.

Key concepts:
- LTreversal strategy with formation and skip periods
- Contrarian signal: long-term returns minus recent returns
- Difference from momentum
- Loser-minus-winner portfolio

Author: Giulio Rossetti
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import PyBondLab as pbl

# =============================================================================
# Generate Synthetic Data with Reversal Patterns
# =============================================================================

def generate_bond_data_reversal(n_bonds=100, n_periods=100, seed=42):
    """
    Generate synthetic bond data with mean reversion.
    Past extreme performers tend to revert to mean.
    """
    np.random.seed(seed)

    dates = pd.date_range('2005-01-31', periods=n_periods, freq='M')
    bond_ids = [f'BOND{i:04d}' for i in range(n_bonds)]

    data = []
    for bond_id in bond_ids:
        # Each bond has mean return + mean-reverting component
        mean_return = np.random.normal(0.005, 0.002)

        returns = []
        deviation = 0  # Deviation from mean
        for t in range(n_periods):
            # Mean reversion: negative autocorrelation
            shock = np.random.normal(0, 0.02)
            deviation = -0.4 * deviation + shock  # Negative AR(1)
            ret = mean_return + deviation

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
# Long-Term Reversal: 36-13-6 Specification
# =============================================================================

print("=" * 80)
print("Long-Term Reversal Strategy")
print("=" * 80)
print()

data = generate_bond_data_reversal(n_bonds=100, n_periods=100)

print("Data summary:")
print(f"  Bonds: {data['ID'].nunique()}")
print(f"  Periods: {data['date'].nunique()}")
print(f"  Mean return: {data['ret'].mean():.4f}")
print()

# Define LT reversal strategy
strategy = pbl.LTreversal(
    holding_period=6,        # Hold portfolios for 6 months
    lookback_period=36,      # Look back 36 months
    skip=13,                 # Skip most recent 13 months
    num_portfolios=10        # Decile portfolios
)

print("Strategy configuration:")
print(f"  Lookback period: {strategy.lookback_period} months")
print(f"  Skip period: {strategy.skip} months")
print(f"  Holding period: {strategy.holding_period} months")
print(f"  Number of portfolios: {strategy.num_portfolios}")
print()

print("Reversal signal:")
print(f"  Long-term returns (t-{strategy.lookback_period} to t)")
print(f"  MINUS")
print(f"  Recent returns (t-{strategy.skip} to t)")
print(f"  = Returns from t-{strategy.lookback_period} to t-{strategy.skip}")
print()

print("Example timeline at March 2010:")
print("  Long-term: March 2007 to March 2010 (36 months)")
print("  Recent: Feb 2009 to March 2010 (13 months)")
print("  Signal: March 2007 to Feb 2009 (23 months)")
print("  Holding: March 2010 to Aug 2010 (6 months)")
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
print("Reversal Portfolio Returns")
print("=" * 80)
print()

# Portfolio returns
labels = results.ret_vw.columns.tolist()
mean_rets = results.ret_vw.mean()

print("Decile portfolios (D1=past losers, D10=past winners):")
for i, label in enumerate(labels):
    percentile = (i + 1) * 10
    print(f"  {label}: {mean_rets[label]:.4f}  ({percentile}th percentile)")

print()

# Loser-minus-winner (note: opposite sign from momentum)
print("High-Minus-Low (Past Losers - Past Winners):")
print(f"  Mean: {results.hml_vw.mean():.4f}")
print(f"  Std:  {results.hml_vw.std():.4f}")
print(f"  Sharpe: {results.hml_vw.mean() / results.hml_vw.std():.4f}")
print(f"  t-stat: {results.hml_vw.mean() / results.hml_vw.std() * np.sqrt(len(results.hml_vw)):.2f}")
print()

print("Note: Positive HML indicates reversal effect")
print("      (past losers outperform past winners)")
print()

# =============================================================================
# Reversal vs Momentum
# =============================================================================

print("=" * 80)
print("Reversal vs Momentum")
print("=" * 80)
print()

# Run momentum for comparison
strategy_mom = pbl.Momentum(
    holding_period=6,
    lookback_period=6,
    skip=1,
    num_portfolios=10,
    verbose=False
)

results_mom = pbl.StrategyFormation(
    data,
    strategy=strategy_mom,
    turnover=False,
    verbose=False
).fit()

print("Key differences:")
print()
print("MOMENTUM (6-1-6):")
print(f"  Formation: Recent 6 months (skip 1)")
print(f"  Signal: Cumulative past returns")
print(f"  Strategy: Buy winners, sell losers")
print(f"  WML mean: {results_mom.hml_vw.mean():.4f}")
print()

print("LONG-TERM REVERSAL (36-13-6):")
print(f"  Formation: Months t-36 to t-13")
print(f"  Signal: Long-term - recent returns")
print(f"  Strategy: Buy losers, sell winners")
print(f"  HML mean: {results.hml_vw.mean():.4f}")
print()

# Correlation between strategies
corr = results.hml_vw.corr(results_mom.hml_vw)
print(f"Correlation between reversal and momentum: {corr:.3f}")
print()

if corr < -0.1:
    print("  → Negative correlation: reversal opposes momentum")
elif corr > 0.1:
    print("  → Positive correlation: both capture similar patterns")
else:
    print("  → Low correlation: independent patterns")

print()

# =============================================================================
# Alternative Specifications
# =============================================================================

print("=" * 80)
print("Alternative Reversal Specifications")
print("=" * 80)
print()

# Shorter lookback
print("Shorter horizon (24-7-6):")
strategy_short = pbl.LTreversal(
    holding_period=6,
    lookback_period=24,
    skip=7,
    num_portfolios=5,
    verbose=False
)

results_short = pbl.StrategyFormation(
    data,
    strategy=strategy_short,
    turnover=False,
    verbose=False
).fit()

print(f"  HML mean: {results_short.hml_vw.mean():.4f}")
print(f"  HML Sharpe: {results_short.hml_vw.mean() / results_short.hml_vw.std():.4f}")
print()

# Longer lookback
print("Longer horizon (60-13-6):")
strategy_long = pbl.LTreversal(
    holding_period=6,
    lookback_period=60,
    skip=13,
    num_portfolios=5,
    verbose=False
)

results_long = pbl.StrategyFormation(
    data,
    strategy=strategy_long,
    turnover=False,
    verbose=False
).fit()

print(f"  HML mean: {results_long.hml_vw.mean():.4f}")
print(f"  HML Sharpe: {results_long.hml_vw.mean() / results_long.hml_vw.std():.4f}")
print()

# =============================================================================
# Signal Decomposition
# =============================================================================

print("=" * 80)
print("Understanding the Signal Decomposition")
print("=" * 80)
print()

print("LT Reversal signal = LongTermReturns - RecentReturns")
print()
print("Why subtract recent returns?")
print("  1. Avoid short-term momentum contamination")
print("  2. Focus on long-term mean reversion")
print("  3. Separate long-term losers from recent losers")
print()
print("Example: Bond with -20% over 36 months")
print("  If -20% all in last 13 months: Recent loser (momentum)")
print("  If -20% in months 13-36, flat recently: LT loser (reversal)")
print()
print("The skip period isolates true long-term reversal from momentum.")
print()

# =============================================================================
# Summary
# =============================================================================

print("=" * 80)
print("Summary")
print("=" * 80)
print()
print("This example demonstrated:")
print("  1. Long-term reversal strategy (contrarian)")
print("  2. Signal: Long-term returns minus recent returns")
print("  3. Classic 36-13-6 specification")
print("  4. Difference from momentum strategies")
print("  5. Alternative time horizons")
print()
print("Common specifications in literature:")
print("  - DeBondt-Thaler (1985): 36-month formation")
print("  - Jegadeesh-Titman (1993): 60-13-month")
print("  - Bonds: 36-13-6 or 24-7-6")
print()
print("Key parameters:")
print("  lookback_period: Total look-back window")
print("  skip: Recent period to exclude (avoid momentum)")
print("  holding_period: How long to hold portfolios")
print()
print("Economic intuition:")
print("  - Overreaction: Prices overreact to news, then revert")
print("  - Mean reversion: Extreme performers revert to average")
print("  - Contrarian profits: Buying out-of-favor bonds")
print()
