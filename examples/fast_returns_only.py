"""
Fast Returns-Only Path Example

Demonstrates the ultra-fast portfolio formation path that activates automatically
when turnover, characteristics, and banding are disabled.

Key concepts:
- Auto-detection of fast path conditions
- Performance comparison: fast vs slow path
- Verifying numerical accuracy matches the standard path

The fast path uses parallel numba kernels to process ALL dates at once,
instead of the standard per-date loop. This provides significant speedup
for research workflows that only need portfolio returns.

Author: Claude Code (Numba Optimization Project)
"""

import numpy as np
import pandas as pd
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import PyBondLab as pbl
from PyBondLab.PyBondLab import StrategyFormation
from PyBondLab.StrategyClass import SingleSort

# =============================================================================
# Generate Synthetic Data
# =============================================================================

def generate_bond_data(n_bonds=500, n_periods=60, seed=42):
    """
    Generate synthetic bond panel data for examples.

    Returns DataFrame with columns: date, ID, ret, RATING_NUM, VW, signal
    """
    np.random.seed(seed)

    dates = pd.date_range('2018-01-31', periods=n_periods, freq='ME')
    bond_ids = [f'BOND{i:04d}' for i in range(n_bonds)]

    data = []
    for bond_id in bond_ids:
        # Assign random but persistent characteristics
        rating = np.random.randint(1, 23)
        signal_base = np.random.normal(0, 1)

        for date in dates:
            # Monthly return with signal premium
            signal = signal_base + np.random.normal(0, 0.3)  # Persistent + noise
            ret = 0.005 + signal * 0.002 + np.random.normal(0, 0.02)

            # Market value
            vw = 1_000_000 * np.exp(np.random.normal(0, 0.5))

            data.append({
                'date': date,
                'ID': bond_id,
                'ret': ret,
                'RATING_NUM': rating,
                'VW': vw,
                'signal': signal
            })

    df = pd.DataFrame(data)
    df = df.sort_values(['ID', 'date']).reset_index(drop=True)

    return df

# =============================================================================
# Fast Path Demonstration
# =============================================================================

print("=" * 80)
print("Fast Returns-Only Path Demonstration")
print("=" * 80)
print()

# Generate data
print("Generating synthetic bond data...")
data = generate_bond_data(n_bonds=500, n_periods=60)
print(f"  Data shape: {data.shape}")
print(f"  Date range: {data['date'].min()} to {data['date'].max()}")
print(f"  Number of bonds: {data['ID'].nunique()}")
print()

# =============================================================================
# Explain Fast Path Conditions
# =============================================================================

print("=" * 80)
print("Fast Path Conditions")
print("=" * 80)
print()
print("The fast returns-only path is AUTOMATICALLY used when ALL of:")
print("  - turnover=False (no turnover computation)")
print("  - chars=None (no characteristics tracking)")
print("  - banding_threshold=None (no banding/transition bands)")
print("  - SingleSort strategy (not DoubleSort)")
print("  - Monthly rebalancing frequency")
print()
print("When these conditions are met, the code processes ALL dates in parallel")
print("using numba's prange, instead of the standard per-date loop.")
print()

# =============================================================================
# Run with Fast Path (Auto-detected)
# =============================================================================

print("=" * 80)
print("Fast Path (Auto-detected)")
print("=" * 80)
print()

# Create strategy with conditions that enable fast path
strategy_fast = SingleSort(
    holding_period=1,
    sort_var='signal',
    num_portfolios=5
)

# Warm up JIT compilation
print("Warming up JIT compilation...")
sf_warmup = StrategyFormation(
    data=data,
    strategy=strategy_fast,
    holding_period=1,
    num_portfolios=5,
    turnover=False,    # Required for fast path
    chars=None,        # Required for fast path
    banding=None,      # Required for fast path
    verbose=False
)
_ = sf_warmup.fit()
print("  JIT warm-up complete!")
print()

# Measure fast path performance
print("Running with fast path (turnover=False, chars=None, banding=None)...")
times_fast = []
for i in range(5):
    strategy = SingleSort(holding_period=1, sort_var='signal', num_portfolios=5)
    sf = StrategyFormation(
        data=data,
        strategy=strategy,
        holding_period=1,
        num_portfolios=5,
        turnover=False,
        chars=None,
        banding=None,
        verbose=False
    )
    t0 = time.time()
    result_fast = sf.fit()
    times_fast.append(time.time() - t0)

print(f"  Average runtime: {np.mean(times_fast):.3f}s (+/- {np.std(times_fast):.3f}s)")
print(f"  EW Long-Short mean: {result_fast.ea.returns.ewls_df.mean().values[0]:.6f}")
print(f"  VW Long-Short mean: {result_fast.ea.returns.vwls_df.mean().values[0]:.6f}")
print()

# =============================================================================
# Run with Slow Path (Forced)
# =============================================================================

print("=" * 80)
print("Slow Path (Forced by disabling fast path)")
print("=" * 80)
print()

print("Running with slow path (forced)...")
times_slow = []
for i in range(5):
    strategy = SingleSort(holding_period=1, sort_var='signal', num_portfolios=5)
    sf = StrategyFormation(
        data=data,
        strategy=strategy,
        holding_period=1,
        num_portfolios=5,
        turnover=False,
        chars=None,
        banding=None,
        verbose=False
    )
    # Force slow path by overriding the check
    sf._can_use_fast_path = lambda: False
    t0 = time.time()
    result_slow = sf.fit()
    times_slow.append(time.time() - t0)

print(f"  Average runtime: {np.mean(times_slow):.3f}s (+/- {np.std(times_slow):.3f}s)")
print(f"  EW Long-Short mean: {result_slow.ea.returns.ewls_df.mean().values[0]:.6f}")
print(f"  VW Long-Short mean: {result_slow.ea.returns.vwls_df.mean().values[0]:.6f}")
print()

# =============================================================================
# Compare Results
# =============================================================================

print("=" * 80)
print("Comparison: Fast Path vs Slow Path")
print("=" * 80)
print()

speedup = np.mean(times_slow) / np.mean(times_fast)
print(f"Speedup: {speedup:.2f}x faster with fast path")
print()

# Verify numerical accuracy
ew_diff = abs(result_fast.ea.returns.ewls_df.mean().values[0] -
              result_slow.ea.returns.ewls_df.mean().values[0])
vw_diff = abs(result_fast.ea.returns.vwls_df.mean().values[0] -
              result_slow.ea.returns.vwls_df.mean().values[0])

print("Numerical accuracy (should be < 1e-10):")
print(f"  EW difference: {ew_diff:.2e}")
print(f"  VW difference: {vw_diff:.2e}")

TOLERANCE = 1e-10
if ew_diff < TOLERANCE and vw_diff < TOLERANCE:
    print("  PASS: Results match within tolerance!")
else:
    print("  WARNING: Results differ more than expected!")
print()

# =============================================================================
# What Disables Fast Path
# =============================================================================

print("=" * 80)
print("What Disables Fast Path")
print("=" * 80)
print()

print("Adding turnover=True forces slow path:")
strategy = SingleSort(holding_period=1, sort_var='signal', num_portfolios=5)
sf_turnover = StrategyFormation(
    data=data,
    strategy=strategy,
    holding_period=1,
    num_portfolios=5,
    turnover=True,     # This disables fast path
    chars=None,
    banding=None,
    verbose=False
)
print(f"  Can use fast path: {sf_turnover._can_use_fast_path()}")
print()

print("Adding chars=['VW'] forces slow path:")
strategy = SingleSort(holding_period=1, sort_var='signal', num_portfolios=5)
sf_chars = StrategyFormation(
    data=data,
    strategy=strategy,
    holding_period=1,
    num_portfolios=5,
    turnover=False,
    chars=['VW'],      # This disables fast path
    banding=None,
    verbose=False
)
print(f"  Can use fast path: {sf_chars._can_use_fast_path()}")
print()

print("Adding banding_threshold=0.2 forces slow path:")
strategy = SingleSort(holding_period=1, sort_var='signal', num_portfolios=5)
sf_banding = StrategyFormation(
    data=data,
    strategy=strategy,
    holding_period=1,
    num_portfolios=5,
    turnover=False,
    chars=None,
    banding_threshold=0.2,  # This disables fast path (threshold = 1/nport)
    verbose=False
)
print(f"  Can use fast path: {sf_banding._can_use_fast_path()}")
print()

# =============================================================================
# Summary
# =============================================================================

print("=" * 80)
print("Summary")
print("=" * 80)
print()
print("The fast returns-only path provides automatic speedup for research")
print("workflows that only need portfolio returns (no turnover/chars/banding).")
print()
print("Key takeaways:")
print(f"  1. Fast path is {speedup:.1f}x faster than slow path")
print("  2. Results are numerically identical (within floating-point tolerance)")
print("  3. Auto-detection means no code changes required")
print("  4. Larger datasets show even greater speedup")
print()
print("When to use:")
print("  - Initial signal screening (many signals, returns only)")
print("  - Quick backtests without turnover analysis")
print("  - Research exploration before full analysis")
print()
print("When slow path is needed:")
print("  - Turnover computation (turnover=True)")
print("  - Characteristics tracking (chars=[...])")
print("  - Transition bands (banding_threshold=0.2)")
print("  - Double sorts (DoubleSort strategy)")
print()
