#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Debug script to reproduce ZeroDivisionError with staggered rebalancing.

The issue occurs when:
1. Using holding_period > 1 (staggered rebalancing)
2. Signal has many NaN values or missing data at certain dates
3. Some portfolios end up with 0 bonds after intersection

Root cause: fastmath=True on compute_portfolio_weights_single allows
the division 1.0/cnt to execute even when cnt=0, bypassing the if check.
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
import PyBondLab as pbl
from PyBondLab.pbl_test import generate_synthetic_data

print("=" * 60)
print("DEBUG: ZeroDivisionError with Staggered Rebalancing")
print("=" * 60)

# Test 1: Direct test of the numba function
print("\n" + "=" * 60)
print("Test 0: Direct numba function test with empty portfolio")
print("=" * 60)

from PyBondLab.numba_core import compute_portfolio_weights_single

# Create scenario where portfolio 5 has no bonds (all ranks are 1-4 or 6-10)
ranks = np.array([1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0, 10.0])
vw = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
nport = 10

print(f"Ranks: {ranks}")
print(f"Portfolio 5 is EMPTY (no rank=5 in array)")

try:
    eweights, vweights, counts = compute_portfolio_weights_single(ranks, vw, nport)
    print(f"SUCCESS: eweights = {eweights}")
    print(f"         counts = {counts}")
except ZeroDivisionError as e:
    print(f"REPRODUCED: ZeroDivisionError - {e}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")


# Test 2: Scenario with very few bonds per date
print("\n" + "=" * 60)
print("Test 1: Extreme case - only 5 bonds per date, 10 portfolios")
print("=" * 60)

# Generate small data
np.random.seed(42)
dates = pd.date_range('2020-01-31', periods=24, freq='M')
n_bonds = 50

rows = []
for d in dates:
    # Only 5 bonds active per date (guarantees some portfolios empty)
    active_bonds = np.random.choice(n_bonds, size=5, replace=False)
    for bond in active_bonds:
        rows.append({
            'date': d,
            'ID': f'bond_{bond}',
            'ret': np.random.randn() * 0.02,
            'VW': np.random.uniform(100, 1000),
            'signal': np.random.randn(),
            'RATING_NUM': np.random.randint(1, 22)
        })

data_small = pd.DataFrame(rows)
print(f"Data shape: {data_small.shape}")
print(f"Bonds per date: {data_small.groupby('date')['ID'].count().mean():.1f}")

try:
    strategy = pbl.SingleSort(
        holding_period=3,
        sort_var='signal',
        num_portfolios=10  # More portfolios than bonds per date!
    )
    sf = pbl.StrategyFormation(data=data_small, strategy=strategy, turnover=True)
    result = sf.fit()
    ew_ls, vw_ls = result.get_long_short()
    print(f"SUCCESS: HP=3 works, {len(ew_ls)} dates")
except ZeroDivisionError as e:
    print(f"REPRODUCED: ZeroDivisionError - {e}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")


# Test 3: Regular data with missing signal values
print("\n" + "=" * 60)
print("Test 2: Regular data with 95% NaN signal")
print("=" * 60)

data = generate_synthetic_data(n_dates=60, n_bonds=500, seed=42)
data['tmat'] = data['signal1'].copy()

# Extreme NaN ratio
nan_mask = np.random.random(len(data)) < 0.95  # 95% NaN!
data.loc[nan_mask, 'tmat'] = np.nan

valid_per_date = data.groupby('date')['tmat'].apply(lambda x: x.notna().sum())
print(f"Valid observations per date: min={valid_per_date.min()}, max={valid_per_date.max()}")

try:
    strategy = pbl.SingleSort(
        holding_period=3,
        sort_var='tmat',
        num_portfolios=10
    )
    sf = pbl.StrategyFormation(data=data, strategy=strategy, turnover=True)
    result = sf.fit()
    ew_ls, vw_ls = result.get_long_short()
    print(f"SUCCESS: HP=3 works, {len(ew_ls)} dates")
except ZeroDivisionError as e:
    print(f"REPRODUCED: ZeroDivisionError - {e}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")


print("\n" + "=" * 60)
print("DIAGNOSIS")
print("=" * 60)
print("""
If ZeroDivisionError is reproduced:
-----------------------------------
The issue is fastmath=True on compute_portfolio_weights_single.

Solution: Change line 98 in numba_core.py from:
    @njit(cache=True, fastmath=True)
To:
    @njit(cache=True)

If no error is reproduced:
--------------------------
The issue may be specific to:
1. User's numba version
2. Specific data patterns in 'tmat'
3. Platform-specific behavior

Check the user's data for:
- Very few bonds per date
- Many NaN values in signal
- Signal values clustered in narrow range (causing empty portfolios)
""")
