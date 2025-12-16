# -*- coding: utf-8 -*-
"""
PyBondLab Baseline Test Script - Source of Truth
=================================================

This script establishes SOURCE OF TRUTH results from the original (slow) code.
Future optimized code MUST match these results exactly.

Test Configurations:
--------------------
1. SingleSort, hp=1, no banding, no turnover
2. SingleSort, hp=1, no banding, turnover=True
3. SingleSort, hp=1, banding=1, turnover=True
4. SingleSort, hp=1, banding=2, turnover=True
5. SingleSort, hp=3, no banding, turnover=True
6. SingleSort, hp=3, banding=1, turnover=True
7. DoubleSort (uncond), hp=1, no banding, turnover=True
8. DoubleSort (cond), hp=1, no banding, turnover=True
9. DoubleSort (uncond), hp=3, banding=1, turnover=True
10. DoubleSort (cond), hp=3, banding=1, turnover=True
11. SingleSort, hp=1, turnover=True, chars=[char1, char2]
12. SingleSort, hp=3, banding=1, turnover=True, chars=[char1, char2, char3]

Values Captured:
----------------
- Factor means: EW and VW long-short returns (mean across time)
- Turnover: Mean EW and VW turnover per portfolio
- Characteristics: Mean characteristic value per portfolio

Created: 2024
Last modified: 2024
"""

import sys
import os
import time
import json
import pickle
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# Import local PyBondLab (not installed package)
# =============================================================================
# Add parent directory to path for local development
_this_dir = Path(__file__).parent
sys.path.insert(0, str(_this_dir.parent))

from PyBondLab.PyBondLab import StrategyFormation
from PyBondLab.StrategyClass import SingleSort, DoubleSort
from PyBondLab.config import (
    StrategyFormationConfig,
    DataConfig,
    FormationConfig,
    FilterConfig
)

# =============================================================================
# Configuration
# =============================================================================

RANDOM_SEED = 42
N_DATES = 60          # Number of monthly periods (5 years) - enough for testing
N_BONDS = 500         # Total number of unique bonds
NPORT = 5             # Number of portfolios for tests
TOLERANCE = 1e-10     # Tolerance for comparing results

# Output paths
RESULTS_DIR = _this_dir / "baseline_results"

# =============================================================================
# Synthetic Data Generation
# =============================================================================

def generate_synthetic_data(
    n_dates: int = N_DATES,
    n_bonds: int = N_BONDS,
    seed: int = RANDOM_SEED,
    n_chars: int = 3
) -> pd.DataFrame:
    """
    Generate synthetic bond panel data for testing.

    Parameters
    ----------
    n_dates : int
        Number of monthly periods
    n_bonds : int
        Total number of unique bonds
    seed : int
        Random seed for reproducibility
    n_chars : int
        Number of random characteristics to generate

    Returns
    -------
    pd.DataFrame
        Synthetic bond panel with all required columns + characteristics
    """
    np.random.seed(seed)

    # Create date range
    dates = pd.date_range('2018-01-31', periods=n_dates, freq='ME')
    bond_ids = [f'BOND_{i:04d}' for i in range(n_bonds)]

    records = []

    for t_idx, date in enumerate(dates):
        # Vary number of active bonds (70-95% of total)
        pct_active = np.random.uniform(0.70, 0.95)
        n_active = int(n_bonds * pct_active)
        active_bonds = np.random.choice(bond_ids, size=n_active, replace=False)

        for bond_id in active_bonds:
            bond_idx = int(bond_id.split('_')[1])

            # Returns: cross-sectional mean + idiosyncratic
            market_ret = np.random.normal(0.004, 0.015)
            idio_ret = np.random.normal(0, 0.025)
            ret = market_ret + idio_ret

            # Rating: relatively stable with occasional changes
            base_rating = 3 + (bond_idx % 18)  # 3-20 (covers IG and NIG)
            rating_shock = np.random.choice([-1, 0, 0, 0, 0, 1])
            rating = int(np.clip(base_rating + rating_shock, 1, 21))

            # Value weight (market cap proxy)
            base_vw = 50 + (bond_idx % 450)
            vw_noise = np.random.uniform(0.9, 1.1)
            vw = base_vw * vw_noise

            # Primary sort variable (e.g., momentum or value signal)
            signal1 = np.random.randn()

            # Secondary sort variable
            signal2 = np.random.randn() * 0.5 + 0.3 * signal1  # Correlated

            # Price
            price = np.random.uniform(85, 115)

            record = {
                'date': date,
                'ID': bond_id,
                'ret': ret,
                'RATING_NUM': rating,
                'VW': vw,
                'PRICE': price,
                'signal1': signal1,
                'signal2': signal2,
            }

            # Add random characteristics
            for c in range(1, n_chars + 1):
                record[f'char{c}'] = np.random.uniform(-10, 10)

            records.append(record)

    df = pd.DataFrame(records)

    # Add some NaN values (realistic missing data)
    nan_mask_ret = np.random.random(len(df)) < 0.01
    nan_mask_signal = np.random.random(len(df)) < 0.02

    df.loc[nan_mask_ret, 'ret'] = np.nan
    df.loc[nan_mask_signal, 'signal1'] = np.nan

    # Sort by ID and date
    df = df.sort_values(['ID', 'date']).reset_index(drop=True)

    return df


def generate_synthetic_data_fast(
    n_dates: int,
    n_bonds: int,
    seed: int = 0,
    n_chars: int = 3,
    balanced_panel: bool = False,   # if True: full date x bond panel
    allow_nans: bool = True,        # if False: no NaNs anywhere
    pct_active_low: float = 0.70,   # used only if balanced_panel=False
    pct_active_high: float = 0.95,  # used only if balanced_panel=False
    exact_active_count: bool = True,# if False: faster Bernoulli mask per date
    id_as_category: bool = True,    # saves memory a lot for big panels
    float_dtype=np.float32,         # float32 is faster + smaller
) -> pd.DataFrame:
    """
    Fast synthetic bond panel generator (vectorized; optional balanced panel).

    For large panels (e.g., 300 dates x 10000 bonds = 3M rows), this is
    significantly faster than the row-by-row generate_synthetic_data function.

    Parameters
    ----------
    n_dates : int
        Number of monthly periods
    n_bonds : int
        Total number of unique bonds
    seed : int
        Random seed for reproducibility
    n_chars : int
        Number of random characteristics to generate
    balanced_panel : bool
        If True, creates full date x bond panel (no missing observations)
    allow_nans : bool
        If True, inject ~1-2% NaN values (realistic missing data)
    pct_active_low, pct_active_high : float
        Range of active bond percentage per date (only if balanced_panel=False)
    exact_active_count : bool
        If True, use exact counts per date; if False, use faster Bernoulli mask
    id_as_category : bool
        If True, convert ID to category dtype (saves memory)
    float_dtype : np.dtype
        Float dtype for numeric columns (float32 is faster + smaller)

    Returns
    -------
    pd.DataFrame
        Synthetic bond panel with columns: date, ID, ret, RATING_NUM, VW,
        PRICE, signal1, signal2, char1, char2, ...
    """
    rng = np.random.default_rng(seed)

    dates = pd.date_range("2018-01-31", periods=n_dates, freq="ME")
    bond_idx_all = np.arange(n_bonds, dtype=np.int32)
    bond_ids = np.array([f"BOND_{i:04d}" for i in range(n_bonds)], dtype=object)

    # ------------------------------------------------------------------
    # Build (date_idx, bond_idx) for either balanced or unbalanced panel
    # ------------------------------------------------------------------
    if balanced_panel:
        date_idx = np.repeat(np.arange(n_dates, dtype=np.int32), n_bonds)
        bond_idx = np.tile(bond_idx_all, n_dates)
        allow_nans = False  # balanced panel implies no missing values anywhere
    else:
        if exact_active_count:
            pct = rng.uniform(pct_active_low, pct_active_high, size=n_dates)
            n_active = (pct * n_bonds).astype(np.int32)

            date_parts = []
            bond_parts = []
            for t in range(n_dates):
                active = rng.choice(bond_idx_all, size=int(n_active[t]), replace=False)
                bond_parts.append(active.astype(np.int32, copy=False))
                date_parts.append(np.full(active.shape[0], t, dtype=np.int32))

            date_idx = np.concatenate(date_parts)
            bond_idx = np.concatenate(bond_parts)
        else:
            # Faster: Bernoulli mask per date (counts fluctuate around pct_active)
            pct = rng.uniform(pct_active_low, pct_active_high, size=n_dates).astype(float_dtype)
            mask = rng.random((n_dates, n_bonds)) < pct[:, None]
            date_idx, bond_idx = np.nonzero(mask)
            date_idx = date_idx.astype(np.int32, copy=False)
            bond_idx = bond_idx.astype(np.int32, copy=False)

    n = bond_idx.size

    # ------------------------------------------------------------------
    # Generate columns (vectorized)
    # ------------------------------------------------------------------
    # Market return per date; idiosyncratic per observation
    market_ret = rng.normal(0.004, 0.015, size=n_dates).astype(float_dtype)
    idio_ret = rng.normal(0.0, 0.025, size=n).astype(float_dtype)
    ret = market_ret[date_idx] + idio_ret

    # Add some extreme returns for filter testing (~2% of observations)
    # These will be useful for trim/bounce filter tests
    extreme_mask = rng.random(n) < 0.02
    extreme_returns = rng.choice(
        np.array([-0.50, -0.30, 0.30, 0.50, 0.80], dtype=float_dtype),
        size=n
    )
    ret = np.where(extreme_mask, extreme_returns, ret)

    # Rating: base by bond + occasional shock per observation
    base_rating = (3 + (bond_idx_all % 18)).astype(np.int16)  # 3..20
    shock = rng.choice(
        np.array([-1, 0, 1], dtype=np.int16),
        size=n,
        p=np.array([1/6, 4/6, 1/6], dtype=np.float64),
    )
    rating = np.clip(base_rating[bond_idx] + shock, 1, 21).astype(np.int16)

    # Value weight proxy
    base_vw = (50 + (bond_idx_all % 450)).astype(float_dtype)
    vw_noise = rng.uniform(0.9, 1.1, size=n).astype(float_dtype)
    vw = base_vw[bond_idx] * vw_noise

    # Signals
    signal1 = rng.standard_normal(size=n).astype(float_dtype)
    signal2 = (rng.standard_normal(size=n).astype(float_dtype) * float_dtype(0.5)
               + float_dtype(0.3) * signal1)

    # Price: wide range 1-1000 with realistic distribution
    # Most bonds 80-120, but some extreme values for filter testing
    base_price = rng.lognormal(mean=4.5, sigma=0.3, size=n)  # centered ~90
    # Add some extreme prices (~5% very low, ~5% very high)
    price_extreme_low = rng.random(n) < 0.05
    price_extreme_high = rng.random(n) < 0.05
    price = np.where(price_extreme_low, rng.uniform(1, 30, size=n),
                     np.where(price_extreme_high, rng.uniform(200, 1000, size=n),
                              np.clip(base_price, 50, 150)))
    price = price.astype(float_dtype)

    # Random characteristics
    chars = {
        f"char{c}": rng.uniform(-10, 10, size=n).astype(float_dtype)
        for c in range(1, n_chars + 1)
    }

    df = pd.DataFrame(
        {
            "date": dates.values[date_idx],
            "ID": bond_ids[bond_idx],
            "ret": ret,
            "RATING_NUM": rating,
            "VW": vw,
            "PRICE": price,
            "signal1": signal1,
            "signal2": signal2,
            **chars,
        }
    )

    # Optional NaNs injection (only if allowed)
    if allow_nans:
        nan_mask_ret = rng.random(n) < 0.01
        nan_mask_signal = rng.random(n) < 0.02
        df.loc[nan_mask_ret, "ret"] = np.nan
        df.loc[nan_mask_signal, "signal1"] = np.nan

    # Sort and optionally category-encode ID
    df = df.sort_values(["ID", "date"], kind="mergesort").reset_index(drop=True)
    if id_as_category:
        df["ID"] = df["ID"].astype("category")

    return df


# =============================================================================
# Result Container
# =============================================================================

@dataclass
class TestResult:
    """Container for test results (source of truth)."""
    test_name: str
    config: Dict[str, Any]

    # Factor returns
    ew_ls_mean: float       # EW long-short mean return
    vw_ls_mean: float       # VW long-short mean return
    ew_returns_by_ptf: List[float]  # EW returns by portfolio
    vw_returns_by_ptf: List[float]  # VW returns by portfolio

    # Turnover (if computed)
    ew_turnover_mean: Optional[float] = None
    vw_turnover_mean: Optional[float] = None
    ew_turnover_by_ptf: Optional[List[float]] = None
    vw_turnover_by_ptf: Optional[List[float]] = None

    # Characteristics (if computed)
    chars_ew_means: Optional[Dict[str, List[float]]] = None
    chars_vw_means: Optional[Dict[str, List[float]]] = None

    # Metadata
    n_dates: int = 0
    runtime_seconds: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'TestResult':
        """Create from dictionary."""
        return cls(**d)


# =============================================================================
# Test Runner
# =============================================================================

def run_single_test(
    data: pd.DataFrame,
    test_name: str,
    strategy_type: str,  # 'single' or 'double'
    holding_period: int,
    nport: int,
    sort_var: str,
    sort_var2: Optional[str] = None,
    how: str = 'unconditional',  # for double sort
    banding: Optional[int] = None,  # integer: 1, 2, etc.
    turnover: bool = False,
    chars: Optional[List[str]] = None,
    verbose: bool = True
) -> TestResult:
    """
    Run a single test configuration and capture results.

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    test_name : str
        Test identifier
    strategy_type : str
        'single' or 'double'
    holding_period : int
        Holding period
    nport : int
        Number of portfolios
    sort_var : str
        Primary sort variable
    sort_var2 : str, optional
        Secondary sort variable (for double sort)
    how : str
        'unconditional' or 'conditional' (for double sort)
    banding : int, optional
        Banding parameter (e.g., 1 means move 1 portfolio to switch)
    turnover : bool
        Whether to compute turnover
    chars : list of str, optional
        Characteristics to compute
    verbose : bool
        Print progress

    Returns
    -------
    TestResult
        Results container with source of truth values
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print(f"{'='*60}")

    t_start = time.time()

    # Create strategy
    if strategy_type == 'single':
        strategy = SingleSort(
            holding_period=holding_period,
            sort_var=sort_var,
            num_portfolios=nport,
            verbose=False
        )
        nport_total = nport
    else:  # double
        # For double sort, use nport for each dimension
        n1 = nport
        n2 = nport
        strategy = DoubleSort(
            holding_period=holding_period,
            sort_var=sort_var,
            sort_var2=sort_var2,
            num_portfolios=n1,
            num_portfolios2=n2,
            how=how,
            verbose=False
        )
        nport_total = n1 * n2

    # Convert banding integer to threshold
    banding_threshold = None
    if banding is not None:
        banding_threshold = banding / nport_total

    # Create configuration
    config = StrategyFormationConfig(
        data=DataConfig(
            rating=None,  # All bonds
            chars=chars
        ),
        formation=FormationConfig(
            dynamic_weights=True,  # Always True per requirements
            compute_turnover=turnover,
            save_idx=True,  # Need for characteristics
            banding_threshold=banding_threshold,
            verbose=False
        )
    )

    # Run portfolio formation
    sf = StrategyFormation(data=data, strategy=strategy, config=config)
    results = sf.fit()

    runtime = time.time() - t_start

    # Extract results
    # Get the EA (ex-ante) results from FormationResults
    ea = results.ea  # StrategyResults object

    # Get returns DataFrames
    ew_ret_df = ea.returns.ewport_df
    vw_ret_df = ea.returns.vwport_df

    # Portfolio labels
    ptf_labels = list(ew_ret_df.columns)
    n_ptf = len(ptf_labels)

    # Compute mean returns by portfolio
    ew_returns_by_ptf = [float(ew_ret_df[col].mean()) for col in ptf_labels]
    vw_returns_by_ptf = [float(vw_ret_df[col].mean()) for col in ptf_labels]

    # Long-short return (last portfolio - first portfolio)
    ew_ls = ew_ret_df[ptf_labels[-1]] - ew_ret_df[ptf_labels[0]]
    vw_ls = vw_ret_df[ptf_labels[-1]] - vw_ret_df[ptf_labels[0]]

    ew_ls_mean = float(ew_ls.mean())
    vw_ls_mean = float(vw_ls.mean())

    # Turnover
    ew_turnover_mean = None
    vw_turnover_mean = None
    ew_turnover_by_ptf = None
    vw_turnover_by_ptf = None

    if turnover and ea.turnover is not None:
        ew_to_df = ea.turnover.ew_turnover_df
        vw_to_df = ea.turnover.vw_turnover_df

        if ew_to_df is not None and not ew_to_df.empty:
            ew_turnover_by_ptf = [float(ew_to_df[col].mean()) for col in ew_to_df.columns]
            vw_turnover_by_ptf = [float(vw_to_df[col].mean()) for col in vw_to_df.columns]
            ew_turnover_mean = float(np.nanmean(ew_turnover_by_ptf))
            vw_turnover_mean = float(np.nanmean(vw_turnover_by_ptf))

    # Characteristics
    chars_ew_means = None
    chars_vw_means = None

    if chars and ea.characteristics is not None:
        chars_ew_means = {}
        chars_vw_means = {}

        for char_name in chars:
            if ea.characteristics.ew_chars is not None:
                if char_name in ea.characteristics.ew_chars:
                    char_df = ea.characteristics.ew_chars[char_name]
                    chars_ew_means[char_name] = [float(char_df[col].mean()) for col in char_df.columns]

            if ea.characteristics.vw_chars is not None:
                if char_name in ea.characteristics.vw_chars:
                    char_df = ea.characteristics.vw_chars[char_name]
                    chars_vw_means[char_name] = [float(char_df[col].mean()) for col in char_df.columns]

    # Build config dict for storage
    config_dict = {
        'strategy_type': strategy_type,
        'holding_period': holding_period,
        'nport': nport,
        'nport_total': nport_total,
        'sort_var': sort_var,
        'sort_var2': sort_var2,
        'how': how if strategy_type == 'double' else None,
        'banding': banding,
        'banding_threshold': banding_threshold,
        'turnover': turnover,
        'chars': chars,
        'dynamic_weights': True
    }

    result = TestResult(
        test_name=test_name,
        config=config_dict,
        ew_ls_mean=ew_ls_mean,
        vw_ls_mean=vw_ls_mean,
        ew_returns_by_ptf=ew_returns_by_ptf,
        vw_returns_by_ptf=vw_returns_by_ptf,
        ew_turnover_mean=ew_turnover_mean,
        vw_turnover_mean=vw_turnover_mean,
        ew_turnover_by_ptf=ew_turnover_by_ptf,
        vw_turnover_by_ptf=vw_turnover_by_ptf,
        chars_ew_means=chars_ew_means,
        chars_vw_means=chars_vw_means,
        n_dates=len(ew_ret_df),
        runtime_seconds=runtime
    )

    if verbose:
        print(f"  Runtime: {runtime:.2f}s")
        print(f"  EW L-S mean: {ew_ls_mean:.6f}")
        print(f"  VW L-S mean: {vw_ls_mean:.6f}")
        if turnover and ew_turnover_mean is not None:
            print(f"  EW Turnover mean: {ew_turnover_mean:.4f}")
            print(f"  VW Turnover mean: {vw_turnover_mean:.4f}")
        if chars and chars_ew_means:
            print(f"  Chars computed: {list(chars_ew_means.keys())}")

    return result


def compare_results(baseline: TestResult, new_result: TestResult, tol: float = TOLERANCE) -> bool:
    """
    Compare new results against baseline (source of truth).

    Returns True if all values match within tolerance.
    """
    all_passed = True

    print(f"\nComparing: {baseline.test_name}")

    # Compare L-S means
    ew_diff = abs(baseline.ew_ls_mean - new_result.ew_ls_mean)
    vw_diff = abs(baseline.vw_ls_mean - new_result.vw_ls_mean)

    if ew_diff > tol:
        print(f"  FAIL: EW L-S mean differs by {ew_diff:.2e}")
        all_passed = False
    else:
        print(f"  PASS: EW L-S mean matches")

    if vw_diff > tol:
        print(f"  FAIL: VW L-S mean differs by {vw_diff:.2e}")
        all_passed = False
    else:
        print(f"  PASS: VW L-S mean matches")

    # Compare portfolio returns
    for i, (b_ew, n_ew) in enumerate(zip(baseline.ew_returns_by_ptf, new_result.ew_returns_by_ptf)):
        diff = abs(b_ew - n_ew)
        if diff > tol:
            print(f"  FAIL: EW portfolio {i+1} return differs by {diff:.2e}")
            all_passed = False

    # Compare turnover if applicable
    if baseline.ew_turnover_mean is not None:
        if new_result.ew_turnover_mean is None:
            print(f"  FAIL: EW turnover missing in new result")
            all_passed = False
        else:
            to_diff = abs(baseline.ew_turnover_mean - new_result.ew_turnover_mean)
            if to_diff > tol:
                print(f"  FAIL: EW turnover differs by {to_diff:.2e}")
                all_passed = False
            else:
                print(f"  PASS: EW turnover matches")

    if baseline.vw_turnover_mean is not None and new_result.vw_turnover_mean is not None:
        to_diff = abs(baseline.vw_turnover_mean - new_result.vw_turnover_mean)
        if to_diff > tol:
            print(f"  FAIL: VW turnover differs by {to_diff:.2e}")
            all_passed = False
        else:
            print(f"  PASS: VW turnover matches")

    # Compare characteristics if applicable
    if baseline.chars_ew_means is not None:
        for char_name, baseline_vals in baseline.chars_ew_means.items():
            if new_result.chars_ew_means is None or char_name not in new_result.chars_ew_means:
                print(f"  FAIL: Char '{char_name}' missing in new result")
                all_passed = False
            else:
                new_vals = new_result.chars_ew_means[char_name]
                max_diff = max(abs(b - n) for b, n in zip(baseline_vals, new_vals))
                if max_diff > tol:
                    print(f"  FAIL: Char '{char_name}' differs by {max_diff:.2e}")
                    all_passed = False
                else:
                    print(f"  PASS: Char '{char_name}' matches")

    return all_passed


# =============================================================================
# Main Test Suite
# =============================================================================

def run_all_baseline_tests(data: pd.DataFrame, verbose: bool = True) -> Dict[str, TestResult]:
    """
    Run all baseline test configurations.

    Returns dictionary of test_name -> TestResult
    """
    results = {}

    # Test 1: SingleSort, hp=1, no banding, no turnover
    results['test_01_single_hp1_noband_noturn'] = run_single_test(
        data=data,
        test_name='test_01_single_hp1_noband_noturn',
        strategy_type='single',
        holding_period=1,
        nport=NPORT,
        sort_var='signal1',
        banding=None,
        turnover=False,
        chars=None,
        verbose=verbose
    )

    # Test 2: SingleSort, hp=1, no banding, turnover=True
    results['test_02_single_hp1_noband_turn'] = run_single_test(
        data=data,
        test_name='test_02_single_hp1_noband_turn',
        strategy_type='single',
        holding_period=1,
        nport=NPORT,
        sort_var='signal1',
        banding=None,
        turnover=True,
        chars=None,
        verbose=verbose
    )

    # Test 3: SingleSort, hp=1, banding=1, turnover=True
    results['test_03_single_hp1_band1_turn'] = run_single_test(
        data=data,
        test_name='test_03_single_hp1_band1_turn',
        strategy_type='single',
        holding_period=1,
        nport=NPORT,
        sort_var='signal1',
        banding=1,
        turnover=True,
        chars=None,
        verbose=verbose
    )

    # Test 4: SingleSort, hp=1, banding=2, turnover=True
    results['test_04_single_hp1_band2_turn'] = run_single_test(
        data=data,
        test_name='test_04_single_hp1_band2_turn',
        strategy_type='single',
        holding_period=1,
        nport=NPORT,
        sort_var='signal1',
        banding=2,
        turnover=True,
        chars=None,
        verbose=verbose
    )

    # Test 5: SingleSort, hp=3, no banding, turnover=True
    results['test_05_single_hp3_noband_turn'] = run_single_test(
        data=data,
        test_name='test_05_single_hp3_noband_turn',
        strategy_type='single',
        holding_period=3,
        nport=NPORT,
        sort_var='signal1',
        banding=None,
        turnover=True,
        chars=None,
        verbose=verbose
    )

    # Test 6: SingleSort, hp=3, banding=1, turnover=True
    results['test_06_single_hp3_band1_turn'] = run_single_test(
        data=data,
        test_name='test_06_single_hp3_band1_turn',
        strategy_type='single',
        holding_period=3,
        nport=NPORT,
        sort_var='signal1',
        banding=1,
        turnover=True,
        chars=None,
        verbose=verbose
    )

    # Test 7: DoubleSort (unconditional), hp=1, no banding, turnover=True
    results['test_07_double_uncond_hp1_noband_turn'] = run_single_test(
        data=data,
        test_name='test_07_double_uncond_hp1_noband_turn',
        strategy_type='double',
        holding_period=1,
        nport=NPORT,
        sort_var='signal1',
        sort_var2='signal2',
        how='unconditional',
        banding=None,
        turnover=True,
        chars=None,
        verbose=verbose
    )

    # Test 8: DoubleSort (conditional), hp=1, no banding, turnover=True
    results['test_08_double_cond_hp1_noband_turn'] = run_single_test(
        data=data,
        test_name='test_08_double_cond_hp1_noband_turn',
        strategy_type='double',
        holding_period=1,
        nport=NPORT,
        sort_var='signal1',
        sort_var2='signal2',
        how='conditional',
        banding=None,
        turnover=True,
        chars=None,
        verbose=verbose
    )

    # Test 9: DoubleSort (unconditional), hp=3, banding=1, turnover=True
    results['test_09_double_uncond_hp3_band1_turn'] = run_single_test(
        data=data,
        test_name='test_09_double_uncond_hp3_band1_turn',
        strategy_type='double',
        holding_period=3,
        nport=NPORT,
        sort_var='signal1',
        sort_var2='signal2',
        how='unconditional',
        banding=1,
        turnover=True,
        chars=None,
        verbose=verbose
    )

    # Test 10: DoubleSort (conditional), hp=3, banding=1, turnover=True
    results['test_10_double_cond_hp3_band1_turn'] = run_single_test(
        data=data,
        test_name='test_10_double_cond_hp3_band1_turn',
        strategy_type='double',
        holding_period=3,
        nport=NPORT,
        sort_var='signal1',
        sort_var2='signal2',
        how='conditional',
        banding=1,
        turnover=True,
        chars=None,
        verbose=verbose
    )

    # Test 11: SingleSort, hp=1, turnover=True, chars=[char1, char2]
    results['test_11_single_hp1_turn_chars2'] = run_single_test(
        data=data,
        test_name='test_11_single_hp1_turn_chars2',
        strategy_type='single',
        holding_period=1,
        nport=NPORT,
        sort_var='signal1',
        banding=None,
        turnover=True,
        chars=['char1', 'char2'],
        verbose=verbose
    )

    # Test 12: SingleSort, hp=3, banding=1, turnover=True, chars=[char1, char2, char3]
    results['test_12_single_hp3_band1_turn_chars3'] = run_single_test(
        data=data,
        test_name='test_12_single_hp3_band1_turn_chars3',
        strategy_type='single',
        holding_period=3,
        nport=NPORT,
        sort_var='signal1',
        banding=1,
        turnover=True,
        chars=['char1', 'char2', 'char3'],
        verbose=verbose
    )

    return results


def save_baseline_results(results: Dict[str, TestResult], output_dir: Path = RESULTS_DIR):
    """Save baseline results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as JSON (human readable)
    json_data = {name: result.to_dict() for name, result in results.items()}
    json_path = output_dir / 'baseline_results.json'
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"\nSaved JSON results to: {json_path}")

    # Save as pickle (for exact floating point preservation)
    pickle_path = output_dir / 'baseline_results.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved pickle results to: {pickle_path}")


def load_baseline_results(input_dir: Path = RESULTS_DIR) -> Dict[str, TestResult]:
    """Load baseline results from disk."""
    pickle_path = input_dir / 'baseline_results.pkl'
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)


# =============================================================================
# Summary Report
# =============================================================================

def print_summary(results: Dict[str, TestResult]):
    """Print summary of all test results."""
    print("\n" + "=" * 70)
    print("BASELINE TEST SUMMARY (SOURCE OF TRUTH)")
    print("=" * 70)

    total_runtime = sum(r.runtime_seconds for r in results.values())

    print(f"\nTotal tests: {len(results)}")
    print(f"Total runtime: {total_runtime:.2f}s")
    print("\n" + "-" * 70)
    print(f"{'Test Name':<45} {'EW L-S':>10} {'VW L-S':>10} {'Time':>8}")
    print("-" * 70)

    for name, result in results.items():
        short_name = name.replace('test_', '').replace('_', ' ')[:42]
        print(f"{short_name:<45} {result.ew_ls_mean:>10.6f} {result.vw_ls_mean:>10.6f} {result.runtime_seconds:>7.2f}s")

    print("-" * 70)

    # Turnover summary
    print("\nTurnover Summary (tests with turnover=True):")
    print(f"{'Test Name':<45} {'EW TO':>10} {'VW TO':>10}")
    print("-" * 70)

    for name, result in results.items():
        if result.ew_turnover_mean is not None:
            short_name = name.replace('test_', '').replace('_', ' ')[:42]
            print(f"{short_name:<45} {result.ew_turnover_mean:>10.4f} {result.vw_turnover_mean:>10.4f}")

    print("-" * 70)


# =============================================================================
# Validation Function (for comparing optimized code)
# =============================================================================

def validate_against_baseline(
    data: pd.DataFrame,
    baseline_results: Dict[str, TestResult],
    run_test_func=run_single_test,
    verbose: bool = True
) -> bool:
    """
    Validate new/optimized code against baseline results.

    Parameters
    ----------
    data : pd.DataFrame
        Same synthetic data used for baseline (same seed!)
    baseline_results : dict
        Loaded baseline results
    run_test_func : callable
        Function to run a single test (default or optimized version)
    verbose : bool
        Print details

    Returns
    -------
    bool
        True if all tests pass
    """
    all_passed = True

    print("\n" + "=" * 70)
    print("VALIDATING AGAINST BASELINE (SOURCE OF TRUTH)")
    print("=" * 70)

    for test_name, baseline in baseline_results.items():
        cfg = baseline.config

        # Run the test with new implementation
        new_result = run_test_func(
            data=data,
            test_name=test_name,
            strategy_type=cfg['strategy_type'],
            holding_period=cfg['holding_period'],
            nport=cfg['nport'],
            sort_var=cfg['sort_var'],
            sort_var2=cfg.get('sort_var2'),
            how=cfg.get('how', 'unconditional'),
            banding=cfg.get('banding'),
            turnover=cfg['turnover'],
            chars=cfg.get('chars'),
            verbose=False
        )

        # Compare
        passed = compare_results(baseline, new_result, tol=TOLERANCE)
        if not passed:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED - Optimized code matches baseline!")
    else:
        print("SOME TESTS FAILED - Check differences above")
    print("=" * 70)

    return all_passed


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for running baseline tests."""
    print("=" * 70)
    print("PyBondLab Baseline Test Suite")
    print("Establishing SOURCE OF TRUTH results")
    print("=" * 70)

    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    t0 = time.time()
    data = generate_synthetic_data(
        n_dates=N_DATES,
        n_bonds=N_BONDS,
        seed=RANDOM_SEED,
        n_chars=3
    )
    print(f"   Generated: {len(data):,} rows, {data['ID'].nunique()} bonds, {data['date'].nunique()} dates")
    print(f"   Time: {time.time() - t0:.2f}s")

    # Run all baseline tests
    print("\n2. Running baseline tests...")
    results = run_all_baseline_tests(data, verbose=True)

    # Print summary
    print_summary(results)

    # Save results
    print("\n3. Saving baseline results...")
    save_baseline_results(results)

    print("\n" + "=" * 70)
    print("BASELINE TESTS COMPLETE")
    print("These results are the SOURCE OF TRUTH for future optimization validation")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = main()
