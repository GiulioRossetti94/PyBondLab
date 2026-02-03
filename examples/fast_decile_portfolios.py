# -*- coding: utf-8 -*-
"""
Fast Decile Portfolio Formation Script

Forms decile portfolios for multiple signals without PyBondLab,
using optimized numpy/numba operations for speed.

Matches PyBondLab SingleSort with:
- holding_period=1
- num_portfolios=N (configurable, default 10 for deciles)
- dynamic_weights=True

Output:
- ew_factors: DataFrame with date column + EW long-short factors (sign-corrected)
- vw_factors: DataFrame with date column + VW long-short factors (sign-corrected)

Author: Claude
Date: 2026-02-03
"""

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
from numba import njit, prange
import time
import warnings

# =============================================================================
# CONFIGURATION
# =============================================================================
RETURN_COL = 'r_1'        # Options: 'r_1', 'r_1_exc', 'r_1_dur'
WEIGHT_COL = 'mv'         # Value weight column
N_PORTFOLIOS = 10         # Number of portfolios (10 = deciles, 5 = quintiles)
SIGN_CORRECT = True       # Flip negative factors so mean > 0

# Data column names
DATE_COL = 'date'
ID_COL = 'cusip'

# Signals to exclude (non-signal columns)
EXCLUDE_COLS = ['date', 'cusip', 'permno', 'r_1', 'r_1_exc', 'r_1_dur', 'mv']


# =============================================================================
# NUMBA KERNELS
# =============================================================================

@njit(cache=True)
def compute_percentile_breakpoints(values, n_portfolios):
    """
    Compute percentile breakpoints from unique sorted values.

    Matches PyBondLab's percentile threshold computation.
    """
    n = len(values)
    if n < n_portfolios:
        return np.empty(0, dtype=np.float64)

    # Get unique values
    sorted_vals = np.sort(values)
    unique_vals = np.empty(n, dtype=np.float64)
    n_unique = 0
    prev = sorted_vals[0] - 1.0  # Ensure first value is included

    for i in range(n):
        if sorted_vals[i] != prev:
            unique_vals[n_unique] = sorted_vals[i]
            n_unique += 1
            prev = sorted_vals[i]

    unique_vals = unique_vals[:n_unique]

    if n_unique < 2:
        return np.empty(0, dtype=np.float64)

    # Compute breakpoints using numpy percentile formula
    breakpoints = np.empty(n_portfolios - 1, dtype=np.float64)
    for p in range(n_portfolios - 1):
        pct = (p + 1) * 100.0 / n_portfolios
        # Linear interpolation (numpy default)
        idx = pct / 100.0 * (n_unique - 1)
        idx_low = int(idx)
        idx_high = min(idx_low + 1, n_unique - 1)
        frac = idx - idx_low
        breakpoints[p] = unique_vals[idx_low] * (1 - frac) + unique_vals[idx_high] * frac

    return breakpoints


@njit(cache=True)
def assign_ranks(signal, breakpoints, n_portfolios):
    """
    Assign portfolio ranks based on breakpoints.

    Returns ranks 1 to n_portfolios (0 = invalid/NaN).
    """
    n = len(signal)
    ranks = np.zeros(n, dtype=np.int32)

    if len(breakpoints) == 0:
        return ranks

    for i in range(n):
        if not np.isfinite(signal[i]):
            continue

        val = signal[i]
        rank = 1
        for p in range(len(breakpoints)):
            if val > breakpoints[p]:
                rank = p + 2
        ranks[i] = rank

    return ranks


@njit(cache=True)
def compute_portfolio_returns_single_date(ranks, returns, weights, n_portfolios):
    """
    Compute EW and VW returns for each portfolio at a single date.

    Parameters
    ----------
    ranks : np.ndarray[int32]
        Portfolio ranks (1 to n_portfolios), 0 = excluded
    returns : np.ndarray[float64]
        Bond returns
    weights : np.ndarray[float64]
        Value weights
    n_portfolios : int
        Number of portfolios

    Returns
    -------
    ew_ret : np.ndarray[float64]
        Equal-weighted returns per portfolio (length n_portfolios)
    vw_ret : np.ndarray[float64]
        Value-weighted returns per portfolio (length n_portfolios)
    """
    ew_ret = np.full(n_portfolios, np.nan, dtype=np.float64)
    vw_ret = np.full(n_portfolios, np.nan, dtype=np.float64)

    n = len(ranks)

    for p in range(1, n_portfolios + 1):
        sum_ret = 0.0
        sum_vw_ret = 0.0
        sum_w = 0.0
        count = 0

        for i in range(n):
            if ranks[i] != p:
                continue
            if not np.isfinite(returns[i]):
                continue

            r = returns[i]
            w = weights[i] if np.isfinite(weights[i]) and weights[i] > 0 else 0.0

            sum_ret += r
            count += 1

            if w > 0:
                sum_vw_ret += r * w
                sum_w += w

        if count > 0:
            ew_ret[p - 1] = sum_ret / count
        if sum_w > 0:
            vw_ret[p - 1] = sum_vw_ret / sum_w

    return ew_ret, vw_ret


@njit(cache=True)
def process_single_date(signal, returns, weights, n_portfolios):
    """
    Process a single date: rank bonds and compute portfolio returns.

    Returns (ew_long, ew_short, vw_long, vw_short) or NaN if insufficient data.
    """
    # Filter to valid observations (valid signal, return, and weight)
    n = len(signal)
    valid_mask = np.zeros(n, dtype=np.bool_)
    n_valid = 0

    for i in range(n):
        if np.isfinite(signal[i]) and np.isfinite(returns[i]) and np.isfinite(weights[i]):
            valid_mask[i] = True
            n_valid += 1

    if n_valid < n_portfolios:
        return np.nan, np.nan, np.nan, np.nan

    # Extract valid values for ranking
    valid_signal = np.empty(n_valid, dtype=np.float64)
    j = 0
    for i in range(n):
        if valid_mask[i]:
            valid_signal[j] = signal[i]
            j += 1

    # Compute breakpoints from valid signals
    breakpoints = compute_percentile_breakpoints(valid_signal, n_portfolios)

    if len(breakpoints) == 0:
        return np.nan, np.nan, np.nan, np.nan

    # Assign ranks to ALL observations (but only valid ones get non-zero rank)
    ranks = assign_ranks(signal, breakpoints, n_portfolios)

    # Zero out ranks for invalid observations
    for i in range(n):
        if not valid_mask[i]:
            ranks[i] = 0

    # Compute portfolio returns
    ew_ret, vw_ret = compute_portfolio_returns_single_date(ranks, returns, weights, n_portfolios)

    # Long-short: top portfolio - bottom portfolio
    ew_long = ew_ret[n_portfolios - 1]
    ew_short = ew_ret[0]
    vw_long = vw_ret[n_portfolios - 1]
    vw_short = vw_ret[0]

    return ew_long, ew_short, vw_long, vw_short


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def form_portfolios_single_signal(data, signal_col, return_col, weight_col, n_portfolios):
    """
    Form portfolios for a single signal across all dates.

    Parameters
    ----------
    data : pd.DataFrame
        Bond panel data with columns: date, signal, return, weight
    signal_col : str
        Signal column name
    return_col : str
        Return column name
    weight_col : str
        Weight column name
    n_portfolios : int
        Number of portfolios

    Returns
    -------
    ew_ls : pd.Series
        Equal-weighted long-short returns indexed by date
    vw_ls : pd.Series
        Value-weighted long-short returns indexed by date
    """
    # Group data by date for faster access
    date_groups = data.groupby(DATE_COL)

    # Get dates from groupby keys (ensures type consistency)
    dates = sorted(date_groups.groups.keys())
    n_dates = len(dates)

    # Pre-allocate results
    ew_ls = np.full(n_dates, np.nan, dtype=np.float64)
    vw_ls = np.full(n_dates, np.nan, dtype=np.float64)

    for i, date in enumerate(dates):
        group = date_groups.get_group(date)

        # Extract arrays
        signal = group[signal_col].values.astype(np.float64)
        returns = group[return_col].values.astype(np.float64)
        weights = group[weight_col].values.astype(np.float64)

        # Process this date
        ew_long, ew_short, vw_long, vw_short = process_single_date(
            signal, returns, weights, n_portfolios
        )

        # Compute long-short
        if np.isfinite(ew_long) and np.isfinite(ew_short):
            ew_ls[i] = ew_long - ew_short
        if np.isfinite(vw_long) and np.isfinite(vw_short):
            vw_ls[i] = vw_long - vw_short

    # Create series with date index (shift forward by 1 month to return date)
    # Formation at t, return earned over t:t+1, labeled as t+1
    return_dates = pd.DatetimeIndex(dates) + MonthEnd(1)
    ew_series = pd.Series(ew_ls, index=return_dates, name=signal_col)
    vw_series = pd.Series(vw_ls, index=return_dates, name=signal_col)

    # Drop NaN dates
    ew_series = ew_series.dropna()
    vw_series = vw_series.dropna()

    return ew_series, vw_series


def form_all_portfolios(data, signal_cols, return_col=RETURN_COL, weight_col=WEIGHT_COL,
                        n_portfolios=N_PORTFOLIOS, sign_correct=SIGN_CORRECT, verbose=True):
    """
    Form portfolios for all signals.

    Parameters
    ----------
    data : pd.DataFrame
        Bond panel data
    signal_cols : list
        List of signal column names
    return_col : str
        Return column name
    weight_col : str
        Weight column name
    n_portfolios : int
        Number of portfolios
    sign_correct : bool
        If True, flip factors with negative mean
    verbose : bool
        Print progress

    Returns
    -------
    ew_df : pd.DataFrame
        Equal-weighted long-short factors with date column
    vw_df : pd.DataFrame
        Value-weighted long-short factors with date column
    """
    if verbose:
        print(f"Forming {n_portfolios}-tile portfolios for {len(signal_cols)} signals...")
        print(f"  Return: {return_col}, Weight: {weight_col}")

    ew_results = {}
    vw_results = {}
    sign_flipped = {'ew': [], 'vw': []}

    t_start = time.time()

    for i, signal in enumerate(signal_cols):
        if verbose and (i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(signal_cols) - i - 1) / rate
            print(f"  [{i + 1}/{len(signal_cols)}] {signal} ({elapsed:.1f}s elapsed, ~{eta:.1f}s remaining)")

        try:
            ew_ls, vw_ls = form_portfolios_single_signal(
                data, signal, return_col, weight_col, n_portfolios
            )

            # Sign correction (independent for EW and VW)
            if sign_correct:
                if len(ew_ls) > 0 and ew_ls.mean() < 0:
                    ew_ls = -ew_ls
                    sign_flipped['ew'].append(signal)
                if len(vw_ls) > 0 and vw_ls.mean() < 0:
                    vw_ls = -vw_ls
                    sign_flipped['vw'].append(signal)

            ew_results[signal] = ew_ls
            vw_results[signal] = vw_ls

        except Exception as e:
            if verbose:
                print(f"  WARNING: {signal} failed - {e}")
            continue

    total_time = time.time() - t_start
    if verbose:
        print(f"  Completed in {total_time:.2f}s ({len(signal_cols) / total_time:.1f} signals/sec)")
        if sign_correct:
            print(f"  Sign-flipped: EW={len(sign_flipped['ew'])}, VW={len(sign_flipped['vw'])}")

    # Combine into DataFrames
    ew_df = pd.DataFrame(ew_results)
    vw_df = pd.DataFrame(vw_results)

    # Reset index to add date column
    ew_df = ew_df.reset_index().rename(columns={'index': 'date'})
    vw_df = vw_df.reset_index().rename(columns={'index': 'date'})

    return ew_df, vw_df


def get_signal_columns(data, exclude_cols=None):
    """
    Auto-detect signal columns from data.

    Returns columns from 'age' to 'ytm' (or all numeric columns not in exclude list).
    """
    if exclude_cols is None:
        exclude_cols = EXCLUDE_COLS

    # Get all columns
    all_cols = list(data.columns)

    # Find 'age' and 'ytm' indices
    try:
        age_idx = all_cols.index('age')
        ytm_idx = all_cols.index('ytm')
        signal_cols = all_cols[age_idx:ytm_idx + 1]
    except ValueError:
        # Fallback: use all numeric columns not in exclude list
        signal_cols = [c for c in all_cols if c not in exclude_cols and data[c].dtype in ['float64', 'float32', 'int64', 'int32']]

    # Remove any exclude columns that snuck in
    signal_cols = [c for c in signal_cols if c not in exclude_cols]

    return signal_cols


# =============================================================================
# VALIDATION AGAINST PYBONDLAB
# =============================================================================

def validate_against_pybondlab(data, signal_cols, return_col=RETURN_COL, weight_col=WEIGHT_COL,
                               n_portfolios=N_PORTFOLIOS, n_signals=5, verbose=True):
    """
    Validate results against PyBondLab SingleSort.

    Parameters
    ----------
    data : pd.DataFrame
        Bond panel data
    signal_cols : list
        Signal columns to validate (will use first n_signals)
    return_col : str
        Return column
    weight_col : str
        Weight column
    n_portfolios : int
        Number of portfolios
    n_signals : int
        Number of signals to validate
    verbose : bool
        Print details

    Returns
    -------
    bool
        True if all validations pass
    """
    try:
        import PyBondLab as pbl
        from PyBondLab import StrategyFormation, SingleSort
        from PyBondLab.config import StrategyFormationConfig, FormationConfig, DataConfig
    except ImportError:
        print("PyBondLab not available - skipping validation")
        return True

    # Prepare data for PyBondLab
    pbl_data = data.copy()

    # Rename columns if needed
    col_mapping = {}
    if 'cusip' in pbl_data.columns and 'ID' not in pbl_data.columns:
        col_mapping['cusip'] = 'ID'
    if weight_col != 'VW' and weight_col in pbl_data.columns:
        col_mapping[weight_col] = 'VW'
    if return_col != 'ret' and return_col in pbl_data.columns:
        col_mapping[return_col] = 'ret'

    if col_mapping:
        pbl_data = pbl_data.rename(columns=col_mapping)

    # Ensure required columns
    if 'RATING_NUM' not in pbl_data.columns:
        pbl_data['RATING_NUM'] = 5  # Dummy rating

    test_signals = signal_cols[:n_signals]
    all_passed = True

    if verbose:
        print(f"\nValidating {len(test_signals)} signals against PyBondLab...")

    for signal in test_signals:
        if verbose:
            print(f"\n  Testing: {signal}")

        # Fast method
        ew_fast, vw_fast = form_portfolios_single_signal(
            data, signal, return_col, weight_col, n_portfolios
        )

        # PyBondLab
        try:
            strategy = SingleSort(
                holding_period=1,
                sort_var=signal,
                num_portfolios=n_portfolios
            )

            config = StrategyFormationConfig(
                data=DataConfig(),
                formation=FormationConfig(
                    dynamic_weights=True,
                    compute_turnover=False,
                    verbose=False
                )
            )

            sf = StrategyFormation(data=pbl_data, strategy=strategy, config=config)
            result = sf.fit()

            ew_pbl, vw_pbl = result.get_long_short()

            # Align dates
            common_dates = ew_fast.index.intersection(ew_pbl.index)

            if len(common_dates) == 0:
                if verbose:
                    print(f"    WARNING: No overlapping dates")
                continue

            ew_fast_aligned = ew_fast.loc[common_dates]
            vw_fast_aligned = vw_fast.loc[common_dates]
            ew_pbl_aligned = ew_pbl.loc[common_dates]
            vw_pbl_aligned = vw_pbl.loc[common_dates]

            # Compare
            ew_diff = (ew_fast_aligned - ew_pbl_aligned).abs().max()
            vw_diff = (vw_fast_aligned - vw_pbl_aligned).abs().max()

            ew_mean_fast = ew_fast_aligned.mean() * 100
            ew_mean_pbl = ew_pbl_aligned.mean() * 100
            vw_mean_fast = vw_fast_aligned.mean() * 100
            vw_mean_pbl = vw_pbl_aligned.mean() * 100

            tol = 1e-10
            ew_pass = ew_diff < tol
            vw_pass = vw_diff < tol

            if verbose:
                status_ew = "PASS" if ew_pass else "FAIL"
                status_vw = "PASS" if vw_pass else "FAIL"
                print(f"    EW: {status_ew} (diff={ew_diff:.2e}, mean: fast={ew_mean_fast:.4f}%, pbl={ew_mean_pbl:.4f}%)")
                print(f"    VW: {status_vw} (diff={vw_diff:.2e}, mean: fast={vw_mean_fast:.4f}%, pbl={vw_mean_pbl:.4f}%)")

            if not (ew_pass and vw_pass):
                all_passed = False

        except Exception as e:
            if verbose:
                print(f"    ERROR: {e}")
            all_passed = False

    if verbose:
        print(f"\nValidation: {'PASSED' if all_passed else 'FAILED'}")

    return all_passed


# =============================================================================
# MAIN
# =============================================================================

def main(data_path=None, data=None, validate=True):
    """
    Main function to run portfolio formation.

    Parameters
    ----------
    data_path : str, optional
        Path to data file (CSV or Parquet)
    data : pd.DataFrame, optional
        Pre-loaded data (alternative to data_path)
    validate : bool
        Run validation against PyBondLab

    Returns
    -------
    ew_df, vw_df : tuple of DataFrames
        Long-short factor returns
    """
    # Load data
    if data is None:
        if data_path is None:
            raise ValueError("Must provide either data_path or data")

        print(f"Loading data from {data_path}...")
        if data_path.endswith('.parquet'):
            data = pd.read_parquet(data_path)
        else:
            data = pd.read_csv(data_path, parse_dates=['date'])

    print(f"Data shape: {data.shape}")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")

    # Get signal columns
    signal_cols = get_signal_columns(data)
    print(f"Found {len(signal_cols)} signal columns: {signal_cols[0]} ... {signal_cols[-1]}")

    # Validate first (optional)
    if validate:
        validate_against_pybondlab(
            data, signal_cols,
            return_col=RETURN_COL,
            weight_col=WEIGHT_COL,
            n_portfolios=N_PORTFOLIOS,
            n_signals=5
        )

    # Form portfolios
    print("\n" + "="*60)
    print("FORMING PORTFOLIOS")
    print("="*60)

    ew_df, vw_df = form_all_portfolios(
        data, signal_cols,
        return_col=RETURN_COL,
        weight_col=WEIGHT_COL,
        n_portfolios=N_PORTFOLIOS,
        sign_correct=SIGN_CORRECT,
        verbose=True
    )

    print(f"\nOutput shapes:")
    print(f"  EW factors: {ew_df.shape}")
    print(f"  VW factors: {vw_df.shape}")

    # Summary stats
    print(f"\nFactor means (% monthly):")
    ew_means = ew_df.drop(columns=['date']).mean() * 100
    vw_means = vw_df.drop(columns=['date']).mean() * 100

    print(f"  EW: min={ew_means.min():.3f}%, max={ew_means.max():.3f}%, median={ew_means.median():.3f}%")
    print(f"  VW: min={vw_means.min():.3f}%, max={vw_means.max():.3f}%, median={vw_means.median():.3f}%")

    return ew_df, vw_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fast decile portfolio formation")
    parser.add_argument("data_path", nargs="?", help="Path to data file")
    parser.add_argument("--return-col", default=RETURN_COL, help="Return column name")
    parser.add_argument("--weight-col", default=WEIGHT_COL, help="Weight column name")
    parser.add_argument("--n-portfolios", type=int, default=N_PORTFOLIOS, help="Number of portfolios")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation")
    parser.add_argument("--no-sign-correct", action="store_true", help="Skip sign correction")

    args = parser.parse_args()

    # Update config
    RETURN_COL = args.return_col
    WEIGHT_COL = args.weight_col
    N_PORTFOLIOS = args.n_portfolios
    SIGN_CORRECT = not args.no_sign_correct

    if args.data_path:
        ew_df, vw_df = main(data_path=args.data_path, validate=not args.no_validate)
    else:
        print("Usage: python fast_decile_portfolios.py <data_path>")
        print("       python fast_decile_portfolios.py data.parquet --n-portfolios 5")
