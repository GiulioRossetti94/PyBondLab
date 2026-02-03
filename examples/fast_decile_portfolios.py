# -*- coding: utf-8 -*-
"""
Fast Decile Portfolio Formation Script

Forms decile portfolios for multiple signals using optimized numpy/numba operations.

Data structure expected:
- date: formation date t
- cusip: bond identifier
- signal: sorting variable at t
- r_1: forward return from t to t+1 (already shifted, i.e., r_1 = ret.shift(-1))
- mv: value weight at t (used for VW portfolios)

This matches PyBondLab SingleSort with holding_period=1.

Key behavior:
- Breakpoints computed from ALL bonds with valid signal at t
- Returns computed only for bonds with valid signal AND valid r_1

Author: Claude
Date: 2026-02-03
"""

import numpy as np
import pandas as pd
from numba import njit
import time

# =============================================================================
# CONFIGURATION
# =============================================================================
DATE_COL = 'date'
ID_COL = 'cusip'
RETURN_COL = 'r_1'        # Forward return at t (return from t to t+1)
WEIGHT_COL = 'mv'         # Value weight at t
N_PORTFOLIOS = 10         # Default: deciles


# =============================================================================
# NUMBA KERNELS
# =============================================================================

@njit(cache=True)
def compute_breakpoints(values, n_portfolios):
    """
    Compute percentile breakpoints using numpy percentile formula.

    Parameters
    ----------
    values : np.ndarray[float64]
        Valid (non-NaN) signal values
    n_portfolios : int
        Number of portfolios

    Returns
    -------
    np.ndarray[float64]
        Breakpoint thresholds (length n_portfolios - 1)
    """
    n = len(values)
    if n < n_portfolios:
        return np.empty(0, dtype=np.float64)

    # Sort values
    sorted_vals = np.sort(values)

    # Compute breakpoints at percentiles: 100/n_port, 200/n_port, ..., (n_port-1)*100/n_port
    breakpoints = np.empty(n_portfolios - 1, dtype=np.float64)
    for p in range(n_portfolios - 1):
        pct = (p + 1) * 100.0 / n_portfolios
        # Linear interpolation (numpy percentile default)
        idx = pct / 100.0 * (n - 1)
        idx_low = int(idx)
        idx_high = min(idx_low + 1, n - 1)
        frac = idx - idx_low
        breakpoints[p] = sorted_vals[idx_low] * (1 - frac) + sorted_vals[idx_high] * frac

    return breakpoints


@njit(cache=True)
def assign_ranks(signal, breakpoints, n_portfolios):
    """
    Assign portfolio ranks based on breakpoints.

    Rank 1 = lowest signal values, Rank N = highest signal values.
    Rank 0 = invalid (NaN signal).

    Parameters
    ----------
    signal : np.ndarray[float64]
        Signal values (may contain NaN)
    breakpoints : np.ndarray[float64]
        Breakpoint thresholds
    n_portfolios : int
        Number of portfolios

    Returns
    -------
    np.ndarray[int32]
        Ranks (1 to n_portfolios), 0 for invalid
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
        for bp in breakpoints:
            if val > bp:
                rank += 1
        ranks[i] = rank

    return ranks


@njit(cache=True)
def compute_portfolio_returns(ranks, returns, weights, n_portfolios):
    """
    Compute EW and VW portfolio returns.

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
        Equal-weighted returns per portfolio
    vw_ret : np.ndarray[float64]
        Value-weighted returns per portfolio
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

            # EW: all valid bonds contribute
            sum_ret += r
            count += 1

            # VW: only bonds with valid weight contribute
            if w > 0:
                sum_vw_ret += r * w
                sum_w += w

        if count > 0:
            ew_ret[p - 1] = sum_ret / count
        if sum_w > 0:
            vw_ret[p - 1] = sum_vw_ret / sum_w

    return ew_ret, vw_ret


# =============================================================================
# MAIN PORTFOLIO FORMATION
# =============================================================================

def form_portfolios(data, signal_col, return_col=RETURN_COL, weight_col=WEIGHT_COL,
                    n_portfolios=N_PORTFOLIOS):
    """
    Form portfolios for a single signal across all dates.

    Matches PyBondLab SingleSort(holding_period=1) behavior exactly:
    1. Compute breakpoints from ALL bonds with valid signal at formation date t
       (including bonds with NaN return - they contribute to ranking universe)
    2. Assign ranks to ALL bonds based on these breakpoints
    3. Filter to bonds that have valid signal AND valid return (r_1)
    4. Compute portfolio returns using the pre-computed ranks

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with columns: date, cusip, signal, r_1 (forward return), mv (weight).
        IMPORTANT: Include ALL bonds with valid signal, even those with NaN r_1.
        Bonds with NaN r_1 contribute to breakpoint computation but are excluded
        from return calculation.
    signal_col : str
        Signal column name
    return_col : str
        Forward return column name (r_1 = return from t to t+1)
    weight_col : str
        Weight column name (for VW portfolios)
    n_portfolios : int
        Number of portfolios (default: 10 for deciles)

    Returns
    -------
    ew_ls : pd.Series
        Equal-weighted long-short returns indexed by return date (t+1)
    vw_ls : pd.Series
        Value-weighted long-short returns indexed by return date (t+1)
    """
    # Group by date
    date_groups = data.groupby(DATE_COL)
    dates = sorted(date_groups.groups.keys())
    n_dates = len(dates)

    # Results arrays
    ew_ls = np.full(n_dates, np.nan, dtype=np.float64)
    vw_ls = np.full(n_dates, np.nan, dtype=np.float64)

    for i, date in enumerate(dates):
        group = date_groups.get_group(date)

        # Extract arrays
        signal = group[signal_col].values.astype(np.float64)
        returns = group[return_col].values.astype(np.float64)
        weights = group[weight_col].values.astype(np.float64)

        # -----------------------------------------------------------------
        # STEP 1: Compute breakpoints from ALL bonds with valid signal
        # -----------------------------------------------------------------
        valid_signal_mask = np.isfinite(signal)
        n_valid_signal = valid_signal_mask.sum()

        if n_valid_signal < n_portfolios:
            continue

        valid_signals = signal[valid_signal_mask]
        breakpoints = compute_breakpoints(valid_signals, n_portfolios)

        if len(breakpoints) == 0:
            continue

        # -----------------------------------------------------------------
        # STEP 2: Assign ranks to ALL bonds
        # -----------------------------------------------------------------
        ranks = assign_ranks(signal, breakpoints, n_portfolios)

        # -----------------------------------------------------------------
        # STEP 3: Filter to bonds with valid signal AND valid return
        # -----------------------------------------------------------------
        valid_mask = (ranks > 0) & np.isfinite(returns)

        # Apply mask
        filtered_ranks = ranks[valid_mask]
        filtered_returns = returns[valid_mask]
        filtered_weights = weights[valid_mask]

        # Check we have enough bonds
        if len(filtered_ranks) < n_portfolios:
            continue

        # -----------------------------------------------------------------
        # STEP 4: Compute portfolio returns
        # -----------------------------------------------------------------
        ew_ret, vw_ret = compute_portfolio_returns(
            filtered_ranks, filtered_returns, filtered_weights, n_portfolios
        )

        # Long-short: top minus bottom
        ew_long = ew_ret[n_portfolios - 1]
        ew_short = ew_ret[0]
        vw_long = vw_ret[n_portfolios - 1]
        vw_short = vw_ret[0]

        if np.isfinite(ew_long) and np.isfinite(ew_short):
            ew_ls[i] = ew_long - ew_short
        if np.isfinite(vw_long) and np.isfinite(vw_short):
            vw_ls[i] = vw_long - vw_short

    # Create series indexed by RETURN date (t+1)
    # Formation at t, return earned t:t+1, labeled as t+1
    from pandas.tseries.offsets import MonthEnd
    return_dates = pd.DatetimeIndex(dates) + MonthEnd(1)

    ew_series = pd.Series(ew_ls, index=return_dates, name=signal_col).dropna()
    vw_series = pd.Series(vw_ls, index=return_dates, name=signal_col).dropna()

    return ew_series, vw_series


def form_all_portfolios(data, signal_cols, return_col=RETURN_COL, weight_col=WEIGHT_COL,
                        n_portfolios=N_PORTFOLIOS, sign_correct=True, verbose=True):
    """
    Form portfolios for multiple signals.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with columns: date, cusip, signal(s), r_1, mv
    signal_cols : list of str
        Signal column names
    return_col : str
        Forward return column (r_1)
    weight_col : str
        Weight column (mv)
    n_portfolios : int
        Number of portfolios
    sign_correct : bool
        Flip factors with negative mean
    verbose : bool
        Print progress

    Returns
    -------
    ew_df : pd.DataFrame
        EW long-short factors
    vw_df : pd.DataFrame
        VW long-short factors
    """
    if verbose:
        print(f"Forming {n_portfolios}-tile portfolios for {len(signal_cols)} signals...")
        print(f"  Return: {return_col}, Weight: {weight_col}")

    ew_results = {}
    vw_results = {}
    t_start = time.time()

    for i, signal in enumerate(signal_cols):
        if verbose and (i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            print(f"  [{i + 1}/{len(signal_cols)}] {elapsed:.1f}s elapsed")

        try:
            ew_ls, vw_ls = form_portfolios(
                data, signal, return_col, weight_col, n_portfolios
            )

            # Sign correction
            if sign_correct:
                if len(ew_ls) > 0 and ew_ls.mean() < 0:
                    ew_ls = -ew_ls
                if len(vw_ls) > 0 and vw_ls.mean() < 0:
                    vw_ls = -vw_ls

            ew_results[signal] = ew_ls
            vw_results[signal] = vw_ls

        except Exception as e:
            if verbose:
                print(f"  WARNING: {signal} failed - {e}")

    if verbose:
        print(f"  Completed in {time.time() - t_start:.2f}s")

    ew_df = pd.DataFrame(ew_results)
    vw_df = pd.DataFrame(vw_results)

    ew_df = ew_df.reset_index().rename(columns={'index': 'date'})
    vw_df = vw_df.reset_index().rename(columns={'index': 'date'})

    return ew_df, vw_df


# =============================================================================
# VALIDATION AGAINST PYBONDLAB
# =============================================================================

def validate_against_pybondlab(n_portfolios=5, n_dates=24, n_bonds=100, seed=42):
    """
    Validate fast code against PyBondLab using synthetic data.

    Creates matching datasets:
    - PyBondLab format: ret at each date t
    - Fast format: r_1 = ret.shift(-1) at each date t

    Returns True if all tests pass.
    """
    try:
        from PyBondLab import StrategyFormation, SingleSort
        from PyBondLab.config import StrategyFormationConfig, FormationConfig, DataConfig
    except ImportError:
        print("PyBondLab not available")
        return False

    np.random.seed(seed)

    print(f"\n{'='*60}")
    print("VALIDATION: Fast Code vs PyBondLab")
    print(f"{'='*60}")
    print(f"Config: {n_portfolios} portfolios, {n_dates} dates, {n_bonds} bonds")

    # -------------------------------------------------------------------------
    # Generate synthetic data
    # -------------------------------------------------------------------------
    dates = pd.date_range('2020-01-31', periods=n_dates, freq='ME')
    bond_ids = [f'BOND{i:03d}' for i in range(n_bonds)]

    # Create balanced panel
    rows = []
    for d in dates:
        for b in bond_ids:
            rows.append({'date': d, 'cusip': b})

    data = pd.DataFrame(rows)

    # Add signal and returns
    n_obs = len(data)
    data['signal'] = np.random.randn(n_obs) * 10
    data['ret'] = np.random.randn(n_obs) * 0.05  # 5% std
    data['mv'] = np.abs(np.random.randn(n_obs)) * 1000 + 100
    data['RATING_NUM'] = 5

    # Create r_1 = ret.shift(-1) within each bond
    data = data.sort_values(['cusip', 'date'])
    data['r_1'] = data.groupby('cusip')['ret'].shift(-1)
    data = data.sort_values(['date', 'cusip']).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Run PyBondLab (uses ret at t+1)
    # -------------------------------------------------------------------------
    print("\nRunning PyBondLab...")

    pbl_data = data[['date', 'cusip', 'ret', 'mv', 'signal', 'RATING_NUM']].copy()
    pbl_data = pbl_data.rename(columns={'cusip': 'ID', 'mv': 'VW'})

    strategy = SingleSort(holding_period=1, sort_var='signal', num_portfolios=n_portfolios)
    config = StrategyFormationConfig(
        data=DataConfig(),
        formation=FormationConfig(dynamic_weights=True, compute_turnover=False, verbose=False)
    )

    sf = StrategyFormation(data=pbl_data, strategy=strategy, config=config)
    result = sf.fit()
    ew_pbl, vw_pbl = result.get_long_short()

    print(f"  PyBondLab dates: {len(ew_pbl)}")
    print(f"  PyBondLab EW mean: {ew_pbl.mean()*100:.4f}%")
    print(f"  PyBondLab VW mean: {vw_pbl.mean()*100:.4f}%")

    # -------------------------------------------------------------------------
    # Run Fast Code (uses r_1 at t)
    # -------------------------------------------------------------------------
    print("\nRunning Fast Code...")

    # Use data with r_1 - do NOT drop NaN rows here!
    # Bonds with NaN r_1 still contribute to breakpoint computation.
    # form_portfolios will filter internally for return computation.
    fast_data = data[['date', 'cusip', 'r_1', 'mv', 'signal']].copy()

    ew_fast, vw_fast = form_portfolios(
        fast_data, 'signal', return_col='r_1', weight_col='mv',
        n_portfolios=n_portfolios
    )

    print(f"  Fast dates: {len(ew_fast)}")
    print(f"  Fast EW mean: {ew_fast.mean()*100:.4f}%")
    print(f"  Fast VW mean: {vw_fast.mean()*100:.4f}%")

    # -------------------------------------------------------------------------
    # Compare
    # -------------------------------------------------------------------------
    print("\nComparing results...")

    common_dates = ew_pbl.index.intersection(ew_fast.index)
    print(f"  Common dates: {len(common_dates)}")

    if len(common_dates) == 0:
        print("  ERROR: No overlapping dates!")
        print(f"  PyBondLab dates: {ew_pbl.index[:5].tolist()}...")
        print(f"  Fast dates: {ew_fast.index[:5].tolist()}...")
        return False

    ew_pbl_aligned = ew_pbl.loc[common_dates]
    vw_pbl_aligned = vw_pbl.loc[common_dates]
    ew_fast_aligned = ew_fast.loc[common_dates]
    vw_fast_aligned = vw_fast.loc[common_dates]

    ew_diff = (ew_fast_aligned - ew_pbl_aligned).abs()
    vw_diff = (vw_fast_aligned - vw_pbl_aligned).abs()

    ew_max_diff = ew_diff.max()
    vw_max_diff = vw_diff.max()

    tol = 1e-10
    ew_pass = ew_max_diff < tol
    vw_pass = vw_max_diff < tol

    print(f"\n  EW max diff: {ew_max_diff:.2e} {'PASS' if ew_pass else 'FAIL'}")
    print(f"  VW max diff: {vw_max_diff:.2e} {'PASS' if vw_pass else 'FAIL'}")

    if not (ew_pass and vw_pass):
        # Show details of first mismatch
        print("\n  Detailed comparison (first 5 dates):")
        for d in common_dates[:5]:
            print(f"    {d.date()}: EW fast={ew_fast_aligned[d]:.6f}, pbl={ew_pbl_aligned[d]:.6f}, diff={ew_diff[d]:.2e}")
            print(f"              VW fast={vw_fast_aligned[d]:.6f}, pbl={vw_pbl_aligned[d]:.6f}, diff={vw_diff[d]:.2e}")

    all_pass = ew_pass and vw_pass
    print(f"\n{'='*60}")
    print(f"VALIDATION: {'PASSED' if all_pass else 'FAILED'}")
    print(f"{'='*60}")

    return all_pass


def validate_unbalanced_panel(n_portfolios=5, seed=42):
    """
    Validate with unbalanced panel (bonds entering/exiting).
    """
    try:
        from PyBondLab import StrategyFormation, SingleSort
        from PyBondLab.config import StrategyFormationConfig, FormationConfig, DataConfig
    except ImportError:
        print("PyBondLab not available")
        return False

    np.random.seed(seed)

    print(f"\n{'='*60}")
    print("VALIDATION: Unbalanced Panel")
    print(f"{'='*60}")

    # Create unbalanced panel: some bonds only exist for part of the sample
    dates = pd.date_range('2020-01-31', periods=6, freq='ME')

    # Group A: exists all dates
    # Group B: enters at date 2 (Feb)
    # Group C: exits after date 3 (Mar)

    rows = []
    for i, d in enumerate(dates):
        # Group A: always present
        for b in range(10):
            rows.append({'date': d, 'cusip': f'A{b:02d}'})

        # Group B: enters at date index 1 (Feb)
        if i >= 1:
            for b in range(5):
                rows.append({'date': d, 'cusip': f'B{b:02d}'})

        # Group C: exits after date index 2 (Mar)
        if i <= 2:
            for b in range(5):
                rows.append({'date': d, 'cusip': f'C{b:02d}'})

    data = pd.DataFrame(rows)
    n_obs = len(data)

    # Add signal and returns
    data['signal'] = np.random.randn(n_obs) * 10
    data['ret'] = np.random.randn(n_obs) * 0.05
    data['mv'] = np.abs(np.random.randn(n_obs)) * 1000 + 100
    data['RATING_NUM'] = 5

    # Create r_1 = ret.shift(-1) within each bond
    data = data.sort_values(['cusip', 'date'])
    data['r_1'] = data.groupby('cusip')['ret'].shift(-1)
    data = data.sort_values(['date', 'cusip']).reset_index(drop=True)

    print(f"Data shape: {data.shape}")
    print(f"Dates: {dates.tolist()}")

    # Show bond counts per date
    for d in dates:
        subset = data[data['date'] == d]
        n_a = subset['cusip'].str.startswith('A').sum()
        n_b = subset['cusip'].str.startswith('B').sum()
        n_c = subset['cusip'].str.startswith('C').sum()
        print(f"  {d.date()}: A={n_a}, B={n_b}, C={n_c}, total={len(subset)}")

    # -------------------------------------------------------------------------
    # Run PyBondLab
    # -------------------------------------------------------------------------
    print("\nRunning PyBondLab...")

    pbl_data = data[['date', 'cusip', 'ret', 'mv', 'signal', 'RATING_NUM']].copy()
    pbl_data = pbl_data.rename(columns={'cusip': 'ID', 'mv': 'VW'})

    strategy = SingleSort(holding_period=1, sort_var='signal', num_portfolios=n_portfolios)
    config = StrategyFormationConfig(
        data=DataConfig(),
        formation=FormationConfig(dynamic_weights=True, compute_turnover=False, verbose=False)
    )

    sf = StrategyFormation(data=pbl_data, strategy=strategy, config=config)
    result = sf.fit()
    ew_pbl, vw_pbl = result.get_long_short()

    # -------------------------------------------------------------------------
    # Run Fast Code
    # -------------------------------------------------------------------------
    print("Running Fast Code...")

    # Do NOT drop NaN r_1 rows - they contribute to breakpoint computation
    fast_data = data[['date', 'cusip', 'r_1', 'mv', 'signal']].copy()

    ew_fast, vw_fast = form_portfolios(
        fast_data, 'signal', return_col='r_1', weight_col='mv',
        n_portfolios=n_portfolios
    )

    # -------------------------------------------------------------------------
    # Compare date by date
    # -------------------------------------------------------------------------
    print("\nDate-by-date comparison:")

    common_dates = ew_pbl.index.intersection(ew_fast.index)
    all_pass = True

    for d in sorted(common_dates):
        ew_diff = abs(ew_fast[d] - ew_pbl[d])
        vw_diff = abs(vw_fast[d] - vw_pbl[d])

        tol = 1e-10
        passed = (ew_diff < tol) and (vw_diff < tol)

        status = "PASS" if passed else "FAIL"
        print(f"  {d.date()}: {status} (EW diff={ew_diff:.2e}, VW diff={vw_diff:.2e})")

        if not passed:
            all_pass = False
            print(f"    Fast: EW={ew_fast[d]:.6f}, VW={vw_fast[d]:.6f}")
            print(f"    PBL:  EW={ew_pbl[d]:.6f}, VW={vw_pbl[d]:.6f}")

    print(f"\n{'='*60}")
    print(f"VALIDATION: {'PASSED' if all_pass else 'FAILED'}")
    print(f"{'='*60}")

    return all_pass


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fast portfolio formation with validation")
    parser.add_argument("--validate", action="store_true", help="Run validation tests")
    parser.add_argument("--n-portfolios", type=int, default=5, help="Number of portfolios")
    parser.add_argument("data_path", nargs="?", help="Path to data file")

    args = parser.parse_args()

    if args.validate or args.data_path is None:
        # Run validation
        print("Running validation tests...\n")

        # Test 1: Balanced panel
        pass1 = validate_against_pybondlab(n_portfolios=args.n_portfolios)

        # Test 2: Unbalanced panel
        pass2 = validate_unbalanced_panel(n_portfolios=args.n_portfolios)

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"  Balanced panel:   {'PASS' if pass1 else 'FAIL'}")
        print(f"  Unbalanced panel: {'PASS' if pass2 else 'FAIL'}")
        print(f"  Overall:          {'PASS' if (pass1 and pass2) else 'FAIL'}")

    else:
        # Run on provided data
        print(f"Loading data from {args.data_path}...")
        if args.data_path.endswith('.parquet'):
            data = pd.read_parquet(args.data_path)
        else:
            data = pd.read_csv(args.data_path, parse_dates=['date'])

        print(f"Data shape: {data.shape}")

        # Auto-detect signal columns (between 'age' and 'ytm')
        cols = list(data.columns)
        try:
            start = cols.index('age')
            end = cols.index('ytm')
            signal_cols = cols[start:end+1]
        except ValueError:
            # Fallback: all numeric columns except known non-signals
            exclude = {'date', 'cusip', 'permno', 'r_1', 'r_1_exc', 'r_1_dur', 'mv'}
            signal_cols = [c for c in cols if c not in exclude and data[c].dtype in ['float64', 'float32']]

        print(f"Found {len(signal_cols)} signals")

        ew_df, vw_df = form_all_portfolios(
            data, signal_cols,
            n_portfolios=args.n_portfolios,
            verbose=True
        )

        print(f"\nOutput: EW {ew_df.shape}, VW {vw_df.shape}")
