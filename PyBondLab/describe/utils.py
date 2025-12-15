"""
Utility functions for computing summary statistics.

This module provides shared functions for computing cross-sectional
and time-series statistics used by PreAnalysisStats and PostAnalysisStats.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import statsmodels.api as sm
from typing import Sequence


# =============================================================================
# Cross-Sectional Statistics
# =============================================================================

def compute_cross_sectional_stats(
    values: pd.Series | np.ndarray,
    percentiles: Sequence[float] = (5, 25, 50, 75, 95),
) -> dict:
    """
    Compute cross-sectional statistics for a single period.

    Parameters
    ----------
    values : pd.Series or np.ndarray
        Values to compute statistics for (e.g., all bond durations at time t).
    percentiles : sequence of float
        Percentiles to compute (values between 0 and 100).
        Default is (5, 25, 50, 75, 95).

    Returns
    -------
    dict
        Dictionary containing all computed statistics:
        - mean, std, skew, kurt (excess), min, max
        - p{X} for each percentile X
        - n (count of valid observations)
    """
    # Convert to numpy array and remove NaN/inf values
    arr = np.asarray(values)
    valid_mask = np.isfinite(arr)
    valid_values = arr[valid_mask]
    n_valid = len(valid_values)

    # Initialize result dictionary
    result = {}

    # Handle empty case
    if n_valid == 0:
        result['mean'] = np.nan
        result['std'] = np.nan
        result['skew'] = np.nan
        result['kurt'] = np.nan
        result['min'] = np.nan
        result['max'] = np.nan
        for p in percentiles:
            result[f'p{int(p)}'] = np.nan
        result['n'] = 0
        return result

    # Basic statistics
    result['mean'] = np.mean(valid_values)
    result['std'] = np.std(valid_values, ddof=1) if n_valid > 1 else np.nan

    # Higher moments (need at least 3 observations for skewness, 4 for kurtosis)
    if n_valid >= 3:
        result['skew'] = scipy_stats.skew(valid_values, bias=False)
    else:
        result['skew'] = np.nan

    if n_valid >= 4:
        # Fisher=True gives excess kurtosis (kurtosis - 3)
        result['kurt'] = scipy_stats.kurtosis(valid_values, fisher=True, bias=False)
    else:
        result['kurt'] = np.nan

    # Min and max
    result['min'] = np.min(valid_values)
    result['max'] = np.max(valid_values)

    # Percentiles
    for p in percentiles:
        result[f'p{int(p)}'] = np.percentile(valid_values, p)

    # Count
    result['n'] = n_valid

    return result


def compute_issuer_stats(
    data: pd.DataFrame,
    id_col: str,
    issuer_col: str,
) -> dict:
    """
    Compute issuer-level statistics for a single period.

    Parameters
    ----------
    data : pd.DataFrame
        Data for a single period.
    id_col : str
        Column name for entity identifier (e.g., 'ID', 'CUSIP').
    issuer_col : str
        Column name for issuer identifier (e.g., 'PERMNO').

    Returns
    -------
    dict
        Dictionary with:
        - n_issuers: number of unique issuers
        - bonds_per_issuer: average number of bonds per issuer
    """
    n_bonds = data[id_col].nunique()
    n_issuers = data[issuer_col].nunique()

    return {
        'n_issuers': n_issuers,
        'bonds_per_issuer': n_bonds / n_issuers if n_issuers > 0 else np.nan,
    }


# =============================================================================
# Time-Series Aggregation
# =============================================================================

def compute_time_series_stats(
    series: pd.Series,
    include_nw: bool = False,
    nw_lag: int = 0,
) -> dict:
    """
    Compute time-series statistics for a single cross-sectional statistic.

    Parameters
    ----------
    series : pd.Series
        Time series of a cross-sectional statistic (e.g., mean over time).
    include_nw : bool
        If True, compute Newey-West t-statistics. Default is False.
    nw_lag : int
        Number of lags for Newey-West standard errors. Default is 0.

    Returns
    -------
    dict
        Dictionary with time-series statistics:
        - mean, std, median, min, max
        - (optional) t_stat, p_value if include_nw=True
    """
    valid_series = series.dropna()
    n_obs = len(valid_series)

    result = {}

    if n_obs == 0:
        result['mean'] = np.nan
        result['std'] = np.nan
        result['median'] = np.nan
        result['min'] = np.nan
        result['max'] = np.nan
        if include_nw:
            result['t_stat'] = np.nan
            result['p_value'] = np.nan
        return result

    result['mean'] = valid_series.mean()
    result['std'] = valid_series.std(ddof=1) if n_obs > 1 else np.nan
    result['median'] = valid_series.median()
    result['min'] = valid_series.min()
    result['max'] = valid_series.max()

    # Newey-West t-statistics
    if include_nw and n_obs > 1:
        t_stat, p_value = compute_nw_tstat(valid_series, nw_lag=nw_lag)
        result['t_stat'] = t_stat
        result['p_value'] = p_value

    return result


def compute_nw_tstat(
    series: pd.Series,
    nw_lag: int = 0,
) -> tuple[float, float]:
    """
    Compute Newey-West t-statistic for testing if mean equals zero.

    Uses HAC (Heteroskedasticity and Autocorrelation Consistent) standard
    errors with the Newey-West estimator.

    Parameters
    ----------
    series : pd.Series
        Time series to test.
    nw_lag : int
        Number of lags for Newey-West. Default is 0 (White robust SE).

    Returns
    -------
    tuple
        (t_statistic, p_value)
    """
    valid_series = series.dropna()

    if len(valid_series) < 2:
        return np.nan, np.nan

    try:
        # Regress on constant to test if mean is zero
        y = valid_series.values
        X = np.ones(len(y))
        model = sm.OLS(y, X, missing='drop').fit(
            cov_type='HAC',
            cov_kwds={'maxlags': nw_lag}
        )
        t_stat = model.tvalues[0]
        p_value = model.pvalues[0]
        return t_stat, p_value
    except Exception:
        return np.nan, np.nan


def aggregate_time_series_stats(
    cs_stats_df: pd.DataFrame,
    include_nw: bool = False,
    nw_lag: int = 0,
) -> pd.DataFrame:
    """
    Aggregate cross-sectional statistics over time.

    For each cross-sectional statistic (column), compute time-series
    properties: mean, std, median, min, max, and optionally NW t-stats.

    Parameters
    ----------
    cs_stats_df : pd.DataFrame
        DataFrame with dates as index and CS statistics as columns.
        Example columns: ['mean', 'std', 'skew', 'p5', 'p25', ..., 'n']
    include_nw : bool
        If True, include Newey-West t-statistics.
    nw_lag : int
        Number of lags for Newey-West standard errors.

    Returns
    -------
    pd.DataFrame
        DataFrame with CS statistics as index and TS aggregates as columns.
        Columns: ['mean', 'std', 'median', 'min', 'max']
        If include_nw: also ['t_stat', 'p_value']
    """
    results = {}

    for col in cs_stats_df.columns:
        ts_stats = compute_time_series_stats(
            cs_stats_df[col],
            include_nw=include_nw,
            nw_lag=nw_lag,
        )
        results[col] = ts_stats

    # Convert to DataFrame with statistics as index
    result_df = pd.DataFrame(results).T

    # Reorder columns for readability
    base_cols = ['mean', 'std', 'median', 'min', 'max']
    if include_nw:
        base_cols.extend(['t_stat', 'p_value'])

    # Only include columns that exist
    result_df = result_df[[c for c in base_cols if c in result_df.columns]]

    return result_df


# =============================================================================
# Formatting Utilities
# =============================================================================

def format_number(x: float, precision: int = 3) -> str:
    """
    Format a number for display, handling large/small values appropriately.

    Parameters
    ----------
    x : float
        Number to format.
    precision : int
        Number of decimal places. Default is 3.

    Returns
    -------
    str
        Formatted number string.
    """
    if np.isnan(x):
        return '--'

    if abs(x) >= 1e4 or (abs(x) < 1e-4 and x != 0):
        return f'{x:.{precision}g}'
    else:
        return f'{x:.{precision}f}'


def stats_to_latex_row(
    stats_dict: dict,
    columns: Sequence[str],
    row_name: str,
    precision: int = 3,
) -> str:
    """
    Convert statistics dictionary to a LaTeX table row.

    Parameters
    ----------
    stats_dict : dict
        Dictionary of statistics.
    columns : sequence of str
        Column names to include (in order).
    row_name : str
        Name for this row.
    precision : int
        Decimal precision for formatting.

    Returns
    -------
    str
        LaTeX formatted row string.
    """
    values = [format_number(stats_dict.get(col, np.nan), precision) for col in columns]
    return f"{row_name} & " + " & ".join(values) + " \\\\"


# =============================================================================
# Correlation Utilities
# =============================================================================

def winsorize_series(
    series: pd.Series,
    pct: float,
    mask: np.ndarray | None = None,
) -> pd.Series:
    """
    Winsorize a series at specified percentile.

    Parameters
    ----------
    series : pd.Series
        Series to winsorize.
    pct : float
        Percentile for winsorization (e.g., 1.0 for 1st/99th percentile).
    mask : np.ndarray or None
        Boolean mask indicating which values to use for computing percentiles.
        If None, uses all finite values.

    Returns
    -------
    pd.Series
        Winsorized series.
    """
    arr = series.values.copy()

    # Determine which values to use for percentile computation
    if mask is not None:
        valid_for_pct = arr[mask & np.isfinite(arr)]
    else:
        valid_for_pct = arr[np.isfinite(arr)]

    if len(valid_for_pct) < 2:
        return series

    # Compute percentile bounds
    lower = np.percentile(valid_for_pct, pct)
    upper = np.percentile(valid_for_pct, 100 - pct)

    # Winsorize
    arr = np.clip(arr, lower, upper)

    return pd.Series(arr, index=series.index)


def winsorize_pairwise(
    x: pd.Series,
    y: pd.Series,
    pct: float,
) -> tuple[pd.Series, pd.Series]:
    """
    Winsorize two series using only observations where both are valid.

    This ensures that the winsorization percentiles are computed on the
    same set of observations that will be used for correlation.

    Parameters
    ----------
    x : pd.Series
        First variable.
    y : pd.Series
        Second variable.
    pct : float
        Percentile for winsorization (e.g., 1.0 for 1st/99th percentile).

    Returns
    -------
    tuple[pd.Series, pd.Series]
        Winsorized (x, y) series.
    """
    # Find observations where both X and Y are valid
    valid_mask = np.isfinite(x.values) & np.isfinite(y.values)

    if valid_mask.sum() < 2:
        return x, y

    # Winsorize each series using the common valid mask
    x_wins = winsorize_series(x, pct, mask=valid_mask)
    y_wins = winsorize_series(y, pct, mask=valid_mask)

    return x_wins, y_wins


def compute_pearson_correlation(
    x: pd.Series,
    y: pd.Series,
    winsorize: bool = True,
    winsorize_pct: float = 1.0,
) -> tuple[float, int]:
    """
    Compute Pearson correlation with optional winsorization.

    Parameters
    ----------
    x : pd.Series
        First variable.
    y : pd.Series
        Second variable.
    winsorize : bool
        If True, winsorize both series before computing correlation.
    winsorize_pct : float
        Percentile for winsorization.

    Returns
    -------
    tuple[float, int]
        (correlation, n_observations)
    """
    # Find pairwise complete observations
    valid_mask = np.isfinite(x.values) & np.isfinite(y.values)
    n_valid = valid_mask.sum()

    if n_valid < 3:
        return np.nan, n_valid

    # Winsorize if requested
    if winsorize:
        x_use, y_use = winsorize_pairwise(x, y, winsorize_pct)
    else:
        x_use, y_use = x, y

    # Extract valid values
    x_valid = x_use.values[valid_mask]
    y_valid = y_use.values[valid_mask]

    # Compute Pearson correlation
    corr, _ = scipy_stats.pearsonr(x_valid, y_valid)

    return corr, n_valid


def compute_spearman_correlation(
    x: pd.Series,
    y: pd.Series,
) -> tuple[float, int]:
    """
    Compute Spearman rank correlation.

    Note: Spearman correlation should NOT be computed on winsorized data
    as it is already robust to outliers (rank-based).

    Parameters
    ----------
    x : pd.Series
        First variable.
    y : pd.Series
        Second variable.

    Returns
    -------
    tuple[float, int]
        (correlation, n_observations)
    """
    # Find pairwise complete observations
    valid_mask = np.isfinite(x.values) & np.isfinite(y.values)
    n_valid = valid_mask.sum()

    if n_valid < 3:
        return np.nan, n_valid

    # Extract valid values (no winsorization for Spearman)
    x_valid = x.values[valid_mask]
    y_valid = y.values[valid_mask]

    # Compute Spearman correlation
    corr, _ = scipy_stats.spearmanr(x_valid, y_valid)

    return corr, n_valid


def compute_pairwise_correlations(
    data: pd.DataFrame,
    var_x: str,
    var_y: str,
    winsorize: bool = True,
    winsorize_pct: float = 1.0,
) -> dict:
    """
    Compute both Pearson and Spearman correlations for a variable pair.

    Parameters
    ----------
    data : pd.DataFrame
        Data for a single period.
    var_x : str
        First variable name.
    var_y : str
        Second variable name.
    winsorize : bool
        If True, winsorize for Pearson correlation.
    winsorize_pct : float
        Percentile for winsorization.

    Returns
    -------
    dict
        Dictionary with:
        - pearson: Pearson correlation
        - spearman: Spearman rank correlation
        - n: number of valid observation pairs
    """
    x = data[var_x]
    y = data[var_y]

    # Pearson (with optional winsorization)
    pearson, n = compute_pearson_correlation(x, y, winsorize, winsorize_pct)

    # Spearman (never winsorized)
    spearman, _ = compute_spearman_correlation(x, y)

    return {
        'pearson': pearson,
        'spearman': spearman,
        'n': n,
    }
