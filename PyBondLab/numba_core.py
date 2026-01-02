# -*- coding: utf-8 -*-
"""
Numba-optimized core functions for PyBondLab portfolio formation.

This module provides high-performance implementations of critical portfolio
formation operations using numba JIT compilation and parallel processing.

Key optimizations:
1. Pre-extract all data to contiguous numpy arrays
2. Replace pandas groupby with vectorized numpy operations
3. Use numba prange for parallel processing where possible

Authors: Optimization layer for PyBondLab
"""

import numpy as np
from numba import njit, prange
from typing import Dict, Tuple, List, Optional
import pandas as pd


# =============================================================================
# Portfolio Return Computation (Numba Kernels)
# =============================================================================

@njit(cache=True)  # NOTE: fastmath=True causes NaN comparison issues
def compute_portfolio_returns_single(
    ranks: np.ndarray,
    returns: np.ndarray,
    weights: np.ndarray,
    nport: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute EW and VW portfolio returns for a single period.

    This replaces pandas groupby operations with direct numpy computation.

    Parameters
    ----------
    ranks : np.ndarray
        Portfolio rank for each bond (1-indexed, NaN for unassigned)
    returns : np.ndarray
        Return for each bond
    weights : np.ndarray
        Value weight for each bond (for VW calculation)
    nport : int
        Total number of portfolios

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (ew_returns, vw_returns) - arrays of length nport
    """
    n = len(ranks)

    # Output arrays
    ew_ret = np.full(nport, np.nan, dtype=np.float64)
    vw_ret = np.full(nport, np.nan, dtype=np.float64)

    # Accumulators for each portfolio
    ew_sum = np.zeros(nport, dtype=np.float64)
    ew_count = np.zeros(nport, dtype=np.int64)
    vw_sum = np.zeros(nport, dtype=np.float64)
    weight_sum = np.zeros(nport, dtype=np.float64)

    # Single pass through data
    for i in range(n):
        r = ranks[i]
        if np.isnan(r):
            continue

        p = int(r) - 1  # Convert to 0-indexed
        if p < 0 or p >= nport:
            continue

        ret_i = returns[i]
        w_i = weights[i]

        if not np.isnan(ret_i):
            ew_sum[p] += ret_i
            ew_count[p] += 1

            if not np.isnan(w_i):
                vw_sum[p] += ret_i * w_i
                weight_sum[p] += w_i

    # Compute final returns
    for p in range(nport):
        if ew_count[p] > 0:
            ew_ret[p] = ew_sum[p] / ew_count[p]

        if weight_sum[p] > 0:
            vw_ret[p] = vw_sum[p] / weight_sum[p]

    return ew_ret, vw_ret


@njit(cache=True)  # fastmath=True breaks conditional checks, causing division by zero
def compute_portfolio_weights_single(
    ranks: np.ndarray,
    value_weights: np.ndarray,
    nport: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute equal weights and value weights for each bond.

    Parameters
    ----------
    ranks : np.ndarray
        Portfolio rank for each bond (1-indexed)
    value_weights : np.ndarray
        Value weight (e.g., market value) for each bond
    nport : int
        Total number of portfolios

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (eweights, vweights, counts) for each bond
    """
    n = len(ranks)

    # First pass: count bonds and sum VW per portfolio
    counts_per_ptf = np.zeros(nport, dtype=np.int64)
    vw_sum_per_ptf = np.zeros(nport, dtype=np.float64)

    for i in range(n):
        r = ranks[i]
        if np.isnan(r):
            continue
        p = int(r) - 1
        if p < 0 or p >= nport:
            continue

        counts_per_ptf[p] += 1
        vw = value_weights[i]
        if not np.isnan(vw):
            vw_sum_per_ptf[p] += vw

    # Second pass: compute weights for each bond
    eweights = np.zeros(n, dtype=np.float64)
    vweights = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.int64)

    for i in range(n):
        r = ranks[i]
        if np.isnan(r):
            continue
        p = int(r) - 1
        if p < 0 or p >= nport:
            continue

        cnt = counts_per_ptf[p]
        counts[i] = cnt

        if cnt > 0:
            eweights[i] = 1.0 / cnt

        vw_sum = vw_sum_per_ptf[p]
        if vw_sum > 0:
            vw = value_weights[i]
            if not np.isnan(vw):
                vweights[i] = vw / vw_sum

    return eweights, vweights, counts


@njit(cache=True)  # Note: fastmath=True breaks conditional checks, causing division by zero
def compute_scaled_weights_single(
    ranks: np.ndarray,
    returns: np.ndarray,
    eweights: np.ndarray,
    vweights: np.ndarray,
    counts: np.ndarray,
    ew_ptf_ret: np.ndarray,
    vw_ptf_ret: np.ndarray,
    nport: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute scaled weights for turnover calculation.

    Scaled weights = (1 + bond_ret) / (1 + ptf_ret) * original_weight

    Parameters
    ----------
    ranks : np.ndarray
        Portfolio rank for each bond
    returns : np.ndarray
        Bond-level returns
    eweights : np.ndarray
        Equal weights
    vweights : np.ndarray
        Value weights
    counts : np.ndarray
        Count of bonds per portfolio
    ew_ptf_ret : np.ndarray
        EW portfolio returns
    vw_ptf_ret : np.ndarray
        VW portfolio returns
    nport : int
        Number of portfolios

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (ew_scaled, vw_scaled) weights
    """
    n = len(ranks)
    ew_scaled = np.zeros(n, dtype=np.float64)
    vw_scaled = np.zeros(n, dtype=np.float64)

    for i in range(n):
        r = ranks[i]
        if np.isnan(r):
            continue
        p = int(r) - 1
        if p < 0 or p >= nport:
            continue

        ret_i = returns[i]
        if np.isnan(ret_i):
            continue

        cnt = counts[i]
        if cnt > 0:
            ew_ptf = ew_ptf_ret[p]
            if not np.isnan(ew_ptf):
                ew_scaled[i] = ((1.0 + ret_i) / (1.0 + ew_ptf)) / cnt

        vw_ptf = vw_ptf_ret[p]
        if not np.isnan(vw_ptf):
            vw_scaled[i] = ((1.0 + ret_i) / (1.0 + vw_ptf)) * vweights[i]

    return ew_scaled, vw_scaled


@njit(cache=True)  # Note: fastmath=True breaks NaN comparisons, do not use here
def compute_characteristics_single(
    ranks: np.ndarray,
    weights: np.ndarray,
    char_values: np.ndarray,
    nport: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute EW and VW portfolio characteristics.

    Parameters
    ----------
    ranks : np.ndarray
        Portfolio rank for each bond
    weights : np.ndarray
        Value weights for VW aggregation
    char_values : np.ndarray
        Characteristic values for each bond
    nport : int
        Number of portfolios

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (ew_char, vw_char) - characteristic values by portfolio
    """
    n = len(ranks)

    # Accumulators
    ew_sum = np.zeros(nport, dtype=np.float64)
    ew_count = np.zeros(nport, dtype=np.int64)
    vw_sum = np.zeros(nport, dtype=np.float64)

    for i in range(n):
        r = ranks[i]
        if np.isnan(r):
            continue
        p = int(r) - 1
        if p < 0 or p >= nport:
            continue

        char_i = char_values[i]
        if np.isnan(char_i):
            continue

        ew_sum[p] += char_i
        ew_count[p] += 1

        w_i = weights[i]
        if not np.isnan(w_i):
            vw_sum[p] += char_i * w_i

    # Compute averages
    ew_char = np.full(nport, np.nan, dtype=np.float64)
    vw_char = np.full(nport, np.nan, dtype=np.float64)

    for p in range(nport):
        if ew_count[p] > 0:
            ew_char[p] = ew_sum[p] / ew_count[p]
        vw_char[p] = vw_sum[p]  # Already weighted

    return ew_char, vw_char


# =============================================================================
# ID Intersection (Numba-optimized)
# =============================================================================

@njit(cache=True)
def intersect_ids_numba(
    ids0: np.ndarray,
    ids1: np.ndarray,
    ids2: np.ndarray,
    dynamic_weights: bool
) -> np.ndarray:
    """
    Find common IDs across arrays using hash-based intersection.

    Parameters
    ----------
    ids0 : np.ndarray
        IDs at formation time (t)
    ids1 : np.ndarray
        IDs at return time (t+h)
    ids2 : np.ndarray
        IDs at weight time (t+h-1)
    dynamic_weights : bool
        Whether to include ids2 in intersection

    Returns
    -------
    np.ndarray
        Array of common IDs
    """
    # Convert to sets for fast intersection
    # Note: This is a simplified version - for string IDs we need different approach
    n0 = len(ids0)
    n1 = len(ids1)

    # Mark IDs present in ids1
    max_id = max(np.max(ids0), np.max(ids1))
    if dynamic_weights:
        max_id = max(max_id, np.max(ids2))

    present_in_1 = np.zeros(max_id + 1, dtype=np.bool_)
    for i in range(n1):
        present_in_1[ids1[i]] = True

    if dynamic_weights:
        present_in_2 = np.zeros(max_id + 1, dtype=np.bool_)
        for i in range(len(ids2)):
            present_in_2[ids2[i]] = True

    # Find common IDs
    common = []
    for i in range(n0):
        id_i = ids0[i]
        if present_in_1[id_i]:
            if not dynamic_weights or present_in_2[id_i]:
                common.append(id_i)

    return np.array(common, dtype=np.int64)


# =============================================================================
# Batch Processing Functions
# =============================================================================

@njit(cache=True, fastmath=True, parallel=True)
def compute_all_portfolio_returns_batch(
    all_ranks: np.ndarray,      # (n_periods, max_bonds)
    all_returns: np.ndarray,    # (n_periods, max_bonds)
    all_weights: np.ndarray,    # (n_periods, max_bonds)
    valid_mask: np.ndarray,     # (n_periods, max_bonds) - True if bond valid
    nport: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute portfolio returns for all periods in parallel.

    This is the main batch computation function that processes all time periods
    in parallel using prange.

    Parameters
    ----------
    all_ranks : np.ndarray
        Portfolio ranks for all periods, shape (n_periods, max_bonds)
    all_returns : np.ndarray
        Bond returns for all periods
    all_weights : np.ndarray
        Value weights for all periods
    valid_mask : np.ndarray
        Boolean mask indicating valid bonds
    nport : int
        Number of portfolios

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (ew_returns, vw_returns) - shape (n_periods, nport)
    """
    n_periods = all_ranks.shape[0]
    max_bonds = all_ranks.shape[1]

    ew_ret_all = np.full((n_periods, nport), np.nan, dtype=np.float64)
    vw_ret_all = np.full((n_periods, nport), np.nan, dtype=np.float64)

    for t in prange(n_periods):
        # Accumulators for this period
        ew_sum = np.zeros(nport, dtype=np.float64)
        ew_count = np.zeros(nport, dtype=np.int64)
        vw_sum = np.zeros(nport, dtype=np.float64)
        weight_sum = np.zeros(nport, dtype=np.float64)

        for i in range(max_bonds):
            if not valid_mask[t, i]:
                continue

            r = all_ranks[t, i]
            if np.isnan(r):
                continue

            p = int(r) - 1
            if p < 0 or p >= nport:
                continue

            ret_i = all_returns[t, i]
            w_i = all_weights[t, i]

            if not np.isnan(ret_i):
                ew_sum[p] += ret_i
                ew_count[p] += 1

                if not np.isnan(w_i):
                    vw_sum[p] += ret_i * w_i
                    weight_sum[p] += w_i

        # Compute final returns for this period
        for p in range(nport):
            if ew_count[p] > 0:
                ew_ret_all[t, p] = ew_sum[p] / ew_count[p]

            if weight_sum[p] > 0:
                vw_ret_all[t, p] = vw_sum[p] / weight_sum[p]

    return ew_ret_all, vw_ret_all


# =============================================================================
# Data Extraction Helpers
# =============================================================================

def extract_period_arrays(
    It1: pd.DataFrame,
    ranks_map: Dict,
    vw_map: Dict,
    date_t: pd.Timestamp,
    ret_col: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract numpy arrays from pandas DataFrames for a single period.

    This is a helper function to convert pandas data structures to
    contiguous numpy arrays for numba processing.

    Parameters
    ----------
    It1 : pd.DataFrame
        Return-period data
    ranks_map : dict
        Precomputed ranks {date: Series(ID -> rank)}
    vw_map : dict
        Precomputed value weights {date: Series(ID -> VW)}
    date_t : pd.Timestamp
        Formation date (for rank lookup)
    ret_col : str
        Return column name

    Returns
    -------
    Tuple of arrays
        (ids, ranks, returns, value_weights, valid_mask)
    """
    if It1.empty:
        return (np.array([], dtype=object),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.bool_))

    ids = It1['ID'].values
    returns = It1[ret_col].values.astype(np.float64)

    # Get ranks from precomputed map
    ranks_series = ranks_map.get(date_t, pd.Series(dtype='float64'))
    ranks = np.array([ranks_series.get(id_val, np.nan) for id_val in ids], dtype=np.float64)

    # Get value weights
    vw_series = vw_map.get(date_t, pd.Series(dtype='float64'))
    value_weights = np.array([vw_series.get(id_val, np.nan) for id_val in ids], dtype=np.float64)

    # Valid mask - bonds with valid ranks
    valid_mask = ~np.isnan(ranks)

    return ids, ranks, returns, value_weights, valid_mask


def form_portfolio_fast(
    It1: pd.DataFrame,
    It1m: pd.DataFrame,
    ranks_map: Dict,
    vw_map: Dict,
    date_t: pd.Timestamp,
    date_t1_minus1: Optional[pd.Timestamp],
    ret_col: str,
    nport: int,
    dynamic_weights: bool,
    compute_turnover: bool = False,
    char_cols: Optional[List[str]] = None
) -> Dict:
    """
    Fast portfolio formation for a single period using numba kernels.

    This is the optimized replacement for _form_single_period that uses
    numba-compiled functions instead of pandas groupby operations.

    Parameters
    ----------
    It1 : pd.DataFrame
        Return-period data (bonds with returns at t+h)
    It1m : pd.DataFrame
        Data at t+h-1 (for dynamic weights and characteristics)
    ranks_map : dict
        Precomputed portfolio ranks
    vw_map : dict
        Precomputed value weights at formation date
    date_t : pd.Timestamp
        Formation date
    date_t1_minus1 : pd.Timestamp
        Date for dynamic weights (t+h-1)
    ret_col : str
        Return column name
    nport : int
        Number of portfolios
    dynamic_weights : bool
        Whether to use dynamic weighting
    compute_turnover : bool
        Whether to compute turnover weights
    char_cols : list of str, optional
        Characteristic columns to aggregate

    Returns
    -------
    dict
        Portfolio formation results
    """
    from .constants import ColumnNames

    # Handle empty data
    if It1.empty:
        return _create_nan_result_fast(nport, char_cols)

    # Get IDs and data arrays
    ids = It1[ColumnNames.ID].values
    returns = It1[ret_col].values.astype(np.float64)

    # Get ranks
    ranks_series = ranks_map.get(date_t, pd.Series(dtype='float64'))
    ranks = np.array([ranks_series.get(id_val, np.nan) for id_val in ids], dtype=np.float64)

    # Filter to valid ranks
    valid_mask = ~np.isnan(ranks)
    if not np.any(valid_mask):
        return _create_nan_result_fast(nport, char_cols)

    # Get value weights
    if dynamic_weights and date_t1_minus1 is not None:
        vw_series = vw_map.get(date_t1_minus1, pd.Series(dtype='float64'))
    else:
        vw_series = vw_map.get(date_t, pd.Series(dtype='float64'))

    value_weights = np.array([vw_series.get(id_val, np.nan) for id_val in ids], dtype=np.float64)

    # Compute weights
    eweights, vweights, counts = compute_portfolio_weights_single(
        ranks, value_weights, nport
    )

    # Compute portfolio returns
    ew_ret, vw_ret = compute_portfolio_returns_single(
        ranks, returns, vweights * value_weights, nport
    )

    # Actually we need to compute VW returns differently - using normalized weights
    # Let me recalculate properly
    vw_ret_corrected = np.full(nport, np.nan, dtype=np.float64)
    for p in range(nport):
        mask = (ranks == (p + 1)) & ~np.isnan(returns)
        if np.any(mask):
            w = vweights[mask]
            r = returns[mask]
            if np.sum(w) > 0:
                vw_ret_corrected[p] = np.sum(r * w)

    # Build result
    result = {
        'returns_ew': ew_ret.tolist(),
        'returns_vw': vw_ret_corrected.tolist(),
        'weights_df': pd.DataFrame(),
        'weights_scaled_df': pd.DataFrame(),
        'chars_ew': None,
        'chars_vw': None
    }

    # Build weights DataFrame if needed for turnover
    if compute_turnover:
        weights_df = pd.DataFrame({
            ColumnNames.ID: ids,
            'ptf_rank': ranks.astype('Int64'),
            'eweights': eweights,
            'vweights': vweights
        })
        weights_df = weights_df[valid_mask].copy()
        result['weights_df'] = weights_df

        # Scaled weights for turnover
        ew_scaled, vw_scaled = compute_scaled_weights_single(
            ranks, returns, eweights, vweights, counts,
            ew_ret, vw_ret_corrected, nport
        )

        weights_scaled_df = pd.DataFrame({
            ColumnNames.ID: ids,
            'ptf_rank': ranks.astype('Int64'),
            'eweights': ew_scaled,
            'vweights': vw_scaled
        })
        weights_scaled_df = weights_scaled_df[valid_mask].copy()
        result['weights_scaled_df'] = weights_scaled_df

    # Compute characteristics if requested
    if char_cols:
        # Merge characteristics from It1m
        chars_ew = pd.DataFrame(index=range(1, nport + 1))
        chars_vw = pd.DataFrame(index=range(1, nport + 1))

        for char in char_cols:
            if char in It1m.columns:
                # Create ID to char mapping
                char_map = It1m.set_index(ColumnNames.ID)[char].to_dict()
                char_values = np.array([char_map.get(id_val, np.nan) for id_val in ids], dtype=np.float64)

                ew_char, vw_char = compute_characteristics_single(
                    ranks, vweights, char_values, nport
                )

                chars_ew[char] = ew_char
                chars_vw[char] = vw_char

        result['chars_ew'] = chars_ew
        result['chars_vw'] = chars_vw

    return result


def _create_nan_result_fast(nport: int, char_cols: Optional[List[str]] = None) -> Dict:
    """Create a NaN result for periods with no data."""
    nan_list = [np.nan] * nport
    result = {
        'returns_ew': nan_list,
        'returns_vw': nan_list,
        'weights_df': pd.DataFrame(),
        'weights_scaled_df': pd.DataFrame(),
        'chars_ew': None,
        'chars_vw': None
    }

    if char_cols:
        nan_df = pd.DataFrame(
            np.full((nport, len(char_cols)), np.nan),
            columns=char_cols,
            index=range(1, nport + 1)
        )
        result['chars_ew'] = nan_df
        result['chars_vw'] = nan_df

    return result


# =============================================================================
# Turnover Computation (Numba-optimized)
# =============================================================================

@njit(cache=True, fastmath=True)
def compute_turnover_all_portfolios(
    ranks: np.ndarray,
    positions: np.ndarray,
    raw_ew: np.ndarray,
    raw_vw: np.ndarray,
    prev_scaled_ew: np.ndarray,
    prev_scaled_vw: np.ndarray,
    prev_sum_ew: np.ndarray,
    prev_sum_vw: np.ndarray,
    prev_seen_ew: np.ndarray,
    prev_seen_vw: np.ndarray,
    cohort: int,
    nport: int,
    n_assets: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute turnover for all portfolios at once.

    This batches the portfolio loop into a single numba call.

    Parameters
    ----------
    ranks : np.ndarray
        Portfolio ranks (1-indexed)
    positions : np.ndarray
        Position indices in the state arrays
    raw_ew : np.ndarray
        Current raw EW weights
    raw_vw : np.ndarray
        Current raw VW weights
    prev_scaled_ew : np.ndarray
        Previous scaled EW weights (nport, n_assets)
    prev_scaled_vw : np.ndarray
        Previous scaled VW weights (nport, n_assets)
    prev_sum_ew : np.ndarray
        Previous EW sum per portfolio (nport,)
    prev_sum_vw : np.ndarray
        Previous VW sum per portfolio (nport,)
    prev_seen_ew : np.ndarray
        Whether we've seen each portfolio before (nport,)
    prev_seen_vw : np.ndarray
        Whether we've seen each portfolio before (nport,)
    cohort : int
        Current cohort index
    nport : int
        Number of portfolios
    n_assets : int
        Total number of assets in state

    Returns
    -------
    Tuple of arrays
        (turn_ew, turn_vw, new_prev_seen_ew, new_prev_seen_vw,
         curr_sum_ew, curr_sum_vw)
    """
    n = len(ranks)

    # Output arrays
    turn_ew = np.full(nport, np.nan, dtype=np.float64)
    turn_vw = np.full(nport, np.nan, dtype=np.float64)
    new_prev_seen_ew = prev_seen_ew.copy()
    new_prev_seen_vw = prev_seen_vw.copy()

    # Accumulate weights per portfolio
    curr_sum_ew = np.zeros(nport, dtype=np.float64)
    curr_sum_vw = np.zeros(nport, dtype=np.float64)

    # Sum of mins per portfolio
    sum_min_ew = np.zeros(nport, dtype=np.float64)
    sum_min_vw = np.zeros(nport, dtype=np.float64)

    # First pass: compute sums and min-sums
    for i in range(n):
        r = ranks[i]
        if np.isnan(r):
            continue

        p = int(r) - 1  # 0-indexed
        if p < 0 or p >= nport:
            continue

        pos = positions[i]
        ew_i = raw_ew[i]
        vw_i = raw_vw[i]

        # Accumulate current sums
        if not np.isnan(ew_i):
            curr_sum_ew[p] += ew_i

            # Sum of min with previous
            if prev_seen_ew[p]:
                prev_w = prev_scaled_ew[p, pos]
                sum_min_ew[p] += min(prev_w, ew_i)

        if not np.isnan(vw_i):
            curr_sum_vw[p] += vw_i

            if prev_seen_vw[p]:
                prev_w = prev_scaled_vw[p, pos]
                sum_min_vw[p] += min(prev_w, vw_i)

    # Second pass: compute turnover
    for p in range(nport):
        if prev_seen_ew[p]:
            turn_ew[p] = prev_sum_ew[p] + curr_sum_ew[p] - 2.0 * sum_min_ew[p]
        else:
            # Entry turnover: going from 0 to full position = sum of weights = 1.0
            turn_ew[p] = curr_sum_ew[p]
            new_prev_seen_ew[p] = True

        if prev_seen_vw[p]:
            turn_vw[p] = prev_sum_vw[p] + curr_sum_vw[p] - 2.0 * sum_min_vw[p]
        else:
            # Entry turnover: going from 0 to full position = sum of weights = 1.0
            turn_vw[p] = curr_sum_vw[p]
            new_prev_seen_vw[p] = True

    return turn_ew, turn_vw, new_prev_seen_ew, new_prev_seen_vw, curr_sum_ew, curr_sum_vw


@njit(cache=True, fastmath=True)
def update_prev_scaled_weights(
    scaled_ew: np.ndarray,
    scaled_vw: np.ndarray,
    ranks: np.ndarray,
    positions: np.ndarray,
    prev_scaled_ew: np.ndarray,
    prev_scaled_vw: np.ndarray,
    nport: int
):
    """
    Update previous scaled weights arrays for next period.

    Only updates portfolios that appear in the current scaled weights.
    Portfolios not present in current data keep their previous values.

    Parameters
    ----------
    scaled_ew : np.ndarray
        Current scaled EW weights
    scaled_vw : np.ndarray
        Current scaled VW weights
    ranks : np.ndarray
        Portfolio ranks
    positions : np.ndarray
        Position indices
    prev_scaled_ew : np.ndarray
        Output: previous EW weights to update (nport, n_assets)
    prev_scaled_vw : np.ndarray
        Output: previous VW weights to update (nport, n_assets)
    nport : int
        Number of portfolios
    """
    n = len(ranks)
    n_assets = prev_scaled_ew.shape[1]

    # First, find which portfolios appear in current data
    portfolio_present = np.zeros(nport, dtype=np.bool_)
    for i in range(n):
        r = ranks[i]
        if np.isnan(r):
            continue
        p = int(r) - 1
        if p >= 0 and p < nport:
            portfolio_present[p] = True

    # Only zero out portfolios that are present in current data
    for p in range(nport):
        if portfolio_present[p]:
            for j in range(n_assets):
                prev_scaled_ew[p, j] = 0.0
                prev_scaled_vw[p, j] = 0.0

    # Fill with current scaled weights
    for i in range(n):
        r = ranks[i]
        if np.isnan(r):
            continue

        p = int(r) - 1
        if p < 0 or p >= nport:
            continue

        pos = positions[i]
        prev_scaled_ew[p, pos] = scaled_ew[i]
        prev_scaled_vw[p, pos] = scaled_vw[i]


# =============================================================================
# Multi-Signal Vectorized Functions (for Batch Processing)
# =============================================================================

@njit(cache=True, fastmath=True)
def _compute_percentile_thresholds(
    values: np.ndarray,
    nport: int
) -> np.ndarray:
    """
    Compute percentile thresholds for a single sorted array.

    Parameters
    ----------
    values : np.ndarray
        SORTED array of non-NaN values
    nport : int
        Number of portfolios

    Returns
    -------
    np.ndarray
        Threshold edges (length nport+1)
    """
    n = len(values)
    thres = np.empty(nport + 1, dtype=np.float64)
    thres[0] = -np.inf

    if n == 0:
        for i in range(1, nport + 1):
            thres[i] = np.nan
        return thres

    for p in range(1, nport + 1):
        pct = p * 100.0 / nport
        if pct >= 100:
            thres[p] = values[n - 1]
        else:
            idx_float = (n - 1) * pct / 100.0
            idx_low = int(np.floor(idx_float))
            idx_high = min(idx_low + 1, n - 1)
            weight = idx_float - idx_low
            thres[p] = values[idx_low] * (1 - weight) + values[idx_high] * weight

    return thres


@njit(cache=True, fastmath=True)
def _assign_ranks_from_thresholds(
    values: np.ndarray,
    thres: np.ndarray,
    nport: int
) -> np.ndarray:
    """
    Assign portfolio ranks based on thresholds.

    Parameters
    ----------
    values : np.ndarray
        Signal values (may contain NaN)
    thres : np.ndarray
        Threshold edges (length nport+1)
    nport : int
        Number of portfolios

    Returns
    -------
    np.ndarray
        Ranks (1-based, NaN for unassigned)
    """
    n = len(values)
    ranks = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        val = values[i]
        if np.isnan(val):
            continue

        for p in range(nport):
            if val > thres[p] and val <= thres[p + 1]:
                ranks[i] = p + 1
                break

    return ranks


@njit(cache=True, parallel=True)
def compute_ranks_multi_signal(
    signal_matrix: np.ndarray,
    nport: int
) -> np.ndarray:
    """
    Compute portfolio ranks for multiple signals simultaneously.

    This is the key function for batch processing speedup - it computes
    ranks for ALL signals in parallel using numba prange.

    Parameters
    ----------
    signal_matrix : np.ndarray
        Signal values, shape (n_bonds, n_signals)
    nport : int
        Number of portfolios

    Returns
    -------
    np.ndarray
        Ranks for all signals, shape (n_bonds, n_signals)
    """
    n_bonds, n_signals = signal_matrix.shape
    all_ranks = np.full((n_bonds, n_signals), np.nan, dtype=np.float64)

    # Process each signal in parallel
    for sig_idx in prange(n_signals):
        # Extract signal values
        values = signal_matrix[:, sig_idx].copy()

        # Count and extract non-NaN values
        n_valid = 0
        for i in range(n_bonds):
            if not np.isnan(values[i]):
                n_valid += 1

        if n_valid == 0:
            continue

        # Create sorted array of non-NaN values
        sorted_vals = np.empty(n_valid, dtype=np.float64)
        j = 0
        for i in range(n_bonds):
            if not np.isnan(values[i]):
                sorted_vals[j] = values[i]
                j += 1

        # Sort
        sorted_vals.sort()

        # Compute thresholds
        thres = _compute_percentile_thresholds(sorted_vals, nport)

        # Assign ranks
        for i in range(n_bonds):
            val = values[i]
            if np.isnan(val):
                continue

            for p in range(nport):
                if val > thres[p] and val <= thres[p + 1]:
                    all_ranks[i, sig_idx] = p + 1
                    break

    return all_ranks


@njit(cache=True, parallel=True)
def compute_portfolio_returns_multi_signal(
    ranks_matrix: np.ndarray,
    returns: np.ndarray,
    weights: np.ndarray,
    nport: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute portfolio returns for multiple signals simultaneously.

    Parameters
    ----------
    ranks_matrix : np.ndarray
        Ranks for all signals, shape (n_bonds, n_signals)
    returns : np.ndarray
        Bond returns, shape (n_bonds,)
    weights : np.ndarray
        Value weights, shape (n_bonds,)
    nport : int
        Number of portfolios

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (ew_returns, vw_returns) - each shape (n_signals, nport)
    """
    n_bonds, n_signals = ranks_matrix.shape

    ew_returns = np.full((n_signals, nport), np.nan, dtype=np.float64)
    vw_returns = np.full((n_signals, nport), np.nan, dtype=np.float64)

    # Process each signal in parallel
    for sig_idx in prange(n_signals):
        ranks = ranks_matrix[:, sig_idx]

        # Accumulators
        sum_ret = np.zeros(nport, dtype=np.float64)
        sum_wret = np.zeros(nport, dtype=np.float64)
        sum_weight = np.zeros(nport, dtype=np.float64)
        count = np.zeros(nport, dtype=np.int64)

        # Accumulate
        for i in range(n_bonds):
            r = ranks[i]
            if np.isnan(r) or np.isnan(returns[i]):
                continue

            p = int(r) - 1
            if p < 0 or p >= nport:
                continue

            ret = returns[i]
            w = weights[i]

            sum_ret[p] += ret
            sum_wret[p] += ret * w
            sum_weight[p] += w
            count[p] += 1

        # Compute final values
        for p in range(nport):
            if count[p] > 0:
                ew_returns[sig_idx, p] = sum_ret[p] / count[p]
                if sum_weight[p] > 0:
                    vw_returns[sig_idx, p] = sum_wret[p] / sum_weight[p]

    return ew_returns, vw_returns


# =============================================================================
# FAST RETURNS-ONLY PATH
# When turnover=False, chars=None, banding=None, we can be MUCH faster
# =============================================================================

@njit(cache=True)
def build_rank_lookup_fast(
    form_date_idx: np.ndarray,   # (n_formation,) date index for each formation row
    form_id_idx: np.ndarray,     # (n_formation,) bond ID index for each formation row
    form_ranks: np.ndarray,      # (n_formation,) portfolio ranks
    n_dates: int,
    n_ids: int
) -> np.ndarray:
    """
    Build a lookup table mapping (date, bond_id) -> rank.

    Replaces the slow Python loop with a fast numba implementation.
    """
    rank_lookup = np.full(n_dates * n_ids, np.nan, dtype=np.float64)
    n = len(form_date_idx)
    for i in range(n):
        d = form_date_idx[i]
        bond_id = form_id_idx[i]
        rank_lookup[d * n_ids + bond_id] = form_ranks[i]
    return rank_lookup


@njit(cache=True, parallel=True)
def align_ranks_for_returns_fast(
    ret_date_idx: np.ndarray,    # (n_returns,) date index for each return row
    ret_id_idx: np.ndarray,      # (n_returns,) bond ID index for each return row
    rank_lookup: np.ndarray,     # (n_dates * n_ids,) flattened lookup table
    n_ids: int
) -> np.ndarray:
    """
    Align formation ranks with return data.

    For each return row at date d, look up the rank from formation date d-1.
    Replaces the slow Python loop with a fast numba implementation.
    """
    n = len(ret_date_idx)
    aligned_ranks = np.full(n, np.nan, dtype=np.float64)
    lookup_size = len(rank_lookup)

    for i in prange(n):
        d = ret_date_idx[i]
        if d == 0:
            continue  # No formation date before first date
        formation_d = d - 1
        bond_id = ret_id_idx[i]
        lookup_idx = formation_d * n_ids + bond_id
        if lookup_idx >= 0 and lookup_idx < lookup_size:
            aligned_ranks[i] = rank_lookup[lookup_idx]

    return aligned_ranks


@njit(cache=True, parallel=True)
def align_ranks_staggered_fast(
    ret_date_idx: np.ndarray,    # (n_returns,) date index for each return row
    ret_id_idx: np.ndarray,      # (n_returns,) bond ID index for each return row
    rank_lookup: np.ndarray,     # (n_dates * n_ids,) flattened lookup table
    n_ids: int,
    n_dates: int,
    hor: int                     # holding period (number of cohorts)
) -> np.ndarray:
    """
    Align formation ranks for staggered portfolios (h > 1).

    For each return row and each cohort, find the formation rank from the
    corresponding formation date.
    """
    n = len(ret_date_idx)
    formation_ranks_matrix = np.full((n, hor), np.nan, dtype=np.float64)
    lookup_size = len(rank_lookup)

    for i in prange(n):
        d = ret_date_idx[i]
        bond_id = ret_id_idx[i]

        for cohort in range(hor):
            if d < cohort + 1:
                continue

            # Formation date for this cohort
            offset = (d - 1 - cohort) % hor
            formation_date = d - 1 - offset

            if formation_date < 0 or formation_date >= n_dates:
                continue

            lookup_idx = formation_date * n_ids + bond_id
            if lookup_idx >= 0 and lookup_idx < lookup_size:
                formation_ranks_matrix[i, cohort] = rank_lookup[lookup_idx]

    return formation_ranks_matrix


@njit(cache=True, parallel=True)
def compute_all_dates_returns_fast(
    date_indices: np.ndarray,      # (n_rows,) - date index for each row
    ranks: np.ndarray,             # (n_rows,) - portfolio rank (1 to nport)
    returns: np.ndarray,           # (n_rows,) - bond returns
    weights: np.ndarray,           # (n_rows,) - value weights
    n_dates: int,
    nport: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute portfolio returns for ALL dates in parallel.

    This is the core fast-path function that replaces the slow per-date loop.
    Uses prange to parallelize across dates.

    Parameters
    ----------
    date_indices : np.ndarray
        Date index (0-based) for each row
    ranks : np.ndarray
        Portfolio ranks (1 to nport) for each row, NaN for unassigned
    returns : np.ndarray
        Bond returns for each row
    weights : np.ndarray
        Value weights for each row
    n_dates : int
        Number of unique dates
    nport : int
        Number of portfolios

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (ew_returns, vw_returns) - each shape (n_dates, nport)
    """
    n_rows = len(date_indices)
    ew_returns = np.full((n_dates, nport), np.nan, dtype=np.float64)
    vw_returns = np.full((n_dates, nport), np.nan, dtype=np.float64)

    # Process each date in parallel
    for d in prange(n_dates):
        # Accumulators for this date
        sum_ret = np.zeros(nport, dtype=np.float64)
        sum_wret = np.zeros(nport, dtype=np.float64)
        sum_weight = np.zeros(nport, dtype=np.float64)
        count = np.zeros(nport, dtype=np.int64)

        # Accumulate across all rows for this date
        for i in range(n_rows):
            if date_indices[i] != d:
                continue

            r = ranks[i]
            ret = returns[i]

            if np.isnan(r) or np.isnan(ret):
                continue

            p = int(r) - 1
            if p < 0 or p >= nport:
                continue

            w = weights[i]
            sum_ret[p] += ret
            sum_wret[p] += ret * w
            sum_weight[p] += w
            count[p] += 1

        # Compute final values for this date
        for p in range(nport):
            if count[p] > 0:
                ew_returns[d, p] = sum_ret[p] / count[p]
                if sum_weight[p] > 0:
                    vw_returns[d, p] = sum_wret[p] / sum_weight[p]

    return ew_returns, vw_returns


@njit(cache=True, parallel=True)
def compute_staggered_returns_fast(
    date_indices: np.ndarray,       # (n_rows,) - date index for each row
    formation_ranks: np.ndarray,    # (n_rows, hor) - ranks from formation date for each cohort
    returns: np.ndarray,            # (n_rows,) - bond returns
    weights: np.ndarray,            # (n_rows,) - value weights
    n_dates: int,
    nport: int,
    hor: int                        # holding period (number of cohorts)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute staggered portfolio returns for ALL dates in parallel.

    For h>1, at each date we have multiple cohorts with different formation dates.
    This function computes the equal-averaged returns across cohorts.

    Parameters
    ----------
    date_indices : np.ndarray
        Date index (0-based) for each row
    formation_ranks : np.ndarray
        Portfolio ranks from formation dates, shape (n_rows, hor)
        Column c contains ranks from the formation date of cohort c
    returns : np.ndarray
        Bond returns for each row
    weights : np.ndarray
        Value weights (from dynamic weights at return date)
    n_dates : int
        Total number of dates
    nport : int
        Number of portfolios
    hor : int
        Holding period (number of cohorts)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (ew_returns, vw_returns) - each shape (n_dates, nport)
        These are equal-averaged across active cohorts
    """
    n_rows = len(date_indices)

    # Accumulate returns for each (date, cohort, portfolio)
    # Then average across cohorts
    ew_returns = np.full((n_dates, nport), np.nan, dtype=np.float64)
    vw_returns = np.full((n_dates, nport), np.nan, dtype=np.float64)

    # Process each date in parallel
    for d in prange(n_dates):
        # For each portfolio, accumulate across cohorts
        cohort_ew = np.zeros((hor, nport), dtype=np.float64)
        cohort_vw = np.zeros((hor, nport), dtype=np.float64)
        cohort_valid = np.zeros((hor, nport), dtype=np.int64)

        for cohort in range(hor):
            # Check if this cohort is active at date d
            # Cohort c is active at date d if d >= c (cohort has been formed)
            if d < cohort:
                continue

            # Accumulators for this (date, cohort)
            sum_ret = np.zeros(nport, dtype=np.float64)
            sum_wret = np.zeros(nport, dtype=np.float64)
            sum_weight = np.zeros(nport, dtype=np.float64)
            count = np.zeros(nport, dtype=np.int64)

            # Accumulate for rows at this date
            for i in range(n_rows):
                if date_indices[i] != d:
                    continue

                r = formation_ranks[i, cohort]
                ret = returns[i]

                if np.isnan(r) or np.isnan(ret):
                    continue

                p = int(r) - 1
                if p < 0 or p >= nport:
                    continue

                w = weights[i]
                sum_ret[p] += ret
                sum_wret[p] += ret * w
                sum_weight[p] += w
                count[p] += 1

            # Store cohort returns
            for p in range(nport):
                if count[p] > 0:
                    cohort_ew[cohort, p] = sum_ret[p] / count[p]
                    cohort_valid[cohort, p] = 1
                    if sum_weight[p] > 0:
                        cohort_vw[cohort, p] = sum_wret[p] / sum_weight[p]

        # Average across valid cohorts
        for p in range(nport):
            n_valid_ew = 0
            n_valid_vw = 0
            sum_ew = 0.0
            sum_vw = 0.0

            for cohort in range(hor):
                if cohort_valid[cohort, p] > 0:
                    sum_ew += cohort_ew[cohort, p]
                    n_valid_ew += 1
                    if not np.isnan(cohort_vw[cohort, p]):
                        sum_vw += cohort_vw[cohort, p]
                        n_valid_vw += 1

            if n_valid_ew > 0:
                ew_returns[d, p] = sum_ew / n_valid_ew
            if n_valid_vw > 0:
                vw_returns[d, p] = sum_vw / n_valid_vw

    return ew_returns, vw_returns


@njit(cache=True)
def precompute_formation_ranks(
    date_indices: np.ndarray,      # (n_rows,) - date index for each row
    id_indices: np.ndarray,        # (n_rows,) - bond ID index for each row
    ranks_by_date: np.ndarray,     # (n_rows,) - ranks computed at each date
    n_dates: int,
    n_ids: int,
    hor: int                       # holding period
) -> np.ndarray:
    """
    Precompute formation ranks for staggered rebalancing.

    For each row at date d and cohort c, find the rank from the
    corresponding formation date.

    Formation date for cohort c at return date d:
    - formation_date = d - 1 - ((d - 1 - c) % hor)
    - This gives the most recent formation date for cohort c before d

    Parameters
    ----------
    date_indices : np.ndarray
        Date index for each row
    id_indices : np.ndarray
        Bond ID index for each row
    ranks_by_date : np.ndarray
        Ranks at each row's date
    n_dates : int
        Total number of dates
    n_ids : int
        Total number of unique bond IDs
    hor : int
        Holding period

    Returns
    -------
    np.ndarray
        Formation ranks, shape (n_rows, hor)
    """
    n_rows = len(date_indices)

    # Build lookup: (date, id) -> rank
    # Using a flat array indexed by date * n_ids + id
    rank_lookup = np.full(n_dates * n_ids, np.nan, dtype=np.float64)

    for i in range(n_rows):
        d = date_indices[i]
        bond_id = id_indices[i]
        rank_lookup[d * n_ids + bond_id] = ranks_by_date[i]

    # For each row, find formation ranks for each cohort
    formation_ranks = np.full((n_rows, hor), np.nan, dtype=np.float64)

    for i in range(n_rows):
        d = date_indices[i]
        bond_id = id_indices[i]

        for cohort in range(hor):
            # Skip if cohort hasn't started yet
            if d < cohort:
                continue

            # Find formation date for this cohort
            # Formation happens at t, returns at t+1 to t+hor
            # For return date d, cohort c was formed at:
            # formation_date = d - 1 - ((d - 1 - c) % hor) for d > c
            if d == 0:
                continue

            # Formation date calculation
            offset = (d - 1 - cohort) % hor
            formation_date = d - 1 - offset

            if formation_date < 0 or formation_date >= n_dates:
                continue

            # Look up rank at formation date
            lookup_idx = formation_date * n_ids + bond_id
            if lookup_idx >= 0 and lookup_idx < len(rank_lookup):
                formation_ranks[i, cohort] = rank_lookup[lookup_idx]

    return formation_ranks


# =============================================================================
# ULTRA-FAST PATH: Skip pandas entirely for large panels
# =============================================================================

@njit(cache=True)
def _argsort_within_date(date_idx: np.ndarray, signal: np.ndarray, n_dates: int):
    """
    Count observations per date and prepare for ranking.
    Returns (counts_per_date, date_starts, sorted_indices_within_date).
    """
    n = len(date_idx)

    # Count per date
    counts = np.zeros(n_dates, dtype=np.int64)
    for i in range(n):
        d = date_idx[i]
        if d >= 0 and d < n_dates:
            counts[d] += 1

    # Compute start positions
    starts = np.zeros(n_dates + 1, dtype=np.int64)
    for d in range(n_dates):
        starts[d + 1] = starts[d] + counts[d]

    return counts, starts


@njit(cache=True, parallel=True)
def build_vw_lookup_and_dynamic_weights(
    date_idx: np.ndarray,
    id_idx: np.ndarray,
    vw: np.ndarray,
    n_dates: int,
    n_ids: int
) -> np.ndarray:
    """
    Build dynamic weights array where each observation's weight
    comes from the previous period (for VW portfolio returns).

    Parameters
    ----------
    date_idx : np.ndarray
        Date index (0-indexed) for each observation
    id_idx : np.ndarray
        Bond ID index for each observation
    vw : np.ndarray
        Value weights for each observation
    n_dates : int
        Total number of dates
    n_ids : int
        Total number of unique IDs

    Returns
    -------
    np.ndarray
        Dynamic weights (VW from previous period) for each observation
    """
    n = len(date_idx)

    # Build VW lookup: (date_idx, id_idx) -> VW
    vw_lookup = np.full(n_dates * n_ids, np.nan, dtype=np.float64)
    for i in range(n):
        d = date_idx[i]
        bid = id_idx[i]
        if d >= 0 and d < n_dates and bid >= 0 and bid < n_ids:
            vw_lookup[d * n_ids + bid] = vw[i]

    # Create dynamic weights array (weight from previous period)
    dynamic_weights = np.full(n, np.nan, dtype=np.float64)
    for i in prange(n):
        d = date_idx[i]
        bid = id_idx[i]
        if d > 0:  # Can look up previous period
            prev_lookup = (d - 1) * n_ids + bid
            if prev_lookup >= 0 and prev_lookup < n_dates * n_ids:
                dynamic_weights[i] = vw_lookup[prev_lookup]

    return dynamic_weights


@njit(cache=True)
def build_vw_lookup(
    date_idx: np.ndarray,
    id_idx: np.ndarray,
    vw: np.ndarray,
    n_dates: int,
    n_ids: int
) -> np.ndarray:
    """
    Build VW lookup table: (date, bond_id) -> VW value.

    This lookup table can be used to get VW from any date, enabling
    both dynamic_weights=True (VW from d-1) and False (VW from formation date).

    Parameters
    ----------
    date_idx : np.ndarray
        Date index (0-indexed) for each observation
    id_idx : np.ndarray
        Bond ID index for each observation
    vw : np.ndarray
        Value weights for each observation
    n_dates : int
        Total number of dates
    n_ids : int
        Total number of unique bond IDs

    Returns
    -------
    np.ndarray
        VW lookup table of shape (n_dates * n_ids,)
        Access as: vw_lookup[date * n_ids + bond_id]
    """
    n = len(date_idx)

    # Build VW lookup: vw_lookup[date * n_ids + bond_id] = VW
    vw_lookup = np.full(n_dates * n_ids, np.nan, dtype=np.float64)
    for i in range(n):
        d = date_idx[i]
        bid = id_idx[i]
        if d >= 0 and d < n_dates and bid >= 0 and bid < n_ids:
            vw_lookup[d * n_ids + bid] = vw[i]

    return vw_lookup


@njit(cache=True, parallel=True)
def compute_ranks_all_dates_fast(
    date_idx: np.ndarray,      # (n,) date index for each row
    signal: np.ndarray,        # (n,) signal values to rank
    n_dates: int,
    nport: int
) -> np.ndarray:
    """
    Compute portfolio ranks for ALL rows across ALL dates in parallel.

    This is the ultra-fast version that bypasses pandas completely.
    Uses percentile-based ranking within each date.

    Parameters
    ----------
    date_idx : np.ndarray
        Date index (0-indexed) for each observation
    signal : np.ndarray
        Signal values to rank (NaN = excluded from ranking)
    n_dates : int
        Total number of dates
    nport : int
        Number of portfolios

    Returns
    -------
    np.ndarray
        Portfolio rank (1-indexed) for each observation, NaN for missing signal
    """
    n = len(date_idx)
    ranks = np.full(n, np.nan, dtype=np.float64)

    # Count valid observations per date (excluding NaN signals)
    counts = np.zeros(n_dates, dtype=np.int64)
    for i in range(n):
        d = date_idx[i]
        if d >= 0 and d < n_dates and not np.isnan(signal[i]):
            counts[d] += 1

    # For each date, collect indices of valid observations
    # Then rank them by signal value
    # Use parallel processing across dates

    # First pass: collect indices per date (serial - needed for setup)
    date_starts = np.zeros(n_dates + 1, dtype=np.int64)
    for d in range(n_dates):
        date_starts[d + 1] = date_starts[d] + counts[d]

    total_valid = date_starts[n_dates]
    valid_indices = np.zeros(total_valid, dtype=np.int64)
    valid_signals = np.zeros(total_valid, dtype=np.float64)

    # Current position for each date
    pos = np.zeros(n_dates, dtype=np.int64)
    for d in range(n_dates):
        pos[d] = date_starts[d]

    # Fill valid indices and signals
    for i in range(n):
        d = date_idx[i]
        if d >= 0 and d < n_dates and not np.isnan(signal[i]):
            valid_indices[pos[d]] = i
            valid_signals[pos[d]] = signal[i]
            pos[d] += 1

    # Process each date in parallel
    for d in prange(n_dates):
        start = date_starts[d]
        end = date_starts[d + 1]
        count = end - start

        if count == 0:
            continue

        # Get signals for this date
        date_signals = valid_signals[start:end]
        date_indices = valid_indices[start:end]

        # Compute order (argsort) - sorted indices
        order = np.argsort(date_signals)

        # Compute percentile thresholds (matching slow path's np.percentile + assign_bond_bins)
        # Percentiles: [0, 20, 40, 60, 80, 100] for nport=5
        thresholds = np.zeros(nport + 1, dtype=np.float64)
        thresholds[0] = -np.inf  # First threshold is always -inf

        for p in range(1, nport + 1):
            # np.percentile position (0-100 scale to 0-(count-1) index)
            pct = (p * 100.0 / nport)
            # Linear interpolation method (matches numpy default)
            pos = (pct / 100.0) * (count - 1)
            idx_low = int(pos)
            idx_high = idx_low + 1
            frac = pos - idx_low

            if idx_high >= count:
                thresholds[p] = date_signals[order[count - 1]]
            else:
                # Linear interpolation
                val_low = date_signals[order[idx_low]]
                val_high = date_signals[order[idx_high]]
                thresholds[p] = val_low + frac * (val_high - val_low)

        # Assign bins based on value > thres[p] AND value <= thres[p+1]
        # (matching slow path's assign_bond_bins)
        for i in range(count):
            orig_idx = date_indices[i]
            val = date_signals[i]

            for p in range(nport):
                if val > thresholds[p] and val <= thresholds[p + 1]:
                    ranks[orig_idx] = p + 1
                    break

    return ranks


@njit(cache=True, parallel=True)
def compute_all_returns_ultrafast(
    ret_date_idx: np.ndarray,    # (n,) date index for return observations
    ret_id_idx: np.ndarray,      # (n,) bond ID index for return observations
    returns: np.ndarray,         # (n,) return values
    weights: np.ndarray,         # (n,) VW weights
    form_date_idx: np.ndarray,   # (m,) date index for formation observations
    form_id_idx: np.ndarray,     # (m,) bond ID index for formation observations
    form_ranks: np.ndarray,      # (m,) portfolio ranks from formation
    n_dates: int,
    n_ids: int,
    nport: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute portfolio returns for ALL dates in one shot.

    Ultra-fast: builds rank lookup and computes returns in parallel.

    Parameters
    ----------
    ret_date_idx : np.ndarray
        Date index for return observations (0-indexed)
    ret_id_idx : np.ndarray
        Bond ID index for return observations
    returns : np.ndarray
        Return values
    weights : np.ndarray
        Value weights for VW calculation
    form_date_idx : np.ndarray
        Date index for formation observations
    form_id_idx : np.ndarray
        Bond ID index for formation observations
    form_ranks : np.ndarray
        Portfolio ranks from formation
    n_dates : int
        Total number of dates
    n_ids : int
        Total number of unique IDs
    nport : int
        Number of portfolios

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (ew_returns, vw_returns) each shape (n_dates, nport)
    """
    # Build rank lookup: (date, id) -> rank
    # Using flat array indexed by date * n_ids + id
    rank_lookup = np.full(n_dates * n_ids, np.nan, dtype=np.float64)

    m = len(form_date_idx)
    for i in range(m):
        d = form_date_idx[i]
        bond_id = form_id_idx[i]
        if d >= 0 and d < n_dates and bond_id >= 0 and bond_id < n_ids:
            rank_lookup[d * n_ids + bond_id] = form_ranks[i]

    # Initialize output arrays
    ew_ret = np.full((n_dates, nport), np.nan, dtype=np.float64)
    vw_ret = np.full((n_dates, nport), np.nan, dtype=np.float64)

    # Temporary accumulators (one set per date to enable parallelism)
    # We'll use atomic-style accumulation

    # First, count observations per (date, portfolio)
    n_ret = len(ret_date_idx)

    # Process in parallel by date
    for d in prange(n_dates):
        # For this return date d, formation was at d-1
        if d == 0:
            continue

        form_d = d - 1

        # Accumulators for this date
        sum_ret = np.zeros(nport, dtype=np.float64)
        sum_wret = np.zeros(nport, dtype=np.float64)
        sum_weight = np.zeros(nport, dtype=np.float64)
        count = np.zeros(nport, dtype=np.int64)

        # Iterate through all return observations to find those at date d
        for i in range(n_ret):
            if ret_date_idx[i] != d:
                continue

            bond_id = ret_id_idx[i]
            ret_val = returns[i]
            weight = weights[i]

            # Look up rank from formation date
            lookup_idx = form_d * n_ids + bond_id
            if lookup_idx < 0 or lookup_idx >= len(rank_lookup):
                continue

            rank = rank_lookup[lookup_idx]
            if np.isnan(rank) or np.isnan(ret_val):
                continue

            # Skip bonds that don't exist at VW date (d-1)
            # This matches slow path's 3-way intersection logic
            if np.isnan(weight):
                continue

            p = int(rank) - 1  # Convert to 0-indexed
            if p < 0 or p >= nport:
                continue

            sum_ret[p] += ret_val
            count[p] += 1

            if weight > 0:
                sum_wret[p] += ret_val * weight
                sum_weight[p] += weight

        # Compute averages
        for p in range(nport):
            if count[p] > 0:
                ew_ret[d, p] = sum_ret[p] / count[p]
            if sum_weight[p] > 0:
                vw_ret[d, p] = sum_wret[p] / sum_weight[p]

    return ew_ret, vw_ret


@njit(cache=True, parallel=True)
def compute_staggered_returns_ultrafast(
    ret_date_idx: np.ndarray,    # (n,) date index for return observations
    ret_id_idx: np.ndarray,      # (n,) bond ID index for return observations
    returns: np.ndarray,         # (n,) return values
    vw_lookup: np.ndarray,       # VW lookup table: vw_lookup[date * n_ids + id]
    form_date_idx: np.ndarray,   # (m,) date index for formation observations
    form_id_idx: np.ndarray,     # (m,) bond ID index for formation observations
    form_ranks: np.ndarray,      # (m,) portfolio ranks from formation
    n_dates: int,
    n_ids: int,
    nport: int,
    hor: int,                    # holding period (number of cohorts)
    use_dynamic_weights: bool    # True: VW from d-1, False: VW from formation date
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute staggered portfolio returns for ALL dates in one shot.

    Ultra-fast version for holding_period > 1.

    Parameters
    ----------
    vw_lookup : np.ndarray
        VW lookup table: vw_lookup[date * n_ids + bond_id] = VW
    hor : int
        Holding period (number of cohorts to average)
    use_dynamic_weights : bool
        If True, use VW from day before return date (d-1) - same for all cohorts.
        If False, use VW from formation date (form_d) - different per cohort.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (ew_returns, vw_returns) each shape (n_dates, nport)
    """
    # Build rank lookup: (date, id) -> rank
    rank_lookup = np.full(n_dates * n_ids, np.nan, dtype=np.float64)

    m = len(form_date_idx)
    for i in range(m):
        d = form_date_idx[i]
        bond_id = form_id_idx[i]
        if d >= 0 and d < n_dates and bond_id >= 0 and bond_id < n_ids:
            rank_lookup[d * n_ids + bond_id] = form_ranks[i]

    # Initialize output arrays
    ew_ret = np.full((n_dates, nport), np.nan, dtype=np.float64)
    vw_ret = np.full((n_dates, nport), np.nan, dtype=np.float64)

    n_ret = len(ret_date_idx)

    # Process in parallel by date
    for d in prange(n_dates):
        if d == 0:
            continue

        # Accumulators for cohort averaging
        cohort_ew = np.full((hor, nport), np.nan, dtype=np.float64)
        cohort_vw = np.full((hor, nport), np.nan, dtype=np.float64)

        for cohort in range(hor):
            # Find formation date for this cohort
            # At return date d, cohort c was formed at formation_date
            if d < cohort + 1:
                continue

            offset = (d - 1 - cohort) % hor
            form_d = d - 1 - offset

            if form_d < 0 or form_d >= n_dates:
                continue

            # Accumulators for this cohort
            sum_ret = np.zeros(nport, dtype=np.float64)
            sum_wret = np.zeros(nport, dtype=np.float64)
            sum_weight = np.zeros(nport, dtype=np.float64)
            count = np.zeros(nport, dtype=np.int64)

            # Find return observations at date d
            for i in range(n_ret):
                if ret_date_idx[i] != d:
                    continue

                bond_id = ret_id_idx[i]
                ret_val = returns[i]

                # STEP 1: INTERSECTION CHECK - Always use formation date
                # This matches slow path's intersect_id(It0, It1, It1m) where
                # It1m is ALWAYS at formation date (form_d), not d-1.
                # The dynamic_weights setting only affects VW weighting, not intersection.
                form_vw_lookup_idx = form_d * n_ids + bond_id
                if form_vw_lookup_idx < 0 or form_vw_lookup_idx >= len(vw_lookup):
                    continue
                form_weight = vw_lookup[form_vw_lookup_idx]
                if np.isnan(form_weight):
                    continue  # Bond doesn't exist at formation date - skip

                # STEP 2: Look up rank from formation date
                lookup_idx = form_d * n_ids + bond_id
                if lookup_idx < 0 or lookup_idx >= len(rank_lookup):
                    continue

                rank = rank_lookup[lookup_idx]
                if np.isnan(rank) or np.isnan(ret_val):
                    continue

                p = int(rank) - 1
                if p < 0 or p >= nport:
                    continue

                # STEP 3: EW - Always include (bond passed intersection check)
                sum_ret[p] += ret_val
                count[p] += 1

                # STEP 4: VW WEIGHTING - Use appropriate date based on dynamic_weights
                # - True: VW from d-1 (day before return date) - same for all cohorts
                # - False: VW from form_d (formation date) - different per cohort
                if use_dynamic_weights:
                    vw_date = d - 1
                else:
                    vw_date = form_d

                vw_lookup_idx = vw_date * n_ids + bond_id
                if vw_lookup_idx >= 0 and vw_lookup_idx < len(vw_lookup):
                    weight = vw_lookup[vw_lookup_idx]
                else:
                    weight = np.nan

                # STEP 5: VW - Only include if valid weight at weighting date
                if not np.isnan(weight) and weight > 0:
                    sum_wret[p] += ret_val * weight
                    sum_weight[p] += weight

            # Compute cohort returns
            for p in range(nport):
                if count[p] > 0:
                    cohort_ew[cohort, p] = sum_ret[p] / count[p]
                if sum_weight[p] > 0:
                    cohort_vw[cohort, p] = sum_wret[p] / sum_weight[p]

        # Average across cohorts (ignoring NaN)
        for p in range(nport):
            ew_sum = 0.0
            vw_sum = 0.0
            ew_count = 0
            vw_count = 0

            for cohort in range(hor):
                if not np.isnan(cohort_ew[cohort, p]):
                    ew_sum += cohort_ew[cohort, p]
                    ew_count += 1
                if not np.isnan(cohort_vw[cohort, p]):
                    vw_sum += cohort_vw[cohort, p]
                    vw_count += 1

            if ew_count > 0:
                ew_ret[d, p] = ew_sum / ew_count
            if vw_count > 0:
                vw_ret[d, p] = vw_sum / vw_count

    return ew_ret, vw_ret


# =============================================================================
# Data Uncertainty Fast Path - Batched Filter Processing
# =============================================================================

@njit(cache=True, parallel=True)
def compute_ranks_with_filter_mask(
    date_idx: np.ndarray,      # (n,) date index for each row
    signal: np.ndarray,        # (n,) signal values to rank
    filter_mask: np.ndarray,   # (n,) boolean mask - True = include in ranking
    n_dates: int,
    nport: int
) -> np.ndarray:
    """
    Compute portfolio ranks for observations that pass the filter.

    Only observations with filter_mask = True are included in ranking.
    Observations with filter_mask = False get rank = NaN.

    Parameters
    ----------
    date_idx : np.ndarray
        Date index (0-indexed) for each observation
    signal : np.ndarray
        Signal values to rank
    filter_mask : np.ndarray
        Boolean mask - True means include in ranking, False means exclude
    n_dates : int
        Total number of dates
    nport : int
        Number of portfolios

    Returns
    -------
    np.ndarray
        Portfolio rank (1-indexed) for each observation, NaN for excluded
    """
    n = len(date_idx)
    ranks = np.full(n, np.nan, dtype=np.float64)

    # Count valid observations per date (filter_mask=True AND signal not NaN)
    counts = np.zeros(n_dates, dtype=np.int64)
    for i in range(n):
        d = date_idx[i]
        if d >= 0 and d < n_dates and filter_mask[i] and not np.isnan(signal[i]):
            counts[d] += 1

    # Build index arrays
    date_starts = np.zeros(n_dates + 1, dtype=np.int64)
    for d in range(n_dates):
        date_starts[d + 1] = date_starts[d] + counts[d]

    total_valid = date_starts[n_dates]
    valid_indices = np.zeros(total_valid, dtype=np.int64)
    valid_signals = np.zeros(total_valid, dtype=np.float64)

    pos = np.zeros(n_dates, dtype=np.int64)
    for d in range(n_dates):
        pos[d] = date_starts[d]

    for i in range(n):
        d = date_idx[i]
        if d >= 0 and d < n_dates and filter_mask[i] and not np.isnan(signal[i]):
            valid_indices[pos[d]] = i
            valid_signals[pos[d]] = signal[i]
            pos[d] += 1

    # Process each date in parallel
    for d in prange(n_dates):
        start = date_starts[d]
        end = date_starts[d + 1]
        count = end - start

        if count == 0:
            continue

        date_signals = valid_signals[start:end]
        date_indices = valid_indices[start:end]

        order = np.argsort(date_signals)

        # Compute percentile thresholds
        thresholds = np.zeros(nport + 1, dtype=np.float64)
        thresholds[0] = -np.inf

        for p in range(1, nport + 1):
            pct = (p * 100.0 / nport)
            pos_f = (pct / 100.0) * (count - 1)
            idx_low = int(pos_f)
            idx_high = idx_low + 1
            frac = pos_f - idx_low

            if idx_high >= count:
                thresholds[p] = date_signals[order[count - 1]]
            else:
                val_low = date_signals[order[idx_low]]
                val_high = date_signals[order[idx_high]]
                thresholds[p] = val_low + frac * (val_high - val_low)

        # Assign bins
        for i in range(count):
            orig_idx = date_indices[i]
            val = date_signals[i]

            for p in range(nport):
                if val > thresholds[p] and val <= thresholds[p + 1]:
                    ranks[orig_idx] = p + 1
                    break

    return ranks


@njit(cache=True, parallel=True)
def compute_returns_multi_filter_hp1(
    ret_date_idx: np.ndarray,      # (n,) date index for return observations
    ret_id_idx: np.ndarray,        # (n,) bond ID index for return observations
    returns_ea: np.ndarray,        # (n,) EA return values (original ret)
    returns_ep: np.ndarray,        # (n, n_filters) EP return values per filter
    weights: np.ndarray,           # (n,) VW weights (from d-1)
    rank_lookup: np.ndarray,       # (n_dates * n_ids,) flat rank lookup
    n_dates: int,
    n_ids: int,
    nport: int,
    n_filters: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute portfolio returns for ALL dates and ALL filters at once (HP=1).

    This is the ultra-fast version for data uncertainty analysis.
    Computes EA returns once, EP returns for each filter.

    Parameters
    ----------
    ret_date_idx : np.ndarray
        Date index for return observations (0-indexed)
    ret_id_idx : np.ndarray
        Bond ID index for return observations
    returns_ea : np.ndarray
        EA return values (same for all filters)
    returns_ep : np.ndarray
        EP return values per filter, shape (n_obs, n_filters)
    weights : np.ndarray
        Value weights for VW calculation (from d-1)
    rank_lookup : np.ndarray
        Pre-computed rank lookup table
    n_dates, n_ids, nport, n_filters : int
        Dimensions

    Returns
    -------
    Tuple of 4 arrays:
        ew_ea: (n_dates, nport) - EW EA returns (same for all filters)
        vw_ea: (n_dates, nport) - VW EA returns (same for all filters)
        ew_ep: (n_dates, nport, n_filters) - EW EP returns per filter
        vw_ep: (n_dates, nport, n_filters) - VW EP returns per filter
    """
    # Initialize output arrays
    ew_ea = np.full((n_dates, nport), np.nan, dtype=np.float64)
    vw_ea = np.full((n_dates, nport), np.nan, dtype=np.float64)
    ew_ep = np.full((n_dates, nport, n_filters), np.nan, dtype=np.float64)
    vw_ep = np.full((n_dates, nport, n_filters), np.nan, dtype=np.float64)

    n_ret = len(ret_date_idx)

    # Process in parallel by date
    for d in prange(n_dates):
        if d == 0:
            continue

        form_d = d - 1

        # EA accumulators (one set for all filters)
        ea_sum_ret = np.zeros(nport, dtype=np.float64)
        ea_sum_wret = np.zeros(nport, dtype=np.float64)
        ea_sum_weight = np.zeros(nport, dtype=np.float64)
        ea_count = np.zeros(nport, dtype=np.int64)

        # EP accumulators (one set per filter)
        ep_sum_ret = np.zeros((nport, n_filters), dtype=np.float64)
        ep_sum_wret = np.zeros((nport, n_filters), dtype=np.float64)
        ep_sum_weight = np.zeros((nport, n_filters), dtype=np.float64)
        ep_count = np.zeros((nport, n_filters), dtype=np.int64)

        # Single pass through return observations
        for i in range(n_ret):
            if ret_date_idx[i] != d:
                continue

            bond_id = ret_id_idx[i]
            ret_ea = returns_ea[i]
            weight = weights[i]

            # Look up rank from formation date
            lookup_idx = form_d * n_ids + bond_id
            if lookup_idx < 0 or lookup_idx >= len(rank_lookup):
                continue

            rank = rank_lookup[lookup_idx]
            if np.isnan(rank):
                continue

            # Skip bonds that don't exist at VW date
            if np.isnan(weight):
                continue

            p = int(rank) - 1
            if p < 0 or p >= nport:
                continue

            # EA aggregation (if EA return is valid)
            if not np.isnan(ret_ea):
                ea_sum_ret[p] += ret_ea
                ea_count[p] += 1
                if weight > 0:
                    ea_sum_wret[p] += ret_ea * weight
                    ea_sum_weight[p] += weight

            # EP aggregation for each filter
            for f in range(n_filters):
                ret_ep_val = returns_ep[i, f]
                if not np.isnan(ret_ep_val):
                    ep_sum_ret[p, f] += ret_ep_val
                    ep_count[p, f] += 1
                    if weight > 0:
                        ep_sum_wret[p, f] += ret_ep_val * weight
                        ep_sum_weight[p, f] += weight

        # Compute EA averages
        for p in range(nport):
            if ea_count[p] > 0:
                ew_ea[d, p] = ea_sum_ret[p] / ea_count[p]
            if ea_sum_weight[p] > 0:
                vw_ea[d, p] = ea_sum_wret[p] / ea_sum_weight[p]

        # Compute EP averages for each filter
        for p in range(nport):
            for f in range(n_filters):
                if ep_count[p, f] > 0:
                    ew_ep[d, p, f] = ep_sum_ret[p, f] / ep_count[p, f]
                if ep_sum_weight[p, f] > 0:
                    vw_ep[d, p, f] = ep_sum_wret[p, f] / ep_sum_weight[p, f]

    return ew_ea, vw_ea, ew_ep, vw_ep


@njit(cache=True, parallel=True)
def compute_returns_multi_filter_staggered(
    ret_date_idx: np.ndarray,      # (n,) date index for return observations
    ret_id_idx: np.ndarray,        # (n,) bond ID index for return observations
    returns_ea: np.ndarray,        # (n,) EA return values
    returns_ep: np.ndarray,        # (n, n_filters) EP return values per filter
    vw_lookup: np.ndarray,         # (n_dates * n_ids,) VW lookup table
    rank_lookup: np.ndarray,       # (n_dates * n_ids,) rank lookup table
    n_dates: int,
    n_ids: int,
    nport: int,
    n_filters: int,
    hor: int,                      # holding period (number of cohorts)
    use_dynamic_weights: bool      # True: VW from d-1, False: VW from formation
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute staggered portfolio returns for ALL dates and ALL filters (HP>1).

    Similar to compute_returns_multi_filter_hp1 but with cohort averaging.
    """
    # Initialize output arrays
    ew_ea = np.full((n_dates, nport), np.nan, dtype=np.float64)
    vw_ea = np.full((n_dates, nport), np.nan, dtype=np.float64)
    ew_ep = np.full((n_dates, nport, n_filters), np.nan, dtype=np.float64)
    vw_ep = np.full((n_dates, nport, n_filters), np.nan, dtype=np.float64)

    n_ret = len(ret_date_idx)

    # Process in parallel by date
    for d in prange(n_dates):
        if d == 0:
            continue

        # Cohort accumulators for EA
        cohort_ew_ea = np.full((hor, nport), np.nan, dtype=np.float64)
        cohort_vw_ea = np.full((hor, nport), np.nan, dtype=np.float64)

        # Cohort accumulators for EP (per filter)
        cohort_ew_ep = np.full((hor, nport, n_filters), np.nan, dtype=np.float64)
        cohort_vw_ep = np.full((hor, nport, n_filters), np.nan, dtype=np.float64)

        for cohort in range(hor):
            if d < cohort + 1:
                continue

            offset = (d - 1 - cohort) % hor
            form_d = d - 1 - offset

            if form_d < 0 or form_d >= n_dates:
                continue

            # EA accumulators for this cohort
            ea_sum_ret = np.zeros(nport, dtype=np.float64)
            ea_sum_wret = np.zeros(nport, dtype=np.float64)
            ea_sum_weight = np.zeros(nport, dtype=np.float64)
            ea_count = np.zeros(nport, dtype=np.int64)

            # EP accumulators for this cohort (per filter)
            ep_sum_ret = np.zeros((nport, n_filters), dtype=np.float64)
            ep_sum_wret = np.zeros((nport, n_filters), dtype=np.float64)
            ep_sum_weight = np.zeros((nport, n_filters), dtype=np.float64)
            ep_count = np.zeros((nport, n_filters), dtype=np.int64)

            for i in range(n_ret):
                if ret_date_idx[i] != d:
                    continue

                bond_id = ret_id_idx[i]
                ret_ea = returns_ea[i]

                # VW date selection
                if use_dynamic_weights:
                    vw_date = d - 1
                else:
                    vw_date = form_d

                vw_lookup_idx = vw_date * n_ids + bond_id
                if vw_lookup_idx >= 0 and vw_lookup_idx < len(vw_lookup):
                    weight = vw_lookup[vw_lookup_idx]
                else:
                    weight = np.nan

                # Rank lookup
                lookup_idx = form_d * n_ids + bond_id
                if lookup_idx < 0 or lookup_idx >= len(rank_lookup):
                    continue

                rank = rank_lookup[lookup_idx]
                if np.isnan(rank) or np.isnan(weight):
                    continue

                p = int(rank) - 1
                if p < 0 or p >= nport:
                    continue

                # EA aggregation
                if not np.isnan(ret_ea):
                    ea_sum_ret[p] += ret_ea
                    ea_count[p] += 1
                    if weight > 0:
                        ea_sum_wret[p] += ret_ea * weight
                        ea_sum_weight[p] += weight

                # EP aggregation for each filter
                for f in range(n_filters):
                    ret_ep_val = returns_ep[i, f]
                    if not np.isnan(ret_ep_val):
                        ep_sum_ret[p, f] += ret_ep_val
                        ep_count[p, f] += 1
                        if weight > 0:
                            ep_sum_wret[p, f] += ret_ep_val * weight
                            ep_sum_weight[p, f] += weight

            # Compute cohort averages for EA
            for p in range(nport):
                if ea_count[p] > 0:
                    cohort_ew_ea[cohort, p] = ea_sum_ret[p] / ea_count[p]
                if ea_sum_weight[p] > 0:
                    cohort_vw_ea[cohort, p] = ea_sum_wret[p] / ea_sum_weight[p]

            # Compute cohort averages for EP
            for p in range(nport):
                for f in range(n_filters):
                    if ep_count[p, f] > 0:
                        cohort_ew_ep[cohort, p, f] = ep_sum_ret[p, f] / ep_count[p, f]
                    if ep_sum_weight[p, f] > 0:
                        cohort_vw_ep[cohort, p, f] = ep_sum_wret[p, f] / ep_sum_weight[p, f]

        # Average across cohorts for EA
        for p in range(nport):
            ew_sum = 0.0
            vw_sum = 0.0
            ew_cnt = 0
            vw_cnt = 0
            for cohort in range(hor):
                if not np.isnan(cohort_ew_ea[cohort, p]):
                    ew_sum += cohort_ew_ea[cohort, p]
                    ew_cnt += 1
                if not np.isnan(cohort_vw_ea[cohort, p]):
                    vw_sum += cohort_vw_ea[cohort, p]
                    vw_cnt += 1
            if ew_cnt > 0:
                ew_ea[d, p] = ew_sum / ew_cnt
            if vw_cnt > 0:
                vw_ea[d, p] = vw_sum / vw_cnt

        # Average across cohorts for EP (per filter)
        for p in range(nport):
            for f in range(n_filters):
                ew_sum = 0.0
                vw_sum = 0.0
                ew_cnt = 0
                vw_cnt = 0
                for cohort in range(hor):
                    if not np.isnan(cohort_ew_ep[cohort, p, f]):
                        ew_sum += cohort_ew_ep[cohort, p, f]
                        ew_cnt += 1
                    if not np.isnan(cohort_vw_ep[cohort, p, f]):
                        vw_sum += cohort_vw_ep[cohort, p, f]
                        vw_cnt += 1
                if ew_cnt > 0:
                    ew_ep[d, p, f] = ew_sum / ew_cnt
                if vw_cnt > 0:
                    vw_ep[d, p, f] = vw_sum / vw_cnt

    return ew_ea, vw_ea, ew_ep, vw_ep


# =============================================================================
# BLAZING FAST: Filter-Specific Ranking Kernels for Data Uncertainty
# =============================================================================

@njit(cache=True, parallel=True)
def compute_ranks_all_filters(
    date_idx: np.ndarray,          # (n,) date index for each row
    signal: np.ndarray,            # (n,) signal values to rank
    filter_masks: np.ndarray,      # (n, n_filters) boolean masks - True = include
    n_dates: int,
    nport: int,
    n_filters: int
) -> np.ndarray:
    """
    Compute portfolio ranks for ALL filters at once in parallel.

    Each filter has its own ranking based on its filter mask.

    Parameters
    ----------
    date_idx : np.ndarray
        Date index (0-indexed) for each observation
    signal : np.ndarray
        Signal values to rank
    filter_masks : np.ndarray
        Boolean masks (n_obs, n_filters) - True means include in ranking
    n_dates : int
        Total number of dates
    nport : int
        Number of portfolios
    n_filters : int
        Number of filters

    Returns
    -------
    np.ndarray
        Shape (n_obs, n_filters) - portfolio rank per observation per filter
    """
    n = len(date_idx)
    ranks = np.full((n, n_filters), np.nan, dtype=np.float64)

    # Process each (date, filter) combination in parallel
    # Flatten to single loop for better parallelization
    n_combos = n_dates * n_filters

    for combo in prange(n_combos):
        d = combo // n_filters
        f = combo % n_filters

        # Count valid observations for this (date, filter)
        count = 0
        for i in range(n):
            if date_idx[i] == d and filter_masks[i, f] and not np.isnan(signal[i]):
                count += 1

        if count == 0:
            continue

        # Gather valid observations
        valid_indices = np.empty(count, dtype=np.int64)
        valid_signals = np.empty(count, dtype=np.float64)
        pos = 0
        for i in range(n):
            if date_idx[i] == d and filter_masks[i, f] and not np.isnan(signal[i]):
                valid_indices[pos] = i
                valid_signals[pos] = signal[i]
                pos += 1

        # Sort by signal
        order = np.argsort(valid_signals)

        # Compute percentile thresholds
        thresholds = np.empty(nport + 1, dtype=np.float64)
        thresholds[0] = -np.inf

        for p in range(1, nport + 1):
            pct = (p * 100.0 / nport)
            pos_f = (pct / 100.0) * (count - 1)
            idx_low = int(pos_f)
            idx_high = idx_low + 1
            frac = pos_f - idx_low

            if idx_high >= count:
                thresholds[p] = valid_signals[order[count - 1]]
            else:
                val_low = valid_signals[order[idx_low]]
                val_high = valid_signals[order[idx_high]]
                thresholds[p] = val_low + frac * (val_high - val_low)

        # Assign bins
        for i in range(count):
            orig_idx = valid_indices[i]
            val = valid_signals[i]

            for p in range(nport):
                if val > thresholds[p] and val <= thresholds[p + 1]:
                    ranks[orig_idx, f] = p + 1
                    break

    return ranks


@njit(cache=True, parallel=True)
def build_rank_lookups_all_filters(
    date_idx: np.ndarray,      # (n,) date index
    id_idx: np.ndarray,        # (n,) bond ID index
    ranks: np.ndarray,         # (n, n_filters) ranks per filter
    n_dates: int,
    n_ids: int,
    n_filters: int
) -> np.ndarray:
    """
    Build rank lookup tables for ALL filters at once.

    Returns
    -------
    np.ndarray
        Shape (n_dates * n_ids, n_filters) - rank lookup per filter
    """
    n = len(date_idx)
    rank_lookups = np.full((n_dates * n_ids, n_filters), np.nan, dtype=np.float64)

    # Parallel over observations (each writes to unique location)
    for i in prange(n):
        d = date_idx[i]
        bond_id = id_idx[i]
        if d >= 0 and d < n_dates and bond_id >= 0 and bond_id < n_ids:
            lookup_idx = d * n_ids + bond_id
            for f in range(n_filters):
                rank_lookups[lookup_idx, f] = ranks[i, f]

    return rank_lookups


@njit(cache=True, parallel=True)
def compute_ls_returns_all_filters_hp1(
    ret_date_idx: np.ndarray,      # (n,) date index for return observations
    ret_id_idx: np.ndarray,        # (n,) bond ID index
    returns_ea: np.ndarray,        # (n,) EA returns (original ret)
    returns_ep: np.ndarray,        # (n, n_filters) EP returns per filter
    weights: np.ndarray,           # (n,) VW weights from d-1
    rank_lookups: np.ndarray,      # (n_dates * n_ids, n_filters) rank lookups
    n_dates: int,
    n_ids: int,
    nport: int,
    n_filters: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute long-short returns for ALL dates and ALL filters (HP=1).

    Each filter uses its own ranking for both EA and EP returns.

    Returns
    -------
    Tuple of 4 arrays:
        ew_ea_ls: (n_dates, n_filters) - EW EA long-short per filter
        vw_ea_ls: (n_dates, n_filters) - VW EA long-short per filter
        ew_ep_ls: (n_dates, n_filters) - EW EP long-short per filter
        vw_ep_ls: (n_dates, n_filters) - VW EP long-short per filter
    """
    n_ret = len(ret_date_idx)

    # Output: long-short returns per date per filter
    ew_ea_ls = np.full((n_dates, n_filters), np.nan, dtype=np.float64)
    vw_ea_ls = np.full((n_dates, n_filters), np.nan, dtype=np.float64)
    ew_ep_ls = np.full((n_dates, n_filters), np.nan, dtype=np.float64)
    vw_ep_ls = np.full((n_dates, n_filters), np.nan, dtype=np.float64)

    # Process in parallel over (date, filter) combinations
    n_combos = n_dates * n_filters

    for combo in prange(n_combos):
        d = combo // n_filters
        f = combo % n_filters

        if d == 0:
            continue

        form_d = d - 1

        # Accumulators for this (date, filter)
        ea_sum_ret = np.zeros(nport, dtype=np.float64)
        ea_sum_wret = np.zeros(nport, dtype=np.float64)
        ea_sum_weight = np.zeros(nport, dtype=np.float64)
        ea_count = np.zeros(nport, dtype=np.int64)

        ep_sum_ret = np.zeros(nport, dtype=np.float64)
        ep_sum_wret = np.zeros(nport, dtype=np.float64)
        ep_sum_weight = np.zeros(nport, dtype=np.float64)
        ep_count = np.zeros(nport, dtype=np.int64)

        # Process all return observations for this date
        for i in range(n_ret):
            if ret_date_idx[i] != d:
                continue

            bond_id = ret_id_idx[i]
            weight = weights[i]

            # Skip bonds without VW at d-1
            if np.isnan(weight):
                continue

            # Look up rank from formation date for THIS filter
            lookup_idx = form_d * n_ids + bond_id
            if lookup_idx < 0 or lookup_idx >= n_dates * n_ids:
                continue

            rank = rank_lookups[lookup_idx, f]
            if np.isnan(rank):
                continue

            p = int(rank) - 1
            if p < 0 or p >= nport:
                continue

            # EA aggregation
            ret_ea = returns_ea[i]
            if not np.isnan(ret_ea):
                ea_sum_ret[p] += ret_ea
                ea_count[p] += 1
                if weight > 0:
                    ea_sum_wret[p] += ret_ea * weight
                    ea_sum_weight[p] += weight

            # EP aggregation
            ret_ep = returns_ep[i, f]
            if not np.isnan(ret_ep):
                ep_sum_ret[p] += ret_ep
                ep_count[p] += 1
                if weight > 0:
                    ep_sum_wret[p] += ret_ep * weight
                    ep_sum_weight[p] += weight

        # Compute portfolio returns
        ew_ea_ptf = np.full(nport, np.nan, dtype=np.float64)
        vw_ea_ptf = np.full(nport, np.nan, dtype=np.float64)
        ew_ep_ptf = np.full(nport, np.nan, dtype=np.float64)
        vw_ep_ptf = np.full(nport, np.nan, dtype=np.float64)

        for p in range(nport):
            if ea_count[p] > 0:
                ew_ea_ptf[p] = ea_sum_ret[p] / ea_count[p]
            if ea_sum_weight[p] > 0:
                vw_ea_ptf[p] = ea_sum_wret[p] / ea_sum_weight[p]
            if ep_count[p] > 0:
                ew_ep_ptf[p] = ep_sum_ret[p] / ep_count[p]
            if ep_sum_weight[p] > 0:
                vw_ep_ptf[p] = ep_sum_wret[p] / ep_sum_weight[p]

        # Long-short (high - low)
        if not np.isnan(ew_ea_ptf[nport-1]) and not np.isnan(ew_ea_ptf[0]):
            ew_ea_ls[d, f] = ew_ea_ptf[nport-1] - ew_ea_ptf[0]
        if not np.isnan(vw_ea_ptf[nport-1]) and not np.isnan(vw_ea_ptf[0]):
            vw_ea_ls[d, f] = vw_ea_ptf[nport-1] - vw_ea_ptf[0]
        if not np.isnan(ew_ep_ptf[nport-1]) and not np.isnan(ew_ep_ptf[0]):
            ew_ep_ls[d, f] = ew_ep_ptf[nport-1] - ew_ep_ptf[0]
        if not np.isnan(vw_ep_ptf[nport-1]) and not np.isnan(vw_ep_ptf[0]):
            vw_ep_ls[d, f] = vw_ep_ptf[nport-1] - vw_ep_ptf[0]

    return ew_ea_ls, vw_ea_ls, ew_ep_ls, vw_ep_ls


@njit(cache=True, parallel=True)
def compute_ls_returns_all_filters_staggered(
    ret_date_idx: np.ndarray,      # (n,) date index
    ret_id_idx: np.ndarray,        # (n,) bond ID index
    returns_ea: np.ndarray,        # (n,) EA returns
    returns_ep: np.ndarray,        # (n, n_filters) EP returns per filter
    vw_lookup: np.ndarray,         # (n_dates * n_ids,) VW lookup table
    rank_lookups: np.ndarray,      # (n_dates * n_ids, n_filters) rank lookups
    n_dates: int,
    n_ids: int,
    nport: int,
    n_filters: int,
    hp: int,
    use_dynamic_weights: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute long-short returns for ALL dates and ALL filters with staggered rebalancing (HP>1).

    Each filter uses its own ranking for both EA and EP returns.
    """
    n_ret = len(ret_date_idx)

    ew_ea_ls = np.full((n_dates, n_filters), np.nan, dtype=np.float64)
    vw_ea_ls = np.full((n_dates, n_filters), np.nan, dtype=np.float64)
    ew_ep_ls = np.full((n_dates, n_filters), np.nan, dtype=np.float64)
    vw_ep_ls = np.full((n_dates, n_filters), np.nan, dtype=np.float64)

    # Process in parallel over (date, filter) combinations
    n_combos = n_dates * n_filters

    for combo in prange(n_combos):
        d = combo // n_filters
        f = combo % n_filters

        if d == 0:
            continue

        # Cohort accumulators
        cohort_ew_ea = np.full((hp, nport), np.nan, dtype=np.float64)
        cohort_vw_ea = np.full((hp, nport), np.nan, dtype=np.float64)
        cohort_ew_ep = np.full((hp, nport), np.nan, dtype=np.float64)
        cohort_vw_ep = np.full((hp, nport), np.nan, dtype=np.float64)

        for cohort in range(hp):
            if d < cohort + 1:
                continue

            offset = (d - 1 - cohort) % hp
            form_d = d - 1 - offset

            if form_d < 0 or form_d >= n_dates:
                continue

            # VW date depends on dynamic_weights
            if use_dynamic_weights:
                vw_d = d - 1
            else:
                vw_d = form_d

            # Accumulators for this cohort
            ea_sum_ret = np.zeros(nport, dtype=np.float64)
            ea_sum_wret = np.zeros(nport, dtype=np.float64)
            ea_sum_weight = np.zeros(nport, dtype=np.float64)
            ea_count = np.zeros(nport, dtype=np.int64)

            ep_sum_ret = np.zeros(nport, dtype=np.float64)
            ep_sum_wret = np.zeros(nport, dtype=np.float64)
            ep_sum_weight = np.zeros(nport, dtype=np.float64)
            ep_count = np.zeros(nport, dtype=np.int64)

            # Process return observations
            for i in range(n_ret):
                if ret_date_idx[i] != d:
                    continue

                bond_id = ret_id_idx[i]

                # Get VW from appropriate date
                vw_lookup_idx = vw_d * n_ids + bond_id
                if vw_lookup_idx < 0 or vw_lookup_idx >= len(vw_lookup):
                    continue
                weight = vw_lookup[vw_lookup_idx]
                if np.isnan(weight):
                    continue

                # Look up rank from formation date
                rank_lookup_idx = form_d * n_ids + bond_id
                if rank_lookup_idx < 0 or rank_lookup_idx >= n_dates * n_ids:
                    continue
                rank = rank_lookups[rank_lookup_idx, f]
                if np.isnan(rank):
                    continue

                p = int(rank) - 1
                if p < 0 or p >= nport:
                    continue

                # EA aggregation
                ret_ea = returns_ea[i]
                if not np.isnan(ret_ea):
                    ea_sum_ret[p] += ret_ea
                    ea_count[p] += 1
                    if weight > 0:
                        ea_sum_wret[p] += ret_ea * weight
                        ea_sum_weight[p] += weight

                # EP aggregation
                ret_ep = returns_ep[i, f]
                if not np.isnan(ret_ep):
                    ep_sum_ret[p] += ret_ep
                    ep_count[p] += 1
                    if weight > 0:
                        ep_sum_wret[p] += ret_ep * weight
                        ep_sum_weight[p] += weight

            # Compute cohort averages
            for p in range(nport):
                if ea_count[p] > 0:
                    cohort_ew_ea[cohort, p] = ea_sum_ret[p] / ea_count[p]
                if ea_sum_weight[p] > 0:
                    cohort_vw_ea[cohort, p] = ea_sum_wret[p] / ea_sum_weight[p]
                if ep_count[p] > 0:
                    cohort_ew_ep[cohort, p] = ep_sum_ret[p] / ep_count[p]
                if ep_sum_weight[p] > 0:
                    cohort_vw_ep[cohort, p] = ep_sum_wret[p] / ep_sum_weight[p]

        # Average across cohorts
        ew_ea_ptf = np.full(nport, np.nan, dtype=np.float64)
        vw_ea_ptf = np.full(nport, np.nan, dtype=np.float64)
        ew_ep_ptf = np.full(nport, np.nan, dtype=np.float64)
        vw_ep_ptf = np.full(nport, np.nan, dtype=np.float64)

        for p in range(nport):
            ew_ea_sum, vw_ea_sum = 0.0, 0.0
            ew_ep_sum, vw_ep_sum = 0.0, 0.0
            ew_ea_cnt, vw_ea_cnt = 0, 0
            ew_ep_cnt, vw_ep_cnt = 0, 0

            for cohort in range(hp):
                if not np.isnan(cohort_ew_ea[cohort, p]):
                    ew_ea_sum += cohort_ew_ea[cohort, p]
                    ew_ea_cnt += 1
                if not np.isnan(cohort_vw_ea[cohort, p]):
                    vw_ea_sum += cohort_vw_ea[cohort, p]
                    vw_ea_cnt += 1
                if not np.isnan(cohort_ew_ep[cohort, p]):
                    ew_ep_sum += cohort_ew_ep[cohort, p]
                    ew_ep_cnt += 1
                if not np.isnan(cohort_vw_ep[cohort, p]):
                    vw_ep_sum += cohort_vw_ep[cohort, p]
                    vw_ep_cnt += 1

            if ew_ea_cnt > 0:
                ew_ea_ptf[p] = ew_ea_sum / ew_ea_cnt
            if vw_ea_cnt > 0:
                vw_ea_ptf[p] = vw_ea_sum / vw_ea_cnt
            if ew_ep_cnt > 0:
                ew_ep_ptf[p] = ew_ep_sum / ew_ep_cnt
            if vw_ep_cnt > 0:
                vw_ep_ptf[p] = vw_ep_sum / vw_ep_cnt

        # Long-short
        if not np.isnan(ew_ea_ptf[nport-1]) and not np.isnan(ew_ea_ptf[0]):
            ew_ea_ls[d, f] = ew_ea_ptf[nport-1] - ew_ea_ptf[0]
        if not np.isnan(vw_ea_ptf[nport-1]) and not np.isnan(vw_ea_ptf[0]):
            vw_ea_ls[d, f] = vw_ea_ptf[nport-1] - vw_ea_ptf[0]
        if not np.isnan(ew_ep_ptf[nport-1]) and not np.isnan(ew_ep_ptf[0]):
            ew_ep_ls[d, f] = ew_ep_ptf[nport-1] - ew_ep_ptf[0]
        if not np.isnan(vw_ep_ptf[nport-1]) and not np.isnan(vw_ep_ptf[0]):
            vw_ep_ls[d, f] = vw_ep_ptf[nport-1] - vw_ep_ptf[0]

    return ew_ea_ls, vw_ea_ls, ew_ep_ls, vw_ep_ls


# =============================================================================
# Momentum/LTreversal Signal Computation (Panel-Based)
# =============================================================================

@njit(cache=True, parallel=True)
def compute_momentum_signals_panel(
    logret_all: np.ndarray,
    bond_starts: np.ndarray,
    lookback: int,
    skip: int
) -> np.ndarray:
    """
    Compute Momentum signals for ALL filters in parallel over bonds.

    This matches pandas groupby('ID').rolling() behavior exactly by processing
    each bond's rows independently (not using a dense date×bond matrix).

    Parameters
    ----------
    logret_all : np.ndarray
        Log returns for all filters, shape (n_obs, n_filters)
        Data must be sorted by (ID, date)
    bond_starts : np.ndarray
        Array of indices where each bond starts, length (n_bonds + 1)
        bond_starts[i] = start index of bond i
        bond_starts[n_bonds] = n_obs (sentinel)
    lookback : int
        Number of periods for momentum calculation (J)
    skip : int
        Number of periods to skip (most recent)

    Returns
    -------
    np.ndarray
        Signal values, shape (n_obs, n_filters)
        signal = exp(rolling_sum) - 1, shifted by skip
    """
    n_obs, n_filters = logret_all.shape
    signals = np.full((n_obs, n_filters), np.nan, dtype=np.float64)
    n_bonds = len(bond_starts) - 1

    # Parallel over bonds
    for bond_idx in prange(n_bonds):
        start = bond_starts[bond_idx]
        end = bond_starts[bond_idx + 1]
        bond_len = end - start

        if bond_len < lookback + skip:
            # Not enough observations for this bond
            continue

        # For each filter
        for f in range(n_filters):
            # Compute rolling sum for this bond's rows
            # First, compute raw rolling sums (before skip shift)
            raw_signals = np.full(bond_len, np.nan, dtype=np.float64)

            for i in range(lookback - 1, bond_len):
                window_sum = 0.0
                valid = True
                for j in range(i - lookback + 1, i + 1):
                    val = logret_all[start + j, f]
                    if np.isnan(val):
                        valid = False
                        break
                    window_sum += val
                if valid:
                    raw_signals[i] = window_sum

            # Apply skip (shift within bond) and exp transform
            for i in range(skip, bond_len):
                if not np.isnan(raw_signals[i - skip]):
                    signals[start + i, f] = np.exp(raw_signals[i - skip]) - 1.0

    return signals


@njit(cache=True, parallel=True)
def compute_ltreversal_signals_panel(
    logret_all: np.ndarray,
    bond_starts: np.ndarray,
    lookback: int,
    skip: int
) -> np.ndarray:
    """
    Compute LT-Reversal signals for ALL filters in parallel over bonds.

    LT-Reversal signal = cumulative return over (lookback) minus cumulative
    return over (skip), i.e., long-term return excluding recent return.

    This matches pandas groupby('ID').rolling() behavior exactly.

    Parameters
    ----------
    logret_all : np.ndarray
        Log returns for all filters, shape (n_obs, n_filters)
        Data must be sorted by (ID, date)
    bond_starts : np.ndarray
        Array of indices where each bond starts, length (n_bonds + 1)
    lookback : int
        Number of periods for long-term calculation (J)
    skip : int
        Number of recent periods to exclude

    Returns
    -------
    np.ndarray
        Signal values, shape (n_obs, n_filters)
    """
    n_obs, n_filters = logret_all.shape
    signals = np.full((n_obs, n_filters), np.nan, dtype=np.float64)
    n_bonds = len(bond_starts) - 1

    # Parallel over bonds
    for bond_idx in prange(n_bonds):
        start = bond_starts[bond_idx]
        end = bond_starts[bond_idx + 1]
        bond_len = end - start

        if bond_len < lookback + skip:
            continue

        for f in range(n_filters):
            # Compute rolling sums for lookback and skip windows
            long_sums = np.full(bond_len, np.nan, dtype=np.float64)
            recent_sums = np.full(bond_len, np.nan, dtype=np.float64)

            # Long-term rolling sum (lookback periods)
            for i in range(lookback - 1, bond_len):
                window_sum = 0.0
                valid = True
                for j in range(i - lookback + 1, i + 1):
                    val = logret_all[start + j, f]
                    if np.isnan(val):
                        valid = False
                        break
                    window_sum += val
                if valid:
                    long_sums[i] = window_sum

            # Recent rolling sum (skip periods)
            for i in range(skip - 1, bond_len):
                window_sum = 0.0
                valid = True
                for j in range(i - skip + 1, i + 1):
                    val = logret_all[start + j, f]
                    if np.isnan(val):
                        valid = False
                        break
                    window_sum += val
                if valid:
                    recent_sums[i] = window_sum

            # Signal = long - recent, shifted by skip, then exp transform
            for i in range(skip, bond_len):
                idx = i - skip
                if not np.isnan(long_sums[idx]) and not np.isnan(recent_sums[idx]):
                    log_signal = long_sums[idx] - recent_sums[idx]
                    signals[start + i, f] = np.exp(log_signal) - 1.0

    return signals


def get_bond_boundaries(id_values: np.ndarray) -> np.ndarray:
    """
    Find bond boundaries (where ID changes) in sorted panel data.

    Parameters
    ----------
    id_values : np.ndarray
        ID values for each observation, must be sorted by ID

    Returns
    -------
    np.ndarray
        Array of boundary indices, length (n_bonds + 1)
        bond_starts[i] = start index of bond i
        bond_starts[-1] = n_obs (sentinel)
    """
    n = len(id_values)
    if n == 0:
        return np.array([0], dtype=np.int64)

    # Find where ID changes
    changes = np.where(id_values[1:] != id_values[:-1])[0] + 1

    # Add start (0) and end (n) sentinels
    bond_starts = np.concatenate([
        np.array([0], dtype=np.int64),
        changes.astype(np.int64),
        np.array([n], dtype=np.int64)
    ])

    return bond_starts


# =============================================================================
# Ex-Ante Winsorization (Fast Path)
# =============================================================================

@njit(cache=True, parallel=True)
def apply_winsorization_fast(
    ret: np.ndarray,
    date_idx: np.ndarray,
    thresholds: np.ndarray,
    loc_code: int
) -> np.ndarray:
    """
    Apply pre-computed thresholds to winsorize returns in parallel.

    Parameters
    ----------
    ret : np.ndarray
        Original returns
    date_idx : np.ndarray
        Date index for each observation
    thresholds : np.ndarray
        Shape (n_dates, 2) with [lb, ub] for each date
    loc_code : int
        0 = both, 1 = right only, 2 = left only

    Returns
    -------
    np.ndarray
        Winsorized returns
    """
    n = len(ret)
    wins_ret = ret.copy()

    for i in prange(n):
        d = date_idx[i]
        lb = thresholds[d, 0]
        ub = thresholds[d, 1]

        if np.isnan(lb) or np.isnan(ub):
            continue

        val = wins_ret[i]
        if np.isnan(val):
            continue

        if loc_code == 0:  # both
            if val > ub:
                wins_ret[i] = ub
            elif val < lb:
                wins_ret[i] = lb
        elif loc_code == 1:  # right
            if val > ub:
                wins_ret[i] = ub
        else:  # left
            if val < lb:
                wins_ret[i] = lb

    return wins_ret


@njit(parallel=True, cache=True)
def _compute_thresholds_parallel(
    ret_sorted: np.ndarray,
    date_sorted: np.ndarray,
    n_dates: int,
    hist_counts: np.ndarray,
    lb_pct: float,
    ub_pct: float
) -> np.ndarray:
    """
    Numba kernel for parallel threshold computation.

    For each date d, finds the lb and ub percentile values among
    returns with date < d, by scanning through the sorted array.

    Uses numpy's 'linear' interpolation method for percentiles to match
    np.nanpercentile exactly.

    Parameters
    ----------
    ret_sorted : np.ndarray
        Returns sorted by VALUE (ascending)
    date_sorted : np.ndarray
        Date indices corresponding to ret_sorted
    n_dates : int
        Number of unique dates
    hist_counts : np.ndarray
        hist_counts[d] = count of values with date < d
    lb_pct : float
        Lower bound percentile fraction (e.g., 0.05 for 5th percentile)
    ub_pct : float
        Upper bound percentile fraction (e.g., 0.95 for 95th percentile)

    Returns
    -------
    np.ndarray
        Shape (n_dates, 2) with [lb, ub] for each date
    """
    n = len(ret_sorted)
    thresholds = np.full((n_dates, 2), np.nan, dtype=np.float64)

    # Parallel over dates - each date processes independently
    for d in prange(1, n_dates):
        hist_count = hist_counts[d]
        if hist_count == 0:
            continue

        # Compute fractional indices using numpy's 'linear' method:
        # index = (n - 1) * percentile / 100
        lb_index = (hist_count - 1) * lb_pct
        ub_index = (hist_count - 1) * ub_pct

        # Integer parts for interpolation
        lb_i = int(np.floor(lb_index))
        lb_j = int(np.ceil(lb_index))
        ub_i = int(np.floor(ub_index))
        ub_j = int(np.ceil(ub_index))

        # Clamp to valid range
        lb_i = max(0, min(lb_i, hist_count - 1))
        lb_j = max(0, min(lb_j, hist_count - 1))
        ub_i = max(0, min(ub_i, hist_count - 1))
        ub_j = max(0, min(ub_j, hist_count - 1))

        # Fractional parts for interpolation
        lb_frac = lb_index - np.floor(lb_index)
        ub_frac = ub_index - np.floor(ub_index)

        # Scan through sorted values, collecting historical values
        # We need values at positions lb_i, lb_j, ub_i, ub_j
        max_pos_needed = max(lb_j, ub_j)

        # Collect values at specific positions
        lb_val_i = np.nan
        lb_val_j = np.nan
        ub_val_i = np.nan
        ub_val_j = np.nan

        count = 0
        for i in range(n):
            if date_sorted[i] < d:
                # This value is part of historical data for date d
                if count == lb_i:
                    lb_val_i = ret_sorted[i]
                if count == lb_j:
                    lb_val_j = ret_sorted[i]
                if count == ub_i:
                    ub_val_i = ret_sorted[i]
                if count == ub_j:
                    ub_val_j = ret_sorted[i]

                count += 1
                if count > max_pos_needed:
                    break

        # Linear interpolation (matching numpy's default)
        if lb_i == lb_j:
            thresholds[d, 0] = lb_val_i
        else:
            thresholds[d, 0] = lb_val_i + (lb_val_j - lb_val_i) * lb_frac

        if ub_i == ub_j:
            thresholds[d, 1] = ub_val_i
        else:
            thresholds[d, 1] = ub_val_i + (ub_val_j - ub_val_i) * ub_frac

    return thresholds


def compute_ex_ante_thresholds_fast(
    ret: np.ndarray,
    date_idx: np.ndarray,
    n_dates: int,
    level: float
) -> np.ndarray:
    """
    Compute ex-ante percentile thresholds for all dates efficiently.

    For each date d, thresholds are computed from all returns with date < d.

    OPTIMIZED: Pre-sorts by VALUE once, then parallelizes over dates using numba prange.
    Complexity: O(n log n) for sort + O(n * n_dates / n_cores) for parallel scan.

    Parameters
    ----------
    ret : np.ndarray
        Original returns
    date_idx : np.ndarray
        Date index for each observation
    n_dates : int
        Number of unique dates
    level : float
        Percentile level (e.g., 99 for 99th percentile)

    Returns
    -------
    np.ndarray
        Shape (n_dates, 2) with [lb, ub] for each date
        lb = percentile(100 - level), ub = percentile(level)
    """
    # Remove NaNs first
    valid_mask = ~np.isnan(ret)
    ret_valid = ret[valid_mask]
    date_valid = date_idx[valid_mask].astype(np.int64)
    n = len(ret_valid)

    if n == 0:
        return np.full((n_dates, 2), np.nan, dtype=np.float64)

    # Sort ALL returns by VALUE (not by date) - O(n log n) once
    value_order = np.argsort(ret_valid)
    ret_sorted = ret_valid[value_order]
    date_sorted = date_valid[value_order]

    # Count returns per date
    date_counts = np.bincount(date_valid, minlength=n_dates)

    # Cumulative historical count for each date: hist_count[d] = count with date < d
    hist_counts = np.zeros(n_dates, dtype=np.int64)
    hist_counts[1:] = np.cumsum(date_counts[:-1])

    # Compute percentile fractions
    lb_pct = (100.0 - level) / 100.0
    ub_pct = level / 100.0

    # Call parallel numba kernel
    thresholds = _compute_thresholds_parallel(
        ret_sorted, date_sorted, n_dates, hist_counts, lb_pct, ub_pct
    )

    return thresholds


# =============================================================================
# Multi-Signal Batch Processing (for BatchStrategyFormation fast path)
# =============================================================================

@njit(parallel=True, cache=True)
def compute_ranks_all_signals(
    date_idx: np.ndarray,          # (n,) date index for each row
    signals: np.ndarray,           # (n, n_signals) signal values
    n_dates: int,
    nport: int,
    n_signals: int
) -> np.ndarray:
    """
    Compute portfolio ranks for ALL signals at once in parallel.

    Each signal is ranked independently within each date.

    Parameters
    ----------
    date_idx : np.ndarray
        Date index (0-indexed) for each observation
    signals : np.ndarray
        Signal values, shape (n_obs, n_signals)
    n_dates : int
        Total number of dates
    nport : int
        Number of portfolios
    n_signals : int
        Number of signals

    Returns
    -------
    np.ndarray
        Shape (n_obs, n_signals) - portfolio rank per observation per signal
    """
    n = len(date_idx)
    ranks = np.full((n, n_signals), np.nan, dtype=np.float64)

    # Process each (date, signal) combination in parallel
    n_combos = n_dates * n_signals

    for combo in prange(n_combos):
        d = combo // n_signals
        s = combo % n_signals

        # Count valid observations for this (date, signal)
        count = 0
        for i in range(n):
            if date_idx[i] == d and not np.isnan(signals[i, s]):
                count += 1

        if count == 0:
            continue

        # Gather valid observations
        valid_indices = np.empty(count, dtype=np.int64)
        valid_signals = np.empty(count, dtype=np.float64)
        pos = 0
        for i in range(n):
            if date_idx[i] == d and not np.isnan(signals[i, s]):
                valid_indices[pos] = i
                valid_signals[pos] = signals[i, s]
                pos += 1

        # Sort by signal
        order = np.argsort(valid_signals)

        # Compute percentile thresholds
        thresholds = np.empty(nport + 1, dtype=np.float64)
        thresholds[0] = -np.inf

        for p in range(1, nport + 1):
            pct = (p * 100.0 / nport)
            pos_f = (pct / 100.0) * (count - 1)
            idx_low = int(pos_f)
            idx_high = idx_low + 1
            frac = pos_f - idx_low

            if idx_high >= count:
                thresholds[p] = valid_signals[order[count - 1]]
            else:
                low_val = valid_signals[order[idx_low]]
                high_val = valid_signals[order[idx_high]]
                thresholds[p] = low_val + frac * (high_val - low_val)

        thresholds[nport] = np.inf

        # Assign ranks using thresholds (matching compute_ranks_all_dates_fast)
        # Logic: val > thresholds[p] and val <= thresholds[p+1]
        for i in range(count):
            orig_idx = valid_indices[i]
            val = signals[orig_idx, s]

            for p in range(nport):
                if val > thresholds[p] and val <= thresholds[p + 1]:
                    ranks[orig_idx, s] = p + 1
                    break

    return ranks


@njit(parallel=True, cache=True)
def build_rank_lookups_all_signals(
    date_idx: np.ndarray,
    id_idx: np.ndarray,
    ranks_all: np.ndarray,
    n_dates: int,
    n_ids: int,
    n_signals: int
) -> np.ndarray:
    """
    Build rank lookup tables for all signals.

    Parameters
    ----------
    date_idx : np.ndarray
        Date index (0-indexed) for each observation
    id_idx : np.ndarray
        Bond ID index for each observation
    ranks_all : np.ndarray
        Ranks from compute_ranks_all_signals, shape (n_obs, n_signals)
    n_dates : int
        Number of dates
    n_ids : int
        Number of unique bond IDs
    n_signals : int
        Number of signals

    Returns
    -------
    np.ndarray
        Shape (n_dates * n_ids, n_signals) - rank lookup table
    """
    n = len(date_idx)
    lookup_size = n_dates * n_ids

    # Initialize with NaN
    rank_lookups = np.full((lookup_size, n_signals), np.nan, dtype=np.float64)

    # Fill lookups in parallel over signals
    for s in prange(n_signals):
        for i in range(n):
            d = date_idx[i]
            bond_id = id_idx[i]
            lookup_idx = d * n_ids + bond_id
            if lookup_idx >= 0 and lookup_idx < lookup_size:
                rank_lookups[lookup_idx, s] = ranks_all[i, s]

    return rank_lookups


@njit(parallel=True, cache=True)
def compute_ls_returns_all_signals_hp1(
    ret_date_idx: np.ndarray,      # (n,) date index for return observations
    ret_id_idx: np.ndarray,        # (n,) bond ID index
    returns: np.ndarray,           # (n,) returns
    weights: np.ndarray,           # (n,) VW weights from d-1
    rank_lookups: np.ndarray,      # (n_dates * n_ids, n_signals) rank lookups
    n_dates: int,
    n_ids: int,
    nport: int,
    n_signals: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute long-short returns for ALL dates and ALL signals (HP=1).

    Returns
    -------
    Tuple of 2 arrays:
        ew_ls: (n_dates, n_signals) - EW long-short per signal
        vw_ls: (n_dates, n_signals) - VW long-short per signal
    """
    n_ret = len(ret_date_idx)

    # Output: long-short returns per date per signal
    ew_ls = np.full((n_dates, n_signals), np.nan, dtype=np.float64)
    vw_ls = np.full((n_dates, n_signals), np.nan, dtype=np.float64)

    # Process in parallel over (date, signal) combinations
    n_combos = n_dates * n_signals

    for combo in prange(n_combos):
        d = combo // n_signals
        s = combo % n_signals

        if d == 0:
            continue

        form_d = d - 1

        # Accumulators for this (date, signal)
        sum_ret = np.zeros(nport, dtype=np.float64)
        sum_wret = np.zeros(nport, dtype=np.float64)
        sum_weight = np.zeros(nport, dtype=np.float64)
        count = np.zeros(nport, dtype=np.int64)

        # Process all return observations for this date
        for i in range(n_ret):
            if ret_date_idx[i] != d:
                continue

            bond_id = ret_id_idx[i]
            weight = weights[i]
            ret_val = returns[i]

            # Skip invalid
            if np.isnan(ret_val) or np.isnan(weight):
                continue

            # Look up rank from formation date for THIS signal
            lookup_idx = form_d * n_ids + bond_id
            if lookup_idx < 0 or lookup_idx >= n_dates * n_ids:
                continue

            rank = rank_lookups[lookup_idx, s]
            if np.isnan(rank):
                continue

            p = int(rank) - 1
            if p < 0 or p >= nport:
                continue

            # Accumulate
            sum_ret[p] += ret_val
            sum_wret[p] += ret_val * weight
            sum_weight[p] += weight
            count[p] += 1

        # Compute portfolio returns
        ew_ret = np.full(nport, np.nan, dtype=np.float64)
        vw_ret = np.full(nport, np.nan, dtype=np.float64)

        for p in range(nport):
            if count[p] > 0:
                ew_ret[p] = sum_ret[p] / count[p]
            if sum_weight[p] > 0:
                vw_ret[p] = sum_wret[p] / sum_weight[p]

        # Long-short = top - bottom
        ew_ls[d, s] = ew_ret[nport - 1] - ew_ret[0]
        vw_ls[d, s] = vw_ret[nport - 1] - vw_ret[0]

    return ew_ls, vw_ls


@njit(parallel=True, cache=True)
def compute_ls_returns_all_signals_staggered(
    ret_date_idx: np.ndarray,
    ret_id_idx: np.ndarray,
    returns: np.ndarray,
    weights: np.ndarray,
    rank_lookups: np.ndarray,
    n_dates: int,
    n_ids: int,
    nport: int,
    n_signals: int,
    holding_period: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute long-short returns for ALL signals with staggered rebalancing (HP > 1).

    Uses cohort averaging: each portfolio's return is averaged across cohorts
    independently, then L-S is computed from the averaged portfolio returns.
    This matches the behavior of compute_staggered_returns_ultrafast.

    Returns
    -------
    Tuple of 2 arrays:
        ew_ls: (n_dates, n_signals) - EW long-short per signal
        vw_ls: (n_dates, n_signals) - VW long-short per signal
    """
    n_ret = len(ret_date_idx)

    # Output: long-short returns per date per signal
    ew_ls = np.full((n_dates, n_signals), np.nan, dtype=np.float64)
    vw_ls = np.full((n_dates, n_signals), np.nan, dtype=np.float64)

    # Process in parallel over (date, signal) combinations
    n_combos = n_dates * n_signals

    for combo in prange(n_combos):
        d = combo // n_signals
        s = combo % n_signals

        if d < holding_period:
            continue

        # Store portfolio returns per cohort for each portfolio
        # Shape: (holding_period, nport)
        cohort_ew = np.full((holding_period, nport), np.nan, dtype=np.float64)
        cohort_vw = np.full((holding_period, nport), np.nan, dtype=np.float64)

        for cohort in range(holding_period):
            form_d = d - 1 - cohort

            if form_d < 0:
                continue

            # Accumulators for this cohort
            sum_ret = np.zeros(nport, dtype=np.float64)
            sum_wret = np.zeros(nport, dtype=np.float64)
            sum_weight = np.zeros(nport, dtype=np.float64)
            count = np.zeros(nport, dtype=np.int64)

            # Process return observations for this date
            for i in range(n_ret):
                if ret_date_idx[i] != d:
                    continue

                bond_id = ret_id_idx[i]
                weight = weights[i]
                ret_val = returns[i]

                if np.isnan(ret_val) or np.isnan(weight):
                    continue

                # Look up rank from formation date
                lookup_idx = form_d * n_ids + bond_id
                if lookup_idx < 0 or lookup_idx >= n_dates * n_ids:
                    continue

                rank = rank_lookups[lookup_idx, s]
                if np.isnan(rank):
                    continue

                p = int(rank) - 1
                if p < 0 or p >= nport:
                    continue

                sum_ret[p] += ret_val
                sum_wret[p] += ret_val * weight
                sum_weight[p] += weight
                count[p] += 1

            # Compute portfolio returns for this cohort (each portfolio independently)
            for p in range(nport):
                if count[p] > 0:
                    cohort_ew[cohort, p] = sum_ret[p] / count[p]
                if sum_weight[p] > 0:
                    cohort_vw[cohort, p] = sum_wret[p] / sum_weight[p]

        # Average each portfolio across cohorts independently
        avg_ew = np.full(nport, np.nan, dtype=np.float64)
        avg_vw = np.full(nport, np.nan, dtype=np.float64)

        for p in range(nport):
            ew_sum = 0.0
            vw_sum = 0.0
            ew_count = 0
            vw_count = 0

            for cohort in range(holding_period):
                if not np.isnan(cohort_ew[cohort, p]):
                    ew_sum += cohort_ew[cohort, p]
                    ew_count += 1
                if not np.isnan(cohort_vw[cohort, p]):
                    vw_sum += cohort_vw[cohort, p]
                    vw_count += 1

            if ew_count > 0:
                avg_ew[p] = ew_sum / ew_count
            if vw_count > 0:
                avg_vw[p] = vw_sum / vw_count

        # Compute L-S from averaged portfolio returns
        if not np.isnan(avg_ew[nport - 1]) and not np.isnan(avg_ew[0]):
            ew_ls[d, s] = avg_ew[nport - 1] - avg_ew[0]
        if not np.isnan(avg_vw[nport - 1]) and not np.isnan(avg_vw[0]):
            vw_ls[d, s] = avg_vw[nport - 1] - avg_vw[0]

    return ew_ls, vw_ls


@njit(parallel=True, cache=True)
def compute_ls_returns_all_signals_staggered_v2(
    ret_date_idx: np.ndarray,
    ret_id_idx: np.ndarray,
    returns: np.ndarray,
    vw_lookup: np.ndarray,
    rank_lookups: np.ndarray,
    n_dates: int,
    n_ids: int,
    nport: int,
    n_signals: int,
    holding_period: int,
    dynamic_weights: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute long-short returns for ALL signals with staggered rebalancing (HP > 1).

    This version supports both dynamic_weights=True and dynamic_weights=False.

    Parameters
    ----------
    vw_lookup : np.ndarray
        VW lookup table with shape (n_dates * n_ids,). Access with:
        vw_lookup[date * n_ids + bond_id]
    dynamic_weights : bool
        If True: use VW from d-1 (return date - 1)
        If False: use VW from formation date for each cohort

    Returns
    -------
    Tuple of 2 arrays:
        ew_ls: (n_dates, n_signals) - EW long-short per signal
        vw_ls: (n_dates, n_signals) - VW long-short per signal
    """
    n_ret = len(ret_date_idx)

    # Output: long-short returns per date per signal
    ew_ls = np.full((n_dates, n_signals), np.nan, dtype=np.float64)
    vw_ls = np.full((n_dates, n_signals), np.nan, dtype=np.float64)

    # Process in parallel over (date, signal) combinations
    n_combos = n_dates * n_signals

    for combo in prange(n_combos):
        d = combo // n_signals
        s = combo % n_signals

        if d < holding_period:
            continue

        # Store portfolio returns per cohort for each portfolio
        # Shape: (holding_period, nport)
        cohort_ew = np.full((holding_period, nport), np.nan, dtype=np.float64)
        cohort_vw = np.full((holding_period, nport), np.nan, dtype=np.float64)

        for cohort in range(holding_period):
            form_d = d - 1 - cohort

            if form_d < 0:
                continue

            # Determine VW date based on dynamic_weights setting
            # True: VW from d-1 (return date - 1)
            # False: VW from form_d (formation date for this cohort)
            if dynamic_weights:
                vw_date = d - 1
            else:
                vw_date = form_d

            # Accumulators for this cohort
            sum_ret = np.zeros(nport, dtype=np.float64)
            sum_wret = np.zeros(nport, dtype=np.float64)
            sum_weight = np.zeros(nport, dtype=np.float64)
            count = np.zeros(nport, dtype=np.int64)

            # Process return observations for this date
            for i in range(n_ret):
                if ret_date_idx[i] != d:
                    continue

                bond_id = ret_id_idx[i]
                ret_val = returns[i]

                if np.isnan(ret_val):
                    continue

                # STEP 1: INTERSECTION CHECK - Always at formation date
                # This matches slow path's intersect_id(It0, It1, It1m) where
                # It1m is ALWAYS at formation date (form_d), not d-1.
                # The dynamic_weights setting only affects VW weighting, not intersection.
                form_vw_lookup_idx = form_d * n_ids + bond_id
                if form_vw_lookup_idx < 0 or form_vw_lookup_idx >= n_dates * n_ids:
                    continue
                form_weight = vw_lookup[form_vw_lookup_idx]
                if np.isnan(form_weight):
                    continue  # Bond doesn't exist at formation date - skip

                # STEP 2: Look up rank from formation date
                rank_lookup_idx = form_d * n_ids + bond_id
                if rank_lookup_idx < 0 or rank_lookup_idx >= n_dates * n_ids:
                    continue

                rank = rank_lookups[rank_lookup_idx, s]
                if np.isnan(rank):
                    continue

                p = int(rank) - 1
                if p < 0 or p >= nport:
                    continue

                # STEP 3: EW - Always include (bond passed intersection check at form_d)
                sum_ret[p] += ret_val
                count[p] += 1

                # STEP 4: VW WEIGHTING - Use appropriate date based on dynamic_weights
                # - True: VW from d-1 (return date - 1) - same for all cohorts
                # - False: VW from form_d (formation date) - different per cohort
                vw_lookup_idx = vw_date * n_ids + bond_id
                if vw_lookup_idx >= 0 and vw_lookup_idx < n_dates * n_ids:
                    weight = vw_lookup[vw_lookup_idx]
                else:
                    weight = np.nan

                # STEP 5: VW - Only include if valid weight at weighting date
                if not np.isnan(weight) and weight > 0:
                    sum_wret[p] += ret_val * weight
                    sum_weight[p] += weight

            # Compute portfolio returns for this cohort (each portfolio independently)
            for p in range(nport):
                if count[p] > 0:
                    cohort_ew[cohort, p] = sum_ret[p] / count[p]
                if sum_weight[p] > 0:
                    cohort_vw[cohort, p] = sum_wret[p] / sum_weight[p]

        # Average each portfolio across cohorts independently
        avg_ew = np.full(nport, np.nan, dtype=np.float64)
        avg_vw = np.full(nport, np.nan, dtype=np.float64)

        for p in range(nport):
            ew_sum = 0.0
            vw_sum = 0.0
            ew_count = 0
            vw_count = 0

            for cohort in range(holding_period):
                if not np.isnan(cohort_ew[cohort, p]):
                    ew_sum += cohort_ew[cohort, p]
                    ew_count += 1
                if not np.isnan(cohort_vw[cohort, p]):
                    vw_sum += cohort_vw[cohort, p]
                    vw_count += 1

            if ew_count > 0:
                avg_ew[p] = ew_sum / ew_count
            if vw_count > 0:
                avg_vw[p] = vw_sum / vw_count

        # Compute L-S from averaged portfolio returns
        if not np.isnan(avg_ew[nport - 1]) and not np.isnan(avg_ew[0]):
            ew_ls[d, s] = avg_ew[nport - 1] - avg_ew[0]
        if not np.isnan(avg_vw[nport - 1]) and not np.isnan(avg_vw[0]):
            vw_ls[d, s] = avg_vw[nport - 1] - avg_vw[0]

    return ew_ls, vw_ls


# =============================================================================
# Non-Staggered Rebalancing Optimization (Phase 15)
# =============================================================================

@njit(cache=True, parallel=True)
def compute_ranks_at_rebal_dates(
    date_idx: np.ndarray,
    signal: np.ndarray,
    rebal_date_indices: np.ndarray,
    n_dates: int,
    nport: int,
) -> np.ndarray:
    """
    Compute portfolio ranks at specific rebalancing dates only.

    This is much faster than computing ranks for all dates when only a subset
    of dates are rebalancing dates (e.g., annual = 10 dates vs monthly = 120).

    Parameters
    ----------
    date_idx : np.ndarray
        Date index for each observation (0 to n_dates-1)
    signal : np.ndarray
        Signal values for ranking
    rebal_date_indices : np.ndarray
        Indices of rebalancing dates (e.g., [5, 17, 29, ...] for June dates)
    n_dates : int
        Total number of unique dates
    nport : int
        Number of portfolios

    Returns
    -------
    np.ndarray
        Portfolio ranks, shape (n_obs,). NaN for observations not at rebal dates.
    """
    n_obs = len(date_idx)
    n_rebal = len(rebal_date_indices)

    # Output: ranks for each observation (NaN for non-rebal dates)
    ranks = np.full(n_obs, np.nan, dtype=np.float64)

    # Create a set-like lookup for rebal dates
    is_rebal_date = np.zeros(n_dates, dtype=np.bool_)
    for i in range(n_rebal):
        is_rebal_date[rebal_date_indices[i]] = True

    # Process each rebalancing date in parallel
    for rebal_i in prange(n_rebal):
        rebal_d = rebal_date_indices[rebal_i]

        # Count valid signals at this date
        count = 0
        for i in range(n_obs):
            if date_idx[i] == rebal_d and not np.isnan(signal[i]):
                count += 1

        if count == 0:
            continue

        # Collect (signal, obs_idx) pairs for this date
        signals_at_date = np.empty(count, dtype=np.float64)
        indices_at_date = np.empty(count, dtype=np.int64)
        k = 0
        for i in range(n_obs):
            if date_idx[i] == rebal_d and not np.isnan(signal[i]):
                signals_at_date[k] = signal[i]
                indices_at_date[k] = i
                k += 1

        # Compute percentile thresholds
        thresholds = np.empty(nport - 1, dtype=np.float64)
        for p in range(nport - 1):
            pct = 100.0 * (p + 1) / nport
            thresholds[p] = np.nanpercentile(signals_at_date, pct)

        # Assign ranks based on thresholds
        for k in range(count):
            sig_val = signals_at_date[k]
            obs_idx = indices_at_date[k]

            # Find rank
            rank = 1
            for p in range(nport - 1):
                if sig_val > thresholds[p]:
                    rank = p + 2

            ranks[obs_idx] = rank

    return ranks


@njit(cache=True)
def build_rank_lookup_nonstaggered(
    date_idx: np.ndarray,
    id_idx: np.ndarray,
    ranks: np.ndarray,
    rebal_date_indices: np.ndarray,
    n_dates: int,
    n_ids: int,
) -> np.ndarray:
    """
    Build a lookup table: rank_lookups[rebal_idx, bond_idx] = portfolio rank.

    For non-staggered rebalancing, we only need ranks at rebalancing dates.

    Parameters
    ----------
    date_idx : np.ndarray
        Date index for each observation
    id_idx : np.ndarray
        Bond ID index for each observation
    ranks : np.ndarray
        Portfolio ranks for each observation (NaN for non-rebal dates)
    rebal_date_indices : np.ndarray
        Indices of rebalancing dates
    n_dates : int
        Total number of dates
    n_ids : int
        Total number of unique bonds

    Returns
    -------
    np.ndarray
        Shape (n_rebal_dates, n_ids), contains portfolio rank or NaN
    """
    n_rebal = len(rebal_date_indices)
    n_obs = len(date_idx)

    # Create mapping from date index to rebal index
    date_to_rebal_idx = np.full(n_dates, -1, dtype=np.int64)
    for i in range(n_rebal):
        date_to_rebal_idx[rebal_date_indices[i]] = i

    # Output: rank lookup table
    rank_lookups = np.full((n_rebal, n_ids), np.nan, dtype=np.float64)

    for i in range(n_obs):
        d = date_idx[i]
        rebal_i = date_to_rebal_idx[d]
        if rebal_i < 0:
            continue  # Not a rebalancing date

        bond = id_idx[i]
        rank = ranks[i]
        if not np.isnan(rank):
            rank_lookups[rebal_i, bond] = rank

    return rank_lookups


@njit(cache=True, parallel=True)
def compute_nonstaggered_returns_fast(
    date_idx: np.ndarray,
    id_idx: np.ndarray,
    ret: np.ndarray,
    vw: np.ndarray,
    rebal_date_indices: np.ndarray,
    holding_period: int,
    rank_lookups: np.ndarray,
    n_dates: int,
    n_ids: int,
    nport: int,
    dynamic_weights: bool,
    vw_lookup: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute portfolio returns for non-staggered rebalancing.

    For each (rebal_date, return_date) pair where return_date is within
    holding_period of rebal_date, compute portfolio returns.

    Parameters
    ----------
    date_idx : np.ndarray
        Date index for each observation
    id_idx : np.ndarray
        Bond ID index for each observation
    ret : np.ndarray
        Returns for each observation
    vw : np.ndarray
        Value weights for each observation
    rebal_date_indices : np.ndarray
        Indices of rebalancing dates
    holding_period : int
        Holding period in months
    rank_lookups : np.ndarray
        Shape (n_rebal, n_ids), portfolio rank for each bond at each rebal date
    n_dates : int
        Total number of dates
    n_ids : int
        Total number of bonds
    nport : int
        Number of portfolios
    dynamic_weights : bool
        If True, use VW from day before return date; if False, use VW from formation
    vw_lookup : np.ndarray
        Shape (n_dates, n_ids), VW for each bond at each date

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (ew_ret, vw_ret) - Shape (n_dates, nport)
    """
    n_obs = len(date_idx)
    n_rebal = len(rebal_date_indices)

    # Output: portfolio returns at each return date
    ew_ret_all = np.full((n_dates, nport), np.nan, dtype=np.float64)
    vw_ret_all = np.full((n_dates, nport), np.nan, dtype=np.float64)

    # Create mapping from date index to rebal index
    date_to_rebal_idx = np.full(n_dates, -1, dtype=np.int64)
    for i in range(n_rebal):
        date_to_rebal_idx[rebal_date_indices[i]] = i

    # Phase 17 fix: For each return date, find which rebal date it belongs to
    # return_date belongs to rebal_date if: rebal_date < return_date <= next_rebal_date
    # We iterate until the next rebalancing date (not just holding_period months)
    return_date_to_rebal = np.full(n_dates, -1, dtype=np.int64)
    for rebal_i in range(n_rebal):
        rebal_d = rebal_date_indices[rebal_i]

        # Find next rebalancing date to determine valid return period
        if rebal_i + 1 < n_rebal:
            next_rebal_d = rebal_date_indices[rebal_i + 1]
        else:
            next_rebal_d = n_dates  # No more rebalancing, collect until end

        # All return dates from rebal_d+1 to next_rebal_d (inclusive) belong to this rebal
        for ret_d in range(rebal_d + 1, next_rebal_d + 1):
            if ret_d < n_dates:
                return_date_to_rebal[ret_d] = rebal_i

    # Process each return date in parallel
    for ret_d in prange(n_dates):
        rebal_i = return_date_to_rebal[ret_d]
        if rebal_i < 0:
            continue  # This date is not a return date for any portfolio

        rebal_d = rebal_date_indices[rebal_i]

        # Accumulators
        sum_ret = np.zeros(nport, dtype=np.float64)
        sum_wret = np.zeros(nport, dtype=np.float64)
        sum_weight = np.zeros(nport, dtype=np.float64)
        count = np.zeros(nport, dtype=np.int64)

        # Loop through all observations at return date
        for i in range(n_obs):
            if date_idx[i] != ret_d:
                continue

            bond = id_idx[i]
            ret_val = ret[i]

            if np.isnan(ret_val):
                continue

            # Get rank from formation date
            rank = rank_lookups[rebal_i, bond]
            if np.isnan(rank):
                continue

            p = int(rank) - 1
            if p < 0 or p >= nport:
                continue

            # Get weight
            if dynamic_weights:
                # VW from day before return date
                vw_date = ret_d - 1
                if vw_date >= 0:
                    weight = vw_lookup[vw_date, bond]
                else:
                    weight = np.nan
            else:
                # VW from formation date
                weight = vw_lookup[rebal_d, bond]

            if np.isnan(weight):
                continue

            sum_ret[p] += ret_val
            count[p] += 1
            sum_wret[p] += ret_val * weight
            sum_weight[p] += weight

        # Compute portfolio returns
        for p in range(nport):
            if count[p] > 0:
                ew_ret_all[ret_d, p] = sum_ret[p] / count[p]
            if sum_weight[p] > 0:
                vw_ret_all[ret_d, p] = sum_wret[p] / sum_weight[p]

    return ew_ret_all, vw_ret_all


@njit(cache=True)
def build_vw_lookup_table(
    date_idx: np.ndarray,
    id_idx: np.ndarray,
    vw: np.ndarray,
    n_dates: int,
    n_ids: int,
) -> np.ndarray:
    """
    Build a lookup table: vw_lookup[date, bond] = value weight.

    Parameters
    ----------
    date_idx : np.ndarray
        Date index for each observation
    id_idx : np.ndarray
        Bond ID index for each observation
    vw : np.ndarray
        Value weights for each observation
    n_dates : int
        Total number of dates
    n_ids : int
        Total number of bonds

    Returns
    -------
    np.ndarray
        Shape (n_dates, n_ids), VW for each bond at each date
    """
    n_obs = len(date_idx)
    vw_lookup = np.full((n_dates, n_ids), np.nan, dtype=np.float64)

    for i in range(n_obs):
        d = date_idx[i]
        bond = id_idx[i]
        w = vw[i]
        if not np.isnan(w):
            vw_lookup[d, bond] = w

    return vw_lookup


@njit(cache=True, parallel=True)
def compute_nonstaggered_ls_returns(
    ew_ret: np.ndarray,
    vw_ret: np.ndarray,
    nport: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute long-short returns from portfolio returns.

    Parameters
    ----------
    ew_ret : np.ndarray
        EW portfolio returns, shape (n_dates, nport)
    vw_ret : np.ndarray
        VW portfolio returns, shape (n_dates, nport)
    nport : int
        Number of portfolios

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (ew_ls, vw_ls) - Long-short returns, shape (n_dates,)
    """
    n_dates = ew_ret.shape[0]

    ew_ls = np.full(n_dates, np.nan, dtype=np.float64)
    vw_ls = np.full(n_dates, np.nan, dtype=np.float64)

    for d in prange(n_dates):
        ew_long = ew_ret[d, nport - 1]
        ew_short = ew_ret[d, 0]
        vw_long = vw_ret[d, nport - 1]
        vw_short = vw_ret[d, 0]

        if not np.isnan(ew_long) and not np.isnan(ew_short):
            ew_ls[d] = ew_long - ew_short
        if not np.isnan(vw_long) and not np.isnan(vw_short):
            vw_ls[d] = vw_long - vw_short

    return ew_ls, vw_ls


# =============================================================================
# Phase 15b: Non-Staggered with Turnover/Chars/Banding
# =============================================================================

@njit(cache=True)
def compute_nonstaggered_weights_at_rebal(
    date_idx: np.ndarray,
    id_idx: np.ndarray,
    ranks: np.ndarray,
    vw: np.ndarray,
    rebal_date: int,
    n_ids: int,
    nport: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute portfolio weights at a single rebalancing date.

    Parameters
    ----------
    date_idx : np.ndarray
        Date index for each observation
    id_idx : np.ndarray
        Bond ID index for each observation
    ranks : np.ndarray
        Portfolio ranks (already computed)
    vw : np.ndarray
        Value weights
    rebal_date : int
        Rebalancing date index
    n_ids : int
        Number of unique bonds
    nport : int
        Number of portfolios

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (ew_weights, vw_weights, bond_ids, bond_ranks)
        - ew_weights: shape (n_bonds_at_date,) - EW for each bond
        - vw_weights: shape (n_bonds_at_date,) - VW for each bond
        - bond_ids: shape (n_bonds_at_date,) - bond indices
        - bond_ranks: shape (n_bonds_at_date,) - portfolio ranks
    """
    n_obs = len(date_idx)

    # First pass: count bonds at this date
    count = 0
    for i in range(n_obs):
        if date_idx[i] == rebal_date and not np.isnan(ranks[i]):
            count += 1

    # Allocate arrays
    bond_ids = np.empty(count, dtype=np.int64)
    bond_ranks = np.empty(count, dtype=np.int64)
    bond_vw = np.empty(count, dtype=np.float64)

    # Second pass: collect data
    idx = 0
    for i in range(n_obs):
        if date_idx[i] == rebal_date and not np.isnan(ranks[i]):
            bond_ids[idx] = id_idx[i]
            bond_ranks[idx] = int(ranks[i])
            bond_vw[idx] = vw[i] if not np.isnan(vw[i]) else 0.0
            idx += 1

    # Compute EW and VW weights for each portfolio
    ew_weights = np.zeros(count, dtype=np.float64)
    vw_weights = np.zeros(count, dtype=np.float64)

    # Count per portfolio and sum VW per portfolio
    ptf_count = np.zeros(nport, dtype=np.int64)
    ptf_vw_sum = np.zeros(nport, dtype=np.float64)

    for i in range(count):
        p = bond_ranks[i] - 1  # 0-indexed
        if p >= 0 and p < nport:
            ptf_count[p] += 1
            ptf_vw_sum[p] += bond_vw[i]

    # Assign weights
    for i in range(count):
        p = bond_ranks[i] - 1
        if p >= 0 and p < nport:
            if ptf_count[p] > 0:
                ew_weights[i] = 1.0 / ptf_count[p]
            if ptf_vw_sum[p] > 0:
                vw_weights[i] = bond_vw[i] / ptf_vw_sum[p]

    return ew_weights, vw_weights, bond_ids, bond_ranks


@njit(cache=True)
def scale_weights_by_returns(
    prev_ew: np.ndarray,
    prev_vw: np.ndarray,
    prev_ids: np.ndarray,
    prev_ranks: np.ndarray,
    ret_lookup: np.ndarray,
    ret_date: int,
    nport: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scale previous weights by returns to get scaled weights.

    scaled_weight[i] = prev_weight[i] * (1 + ret[i]) / sum(prev_weight * (1 + ret))

    Parameters
    ----------
    prev_ew : np.ndarray
        Previous EW weights
    prev_vw : np.ndarray
        Previous VW weights
    prev_ids : np.ndarray
        Bond IDs for previous weights
    prev_ranks : np.ndarray
        Portfolio ranks for previous weights
    ret_lookup : np.ndarray
        Returns lookup table, shape (n_dates, n_ids)
    ret_date : int
        Date to get returns from
    nport : int
        Number of portfolios

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (scaled_ew, scaled_vw) - Scaled weights
    """
    n = len(prev_ew)
    scaled_ew = np.zeros(n, dtype=np.float64)
    scaled_vw = np.zeros(n, dtype=np.float64)

    # Compute sum of weight * (1 + ret) per portfolio
    ew_sum = np.zeros(nport, dtype=np.float64)
    vw_sum = np.zeros(nport, dtype=np.float64)

    # First pass: compute weighted sums
    for i in range(n):
        bond_id = prev_ids[i]
        ret = ret_lookup[ret_date, bond_id]
        if np.isnan(ret):
            ret = 0.0  # Assume 0 return if missing

        p = prev_ranks[i] - 1
        if p >= 0 and p < nport:
            ew_sum[p] += prev_ew[i] * (1.0 + ret)
            vw_sum[p] += prev_vw[i] * (1.0 + ret)

    # Second pass: compute scaled weights
    for i in range(n):
        bond_id = prev_ids[i]
        ret = ret_lookup[ret_date, bond_id]
        if np.isnan(ret):
            ret = 0.0

        p = prev_ranks[i] - 1
        if p >= 0 and p < nport:
            if ew_sum[p] > 0:
                scaled_ew[i] = prev_ew[i] * (1.0 + ret) / ew_sum[p]
            if vw_sum[p] > 0:
                scaled_vw[i] = prev_vw[i] * (1.0 + ret) / vw_sum[p]

    return scaled_ew, scaled_vw


@njit(cache=True)
def compute_turnover_single_rebal(
    curr_ew: np.ndarray,
    curr_vw: np.ndarray,
    curr_ids: np.ndarray,
    curr_ranks: np.ndarray,
    scaled_ew: np.ndarray,
    scaled_vw: np.ndarray,
    scaled_ids: np.ndarray,
    scaled_ranks: np.ndarray,
    nport: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute turnover between current and scaled previous weights.

    Turnover = sum(|curr_weight - prev_scaled_weight|) / 2

    Parameters
    ----------
    curr_ew, curr_vw : np.ndarray
        Current weights
    curr_ids, curr_ranks : np.ndarray
        Current bond IDs and ranks
    scaled_ew, scaled_vw : np.ndarray
        Scaled previous weights
    scaled_ids, scaled_ranks : np.ndarray
        Previous bond IDs and ranks
    nport : int
        Number of portfolios

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (ew_turnover, vw_turnover) - Turnover per portfolio, shape (nport,)
    """
    ew_turnover = np.zeros(nport, dtype=np.float64)
    vw_turnover = np.zeros(nport, dtype=np.float64)

    # Build lookup from bond_id to (scaled_ew, scaled_vw, rank) for previous
    # Use a simple approach: iterate over all prev bonds
    n_prev = len(scaled_ids)
    n_curr = len(curr_ids)

    # For each portfolio, compute turnover
    for p in range(nport):
        ptf = p + 1  # 1-indexed portfolio

        # Sum of current weights in portfolio
        curr_sum_ew = 0.0
        curr_sum_vw = 0.0
        for i in range(n_curr):
            if curr_ranks[i] == ptf:
                curr_sum_ew += curr_ew[i]
                curr_sum_vw += curr_vw[i]

        # Sum of scaled previous weights in portfolio
        prev_sum_ew = 0.0
        prev_sum_vw = 0.0
        for i in range(n_prev):
            if scaled_ranks[i] == ptf:
                prev_sum_ew += scaled_ew[i]
                prev_sum_vw += scaled_vw[i]

        # Sum of min(curr, scaled_prev) for matching bonds
        sum_min_ew = 0.0
        sum_min_vw = 0.0

        # For each current bond in portfolio, find if it was in previous
        for i in range(n_curr):
            if curr_ranks[i] != ptf:
                continue
            bond_id = curr_ids[i]

            # Find this bond in previous
            for j in range(n_prev):
                if scaled_ids[j] == bond_id and scaled_ranks[j] == ptf:
                    sum_min_ew += min(curr_ew[i], scaled_ew[j])
                    sum_min_vw += min(curr_vw[i], scaled_vw[j])
                    break

        # Turnover = prev_sum + curr_sum - 2 * sum_min
        ew_turnover[p] = prev_sum_ew + curr_sum_ew - 2.0 * sum_min_ew
        vw_turnover[p] = prev_sum_vw + curr_sum_vw - 2.0 * sum_min_vw

    return ew_turnover, vw_turnover


@njit(cache=True)
def compute_nonstaggered_chars_single(
    date_idx: np.ndarray,
    id_idx: np.ndarray,
    ranks: np.ndarray,
    vw: np.ndarray,
    char_values: np.ndarray,
    rebal_date: int,
    nport: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute EW and VW characteristics at a single rebalancing date.

    Parameters
    ----------
    date_idx : np.ndarray
        Date index for each observation
    id_idx : np.ndarray
        Bond ID index for each observation
    ranks : np.ndarray
        Portfolio ranks
    vw : np.ndarray
        Value weights
    char_values : np.ndarray
        Characteristic values
    rebal_date : int
        Rebalancing date index
    nport : int
        Number of portfolios

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (ew_char, vw_char) - Characteristics per portfolio, shape (nport,)
    """
    n_obs = len(date_idx)

    ew_sum = np.zeros(nport, dtype=np.float64)
    ew_count = np.zeros(nport, dtype=np.int64)
    vw_sum = np.zeros(nport, dtype=np.float64)
    vw_weight_sum = np.zeros(nport, dtype=np.float64)

    for i in range(n_obs):
        if date_idx[i] != rebal_date:
            continue
        if np.isnan(ranks[i]):
            continue

        p = int(ranks[i]) - 1
        if p < 0 or p >= nport:
            continue

        char_val = char_values[i]
        if np.isnan(char_val):
            continue

        w = vw[i] if not np.isnan(vw[i]) else 0.0

        ew_sum[p] += char_val
        ew_count[p] += 1
        vw_sum[p] += char_val * w
        vw_weight_sum[p] += w

    # Compute means
    ew_char = np.full(nport, np.nan, dtype=np.float64)
    vw_char = np.full(nport, np.nan, dtype=np.float64)

    for p in range(nport):
        if ew_count[p] > 0:
            ew_char[p] = ew_sum[p] / ew_count[p]
        if vw_weight_sum[p] > 0:
            vw_char[p] = vw_sum[p] / vw_weight_sum[p]

    return ew_char, vw_char


@njit(cache=True)
def apply_banding_single_rebal(
    curr_ranks: np.ndarray,
    curr_ids: np.ndarray,
    prev_rank_lookup: np.ndarray,
    nport: int,
    banding_threshold: float,
) -> np.ndarray:
    """
    Apply banding to ranks at a single rebalancing date.

    A bond only moves to a new portfolio if the rank difference exceeds
    banding_threshold * nport.

    Parameters
    ----------
    curr_ranks : np.ndarray
        Current signal-based ranks
    curr_ids : np.ndarray
        Bond IDs
    prev_rank_lookup : np.ndarray
        Previous ranks lookup, shape (n_ids,). NaN if bond not seen before.
    nport : int
        Number of portfolios
    banding_threshold : float
        Banding threshold (e.g., 1 means bond must move by 1 full portfolio)

    Returns
    -------
    np.ndarray
        Adjusted ranks after banding
    """
    n = len(curr_ranks)
    new_ranks = np.empty(n, dtype=np.float64)
    threshold_portfolios = banding_threshold * nport

    for i in range(n):
        bond_id = curr_ids[i]
        curr_rank = curr_ranks[i]

        if np.isnan(curr_rank):
            new_ranks[i] = np.nan
            continue

        prev_rank = prev_rank_lookup[bond_id]

        if np.isnan(prev_rank):
            # New bond - use current rank
            new_ranks[i] = curr_rank
        else:
            # Existing bond - apply banding
            rank_diff = abs(curr_rank - prev_rank)
            if rank_diff < threshold_portfolios:
                # Stay in previous portfolio
                new_ranks[i] = prev_rank
            else:
                # Move to new portfolio
                new_ranks[i] = curr_rank

    return new_ranks


@njit(cache=True)
def build_ret_lookup(
    date_idx: np.ndarray,
    id_idx: np.ndarray,
    ret: np.ndarray,
    n_dates: int,
    n_ids: int,
) -> np.ndarray:
    """
    Build returns lookup table.

    Parameters
    ----------
    date_idx : np.ndarray
        Date index for each observation
    id_idx : np.ndarray
        Bond ID index for each observation
    ret : np.ndarray
        Returns for each observation
    n_dates : int
        Number of dates
    n_ids : int
        Number of bonds

    Returns
    -------
    np.ndarray
        Shape (n_dates, n_ids), returns for each bond at each date
    """
    n_obs = len(date_idx)
    ret_lookup = np.full((n_dates, n_ids), np.nan, dtype=np.float64)

    for i in range(n_obs):
        d = date_idx[i]
        bond = id_idx[i]
        r = ret[i]
        if not np.isnan(r):
            ret_lookup[d, bond] = r

    return ret_lookup


@njit(cache=True)
def compute_nonstaggered_full_fast(
    date_idx: np.ndarray,
    id_idx: np.ndarray,
    signal: np.ndarray,
    ret: np.ndarray,
    vw: np.ndarray,
    char_values: np.ndarray,
    rebal_date_indices: np.ndarray,
    hp: int,
    n_dates: int,
    n_ids: int,
    nport: int,
    n_chars: int,
    compute_turnover: bool,
    compute_chars: bool,
    banding_threshold: float,
    dynamic_weights: bool,
    vw_lookup: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full non-staggered portfolio formation with turnover, chars, and banding.

    This is the main entry point for Phase 15b optimization.

    Turnover is computed at EVERY date (not just rebalancing dates) to match
    the slow path behavior. At each date:
    1. Current raw weights from VW at that date
    2. Previous scaled weights: prev_weight * (1 + bond_ret) / (1 + ptf_ret)
    3. Turnover = prev_sum + curr_sum - 2 * sum_min
    """
    n_obs = len(date_idx)
    n_rebal = len(rebal_date_indices)

    # Output arrays
    ew_ret = np.full((n_dates, nport), np.nan, dtype=np.float64)
    vw_ret = np.full((n_dates, nport), np.nan, dtype=np.float64)
    ew_turnover = np.full((n_dates, nport), np.nan, dtype=np.float64)
    vw_turnover = np.full((n_dates, nport), np.nan, dtype=np.float64)

    if compute_chars and n_chars > 0:
        ew_chars = np.full((n_dates, n_chars, nport), np.nan, dtype=np.float64)
        vw_chars = np.full((n_dates, n_chars, nport), np.nan, dtype=np.float64)
    else:
        ew_chars = np.empty((0, 0, 0), dtype=np.float64)
        vw_chars = np.empty((0, 0, 0), dtype=np.float64)

    # Build returns lookup for scaling weights
    ret_lookup = np.full((n_dates, n_ids), np.nan, dtype=np.float64)
    for i in range(n_obs):
        d = date_idx[i]
        bond = id_idx[i]
        r = ret[i]
        if not np.isnan(r):
            ret_lookup[d, bond] = r

    # Build characteristics lookup for computing chars at return dates
    # Shape: (n_dates, n_ids, n_chars)
    if compute_chars and n_chars > 0:
        char_lookup = np.full((n_dates, n_ids, n_chars), np.nan, dtype=np.float64)
        for i in range(n_obs):
            d = date_idx[i]
            bond = id_idx[i]
            for c in range(n_chars):
                char_lookup[d, bond, c] = char_values[i, c]
    else:
        char_lookup = np.empty((0, 0, 0), dtype=np.float64)

    # Rank lookup: stores ranks assigned at each formation date
    # Shape: (n_dates, n_ids) - only filled at rebalancing dates
    rank_lookup = np.full((n_dates, n_ids), np.nan, dtype=np.float64)

    # Previous rank lookup for banding (persists across rebalancing dates)
    prev_rank_lookup = np.full(n_ids, np.nan, dtype=np.float64)

    # For turnover: track per-bond weights and state
    # These are updated at each date (not just rebalancing)
    prev_ew_weights = np.zeros(n_ids, dtype=np.float64)  # Scaled weights from prev date
    prev_vw_weights = np.zeros(n_ids, dtype=np.float64)
    prev_ranks_for_turnover = np.full(n_ids, np.nan, dtype=np.float64)  # Ranks at prev date
    prev_sum_ew = np.zeros(nport, dtype=np.float64)  # Sum of prev scaled weights per portfolio
    prev_sum_vw = np.zeros(nport, dtype=np.float64)
    prev_seen = np.zeros(nport, dtype=np.bool_)  # Whether each portfolio has been seen

    # Process each rebalancing date to compute ranks
    for r_idx in range(n_rebal):
        rebal_d = rebal_date_indices[r_idx]

        # Collect signals at this date
        n_bonds = 0
        for i in range(n_obs):
            if date_idx[i] == rebal_d and not np.isnan(signal[i]):
                n_bonds += 1

        if n_bonds == 0:
            continue

        # Allocate arrays
        sig_arr = np.empty(n_bonds, dtype=np.float64)
        id_arr = np.empty(n_bonds, dtype=np.int64)
        obs_arr = np.empty(n_bonds, dtype=np.int64)

        idx = 0
        for i in range(n_obs):
            if date_idx[i] == rebal_d and not np.isnan(signal[i]):
                sig_arr[idx] = signal[i]
                id_arr[idx] = id_idx[i]
                obs_arr[idx] = i
                idx += 1

        # Compute percentile thresholds
        sorted_sig = np.sort(sig_arr)
        thresholds = np.empty(nport - 1, dtype=np.float64)
        for p in range(nport - 1):
            pct = (p + 1) * 100.0 / nport
            idx_f = pct * (n_bonds - 1) / 100.0
            idx_lo = int(idx_f)
            idx_hi = min(idx_lo + 1, n_bonds - 1)
            frac = idx_f - idx_lo
            thresholds[p] = sorted_sig[idx_lo] * (1.0 - frac) + sorted_sig[idx_hi] * frac

        # Assign ranks
        for j in range(n_bonds):
            s = sig_arr[j]
            rank = 1
            for p in range(nport - 1):
                if s > thresholds[p]:
                    rank = p + 2
            bond_id = id_arr[j]

            # Apply banding if needed
            if banding_threshold > 0:
                prev_rank = prev_rank_lookup[bond_id]
                if not np.isnan(prev_rank):
                    rank_diff = abs(float(rank) - prev_rank)
                    if rank_diff < banding_threshold * nport:
                        rank = int(prev_rank)
                prev_rank_lookup[bond_id] = float(rank)

            rank_lookup[rebal_d, bond_id] = float(rank)

    # Now process each date for returns, turnover, and characteristics
    for d in range(n_dates):
        # Find which rebalancing date this return date belongs to
        form_d = -1
        for r in range(n_rebal):
            if rebal_date_indices[r] < d:
                form_d = rebal_date_indices[r]
            elif rebal_date_indices[r] == d:
                # This is a rebalancing date - use previous rebal for return calculation
                if r > 0:
                    form_d = rebal_date_indices[r - 1]
                break
            else:
                break

        if form_d < 0:
            continue

        # Phase 17 fix: Find next rebalancing date to determine valid return period
        # Instead of using hp (which is wrong for non-staggered), we collect returns
        # until the next rebalancing date
        next_rebal_d = n_dates  # default: no more rebalancing (collect until end)
        for r in range(n_rebal):
            if rebal_date_indices[r] > form_d:
                next_rebal_d = rebal_date_indices[r]
                break

        # Skip if return date is past next rebalancing (belongs to next period)
        # Note: d == next_rebal_d still belongs to current period (returns collected at rebal date)
        if d > next_rebal_d:
            continue

        # Get bonds at this return date
        n_bonds_d = 0
        for i in range(n_obs):
            if date_idx[i] == d:
                n_bonds_d += 1

        if n_bonds_d == 0:
            continue

        # Collect bond data at this date
        id_arr_d = np.empty(n_bonds_d, dtype=np.int64)
        ret_arr_d = np.empty(n_bonds_d, dtype=np.float64)
        vw_arr_d = np.empty(n_bonds_d, dtype=np.float64)

        idx = 0
        for i in range(n_obs):
            if date_idx[i] == d:
                id_arr_d[idx] = id_idx[i]
                ret_arr_d[idx] = ret[i]
                vw_arr_d[idx] = vw[i]
                idx += 1

        # Get VW from appropriate date for weighting
        vw_date = d - 1 if (dynamic_weights and d > 0) else form_d

        # Compute returns and weights for this date
        ptf_count = np.zeros(nport, dtype=np.int64)
        ptf_vw_sum = np.zeros(nport, dtype=np.float64)
        ew_ret_sum = np.zeros(nport, dtype=np.float64)
        vw_ret_sum = np.zeros(nport, dtype=np.float64)

        # Arrays for current weights (for turnover)
        curr_ew = np.zeros(n_ids, dtype=np.float64)
        curr_vw = np.zeros(n_ids, dtype=np.float64)
        curr_ranks = np.full(n_ids, np.nan, dtype=np.float64)

        # First pass: collect bonds with valid ranks and VW
        for j in range(n_bonds_d):
            bond_id = id_arr_d[j]
            rank = rank_lookup[form_d, bond_id]
            if np.isnan(rank):
                continue

            r = ret_arr_d[j]
            if np.isnan(r):
                continue

            w = vw_lookup[vw_date, bond_id] if vw_date >= 0 and not np.isnan(vw_lookup[vw_date, bond_id]) else np.nan
            if np.isnan(w):
                continue

            p = int(rank) - 1
            ptf_count[p] += 1
            ptf_vw_sum[p] += w
            curr_ranks[bond_id] = rank

        # Second pass: compute returns and weights
        for j in range(n_bonds_d):
            bond_id = id_arr_d[j]
            rank = rank_lookup[form_d, bond_id]
            if np.isnan(rank):
                continue

            r = ret_arr_d[j]
            if np.isnan(r):
                continue

            w = vw_lookup[vw_date, bond_id] if vw_date >= 0 and not np.isnan(vw_lookup[vw_date, bond_id]) else np.nan
            if np.isnan(w):
                continue

            p = int(rank) - 1

            # EW return contribution
            if ptf_count[p] > 0:
                ew_ret_sum[p] += r
                curr_ew[bond_id] = 1.0 / ptf_count[p]

            # VW return contribution
            if ptf_vw_sum[p] > 0:
                vw_ret_sum[p] += r * w
                curr_vw[bond_id] = w / ptf_vw_sum[p]

        # Store returns
        for p in range(nport):
            if ptf_count[p] > 0:
                ew_ret[d, p] = ew_ret_sum[p] / ptf_count[p]
            if ptf_vw_sum[p] > 0:
                vw_ret[d, p] = vw_ret_sum[p] / ptf_vw_sum[p]

        # Compute characteristics at this return date
        # Slow path: It1m = precomp.It1m.get(date_t1_minus1 if date_t1_minus1 else date_t1)
        # where date_t1_minus1 is only set when dynamic_weights=True
        # So: char_date = d-1 if dynamic_weights=True AND d>0, else d (return date)
        if compute_chars and n_chars > 0:
            if dynamic_weights and d > 0:
                char_date = d - 1
            else:
                char_date = d  # Return date (matches slow path when dynamic_weights=False)

            for c_idx in range(n_chars):
                ew_char_sum = np.zeros(nport, dtype=np.float64)
                ew_char_cnt = np.zeros(nport, dtype=np.int64)
                vw_char_sum = np.zeros(nport, dtype=np.float64)

                for j in range(n_bonds_d):
                    bond_id = id_arr_d[j]

                    # Slow path's It1 only contains bonds with valid returns
                    # So we must check return at return date
                    if np.isnan(ret_arr_d[j]):
                        continue

                    rank = rank_lookup[form_d, bond_id]
                    if np.isnan(rank):
                        continue

                    # Get characteristic value from char_date
                    if char_date >= 0:
                        char_val = char_lookup[char_date, bond_id, c_idx]
                    else:
                        char_val = np.nan

                    if np.isnan(char_val):
                        continue

                    # Slow path's It1m only contains bonds with valid VW at char_date
                    # So we must check VW at char_date for characteristics (not vw_date!)
                    vw_at_char_date = vw_lookup[char_date, bond_id] if char_date >= 0 else np.nan
                    if np.isnan(vw_at_char_date):
                        continue  # Bond not in It1m, skip for BOTH EW and VW chars

                    p = int(rank) - 1

                    # EW: include all bonds that would be in It1m_aug
                    ew_char_sum[p] += char_val
                    ew_char_cnt[p] += 1

                    # VW: use weight from vw_date (for return weighting), normalized by ptf_vw_sum
                    w = vw_lookup[vw_date, bond_id] if vw_date >= 0 and not np.isnan(vw_lookup[vw_date, bond_id]) else np.nan
                    if not np.isnan(w) and ptf_vw_sum[p] > 0:
                        # Normalized weight = VW / ptf_vw_sum (sum over ALL bonds in portfolio)
                        vw_char_sum[p] += char_val * (w / ptf_vw_sum[p])

                for p in range(nport):
                    if ew_char_cnt[p] > 0:
                        ew_chars[d, c_idx, p] = ew_char_sum[p] / ew_char_cnt[p]
                    # VW chars: already weighted by normalized weights, no division needed
                    if vw_char_sum[p] != 0.0:
                        vw_chars[d, c_idx, p] = vw_char_sum[p]

        # Compute turnover if enabled
        if compute_turnover:
            # Check if this is the first return date after rebalancing
            # Only compute actual turnover at rebalancing dates (d == form_d + 1)
            # At other dates (holding period), set turnover to 0
            is_rebalancing_date = (d == form_d + 1)

            if is_rebalancing_date:
                # Compute portfolio returns for scaling
                ew_ptf_ret = np.zeros(nport, dtype=np.float64)
                vw_ptf_ret = np.zeros(nport, dtype=np.float64)
                for p in range(nport):
                    if ptf_count[p] > 0:
                        ew_ptf_ret[p] = ew_ret_sum[p] / ptf_count[p]
                    if ptf_vw_sum[p] > 0:
                        vw_ptf_ret[p] = vw_ret_sum[p] / ptf_vw_sum[p]

                # Compute turnover per portfolio
                for p in range(nport):
                    # Current weights sum (needed for both entry and regular turnover)
                    curr_sum_ew = 0.0
                    curr_sum_vw = 0.0
                    for bond_id in range(n_ids):
                        if curr_ranks[bond_id] == p + 1:
                            curr_sum_ew += curr_ew[bond_id]
                            curr_sum_vw += curr_vw[bond_id]

                    if not prev_seen[p]:
                        # First time seeing this portfolio - entry turnover = sum of weights = 1.0
                        prev_seen[p] = True
                        ew_turnover[d, p] = curr_sum_ew
                        vw_turnover[d, p] = curr_sum_vw
                        continue

                    # Sum of min(prev_scaled, curr)
                    sum_min_ew = 0.0
                    sum_min_vw = 0.0
                    for bond_id in range(n_ids):
                        if curr_ranks[bond_id] == p + 1 and prev_ranks_for_turnover[bond_id] == p + 1:
                            sum_min_ew += min(curr_ew[bond_id], prev_ew_weights[bond_id])
                            sum_min_vw += min(curr_vw[bond_id], prev_vw_weights[bond_id])

                    # Turnover = prev_sum + curr_sum - 2 * sum_min
                    turn_ew = prev_sum_ew[p] + curr_sum_ew - 2.0 * sum_min_ew
                    turn_vw = prev_sum_vw[p] + curr_sum_vw - 2.0 * sum_min_vw

                    ew_turnover[d, p] = turn_ew
                    vw_turnover[d, p] = turn_vw
            else:
                # Holding period - no trading, turnover = 0
                # This matches staggered holding cohort behavior
                for p in range(nport):
                    if prev_seen[p]:
                        ew_turnover[d, p] = 0.0
                        vw_turnover[d, p] = 0.0

            # Update previous scaled weights for next date
            # scaled_weight = curr_weight * (1 + bond_ret) / (1 + ptf_ret)
            for bond_id in range(n_ids):
                prev_ew_weights[bond_id] = 0.0
                prev_vw_weights[bond_id] = 0.0
                prev_ranks_for_turnover[bond_id] = np.nan

            for p in range(nport):
                prev_sum_ew[p] = 0.0
                prev_sum_vw[p] = 0.0

            for j in range(n_bonds_d):
                bond_id = id_arr_d[j]
                rank = curr_ranks[bond_id]
                if np.isnan(rank):
                    continue

                p = int(rank) - 1
                r = ret_arr_d[j]
                if np.isnan(r):
                    r = 0.0

                # Scale current weight by (1 + bond_ret) / (1 + ptf_ret)
                ew_scale = (1.0 + r) / (1.0 + ew_ptf_ret[p]) if (1.0 + ew_ptf_ret[p]) != 0 else 1.0
                vw_scale = (1.0 + r) / (1.0 + vw_ptf_ret[p]) if (1.0 + vw_ptf_ret[p]) != 0 else 1.0

                prev_ew_weights[bond_id] = curr_ew[bond_id] * ew_scale
                prev_vw_weights[bond_id] = curr_vw[bond_id] * vw_scale
                prev_ranks_for_turnover[bond_id] = rank

                prev_sum_ew[p] += prev_ew_weights[bond_id]
                prev_sum_vw[p] += prev_vw_weights[bond_id]

    # ===== LIQUIDATION TURNOVER =====
    # At the last date, assume all positions are liquidated
    # This matches the slow path's finalize_turnover behavior
    if compute_turnover:
        tau_last = n_dates - 1
        for p in range(nport):
            # Liquidation turnover = sum of scaled weights (all sold, nothing bought)
            ew_turnover[tau_last, p] = prev_sum_ew[p]
            vw_turnover[tau_last, p] = prev_sum_vw[p]

    return ew_ret, vw_ret, ew_turnover, vw_turnover, ew_chars, vw_chars


# =============================================================================
# WithinFirmSort Aggregation (Phase 16)
# =============================================================================

@njit(cache=True)
def compute_withinfirm_assignments_all_dates(
    signal: np.ndarray,         # Signal values (sorted by date, rating_terc, firm)
    vw: np.ndarray,             # Value weights (sorted same way)
    group_starts: np.ndarray,   # Start index for each (date, rating_terc, firm) group
    group_ends: np.ndarray,     # End index for each group
    min_bonds: int,             # Minimum bonds required per firm
) -> np.ndarray:
    """
    Compute within-firm HIGH/LOW portfolio assignments for all groups at once.

    This is the fast path version that processes all (date, rating_terc, firm)
    groups in a single pass through the data.

    Parameters
    ----------
    signal : np.ndarray
        Signal values, sorted by (date, rating_terc, firm)
    vw : np.ndarray
        Value weights, sorted same way
    group_starts : np.ndarray
        Start indices for each group
    group_ends : np.ndarray
        End indices for each group
    min_bonds : int
        Minimum bonds required per firm to be included

    Returns
    -------
    ptf_rank : np.ndarray
        Portfolio rank for each observation: 0=unassigned, 1=LOW, 2=HIGH
    """
    n_obs = len(signal)
    n_groups = len(group_starts)
    ptf_rank = np.zeros(n_obs, dtype=np.float64)

    for g in range(n_groups):
        start = group_starts[g]
        end = group_ends[g]
        n = end - start

        if n < min_bonds:
            continue

        # Get slice for this group
        s = signal[start:end]
        w = vw[start:end]

        # Check for finite values (signal and weight must be valid)
        finite_count = 0
        for i in range(n):
            if np.isfinite(s[i]) and np.isfinite(w[i]) and w[i] > 0:
                finite_count += 1

        if finite_count < min_bonds:
            continue

        # Collect finite signal values for percentile computation
        s_finite = np.empty(finite_count, dtype=np.float64)
        j = 0
        for i in range(n):
            if np.isfinite(s[i]) and np.isfinite(w[i]) and w[i] > 0:
                s_finite[j] = s[i]
                j += 1

        # Get UNIQUE values (slow path uses np.unique before percentile)
        s_sorted = np.sort(s_finite)
        n_unique = 1
        for i in range(1, len(s_sorted)):
            if s_sorted[i] != s_sorted[i-1]:
                n_unique += 1

        if n_unique < 2:
            continue

        # Extract unique values (matches slow path behavior)
        s_uniq = np.empty(n_unique, dtype=np.float64)
        s_uniq[0] = s_sorted[0]
        k = 1
        for i in range(1, len(s_sorted)):
            if s_sorted[i] != s_sorted[i-1]:
                s_uniq[k] = s_sorted[i]
                k += 1

        # Compute percentile thresholds on UNIQUE values (matches slow path)
        high_bp = np.percentile(s_uniq, 66.66666666666667)
        low_bp = np.percentile(s_uniq, 33.33333333333333)

        # Assign to HIGH/LOW portfolios
        has_high = False
        has_low = False

        for i in range(n):
            idx = start + i
            if np.isfinite(s[i]) and np.isfinite(w[i]) and w[i] > 0:
                if s[i] > high_bp:
                    ptf_rank[idx] = 2.0  # HIGH
                    has_high = True
                elif s[i] < low_bp:
                    ptf_rank[idx] = 1.0  # LOW
                    has_low = True

        # If no bonds in either portfolio, reset assignments for this group
        if not (has_high and has_low):
            for i in range(n):
                ptf_rank[start + i] = 0.0

    return ptf_rank


@njit(cache=True)
def compute_within_firm_aggregation_fast(
    date_idx: np.ndarray,       # Date index for each bond-period
    firm_idx: np.ndarray,       # Firm index for each bond-period
    rating_terc: np.ndarray,    # Rating tercile (1, 2, 3) for each bond-period
    ptf_rank: np.ndarray,       # Portfolio rank (1=LOW, 2=HIGH) for each bond-period
    ret: np.ndarray,            # Return for each bond-period
    vw: np.ndarray,             # Value weight for each bond-period
    n_dates: int,               # Total number of dates
    n_firms: int,               # Total number of unique firms
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast within-firm return aggregation using numba.

    Computes BOTH EW and VW long-short factors:
    - EW: EW returns within firm -> equal-weighted across firms -> avg across ratings
    - VW: VW returns within firm -> cap-weighted across firms -> avg across ratings

    Parameters
    ----------
    date_idx : np.ndarray
        Date index (0 to n_dates-1) for each observation
    firm_idx : np.ndarray
        Firm index (0 to n_firms-1) for each observation
    rating_terc : np.ndarray
        Rating tercile (1, 2, or 3) for each observation
    ptf_rank : np.ndarray
        Portfolio rank (1=LOW, 2=HIGH) for each observation
    ret : np.ndarray
        Return for each observation
    vw : np.ndarray
        Value weight for each observation
    n_dates : int
        Number of unique dates
    n_firms : int
        Number of unique firms

    Returns
    -------
    ew_long_short : np.ndarray
        Shape (n_dates,) - EW within firm -> equal-weight across firms -> avg across ratings
    vw_long_short : np.ndarray
        Shape (n_dates,) - VW within firm -> cap-weight across firms -> avg across ratings
    ew_high_ret : np.ndarray
        Shape (n_dates,) - Simple EW return of HIGH portfolio
    ew_low_ret : np.ndarray
        Shape (n_dates,) - Simple EW return of LOW portfolio
    vw_high_ret : np.ndarray
        Shape (n_dates,) - Simple VW return of HIGH portfolio
    vw_low_ret : np.ndarray
        Shape (n_dates,) - Simple VW return of LOW portfolio
    """
    n_obs = len(date_idx)

    # Accumulators for (date, rating_terc, firm, portfolio)
    # Shape: (n_dates, 3 rating terciles, n_firms, 2 portfolios)
    # portfolio: 0=LOW, 1=HIGH
    ret_sum = np.zeros((n_dates, 3, n_firms, 2), dtype=np.float64)      # sum(ret) for EW
    wret_sum = np.zeros((n_dates, 3, n_firms, 2), dtype=np.float64)     # sum(ret * w) for VW
    vw_sum = np.zeros((n_dates, 3, n_firms, 2), dtype=np.float64)       # sum(w) for VW
    count = np.zeros((n_dates, 3, n_firms, 2), dtype=np.int64)          # count for EW

    # Simple aggregation for reporting (date, portfolio)
    simple_ret_sum = np.zeros((n_dates, 2), dtype=np.float64)
    simple_wret_sum = np.zeros((n_dates, 2), dtype=np.float64)
    simple_vw_sum = np.zeros((n_dates, 2), dtype=np.float64)
    simple_count = np.zeros((n_dates, 2), dtype=np.int64)

    # First pass: accumulate returns and weights
    for i in range(n_obs):
        d = date_idx[i]
        f = firm_idx[i]
        rt = rating_terc[i]
        p = ptf_rank[i]
        r = ret[i]
        w = vw[i]

        # Skip invalid entries
        if d < 0 or f < 0:
            continue
        if np.isnan(r):
            continue
        if np.isnan(rt) or rt < 1 or rt > 3:
            continue
        if np.isnan(p) or p < 1 or p > 2:
            continue

        rt_idx = int(rt) - 1  # 0, 1, or 2
        p_idx = int(p) - 1    # 0=LOW, 1=HIGH

        # EW: just sum returns and count
        ret_sum[d, rt_idx, f, p_idx] += r
        count[d, rt_idx, f, p_idx] += 1

        # VW: weighted sum (only if weight is valid)
        if not np.isnan(w) and w > 0:
            wret_sum[d, rt_idx, f, p_idx] += r * w
            vw_sum[d, rt_idx, f, p_idx] += w

        # Simple aggregation for reporting
        simple_ret_sum[d, p_idx] += r
        simple_count[d, p_idx] += 1
        if not np.isnan(w) and w > 0:
            simple_wret_sum[d, p_idx] += r * w
            simple_vw_sum[d, p_idx] += w

    # Output arrays
    ew_long_short = np.full(n_dates, np.nan, dtype=np.float64)
    vw_long_short = np.full(n_dates, np.nan, dtype=np.float64)
    ew_high_ret = np.full(n_dates, np.nan, dtype=np.float64)
    ew_low_ret = np.full(n_dates, np.nan, dtype=np.float64)
    vw_high_ret = np.full(n_dates, np.nan, dtype=np.float64)
    vw_low_ret = np.full(n_dates, np.nan, dtype=np.float64)

    # Second pass: compute aggregated returns for each date
    for d in range(n_dates):
        # Compute simple returns for reporting (HIGH and LOW portfolios)
        if simple_count[d, 0] > 0:
            ew_low_ret[d] = simple_ret_sum[d, 0] / simple_count[d, 0]
        if simple_count[d, 1] > 0:
            ew_high_ret[d] = simple_ret_sum[d, 1] / simple_count[d, 1]
        if simple_vw_sum[d, 0] > 0:
            vw_low_ret[d] = simple_wret_sum[d, 0] / simple_vw_sum[d, 0]
        if simple_vw_sum[d, 1] > 0:
            vw_high_ret[d] = simple_wret_sum[d, 1] / simple_vw_sum[d, 1]

        # Compute EW and VW H-L factors with rating-averaged aggregation
        ew_rating_factors = np.zeros(3, dtype=np.float64)
        vw_rating_factors = np.zeros(3, dtype=np.float64)
        ew_rating_valid = np.zeros(3, dtype=np.bool_)
        vw_rating_valid = np.zeros(3, dtype=np.bool_)

        for rt_idx in range(3):
            # EW: equal-weight across firms (simple average of firm H-L factors)
            ew_firm_hl_sum = 0.0
            ew_firm_count = 0

            # VW: cap-weight across firms
            vw_firm_hl_sum = 0.0
            vw_firm_weight_sum = 0.0

            for f in range(n_firms):
                # EW within firm
                low_cnt = count[d, rt_idx, f, 0]
                high_cnt = count[d, rt_idx, f, 1]

                if low_cnt > 0 and high_cnt > 0:
                    ew_low_r = ret_sum[d, rt_idx, f, 0] / low_cnt
                    ew_high_r = ret_sum[d, rt_idx, f, 1] / high_cnt
                    ew_firm_hl = ew_high_r - ew_low_r

                    # Equal-weight across firms
                    ew_firm_hl_sum += ew_firm_hl
                    ew_firm_count += 1

                # VW within firm
                low_vw = vw_sum[d, rt_idx, f, 0]
                high_vw = vw_sum[d, rt_idx, f, 1]

                if low_vw > 0 and high_vw > 0:
                    vw_low_r = wret_sum[d, rt_idx, f, 0] / low_vw
                    vw_high_r = wret_sum[d, rt_idx, f, 1] / high_vw
                    vw_firm_hl = vw_high_r - vw_low_r
                    firm_weight = low_vw + high_vw

                    # Cap-weight across firms
                    vw_firm_hl_sum += vw_firm_hl * firm_weight
                    vw_firm_weight_sum += firm_weight

            # Store rating-level factors
            if ew_firm_count > 0:
                ew_rating_factors[rt_idx] = ew_firm_hl_sum / ew_firm_count
                ew_rating_valid[rt_idx] = True
            if vw_firm_weight_sum > 0:
                vw_rating_factors[rt_idx] = vw_firm_hl_sum / vw_firm_weight_sum
                vw_rating_valid[rt_idx] = True

        # Average across valid rating terciles
        ew_n_valid = 0
        ew_rating_sum = 0.0
        vw_n_valid = 0
        vw_rating_sum = 0.0

        for rt_idx in range(3):
            if ew_rating_valid[rt_idx]:
                ew_rating_sum += ew_rating_factors[rt_idx]
                ew_n_valid += 1
            if vw_rating_valid[rt_idx]:
                vw_rating_sum += vw_rating_factors[rt_idx]
                vw_n_valid += 1

        if ew_n_valid > 0:
            ew_long_short[d] = ew_rating_sum / ew_n_valid
        if vw_n_valid > 0:
            vw_long_short[d] = vw_rating_sum / vw_n_valid

    return ew_long_short, vw_long_short, ew_high_ret, ew_low_ret, vw_high_ret, vw_low_ret


@njit(cache=True)
def compute_within_firm_aggregation_with_lookup(
    date_idx: np.ndarray,       # Return date index for each observation
    id_idx: np.ndarray,         # Bond ID index for each observation
    firm_idx: np.ndarray,       # Firm index for each observation
    ret: np.ndarray,            # Return for each observation
    vw: np.ndarray,             # Value weight for each observation
    rank_lookup: np.ndarray,    # (n_dates, n_ids, 3): [rank, rating_terc, firm_idx] at formation
    vw_lookup: np.ndarray,      # (n_dates, n_ids): VW at each date for dynamic weights
    n_dates: int,
    n_ids: int,
    n_firms: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Within-firm return aggregation using lookup tables for formation/return date separation.

    For HP=1:
    - Return date = t+1
    - Formation date = t = return_date - 1
    - Look up rank, rating_terc, firm_idx from formation date

    Aggregation hierarchy:
    - EW: EW returns within firm -> equal-weight across firms -> avg across ratings
    - VW: VW returns within firm -> cap-weight across firms -> avg across ratings

    Parameters
    ----------
    date_idx : np.ndarray
        Return date index (0 to n_dates-1) for each observation
    id_idx : np.ndarray
        Bond ID index (0 to n_ids-1) for each observation
    firm_idx : np.ndarray
        Firm index for each observation (used as fallback)
    ret : np.ndarray
        Return for each observation (at return date)
    vw : np.ndarray
        Value weight for each observation
    rank_lookup : np.ndarray
        Shape (n_dates, n_ids, 3): [rank, rating_terc, firm_idx] at each formation date
    vw_lookup : np.ndarray
        Shape (n_dates, n_ids): VW at each date (for dynamic weights)
    n_dates : int
        Number of unique dates
    n_ids : int
        Number of unique bond IDs
    n_firms : int
        Number of unique firms

    Returns
    -------
    ew_long_short, vw_long_short : np.ndarray
        EW and VW long-short factors at each return date
    ew_high_ret, ew_low_ret : np.ndarray
        Simple EW portfolio returns (HIGH and LOW)
    vw_high_ret, vw_low_ret : np.ndarray
        Simple VW portfolio returns (HIGH and LOW)
    """
    n_obs = len(date_idx)

    # Accumulation arrays: (date, rating_terc, firm, portfolio)
    # portfolio: 0=LOW, 1=HIGH
    ret_sum = np.zeros((n_dates, 3, n_firms, 2), dtype=np.float64)
    count = np.zeros((n_dates, 3, n_firms, 2), dtype=np.float64)
    wret_sum = np.zeros((n_dates, 3, n_firms, 2), dtype=np.float64)
    vw_sum = np.zeros((n_dates, 3, n_firms, 2), dtype=np.float64)

    # Simple aggregation for reporting (across all firms/ratings)
    simple_ret_sum = np.zeros((n_dates, 2), dtype=np.float64)
    simple_count = np.zeros((n_dates, 2), dtype=np.float64)
    simple_wret_sum = np.zeros((n_dates, 2), dtype=np.float64)
    simple_vw_sum = np.zeros((n_dates, 2), dtype=np.float64)

    # First pass: accumulate returns using formation date lookup
    for i in range(n_obs):
        return_d = date_idx[i]  # Return date
        form_d = return_d - 1   # Formation date = return_date - 1
        bond_id = id_idx[i]
        r = ret[i]

        # Skip if no formation date (first date has no prior formation)
        if form_d < 0:
            continue

        # Skip if return is NaN
        if np.isnan(r):
            continue

        # Look up rank from formation date
        rank = rank_lookup[form_d, bond_id, 0]
        rating_terc_f = rank_lookup[form_d, bond_id, 1]
        firm_idx_f = rank_lookup[form_d, bond_id, 2]

        # Skip if not assigned to a portfolio
        if rank < 1 or rank > 2:
            continue

        # Skip if invalid rating tercile or firm
        if np.isnan(rating_terc_f) or rating_terc_f < 1 or rating_terc_f > 3:
            continue
        if np.isnan(firm_idx_f) or firm_idx_f < 0:
            continue

        rt_idx = int(rating_terc_f) - 1  # 0, 1, or 2
        f = int(firm_idx_f)
        p_idx = int(rank) - 1  # 0=LOW, 1=HIGH

        # Get VW from formation date (d-1) for dynamic weights
        # This matches slow path: vw_map_t1m[date_t1_minus1] where date_t1_minus1 = form_d
        w = vw_lookup[form_d, bond_id]

        # Skip if no valid VW at formation date (matching intersection logic)
        # The slow path's intersect_id() filters out bonds without valid VW
        if np.isnan(w) or w <= 0:
            continue

        # EW: just sum returns and count
        ret_sum[return_d, rt_idx, f, p_idx] += r
        count[return_d, rt_idx, f, p_idx] += 1

        # VW: weighted sum (weight already validated above)
        wret_sum[return_d, rt_idx, f, p_idx] += r * w
        vw_sum[return_d, rt_idx, f, p_idx] += w

        # Simple aggregation for reporting (weight already validated)
        simple_ret_sum[return_d, p_idx] += r
        simple_count[return_d, p_idx] += 1
        simple_wret_sum[return_d, p_idx] += r * w
        simple_vw_sum[return_d, p_idx] += w

    # Output arrays (indexed by return date)
    ew_long_short = np.full(n_dates, np.nan, dtype=np.float64)
    vw_long_short = np.full(n_dates, np.nan, dtype=np.float64)
    ew_high_ret = np.full(n_dates, np.nan, dtype=np.float64)
    ew_low_ret = np.full(n_dates, np.nan, dtype=np.float64)
    vw_high_ret = np.full(n_dates, np.nan, dtype=np.float64)
    vw_low_ret = np.full(n_dates, np.nan, dtype=np.float64)

    # Second pass: compute aggregated returns for each return date
    for d in range(n_dates):
        # Compute simple returns for reporting (HIGH and LOW portfolios)
        if simple_count[d, 0] > 0:
            ew_low_ret[d] = simple_ret_sum[d, 0] / simple_count[d, 0]
        if simple_count[d, 1] > 0:
            ew_high_ret[d] = simple_ret_sum[d, 1] / simple_count[d, 1]
        if simple_vw_sum[d, 0] > 0:
            vw_low_ret[d] = simple_wret_sum[d, 0] / simple_vw_sum[d, 0]
        if simple_vw_sum[d, 1] > 0:
            vw_high_ret[d] = simple_wret_sum[d, 1] / simple_vw_sum[d, 1]

        # Compute EW and VW H-L factors with rating-averaged aggregation
        ew_rating_factors = np.zeros(3, dtype=np.float64)
        vw_rating_factors = np.zeros(3, dtype=np.float64)
        ew_rating_valid = np.zeros(3, dtype=np.bool_)
        vw_rating_valid = np.zeros(3, dtype=np.bool_)

        for rt_idx in range(3):
            # EW: equal-weight across firms (simple average of firm H-L factors)
            ew_firm_hl_sum = 0.0
            ew_firm_count = 0

            # VW: cap-weight across firms
            vw_firm_hl_sum = 0.0
            vw_firm_weight_sum = 0.0

            for f in range(n_firms):
                # EW within firm
                low_cnt = count[d, rt_idx, f, 0]
                high_cnt = count[d, rt_idx, f, 1]

                if low_cnt > 0 and high_cnt > 0:
                    ew_low_r = ret_sum[d, rt_idx, f, 0] / low_cnt
                    ew_high_r = ret_sum[d, rt_idx, f, 1] / high_cnt
                    ew_firm_hl = ew_high_r - ew_low_r

                    # Equal-weight across firms
                    ew_firm_hl_sum += ew_firm_hl
                    ew_firm_count += 1

                # VW within firm
                low_vw = vw_sum[d, rt_idx, f, 0]
                high_vw = vw_sum[d, rt_idx, f, 1]

                if low_vw > 0 and high_vw > 0:
                    vw_low_r = wret_sum[d, rt_idx, f, 0] / low_vw
                    vw_high_r = wret_sum[d, rt_idx, f, 1] / high_vw
                    vw_firm_hl = vw_high_r - vw_low_r
                    firm_weight = low_vw + high_vw

                    # Cap-weight across firms
                    vw_firm_hl_sum += vw_firm_hl * firm_weight
                    vw_firm_weight_sum += firm_weight

            # Store rating-level factors
            if ew_firm_count > 0:
                ew_rating_factors[rt_idx] = ew_firm_hl_sum / ew_firm_count
                ew_rating_valid[rt_idx] = True
            if vw_firm_weight_sum > 0:
                vw_rating_factors[rt_idx] = vw_firm_hl_sum / vw_firm_weight_sum
                vw_rating_valid[rt_idx] = True

        # Average across valid rating terciles
        ew_n_valid = 0
        ew_rating_sum = 0.0
        vw_n_valid = 0
        vw_rating_sum = 0.0

        for rt_idx in range(3):
            if ew_rating_valid[rt_idx]:
                ew_rating_sum += ew_rating_factors[rt_idx]
                ew_n_valid += 1
            if vw_rating_valid[rt_idx]:
                vw_rating_sum += vw_rating_factors[rt_idx]
                vw_n_valid += 1

        if ew_n_valid > 0:
            ew_long_short[d] = ew_rating_sum / ew_n_valid
        if vw_n_valid > 0:
            vw_long_short[d] = vw_rating_sum / vw_n_valid

    return ew_long_short, vw_long_short, ew_high_ret, ew_low_ret, vw_high_ret, vw_low_ret


@njit(cache=True)
def compute_within_firm_chars_aggregation(
    date_idx: np.ndarray,       # Formation date index for each observation
    id_idx: np.ndarray,         # Bond ID index for each observation
    firm_idx: np.ndarray,       # Firm index for each observation
    rating_terc: np.ndarray,    # Rating tercile (1, 2, 3) for each observation
    ptf_rank: np.ndarray,       # Portfolio rank (1=LOW, 2=HIGH) for each observation
    char_values: np.ndarray,    # Characteristic values for each observation
    vw: np.ndarray,             # Value weights for each observation
    n_dates: int,
    n_firms: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Within-firm characteristics aggregation using hierarchical structure.

    Aggregation hierarchy (same as returns):
    - Within-firm: VW-weighted char for HIGH and LOW portfolios
    - Across firms: Cap-weight the firm-level chars within each rating tercile
    - Across ratings: Simple average across rating terciles

    Parameters
    ----------
    date_idx : np.ndarray
        Formation date index for each observation
    id_idx : np.ndarray
        Bond ID index for each observation
    firm_idx : np.ndarray
        Firm index for each observation
    rating_terc : np.ndarray
        Rating tercile (1, 2, 3) for each observation
    ptf_rank : np.ndarray
        Portfolio rank (1=LOW, 2=HIGH) for each observation
    char_values : np.ndarray
        Characteristic values for each observation
    vw : np.ndarray
        Value weights for each observation
    n_dates : int
        Number of unique dates
    n_firms : int
        Number of unique firms

    Returns
    -------
    ew_low, ew_high : np.ndarray
        EW characteristic values for LOW and HIGH portfolios at each date
    vw_low, vw_high : np.ndarray
        VW characteristic values for LOW and HIGH portfolios at each date
    """
    n_obs = len(date_idx)

    # Accumulation arrays: (date, rating_terc, firm, portfolio)
    # portfolio: 0=LOW, 1=HIGH
    char_wsum = np.zeros((n_dates, 3, n_firms, 2), dtype=np.float64)
    vw_sum = np.zeros((n_dates, 3, n_firms, 2), dtype=np.float64)
    char_sum = np.zeros((n_dates, 3, n_firms, 2), dtype=np.float64)
    count = np.zeros((n_dates, 3, n_firms, 2), dtype=np.float64)

    # First pass: accumulate char values at formation date
    for i in range(n_obs):
        d = date_idx[i]
        f = firm_idx[i]
        rt = rating_terc[i]
        p = ptf_rank[i]
        c = char_values[i]
        w = vw[i]

        if d < 0 or f < 0:
            continue
        if np.isnan(rt) or rt < 1 or rt > 3:
            continue
        if np.isnan(p) or p < 1 or p > 2:
            continue
        if np.isnan(c):
            continue

        rt_idx = int(rt) - 1
        p_idx = int(p) - 1

        char_sum[d, rt_idx, f, p_idx] += c
        count[d, rt_idx, f, p_idx] += 1

        if not np.isnan(w) and w > 0:
            char_wsum[d, rt_idx, f, p_idx] += c * w
            vw_sum[d, rt_idx, f, p_idx] += w

    # Output arrays
    ew_low = np.full(n_dates, np.nan, dtype=np.float64)
    ew_high = np.full(n_dates, np.nan, dtype=np.float64)
    vw_low = np.full(n_dates, np.nan, dtype=np.float64)
    vw_high = np.full(n_dates, np.nan, dtype=np.float64)

    # Second pass: compute aggregated chars for each date
    for d in range(n_dates):
        for p_idx in range(2):
            ew_rating_chars = np.zeros(3, dtype=np.float64)
            vw_rating_chars = np.zeros(3, dtype=np.float64)
            ew_rating_valid = np.zeros(3, dtype=np.bool_)
            vw_rating_valid = np.zeros(3, dtype=np.bool_)

            for rt_idx in range(3):
                ew_firm_char_sum = 0.0
                ew_firm_count = 0
                vw_firm_char_sum = 0.0
                vw_firm_weight_sum = 0.0

                for f in range(n_firms):
                    cnt = count[d, rt_idx, f, p_idx]
                    if cnt > 0:
                        ew_firm_char = char_sum[d, rt_idx, f, p_idx] / cnt
                        ew_firm_char_sum += ew_firm_char
                        ew_firm_count += 1

                    vw_w = vw_sum[d, rt_idx, f, p_idx]
                    if vw_w > 0:
                        vw_firm_char = char_wsum[d, rt_idx, f, p_idx] / vw_w
                        vw_firm_char_sum += vw_firm_char * vw_w
                        vw_firm_weight_sum += vw_w

                if ew_firm_count > 0:
                    ew_rating_chars[rt_idx] = ew_firm_char_sum / ew_firm_count
                    ew_rating_valid[rt_idx] = True
                if vw_firm_weight_sum > 0:
                    vw_rating_chars[rt_idx] = vw_firm_char_sum / vw_firm_weight_sum
                    vw_rating_valid[rt_idx] = True

            ew_n_valid = 0
            ew_rating_sum = 0.0
            vw_n_valid = 0
            vw_rating_sum = 0.0

            for rt_idx in range(3):
                if ew_rating_valid[rt_idx]:
                    ew_rating_sum += ew_rating_chars[rt_idx]
                    ew_n_valid += 1
                if vw_rating_valid[rt_idx]:
                    vw_rating_sum += vw_rating_chars[rt_idx]
                    vw_n_valid += 1

            if p_idx == 0:  # LOW
                if ew_n_valid > 0:
                    ew_low[d] = ew_rating_sum / ew_n_valid
                if vw_n_valid > 0:
                    vw_low[d] = vw_rating_sum / vw_n_valid
            else:  # HIGH
                if ew_n_valid > 0:
                    ew_high[d] = ew_rating_sum / ew_n_valid
                if vw_n_valid > 0:
                    vw_high[d] = vw_rating_sum / vw_n_valid

    return ew_low, ew_high, vw_low, vw_high
