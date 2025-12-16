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

@njit(cache=True, fastmath=True)
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


@njit(cache=True, fastmath=True)
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


@njit(cache=True, fastmath=True)
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


@njit(cache=True, fastmath=True)
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
            new_prev_seen_ew[p] = True

        if prev_seen_vw[p]:
            turn_vw[p] = prev_sum_vw[p] + curr_sum_vw[p] - 2.0 * sum_min_vw[p]
        else:
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
