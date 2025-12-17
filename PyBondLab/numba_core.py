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

                # Look up VW based on dynamic_weights setting:
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

                # Look up rank from formation date
                lookup_idx = form_d * n_ids + bond_id
                if lookup_idx < 0 or lookup_idx >= len(rank_lookup):
                    continue

                rank = rank_lookup[lookup_idx]
                if np.isnan(rank) or np.isnan(ret_val):
                    continue

                # Skip bonds that don't exist at VW date
                # This matches slow path's 3-way intersection logic
                if np.isnan(weight):
                    continue

                p = int(rank) - 1
                if p < 0 or p >= nport:
                    continue

                sum_ret[p] += ret_val
                count[p] += 1

                if weight > 0:
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
