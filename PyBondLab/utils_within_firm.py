# -*- coding: utf-8 -*-
"""
Created: 2025-11-11
@authors: Giulio Rossetti & Alex Dickerson

Within-firm portfolio formation utilities for PyBondLab.

The key innovation is that portfolio formation happens within each firm, isolating
within-firm bond dispersion from cross-firm differences.
"""

import numpy as np
import pandas as pd
from numba import njit
from typing import Tuple

from .constants import ColumnNames


@njit
def compute_within_firm_assignments_numba(
    sig_vals, w_vals, group_starts, group_ends, min_bonds=2
):
    """
    Numba-compiled core: compute within-firm portfolio assignments.

    For each firm-date-rating group:
    1. Sort bonds by signal into high (>66.7th pctile) and low (<33.3rd pctile)
    2. Assign bonds to high (Q2) or low (Q1) portfolios
    3. Track weights for aggregation across firms

    Parameters
    ----------
    sig_vals : np.ndarray
        Signal values (e.g., yield)
    w_vals : np.ndarray
        Weight values (e.g., market value)
    group_starts : np.ndarray
        Start index for each group
    group_ends : np.ndarray
        End index for each group
    min_bonds : int
        Minimum bonds required per firm

    Returns
    -------
    portfolio_assignments : np.ndarray
        Shape (n_bonds,) with portfolio ranks: 0=unassigned, 1=low, 2=high
    firm_weights : np.ndarray
        Shape (n_groups,) with total market value for each firm-date-rating group
    """
    n_groups = len(group_starts)
    portfolio_assignments = np.zeros(len(sig_vals), dtype=np.int32)
    firm_weights = np.zeros(n_groups, dtype=np.float64)

    for g in range(n_groups):
        start, end = group_starts[g], group_ends[g]
        n = end - start

        if n < min_bonds:
            continue

        s = sig_vals[start:end]
        w = w_vals[start:end]

        # Check for finite values (signal and weight must be valid)
        finite_mask = np.isfinite(s) & np.isfinite(w) & (w > 0)
        if np.sum(finite_mask) < min_bonds:
            continue

        s_finite = s[finite_mask]
        uniq = np.unique(s_finite)

        if len(uniq) < 2:
            continue

        # Compute percentile thresholds
        high_bp = np.percentile(uniq, 66.66666666666667)
        low_bp = np.percentile(uniq, 33.33333333333333)

        # Assign to high/low portfolios
        high_mask = finite_mask & (s > high_bp)
        low_mask = finite_mask & (s < low_bp)

        if not (np.any(high_mask) and np.any(low_mask)):
            continue

        # Store total firm weight (sum of high + low weights)
        firm_weights[g] = np.sum(w[high_mask]) + np.sum(w[low_mask])

        # Assign portfolio ranks
        for i in range(n):
            idx = start + i
            if high_mask[i]:
                portfolio_assignments[idx] = 2  # High = Q2
            elif low_mask[i]:
                portfolio_assignments[idx] = 1  # Low = Q1
            # else: remains 0 (unassigned)

    return portfolio_assignments, firm_weights


def compute_within_firm_portfolios(
    data: pd.DataFrame,
    signal_col: str,
    return_col: str,  # Not used, kept for API compatibility
    weight_col: str,
    firm_id_col: str,
    rating_col: str,
    rating_bins: list,
    min_bonds_per_firm: int = 2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute within-firm portfolio assignments.

    This function implements the core within-firm sorting logic:
    1. Create rating terciles
    2. Group by date, rating tercile, and firm
    3. Within each group, assign bonds to high/low portfolios based on signal

    NOTE: This function only assigns bonds to portfolios. It does NOT compute
    returns. Return computation is handled by PyBondLab's standard machinery.

    Parameters
    ----------
    data : pd.DataFrame
        Bond panel data with all required columns
    signal_col : str
        Signal column name (e.g., 'eff_yld')
    return_col : str
        Return column name (unused, for API compatibility)
    weight_col : str
        Weight column name (e.g., 'VW')
    firm_id_col : str
        Firm identifier column name (e.g., 'PERMNO')
    rating_col : str
        Rating column name (e.g., 'RATING_NUM')
    rating_bins : list
        Rating bin edges (e.g., [-inf, 7, 10, inf])
    min_bonds_per_firm : int, default 2
        Minimum bonds required per firm-date-rating group

    Returns
    -------
    bond_assignments : pd.DataFrame
        DataFrame with columns: ID, date, ptf_rank (1=low, 2=high)
    firm_weights : pd.DataFrame
        DataFrame with firm-level weights by date and rating tercile

    Notes
    -----
    - Only bonds assigned to portfolios (high or low) are included in bond_assignments
    - Bonds in the middle tercile are excluded from portfolios
    - Portfolio ranks: 1 = Low, 2 = High
    """
    # Create rating terciles
    rating_terc = pd.cut(
        pd.to_numeric(data[rating_col], errors='coerce'),
        bins=rating_bins,
        labels=[1, 2, 3],
        include_lowest=True
    ).astype('Int64')

    # Filter valid observations
    valid_mask = (
        rating_terc.notna() &
        data[signal_col].notna() &
        data[weight_col].notna() &
        data[firm_id_col].notna()
    )

    # Create subset for processing
    sub = data.loc[valid_mask].copy()
    sub['rating_terc'] = rating_terc[valid_mask].values

    # Clip weights to avoid division by zero
    sub[weight_col] = sub[weight_col].clip(lower=1e-10)

    # Create grouping variable: date + rating_terc + firm_id
    grp_cols = [ColumnNames.DATE, 'rating_terc', firm_id_col]
    sub = sub.sort_values(grp_cols).reset_index(drop=True)
    sub['_grp'] = sub.groupby(grp_cols, sort=False).ngroup()

    # Find group boundaries
    group_changes = np.concatenate([
        [0],
        np.where(np.diff(sub['_grp']) != 0)[0] + 1,
        [len(sub)]
    ])

    group_starts = group_changes[:-1].astype(np.int64)
    group_ends = group_changes[1:].astype(np.int64)
    n_groups = len(group_starts)

    # Convert to numpy arrays for numba
    sig_vals = sub[signal_col].to_numpy(dtype=np.float64)
    w_vals = sub[weight_col].to_numpy(dtype=np.float64)

    # Get group metadata
    group_meta = sub.groupby('_grp').agg({
        ColumnNames.DATE: 'first',
        'rating_terc': 'first',
        firm_id_col: 'first'
    }).reset_index()

    # Run numba-compiled core computation
    portfolio_ranks, firm_weights_arr = compute_within_firm_assignments_numba(
        sig_vals, w_vals, group_starts, group_ends, min_bonds_per_firm
    )

    # Create bond-level portfolio assignments
    # Include all bonds (even those not assigned)
    bond_assignments = pd.DataFrame({
        ColumnNames.ID: sub[ColumnNames.ID].values,
        ColumnNames.DATE: sub[ColumnNames.DATE].values,
        ColumnNames.PORTFOLIO_RANK: portfolio_ranks.astype(int),
    })

    # Filter to only assigned bonds (rank > 0)
    bond_assignments = bond_assignments[bond_assignments[ColumnNames.PORTFOLIO_RANK] > 0]

    # Create firm weights dataframe (for debugging/analysis)
    firm_weights_df = pd.DataFrame({
        '_grp': np.arange(n_groups),
        'firm_weight': firm_weights_arr
    })
    firm_weights_df = firm_weights_df.merge(group_meta, on='_grp')
    firm_weights_df = firm_weights_df[firm_weights_df['firm_weight'] > 0]  # Filter out empty firms

    return bond_assignments, firm_weights_df


def compute_within_firm_returns_aggregation_fast(
    portfolio_indices: dict,
    data_raw: pd.DataFrame,
    datelist: list,
    firm_id_col: str,
    rating_col: str,
    rating_bins: list
) -> dict:
    """
    Fast numba-based within-firm return aggregation.

    This is a drop-in replacement for compute_within_firm_returns_aggregation
    that uses numba kernels for ~350x speedup.
    """
    from .numba_core import compute_within_firm_aggregation_fast

    # Collect all portfolio data into a single DataFrame
    # Note: portfolio_indices is dict {date -> DataFrame}, DataFrame has no date column
    all_port_dfs = []
    for date_t in datelist:
        if date_t not in portfolio_indices:
            continue
        port_df = portfolio_indices[date_t]
        if port_df.empty:
            continue
        # Add date column to each DataFrame
        port_df_copy = port_df.copy()
        port_df_copy[ColumnNames.DATE] = date_t
        all_port_dfs.append(port_df_copy)

    if not all_port_dfs:
        return {
            'long_short': pd.Series(dtype=float),
            'long_leg': pd.Series(dtype=float),
            'short_leg': pd.Series(dtype=float),
        }

    # Concatenate all portfolio data
    combined_df = pd.concat(all_port_dfs, ignore_index=True)

    # Get firm and rating lookup
    firm_rating_lookup = data_raw[[ColumnNames.ID, firm_id_col, rating_col]].drop_duplicates()

    # Merge with firm and rating info
    combined_df = combined_df.merge(firm_rating_lookup, on=ColumnNames.ID, how='left')

    # Create rating terciles
    combined_df['rating_terc'] = pd.cut(
        pd.to_numeric(combined_df[rating_col], errors='coerce'),
        bins=rating_bins,
        labels=[1, 2, 3],
        include_lowest=True
    ).astype('Int64')

    # Create index mappings
    unique_dates = sorted(combined_df[ColumnNames.DATE].unique())
    unique_firms = sorted(combined_df[firm_id_col].dropna().unique())

    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    firm_to_idx = {f: i for i, f in enumerate(unique_firms)}

    n_dates = len(unique_dates)
    n_firms = len(unique_firms)

    # Convert to numpy arrays
    date_idx = combined_df[ColumnNames.DATE].map(date_to_idx).values.astype(np.int64)
    firm_idx = combined_df[firm_id_col].map(firm_to_idx).fillna(-1).values.astype(np.int64)
    rating_terc = combined_df['rating_terc'].values.astype(np.float64)
    ptf_rank = combined_df['ptf_rank'].values.astype(np.float64)
    ret = combined_df['ret'].values.astype(np.float64)
    vw = combined_df['VW'].values.astype(np.float64)

    # Run fast numba aggregation
    long_short, high_ret, low_ret, valid_dates = compute_within_firm_aggregation_fast(
        date_idx, firm_idx, rating_terc, ptf_rank, ret, vw, n_dates, n_firms
    )

    # Convert to pandas Series with date index
    long_short_series = pd.Series(long_short, index=unique_dates)
    high_returns_series = pd.Series(high_ret, index=unique_dates)
    low_returns_series = pd.Series(low_ret, index=unique_dates)

    # Filter to valid dates only
    valid_mask = ~np.isnan(long_short_series.values)
    long_short_series = long_short_series[valid_mask]
    high_returns_series = high_returns_series[valid_mask]
    low_returns_series = low_returns_series[valid_mask]

    return {
        'long_short': long_short_series,
        'long_leg': high_returns_series,
        'short_leg': low_returns_series,
    }


def compute_within_firm_returns_aggregation(
    portfolio_indices: dict,
    data_raw: pd.DataFrame,
    datelist: list,
    firm_id_col: str,
    rating_col: str,
    rating_bins: list,
    use_fast_path: bool = True
) -> dict:
    """
    Post-process portfolio assignments to compute returns with custom aggregation.

    This implements the within-firm return aggregation scheme:
    1. For each portfolio (Q1=low, Q2=high), group bonds by rating tercile and firm
    2. Compute firm-level value-weighted returns
    3. Aggregate across firms using firm cap-weighting
    4. Average across rating groups
    5. Compute long-short (Q2 - Q1)

    Parameters
    ----------
    portfolio_indices : dict
        Dictionary mapping date -> DataFrame with columns:
        [ID, date, ptf_rank, eweights, vweights, ret, VW]
    data_raw : pd.DataFrame
        Original data with firm_id_col and rating_col
    datelist : list
        List of dates
    firm_id_col : str
        Firm identifier column name
    rating_col : str
        Rating column name
    rating_bins : list
        Rating bin edges
    use_fast_path : bool, default True
        If True, use fast numba-based aggregation (~350x speedup)

    Returns
    -------
    dict
        Dictionary with keys:
        - 'long_short': pd.Series of long-short returns (Q2 - Q1)
        - 'long_leg': pd.Series of long portfolio returns (Q2)
        - 'short_leg': pd.Series of short portfolio returns (Q1)
    """
    # Use fast path if enabled
    if use_fast_path:
        return compute_within_firm_returns_aggregation_fast(
            portfolio_indices, data_raw, datelist, firm_id_col, rating_col, rating_bins
        )

    # === SLOW PATH (pandas-based, kept for validation) ===

    # Create firm and rating lookup
    firm_rating_lookup = data_raw[[ColumnNames.ID, firm_id_col, rating_col]].drop_duplicates()

    # Initialize results storage
    high_returns = []
    low_returns = []
    long_short_factors = []  # The actual H-L factors computed firm-by-firm
    dates_with_data = []
    rating_factors = {1: [], 2: [], 3: []}  # Store by rating tercile

    # Process each date
    for date_t in datelist:
        if date_t not in portfolio_indices:
            continue

        port_df = portfolio_indices[date_t]

        # Skip if empty
        if port_df.empty:
            continue

        # Merge with firm and rating info
        port_df_enriched = port_df.merge(
            firm_rating_lookup,
            on=ColumnNames.ID,
            how='left'
        )

        # Create rating terciles
        port_df_enriched['rating_terc'] = pd.cut(
            pd.to_numeric(port_df_enriched[rating_col], errors='coerce'),
            bins=rating_bins,
            labels=[1, 2, 3],
            include_lowest=True
        ).astype('Int64')

        # Filter out bonds without valid rating tercile
        port_df_enriched = port_df_enriched[port_df_enriched['rating_terc'].notna()]

        if port_df_enriched.empty:
            continue

        # Compute firm-level high-low factors first, then aggregate
        # This matches the standalone function's logic exactly

        # Group by rating tercile
        rating_group_factors = []

        for rating_terc in [1, 2, 3]:
            rating_bonds = port_df_enriched[port_df_enriched['rating_terc'] == rating_terc]

            if rating_bonds.empty:
                continue

            # Group by firm within this rating tercile
            firm_grouped = rating_bonds.groupby(firm_id_col)

            firm_hl_factors = []
            firm_weights = []

            for firm, firm_bonds in firm_grouped:
                # Get high and low bonds for this firm
                low_bonds = firm_bonds[firm_bonds['ptf_rank'] == 1]
                high_bonds = firm_bonds[firm_bonds['ptf_rank'] == 2]

                # Need both high and low bonds to compute factor
                if low_bonds.empty or high_bonds.empty:
                    continue

                # Compute VW returns within firm for high and low
                low_vw = low_bonds['VW'].sum()
                high_vw = high_bonds['VW'].sum()

                if low_vw > 0 and high_vw > 0:
                    low_ret = (low_bonds['ret'] * low_bonds['VW']).sum() / low_vw
                    high_ret = (high_bonds['ret'] * high_bonds['VW']).sum() / high_vw

                    if not (np.isnan(low_ret) or np.isnan(high_ret)):
                        # Firm-level H-L factor
                        firm_hl = high_ret - low_ret
                        firm_weight = low_vw + high_vw

                        firm_hl_factors.append(firm_hl)
                        firm_weights.append(firm_weight)

            # Aggregate across firms (firm cap-weighted)
            if len(firm_hl_factors) > 0:
                firm_hl_arr = np.array(firm_hl_factors)
                firm_w_arr = np.array(firm_weights)

                rating_factor = np.sum(firm_hl_arr * firm_w_arr) / np.sum(firm_w_arr)
                rating_group_factors.append(rating_factor)

        # Average across rating groups
        if len(rating_group_factors) > 0:
            overall_factor = np.mean(rating_group_factors)
            dates_with_data.append(date_t)
            long_short_factors.append(overall_factor)

            # For long and short legs, compute them separately
            # Note: These are computed differently than the factor (simple VW, not firm-weighted)
            # This is just for reporting/display purposes

            all_low = port_df_enriched[port_df_enriched['ptf_rank'] == 1]
            all_high = port_df_enriched[port_df_enriched['ptf_rank'] == 2]

            if not all_low.empty and not all_high.empty:
                low_ret_overall = (all_low['ret'] * all_low['VW']).sum() / all_low['VW'].sum()
                high_ret_overall = (all_high['ret'] * all_high['VW']).sum() / all_high['VW'].sum()

                low_returns.append(low_ret_overall)
                high_returns.append(high_ret_overall)
            else:
                low_returns.append(np.nan)
                high_returns.append(np.nan)

    # Create return series
    if len(dates_with_data) == 0:
        # Return empty series
        return {
            'long_short': pd.Series(dtype=float),
            'long_leg': pd.Series(dtype=float),
            'short_leg': pd.Series(dtype=float),
        }

    low_returns_series = pd.Series(low_returns, index=dates_with_data)
    high_returns_series = pd.Series(high_returns, index=dates_with_data)
    long_short_series = pd.Series(long_short_factors, index=dates_with_data)  # Use the correctly computed factors!

    return {
        'long_short': long_short_series,
        'long_leg': high_returns_series,
        'short_leg': low_returns_series,
    }
