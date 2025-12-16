# -*- coding: utf-8 -*-
"""
Created: 03-11-2025
Last modified: 03-11-2025

@authors: Giulio Rossetti 

Portfolio formation and management utilities.

This module handles the core portfolio formation logic, including:
- Portfolio assignment and ranking
- Weight calculations
- Turnover computations
- Characteristic aggregation
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

from .constants import ColumnNames, NumericConstants
# Use numba-optimized versions for performance (2-5x faster)
from .utils_optimized import (
    assign_bond_bins_optimized as assign_bond_bins,
    double_sort_uncond_optimized as double_sort_uncond,
    double_sort_cond_optimized as double_sort_cond,
    compute_thresholds_optimized as compute_thresholds,
    intersect_id_optimized as intersect_id,
    create_subset_mask,
)

# =============================================================================
# Portfolio Ranking
# =============================================================================
def compute_portfolio_ranks(
    data: pd.DataFrame,
    sort_var: str,
    num_portfolios: int,
    breakpoints: Optional[List[float]] = None,
    subset_func: Optional[callable] = None,
    double_sort_params: Optional[Dict] = None
) -> pd.Series:
    """
    Compute portfolio ranks for each bond based on sorting variable(s).
    
    Parameters
    ----------
    data : pd.DataFrame
        Data containing bonds and sorting variables
    sort_var : str
        Primary sorting variable name
    num_portfolios : int
        Number of portfolios for primary sort
    breakpoints : list of float, optional
        Custom breakpoints (percentiles) instead of equal splits
    subset_func : callable, optional
        Function to create subset mask for breakpoint calculation
    double_sort_params : dict, optional
        Parameters for double sorting:
        - 'sort_var2': str, secondary sorting variable
        - 'num_portfolios2': int, number of portfolios for secondary sort
        - 'method': str, 'conditional' or 'unconditional'
    
    Returns
    -------
    pd.Series
        Portfolio ranks (1-indexed)
    """
    # Get subset mask if provided
    subset_mask = None
    if subset_func is not None:
        subset_mask = create_subset_mask(data, subset_func)
    
    # Compute thresholds
    breakpoints_to_use = breakpoints if breakpoints is not None else num_portfolios
    thresholds = compute_thresholds(
        data=data,
        sig=sort_var,
        breakpoints=breakpoints_to_use,
        subset=subset_mask
    )
    
    # Assign primary portfolio ranks
    ranks = assign_bond_bins(
        sortvar=data[sort_var].values,
        thres=thresholds,
        nport=num_portfolios
    )
    
    # Handle double sorting if requested
    if double_sort_params is not None:
        sort_var2 = double_sort_params['sort_var2']
        num_portfolios2 = double_sort_params['num_portfolios2']
        method = double_sort_params.get('method', 'conditional')
        
        if method == 'conditional':
            ranks = double_sort_cond(
                sortvar2=data[sort_var2].values,
                idx1=ranks,
                n1=num_portfolios,
                n2=num_portfolios2
            )
        else:  # unconditional
            # Compute second sort independently
            thresholds2 = compute_thresholds(
                data=data,
                sig=sort_var2,
                breakpoints=num_portfolios2,
                subset=subset_mask
            )
            ranks2 = assign_bond_bins(
                sortvar=data[sort_var2].values,
                thres=thresholds2,
                nport=num_portfolios2
            )
            ranks = double_sort_uncond(
                idx1=ranks,
                idx2=ranks2,
                n1=num_portfolios,
                n2=num_portfolios2
            )
    
    return pd.Series(ranks, index=data.index, name=ColumnNames.PORTFOLIO_RANK)

# =============================================================================
# Weight Calculations
# =============================================================================
def compute_portfolio_weights(
    df: pd.DataFrame,
    portfolio_col: str = ColumnNames.PORTFOLIO_RANK,
    value_weight_col: str = ColumnNames.VALUE_WEIGHT
) -> pd.DataFrame:
    """
    Compute equal-weight and value-weight for each bond within its portfolio.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with portfolio assignments and value weights
    portfolio_col : str
        Column name for portfolio ranks
    value_weight_col : str
        Column name for value weights (e.g., market value)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added columns: 'eweights', 'vweights', 'count'
    """
    df = df.copy()
    
    # Count bonds in each portfolio
    counts = df.groupby(portfolio_col, sort=False)[ColumnNames.ID].size()
    df[ColumnNames.COUNT] = df[portfolio_col].map(counts)
    
    # Equal weights: 1/N within each portfolio
    df[ColumnNames.EQ_WEIGHTS] = 1.0 / df[ColumnNames.COUNT]
    
    # Value weights: proportional to value weight within portfolio
    sums = df.groupby(portfolio_col, sort=False)[value_weight_col].sum()
    df[ColumnNames.VAL_WEIGHTS] = df[value_weight_col] / df[portfolio_col].map(sums)
    df[ColumnNames.VAL_WEIGHTS] = df[ColumnNames.VAL_WEIGHTS].fillna(0.0)
    
    return df

def compute_scaled_weights(
    df: pd.DataFrame,
    ret_col: str,
    portfolio_col: str = ColumnNames.PORTFOLIO_RANK
) -> pd.DataFrame:
    """
    Compute scaled weights for turnover calculation.
    
    Scaled weights account for differential returns within portfolios,
    adjusting weights to reflect positions at period end.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with returns, portfolio ranks, and weights
    ret_col : str
        Return column name
    portfolio_col : str
        Portfolio rank column name
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 'ewret_scaled' and 'vwret_scaled' columns
    """
    df = df.copy()
    
    # Portfolio-level returns
    ew_ret = df.groupby(portfolio_col, sort=False)[ret_col].mean()
    vw_ret = (df[ret_col] * df[ColumnNames.VAL_WEIGHTS]).groupby(
        df[portfolio_col], sort=False
    ).sum()
    
    # Merge portfolio returns back
    df['ewret'] = df[portfolio_col].map(ew_ret)
    df['vwret'] = df[portfolio_col].map(vw_ret)
    
    # Scaled weights
    df['ewret_scaled'] = (
        (1.0 + df[ret_col]) / (1.0 + df['ewret'])
    ) / df[ColumnNames.COUNT]
    
    df['vwret_scaled'] = (
        (1.0 + df[ret_col]) / (1.0 + df['vwret'])
    ) * df[ColumnNames.VAL_WEIGHTS]
    
    return df[['ewret_scaled', 'vwret_scaled']].rename(
        columns={'ewret_scaled': 'eweights', 'vwret_scaled': 'vweights'}
    )

# =============================================================================
# Portfolio Returns
# =============================================================================
def compute_portfolio_returns(
    df: pd.DataFrame,
    ret_col: str,
    num_portfolios: int,
    portfolio_col: str = ColumnNames.PORTFOLIO_RANK
) -> Tuple[List[float], List[float]]:
    """
    Compute equal-weighted and value-weighted portfolio returns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with returns, portfolio ranks, and weights
    ret_col : str
        Return column name
    num_portfolios : int
        Total number of portfolios
    portfolio_col : str
        Portfolio rank column name
    
    Returns
    -------
    tuple of (list, list)
        (equal_weighted_returns, value_weighted_returns)
    """
    # Equal-weighted returns
    ew_returns = df.groupby(portfolio_col, sort=False)[ret_col].mean()
    
    # Value-weighted returns
    vw_returns = (df[ret_col] * df[ColumnNames.WEIGHTS]).groupby(
        df[portfolio_col], sort=False
    ).sum()
    
    # Reindex to ensure all portfolios are present
    portfolio_idx = range(1, num_portfolios + 1)
    ew_returns = ew_returns.reindex(portfolio_idx)
    vw_returns = vw_returns.reindex(portfolio_idx)
    
    return ew_returns.tolist(), vw_returns.tolist()

# =============================================================================
# Characteristic Aggregation
# =============================================================================
def compute_portfolio_characteristics(
    df: pd.DataFrame,
    char_cols: List[str],
    num_portfolios: int,
    portfolio_col: str = ColumnNames.PORTFOLIO_RANK
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute portfolio-level characteristics (equal and value weighted).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with portfolio assignments, weights, and characteristics
    char_cols : list of str
        List of characteristic column names
    num_portfolios : int
        Total number of portfolios
    portfolio_col : str
        Portfolio rank column name
    
    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        (equal_weighted_chars, value_weighted_chars)
        Each DataFrame has portfolios as rows and characteristics as columns
    """
    portfolio_idx = range(1, num_portfolios + 1)
    
    ew_chars = pd.DataFrame()
    vw_chars = pd.DataFrame()
    
    for char in char_cols:
        # Equal-weighted characteristic
        char_ew = df.groupby(portfolio_col, sort=False)[char].mean().reindex(portfolio_idx)
        
        # Value-weighted characteristic
        char_vw = (df[char] * df[ColumnNames.WEIGHTS]).groupby(
            df[portfolio_col], sort=False
        ).sum().reindex(portfolio_idx)
        
        ew_chars = pd.concat([ew_chars, char_ew], axis=1)
        vw_chars = pd.concat([vw_chars, char_vw], axis=1)
    
    vw_chars.columns = ew_chars.columns = char_cols
    
    return ew_chars, vw_chars

# =============================================================================
# Banding 
# =============================================================================
def apply_banding(
    current_ranks: pd.DataFrame,
    previous_ranks: pd.DataFrame,
    threshold: float,
    num_portfolios: int,
    id_col: str = ColumnNames.ID,
    rank_col: str = ColumnNames.PORTFOLIO_RANK
) -> pd.DataFrame:
    """
    Apply banding to stabilize portfolio ranks across periods.
    
    Bonds stay in their current portfolio unless they move more than
    threshold fraction of portfolios.
    This reduces turnover.
    
    Parameters
    ----------
    current_ranks : pd.DataFrame
        Current period ranks with columns [id_col, rank_col]
    previous_ranks : pd.DataFrame
        Previous period ranks with columns [id_col, rank_col]
    threshold : float
        Threshold for rank changes (as fraction of num_portfolios)
        Example: 0.1 means bonds must move 10% of portfolios
    num_portfolios : int
        Total number of portfolios
    id_col : str
        ID column name
    rank_col : str
        Portfolio rank column name
    
    Returns
    -------
    pd.DataFrame
        Updated ranks after applying banding
    """
    # Merge current and previous ranks
    merged = current_ranks.merge(
        previous_ranks[[id_col, rank_col]],
        on=id_col,
        how='left',
        suffixes=('_current', '_previous')
    )
    
    # Calculate threshold in portfolio units
    threshold_portfolios = threshold * num_portfolios
    
    # Apply banding logic
    rank_diff = np.abs(
        merged[f'{rank_col}_current'] - merged[f'{rank_col}_previous']
    )
    
    # Keep previous rank if change is below threshold
    merged[rank_col] = np.where(
        rank_diff < threshold_portfolios,
        merged[f'{rank_col}_previous'],
        merged[f'{rank_col}_current']
    )
    
    # For new bonds (no previous rank), use current rank
    merged[rank_col] = merged[rank_col].fillna(merged[f'{rank_col}_current'])
    
    return merged[[id_col, rank_col]]

def calculate_qnew_vectorized(
    q_old: pd.Series,
    q_sig: pd.Series,
    nport: int,
    threshold: float
) -> pd.Series:
    """
    Vectorized banding calculation (legacy compatible version).
    
    Parameters
    ----------
    q_old : pd.Series
        Previous portfolio ranks
    q_sig : pd.Series
        Current signal-based ranks
    nport : int
        Total number of portfolios
    threshold : float
        Banding threshold
    
    Returns
    -------
    pd.Series
        Adjusted portfolio ranks
    """
    threshold_portfolios = threshold * nport
    rank_diff = np.abs(q_sig - q_old)
    
    q_new = np.where(
        rank_diff < threshold_portfolios,
        q_old,
        q_sig
    )
    
    # Handle NaNs (new bonds)
    q_new = np.where(pd.isna(q_old), q_sig, q_new)
    
    return pd.Series(q_new, index=q_sig.index)

# =============================================================================
# Period Preparation
# =============================================================================
@dataclass
class PeriodData:
    """Container for data at a specific time period."""
    date: pd.Timestamp
    df: pd.DataFrame
    
    @property
    def is_empty(self) -> bool:
        return self.df.empty
    
    @property
    def ids(self) -> pd.Series:
        return self.df[ColumnNames.ID]


def prepare_period_data(
    It0: pd.DataFrame,
    It1: pd.DataFrame,
    It1m: pd.DataFrame,
    dynamic_weights: bool
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare and intersect data across time periods.
    
    Parameters
    ----------
    It0 : pd.DataFrame
        Data at portfolio formation time (t)
    It1 : pd.DataFrame
        Data at return realization time (t+h)
    It1m : pd.DataFrame
        Data at weight rebalance time (t+h-1) for dynamic weights
    dynamic_weights : bool
        Whether to use dynamic weighting
    
    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame, pd.DataFrame)
        Filtered versions of (It0, It1, It1m) with common IDs
    """
    return intersect_id(It0, It1, It1m, dynamic_weights)

# =============================================================================
# Portfolio Formation Result
# =============================================================================
@dataclass
class PortfolioFormationResult:
    """
    Container for portfolio formation results for a single period.
    
    Attributes
    ----------
    returns_ew : list
        Equal-weighted returns for each portfolio
    returns_vw : list
        Value-weighted returns for each portfolio
    weights : pd.DataFrame
        Bond-level weights (ID, ptf_rank, eweights, vweights)
    weights_scaled : pd.DataFrame
        Scaled weights for turnover calculation
    chars_ew : pd.DataFrame, optional
        Equal-weighted characteristics
    chars_vw : pd.DataFrame, optional
        Value-weighted characteristics
    date : pd.Timestamp
        Period date
    """
    returns_ew: List[float]
    returns_vw: List[float]
    weights: pd.DataFrame
    weights_scaled: pd.DataFrame
    chars_ew: Optional[pd.DataFrame] = None
    chars_vw: Optional[pd.DataFrame] = None
    date: Optional[pd.Timestamp] = None
    
    @property
    def has_characteristics(self) -> bool:
        """Check if characteristics are available."""
        return self.chars_ew is not None and self.chars_vw is not None
    
    def to_nan(self, num_portfolios: int, num_chars: int = 0):
        """Create a NaN result (for periods with no valid data)."""
        nan_list = [np.nan] * num_portfolios
        self.returns_ew = nan_list
        self.returns_vw = nan_list
        self.weights = pd.DataFrame()
        self.weights_scaled = pd.DataFrame()
        
        if num_chars > 0:
            nan_df = pd.DataFrame(
                np.full((num_portfolios, num_chars), np.nan)
            )
            self.chars_ew = nan_df
            self.chars_vw = nan_df

# =============================================================================
# Main Portfolio Formation Function
# =============================================================================
def form_portfolio_single_period(
    It0: pd.DataFrame,
    It1: pd.DataFrame,
    It1m: pd.DataFrame,
    ranks_map: Dict[pd.Timestamp, pd.Series],
    vw_map_t0: Dict[pd.Timestamp, pd.Series],
    vw_map_t1m: Dict[pd.Timestamp, pd.Series],
    date_t: pd.Timestamp,
    date_t1_minus1: Optional[pd.Timestamp],
    ret_col: str,
    num_portfolios: int,
    dynamic_weights: bool = False,
    char_cols: Optional[List[str]] = None,
    banding_params: Optional[Dict] = None
) -> PortfolioFormationResult:
    """
    Form portfolios for a single time period.
    
    This is the main function that orchestrates portfolio formation for one period.
    
    Parameters
    ----------
    It0, It1, It1m : pd.DataFrame
        Data at different time periods
    ranks_map : dict
        Precomputed portfolio ranks by date
    vw_map_t0, vw_map_t1m : dict
        Precomputed value weights by date
    date_t : pd.Timestamp
        Formation date
    date_t1_minus1 : pd.Timestamp, optional
        Date for dynamic weights (t+h-1)
    ret_col : str
        Return column name
    num_portfolios : int
        Number of portfolios
    dynamic_weights : bool
        Use dynamic weighting
    char_cols : list of str, optional
        Characteristic columns to aggregate
    banding_params : dict, optional
        Banding parameters if applicable
    
    Returns
    -------
    PortfolioFormationResult
        Complete formation results for the period
    """
    # Handle empty data
    if It0.empty or It1.empty:
        result = PortfolioFormationResult(
            returns_ew=[np.nan] * num_portfolios,
            returns_vw=[np.nan] * num_portfolios,
            weights=pd.DataFrame(),
            weights_scaled=pd.DataFrame(),
            date=date_t
        )
        if char_cols:
            result.to_nan(num_portfolios, len(char_cols))
        return result
    
    # Intersect IDs
    It0, It1, It1m = prepare_period_data(It0, It1, It1m, dynamic_weights)
    
    if It0.shape[0] == 0:
        result = PortfolioFormationResult(
            returns_ew=[np.nan] * num_portfolios,
            returns_vw=[np.nan] * num_portfolios,
            weights=pd.DataFrame(),
            weights_scaled=pd.DataFrame(),
            date=date_t
        )
        if char_cols:
            result.to_nan(num_portfolios, len(char_cols))
        return result
    
    # Map portfolio ranks
    It1[ColumnNames.PORTFOLIO_RANK] = It1[ColumnNames.ID].map(
        ranks_map.get(date_t, pd.Series(dtype='Int64'))
    )
    It1 = It1.dropna(subset=[ColumnNames.PORTFOLIO_RANK])
    
    if It1.empty:
        result = PortfolioFormationResult(
            returns_ew=[np.nan] * num_portfolios,
            returns_vw=[np.nan] * num_portfolios,
            weights=pd.DataFrame(),
            weights_scaled=pd.DataFrame(),
            date=date_t
        )
        if char_cols:
            result.to_nan(num_portfolios, len(char_cols))
        return result
    
    It1[ColumnNames.PORTFOLIO_RANK] = It1[ColumnNames.PORTFOLIO_RANK].astype(int)
    
    # Get value weights
    if dynamic_weights and date_t1_minus1 is not None:
        vw_map = vw_map_t1m.get(date_t1_minus1, pd.Series(dtype=float))
    else:
        vw_map = vw_map_t0.get(date_t, pd.Series(dtype=float))
    
    It1[ColumnNames.VALUE_WEIGHT] = It1[ColumnNames.ID].map(vw_map)
    
    # Apply banding if requested
    if banding_params is not None:
        # This would be called from the parent with appropriate previous ranks
        pass
    
    # Compute weights
    It1 = compute_portfolio_weights(It1)
    
    # Copy weights column for return calculation
    It1[ColumnNames.WEIGHTS] = It1[ColumnNames.VAL_WEIGHTS]
    
    # Compute returns
    returns_ew, returns_vw = compute_portfolio_returns(
        It1, ret_col, num_portfolios
    )
    
    # Prepare weight outputs
    weights_df = It1[[
        ColumnNames.ID, 
        ColumnNames.PORTFOLIO_RANK, 
        ColumnNames.EQ_WEIGHTS, 
        ColumnNames.VAL_WEIGHTS
    ]].copy()
    
    weights_scaled_df = compute_scaled_weights(
        It1[It1.columns.intersection([
            ColumnNames.ID, ColumnNames.PORTFOLIO_RANK, 
            ret_col, ColumnNames.VAL_WEIGHTS, ColumnNames.COUNT
        ])],
        ret_col
    )
    
    # Compute characteristics if requested
    chars_ew, chars_vw = None, None
    if char_cols:
        # Merge characteristics from It1m
        It1_aug = It1[[ColumnNames.ID, ColumnNames.PORTFOLIO_RANK, ColumnNames.WEIGHTS]].merge(
            It1m[[ColumnNames.ID] + char_cols],
            on=ColumnNames.ID,
            how='inner'
        )
        chars_ew, chars_vw = compute_portfolio_characteristics(
            It1_aug, char_cols, num_portfolios
        )
    
    return PortfolioFormationResult(
        returns_ew=returns_ew,
        returns_vw=returns_vw,
        weights=weights_df,
        weights_scaled=weights_scaled_df,
        chars_ew=chars_ew,
        chars_vw=chars_vw,
        date=date_t
    )
