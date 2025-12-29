# -*- coding: utf-8 -*-
"""
Created: 17-06-2024
Last modified: 03-11-2025
@authors: Giulio Rossetti

Turnover-related helper functions for PyBondLab.

This module provides unified turnover computation for both staggered and non-staggered
portfolio rebalancing strategies. It tracks portfolio composition changes over time
and calculates turnover metrics for both equal-weighted (EW) and value-weighted (VW) portfolios.

Key concepts:
- Turnover = sum(|current_weight - previous_weight|) / 2
- Scaled weights account for portfolio returns between rebalancing dates
- Staggered: Multiple overlapping cohorts rebalanced at different times
- Non-staggered: Single portfolio rebalanced at specific dates
"""

import numpy as np
import pandas as pd
import warnings
from typing import Optional

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

# Import optimized turnover computation
try:
    from .numba_core import compute_turnover_all_portfolios, update_prev_scaled_weights
    NUMBA_TURNOVER_AVAILABLE = True
except ImportError:
    NUMBA_TURNOVER_AVAILABLE = False

# Import turnover tracking system
try:
    from .turnover_tracking import (
        TurnoverLogger,
        compute_turnover_diagnostic,
        classify_turnover_case
    )
    TRACKING_AVAILABLE = True
except ImportError:
    TRACKING_AVAILABLE = False


# =============================================================================
# Optional Numba acceleration for inner min-sum computation
# =============================================================================
if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True)
    def _sum_min_prev_raw(prev_vec, pos, raw_w):
        """
        Compute sum of element-wise minimums between previous and current weights.
        Numba-accelerated version for performance.
        
        Parameters
        ----------
        prev_vec : np.ndarray
            Previous scaled weights for all assets
        pos : np.ndarray
            Position indices for current portfolio assets
        raw_w : np.ndarray
            Current raw weights for portfolio assets
            
        Returns
        -------
        float
            Sum of minimums
        """
        s = 0.0
        m = pos.size
        for j in range(m):
            pv = prev_vec[pos[j]]
            rw = raw_w[j]
            s += pv if pv < rw else rw
        return s
else:
    def _sum_min_prev_raw(prev_vec, pos, raw_w):
        """Pure numpy fallback when Numba is not available."""
        return float(np.minimum(prev_vec[pos], raw_w).sum())


# =============================================================================
# Turnover manager class
# =============================================================================

class TurnoverState:
    """
    Manages turnover state for either staggered or non-staggered rebalancing.
    
    For staggered rebalancing:
    - Tracks multiple cohorts (hor cohorts, one per month)
    - Each cohort maintains separate weight history
    - Arrays are shaped (hor, tot_nport, n_assets)
    
    For non-staggered rebalancing:
    - Tracks single cohort
    - Arrays are shaped (1, tot_nport, n_assets)
    - Simpler state management
    
    Attributes
    ----------
    id_to_pos : pd.Series
        Maps asset IDs to array positions for fast lookup
    prev_scaled_ew : np.ndarray
        Previous scaled equal weights by cohort and portfolio
    prev_scaled_vw : np.ndarray
        Previous scaled value weights by cohort and portfolio
    prev_sum_ew : np.ndarray
        Sum of previous EW for each cohort/portfolio (for turnover calculation)
    prev_sum_vw : np.ndarray
        Sum of previous VW for each cohort/portfolio (for turnover calculation)
    prev_seen_ew : np.ndarray (bool)
        Whether each cohort/portfolio has been initialized (EW)
    prev_seen_vw : np.ndarray (bool)
        Whether each cohort/portfolio has been initialized (VW)
    ew_turn_ea : np.ndarray
        Equal-weighted turnover time series (EA strategy)
    vw_turn_ea : np.ndarray
        Value-weighted turnover time series (EA strategy)
    is_staggered : bool
        Whether this is for staggered rebalancing
    """
    
    def __init__(self, hor: int, tot_nport: int, all_ids: np.ndarray, is_staggered: bool = True,
                 logger: Optional['TurnoverLogger'] = None):
        """
        Initialize turnover state.

        Parameters
        ----------
        hor : int
            Holding horizon (number of cohorts for staggered, or max holding for non-staggered)
        tot_nport : int
            Total number of portfolios
        all_ids : np.ndarray
            Sorted array of all asset IDs in the dataset
        is_staggered : bool, default=True
            Whether this is for staggered rebalancing
        logger : TurnoverLogger, optional
            Logger for tracking turnover diagnostics. If provided, detailed logging will be enabled.
        """
        self.is_staggered = is_staggered
        self.hor = hor
        self.logger = logger  # Store logger for tracking

        # Create ID to position mapping for fast lookups
        self.id_to_pos = pd.Series(np.arange(all_ids.size, dtype=np.int64), index=all_ids)

        n_ids = all_ids.size

        # For staggered: track all cohorts; for non-staggered: track single cohort
        n_cohorts = hor if is_staggered else 1

        # Previous scaled weights (for computing turnover at t+1)
        self.prev_scaled_ew = np.zeros((n_cohorts, tot_nport, n_ids), dtype=np.float64)
        self.prev_scaled_vw = np.zeros_like(self.prev_scaled_ew)

        # Sums of previous weights (for turnover formula)
        self.prev_sum_ew = np.zeros((n_cohorts, tot_nport), dtype=np.float64)
        self.prev_sum_vw = np.zeros((n_cohorts, tot_nport), dtype=np.float64)

        # Track whether each cohort/portfolio has been initialized
        self.prev_seen_ew = np.zeros((n_cohorts, tot_nport), dtype=bool)
        self.prev_seen_vw = np.zeros((n_cohorts, tot_nport), dtype=bool)

        # Turnover time series (will be sized during init)
        self.ew_turn_ea = np.full((0, n_cohorts, tot_nport), np.nan)
        self.vw_turn_ea = np.full((0, n_cohorts, tot_nport), np.nan)

class TurnoverManager:
    """
    Manages all turnover computation for portfolio strategies.
    
    Attributes
    ----------
    data : pd.DataFrame
        Portfolio data
    datelist : list
        List of dates
    hor : int
        Holding horizon
    rebalance_frequency : str or int
        Rebalancing frequency
    is_staggered : bool
        Whether using staggered rebalancing
    state : TurnoverState
        Current turnover state
    """
    
    def __init__(self, data, datelist, hor, rebalance_frequency='monthly',
                 use_nanmean=True):
        """
        Initialize TurnoverManager.

        Parameters
        ----------
        use_nanmean : bool, default=True
            If True, use np.nanmean for cohort averaging (ignores NaN values).
            If False, use np.mean (propagates NaN values, matches old behavior).
            Set to False for backward compatibility with old package.
        """
        self.data = data
        self.datelist = datelist
        self.hor = hor
        self.rebalance_frequency = rebalance_frequency
        self.is_staggered = (rebalance_frequency == 'monthly')
        self.use_nanmean = use_nanmean
        self.state = None
    
    def init_state(self, TM, tot_nport):
        """
        Initialize turnover state for portfolio formation.
        
        Parameters
        ----------
        TM : int
            Total number of time periods
        tot_nport : int
            Total number of portfolios
            
        Returns
        -------
        TurnoverState
            Initialized turnover state object
        """
        # Get sorted list of all unique asset IDs
        all_ids = np.sort(self.data['ID'].unique())
        
        # Create state object
        self.state = TurnoverState(self.hor, tot_nport, all_ids, self.is_staggered)
        
        # Size the turnover arrays based on number of time periods
        if self.is_staggered:
            # Staggered: (time, cohort, portfolio)
            self.state.ew_turn_ea = np.full((TM, self.hor, tot_nport), np.nan)
            self.state.vw_turn_ea = np.full((TM, self.hor, tot_nport), np.nan)
        else:
            # Non-staggered: (time, portfolio) - no cohort dimension
            self.state.ew_turn_ea = np.full((TM, tot_nport), np.nan)
            self.state.vw_turn_ea = np.full((TM, tot_nport), np.nan)
        
        return self.state
    
    def accumulate(self, state, cohort_or_key, tot_nport, tau,
                   weights_df, weights_scaled_df):
        """Compute turnover (staggered)."""
        accumulate_turnover(state, cohort_or_key, tot_nport, tau,
                          weights_df, weights_scaled_df)

    def set_zero_for_holding_cohorts(self, state, rebalancing_cohort, tot_nport, tau):
        """
        Set turnover to 0.0 for active cohorts that are NOT rebalancing at time tau.

        This reflects that holding cohorts do not trade and therefore have zero turnover.
        Non-existent cohorts (not yet formed) remain as NaN.

        Parameters
        ----------
        state : TurnoverState
            The turnover state object
        rebalancing_cohort : int
            The cohort that IS rebalancing at this time (will be skipped)
        tot_nport : int
            Total number of portfolios
        tau : int
            Current time index (formation time)

        Notes
        -----
        A cohort c is considered "active" (exists) at time tau if c <= tau.
        - Cohort 0 is formed at t=0
        - Cohort 1 is formed at t=1
        - Cohort c is formed at t=c (first formation)

        After first formation, each cohort rebalances every `hor` periods.
        """
        if not state.is_staggered:
            return  # Only applies to staggered rebalancing

        hor = state.hor

        for cohort in range(hor):
            # Skip the rebalancing cohort (it already has computed turnover)
            if cohort == rebalancing_cohort:
                continue

            # Check if this cohort exists (has been formed at least once)
            # Cohort c is first formed at time t=c
            if cohort > tau:
                # Cohort doesn't exist yet - leave as NaN
                continue

            # Cohort exists but is not rebalancing - set to 0.0 for all portfolios
            for k in range(tot_nport):
                # Only set to 0.0 if the cohort has been seen (prev_seen is True)
                # This ensures we don't set zeros before the cohort has any data
                if state.prev_seen_ew[cohort, k]:
                    state.ew_turn_ea[tau, cohort, k] = 0.0
                if state.prev_seen_vw[cohort, k]:
                    state.vw_turn_ea[tau, cohort, k] = 0.0

    def compute(self, state, weights_df, weights_scaled_df, tau, tot_nport,
                is_rebalancing_date=True):
        """Compute turnover (non-staggered).

        Parameters
        ----------
        state : TurnoverState
            Turnover state object
        weights_df : pd.DataFrame
            Current weights
        weights_scaled_df : pd.DataFrame
            Scaled weights for next period
        tau : int
            Time index
        tot_nport : int
            Total number of portfolios
        is_rebalancing_date : bool, default=True
            If True, compute actual turnover (trading occurs).
            If False, set turnover to 0.0 (holding period, no trading).
            This is similar to staggered rebalancing where holding cohorts get 0.
        """
        if not is_rebalancing_date:
            # Between rebalancing dates: no trading, turnover = 0
            # This matches staggered holding cohort behavior
            for k in range(tot_nport):
                state.ew_turn_ea[tau, k] = 0.0
                state.vw_turn_ea[tau, k] = 0.0
            # Still need to update previous weights for next comparison
            # but skip turnover computation
            _update_prev_weights_only(state, weights_scaled_df, tot_nport)
            return

        # For non-staggered, we use cohort index 0 (single cohort)
        compute_nonstaggered_turnover_state(
            state, weights_df, weights_scaled_df, tau, tot_nport
        )
    
    def finalize(self, state, ptf_labels):
        """Finalize and return turnover DataFrames."""
        return finalize_turnover(state, self.datelist, ptf_labels,
                                use_nanmean=self.use_nanmean)


# =============================================================================
# Turnover state management
# =============================================================================

# =============================================================================
# Core turnover computation
# =============================================================================

def _accumulate_turnover_fast(state: TurnoverState,
                              cohort: int,
                              tot_nport: int,
                              tau: int,
                              weights_df: pd.DataFrame,
                              weights_scaled_df: pd.DataFrame):
    """
    Fast path for turnover accumulation using numba.

    Extracts all data to numpy arrays once, then processes all portfolios
    in a single pass instead of looping with DataFrame filtering.
    """
    id_to_pos = state.id_to_pos
    n_assets = state.prev_scaled_ew.shape[2]

    # Extract arrays from weights_df
    ranks = weights_df['ptf_rank'].values.astype(np.float64)
    ids = weights_df['ID'].values
    positions = id_to_pos.loc[ids].values.astype(np.int64)
    raw_ew = weights_df['eweights'].values.astype(np.float64)
    raw_vw = weights_df['vweights'].values.astype(np.float64)

    # Get previous state for this cohort
    prev_scaled_ew_cohort = state.prev_scaled_ew[cohort]  # (nport, n_assets)
    prev_scaled_vw_cohort = state.prev_scaled_vw[cohort]
    prev_sum_ew_cohort = state.prev_sum_ew[cohort]  # (nport,)
    prev_sum_vw_cohort = state.prev_sum_vw[cohort]
    prev_seen_ew_cohort = state.prev_seen_ew[cohort]  # (nport,)
    prev_seen_vw_cohort = state.prev_seen_vw[cohort]

    # Compute turnover for all portfolios at once
    turn_ew, turn_vw, new_seen_ew, new_seen_vw, curr_sum_ew, curr_sum_vw = \
        compute_turnover_all_portfolios(
            ranks, positions, raw_ew, raw_vw,
            prev_scaled_ew_cohort, prev_scaled_vw_cohort,
            prev_sum_ew_cohort, prev_sum_vw_cohort,
            prev_seen_ew_cohort, prev_seen_vw_cohort,
            cohort, tot_nport, n_assets
        )

    # Store turnover values
    for k0 in range(tot_nport):
        if not np.isnan(turn_ew[k0]):
            state.ew_turn_ea[tau, cohort, k0] = turn_ew[k0]
        if not np.isnan(turn_vw[k0]):
            state.vw_turn_ea[tau, cohort, k0] = turn_vw[k0]

    # Update seen flags
    state.prev_seen_ew[cohort] = new_seen_ew
    state.prev_seen_vw[cohort] = new_seen_vw

    # Update previous scaled weights and sums from scaled weights DataFrame
    scaled_ranks = weights_scaled_df['ptf_rank'].values.astype(np.float64)
    scaled_ids = weights_scaled_df['ID'].values
    scaled_positions = id_to_pos.loc[scaled_ids].values.astype(np.int64)
    scaled_ew = weights_scaled_df['eweights'].values.astype(np.float64)
    scaled_vw = weights_scaled_df['vweights'].values.astype(np.float64)

    # Use numba to update previous weights
    update_prev_scaled_weights(
        scaled_ew, scaled_vw, scaled_ranks, scaled_positions,
        state.prev_scaled_ew[cohort], state.prev_scaled_vw[cohort],
        tot_nport
    )

    # Update sums
    for k0 in range(tot_nport):
        k = k0 + 1
        mask = scaled_ranks == k
        if np.any(mask):
            state.prev_sum_ew[cohort, k0] = np.sum(scaled_ew[mask])
            state.prev_sum_vw[cohort, k0] = np.sum(scaled_vw[mask])


def accumulate_turnover(state: TurnoverState,
                       cohort: int,
                       tot_nport: int,
                       tau: int,
                       weights_df: pd.DataFrame,
                       weights_scaled_df: pd.DataFrame):
    """
    Accumulate turnover for staggered rebalancing at a given time and cohort.

    This function:
    1. Computes turnover by comparing current weights with previous scaled weights
    2. Updates the state with current scaled weights for next period
    3. Stores turnover values in the state arrays

    Parameters
    ----------
    state : TurnoverState
        Turnover state object containing previous weights and turnover arrays
    cohort : int
        Cohort index (0 to hor-1)
    tot_nport : int
        Total number of portfolios
    tau : int
        Time index in the overall time series
    weights_df : pd.DataFrame
        Current weights with columns: ID, ptf_rank, eweights, vweights
    weights_scaled_df : pd.DataFrame
        Scaled weights accounting for returns since formation
    """
    # Fast path using numba-optimized batch computation
    if NUMBA_TURNOVER_AVAILABLE and state.logger is None:
        _accumulate_turnover_fast(state, cohort, tot_nport, tau, weights_df, weights_scaled_df)
        return

    # Get ID positions for fast lookup
    id_to_pos = state.id_to_pos

    # Process each portfolio
    for k in range(1, tot_nport + 1):
        k0 = k - 1  # Zero-based portfolio index
        
        # Filter current portfolio
        curr_ptf = weights_df[weights_df['ptf_rank'] == k].copy()
        
        if curr_ptf.empty:
            continue
        
        # Get asset IDs and map to positions
        curr_ids = curr_ptf['ID'].values
        pos = id_to_pos.loc[curr_ids].values.astype(np.int64)
        
        # Current raw weights
        raw_ew = curr_ptf['eweights'].values
        raw_vw = curr_ptf['vweights'].values
        
        # ===== EQUAL WEIGHTS =====
        if state.prev_seen_ew[cohort, k0]:
            # Get previous scaled weights for this cohort/portfolio
            prev_vec_ew = state.prev_scaled_ew[cohort, k0, :]

            # Get previous IDs (non-zero weights)
            prev_ids = np.where(prev_vec_ew > 0)[0]
            prev_weights_nonzero = prev_vec_ew[prev_ids]

            # Compute sum of min(prev, current) for matching assets
            sum_min_ew = _sum_min_prev_raw(prev_vec_ew, pos, raw_ew)

            # Turnover = prev_sum + curr_sum - 2*sum_min
            prev_sum_ew = state.prev_sum_ew[cohort, k0]
            curr_sum_ew = raw_ew.sum()
            turn_ew = prev_sum_ew + curr_sum_ew - 2.0 * sum_min_ew

            # Store turnover
            state.ew_turn_ea[tau, cohort, k0] = turn_ew

            # === TRACKING LOGIC ===
            if state.logger is not None and TRACKING_AVAILABLE:
                # Get date string (need to pass it somehow - will fix in next iteration)
                # For now, use tau as date identifier
                date_str = f"tau_{tau}"
                portfolio_id = f"P{k}"

                # Compute diagnostic
                try:
                    diagnostic = compute_turnover_diagnostic(
                        date=date_str,
                        portfolio_id=portfolio_id,
                        ids_t=curr_ids,
                        weights_t=raw_ew,
                        ids_t_minus_1=prev_ids,
                        weights_t_minus_1=prev_weights_nonzero,
                        turnover_value=turn_ew,
                        returns=None,  # Not available here
                        cohort_id=cohort
                    )
                    state.logger.log(diagnostic)
                except Exception:
                    # Silently skip logging on error to avoid breaking calculation
                    pass
        else:
            # First time seeing this portfolio - mark as seen
            state.prev_seen_ew[cohort, k0] = True

            # === TRACKING LOGIC FOR FIRST OBSERVATION ===
            if state.logger is not None and TRACKING_AVAILABLE:
                date_str = f"tau_{tau}"
                portfolio_id = f"P{k}"

                try:
                    diagnostic = compute_turnover_diagnostic(
                        date=date_str,
                        portfolio_id=portfolio_id,
                        ids_t=curr_ids,
                        weights_t=raw_ew,
                        ids_t_minus_1=np.array([]),
                        weights_t_minus_1=np.array([]),
                        turnover_value=np.nan,
                        returns=None,
                        cohort_id=cohort
                    )
                    state.logger.log(diagnostic)
                except Exception:
                    pass
        
        # ===== VALUE WEIGHTS =====
        if state.prev_seen_vw[cohort, k0]:
            # Get previous scaled weights for this cohort/portfolio
            prev_vec_vw = state.prev_scaled_vw[cohort, k0, :]

            # Get previous IDs (non-zero weights)
            prev_ids_vw = np.where(prev_vec_vw > 0)[0]
            prev_weights_vw_nonzero = prev_vec_vw[prev_ids_vw]

            # Compute sum of min(prev, current) for matching assets
            sum_min_vw = _sum_min_prev_raw(prev_vec_vw, pos, raw_vw)

            # Turnover = prev_sum + curr_sum - 2*sum_min
            prev_sum_vw = state.prev_sum_vw[cohort, k0]
            curr_sum_vw = raw_vw.sum()
            turn_vw = prev_sum_vw + curr_sum_vw - 2.0 * sum_min_vw

            # Store turnover
            state.vw_turn_ea[tau, cohort, k0] = turn_vw

            # === TRACKING LOGIC ===
            if state.logger is not None and TRACKING_AVAILABLE:
                date_str = f"tau_{tau}"
                portfolio_id = f"P{k}_VW"

                try:
                    diagnostic = compute_turnover_diagnostic(
                        date=date_str,
                        portfolio_id=portfolio_id,
                        ids_t=curr_ids,
                        weights_t=raw_vw,
                        ids_t_minus_1=prev_ids_vw,
                        weights_t_minus_1=prev_weights_vw_nonzero,
                        turnover_value=turn_vw,
                        returns=None,
                        cohort_id=cohort
                    )
                    state.logger.log(diagnostic)
                except Exception:
                    pass
        else:
            # First time seeing this portfolio - mark as seen
            state.prev_seen_vw[cohort, k0] = True

            # === TRACKING LOGIC FOR FIRST OBSERVATION ===
            if state.logger is not None and TRACKING_AVAILABLE:
                date_str = f"tau_{tau}"
                portfolio_id = f"P{k}_VW"

                try:
                    diagnostic = compute_turnover_diagnostic(
                        date=date_str,
                        portfolio_id=portfolio_id,
                        ids_t=curr_ids,
                        weights_t=raw_vw,
                        ids_t_minus_1=np.array([]),
                        weights_t_minus_1=np.array([]),
                        turnover_value=np.nan,
                        returns=None,
                        cohort_id=cohort
                    )
                    state.logger.log(diagnostic)
                except Exception:
                    pass
        
        # ===== UPDATE STATE WITH SCALED WEIGHTS FOR NEXT PERIOD =====
        # Get scaled weights for this portfolio
        curr_ptf_scaled = weights_scaled_df[weights_scaled_df['ptf_rank'] == k].copy()
        
        if not curr_ptf_scaled.empty:
            # Reset previous weights to zero
            state.prev_scaled_ew[cohort, k0, :] = 0.0
            state.prev_scaled_vw[cohort, k0, :] = 0.0
            
            # Get IDs and positions for scaled weights
            scaled_ids = curr_ptf_scaled['ID'].values
            scaled_pos = id_to_pos.loc[scaled_ids].values.astype(np.int64)
            
            # Store new scaled weights
            state.prev_scaled_ew[cohort, k0, scaled_pos] = curr_ptf_scaled['eweights'].values
            state.prev_scaled_vw[cohort, k0, scaled_pos] = curr_ptf_scaled['vweights'].values
            
            # Store sums for next turnover calculation
            state.prev_sum_ew[cohort, k0] = curr_ptf_scaled['eweights'].sum()
            state.prev_sum_vw[cohort, k0] = curr_ptf_scaled['vweights'].sum()


# =============================================================================
# Non-staggered turnover computation with TurnoverState
# =============================================================================

def _update_prev_weights_only(state: TurnoverState, weights_scaled_df: pd.DataFrame,
                              tot_nport: int):
    """
    Update previous weights without computing turnover.

    This is used between rebalancing dates when we want to track weights
    but not compute turnover (since no trading occurs).

    Parameters
    ----------
    state : TurnoverState
        Turnover state object
    weights_scaled_df : pd.DataFrame
        Scaled weights for next period
    tot_nport : int
        Total number of portfolios
    """
    cohort = 0  # Non-staggered uses single cohort
    id_to_pos = state.id_to_pos

    for k in range(1, tot_nport + 1):
        k0 = k - 1

        curr_ptf_scaled = weights_scaled_df[weights_scaled_df['ptf_rank'] == k].copy()

        if curr_ptf_scaled.empty:
            continue

        # Reset previous weights to zero
        state.prev_scaled_ew[cohort, k0, :] = 0.0
        state.prev_scaled_vw[cohort, k0, :] = 0.0

        # Get IDs and positions
        scaled_ids = curr_ptf_scaled['ID'].values
        scaled_pos = id_to_pos.loc[scaled_ids].values.astype(np.int64)

        # Store new scaled weights
        state.prev_scaled_ew[cohort, k0, scaled_pos] = curr_ptf_scaled['eweights'].values
        state.prev_scaled_vw[cohort, k0, scaled_pos] = curr_ptf_scaled['vweights'].values

        # Store sums for next turnover calculation
        state.prev_sum_ew[cohort, k0] = curr_ptf_scaled['eweights'].sum()
        state.prev_sum_vw[cohort, k0] = curr_ptf_scaled['vweights'].sum()

        # Mark as seen
        state.prev_seen_ew[cohort, k0] = True
        state.prev_seen_vw[cohort, k0] = True


def compute_nonstaggered_turnover_state(
    state: TurnoverState,
    weights_df: pd.DataFrame,
    weights_scaled_df: pd.DataFrame,
    tau: int,
    tot_nport: int
):
    """
    Compute turnover for non-staggered rebalancing using TurnoverState.

    This function is similar to accumulate_turnover but designed for
    non-staggered rebalancing (single cohort at index 0).

    Parameters
    ----------
    state : TurnoverState
        Turnover state object
    weights_df : pd.DataFrame
        Current weights with columns: ID, ptf_rank, eweights, vweights
    weights_scaled_df : pd.DataFrame
        Scaled weights for next period
    tau : int
        Time index in the overall time series
    tot_nport : int
        Total number of portfolios
    """
    # For non-staggered, always use cohort index 0
    cohort = 0

    # Get ID positions for fast lookup
    id_to_pos = state.id_to_pos

    # Process each portfolio
    for k in range(1, tot_nport + 1):
        k0 = k - 1  # Zero-based portfolio index

        # Filter current portfolio
        curr_ptf = weights_df[weights_df['ptf_rank'] == k].copy()

        if curr_ptf.empty:
            continue

        # Get asset IDs and map to positions
        curr_ids = curr_ptf['ID'].values
        pos = id_to_pos.loc[curr_ids].values.astype(np.int64)

        # Current raw weights
        raw_ew = curr_ptf['eweights'].values
        raw_vw = curr_ptf['vweights'].values

        # ===== EQUAL WEIGHTS =====
        if state.prev_seen_ew[cohort, k0]:
            # Get previous scaled weights
            prev_vec_ew = state.prev_scaled_ew[cohort, k0, :]

            # Get previous IDs (non-zero weights)
            prev_ids = np.where(prev_vec_ew > 0)[0]
            prev_weights_nonzero = prev_vec_ew[prev_ids]

            # Compute sum of min(prev, current)
            sum_min_ew = _sum_min_prev_raw(prev_vec_ew, pos, raw_ew)

            # Turnover = prev_sum + curr_sum - 2*sum_min
            prev_sum_ew = state.prev_sum_ew[cohort, k0]
            curr_sum_ew = raw_ew.sum()
            turn_ew = prev_sum_ew + curr_sum_ew - 2.0 * sum_min_ew

            # Store turnover at correct index
            state.ew_turn_ea[tau, k0] = turn_ew

            # === TRACKING LOGIC ===
            if state.logger is not None and TRACKING_AVAILABLE:
                date_str = f"tau_{tau}"
                portfolio_id = f"P{k}"

                try:
                    diagnostic = compute_turnover_diagnostic(
                        date=date_str,
                        portfolio_id=portfolio_id,
                        ids_t=curr_ids,
                        weights_t=raw_ew,
                        ids_t_minus_1=prev_ids,
                        weights_t_minus_1=prev_weights_nonzero,
                        turnover_value=turn_ew,
                        returns=None,
                        cohort_id=None  # Non-staggered has no cohort
                    )
                    state.logger.log(diagnostic)
                except Exception:
                    pass
        else:
            # First time - mark as seen
            state.prev_seen_ew[cohort, k0] = True

            # === TRACKING LOGIC FOR FIRST OBSERVATION ===
            if state.logger is not None and TRACKING_AVAILABLE:
                date_str = f"tau_{tau}"
                portfolio_id = f"P{k}"

                try:
                    diagnostic = compute_turnover_diagnostic(
                        date=date_str,
                        portfolio_id=portfolio_id,
                        ids_t=curr_ids,
                        weights_t=raw_ew,
                        ids_t_minus_1=np.array([]),
                        weights_t_minus_1=np.array([]),
                        turnover_value=np.nan,
                        returns=None,
                        cohort_id=None
                    )
                    state.logger.log(diagnostic)
                except Exception:
                    pass

        # ===== VALUE WEIGHTS =====
        if state.prev_seen_vw[cohort, k0]:
            # Get previous scaled weights
            prev_vec_vw = state.prev_scaled_vw[cohort, k0, :]

            # Get previous IDs (non-zero weights)
            prev_ids_vw = np.where(prev_vec_vw > 0)[0]
            prev_weights_vw_nonzero = prev_vec_vw[prev_ids_vw]

            # Compute sum of min(prev, current)
            sum_min_vw = _sum_min_prev_raw(prev_vec_vw, pos, raw_vw)

            # Turnover = prev_sum + curr_sum - 2*sum_min
            prev_sum_vw = state.prev_sum_vw[cohort, k0]
            curr_sum_vw = raw_vw.sum()
            turn_vw = prev_sum_vw + curr_sum_vw - 2.0 * sum_min_vw

            # Store turnover at correct index
            state.vw_turn_ea[tau, k0] = turn_vw

            # === TRACKING LOGIC ===
            if state.logger is not None and TRACKING_AVAILABLE:
                date_str = f"tau_{tau}"
                portfolio_id = f"P{k}_VW"

                try:
                    diagnostic = compute_turnover_diagnostic(
                        date=date_str,
                        portfolio_id=portfolio_id,
                        ids_t=curr_ids,
                        weights_t=raw_vw,
                        ids_t_minus_1=prev_ids_vw,
                        weights_t_minus_1=prev_weights_vw_nonzero,
                        turnover_value=turn_vw,
                        returns=None,
                        cohort_id=None
                    )
                    state.logger.log(diagnostic)
                except Exception:
                    pass
        else:
            # First time - mark as seen
            state.prev_seen_vw[cohort, k0] = True

            # === TRACKING LOGIC FOR FIRST OBSERVATION ===
            if state.logger is not None and TRACKING_AVAILABLE:
                date_str = f"tau_{tau}"
                portfolio_id = f"P{k}_VW"

                try:
                    diagnostic = compute_turnover_diagnostic(
                        date=date_str,
                        portfolio_id=portfolio_id,
                        ids_t=curr_ids,
                        weights_t=raw_vw,
                        ids_t_minus_1=np.array([]),
                        weights_t_minus_1=np.array([]),
                        turnover_value=np.nan,
                        returns=None,
                        cohort_id=None
                    )
                    state.logger.log(diagnostic)
                except Exception:
                    pass

        # ===== UPDATE STATE WITH SCALED WEIGHTS =====
        curr_ptf_scaled = weights_scaled_df[weights_scaled_df['ptf_rank'] == k].copy()

        if not curr_ptf_scaled.empty:
            # Reset previous weights to zero
            state.prev_scaled_ew[cohort, k0, :] = 0.0
            state.prev_scaled_vw[cohort, k0, :] = 0.0

            # Get IDs and positions for scaled weights
            scaled_ids = curr_ptf_scaled['ID'].values
            scaled_pos = id_to_pos.loc[scaled_ids].values.astype(np.int64)

            # Store new scaled weights
            state.prev_scaled_ew[cohort, k0, scaled_pos] = curr_ptf_scaled['eweights'].values
            state.prev_scaled_vw[cohort, k0, scaled_pos] = curr_ptf_scaled['vweights'].values

            # Store sums for next turnover calculation
            state.prev_sum_ew[cohort, k0] = curr_ptf_scaled['eweights'].sum()
            state.prev_sum_vw[cohort, k0] = curr_ptf_scaled['vweights'].sum()


# =============================================================================
# Initialization
# =============================================================================

def compute_nonstaggered_turnover(weights_df: pd.DataFrame,
                                 weights_scaled_df: pd.DataFrame,
                                 prev_weights: dict,
                                 tot_nport: int,
                                 is_first: bool = False) -> tuple:
    """
    Compute turnover for non-staggered rebalancing at a rebalancing date.
    
    This function compares current weights with previous scaled weights
    to compute turnover for each portfolio.
    
    Parameters
    ----------
    weights_df : pd.DataFrame
        Current weights with columns: ID, ptf_rank, eweights, vweights
    weights_scaled_df : pd.DataFrame
        Scaled weights for next period
    prev_weights : dict
        Previous scaled weights: {'prev_ew': {k: {id: w}}, 'prev_vw': {k: {id: w}}}
    tot_nport : int
        Total number of portfolios
    is_first : bool, default=False
        Whether this is the first rebalancing (no previous weights)
        
    Returns
    -------
    tuple
        (ew_turn_array, vw_turn_array, updated_prev_weights)
        - ew_turn_array: np.ndarray of shape (tot_nport,) with EW turnover
        - vw_turn_array: np.ndarray of shape (tot_nport,) with VW turnover
        - updated_prev_weights: dict with updated previous weights for next period
    """
    # Initialize turnover arrays
    ew_turn = np.full(tot_nport, np.nan)
    vw_turn = np.full(tot_nport, np.nan)
    
    if not is_first:
        # Compute turnover for each portfolio
        for k in range(1, tot_nport + 1):
            k0 = k - 1  # Zero-based index
            
            # Filter current portfolio
            curr_ptf = weights_df[weights_df['ptf_rank'] == k]
            
            if not curr_ptf.empty:
                # Current weights as dictionaries
                curr_ew = dict(zip(curr_ptf['ID'], curr_ptf['eweights']))
                curr_vw = dict(zip(curr_ptf['ID'], curr_ptf['vweights']))
                
                # Previous weights
                prev_ew = prev_weights['prev_ew'].get(k, {})
                prev_vw = prev_weights['prev_vw'].get(k, {})
                
                # Union of all asset IDs
                all_ids = set(curr_ew.keys()) | set(prev_ew.keys())
                
                # Compute turnover = sum(|current - previous|) / 2
                ew_turn[k0] = sum(
                    abs(curr_ew.get(id_, 0.0) - prev_ew.get(id_, 0.0))
                    for id_ in all_ids
                ) / 2.0
                
                vw_turn[k0] = sum(
                    abs(curr_vw.get(id_, 0.0) - prev_vw.get(id_, 0.0))
                    for id_ in all_ids
                ) / 2.0
    
    # Update previous weights with scaled weights for next rebalancing
    updated_prev = {'prev_ew': {}, 'prev_vw': {}}
    
    for k in range(1, tot_nport + 1):
        # Filter scaled weights for portfolio k
        curr_ptf_scaled = weights_scaled_df[weights_scaled_df['ptf_rank'] == k]
        
        if not curr_ptf_scaled.empty:
            updated_prev['prev_ew'][k] = dict(
                zip(curr_ptf_scaled['ID'], curr_ptf_scaled['eweights'])
            )
            updated_prev['prev_vw'][k] = dict(
                zip(curr_ptf_scaled['ID'], curr_ptf_scaled['vweights'])
            )
    
    return ew_turn, vw_turn, updated_prev


# =============================================================================
# Finalization
# =============================================================================
def finalize_turnover(state: TurnoverState, datelist: list, ptf_labels: list,
                     use_nanmean: bool = True):
    """
    Finalize turnover computation: handle last-row liquidation and average across cohorts.

    For the last time period, we assume portfolio liquidation (turnover = sum of weights).
    For staggered rebalancing, we also average turnover across overlapping cohorts.

    Parameters
    ----------
    state : TurnoverState
        Turnover state object
    datelist : list
        List of dates
    ptf_labels : list
        Portfolio labels (e.g., ['Q1', 'Q2', ...])
    use_nanmean : bool, default=True
        If True, use np.nanmean for averaging (ignores NaN values).
        If False, use np.mean (propagates NaN, matches old package behavior).

    Returns
    -------
    tuple
        (ew_turnover_df, vw_turnover_df) - DataFrames with turnover metrics
    """
    TM = len(datelist)
    hor = state.hor
    tot_nport = len(ptf_labels)
    tau_last = TM - 1  # Last time index
    
    # ===== LAST-ROW LIQUIDATION =====
    # Assume all positions are liquidated at the end
    if state.is_staggered:
        for c in range(hor):
            for k in range(tot_nport):
                if state.prev_seen_ew[c, k]:
                    turn_ew_liquidation = state.prev_sum_ew[c, k]
                    state.ew_turn_ea[tau_last, c, k] = turn_ew_liquidation

                    # === TRACKING LOGIC FOR LIQUIDATION ===
                    if state.logger is not None and TRACKING_AVAILABLE:
                        try:
                            # Get previous IDs (non-zero weights)
                            prev_vec = state.prev_scaled_ew[c, k, :]
                            prev_ids = np.where(prev_vec > 0)[0]
                            prev_weights = prev_vec[prev_ids]

                            diagnostic = compute_turnover_diagnostic(
                                date=f"tau_{tau_last}_LIQUIDATION",
                                portfolio_id=f"P{k+1}",
                                ids_t=np.array([]),  # No current holdings
                                weights_t=np.array([]),
                                ids_t_minus_1=prev_ids,
                                weights_t_minus_1=prev_weights,
                                turnover_value=turn_ew_liquidation,
                                returns=None,
                                cohort_id=c
                            )
                            state.logger.log(diagnostic)
                        except Exception:
                            pass

                if state.prev_seen_vw[c, k]:
                    turn_vw_liquidation = state.prev_sum_vw[c, k]
                    state.vw_turn_ea[tau_last, c, k] = turn_vw_liquidation

                    # === TRACKING LOGIC FOR VW LIQUIDATION ===
                    if state.logger is not None and TRACKING_AVAILABLE:
                        try:
                            prev_vec = state.prev_scaled_vw[c, k, :]
                            prev_ids = np.where(prev_vec > 0)[0]
                            prev_weights = prev_vec[prev_ids]

                            diagnostic = compute_turnover_diagnostic(
                                date=f"tau_{tau_last}_LIQUIDATION",
                                portfolio_id=f"P{k+1}_VW",
                                ids_t=np.array([]),
                                weights_t=np.array([]),
                                ids_t_minus_1=prev_ids,
                                weights_t_minus_1=prev_weights,
                                turnover_value=turn_vw_liquidation,
                                returns=None,
                                cohort_id=c
                            )
                            state.logger.log(diagnostic)
                        except Exception:
                            pass
    else:
        # Non-staggered: single cohort (index 0)
        for k in range(tot_nport):
            if state.prev_seen_ew[0, k]:
                turn_ew_liquidation = state.prev_sum_ew[0, k]
                state.ew_turn_ea[tau_last, k] = turn_ew_liquidation

                # === TRACKING LOGIC FOR LIQUIDATION ===
                if state.logger is not None and TRACKING_AVAILABLE:
                    try:
                        prev_vec = state.prev_scaled_ew[0, k, :]
                        prev_ids = np.where(prev_vec > 0)[0]
                        prev_weights = prev_vec[prev_ids]

                        diagnostic = compute_turnover_diagnostic(
                            date=f"tau_{tau_last}_LIQUIDATION",
                            portfolio_id=f"P{k+1}",
                            ids_t=np.array([]),
                            weights_t=np.array([]),
                            ids_t_minus_1=prev_ids,
                            weights_t_minus_1=prev_weights,
                            turnover_value=turn_ew_liquidation,
                            returns=None,
                            cohort_id=None
                        )
                        state.logger.log(diagnostic)
                    except Exception:
                        pass

            if state.prev_seen_vw[0, k]:
                turn_vw_liquidation = state.prev_sum_vw[0, k]
                state.vw_turn_ea[tau_last, k] = turn_vw_liquidation

                # === TRACKING LOGIC FOR VW LIQUIDATION ===
                if state.logger is not None and TRACKING_AVAILABLE:
                    try:
                        prev_vec = state.prev_scaled_vw[0, k, :]
                        prev_ids = np.where(prev_vec > 0)[0]
                        prev_weights = prev_vec[prev_ids]

                        diagnostic = compute_turnover_diagnostic(
                            date=f"tau_{tau_last}_LIQUIDATION",
                            portfolio_id=f"P{k+1}_VW",
                            ids_t=np.array([]),
                            weights_t=np.array([]),
                            ids_t_minus_1=prev_ids,
                            weights_t_minus_1=prev_weights,
                            turnover_value=turn_vw_liquidation,
                            returns=None,
                            cohort_id=None
                        )
                        state.logger.log(diagnostic)
                    except Exception:
                        pass
    
    # ===== AVERAGE ACROSS COHORTS (STAGGERED ONLY) =====
    if state.is_staggered:
        # Ignore errors from NaN slices
        with np.errstate(invalid='ignore'):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Mean of empty slice")

                # Average over cohort dimension (axis=1)
                # Skip first row (tau=0) as it has no turnover
                if use_nanmean:
                    # New behavior: ignore NaN values when averaging
                    ew_mean_over_h = np.nanmean(state.ew_turn_ea[1:, :, :], axis=1)
                    vw_mean_over_h = np.nanmean(state.vw_turn_ea[1:, :, :], axis=1)
                else:
                    # Old behavior: propagate NaN values (backward compatibility)
                    ew_mean_over_h = np.mean(state.ew_turn_ea[1:, :, :], axis=1)
                    vw_mean_over_h = np.mean(state.vw_turn_ea[1:, :, :], axis=1)
        
        # Ensure fully-empty rows stay as NaN (not 0 from nanmean of all-NaN)
        ew_mean_over_h = np.where(
            np.all(np.isnan(ew_mean_over_h), axis=1, keepdims=True),
            np.nan,
            ew_mean_over_h
        )
        vw_mean_over_h = np.where(
            np.all(np.isnan(vw_mean_over_h), axis=1, keepdims=True),
            np.nan,
            vw_mean_over_h
        )
        
        # Create DataFrames (skip first date)
        ew_turnover_df = pd.DataFrame(
            ew_mean_over_h, index=datelist[1:], columns=ptf_labels
        )
        vw_turnover_df = pd.DataFrame(
            vw_mean_over_h, index=datelist[1:], columns=ptf_labels
        )

        # ===== SHIFT(1) ALIGNMENT =====
        # Shift turnover by 1 so that turnover[t] represents the cost incurred
        # to generate return[t]. Without shift, turnover at index t is the cost
        # incurred at formation date t-1. After shift(1):
        # - turnover[t] = cost of entering positions that generate returns at t
        # - First row becomes NaN (warmup period - no previous to compare)
        ew_turnover_df = ew_turnover_df.shift(1)
        vw_turnover_df = vw_turnover_df.shift(1)

        return ew_turnover_df, vw_turnover_df
    else:
        # For non-staggered, return the turnover arrays directly as DataFrames
        ew_turnover_df = pd.DataFrame(
            state.ew_turn_ea[1:, :], index=datelist[1:], columns=ptf_labels
        )
        vw_turnover_df = pd.DataFrame(
            state.vw_turn_ea[1:, :], index=datelist[1:], columns=ptf_labels
        )

        # ===== SHIFT(1) ALIGNMENT =====
        # Same logic as staggered: shift so turnover[t] = cost for return[t]
        ew_turnover_df = ew_turnover_df.shift(1)
        vw_turnover_df = vw_turnover_df.shift(1)

        return ew_turnover_df, vw_turnover_df