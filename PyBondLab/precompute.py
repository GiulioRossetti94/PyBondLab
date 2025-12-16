# -*- coding: utf-8 -*-
"""
Created: 03-11-2025
Last modified: 03-11-2025

@authors: Giulio Rossetti

precomputation of formation dates, ranks, and weights
for fast portfolio formation.

"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Tuple
import warnings
import numpy as np
import pandas as pd

# Use numba-optimized versions for performance (2-5x faster)
from .utils_optimized import (
    compute_thresholds_optimized as compute_thresholds,
    assign_bond_bins_optimized as assign_bond_bins,
    double_sort_uncond_optimized as double_sort_uncond,
    double_sort_cond_optimized as double_sort_cond,
    create_subset_mask,
)
from .constants import ColumnNames

# =============================================================================
# Precomputation Data Container
# =============================================================================
@dataclass
class PrecomputedData:
    """
    Container for precomputed portfolio formation data.
    
    This stores all the data needed for fast portfolio formation,
    computed once upfront to avoid repeated filtering and ranking.
    
    Attributes
    ----------
    It0 : dict
        Formation-date data: {date: DataFrame}
    It1 : dict
        Return-period data (EA): {date: DataFrame}
    It1m : dict
        Dynamic weighting data: {date: DataFrame}
    ranks_map : dict
        Portfolio ranks: {date: Series(ID -> rank)}
    vw_map_t0 : dict
        Value weights at formation: {date: Series(ID -> VW)}
    vw_map_t1m : dict
        Value weights at t+h-1: {date: Series(ID -> VW)}
    It2 : dict
        Return-period data (EP): {date: DataFrame}
    """
    
    It0: Dict[pd.Timestamp, pd.DataFrame]
    It1: Dict[pd.Timestamp, pd.DataFrame]
    It1m: Dict[pd.Timestamp, pd.DataFrame]
    ranks_map: Dict[pd.Timestamp, pd.Series]
    vw_map_t0: Dict[pd.Timestamp, pd.Series]
    vw_map_t1m: Dict[pd.Timestamp, pd.Series]
    It2: Dict[pd.Timestamp, pd.DataFrame]
# =============================================================================
# Precomputation Builder
# =============================================================================
class PrecomputeBuilder:
    """
    Builder class for creating precomputed data.
    """
    
    def __init__(self, strategy_formation):
        """
        Initialize builder with strategy formation object.
        
        Parameters
        ----------
        strategy_formation : StrategyFormation
            The main strategy formation object
        """
        self.sf = strategy_formation
    
    def build(
        self,
        tab: pd.DataFrame,
        tab_raw: pd.DataFrame,
        datelist: List[pd.Timestamp],
        sort_var: str,
        sort_var2: Optional[str],
        use_double_sort: bool,
        how: str,
        adj: Optional[str],
        ret_var: str,
        nport: int,
        nport2: Optional[int],
        breakpoints: Optional[List[float]] = None,
        breakpoints2: Optional[List[float]] = None,
        cached_precomp: Optional[Dict] = None
    ) -> PrecomputedData:
        """
        Build all precomputed data.

        This orchestrates three main precomputation steps:
        1. Formation-date data (It0) with ranks and weights
        2. Return-period data (It1 for EA, It2 for EP)
        3. Dynamic weighting data (It1m)

        Parameters
        ----------
        tab : pd.DataFrame
            Processed bond data
        tab_raw : pd.DataFrame
            Raw bond data
        datelist : list of pd.Timestamp
            List of dates to process
        sort_var : str
            Primary sorting variable
        sort_var2 : str, optional
            Secondary sorting variable
        use_double_sort : bool
            Whether using double sorting
        how : str
            Double sort method ('conditional' or 'unconditional')
        adj : str, optional
            Filter adjustment type
        ret_var : str
            Return variable name
        nport : int
            Number of primary portfolios
        nport2 : int, optional
            Number of secondary portfolios
        breakpoints : list of float, optional
            Custom percentile breakpoints for primary sort (e.g., [30, 70] for terciles)
        breakpoints2 : list of float, optional
            Custom percentile breakpoints for secondary sort
        cached_precomp : dict, optional
            Cached precomputed data from a previous run with same (rating, filt, retcol).
            Contains 'It0' and 'vw_map_t0' that can be reused.

        Returns
        -------
        PrecomputedData
            All precomputed views
        """
        # Step 0: Get breakpoint universe functions if any
        subset_func, subset_func2 = self._get_subset_functions(use_double_sort)

        # OPTIMIZATION: Pre-index data by date (avoids repeated boolean filtering)
        # This creates dictionary lookups instead of filtering tab[tab['date'] == date]
        tab_by_date = self._create_date_index(tab, datelist)
        tab_raw_by_date = self._create_date_index(tab_raw, datelist)

        # Option 6: Check if we can reuse cached formation data
        if cached_precomp is not None and 'It0' in cached_precomp:
            # Reuse cached It0 and vw_map_t0 (independent of hp/nport)
            It0 = cached_precomp['It0']
            vw_map_t0 = cached_precomp['vw_map_t0']

            # Recompute ranks using cached It0 (ranks depend on nport)
            ranks_map = self._recompute_ranks_from_cached_It0(
                It0, datelist, sort_var, sort_var2, use_double_sort,
                how, nport, nport2, subset_func, subset_func2,
                breakpoints, breakpoints2
            )
        else:
            # Step 1: Precompute formation data (filters, ranks, VW at t)
            # this returns:
            # pre_It0: formation data (filtered data at time t)
            # ranks_map: rank mappings (rank of each bond at time t based on sorting vars)
            # vw_map_t0: volume-weighted mappings (VW of each bond at time t)
            It0, ranks_map, vw_map_t0 = self._precompute_formation_data(
                tab, datelist, sort_var, sort_var2, use_double_sort,
                how, adj, ret_var, nport, nport2, subset_func, subset_func2,
                breakpoints, breakpoints2,
                tab_by_date=tab_by_date  # Pass pre-indexed data
            )

        # Step 2: Precompute return data (EA and EP paths)
        # Check if we can reuse cached return data (shared across signals in batch mode)
        if cached_precomp is not None and 'It1' in cached_precomp:
            # Reuse cached return data (independent of signal/sort_var)
            It1 = cached_precomp['It1']
            It2 = cached_precomp.get('It2', {})
        else:
            # this returns:
            # pre_It1: EA return data (unadjusted returns at time t)
            # pre_It2: EP return data (adjusted returns at time t based on adj)
            It1, It2 = self._precompute_return_data(
                tab, tab_raw, datelist, ret_var, adj,
                tab_by_date=tab_by_date,  # Pass pre-indexed data
                tab_raw_by_date=tab_raw_by_date
            )

        # Step 3: Precompute dynamic weighting data (VW at t+h-1)
        # Check if we can reuse cached dynamic weights (shared across signals)
        if cached_precomp is not None and 'It1m' in cached_precomp:
            # Reuse cached dynamic weights (independent of signal/sort_var)
            It1m = cached_precomp['It1m']
            vw_map_t1m = cached_precomp['vw_map_t1m']
        else:
            # this returns:
            # pre_It1m: data at t+h-1 with valid VW
            It1m, vw_map_t1m = self._precompute_dynamic_weights(
                tab, datelist,
                tab_by_date=tab_by_date  # Pass pre-indexed data
            )

        return PrecomputedData(
            It0=It0,
            It1=It1,
            It1m=It1m,
            ranks_map=ranks_map,
            vw_map_t0=vw_map_t0,
            vw_map_t1m=vw_map_t1m,
            It2=It2
        )

    def _create_date_index(
        self,
        tab: pd.DataFrame,
        datelist: List[pd.Timestamp]
    ) -> Dict[pd.Timestamp, pd.DataFrame]:
        """
        Pre-index data by date for faster lookups.

        Instead of repeatedly filtering tab[tab['date'] == date], we create
        a dictionary mapping date -> DataFrame once upfront.

        Parameters
        ----------
        tab : pd.DataFrame
            Bond panel data with 'date' column
        datelist : list of pd.Timestamp
            List of dates to index

        Returns
        -------
        Dict[pd.Timestamp, pd.DataFrame]
            Dictionary mapping each date to its subset of data
        """
        # Use groupby for efficient date indexing
        # This is faster than repeated boolean filtering
        date_col = ColumnNames.DATE if ColumnNames.DATE in tab.columns else 'date'

        # Group by date and convert to dictionary
        grouped = tab.groupby(date_col, sort=False)

        # Create dictionary with only the dates we need
        date_index = {}
        for date in datelist:
            if date in grouped.groups:
                date_index[date] = grouped.get_group(date)
            else:
                # Empty DataFrame with same columns
                date_index[date] = tab.iloc[:0]

        return date_index

    # -------------------------------------------------------------------------
    # Helpers for pre computation 
    # -------------------------------------------------------------------------     
    def _get_subset_functions(self, use_double_sort: bool) -> Tuple[Optional[Callable], Optional[Callable]]:
        """
        Extract breakpoint universe functions from strategy.
        
        Parameters
        ----------
        use_double_sort : bool
            Whether using double sorting
        
        Returns
        -------
        tuple
            (subset_func, subset_func2) - functions for breakpoint computation
        """
        subset_func = getattr(self.sf.strategy, 'breakpoint_universe_func', None)
        subset_func2 = (getattr(self.sf.strategy, 'breakpoint_universe_func2', None) 
                        if use_double_sort else None)
        return subset_func, subset_func2
    
    def _extract_vw(self, sub: pd.DataFrame) -> pd.Series:
        """
        Extract VW (value weights) from dataframe, handling duplicates.
        
        If the dataframe has duplicate indices, take the last VW for each ID.
        Otherwise, use the VW directly.
        
        Parameters
        ----------
        sub : pd.DataFrame
            Data with ID and VW columns
        
        Returns
        -------
        pd.Series
            VW indexed by ID
        """
        if sub.index.has_duplicates:
            print("Warning: Duplicate IDs found when extracting VW. Taking last occurrence.")
            return sub.groupby('ID', as_index=True)['VW'].last()
        else:
            return sub.set_index('ID')['VW']
        
    def _compute_ranks(
        self,
        sub: pd.DataFrame,
        sort_var: str,
        sort_var2: Optional[str],
        nport: int,
        nport2: Optional[int],
        use_double_sort: bool,
        how: str,
        subset_func: Optional[Callable],
        subset_func2: Optional[Callable],
        breakpoints: Optional[List[float]] = None,
        breakpoints2: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Compute portfolio ranks for bonds on a single date.

        This handles both single and double sorting with optional custom
        breakpoint universes, as well as within-firm sorting.

        Parameters
        ----------
        sub : pd.DataFrame
            Filtered bond data for this date
        sort_var : str
            Primary sorting variable
        sort_var2 : str, optional
            Secondary sorting variable
        nport : int
            Number of primary portfolios
        nport2 : int, optional
            Number of secondary portfolios
        use_double_sort : bool
            Whether to perform double sorting
        how : str
            Double sort method: 'conditional' or 'unconditional'
        subset_func : callable, optional
            Function to create subset mask for primary breakpoints
        subset_func2 : callable, optional
            Function to create subset mask for secondary breakpoints
        breakpoints : list of float, optional
            Custom percentile breakpoints for primary sort (e.g., [30, 70])
        breakpoints2 : list of float, optional
            Custom percentile breakpoints for secondary sort

        Returns
        -------
        np.ndarray
            Portfolio ranks (1 to tot_nport)

        Examples
        --------
        Single sort:
        >>> ranks = self._compute_ranks(
        ...     bonds_df, 'duration', None, 10, None, False, 'unconditional', None, None
        ... )
        >>> # Returns array with values 1-10

        Double sort:
        >>> ranks = self._compute_ranks(
        ...     bonds_df, 'duration', 'size', 5, 5, True, 'conditional', None, None
        ... )
        >>> # Returns array with values 1-25 (5 x 5 portfolios)
        """
        # Check if this is WithinFirmSort strategy
        if self.sf.strategy.__strategy_name__ == "Within-Firm Sort":
            # Use within-firm sorting logic
            from .utils_within_firm import compute_within_firm_portfolios

            firm_id_col = getattr(self.sf.strategy, 'firm_id_col', 'PERMNO')
            rating_bins = getattr(self.sf.strategy, 'rating_bins', [-np.inf, 7, 10, np.inf])
            min_bonds = getattr(self.sf.strategy, 'min_bonds_per_firm', 2)

            # Get the current date from sub (all rows should have same date)
            current_date = sub[ColumnNames.DATE].iloc[0] if len(sub) > 0 else None

            if current_date is None:
                return np.array([], dtype=np.int32)

            # sub already has firm_id_col from required columns
            # Call within-firm portfolio computation directly on sub
            bond_assignments, _ = compute_within_firm_portfolios(
                data=sub,
                signal_col=sort_var,
                return_col=ColumnNames.RETURN,  # Placeholder, not used in ranking
                weight_col=ColumnNames.VALUE_WEIGHT,
                firm_id_col=firm_id_col,
                rating_col=ColumnNames.RATING,
                rating_bins=rating_bins,
                min_bonds_per_firm=min_bonds
            )

            # Create ranks array for all bonds in sub
            # Bonds not assigned to portfolios get rank 0
            ranks = np.zeros(len(sub), dtype=np.int32)

            # Map bond assignments back to original indices
            if not bond_assignments.empty:
                # Create a mapping from ID to rank
                id_to_rank = dict(zip(
                    bond_assignments[ColumnNames.ID],
                    bond_assignments[ColumnNames.PORTFOLIO_RANK]
                ))

                # Assign ranks based on ID in sub
                for i, bond_id in enumerate(sub[ColumnNames.ID].values):
                    ranks[i] = id_to_rank.get(bond_id, 0)

            return ranks

        # Standard sorting logic (existing code)
        # Create subset masks for custom breakpoint computation
        subset_mask = create_subset_mask(sub, subset_func) if subset_func else None
        subset_mask2 = create_subset_mask(sub, subset_func2) if subset_func2 else None

        # Primary sort: compute thresholds and assign ranks
        # Use custom breakpoints if provided, otherwise use nport for even percentiles
        bp1 = breakpoints if breakpoints is not None else nport
        th1 = compute_thresholds(sub, sort_var, bp1, subset=subset_mask)
        idx1 = assign_bond_bins(sub[sort_var].to_numpy(), th1, nport)

        # Ensure ranks start at 1 (not 0)
        if np.min(idx1) == 0:
            idx1 = idx1 + 1

        # Double sort (if requested)
        if use_double_sort:
            if how == 'unconditional':
                # Independent sorting on both variables
                # Use custom breakpoints2 if provided, otherwise use nport2
                bp2 = breakpoints2 if breakpoints2 is not None else nport2
                th2 = compute_thresholds(sub, sort_var2, bp2, subset=subset_mask2)
                idx2 = assign_bond_bins(sub[sort_var2].to_numpy(), th2, nport2)
                if np.min(idx2) == 0:
                    idx2 = idx2 + 1
                ranks = double_sort_uncond(idx1, idx2, nport, nport2)
            elif how == 'conditional':
                # Sort second variable within first variable bins
                # Warn if user provided breakpoints2 or subset_func2 - these are ignored
                if breakpoints2 is not None:
                    warnings.warn(
                        "breakpoints2 is ignored in conditional double sort. "
                        "Conditional sorting computes even percentiles within each primary bin.",
                        UserWarning
                    )
                if subset_mask2 is not None:
                    warnings.warn(
                        "breakpoint_universe_func2 is ignored in conditional double sort. "
                        "Conditional sorting uses all assets within each primary bin.",
                        UserWarning
                    )
                ranks = double_sort_cond(sub[sort_var2], idx1, nport, nport2)
            else:
                raise ValueError(f"Unknown double sort method: {how}")
        else:
            ranks = idx1

        return ranks

    def _recompute_ranks_from_cached_It0(
        self,
        It0: Dict[pd.Timestamp, pd.DataFrame],
        datelist: List[pd.Timestamp],
        sort_var: str,
        sort_var2: Optional[str],
        use_double_sort: bool,
        how: str,
        nport: int,
        nport2: Optional[int],
        subset_func: Optional[Callable],
        subset_func2: Optional[Callable],
        breakpoints: Optional[List[float]] = None,
        breakpoints2: Optional[List[float]] = None
    ) -> Dict[pd.Timestamp, pd.Series]:
        """
        Recompute ranks using cached It0 data.

        This is used in Option 6 batching when we have cached filtered data (It0)
        but need to recompute ranks with different nport/breakpoints.

        Parameters
        ----------
        It0 : dict
            Cached formation-date data: {date: DataFrame}
        datelist : list of pd.Timestamp
            Dates to process
        sort_var : str
            Primary sorting variable
        sort_var2 : str, optional
            Secondary sorting variable
        use_double_sort : bool
            Whether using double sorting
        how : str
            Double sort method
        nport : int
            Number of primary portfolios
        nport2 : int, optional
            Number of secondary portfolios
        subset_func : callable, optional
            Breakpoint universe function for primary sort
        subset_func2 : callable, optional
            Breakpoint universe function for secondary sort
        breakpoints : list of float, optional
            Custom percentile breakpoints for primary sort
        breakpoints2 : list of float, optional
            Custom percentile breakpoints for secondary sort

        Returns
        -------
        Dict[pd.Timestamp, pd.Series]
            Portfolio ranks indexed by ID for each date
        """
        ranks_map = {}

        for date_t in datelist:
            sub = It0.get(date_t, pd.DataFrame())

            if sub.empty:
                ranks_map[date_t] = pd.Series(dtype="Int64")
                continue

            # Compute ranks using existing method
            ranks = self._compute_ranks(
                sub, sort_var, sort_var2, nport, nport2,
                use_double_sort, how, subset_func, subset_func2,
                breakpoints, breakpoints2
            )

            # Store ranks as Series indexed by ID
            ranks_map[date_t] = pd.Series(
                ranks,
                index=sub['ID'].values,
                dtype="Int64"
            )

        return ranks_map

    def _precompute_formation_data(
        self,
        tab: pd.DataFrame,
        datelist: List[pd.Timestamp],
        sort_var: str,
        sort_var2: Optional[str],
        use_double_sort: bool,
        how: str,
        adj: Optional[str],
        ret_var: str,
        nport: int,
        nport2: Optional[int],
        subset_func: Optional[Callable],
        subset_func2: Optional[Callable],
        breakpoints: Optional[List[float]] = None,
        breakpoints2: Optional[List[float]] = None,
        tab_by_date: Optional[Dict[pd.Timestamp, pd.DataFrame]] = None
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Precompute formation-date data including ranks and VW.

        For each date in datelist, this function:
        1. Filters bonds by rating, characteristics, and universe matching
        2. Computes portfolio ranks based on sorting variables
        3. Extracts value weights (VW) for later portfolio weighting

        Parameters
        ----------
        tab : pd.DataFrame
            Full bond panel data
        datelist : list of pd.Timestamp
            Dates to process
        sort_var : str
            Primary sorting variable
        sort_var2 : str, optional
            Secondary sorting variable
        use_double_sort : bool
            Whether using double sorting
        how : str
            Double sort method
        adj : str, optional
            Adjustment type for EP
        ret_var : str
            Return variable name
        nport : int
            Number of primary portfolios
        nport2 : int, optional
            Number of secondary portfolios
        subset_func : callable, optional
            Breakpoint universe function for primary sort
        subset_func2 : callable, optional
            Breakpoint universe function for secondary sort
        breakpoints : list of float, optional
            Custom percentile breakpoints for primary sort
        breakpoints2 : list of float, optional
            Custom percentile breakpoints for secondary sort
        tab_by_date : dict, optional
            Pre-indexed data by date for faster lookups

        Returns
        -------
        tuple
            (pre_It0, ranks_map, vw_map_t0) - formation data, ranks, and weights
        """
        It0, ranks_map, vw_map_t0 = {}, {}, {}

        for date_t in datelist:
            # Filter data (now uses unified filter methods)
            # Pass pre-indexed data if available for faster lookup
            date_sub = tab_by_date.get(date_t) if tab_by_date else None
            sub = self.sf.filter_by_rating(tab, date_t, sort_var, sort_var2, date_sub=date_sub)

            if self.sf.subset_filter:
                sub = self.sf.filter_by_char(sub, date_t, sort_var, sort_var2)

            if adj in ['trim', 'bounce', 'price']:
                sub = self.sf.filter_by_universe_matching(sub, adj, ret_var)

            # Store filtered data
            It0[date_t] = sub

            # Handle empty data
            if sub.empty:
                ranks_map[date_t] = pd.Series(dtype="Int64")
                vw_map_t0[date_t] = pd.Series(dtype=float)
                continue

            # Compute ranks using extracted method
            ranks = self._compute_ranks(
                sub, sort_var, sort_var2, nport, nport2,
                use_double_sort, how, subset_func, subset_func2,
                breakpoints, breakpoints2
            )

            # Store ranks as Series indexed by ID
            # Use pandas Int64 (nullable integer) to preserve NaN values
            ranks_map[date_t] = pd.Series(
                ranks,
                index=sub['ID'].values,
                dtype="Int64"
            )

            # Extract and store VW
            vw_map_t0[date_t] = self._extract_vw(sub)

        return It0, ranks_map, vw_map_t0
    
    def _precompute_return_data(
        self,
        tab: pd.DataFrame,  # filtered data for EP
        tab_raw: pd.DataFrame,  # raw data for EA
        datelist: List[pd.Timestamp],
        ret_var: str,
        adj: Optional[str],
        tab_by_date: Optional[Dict[pd.Timestamp, pd.DataFrame]] = None,
        tab_raw_by_date: Optional[Dict[pd.Timestamp, pd.DataFrame]] = None
    ) -> Tuple[Dict, Dict]:
        """
        Precompute return data for EA and EP paths.

        EA (Ex-Ante): Always uses unadjusted returns
        EP (Ex-Post): Uses adjusted returns based on filter type (this is infeasible)

        Parameters
        ----------
        tab : pd.DataFrame
            Full bond panel data
        datelist : list of pd.Timestamp
            Dates to process
        ret_var : str
            Return variable name
        adj : str, optional
            Adjustment type ('wins', 'trim', 'price', 'bounce', or None)
        tab_by_date : dict, optional
            Pre-indexed tab data by date
        tab_raw_by_date : dict, optional
            Pre-indexed tab_raw data by date

        Returns
        -------
        tuple
            (pre_It1, pre_It2) - EA and EP return data by date
        """
        It1, It2 = {}, {}

        # Pre-index winsorized data if needed
        wdf_by_date = None
        if adj == 'wins':
            wdf = self.sf.filter_obj.data_winsorized_ex_post
            wdf_by_date = self._create_date_index(wdf, datelist)

        for date in datelist:
            # EA: always unadjusted returns
            # Use pre-indexed data if available
            if tab_raw_by_date and date in tab_raw_by_date:
                raw_sub = tab_raw_by_date[date]
                It1[date] = raw_sub[~raw_sub[ret_var].isna()]
            else:
                It1[date] = tab_raw[(tab_raw['date'] == date) & (~tab_raw[ret_var].isna())]

            # EP: adjusted returns based on filter type
            if adj == 'wins':
                ret_col_ep = f"{ret_var}_{adj}"  # 'ret_wins'
                if wdf_by_date and date in wdf_by_date:
                    wdf_sub = wdf_by_date[date]
                    It2[date] = wdf_sub[~wdf_sub[ret_col_ep].isna()]
                else:
                    wdf = self.sf.filter_obj.data_winsorized_ex_post
                    It2[date] = wdf[(wdf['date'] == date) & (~wdf[ret_col_ep].isna())]
            elif adj in ['trim', 'price', 'bounce']:
                ret_col_ep = f"{ret_var}_{adj}"
                if tab_by_date and date in tab_by_date:
                    tab_sub = tab_by_date[date]
                    It2[date] = tab_sub[~tab_sub[ret_col_ep].isna()]
                else:
                    It2[date] = tab[(tab['date'] == date) & (~tab[ret_col_ep].isna())]
            else:
                # No EP adjustment
                It2[date] = It1[date]

        return It1, It2

    def _precompute_dynamic_weights(
        self,
        tab: pd.DataFrame,
        datelist: List[pd.Timestamp],
        tab_by_date: Optional[Dict[pd.Timestamp, pd.DataFrame]] = None
    ):
        """
        Precompute dynamic weighting data (VW at t+h-1).

        For dynamic weighting, we need VW at the end of the holding period
        (t+h-1) rather than at portfolio formation (t).

        Parameters
        ----------
        tab : pd.DataFrame
            Full bond panel data
        datelist : list of pd.Timestamp
            Dates to process
        tab_by_date : dict, optional
            Pre-indexed data by date for faster lookups

        Returns
        -------
        tuple
            (pre_It1m, vw_map_t1m) - data and weights for dynamic weighting
        """
        It1m, vw_map_t1m = {}, {}

        for date in datelist:
            # Get data with valid VW
            # Use pre-indexed data if available
            if tab_by_date and date in tab_by_date:
                tab_sub = tab_by_date[date]
                It1m[date] = tab_sub[~tab_sub[ColumnNames.VALUE_WEIGHT].isna()]
            else:
                It1m[date] = tab[
                    (tab[ColumnNames.DATE] == date) &
                    (~tab[ColumnNames.VALUE_WEIGHT].isna())
                ]
            It1m_d = It1m[date]

            # Extract VW if data available
            if not It1m_d.empty:
                vw_map_t1m[date] = self._extract_vw(It1m_d)
            else:
                vw_map_t1m[date] = pd.Series(dtype=float)

        return It1m, vw_map_t1m
    
# =============================================================================
# Convenience Function
# =============================================================================
def build_precomputed_data(
    strategy_formation,
    tab: pd.DataFrame,
    tab_raw: pd.DataFrame,
    datelist: List[pd.Timestamp],
    sort_var: str,
    sort_var2: Optional[str],
    use_double_sort: bool,
    how: str,
    adj: Optional[str],
    ret_var: str,
    nport: int,
    nport2: Optional[int]
) -> PrecomputedData:
    """
    Convenience function to build precomputed data.
    
    This is the main entry point for precomputation. It creates
    a PrecomputeBuilder and uses it to build all necessary data.
    
    Parameters
    ----------
    strategy_formation : StrategyFormation
        Main strategy formation object
    ... (other parameters same as PrecomputeBuilder.build)
    
    Returns
    -------
    PrecomputedData
        All precomputed data: formation data, ranks, and weights
    
    Examples
    --------
    >>> precomp = build_precomputed_data(
    ...     sf, data, data_raw, dates, 'duration', None,
    ...     False, 'unconditional', None, 'ret', 10, None
    ... )
    >>> # Access precomputed ranks
    >>> ranks = precomp.ranks_map[date]
    """
    builder = PrecomputeBuilder(strategy_formation)
    return builder.build(
        tab, tab_raw, datelist, sort_var, sort_var2,
        use_double_sort, how, adj, ret_var, nport, nport2
    )

