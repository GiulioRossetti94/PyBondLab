# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:28:52 2024
Last modified: 03-11-2025

@authors: Giulio Rossetti & Alex Dickerson
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Union, Any, List
import warnings

import numpy as np
import pandas as pd

from .FilterClass import Filter
from .StrategyClass import *

from .config import (
    StrategyFormationConfig,
    DataConfig,
    FormationConfig,
    FilterConfig
)
from .constants import (
    Defaults,
    ColumnNames,
    RatingBounds,
    ValidationMessages,
    get_rating_bounds,
    get_portfolio_labels,
    get_signal_based_labels
)

from .utils import summarize_ranks, _get_rebalancing_dates
from .precompute import PrecomputedData, PrecomputeBuilder
from .utils_portfolio import (
    compute_portfolio_ranks,
    form_portfolio_single_period,
    apply_banding,
    calculate_qnew_vectorized
)

# from .iotools.PyBondLabResults import StrategyResults
from .results import build_strategy_results, build_formation_results


from .utils import (
    summarize_ranks,
    _get_rebalancing_dates,
)

# Turnover utils
from .utils_turnover import TurnoverManager

# Numba-optimized core functions
from .numba_core import (
    compute_portfolio_returns_single,
    compute_portfolio_weights_single,
    compute_scaled_weights_single,
    compute_characteristics_single,
    # Fast returns-only path
    compute_all_dates_returns_fast,
    compute_staggered_returns_fast,
    precompute_formation_ranks,
    build_rank_lookup_fast,
    align_ranks_for_returns_fast,
    align_ranks_staggered_fast,
    # Ultra-fast path (bypasses pandas)
    compute_ranks_all_dates_fast,
    compute_all_returns_ultrafast,
    compute_staggered_returns_ultrafast,
    build_vw_lookup_and_dynamic_weights,
    build_vw_lookup,
)

import statsmodels.api as sm
import matplotlib.pyplot as plt
from PyBondLab.data.WRDS import load

Number = Union[int, float]
SubsetFilter = Dict[str, Tuple[Number, Number]]

# Try to import numba for performance
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True)
    def _sum_min_prev_raw(prev_vec, pos, raw_w):
        s = 0.0
        m = pos.size
        for j in range(m):
            pv = prev_vec[pos[j]]
            rw = raw_w[j]
            s += pv if pv < rw else rw
        return s
else:
    def _sum_min_prev_raw(prev_vec, pos, raw_w):
        return float(np.minimum(prev_vec[pos], raw_w).sum())


# =============================================================================
# Precomputation Data Structure
# =============================================================================
@dataclass
class PrecomputedData:
    """Container for precomputed time-series data."""
    It0: Dict[pd.Timestamp, pd.DataFrame]   # Data at portfolio formation time
    It1: Dict[pd.Timestamp, pd.DataFrame]   # Data at return realization time
    It1m: Dict[pd.Timestamp, pd.DataFrame]  # Data for dynamic weights
    ranks_map: Dict[pd.Timestamp, pd.Series]  # Precomputed ranks
    vw_map_t0: Dict[pd.Timestamp, pd.Series]  # Value weights at t
    vw_map_t1m: Dict[pd.Timestamp, pd.Series]  # Value weights at t+h-1



# =============================================================================
# Main StrategyFormation Class
# =============================================================================
class StrategyFormation:
    """
    Form and analyze bond portfolios based on trading strategies.

    This is the main class for portfolio-based analysis. It handles:
    - Signal computation from strategies
    - Portfolio formation and ranking
    - Return calculations (equal and value weighted)
    - Optional: turnover, characteristics, filtering

    Parameters
    ----------
    data : pd.DataFrame
        Bond panel data with required columns: date, ID, ret, RATING_NUM, VW
    strategy : Strategy
        Trading strategy object (Momentum, LTreversal, SingleSort, etc.)
    config : StrategyFormationConfig, optional
        Configuration object. If not provided, default configuration is used.
        For backward compatibility, can also pass individual parameters as **kwargs

    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        config: Optional[StrategyFormationConfig] = None,
        **kwargs
    ):
        """Initialize StrategyFormation with data and strategy."""
        # Extract Option 6 caching parameter before passing to config
        self._cached_precomp = kwargs.pop('cached_precomp', None)

        # Store raw inputs
        self.data_raw = data.copy()
        self.strategy = strategy

        # Handle configuration
        if config is None:
            # Backward compatibility: create config from kwargs
            config = StrategyFormationConfig.from_legacy_params(**kwargs)

        # Handle configuration
        if config is None:
            # Backward compatibility: create config from kwargs
            config = StrategyFormationConfig.from_legacy_params(**kwargs)
        self.config = config

        # Extract configuration components for easier access
        self._extract_config()

        # Initialize state
        self._initialize_state()



        if self.verbose:
            self._print_initialization_summary()


    def _extract_config(self):
        """Extract configuration components into instance variables."""
        # Data configuration
        self.rating = self.config.data.rating
        self.subset_filter = self.config.data.subset_filter
        self.chars = self.config.data.chars

        # Formation configuration
        self.dynamic_weights = self.config.formation.dynamic_weights
        self.turnover = self.config.formation.compute_turnover
        self.save_idx = self.config.formation.save_idx
        self.banding_threshold = self.config.formation.banding_threshold
        self.verbose = self.config.formation.verbose

        # Filter configuration
        self.filters = self.config.filters.to_dict() if self.config.has_filters else None
        self.adj = self.config.filters.adj if self.config.has_filters else None

        # Strategy parameters (from strategy object)
        self.nport = self.strategy.num_portfolios
        self.hor = self.strategy.holding_period
        self.rebalance_frequency = self.strategy.rebalance_frequency
        self.rebalance_month = self.strategy.rebalance_month

    def _initialize_state(self):
        """Initialize instance state variables."""
        # For WithinFirmSort, always save portfolio indices (required for custom aggregation)
        if self.strategy.__strategy_name__ == "Within-Firm Sort":
            self.save_idx = True

        # Results containers
        self.results = None
        self.port_idx = {} if self.save_idx else None

        # Name for column naming (will be set after filters are applied)
        self.name = None

        # Turnover tracking
        if self.turnover:
            self.turnover_state = None

        # Banding tracking
        if self.banding_threshold is not None:
            self.lag_rank = {}

        # Characteristics tracking
        if self.chars:
            self.ew_ep_chars_dict = {}
            self.vw_ep_chars_dict = {}

    def _create_name(self, rating, strategy_name):
        """
        Create a descriptive name for the strategy including rating and strategy parameters.

        Parameters
        ----------
        rating : str or None
            Rating category ('NIG', 'IG', or None for 'ALL')
        strategy_name : str
            Strategy name string from strategy.str_name

        Returns
        -------
        str
            Combined name (e.g., 'NIG_3_3_1' or 'ALL_6_6_1')
        """
        if rating is None:
            return f"ALL_{strategy_name}"
        return f"{rating}_{strategy_name}"

    def _validate_data(self):
        """Validate input data has required columns."""
        required_cols = set(ColumnNames.REQUIRED)
        missing_cols = required_cols - set(self.data_raw.columns)

        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Required: {required_cols}"
            )

    def _apply_column_mapping(self, IDvar, DATEvar, RETvar, RATINGvar, VWvar, PRICEvar):
        """
        Apply column name mapping to standardize variable names.

        This method renames columns in self.data_raw to match expected column names,
        then re-runs validation and preparation.

        Parameters
        ----------
        IDvar, DATEvar, RETvar, RATINGvar, VWvar, PRICEvar : str or None
            Custom column names to map to standard names
        """
        # Build mapping dictionary
        column_mapping = {}

        if IDvar is not None and IDvar != 'ID':
            column_mapping[IDvar] = 'ID'
        if DATEvar is not None and DATEvar != 'date':
            column_mapping[DATEvar] = 'date'
        if RETvar is not None and RETvar != 'ret':
            column_mapping[RETvar] = 'ret'
        if RATINGvar is not None and RATINGvar != 'RATING_NUM':
            column_mapping[RATINGvar] = 'RATING_NUM'
        if VWvar is not None and VWvar != 'VW':
            column_mapping[VWvar] = 'VW'
        if PRICEvar is not None and PRICEvar != 'PRICE':
            column_mapping[PRICEvar] = 'PRICE'

        # Apply mapping if any columns need to be renamed
        if column_mapping:
            # Check if all source columns exist in raw data
            missing_cols = set(column_mapping.keys()) - set(self.data_raw.columns)
            if missing_cols:
                raise ValueError(
                    f"Specified column names not found in data: {missing_cols}. "
                    f"Available columns: {list(self.data_raw.columns)}"
                )

            # Rename columns in raw data
            self.data_raw = self.data_raw.rename(columns=column_mapping)

            if self.verbose:
                print(f"Mapped columns: {column_mapping}")

            # Re-validate and re-prepare data with new column names
            self._validate_data()
            self._prepare_data()


    def _prepare_data(self):
        """Prepare data: filtering, signal computation, and indexing."""
        # Start with raw data
        self.data = self.data_raw.copy()

        # this might introduce lookahead bias if done here
        # Apply rating filter. this is done later
        # if self.rating is not None:
        #     self._apply_rating_filter()

        # Apply subset filters: this is done later.
        # if self.subset_filter is not None:
        #     self._apply_subset_filters()

        # Compute strategy signal
        self.data = self.strategy.compute_signal(self.data)

        # Create base name for column naming
        strategy_str_name = self.strategy.str_name if hasattr(self.strategy, 'str_name') else 'strategy'
        self.name = self._create_name(self.rating, strategy_str_name)

        # Apply ex-post filters if requested
        if self.filters is not None:
            self._apply_filters()
            # Add filter name to the column name
            self.name += self.filter_obj.name_filt
            # Recompute signal for momentum-based strategies with adjusted returns
            self._recompute_signal_if_needed()
        # Prepare indices and IDs
        self._prepare_index_and_ids()

        # Build required columns list
        sort_var_main, sort_var2 = self._get_sort_vars()
        self.required_cols = self._build_required_columns(sort_var_main, sort_var2)

        # Validate date coverage for double sorts
        if sort_var2 is not None:
            self._validate_double_sort_date_coverage(sort_var_main, sort_var2)


    def _apply_rating_filter(self):
        """Apply rating filter to data."""
        if isinstance(self.rating, str):
            min_rating, max_rating = get_rating_bounds(self.rating)
        else:
            min_rating, max_rating = self.rating

        self.data = self.data[
            (self.data[ColumnNames.RATING] >= min_rating) &
            (self.data[ColumnNames.RATING] <= max_rating)
        ].copy()

        if self.verbose:
            print(f"Applied rating filter: {self.rating}")
            print(f"Remaining observations: {len(self.data)}")

    def _apply_subset_filters(self):
        """Apply characteristic-based subset filters."""
        for col, (min_val, max_val) in self.subset_filter.items():
            if col not in self.data.columns:
                raise ValueError(
                    ValidationMessages.MISSING_COLUMN.format(
                        col=col,
                        available=list(self.data.columns)
                    )
                )

            self.data = self.data[
                (self.data[col] >= min_val) &
                (self.data[col] <= max_val)
            ].copy()

            if self.verbose:
                print(f"Applied filter on {col}: [{min_val}, {max_val}]")
                print(f"Remaining observations: {len(self.data)}")

    def _apply_filters(self):
        """Apply ex-post filters (trim, winsorize, etc.)."""
        filter_obj = Filter(
            data=self.data,
            adj=self.filters['adj'],
            w=self.filters.get('level'),
            loc=self.filters.get('location'),
            percentile_breakpoints=self.filters.get('df_breakpoints'),
            price_threshold=self.filters.get('price_threshold', Defaults.PRICE_THRESHOLD)
        )

        self.data = filter_obj.apply_filters()
        # return also filter obj
        self.filter_obj = filter_obj

        if self.verbose:
            print(f"Applied filter: {self.filters['adj']}")

    def _recompute_signal_if_needed(self):
        """
        Recompute signal for priced-based strategies using adjusted returns.

        For Momentum and LTreversal strategies, the signal depends on past returns.
        When winsorization or other filters are applied, we need to recompute the
        signal using the adjusted returns.
        """
        strategy_name = self.strategy.__strategy_name__
        adj = self.adj

        # Only recompute for strategies that compute signal from returns
        if strategy_name not in ["MOMENTUM", "LT-REVERSAL"]:
            return

        # Check if sort_var contains 'signal' to confirm signal-based strategy
        sort_var = self.strategy.get_sort_var(adj)
        if 'signal' not in sort_var:
            return

        # Recompute signal using adjusted returns
        if strategy_name == "MOMENTUM":
            J = self.strategy.J
            skip = self.strategy.skip
            varname = f'ret_{adj}'
            signal_col = f'signal_{adj}'

            # Get NaN handling parameters from strategy
            no_gap = getattr(self.strategy, 'no_gap', False)
            fill_na = getattr(self.strategy, 'fill_na', False)
            drop_na = getattr(self.strategy, 'drop_na', False)

            # Ensure data is sorted
            self.data = self.data.sort_values(['ID', 'date'], ignore_index=True)

            # Create month index for gap detection
            if no_gap:
                self.data['month_idx'] = (
                    self.data['date'].dt.year * 12 + self.data['date'].dt.month
                )

            if drop_na:
                # Option B: Variable window to accumulate J valid (non-NaN) returns
                self.data['logret'] = np.log(self.data[varname] + 1)
                self.data['cumlogret'] = self.data.groupby('ID')['logret'].cumsum()
                self.data['cumvalid'] = self.data.groupby('ID')[varname].transform(
                    lambda x: x.notna().cumsum()
                )

                def compute_drop_na_signal(group):
                    """Compute signal using exactly J valid returns."""
                    group = group.copy()
                    n = len(group)
                    signal = np.full(n, np.nan)
                    cumlogret = group['cumlogret'].values
                    cumvalid = group['cumvalid'].values

                    for i in range(n):
                        if np.isnan(cumvalid[i]) or cumvalid[i] < J:
                            continue
                        target_cumvalid = cumvalid[i] - J
                        if target_cumvalid == 0:
                            signal[i] = cumlogret[i]
                        else:
                            for j in range(i - 1, -1, -1):
                                if cumvalid[j] == target_cumvalid:
                                    signal[i] = cumlogret[i] - cumlogret[j]
                                    break
                    group[signal_col] = signal
                    return group

                self.data = self.data.groupby('ID', group_keys=False).apply(compute_drop_na_signal)
                self.data[signal_col] = np.exp(self.data[signal_col]) - 1
                self.data.drop(columns=['cumlogret', 'cumvalid', 'logret'], inplace=True)

            elif fill_na:
                # Option A: Fixed window of J rows, NaN returns treated as 0%
                ret_col = self.data[varname].fillna(0)
                self.data['logret'] = np.log(ret_col + 1)
                self.data[signal_col] = (
                    self.data.groupby(['ID'], group_keys=False)['logret']
                    .rolling(J, min_periods=J)
                    .sum()
                    .values
                )
                self.data[signal_col] = np.exp(self.data[signal_col]) - 1
                self.data.drop(columns=['logret'], inplace=True)

            else:
                # Default: Standard rolling, NaN propagates
                self.data['logret'] = np.log(self.data[varname] + 1)
                self.data[signal_col] = (
                    self.data.groupby(['ID'], group_keys=False)['logret']
                    .rolling(J, min_periods=J)
                    .sum()
                    .values
                )
                self.data[signal_col] = np.exp(self.data[signal_col]) - 1
                self.data.drop(columns=['logret'], inplace=True)

            # Apply no_gap check: invalidate signal if months are not consecutive
            if no_gap:
                self.data['lmonth_idx'] = self.data.groupby('ID')['month_idx'].shift(J - 1)
                self.data['month_diff'] = self.data['month_idx'] - self.data['lmonth_idx']
                self.data.loc[self.data['month_diff'] != (J - 1), signal_col] = np.nan
                self.data.drop(columns=['month_idx', 'lmonth_idx', 'month_diff'], inplace=True)

            # Apply skip period
            self.data[signal_col] = self.data.groupby("ID")[signal_col].shift(skip)

        elif strategy_name == "LT-REVERSAL":
            J = self.strategy.J
            skip = self.strategy.skip
            varname = f'ret_{adj}'
            signal_col = f'signal_{adj}'

            # Get NaN handling parameters from strategy
            no_gap = getattr(self.strategy, 'no_gap', False)
            fill_na = getattr(self.strategy, 'fill_na', False)
            drop_na = getattr(self.strategy, 'drop_na', False)

            # Ensure data is sorted
            self.data = self.data.sort_values(['ID', 'date'], ignore_index=True)

            # Create month index for gap detection
            if no_gap:
                self.data['month_idx'] = (
                    self.data['date'].dt.year * 12 + self.data['date'].dt.month
                )

            if drop_na:
                # Option B: Variable window to accumulate valid (non-NaN) returns
                # Fill NaN with 0 for cumsum, but track validity separately
                self.data['ret_filled'] = self.data[varname].fillna(0)
                self.data['cumret'] = self.data.groupby('ID')['ret_filled'].cumsum()
                self.data['cumvalid'] = self.data.groupby('ID')[varname].transform(
                    lambda x: x.notna().cumsum()
                )
                self.data['is_valid'] = self.data[varname].notna()

                def compute_drop_na_signal(group):
                    """Compute LT reversal signal using valid returns."""
                    group = group.copy()
                    n = len(group)
                    signal = np.full(n, np.nan)
                    cumret = group['cumret'].values
                    cumvalid = group['cumvalid'].values
                    is_valid = group['is_valid'].values

                    for i in range(n):
                        if not is_valid[i] or cumvalid[i] < J:
                            continue
                        target_lt = cumvalid[i] - J
                        target_recent = cumvalid[i] - skip
                        lt_sum = None
                        recent_sum = None

                        if target_lt == 0:
                            lt_sum = cumret[i]
                        else:
                            # Find the last VALID row where cumvalid == target_lt
                            for j in range(i - 1, -1, -1):
                                if is_valid[j] and cumvalid[j] == target_lt:
                                    lt_sum = cumret[i] - cumret[j]
                                    break

                        if target_recent == 0:
                            recent_sum = cumret[i]
                        else:
                            # Find the last VALID row where cumvalid == target_recent
                            for j in range(i - 1, -1, -1):
                                if is_valid[j] and cumvalid[j] == target_recent:
                                    recent_sum = cumret[i] - cumret[j]
                                    break

                        if lt_sum is not None and recent_sum is not None:
                            signal[i] = lt_sum - recent_sum

                    group[signal_col] = signal
                    return group

                self.data = self.data.groupby('ID', group_keys=False).apply(compute_drop_na_signal)
                self.data.drop(columns=['cumret', 'cumvalid', 'ret_filled', 'is_valid'], inplace=True)

            elif fill_na:
                # Option A: Fixed window, NaN returns treated as 0%
                self.data['ret_filled'] = self.data[varname].fillna(0)

                long_term = (
                    self.data.groupby(['ID'], group_keys=False)['ret_filled']
                    .rolling(window=J, min_periods=J)
                    .sum()
                )
                recent = (
                    self.data.groupby(['ID'], group_keys=False)['ret_filled']
                    .rolling(window=skip, min_periods=skip)
                    .sum()
                )
                self.data[signal_col] = long_term.values - recent.values
                self.data.drop(columns=['ret_filled'], inplace=True)

            else:
                # Default: Standard rolling, NaN propagates
                long_term = (
                    self.data.groupby(['ID'], group_keys=False)[varname]
                    .rolling(window=J, min_periods=J)
                    .sum()
                )
                recent = (
                    self.data.groupby(['ID'], group_keys=False)[varname]
                    .rolling(window=skip, min_periods=skip)
                    .sum()
                )
                self.data[signal_col] = long_term.values - recent.values

            # Apply no_gap check: invalidate signal if months are not consecutive
            if no_gap:
                self.data['lmonth_idx'] = self.data.groupby('ID')['month_idx'].shift(J - 1)
                self.data['month_diff'] = self.data['month_idx'] - self.data['lmonth_idx']
                self.data.loc[self.data['month_diff'] != (J - 1), signal_col] = np.nan
                self.data.drop(columns=['month_idx', 'lmonth_idx', 'month_diff'], inplace=True)

        if self.verbose:
            print(f"Recomputed signal using {varname} for {strategy_name}")

    def _prepare_index_and_ids(self):
        """Prepare monotonic indices and stable integer IDs."""
        # Create monotonic index
        self.data[ColumnNames.INDEX] = np.arange(1, len(self.data) + 1, dtype=np.int64)
        self.data_raw[ColumnNames.INDEX] = np.arange(1, len(self.data_raw) + 1, dtype=np.int64)

        # Vectorized, deterministic ID mapping
        codes, uniques = pd.factorize(self.data[ColumnNames.ID], sort=True)
        self.data[ColumnNames.ID] = codes.astype(np.int64) + 1
        self.data_raw[ColumnNames.ID] = pd.Categorical(
            self.data_raw[ColumnNames.ID],
            categories=uniques
        ).codes.astype(np.int64) + 1

        # Also factorize IDs in data_winsorized_ex_post if it exists (wins filter)
        # This is needed because the Filter creates data_winsorized_ex_post before
        # ID factorization, so the IDs would be mismatched otherwise
        if hasattr(self, 'filter_obj') and self.filter_obj is not None:
            if hasattr(self.filter_obj, 'data_winsorized_ex_post') and self.filter_obj.data_winsorized_ex_post is not None:
                wdf = self.filter_obj.data_winsorized_ex_post
                wdf[ColumnNames.ID] = pd.Categorical(
                    wdf[ColumnNames.ID],
                    categories=uniques
                ).codes.astype(np.int64) + 1

        # Normalize dates
        if not pd.api.types.is_datetime64_any_dtype(self.data[ColumnNames.DATE]):
            self.data[ColumnNames.DATE] = pd.to_datetime(self.data[ColumnNames.DATE])
        if not pd.api.types.is_datetime64_any_dtype(self.data_raw[ColumnNames.DATE]):
            self.data_raw[ColumnNames.DATE] = pd.to_datetime(self.data_raw[ColumnNames.DATE])

        # Canonical sort
        self.data.sort_values([ColumnNames.ID, ColumnNames.DATE], inplace=True)
        self.data_raw.sort_values([ColumnNames.ID, ColumnNames.DATE], inplace=True)

        # Cache useful info
        self.datelist = pd.Index(self.data[ColumnNames.DATE].unique()).sort_values().tolist()
        self.unique_bonds = int(self.data[ColumnNames.ID].nunique())

        if self.verbose:
            print(f"Data prepared: {self.unique_bonds} unique bonds, {len(self.datelist)} periods")


    def _get_sort_vars(self) -> Tuple[str, Optional[str]]:
        """Get primary and secondary sorting variables."""
        use_adj = self.filters is not None
        main = self.strategy.get_sort_var(self.adj) if use_adj else self.strategy.get_sort_var()

        # Check for double sort
        is_double = getattr(self.strategy, "double_sort", 0) or getattr(self.strategy, "DoubleSort", 0)

        if is_double:
            if hasattr(self.strategy, "get_sort_var2"):
                second = self.strategy.get_sort_var2(self.adj if use_adj else None)
            else:
                second = getattr(self.strategy, "sort_var2", None)
        else:
            second = None

        return main, second

    def _build_required_columns(self, sort_var_main: str, sort_var2: Optional[str]) -> list:
        """Build list of required columns for analysis."""
        required = {
            ColumnNames.INDEX,
            ColumnNames.DATE,
            ColumnNames.ID,
            ColumnNames.RETURN,
            ColumnNames.RATING,
            ColumnNames.VALUE_WEIGHT
        }

        if self.adj == "price":
            required.add(ColumnNames.PRICE)

        if self.chars:
            required.update(self.chars)

        if self.subset_filter:
            required.update(self.subset_filter.keys())

        # Add adjusted return column if applicable
        if self.adj in ["trim", "price", "bounce"]:
            adj_ret_col = f"{ColumnNames.RETURN}_{self.adj}"
            if adj_ret_col in self.data.columns:
                required.add(adj_ret_col)

        # Add firm ID column for WithinFirmSort strategy
        if self.strategy.__strategy_name__ == "Within-Firm Sort":
            firm_id_col = getattr(self.strategy, 'firm_id_col', 'PERMNO')
            if firm_id_col in self.data.columns:
                required.add(firm_id_col)
            else:
                raise ValueError(
                    f"WithinFirmSort requires firm ID column '{firm_id_col}' which is missing from data. "
                    f"Available columns: {list(self.data.columns)}"
                )

        # Keep signal columns
        for s in filter(None, [sort_var_main, sort_var2]):
            if s in self.data.columns:
                required.add(s)

        # Add columns required by breakpoint_universe_func if it's a string column name
        bp_func = getattr(self.strategy, 'breakpoint_universe_func', None)
        bp_func2 = getattr(self.strategy, 'breakpoint_universe_func2', None)
        for func in [bp_func, bp_func2]:
            if isinstance(func, str) and func in self.data.columns:
                required.add(func)

        # Build ordered list
        cols = [
            ColumnNames.INDEX, ColumnNames.DATE, ColumnNames.ID,
            ColumnNames.RETURN, ColumnNames.RATING, ColumnNames.VALUE_WEIGHT
        ]

        if ColumnNames.PRICE in required and ColumnNames.PRICE not in cols:
            cols.append(ColumnNames.PRICE)

        for c in sorted(required - set(cols)):
            cols.append(c)

        return cols

    def _validate_double_sort_date_coverage(self, sort_var_main: str, sort_var2: str):
        """
        Validate that both sorting variables have consistent date coverage.

        For double sorts, this method checks two conditions:
        1. Both signals should have the same date range
        2. At each date, at least some bonds must have non-NaN values for BOTH signals

        Behavior depends on strategy.auto_match_signals:
        - False (default): Raises ValueError, user must fix data manually
        - True: Warns and automatically truncates data to overlapping period

        Parameters
        ----------
        sort_var_main : str
            Primary sorting variable name
        sort_var2 : str
            Secondary sorting variable name

        Raises
        ------
        ValueError
            If auto_match_signals=False and signals have mismatched coverage
        """
        if sort_var_main not in self.data.columns or sort_var2 not in self.data.columns:
            return  # Column validation happens elsewhere

        date_col = ColumnNames.DATE

        # Check if auto-matching is enabled
        auto_match = getattr(self.strategy, 'auto_match_signals', False)

        # Find date ranges with non-NaN values for each signal
        df1 = self.data[self.data[sort_var_main].notna()]
        df2 = self.data[self.data[sort_var2].notna()]

        if df1.empty or df2.empty:
            return  # Empty data validation happens elsewhere

        max_date1 = df1[date_col].max()
        max_date2 = df2[date_col].max()
        min_date1 = df1[date_col].min()
        min_date2 = df2[date_col].min()

        # Check 1: Different date ranges
        if max_date1 != max_date2 or min_date1 != min_date2:
            if auto_match:
                # Auto-truncate with warning
                new_min = max(min_date1, min_date2)
                new_max = min(max_date1, max_date2)

                warnings.warn(
                    f"\nDouble sort date coverage mismatch detected:\n"
                    f"  - '{sort_var_main}' has data from {min_date1} to {max_date1}\n"
                    f"  - '{sort_var2}' has data from {min_date2} to {max_date2}\n"
                    f"auto_match_signals=True: Truncating data to {new_min} - {new_max}",
                    UserWarning
                )

                n_before = len(self.data)
                self.data = self.data[
                    (self.data[date_col] >= new_min) & (self.data[date_col] <= new_max)
                ].copy()
                n_after = len(self.data)

                if self.verbose:
                    print(f"Truncated data from {n_before:,} to {n_after:,} observations")
            else:
                # Raise error - user must fix manually
                raise ValueError(
                    f"\nDouble sort date coverage mismatch:\n"
                    f"  - '{sort_var_main}' has data from {min_date1} to {max_date1}\n"
                    f"  - '{sort_var2}' has data from {min_date2} to {max_date2}\n\n"
                    f"Both sorting variables must have the same date range.\n"
                    f"Either:\n"
                    f"  1. Filter your data to the overlapping period before calling StrategyFormation\n"
                    f"  2. Use auto_match_signals=True in DoubleSort() to auto-truncate"
                )

        # Check 2: At each date, verify some bonds have non-NaN for BOTH signals
        both_valid = self.data[
            self.data[sort_var_main].notna() & self.data[sort_var2].notna()
        ]

        if both_valid.empty:
            raise ValueError(
                f"\nDouble sort error: No observations have valid (non-NaN) values for BOTH "
                f"'{sort_var_main}' and '{sort_var2}' at any date.\n"
                f"For double sorting, each bond needs non-NaN values in both sorting variables.\n"
                f"Please check your data preparation."
            )

        # Find dates with no overlap (all bonds have NaN in at least one signal)
        dates_with_overlap = set(both_valid[date_col].unique())
        all_dates = set(self.data[date_col].unique())
        dates_without_overlap = all_dates - dates_with_overlap

        if dates_without_overlap:
            n_missing = len(dates_without_overlap)
            n_total = len(all_dates)

            # Show some examples
            examples = sorted(dates_without_overlap)[:5]
            examples_str = ", ".join(str(d)[:10] for d in examples)
            if n_missing > 5:
                examples_str += f", ... ({n_missing - 5} more)"

            if auto_match:
                # Auto-remove dates without overlap
                warnings.warn(
                    f"\nDouble sort signal overlap warning:\n"
                    f"  {n_missing} of {n_total} dates have no bonds with valid values for BOTH signals.\n"
                    f"  Dates without overlap: {examples_str}\n"
                    f"auto_match_signals=True: Removing these dates from data.",
                    UserWarning
                )

                n_before = len(self.data)
                self.data = self.data[self.data[date_col].isin(dates_with_overlap)].copy()
                n_after = len(self.data)

                if self.verbose:
                    print(f"Removed {n_missing} dates without signal overlap "
                          f"({n_before:,} -> {n_after:,} observations)")
            else:
                raise ValueError(
                    f"\nDouble sort signal overlap error:\n"
                    f"  {n_missing} of {n_total} dates have no bonds with valid values for BOTH signals.\n"
                    f"  Dates without overlap: {examples_str}\n\n"
                    f"For double sorting, at each date there must be at least one bond with non-NaN\n"
                    f"values for both '{sort_var_main}' and '{sort_var2}'.\n"
                    f"Either:\n"
                    f"  1. Check your data - these dates have no overlapping coverage\n"
                    f"  2. Use auto_match_signals=True in DoubleSort() to auto-remove these dates"
                )

    def _print_initialization_summary(self):
        """Print initialization summary."""
        print("=" * 60)
        print("StrategyFormation Initialization")
        print("=" * 60)
        print(f"Strategy: {self.strategy.strategy_name}")

        # Print sorting variable(s)
        try:
            sort_var = self.strategy.get_sort_var()
            print(f"Sort variable: {sort_var}")
            # For DoubleSort, also print second sort variable
            if hasattr(self.strategy, 'get_sort_var2'):
                sort_var2 = self.strategy.get_sort_var2()
                if sort_var2:
                    print(f"Sort variable 2: {sort_var2}")
        except Exception:
            pass

        print(f"Holding period: {self.hor}")
        print(f"Number of portfolios: {self.nport}")
        print(f"Rebalancing frequency: {self.rebalance_frequency}")

        if self.rating:
            print(f"Rating filter: {self.rating}")
        if self.subset_filter:
            print(f"Subset filters: {self.subset_filter}")
        if self.chars:
            print(f"Tracking characteristics: {self.chars}")
        if self.turnover:
            print("Computing turnover: Yes")
        if self.banding_threshold:
            print(f"Banding threshold: {self.banding_threshold}")
        if self.filters:
            print(f"Ex-post filters: {self.filters}")

        print("=" * 60)


    # =========================================================================
    # Portfolio Formation Methods
    # =========================================================================


    def fit(self,
        IDvar: Optional[str] = None,
        DATEvar: Optional[str] = None,
        RETvar: Optional[str] = None,
        RATINGvar: Optional[str] = None,
        VWvar: Optional[str] = None,
        PRICEvar: Optional[str] = None):
        """
        Form portfolios and compute returns.

        This is the main method that executes the portfolio formation process.
            Parameters
        ----------
        IDvar : str, optional
            Name of the ID column in the data (default: 'ID')
        DATEvar : str, optional
            Name of the date column in the data (default: 'date')
        RETvar : str, optional
            Name of the return column in the data (default: 'ret')
        RATINGvar : str, optional
            Name of the rating column in the data (default: 'RATING_NUM')
        VWvar : str, optional
            Name of the value weight column in the data (default: 'VW')
        PRICEvar : str, optional
            Name of the price column in the data (default: 'PRICE')

        Returns
        -------
        StrategyResults
            Object containing all results (returns, characteristics, turnover, etc.)
        """
        # Update column names if custom names are provided
        self._apply_column_mapping(IDvar, DATEvar, RETvar, RATINGvar, VWvar, PRICEvar)

        # Validate and prepare data
        self._validate_data()
        self._prepare_data()

        self._computing_ep = False
        if self.verbose:
            print("\nStarting portfolio formation...")

        # Determine if using staggered or non-staggered rebalancing
        is_staggered = self.rebalance_frequency == 'monthly'

        # Check if fast returns-only path can be used
        use_fast_path = self._can_use_fast_path()
        use_nonstaggered_fast_path = self._can_use_nonstaggered_fast_path()
        use_withinfirm_fast_path = self._can_use_withinfirm_fast_path()

        # Form portfolios (EA results)
        if use_withinfirm_fast_path:
            ea_results = self._fit_withinfirm_fast()
        elif use_fast_path:
            ea_results = self._fit_fast_returns_only()
        elif use_nonstaggered_fast_path:
            ea_results = self._fit_nonstaggered_fast()
        elif is_staggered:
            ea_results = self._fit_staggered()
        else:
            ea_results = self._fit_nonstaggered()

        # Form EP results if filters are applied
        ep_results = None
        if self.config.has_filters:
            # Apply filters to data
            self._computing_ep = True

            # Re-run portfolio formation. # uses It2
            if use_withinfirm_fast_path:
                ep_results = self._fit_withinfirm_fast()
            elif use_fast_path:
                ep_results = self._fit_fast_returns_only()
            elif use_nonstaggered_fast_path:
                ep_results = self._fit_nonstaggered_fast()
            elif is_staggered:
                ep_results = self._fit_staggered()
            else:
                ep_results = self._fit_nonstaggered()
            self._computing_ep = False

        # Get strategy name
        strategy_name = self.strategy.str_name if hasattr(self.strategy, 'str_name') else 'strategy'

        # Build FormationResults
        results = build_formation_results(
            name=strategy_name,
            ea_results=ea_results,
            ep_results=ep_results,
            config=self._get_config_dict(),
            metadata=self._get_metadata_dict(),
            port_idx=self.port_idx if self.save_idx else None,
        )

        # Store results
        self.results = results

        if self.verbose:
            print("Portfolio formation complete!")

        return results

    def _get_config_dict(self) -> dict:
        """Get configuration dictionary."""
        return {
            'nport': self.nport,
            'holding_period': self.hor,
            'rebalance_frequency': self.rebalance_frequency,
            'rating': self.rating,
            'turnover': self.turnover,
            'chars': self.chars,
            'filters': self.filters is not None,
        }
    def _get_metadata_dict(self) -> dict:
        """Get metadata dictionary."""
        is_double = getattr(self.strategy, "double_sort", 0) or getattr(self.strategy, "DoubleSort", 0)
        metadata = {
            'n_periods': len(self.datelist),
            'n_portfolios': self._get_total_portfolios(),
            'is_double_sort': bool(is_double),
        }
        if is_double:
            metadata['n1'] = self.nport
            metadata['n2'] = getattr(self.strategy, "num_portfolios2", None) or getattr(self.strategy, "nport2", None)
        return metadata

    def _get_return_data(self, precomputed, date):
        """
        Get return data for portfolio formation.

        Parameters
        ----------
        precomputed : PrecomputedData
            Precomputed portfolio data
        date : pd.Timestamp
            Date to get returns for

        Returns
        -------
        pd.DataFrame
            Return data (It1 for EA, It2 for EP)
        """
        if getattr(self, '_use_ep_data', False):
            # EP: Use filtered returns from It2
            return precomputed.It2[date]
        else:
            # EA: Use unfiltered returns from It1
            return precomputed.It1[date]

    def _fit_staggered(self):
        """Portfolio formation with staggered overlapping portfolios (monthly)."""
        if self.verbose:
            print("Using staggered (monthly) rebalancing...")

        # Precompute time-series data
        precomp = self._precompute_data()

        # Initialize results arrays
        TM = len(self.datelist)
        tot_nport = self._get_total_portfolios()

        ew_ret_arr = np.full((TM, self.hor, tot_nport), np.nan)
        vw_ret_arr = np.full((TM, self.hor, tot_nport), np.nan)

        # Initialize characteristics arrays if requested
        if self.chars:
            num_chars = len(self.chars)
            ew_chars_arr = {c: np.full((TM, self.hor, tot_nport), np.nan) for c in self.chars}
            vw_chars_arr = {c: np.full((TM, self.hor, tot_nport), np.nan) for c in self.chars}
        else:
            ew_chars_arr = None
            vw_chars_arr = None

        # Initialize turnover if requested
        if self.turnover:
            self.turnover_manager = TurnoverManager(
                self.data, self.datelist, self.hor, 'monthly',
                use_nanmean=self.config.formation.turnover_nanmean
                )
            self.turnover_state = self.turnover_manager.init_state(TM, tot_nport)

        # Main loop: form portfolios for each cohort
        for t_idx, date_t in enumerate(self.datelist):
            # Store cohort for banding (matching old version: t % hor)
            self.cohort = t_idx % self.hor

            # Form portfolio for this cohort
            self._form_cohort_portfolios(
                t_idx, date_t, precomp,
                ew_ret_arr, vw_ret_arr,
                ew_chars_arr, vw_chars_arr
            )

        # Aggregate results
        results = self._aggregate_results_staggered(
            ew_ret_arr, vw_ret_arr,
            ew_chars_arr, vw_chars_arr
        )

        return results

    def _fit_nonstaggered(self):
        """Portfolio formation with non-staggered rebalancing e.g., Fama-French type portfolios."""
        if self.verbose:
            print(f"Using non-staggered rebalancing: {self.rebalance_frequency}...")

        # Get rebalancing dates
        rebal_dates_idx = _get_rebalancing_dates(
            self.datelist,
            self.rebalance_frequency,
            self.rebalance_month
        )

        if self.verbose:
            print(f"Rebalancing at {len(rebal_dates_idx)} dates")

        # Precompute data
        precomp = self._precompute_data()

        # Initialize results
        TM = len(self.datelist)
        tot_nport = self._get_total_portfolios()

        ew_ret_arr = np.full((TM, tot_nport), np.nan)
        vw_ret_arr = np.full((TM, tot_nport), np.nan)

        # Initialize characteristics arrays if requested
        if self.chars:
            ew_chars_arr = {c: np.full((TM, tot_nport), np.nan) for c in self.chars}
            vw_chars_arr = {c: np.full((TM, tot_nport), np.nan) for c in self.chars}
        else:
            ew_chars_arr = None
            vw_chars_arr = None

        # Initialize turnover
        if self.turnover:
            self.turnover_manager = TurnoverManager(
                self.data, self.datelist, self.hor, self.rebalance_frequency,
                use_nanmean=self.config.formation.turnover_nanmean
            )
            self.turnover_state = self.turnover_manager.init_state(TM, tot_nport)
        # Main loop
        for rebal_idx in rebal_dates_idx:
            # Store rebalance date for banding
            self.current_rebal_date = self.datelist[rebal_idx]

            self._form_nonstaggered_portfolio(
                rebal_idx, precomp,
                ew_ret_arr, vw_ret_arr,
                ew_chars_arr, vw_chars_arr,
                rebal_dates_idx  # Phase 17: pass rebalancing dates for proper iteration
            )

        # Aggregate results
        results = self._aggregate_results_nonstaggered(
            ew_ret_arr, vw_ret_arr,
            ew_chars_arr, vw_chars_arr
        )

        return results

    def _can_use_nonstaggered_fast_path(self) -> bool:
        """Check if fast non-staggered path can be used.

        Phase 15b: Now supports turnover, chars, and banding!
        """
        # Must be non-staggered rebalancing
        if self.rebalance_frequency == 'monthly':
            return False
        # Only SingleSort supported (no DoubleSort)
        is_double = getattr(self.strategy, "double_sort", 0) or getattr(self.strategy, "DoubleSort", 0)
        if is_double:
            return False
        # Disable fast path for filters - need to validate filter handling first
        if self.config.has_filters:
            return False
        # Phase 15b: turnover, chars, and banding are now supported
        return True

    def _fit_nonstaggered_fast(self):
        """
        Ultra-fast non-staggered portfolio formation using numba.

        This is significantly faster than the standard path because:
        1. Computes ranks only at rebalancing dates (not all dates)
        2. Uses parallel numba kernels for return computation
        3. Bypasses pandas precomputation entirely

        Phase 15b: Now supports turnover, chars, and banding!

        Requirements:
        - Non-staggered rebalancing (quarterly, semi-annual, annual)
        - SingleSort only
        """
        from .numba_core import (
            compute_nonstaggered_full_fast,
            compute_nonstaggered_ls_returns,
            build_vw_lookup_table,
        )
        from .utils_optimized import _get_rebalancing_dates

        if self.verbose:
            features = []
            if self.turnover:
                features.append("turnover")
            if self.chars:
                features.append("chars")
            if self.banding_threshold is not None:
                features.append("banding")
            feature_str = f" ({', '.join(features)})" if features else ""
            print(f"Using ULTRA-FAST non-staggered path ({self.rebalance_frequency}){feature_str}...")

        TM = len(self.datelist)
        tot_nport = self._get_total_portfolios()
        sort_var_main, _ = self._get_sort_vars()

        # Get rebalancing dates
        rebal_dates_idx = _get_rebalancing_dates(
            self.datelist,
            self.rebalance_frequency,
            self.rebalance_month
        )
        rebal_dates_idx = np.array(rebal_dates_idx, dtype=np.int64)

        if self.verbose:
            print(f"  Rebalancing at {len(rebal_dates_idx)} dates")

        # Get filtered data
        tab = self.data

        # Create date-to-index mapping
        date_to_idx = {d: i for i, d in enumerate(self.datelist)}

        # Only keep rows with valid dates in our datelist
        valid_mask = tab[ColumnNames.DATE].isin(self.datelist)
        data = tab[valid_mask].copy()

        if data.empty:
            return self._create_empty_results()

        # Determine which return column to use
        if self._computing_ep and self.adj:
            ret_col = f'ret_{self.adj}'
        else:
            ret_col = ColumnNames.RETURN

        # Create ID mapping
        all_ids = data[ColumnNames.ID].unique()
        id_to_idx = {id_val: idx for idx, id_val in enumerate(all_ids)}
        n_ids = len(all_ids)

        # Convert data to numpy arrays
        date_idx = data[ColumnNames.DATE].map(date_to_idx).values.astype(np.int64)
        id_idx = data[ColumnNames.ID].map(id_to_idx).values.astype(np.int64)
        signal = data[sort_var_main].values.astype(np.float64)
        ret = data[ret_col].values.astype(np.float64)
        vw = data[ColumnNames.VALUE_WEIGHT].values.astype(np.float64)

        # Build VW lookup table
        vw_lookup = build_vw_lookup_table(date_idx, id_idx, vw, TM, n_ids)

        # Prepare characteristics array if needed
        n_chars = len(self.chars) if self.chars else 0
        if n_chars > 0:
            char_values = np.column_stack([
                data[c].values.astype(np.float64) for c in self.chars
            ])
        else:
            char_values = np.empty((len(data), 0), dtype=np.float64)

        # Banding threshold (-1 means no banding)
        banding = self.banding_threshold if self.banding_threshold is not None else -1.0

        # Call the full-featured numba kernel
        # Note: For non-staggered rebalancing, always use dynamic_weights=False
        # (VW from formation date, not d-1) per PyBondLab specification
        ew_ret_arr, vw_ret_arr, ew_turn_arr, vw_turn_arr, ew_chars_arr, vw_chars_arr = \
            compute_nonstaggered_full_fast(
                date_idx, id_idx, signal, ret, vw, char_values,
                rebal_dates_idx, self.hor, TM, n_ids, tot_nport, n_chars,
                self.turnover, n_chars > 0, banding, False, vw_lookup  # dynamic_weights=False
            )

        # Compute long-short returns
        ew_ls, vw_ls = compute_nonstaggered_ls_returns(ew_ret_arr, vw_ret_arr, tot_nport)

        # Build results structure
        sort_var_main, sort_var2 = self._get_sort_vars()
        is_double = getattr(self.strategy, "double_sort", 0) or getattr(self.strategy, "DoubleSort", 0)

        if is_double:
            nport2 = getattr(self.strategy, "num_portfolios2", None) or getattr(self.strategy, "nport2", None)
            ptf_labels = get_signal_based_labels(sort_var_main, self.nport, sort_var2, nport2)
        else:
            ptf_labels = get_signal_based_labels(sort_var_main, self.nport)

        # Create DataFrames
        ew_df = pd.DataFrame(ew_ret_arr, index=self.datelist, columns=ptf_labels)
        vw_df = pd.DataFrame(vw_ret_arr, index=self.datelist, columns=ptf_labels)

        # Create long-short DataFrames
        prefix = 'EWEP' if self._computing_ep else 'EWEA'
        vw_prefix = 'VWEP' if self._computing_ep else 'VWEA'

        ewls_df = pd.DataFrame(ew_ls, index=self.datelist, columns=[f'{prefix}_{self.name}'])
        vwls_df = pd.DataFrame(vw_ls, index=self.datelist, columns=[f'{vw_prefix}_{self.name}'])

        # Long and short portfolios
        ew_long = ew_ret_arr[:, -1]
        vw_long = vw_ret_arr[:, -1]
        ew_short = ew_ret_arr[:, 0]
        vw_short = vw_ret_arr[:, 0]

        ew_long_df = pd.DataFrame(ew_long, index=self.datelist, columns=[f'LONG_{prefix}_{self.name}'])
        vw_long_df = pd.DataFrame(vw_long, index=self.datelist, columns=[f'LONG_{vw_prefix}_{self.name}'])
        ew_short_df = pd.DataFrame(ew_short, index=self.datelist, columns=[f'SHORT_{prefix}_{self.name}'])
        vw_short_df = pd.DataFrame(vw_short, index=self.datelist, columns=[f'SHORT_{vw_prefix}_{self.name}'])

        # Process turnover if computed
        if self.turnover:
            # Skip first row (no turnover at first rebalancing)
            ew_turnover_df = pd.DataFrame(
                ew_turn_arr[1:, :], index=self.datelist[1:], columns=ptf_labels
            )
            vw_turnover_df = pd.DataFrame(
                vw_turn_arr[1:, :], index=self.datelist[1:], columns=ptf_labels
            )
        else:
            ew_turnover_df = None
            vw_turnover_df = None

        # Process characteristics if computed
        if n_chars > 0:
            chars_ew = {}
            chars_vw = {}
            for c_idx, c_name in enumerate(self.chars):
                chars_ew[c_name] = pd.DataFrame(
                    ew_chars_arr[:, c_idx, :], index=self.datelist, columns=ptf_labels
                )
                chars_vw[c_name] = pd.DataFrame(
                    vw_chars_arr[:, c_idx, :], index=self.datelist, columns=ptf_labels
                )
        else:
            chars_ew = None
            chars_vw = None

        # Build and return StrategyResults (same format as slow path)
        return build_strategy_results(
            ewport_df=ew_df,
            vwport_df=vw_df,
            ewls_df=ewls_df,
            vwls_df=vwls_df,
            ewls_long_df=ew_long_df,
            vwls_long_df=vw_long_df,
            ewls_short_df=ew_short_df,
            vwls_short_df=vw_short_df,
            turnover_ew_df=ew_turnover_df,
            turnover_vw_df=vw_turnover_df,
            chars_ew=chars_ew,
            chars_vw=chars_vw,
        )

    def _can_use_fast_path(self) -> bool:
        """Check if fast returns-only path can be used."""
        # Fast path requires: no turnover, no chars, no banding, SingleSort only
        if self.turnover:
            return False
        if self.chars:
            return False
        if self.banding_threshold is not None:
            return False
        # Only SingleSort supported (no DoubleSort)
        is_double = getattr(self.strategy, "double_sort", 0) or getattr(self.strategy, "DoubleSort", 0)
        if is_double:
            return False
        # WithinFirmSort requires special handling (within-firm grouping, rating bins, etc.)
        is_within_firm = getattr(self.strategy, "__strategy_name__", "") == "Within-Firm Sort"
        if is_within_firm:
            return False
        # Only monthly (staggered) rebalancing for now
        if self.rebalance_frequency != 'monthly':
            return False
        # Disable fast path for filters - ID intersection logic differs from slow path
        # TODO: Fix fast path to properly handle (formation_date, return_date) ID intersection
        if self.config.has_filters:
            return False
        # Fast path supports both dynamic_weights=True and False:
        # - True: VW from day before return date (d-1)
        # - False: VW from formation date (different per cohort for hp>1)
        return True

    def _can_use_withinfirm_fast_path(self) -> bool:
        """Check if WithinFirmSort fast path can be used."""
        # Only for WithinFirmSort strategy
        if self.strategy.__strategy_name__ != "Within-Firm Sort":
            return False
        # No turnover (uses slow path with multiprocessing for turnover)
        if self.turnover:
            return False
        # No chars (uses slow path with multiprocessing for chars)
        if self.chars:
            return False
        # HP=1 only (HP>1 is disabled for WithinFirmSort)
        if self.hor != 1:
            return False
        # Monthly rebalancing only
        if self.rebalance_frequency != 'monthly':
            return False
        return True

    def _fit_withinfirm_fast(self):
        """
        Ultra-fast portfolio formation for WithinFirmSort (HP=1, no turnover, no chars).

        This bypasses _precompute_data() entirely and works directly with numpy arrays.

        For HP=1:
        - Formation date t: rank bonds using signal_t within (rating_terc_t, firm_t) groups
        - Return date t+1: collect returns from date t+1 for those ranked bonds
        - Result indexed at t+1 (return date)

        Requirements:
        - WithinFirmSort strategy
        - turnover=False
        - chars=None
        - holding_period=1
        - Monthly rebalancing
        """
        from .numba_core import (
            compute_withinfirm_assignments_all_dates,
            compute_within_firm_aggregation_with_lookup
        )

        if self.verbose:
            print("Using ULTRA-FAST WithinFirmSort path...")

        # Get strategy parameters
        firm_id_col = getattr(self.strategy, 'firm_id_col', 'PERMNO')
        rating_bins = getattr(self.strategy, 'rating_bins', [-np.inf, 7, 10, np.inf])
        min_bonds = getattr(self.strategy, 'min_bonds_per_firm', 2)
        sort_var = self.strategy.sort_var

        # Get filtered data
        tab = self.data

        # Create mappings
        date_to_idx = {d: i for i, d in enumerate(self.datelist)}
        n_dates = len(self.datelist)

        # Filter to valid dates
        valid_mask = tab[ColumnNames.DATE].isin(self.datelist)
        data = tab[valid_mask].copy()

        if data.empty:
            return self._create_empty_results()

        # Create ID mapping
        unique_ids = data[ColumnNames.ID].unique()
        id_to_idx = {id_: i for i, id_ in enumerate(unique_ids)}
        n_ids = len(unique_ids)

        # Create firm mapping
        unique_firms = data[firm_id_col].dropna().unique()
        firm_to_idx = {f: i for i, f in enumerate(unique_firms)}
        n_firms = len(unique_firms)

        # Create rating terciles
        rating_terc = pd.cut(
            pd.to_numeric(data[ColumnNames.RATING], errors='coerce'),
            bins=rating_bins,
            labels=[1, 2, 3],
            include_lowest=True
        ).astype('Int64').fillna(0).values.astype(np.int64)

        # Convert to numpy arrays
        date_idx = data[ColumnNames.DATE].map(date_to_idx).values.astype(np.int64)
        id_idx = data[ColumnNames.ID].map(id_to_idx).values.astype(np.int64)
        firm_idx = data[firm_id_col].map(firm_to_idx).fillna(-1).values.astype(np.int64)
        signal = data[sort_var].values.astype(np.float64)
        ret = data[ColumnNames.RETURN].values.astype(np.float64)
        vw = data[ColumnNames.VALUE_WEIGHT].values.astype(np.float64)

        # Sort by (date, rating_terc, firm) for group processing
        sort_order = np.lexsort((firm_idx, rating_terc, date_idx))
        date_idx_sorted = date_idx[sort_order]
        rating_terc_sorted = rating_terc[sort_order].astype(np.float64)
        firm_idx_sorted = firm_idx[sort_order]
        id_idx_sorted = id_idx[sort_order]
        signal_sorted = signal[sort_order]
        vw_sorted = vw[sort_order]

        # Find group boundaries (date, rating_terc, firm)
        n_obs = len(date_idx_sorted)
        group_keys = date_idx_sorted * 1000000 + rating_terc_sorted.astype(np.int64) * 10000 + firm_idx_sorted
        group_changes = np.concatenate([
            [0],
            np.where(np.diff(group_keys) != 0)[0] + 1,
            [n_obs]
        ])
        group_starts = group_changes[:-1].astype(np.int64)
        group_ends = group_changes[1:].astype(np.int64)

        # Step 1: Compute HIGH/LOW assignments at formation dates
        ptf_rank = compute_withinfirm_assignments_all_dates(
            signal_sorted, vw_sorted, group_starts, group_ends, min_bonds
        )

        # Step 2: Build rank lookup table: (formation_date, bond_id) -> (rank, rating_terc, firm_idx)
        # Lookup shape: (n_dates, n_ids, 3) for [rank, rating_terc, firm_idx]
        rank_lookup = np.zeros((n_dates, n_ids, 3), dtype=np.float64)
        for i in range(n_obs):
            d = date_idx_sorted[i]
            b = id_idx_sorted[i]
            rank_lookup[d, b, 0] = ptf_rank[i]  # rank (1=LOW, 2=HIGH, 0=unassigned)
            rank_lookup[d, b, 1] = rating_terc_sorted[i]  # rating tercile
            rank_lookup[d, b, 2] = firm_idx_sorted[i]  # firm index

        # Step 3: Build VW lookup for dynamic weights
        # VW at date d-1 for each bond
        vw_lookup = np.full((n_dates, n_ids), np.nan, dtype=np.float64)
        for i in range(len(data)):
            d = date_idx[i]
            b = id_idx[i]
            vw_lookup[d, b] = vw[i]

        # Step 4: Aggregate returns at return dates
        # For each return date t+1, look up ranks from formation date t
        (ew_long_short, vw_long_short,
         ew_high_ret, ew_low_ret,
         vw_high_ret, vw_low_ret) = compute_within_firm_aggregation_with_lookup(
            date_idx, id_idx, firm_idx, ret, vw,
            rank_lookup, vw_lookup, n_dates, n_ids, n_firms
        )

        # Create result series
        ewls_series = pd.Series(ew_long_short, index=self.datelist)
        vwls_series = pd.Series(vw_long_short, index=self.datelist)
        ew_high_series = pd.Series(ew_high_ret, index=self.datelist)
        ew_low_series = pd.Series(ew_low_ret, index=self.datelist)
        vw_high_series = pd.Series(vw_high_ret, index=self.datelist)
        vw_low_series = pd.Series(vw_low_ret, index=self.datelist)

        # Build result DataFrames
        sort_var_main, _ = self._get_sort_vars()
        ptf_labels = ['LOW', 'HIGH']

        # Portfolio returns
        ew_port = pd.DataFrame(
            np.column_stack([ew_low_series.values, ew_high_series.values]),
            index=self.datelist,
            columns=ptf_labels
        )
        vw_port = pd.DataFrame(
            np.column_stack([vw_low_series.values, vw_high_series.values]),
            index=self.datelist,
            columns=ptf_labels
        )

        # Long-short DataFrames (match format expected by build_strategy_results)
        prefix = 'EWEP' if self._computing_ep else 'EWEA'
        vw_prefix = 'VWEP' if self._computing_ep else 'VWEA'

        ewls_df = pd.DataFrame(ewls_series.values, index=self.datelist, columns=[f'{prefix}_{self.name}'])
        vwls_df = pd.DataFrame(vwls_series.values, index=self.datelist, columns=[f'{vw_prefix}_{self.name}'])
        ew_long_df = pd.DataFrame(ew_high_series.values, index=self.datelist, columns=[f'LONG_{prefix}_{self.name}'])
        vw_long_df = pd.DataFrame(vw_high_series.values, index=self.datelist, columns=[f'LONG_{vw_prefix}_{self.name}'])
        ew_short_df = pd.DataFrame(ew_low_series.values, index=self.datelist, columns=[f'SHORT_{prefix}_{self.name}'])
        vw_short_df = pd.DataFrame(vw_low_series.values, index=self.datelist, columns=[f'SHORT_{vw_prefix}_{self.name}'])

        # Build results object using same function as other fast paths
        self.results = build_strategy_results(
            ewport_df=ew_port,
            vwport_df=vw_port,
            ewls_df=ewls_df,
            vwls_df=vwls_df,
            ewls_long_df=ew_long_df,
            vwls_long_df=vw_long_df,
            ewls_short_df=ew_short_df,
            vwls_short_df=vw_short_df,
            turnover_ew_df=None,  # No turnover for fast path
            turnover_vw_df=None,
            chars_ew=None,
            chars_vw=None,
        )

        return self.results

    def _fit_fast_returns_only(self):
        """
        Ultra-fast portfolio formation when only returns are needed.

        This is 10-20x faster than the standard path because:
        1. Bypasses pandas precomputation entirely - works directly with numpy arrays
        2. All dates are processed in parallel using prange
        3. No turnover/scaled weights computation
        4. No characteristics computation
        5. No banding state tracking

        Requirements:
        - turnover=False
        - chars=None
        - banding=None
        - SingleSort only
        - Monthly rebalancing

        Supports filters:
        - EA (Ex-Ante): Uses original returns ('ret')
        - EP (Ex-Post): Uses filtered returns ('ret_{adj}')
        - Both use the same ranking (from filtered signal)
        """
        path_type = "EP" if self._computing_ep else "EA"
        if self.verbose:
            print(f"Using ULTRA-FAST returns-only path ({path_type})...")

        TM = len(self.datelist)
        tot_nport = self._get_total_portfolios()
        sort_var_main, _ = self._get_sort_vars()

        # Get the filtered data (this applies rating/char filters if any)
        tab = self.data

        # Create date-to-index mapping
        date_to_idx = {d: i for i, d in enumerate(self.datelist)}

        # Step 1: Convert data to numpy arrays ONCE (no per-date loops)
        # Only keep rows with valid dates in our datelist
        valid_mask = tab[ColumnNames.DATE].isin(self.datelist)
        data = tab[valid_mask].copy()

        if data.empty:
            return self._create_empty_results()

        # Determine which return column to use
        # EA: Use original returns (ColumnNames.RETURN = 'ret')
        # EP: Use filtered returns ('ret_{adj}') when filters are applied
        if self._computing_ep and self.adj:
            ret_col = f'ret_{self.adj}'
        else:
            ret_col = ColumnNames.RETURN

        if data.empty:
            return self._create_empty_results()

        # Create ID mapping (from ALL data - needed for rank lookup)
        all_ids = data[ColumnNames.ID].unique()
        id_to_idx = {id_val: idx for idx, id_val in enumerate(all_ids)}
        n_ids = len(all_ids)

        # Convert to numpy arrays
        # Formation data: use all rows with valid signal (NaN signals excluded by rank computation)
        date_idx = data[ColumnNames.DATE].map(date_to_idx).values.astype(np.int64)
        id_idx = data[ColumnNames.ID].map(id_to_idx).values.astype(np.int64)
        signal = data[sort_var_main].values.astype(np.float64)
        vw = data[ColumnNames.VALUE_WEIGHT].values.astype(np.float64)

        # Step 1b: Build filter mask for rating and subset_filter (Phase 14)
        # Filter mask: True = observation passes filter at this date
        # We apply filter by setting signal to NaN for filtered-out observations.
        # This ensures filtered bonds won't be ranked at formation date.
        # BUT their returns are still collected if they were in a portfolio.
        if self.rating is not None or self.subset_filter is not None:
            filter_mask = np.ones(len(data), dtype=np.bool_)

            if self.rating is not None:
                rating_vals = data[ColumnNames.RATING].values
                if isinstance(self.rating, str):
                    min_r, max_r = get_rating_bounds(self.rating)
                else:
                    min_r, max_r = self.rating
                filter_mask &= (rating_vals >= min_r) & (rating_vals <= max_r)

            if self.subset_filter is not None:
                for col, (min_val, max_val) in self.subset_filter.items():
                    if col not in data.columns:
                        raise ValueError(f"subset_filter column '{col}' not found in data")
                    col_vals = data[col].values
                    filter_mask &= (col_vals >= min_val) & (col_vals <= max_val)

            # Set signal to NaN for observations that don't pass filter
            # This ensures they won't be ranked (rank computation skips NaN)
            signal[~filter_mask] = np.nan

            if self.verbose:
                n_filtered = (~filter_mask).sum()
                pct_filtered = 100 * n_filtered / len(data)
                print(f"  Filter excludes {n_filtered:,} observations ({pct_filtered:.1f}%) from ranking")

        # Return data: need to handle NaN in ret_col properly
        # For EP, set NaN returns so they're excluded from return calculation
        returns = data[ret_col].values.astype(np.float64)

        # Step 2: Compute ranks for ALL dates in parallel using numba
        # This replaces the slow per-date pandas groupby/rank operations
        ranks = compute_ranks_all_dates_fast(date_idx, signal, TM, tot_nport)

        # Step 3: Prepare VW data for portfolio weighting
        # Build VW lookup table: vw_lookup[date * n_ids + bond_id] = VW
        vw_lookup = build_vw_lookup(date_idx, id_idx, vw, TM, n_ids)

        # Step 4: Compute portfolio returns using ultra-fast numba functions
        if self.hor == 1:
            # Simple case: h=1
            # For hp=1, both dynamic_weights modes use same VW date:
            #   - Formation date is d-1, return date is d
            #   - dynamic_weights=True: VW from d-1 (= formation date)
            #   - dynamic_weights=False: VW from formation date = d-1
            # So we can use pre-computed dynamic_weights (faster lookup)
            dynamic_weights = build_vw_lookup_and_dynamic_weights(
                date_idx, id_idx, vw, TM, n_ids
            )
            ew_ret_raw, vw_ret_raw = compute_all_returns_ultrafast(
                date_idx,        # return date indices
                id_idx,          # return bond ID indices
                returns,         # returns
                dynamic_weights, # VW weights (from previous period)
                date_idx,        # formation date indices (same data)
                id_idx,          # formation bond ID indices
                ranks,           # formation ranks
                TM, n_ids, tot_nport
            )
        else:
            # Staggered case: h>1
            # For hp>1, dynamic_weights modes differ:
            #   - True: VW from d-1 (day before return) - same for all cohorts
            #   - False: VW from formation date - different per cohort
            # Pass vw_lookup and let the function look up VW per cohort
            ew_ret_raw, vw_ret_raw = compute_staggered_returns_ultrafast(
                date_idx, id_idx, returns, vw_lookup,
                date_idx, id_idx, ranks,
                TM, n_ids, tot_nport, self.hor,
                self.dynamic_weights  # True: VW from d-1, False: VW from form_d
            )

        # Aggregate results (same format as standard path)
        return self._aggregate_fast_results(ew_ret_raw, vw_ret_raw)

    def _aggregate_fast_results(self, ew_ret: np.ndarray, vw_ret: np.ndarray):
        """Aggregate fast path results into same format as standard path."""
        # Compute long-short returns
        sort_var_main, _ = self._get_sort_vars()

        ewls, vwls, ew_long, vw_long, ew_short, vw_short = self._compute_single_sort_longshort(
            ew_ret, vw_ret
        )
        ptf_labels = get_signal_based_labels(sort_var_main, self.nport)

        # Create DataFrames
        ew_df = pd.DataFrame(ew_ret, index=self.datelist, columns=ptf_labels)
        vw_df = pd.DataFrame(vw_ret, index=self.datelist, columns=ptf_labels)

        # Create long-short DataFrames
        prefix = 'EWEP' if self._computing_ep else 'EWEA'
        vw_prefix = 'VWEP' if self._computing_ep else 'VWEA'

        ewls_df = pd.DataFrame(ewls, index=self.datelist, columns=[f'{prefix}_{self.name}'])
        vwls_df = pd.DataFrame(vwls, index=self.datelist, columns=[f'{vw_prefix}_{self.name}'])
        ew_long_df = pd.DataFrame(ew_long, index=self.datelist, columns=[f'LONG_{prefix}_{self.name}'])
        vw_long_df = pd.DataFrame(vw_long, index=self.datelist, columns=[f'LONG_{vw_prefix}_{self.name}'])
        ew_short_df = pd.DataFrame(ew_short, index=self.datelist, columns=[f'SHORT_{prefix}_{self.name}'])
        vw_short_df = pd.DataFrame(vw_short, index=self.datelist, columns=[f'SHORT_{vw_prefix}_{self.name}'])

        # Build and return StrategyResults (same format as _aggregate_results_staggered)
        return build_strategy_results(
            ewport_df=ew_df,
            vwport_df=vw_df,
            ewls_df=ewls_df,
            vwls_df=vwls_df,
            ewls_long_df=ew_long_df,
            vwls_long_df=vw_long_df,
            ewls_short_df=ew_short_df,
            vwls_short_df=vw_short_df,
            turnover_ew_df=None,  # No turnover for fast path
            turnover_vw_df=None,
            chars_ew=None,
            chars_vw=None,
        )

    def _create_empty_results(self):
        """Create empty results for edge cases."""
        TM = len(self.datelist)
        tot_nport = self._get_total_portfolios()

        ew_ret = np.full((TM, tot_nport), np.nan)
        vw_ret = np.full((TM, tot_nport), np.nan)

        return self._aggregate_fast_results(ew_ret, vw_ret)

    def _get_total_portfolios(self) -> int:
        """Get total number of portfolios (accounting for double sorts)."""
        is_double = getattr(self.strategy, "double_sort", 0) or getattr(self.strategy, "DoubleSort", 0)
        if is_double:
            nport2 = getattr(self.strategy, "num_portfolios2", None) or getattr(self.strategy, "nport2", None)
            return self.nport * nport2 if nport2 else self.nport
        return self.nport

    # =========================================================================
    # Helper Methods (Precomputation, etc.)
    # =========================================================================
    def _precompute_data(self) -> PrecomputedData:
        """Precompute time-indexed DataFrames."""
        # Get sorting variables
        sort_var_main, sort_var2 = self._get_sort_vars()

        # Determine double sort parameters
        is_double = getattr(self.strategy, "double_sort", 0) or getattr(self.strategy, "DoubleSort", 0)
        if is_double:
            nport2 = getattr(self.strategy, "num_portfolios2", None) or getattr(self.strategy, "nport2", None)
            how = getattr(self.strategy, "how", 'conditional')
            breakpoints = getattr(self.strategy, "breakpoints", None)
            breakpoints2 = getattr(self.strategy, "breakpoints2", None)
        else:
            nport2 = None
            how = 'conditional'
            breakpoints = getattr(self.strategy, "breakpoints", None)
            breakpoints2 = None

        # Determine return variable
        ret_var = ColumnNames.RETURN

        # Build using PrecomputeBuilder
        builder = PrecomputeBuilder(self)

        precomp = builder.build(
            tab=self.data[self.required_cols],
            tab_raw=self.data_raw,
            datelist=self.datelist,
            sort_var=sort_var_main,
            sort_var2=sort_var2,
            use_double_sort=bool(is_double),
            how=how,
            adj=self.adj,
            ret_var=ret_var,
            nport=self.nport,
            nport2=nport2,
            breakpoints=breakpoints,
            breakpoints2=breakpoints2,
            cached_precomp=self._cached_precomp  # Option 6: Pass cached data
        )

        # Option 6: Store shareable parts for caching by AssayAnomalyRunner
        # It0 and vw_map_t0 are independent of hp/nport
        # It1, It2, It1m, vw_map_t1m are independent of signal (for batch processing)
        self._shareable_precomp = {
            'It0': precomp.It0,
            'vw_map_t0': precomp.vw_map_t0,
            # Batch processing: return data is same for all signals
            'It1': precomp.It1,
            'It2': precomp.It2,
            'It1m': precomp.It1m,
            'vw_map_t1m': precomp.vw_map_t1m,
        }

        return precomp

    # Methods needed by PrecomputeBuilder
    def filter_by_rating(self, tab, date_t, sort_var, sort_var2, date_sub=None):
        """
        Filter by rating - needed by PrecomputeBuilder.

        Parameters
        ----------
        tab : pd.DataFrame
            Full bond panel data
        date_t : pd.Timestamp
            Date to filter for
        sort_var : str
            Primary sorting variable
        sort_var2 : str, optional
            Secondary sorting variable
        date_sub : pd.DataFrame, optional
            Pre-filtered data for this date (optimization).
            If provided, skips the date filtering step.

        Returns
        -------
        pd.DataFrame
            Filtered data for the given date
        """
        # Use pre-indexed data if available, otherwise filter by date
        if date_sub is not None:
            sub = date_sub.copy()
        else:
            sub = tab[tab[ColumnNames.DATE] == date_t].copy()

        # Remove rows where sort variables are NaN
        if sort_var in sub.columns:
            sub = sub[~sub[sort_var].isna()]
        if sort_var2 and sort_var2 in sub.columns:
            sub = sub[~sub[sort_var2].isna()]

        # Apply rating filter if specified
        if self.rating is not None:
            if isinstance(self.rating, str):
                min_r, max_r = get_rating_bounds(self.rating)
            else:
                min_r, max_r = self.rating

            sub = sub[
                (sub[ColumnNames.RATING] >= min_r) &
                (sub[ColumnNames.RATING] <= max_r)
            ]

        return sub

    def filter_by_char(self, sub, date_t, sort_var, sort_var2):
        """Filter by characteristics - needed by PrecomputeBuilder."""
        if self.subset_filter is None:
            return sub

        for col, (min_val, max_val) in self.subset_filter.items():
            if col in sub.columns:
                sub = sub[
                    (sub[col] >= min_val) &
                    (sub[col] <= max_val)
                ]

        return sub

    def filter_by_universe_matching(self, sub, adj, ret_var):
        """Filter by universe matching - needed by PrecomputeBuilder."""
        if adj in ['trim', 'bounce', 'price']:
            adj_col = f"{ret_var}_{adj}"
            if adj_col in sub.columns:
                sub = sub[~sub[adj_col].isna()]

        return sub

    def _form_cohort_portfolios(self, t_idx, date_t, precomp,
                                ew_ret_arr, vw_ret_arr, ew_chars_arr, vw_chars_arr):
        """Form portfolios for one cohort (staggered rebalancing)."""
        tot_nport = self._get_total_portfolios()

        # Loop over holding period horizons
        for h in range(self.hor):
            t1_idx = t_idx + h + 1

            # Check if we're beyond available data
            if t1_idx >= len(self.datelist):
                break

            date_t1 = self.datelist[t1_idx]

            # Get date for dynamic weights if needed
            date_t1_minus1 = None
            if self.dynamic_weights and t1_idx > 0:
                date_t1_minus1 = self.datelist[t1_idx - 1]

            # Get data for this period
            It0 = precomp.It0.get(date_t, pd.DataFrame())

            # Determine which return data to use (EA vs EP)
            if self._computing_ep:
                It1 = precomp.It2.get(date_t1, pd.DataFrame())
            else:
                It1 = precomp.It1.get(date_t1, pd.DataFrame())

            It1m = precomp.It1m.get(date_t1_minus1 if date_t1_minus1 else date_t1, pd.DataFrame())

            # Determine return column based on EA vs EP
            if self._computing_ep and self.adj:
                ret_col = f"{ColumnNames.RETURN}_{self.adj}"  # EP uses adjusted returns
            else:
                ret_col = ColumnNames.RETURN  # EA uses original returns

            # Form portfolio
            result = self._form_single_period(
                It0, It1, It1m,
                precomp.ranks_map,
                precomp.vw_map_t0,
                precomp.vw_map_t1m,
                date_t,
                date_t1,
                date_t1_minus1,
                ret_col)


            # Store results at realization time (t1_idx) and cohort dimension
            # This matches the old version: ewport_hor_ea[t + h, self.cohort, :]
            ew_ret_arr[t1_idx, self.cohort, :] = result['returns_ew']
            vw_ret_arr[t1_idx, self.cohort, :] = result['returns_vw']

            # Store characteristics if requested
            if self.chars and result['chars_ew'] is not None:
                for c in self.chars:
                    ew_chars_arr[c][t1_idx, self.cohort, :] = result['chars_ew'][c].values
                    vw_chars_arr[c][t1_idx, self.cohort, :] = result['chars_vw'][c].values

            # Handle turnover if requested
            if self.turnover and not result['weights_df'].empty:
                self.turnover_manager.accumulate(
                    self.turnover_state,
                    self.cohort,  # cohort index (t % hor)
                    tot_nport,
                    t_idx,  # tau (time index for formation)
                    result['weights_df'],
                    result['weights_scaled_df']
                )
                # Set zero turnover for active cohorts that are NOT rebalancing
                # (they hold positions and don't trade)
                self.turnover_manager.set_zero_for_holding_cohorts(
                    self.turnover_state,
                    self.cohort,  # rebalancing cohort
                    tot_nport,
                    t_idx  # tau
                )

    def _form_nonstaggered_portfolio(self, rebal_idx, precomp,
                                    ew_ret_arr, vw_ret_arr, ew_chars_arr, vw_chars_arr,
                                    rebal_dates_idx=None):
        """Form portfolio for one rebalancing period (non-staggered).

        Phase 17 fix: Now iterates through ALL months until next rebalancing,
        not just holding_period months.
        """
        tot_nport = self._get_total_portfolios()
        date_t = self.datelist[rebal_idx]

        # Phase 17: Find next rebalancing date to determine how many months to iterate
        if rebal_dates_idx is not None:
            # Find next rebalancing date
            next_rebal_idx = len(self.datelist)  # default: end of data
            for idx in rebal_dates_idx:
                if idx > rebal_idx:
                    next_rebal_idx = idx
                    break
            # Iterate from rebal_idx+1 to next_rebal_idx (inclusive)
            # This gives us all return dates until (and including) the next rebalancing date
            n_months = next_rebal_idx - rebal_idx
        else:
            # Fallback to old behavior for backward compatibility
            n_months = self.hor

        # Loop over all months until next rebalancing
        for h in range(n_months):
            t1_idx = rebal_idx + h + 1

            if t1_idx >= len(self.datelist):
                break

            date_t1 = self.datelist[t1_idx]

            # For non-staggered rebalancing, always use VW from formation date (not d-1)
            # This is the correct behavior per PyBondLab specification
            # date_t1_minus1 = None means use VW from formation date (date_t)
            date_t1_minus1 = None  # Always None for non-staggered

            # Get data
            It0 = precomp.It0.get(date_t, pd.DataFrame())

            # Determine which return data to use (EA vs EP)
            if self._computing_ep:
                # EP: Always use filtered returns from It2
                It1 = precomp.It2.get(date_t1, pd.DataFrame())
            else:
                # EA: Always use unfiltered returns from It1
                It1 = precomp.It1.get(date_t1, pd.DataFrame())

            # For non-staggered, always use VW from formation date (date_t)
            It1m = precomp.It1m.get(date_t, pd.DataFrame())

            # Form portfolio
            result = self._form_single_period(
                It0, It1, It1m,
                precomp.ranks_map,
                precomp.vw_map_t0,
                precomp.vw_map_t1m,
                date_t,
                date_t1,
                date_t1_minus1,
                ColumnNames.RETURN if self.adj not in ['trim', 'price', 'bounce'] else f"{ColumnNames.RETURN}_{self.adj}"
            )

            # Store results at t1_idx (not rebal_idx)
            ew_ret_arr[t1_idx, :] = result['returns_ew']
            vw_ret_arr[t1_idx, :] = result['returns_vw']

            # Store characteristics
            if self.chars and result['chars_ew'] is not None:
                for c in self.chars:
                    ew_chars_arr[c][t1_idx, :] = result['chars_ew'][c].values
                    vw_chars_arr[c][t1_idx, :] = result['chars_vw'][c].values

            # Handle turnover
            if self.turnover and not result['weights_df'].empty:
                self.turnover_manager.compute(
                    self.turnover_state,
                    result['weights_df'],
                    result['weights_scaled_df'],
                    t1_idx, tot_nport
                )

    def _form_single_period(
        self,
        It0: pd.DataFrame,
        It1: pd.DataFrame,
        It1m: pd.DataFrame,
        ranks_map: Dict,
        vw_map_t0: Dict,
        vw_map_t1m: Dict,
        date_t: pd.Timestamp,
        date_t1: pd.Timestamp,
        date_t1_minus1: Optional[pd.Timestamp],
        ret_col: str
    ) -> Dict:
        """
        Form portfolio for a single period.

        Parameters
        ----------
        date_t : pd.Timestamp
            Formation date (used for rank lookup)
        date_t1 : pd.Timestamp
            Return date (used for port_idx key - results indexed by return date)
        """
        from .utils import intersect_id

        tot_nport = self._get_total_portfolios()

        # Handle empty data
        if It0.empty or It1.empty:
            return self._create_nan_result(tot_nport)

        # Intersect IDs
        It0, It1, It1m = intersect_id(It0, It1, It1m, self.dynamic_weights)

        if It0.shape[0] == 0:
            return self._create_nan_result(tot_nport)

        # Map ranks
        It1['ptf_rank'] = It1[ColumnNames.ID].map(
            ranks_map.get(date_t, pd.Series(dtype='Int64'))
        )
        It1 = It1.dropna(subset=['ptf_rank'])

        if It1.empty:
            return self._create_nan_result(tot_nport)

        It1['ptf_rank'] = It1['ptf_rank'].astype(int)

        # Get value weights
        if self.dynamic_weights and date_t1_minus1 is not None:
            vw_map = vw_map_t1m.get(date_t1_minus1, pd.Series(dtype=float))
        else:
            vw_map = vw_map_t0.get(date_t, pd.Series(dtype=float))

        It1[ColumnNames.VALUE_WEIGHT] = It1[ColumnNames.ID].map(vw_map)

        # Apply banding if needed
        if self.banding_threshold is not None:
            It1 = self._apply_banding_to_period(It1, tot_nport)

        # Extract numpy arrays for numba processing
        ranks_arr = It1['ptf_rank'].values.astype(np.float64)
        returns_arr = It1[ret_col].values.astype(np.float64)
        vw_arr = It1[ColumnNames.VALUE_WEIGHT].values.astype(np.float64)

        # Compute weights using numba kernel (replaces groupby)
        eweights_arr, vweights_arr, counts_arr = compute_portfolio_weights_single(
            ranks_arr, vw_arr, tot_nport
        )
        It1['weights'] = vweights_arr
        It1['eweights'] = eweights_arr
        It1['count'] = counts_arr

        # Compute portfolio returns using numba kernel (replaces groupby)
        ew_ret_arr, vw_ret_arr = compute_portfolio_returns_single(
            ranks_arr, returns_arr, vw_arr, tot_nport
        )

        # Convert to pandas Series with proper index (for compatibility)
        nport_idx = range(1, tot_nport + 1)
        ptf_ret_ew = pd.Series(ew_ret_arr, index=nport_idx)
        ptf_ret_vw = pd.Series(vw_ret_arr, index=nport_idx)

        # Prepare weight outputs
        weights_df = pd.DataFrame()
        weights_scaled_df = pd.DataFrame()

        if self.turnover or self.save_idx:
            # Build rank DataFrame using already-computed weights
            rank = It1[[ColumnNames.ID, 'ptf_rank', ret_col, 'eweights', 'weights', 'count']].copy()
            rank = rank.rename(columns={ret_col: 'ret', 'weights': 'vweights'})
            rank['vweights'] = rank['vweights'].fillna(0.0)

            # For WithinFirmSort, also include VW column (needed for custom aggregation)
            if self.strategy.__strategy_name__ == "Within-Firm Sort":
                rank['VW'] = It1[ColumnNames.VALUE_WEIGHT].values
                weights_df = rank[[ColumnNames.ID, 'ptf_rank', 'eweights', 'vweights', 'ret', 'VW']]
            else:
                weights_df = rank[[ColumnNames.ID, 'ptf_rank', 'eweights', 'vweights']]

            if self.save_idx:
                if not hasattr(self, 'port_idx'):
                    self.port_idx = {}
                self.port_idx[date_t1] = weights_df

            # Scaled weights using numba kernel
            ew_scaled_arr, vw_scaled_arr = compute_scaled_weights_single(
                ranks_arr, returns_arr, eweights_arr, vweights_arr,
                counts_arr, ew_ret_arr, vw_ret_arr, tot_nport
            )

            weights_scaled_df = pd.DataFrame({
                ColumnNames.ID: It1[ColumnNames.ID].values,
                'ptf_rank': It1['ptf_rank'].values,
                'eweights': ew_scaled_arr,
                'vweights': vw_scaled_arr
            })
            weights_scaled_df['vweights'] = weights_scaled_df['vweights'].fillna(0.0)

        # Compute characteristics using numba kernels
        chars_ew = None
        chars_vw = None
        if self.chars:
            # Merge to get characteristics aligned with returns data
            nm = [ColumnNames.ID, 'ptf_rank', 'weights']
            sub = It1[nm]
            It1m_aug = It1m.merge(sub, on=ColumnNames.ID, how='inner')

            # Extract arrays for numba processing
            char_ranks = It1m_aug['ptf_rank'].values.astype(np.float64)
            char_weights = It1m_aug['weights'].values.astype(np.float64)

            chars_ew = pd.DataFrame(index=nport_idx)
            chars_vw = pd.DataFrame(index=nport_idx)
            for c in self.chars:
                char_values = It1m_aug[c].values.astype(np.float64)
                c_ew_arr, c_vw_arr = compute_characteristics_single(
                    char_ranks, char_weights, char_values, tot_nport
                )
                chars_ew[c] = c_ew_arr
                chars_vw[c] = c_vw_arr

        return {
            'returns_ew': ptf_ret_ew.tolist(),
            'returns_vw': ptf_ret_vw.tolist(),
            'weights_df': weights_df,
            'weights_scaled_df': weights_scaled_df,
            'chars_ew': chars_ew,
            'chars_vw': chars_vw
        }

    def _create_nan_result(self, tot_nport: int) -> Dict:
        """Create a NaN result for periods with no data."""
        nan_list = [np.nan] * tot_nport
        result = {
            'returns_ew': nan_list,
            'returns_vw': nan_list,
            'weights_df': pd.DataFrame(),
            'weights_scaled_df': pd.DataFrame(),
            'chars_ew': None,
            'chars_vw': None
        }

        if self.chars:
            nan_df = pd.DataFrame(np.full((tot_nport, len(self.chars)), np.nan), columns=self.chars)
            result['chars_ew'] = nan_df
            result['chars_vw'] = nan_df

        return result

    def _apply_banding_to_period(self, It1: pd.DataFrame, nportmax: int) -> pd.DataFrame:
        """Apply banding to a period's ranks."""
        # Get appropriate cohort key
        if hasattr(self, 'cohort'):
            cohort_key = self.cohort
        elif hasattr(self, 'current_rebal_date'):
            cohort_key = self.current_rebal_date
        else:
            return It1

        prev = self.lag_rank.get(cohort_key)

        if prev is None or (hasattr(prev, "empty") and prev.empty):
            # First time - save current ranks
            self.lag_rank[cohort_key] = It1[[ColumnNames.ID, 'ptf_rank']].copy()
        else:
            # Apply banding
            It1 = It1.merge(prev, on=ColumnNames.ID, how='left', suffixes=('_current', '_lag'))
            It1['ptf_rank'] = self.calculate_qnew_vectorized(
                It1['ptf_rank_lag'], It1['ptf_rank_current'], nportmax, self.banding_threshold
            )
            # Update lag_rank
            self.lag_rank[cohort_key] = It1[[ColumnNames.ID, 'ptf_rank']].copy()

        return It1

    def calculate_qnew_vectorized(self, q_old, q_sig, nport, threshold):
        """Vectorized banding calculation."""
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

    def _compute_single_sort_longshort(self, ew_ret, vw_ret):
        """
        Compute long-short for single sort.

        Long = last portfolio (highest)
        Short = first portfolio (lowest)
        LS = Long - Short

        Parameters
        ----------
        ew_ret : np.ndarray
            EW portfolio returns (time × nport)
        vw_ret : np.ndarray
            VW portfolio returns (time × nport)

        Returns
        -------
        tuple
            (ewls, vwls, ew_long, vw_long, ew_short, vw_short)
        """
        ew_long = ew_ret[:, -1]
        ew_short = ew_ret[:, 0]
        ewls = ew_long - ew_short

        vw_long = vw_ret[:, -1]
        vw_short = vw_ret[:, 0]
        vwls = vw_long - vw_short

        return ewls, vwls, ew_long, vw_long, ew_short, vw_short

    def _compute_double_sort_longshort(self, ew_ret, vw_ret, n1, n2):
        """
        Compute long-short for double sort.

        For each primary portfolio i, compute:
            LS_i = portfolio(i, n2) - portfolio(i, 1)
        Then average across all primary portfolios.

        This matches the old PyBondLab behavior.

        Parameters
        ----------
        ew_ret : np.ndarray
            EW portfolio returns (time × n1*n2)
        vw_ret : np.ndarray
            VW portfolio returns (time × n1*n2)
        n1 : int
            Number of primary portfolios
        n2 : int
            Number of secondary portfolios

        Returns
        -------
        tuple
            (ewls, vwls, ew_long, vw_long, ew_short, vw_short)
            where long/short are also averaged across primary bins
        """
        ew_ls_by_primary = []
        vw_ls_by_primary = []
        ew_long_by_primary = []
        vw_long_by_primary = []
        ew_short_by_primary = []
        vw_short_by_primary = []

        for i in range(n1):
            # For primary portfolio i:
            # Long = portfolio (i, n2) - last in this primary bin
            # Short = portfolio (i, 1) - first in this primary bin
            long_idx = i * n2 + (n2 - 1)
            short_idx = i * n2

            ew_long_i = ew_ret[:, long_idx]
            ew_short_i = ew_ret[:, short_idx]
            ew_ls_i = ew_long_i - ew_short_i

            vw_long_i = vw_ret[:, long_idx]
            vw_short_i = vw_ret[:, short_idx]
            vw_ls_i = vw_long_i - vw_short_i

            ew_ls_by_primary.append(ew_ls_i)
            vw_ls_by_primary.append(vw_ls_i)
            ew_long_by_primary.append(ew_long_i)
            vw_long_by_primary.append(vw_long_i)
            ew_short_by_primary.append(ew_short_i)
            vw_short_by_primary.append(vw_short_i)

        # Average across primary portfolios
        ewls = np.mean(ew_ls_by_primary, axis=0)
        vwls = np.mean(vw_ls_by_primary, axis=0)
        ew_long = np.mean(ew_long_by_primary, axis=0)
        vw_long = np.mean(vw_long_by_primary, axis=0)
        ew_short = np.mean(ew_short_by_primary, axis=0)
        vw_short = np.mean(vw_short_by_primary, axis=0)

        return ewls, vwls, ew_long, vw_long, ew_short, vw_short

    def _aggregate_within_firm_results(self, ew_ret_arr, vw_ret_arr,
                                      ew_chars_arr, vw_chars_arr):
        """
        Aggregate results for WithinFirmSort strategy using custom aggregation.

        This implements the within-firm return aggregation scheme:
        - Group bonds by rating tercile and firm
        - Compute firm-level VW returns
        - Aggregate across firms (firm cap-weighted)
        - Average across rating groups
        """
        from .utils_within_firm import compute_within_firm_returns_aggregation

        # Get strategy parameters
        firm_id_col = getattr(self.strategy, 'firm_id_col', 'PERMNO')
        rating_bins = getattr(self.strategy, 'rating_bins', [-np.inf, 7, 10, np.inf])

        # Use custom aggregation
        custom_returns = compute_within_firm_returns_aggregation(
            portfolio_indices=self.port_idx,
            data_raw=self.data_raw,
            datelist=self.datelist,
            firm_id_col=firm_id_col,
            rating_col=ColumnNames.RATING,
            rating_bins=rating_bins
        )

        # Extract custom returns and align with datelist
        # custom_returns may have fewer dates than datelist
        # Now we have both EW and VW properly computed:
        # - EW: EW within firm → equal-weight across firms → avg across ratings
        # - VW: VW within firm → cap-weight across firms → avg across ratings

        # Use new keys if available (fast path), fall back to legacy keys (slow path)
        if 'ew_long_short' in custom_returns:
            ewls_series = custom_returns['ew_long_short'].reindex(self.datelist)
            ew_long_series = custom_returns['ew_long_leg'].reindex(self.datelist)
            ew_short_series = custom_returns['ew_short_leg'].reindex(self.datelist)
            vwls_series = custom_returns['vw_long_short'].reindex(self.datelist)
            vw_long_series = custom_returns['vw_long_leg'].reindex(self.datelist)
            vw_short_series = custom_returns['vw_short_leg'].reindex(self.datelist)
        else:
            # Legacy slow path only computes VW
            vwls_series = custom_returns['long_short'].reindex(self.datelist)
            vw_long_series = custom_returns['long_leg'].reindex(self.datelist)
            vw_short_series = custom_returns['short_leg'].reindex(self.datelist)
            ewls_series = vwls_series
            ew_long_series = vw_long_series
            ew_short_series = vw_short_series

        ewls = ewls_series.values
        ew_long = ew_long_series.values
        ew_short = ew_short_series.values
        vwls = vwls_series.values
        vw_long = vw_long_series.values
        vw_short = vw_short_series.values

        # Create portfolio labels
        sort_var_main, _ = self._get_sort_vars()
        ptf_labels = ['LOW', 'HIGH']

        # Create placeholder portfolio returns (we don't use these for within-firm)
        # But need them for compatibility with StrategyResults structure
        ew_ret = np.zeros((len(self.datelist), 2))
        vw_ret = np.zeros((len(self.datelist), 2))

        # Fill with long/short leg returns for display
        for i, date_t in enumerate(self.datelist):
            if date_t in custom_returns['long_short'].index:
                idx = custom_returns['long_short'].index.get_loc(date_t)
                ew_ret[i, 0] = custom_returns['short_leg'].iloc[idx]
                ew_ret[i, 1] = custom_returns['long_leg'].iloc[idx]
                vw_ret[i, 0] = custom_returns['short_leg'].iloc[idx]
                vw_ret[i, 1] = custom_returns['long_leg'].iloc[idx]
            else:
                ew_ret[i, :] = np.nan
                vw_ret[i, :] = np.nan

        # Create DataFrames
        ew_df = pd.DataFrame(ew_ret, index=self.datelist, columns=ptf_labels)
        vw_df = pd.DataFrame(vw_ret, index=self.datelist, columns=ptf_labels)

        # Create long-short DataFrames with proper naming (matching old version)
        # Use EA/EP prefix depending on whether filters are being computed
        prefix = 'EWEP' if self._computing_ep else 'EWEA'
        vw_prefix = 'VWEP' if self._computing_ep else 'VWEA'

        ewls_df = pd.DataFrame(ewls, index=self.datelist, columns=[f'{prefix}_{self.name}'])
        vwls_df = pd.DataFrame(vwls, index=self.datelist, columns=[f'{vw_prefix}_{self.name}'])
        ew_long_df = pd.DataFrame(ew_long, index=self.datelist, columns=[f'LONG_{prefix}_{self.name}'])
        vw_long_df = pd.DataFrame(vw_long, index=self.datelist, columns=[f'LONG_{vw_prefix}_{self.name}'])
        ew_short_df = pd.DataFrame(ew_short, index=self.datelist, columns=[f'SHORT_{prefix}_{self.name}'])
        vw_short_df = pd.DataFrame(vw_short, index=self.datelist, columns=[f'SHORT_{vw_prefix}_{self.name}'])

        # Handle characteristics using hierarchical aggregation
        chars_ew_dict = None
        chars_vw_dict = None

        if self.chars and self.port_idx:
            from .numba_core import compute_within_firm_chars_aggregation

            # Collect portfolio data from all dates
            all_port_dfs = []
            for date_t in self.datelist:
                if date_t not in self.port_idx:
                    continue
                port_df = self.port_idx[date_t]
                if port_df.empty:
                    continue
                port_df_copy = port_df.copy()
                port_df_copy[ColumnNames.DATE] = date_t
                all_port_dfs.append(port_df_copy)

            if all_port_dfs:
                combined_df = pd.concat(all_port_dfs, ignore_index=True)

                # Get firm and rating lookup from raw data
                firm_rating_lookup = self.data_raw[
                    [ColumnNames.ID, firm_id_col, ColumnNames.RATING]
                ].drop_duplicates()
                combined_df = combined_df.merge(firm_rating_lookup, on=ColumnNames.ID, how='left')

                # Create rating terciles
                combined_df['rating_terc'] = pd.cut(
                    pd.to_numeric(combined_df[ColumnNames.RATING], errors='coerce'),
                    bins=rating_bins,
                    labels=[1, 2, 3],
                    include_lowest=True
                ).astype('Int64').fillna(0).values

                # Get char values from raw data at FORMATION date (t-1, not return date)
                # port_idx is indexed by return date (t+1), so we need chars from t = t+1 - 1
                char_cols = [ColumnNames.ID, ColumnNames.DATE] + list(self.chars)
                char_data = self.data_raw[char_cols].copy()

                # Create formation date column (return_date - 1 period)
                date_to_prev = {self.datelist[i]: self.datelist[i-1] for i in range(1, len(self.datelist))}
                combined_df['form_date'] = combined_df[ColumnNames.DATE].map(date_to_prev)

                # Merge chars using formation date
                char_data = char_data.rename(columns={ColumnNames.DATE: 'form_date'})
                combined_df = combined_df.merge(char_data, on=[ColumnNames.ID, 'form_date'], how='left')

                # Create mappings
                unique_firms = combined_df[firm_id_col].dropna().unique()
                firm_to_idx = {f: i for i, f in enumerate(unique_firms)}
                n_firms = len(unique_firms)

                date_to_idx = {d: i for i, d in enumerate(self.datelist)}
                n_dates = len(self.datelist)

                # Convert to numpy arrays
                date_idx = combined_df[ColumnNames.DATE].map(date_to_idx).values.astype(np.int64)
                id_idx = np.zeros(len(combined_df), dtype=np.int64)  # Not used in aggregation
                firm_idx = combined_df[firm_id_col].map(firm_to_idx).fillna(-1).values.astype(np.int64)
                rating_terc = combined_df['rating_terc'].values.astype(np.float64)
                ptf_rank = combined_df['ptf_rank'].values.astype(np.float64)
                vw = combined_df[ColumnNames.VALUE_WEIGHT].values.astype(np.float64)

                # Compute chars for each characteristic
                chars_ew_dict = {}
                chars_vw_dict = {}
                ptf_labels_chars = ['LOW', 'HIGH']

                for char_name in self.chars:
                    char_values = combined_df[char_name].values.astype(np.float64)

                    ew_low, ew_high, vw_low, vw_high = compute_within_firm_chars_aggregation(
                        date_idx, id_idx, firm_idx, rating_terc, ptf_rank,
                        char_values, vw, n_dates, n_firms
                    )

                    chars_ew_dict[char_name] = pd.DataFrame(
                        np.column_stack([ew_low, ew_high]),
                        index=self.datelist,
                        columns=ptf_labels_chars
                    )
                    chars_vw_dict[char_name] = pd.DataFrame(
                        np.column_stack([vw_low, vw_high]),
                        index=self.datelist,
                        columns=ptf_labels_chars
                    )

        # Finalize turnover (use standard PyBondLab machinery)
        turnover_ew = None
        turnover_vw = None
        if self.turnover:
            turnover_ew, turnover_vw = self.turnover_manager.finalize(
                self.turnover_state, ptf_labels)

        # Build and return StrategyResults
        return build_strategy_results(
            ewport_df=ew_df,
            vwport_df=vw_df,
            ewls_df=ewls_df,
            vwls_df=vwls_df,
            ewls_long_df=ew_long_df,
            vwls_long_df=vw_long_df,
            ewls_short_df=ew_short_df,
            vwls_short_df=vw_short_df,
            turnover_ew_df=turnover_ew,
            turnover_vw_df=turnover_vw,
            chars_ew=chars_ew_dict,
            chars_vw=chars_vw_dict,
        )

    def _aggregate_results_staggered(self, ew_ret_arr, vw_ret_arr,
                                    ew_chars_arr, vw_chars_arr):
        """Aggregate staggered portfolio results."""
        # Check if this is WithinFirmSort - if so, use custom aggregation
        is_within_firm = self.strategy.__strategy_name__ == "Within-Firm Sort"

        if is_within_firm:
            return self._aggregate_within_firm_results(ew_ret_arr, vw_ret_arr,
                                                       ew_chars_arr, vw_chars_arr)

        # Standard aggregation (existing code)
        # Average over horizons
        # Suppress "Mean of empty slice" warning - expected when some portfolios have no bonds
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)
            ew_ret = np.nanmean(ew_ret_arr, axis=1)  # (TM, nport)
            vw_ret = np.nanmean(vw_ret_arr, axis=1)

        # Compute long-short returns
        is_double = getattr(self.strategy, "double_sort", 0) or getattr(self.strategy, "DoubleSort", 0)
        sort_var_main, sort_var2 = self._get_sort_vars()

        if is_double:
            nport2 = getattr(self.strategy, "num_portfolios2", None) or getattr(self.strategy, "nport2", None)
            ewls, vwls, ew_long, vw_long, ew_short, vw_short = self._compute_double_sort_longshort(
                ew_ret, vw_ret, self.nport, nport2
            )
            ptf_labels = get_signal_based_labels(sort_var_main, self.nport, sort_var2, nport2)
        else:
            ewls, vwls, ew_long, vw_long, ew_short, vw_short = self._compute_single_sort_longshort(
                ew_ret, vw_ret
            )
            ptf_labels = get_signal_based_labels(sort_var_main, self.nport)

        # Create DataFrames
        ew_df = pd.DataFrame(ew_ret, index=self.datelist, columns=ptf_labels)
        vw_df = pd.DataFrame(vw_ret, index=self.datelist, columns=ptf_labels)


        # Create long-short DataFrames with proper naming (matching old version)
        # Use EA/EP prefix depending on whether filters are being computed
        prefix = 'EWEP' if self._computing_ep else 'EWEA'
        vw_prefix = 'VWEP' if self._computing_ep else 'VWEA'

        ewls_df = pd.DataFrame(ewls, index=self.datelist, columns=[f'{prefix}_{self.name}'])
        vwls_df = pd.DataFrame(vwls, index=self.datelist, columns=[f'{vw_prefix}_{self.name}'])
        ew_long_df = pd.DataFrame(ew_long, index=self.datelist, columns=[f'LONG_{prefix}_{self.name}'])
        vw_long_df = pd.DataFrame(vw_long, index=self.datelist, columns=[f'LONG_{vw_prefix}_{self.name}'])
        ew_short_df = pd.DataFrame(ew_short, index=self.datelist, columns=[f'SHORT_{prefix}_{self.name}'])
        vw_short_df = pd.DataFrame(vw_short, index=self.datelist, columns=[f'SHORT_{vw_prefix}_{self.name}'])


        # Handle characteristics
        chars_ew_dict = None
        chars_vw_dict = None
        if self.chars:
            chars_ew_dict = {}
            chars_vw_dict = {}
            for c in self.chars:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)
                    chars_ew_dict[c] = pd.DataFrame(
                        np.nanmean(ew_chars_arr[c], axis=1),
                        index=self.datelist,
                        columns=ptf_labels
                    )
                    chars_vw_dict[c] = pd.DataFrame(
                        np.nanmean(vw_chars_arr[c], axis=1),
                        index=self.datelist,
                        columns=ptf_labels
                    )

        # Finalize turnover
        turnover_ew = None
        turnover_vw = None
        if self.turnover:
            turnover_ew, turnover_vw = self.turnover_manager.finalize(
                self.turnover_state, ptf_labels)

        # Build and return StrategyResults
        return build_strategy_results(
            ewport_df=ew_df,
            vwport_df=vw_df,
            ewls_df=ewls_df,
            vwls_df=vwls_df,
            ewls_long_df=ew_long_df,
            vwls_long_df=vw_long_df,
            ewls_short_df=ew_short_df,
            vwls_short_df=vw_short_df,
            turnover_ew_df=turnover_ew,
            turnover_vw_df=turnover_vw,
            chars_ew=chars_ew_dict,
            chars_vw=chars_vw_dict,
        )

    def _aggregate_results_nonstaggered(self, ew_ret_arr, vw_ret_arr,
                                       ew_chars_arr, vw_chars_arr):
        """Aggregate non-staggered portfolio results."""
        # tot_nport = self._get_total_portfolios()
        # ptf_labels = get_portfolio_labels(tot_nport)



         # Compute long-short returns
        is_double = getattr(self.strategy, "double_sort", 0) or getattr(self.strategy, "DoubleSort", 0)
        sort_var_main, sort_var2 = self._get_sort_vars()

        if is_double:
            nport2 = getattr(self.strategy, "num_portfolios2", None) or getattr(self.strategy, "nport2", None)
            ewls, vwls, ew_long, vw_long, ew_short, vw_short = self._compute_double_sort_longshort(
                ew_ret_arr, vw_ret_arr, self.nport, nport2
            )
            ptf_labels = get_signal_based_labels(sort_var_main, self.nport, sort_var2, nport2)

        else:
            ewls, vwls, ew_long, vw_long, ew_short, vw_short = self._compute_single_sort_longshort(
                ew_ret_arr, vw_ret_arr
            )
            ptf_labels = get_signal_based_labels(sort_var_main, self.nport)

        # Create DataFrames (already in correct shape)
        ew_df = pd.DataFrame(ew_ret_arr, index=self.datelist, columns=ptf_labels)
        vw_df = pd.DataFrame(vw_ret_arr, index=self.datelist, columns=ptf_labels)

        # Create long-short DataFrames with proper naming (matching old version)
        # Use EA/EP prefix depending on whether filters are being computed
        prefix = 'EWEP' if self._computing_ep else 'EWEA'
        vw_prefix = 'VWEP' if self._computing_ep else 'VWEA'

        ewls_df = pd.DataFrame(ewls, index=self.datelist, columns=[f'{prefix}_{self.name}'])
        vwls_df = pd.DataFrame(vwls, index=self.datelist, columns=[f'{vw_prefix}_{self.name}'])
        ew_long_df = pd.DataFrame(ew_long, index=self.datelist, columns=[f'LONG_{prefix}_{self.name}'])
        vw_long_df = pd.DataFrame(vw_long, index=self.datelist, columns=[f'LONG_{vw_prefix}_{self.name}'])
        ew_short_df = pd.DataFrame(ew_short, index=self.datelist, columns=[f'SHORT_{prefix}_{self.name}'])
        vw_short_df = pd.DataFrame(vw_short, index=self.datelist, columns=[f'SHORT_{vw_prefix}_{self.name}'])

        # Handle characteristics
        chars_ew_dict = None
        chars_vw_dict = None
        if self.chars:
            chars_ew_dict = {}
            chars_vw_dict = {}
            for c in self.chars:
                chars_ew_dict[c] = pd.DataFrame(
                    ew_chars_arr[c],
                    index=self.datelist,
                    columns=ptf_labels
                )
                chars_vw_dict[c] = pd.DataFrame(
                    vw_chars_arr[c],
                    index=self.datelist,
                    columns=ptf_labels
                )

        # Finalize turnover
        turnover_ew = None
        turnover_vw = None
        if self.turnover:
            turnover_ew, turnover_vw = self.turnover_manager.finalize(
                self.turnover_state, ptf_labels
            )

        # Build and return StrategyResults
        return build_strategy_results(
            ewport_df=ew_df,
            vwport_df=vw_df,
            ewls_df=ewls_df,
            vwls_df=vwls_df,
            ewls_long_df=ew_long_df,
            vwls_long_df=vw_long_df,
            ewls_short_df=ew_short_df,
            vwls_short_df=vw_short_df,
            turnover_ew_df=turnover_ew,
            turnover_vw_df=turnover_vw,
            chars_ew=chars_ew_dict,
            chars_vw=chars_vw_dict,
        )


    # =========================================================================
    # Utility Methods
    # =========================================================================
    def summarize_portfolio_composition(self) -> pd.DataFrame:
        """
        Summarize portfolio composition over time.

        Returns
        -------
        pd.DataFrame
            Summary statistics by date (number of bonds in long, short, etc.)
        """
        if not hasattr(self, 'port_idx') or self.port_idx is None:
            raise ValueError("No portfolio indices saved. Set save_idx=True when initializing.")

        return summarize_ranks(self.port_idx)


# #### OLD CODE



def load_breakpoints_WRDS() -> pd.DataFrame:
    """
    Load the breakpoints (rolling percentiles) WRDS data
    """
    return load()
