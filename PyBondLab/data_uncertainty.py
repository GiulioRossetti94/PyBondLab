# -*- coding: utf-8 -*-
"""
Data Uncertainty Analysis Module
================================

High-level wrapper for running data uncertainty analysis across multiple
holding periods, filter configurations, and signals.

Example Usage:
--------------
>>> from PyBondLab import DataUncertaintyAnalysis
>>> results = DataUncertaintyAnalysis(
...     data=data,
...     signals=['momentum'],
...     holding_periods=[1, 3, 6],
...     filters={
...         'trim': [0.2, 0.5],
...         'price': [50, 200],
...         'wins': [(99, 'both')],
...     },
...     columns={  # Map your column names to expected names
...         'date': 'date',
...         'ID': 'cusip_id',
...         'ret': 'ret',
...         'VW': 'mcap_e',
...         'RATING_NUM': 'spc_rat',
...         'PRICE': 'prc_eom',
...     },
...     n_jobs=4,
... ).fit()
>>> print(results.summary())
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple
from numbers import Number

# Type alias for subset filter
SubsetFilter = Dict[str, Tuple[float, float]]
from concurrent.futures import ProcessPoolExecutor, as_completed
import platform

import numpy as np
import pandas as pd

# Import PyBondLab components
from .PyBondLab import StrategyFormation
from .StrategyClass import SingleSort, Momentum, LTreversal


# =============================================================================
# Default Column Mapping
# =============================================================================

# Default expected column names (PyBondLab internal names)
DEFAULT_COLUMNS = {
    'date': 'date',           # Date column
    'ID': 'ID',               # Bond identifier
    'ret': 'ret',             # Return column
    'VW': 'VW',               # Value weight column
    'RATING_NUM': 'RATING_NUM',  # Rating column (optional)
    'PRICE': 'PRICE',         # Price column (for price filters, optional)
}

# Required columns (must be present)
# Note: RATING_NUM is required by StrategyFormation even if no rating filter is used
REQUIRED_COLUMNS = {'date', 'ID', 'ret', 'VW', 'RATING_NUM'}

# Optional columns (used if available)
OPTIONAL_COLUMNS = {'PRICE'}


# =============================================================================
# Configuration Dataclass
# =============================================================================

@dataclass
class FilterConfig:
    """Configuration for a single filter."""
    filter_type: str          # 'baseline', 'trim', 'price', 'bounce', 'wins'
    level: Any                # Filter level (float, list, or tuple)
    location: Optional[str]   # 'left', 'right', 'both', or None

    def to_pbl_filter(self) -> Optional[Dict]:
        """Convert to PyBondLab filter dict format."""
        if self.filter_type == 'baseline':
            return None

        if self.filter_type == 'wins':
            return {
                'adj': 'wins',
                'level': self.level,
                'location': self.location
            }
        else:
            return {
                'adj': self.filter_type,
                'level': self.level
            }

    def get_column_suffix(self) -> str:
        """Generate column name suffix for this filter."""
        if self.filter_type == 'baseline':
            return 'baseline'

        if isinstance(self.level, (list, tuple)) and self.filter_type != 'wins':
            level_str = f"{self.level[0]}_{self.level[1]}"
        else:
            level_str = str(self.level)

        if self.filter_type == 'wins':
            return f"wins_{level_str}_{self.location}"
        elif self.location and isinstance(self.level, (list, tuple)):
            return f"{self.filter_type}_{level_str}"
        else:
            return f"{self.filter_type}_{level_str}"


@dataclass
class AnalysisConfig:
    """Configuration for a single analysis run."""
    signal: str
    hp: int
    filter_config: FilterConfig
    column_name: str


# =============================================================================
# Newey-West T-Statistics
# =============================================================================

def compute_newey_west_tstat(series: pd.Series) -> Tuple[float, float, int]:
    """
    Compute Newey-West t-statistic for a time series.

    Parameters
    ----------
    series : pd.Series
        Time series of returns

    Returns
    -------
    Tuple[float, float, int]
        (mean, t-statistic, n_obs)
    """
    # Drop NaN values
    clean_series = series.dropna()
    n_obs = len(clean_series)

    if n_obs < 2:
        return np.nan, np.nan, n_obs

    mean = clean_series.mean()

    # Lag length: int(T^0.25)
    lag = int(n_obs ** 0.25)

    try:
        from statsmodels.stats.sandwich_covariance import cov_hac
        from statsmodels.regression.linear_model import OLS
        import statsmodels.api as sm

        # Regress returns on constant to get NW standard error
        X = sm.add_constant(np.ones(n_obs))
        model = OLS(clean_series.values, X[:, :1])  # Just constant
        results = model.fit()

        # Get HAC standard errors
        cov = cov_hac(results, nlags=lag)
        se = np.sqrt(cov[0, 0])

        if se > 0:
            tstat = mean / se
        else:
            tstat = np.nan

    except ImportError:
        # Fallback: simple t-stat without NW correction
        warnings.warn("statsmodels not available, using simple t-stat")
        se = clean_series.std() / np.sqrt(n_obs)
        tstat = mean / se if se > 0 else np.nan

    return mean, tstat, n_obs


# =============================================================================
# Results Container
# =============================================================================

class DataUncertaintyResults:
    """
    Container for data uncertainty analysis results.

    Provides access to factor returns (EW/VW × EA/EP), summary statistics
    with Newey-West t-statistics, and filtering capabilities.

    Attributes
    ----------
    ew_ea : pd.DataFrame
        EW Ex-Ante long-short factors (dates × configs)
    vw_ea : pd.DataFrame
        VW Ex-Ante long-short factors (dates × configs)
    ew_ep : pd.DataFrame
        EW Ex-Post long-short factors (dates × configs)
    vw_ep : pd.DataFrame
        VW Ex-Post long-short factors (dates × configs)
    configs : pd.DataFrame
        Metadata for all configurations
    """

    def __init__(
        self,
        ew_ea: pd.DataFrame,
        vw_ea: pd.DataFrame,
        ew_ep: pd.DataFrame,
        vw_ep: pd.DataFrame,
        configs: pd.DataFrame
    ):
        self._ew_ea = ew_ea
        self._vw_ea = vw_ea
        self._ew_ep = ew_ep
        self._vw_ep = vw_ep
        self._configs = configs
        self._summary_cache = None

    @property
    def ew_ea(self) -> pd.DataFrame:
        """EW Ex-Ante long-short factors."""
        return self._ew_ea

    @property
    def vw_ea(self) -> pd.DataFrame:
        """VW Ex-Ante long-short factors."""
        return self._vw_ea

    @property
    def ew_ep(self) -> pd.DataFrame:
        """EW Ex-Post long-short factors."""
        return self._ew_ep

    @property
    def vw_ep(self) -> pd.DataFrame:
        """VW Ex-Post long-short factors."""
        return self._vw_ep

    @property
    def configs(self) -> pd.DataFrame:
        """Metadata for all configurations."""
        return self._configs

    def summary(
        self,
        aggregate_by: Optional[Union[str, List[str]]] = None
    ) -> pd.DataFrame:
        """
        Compute summary statistics for all configurations.

        Parameters
        ----------
        aggregate_by : str or list of str, optional
            Group by these column(s) and compute average statistics.
            Valid values: 'filter_type', 'signal', 'hp', 'rating', 'location'
            Example: aggregate_by='filter_type' returns one row per filter type
            with averaged statistics.

        Returns
        -------
        pd.DataFrame
            Summary statistics with columns:
            - signal, hp, rating, filter_type, level, location
            - ew_ea_mean, ew_ea_tstat (Newey-West)
            - vw_ea_mean, vw_ea_tstat
            - ew_ep_mean, ew_ep_tstat
            - vw_ep_mean, vw_ep_tstat
            - ea_ep_diff_ew, ea_ep_diff_vw (EP - EA difference)
            - n_obs, sharpe (annualized EW EA Sharpe)

            All means are in % (×100).

        Notes
        -----
        For 'wins' filter, EA values are NaN because winsorization doesn't
        exclude bonds from ranking (unlike trim/price/bounce filters).
        The wins rankings are identical to baseline, so EA returns would
        duplicate baseline. EP returns show the effect of winsorized returns.
        """
        if self._summary_cache is None:
            rows = []
            for _, config in self._configs.iterrows():
                col = config['column_name']

                # Compute stats for each panel
                ew_ea_mean, ew_ea_tstat, n_obs = compute_newey_west_tstat(self._ew_ea[col])
                vw_ea_mean, vw_ea_tstat, _ = compute_newey_west_tstat(self._vw_ea[col])
                ew_ep_mean, ew_ep_tstat, _ = compute_newey_west_tstat(self._ew_ep[col])
                vw_ep_mean, vw_ep_tstat, _ = compute_newey_west_tstat(self._vw_ep[col])

                # Annualized Sharpe ratio (EW EA)
                ew_ea_series = self._ew_ea[col].dropna()
                if len(ew_ea_series) > 1 and ew_ea_series.std() > 0:
                    sharpe = (ew_ea_series.mean() / ew_ea_series.std()) * np.sqrt(12)
                else:
                    sharpe = np.nan

                # EA-EP difference (EP - EA), in percentage points
                ea_ep_diff_ew = (ew_ep_mean - ew_ea_mean) * 100 if not (np.isnan(ew_ep_mean) or np.isnan(ew_ea_mean)) else np.nan
                ea_ep_diff_vw = (vw_ep_mean - vw_ea_mean) * 100 if not (np.isnan(vw_ep_mean) or np.isnan(vw_ea_mean)) else np.nan

                row = {
                    'signal': config['signal'],
                    'hp': config['hp'],
                    'filter_type': config['filter_type'],
                    'level': config['level'],
                    'location': config['location'],
                    'ew_ea_mean': ew_ea_mean * 100,  # Convert to %
                    'ew_ea_tstat': ew_ea_tstat,
                    'vw_ea_mean': vw_ea_mean * 100,
                    'vw_ea_tstat': vw_ea_tstat,
                    'ew_ep_mean': ew_ep_mean * 100,
                    'ew_ep_tstat': ew_ep_tstat,
                    'vw_ep_mean': vw_ep_mean * 100,
                    'vw_ep_tstat': vw_ep_tstat,
                    'ea_ep_diff_ew': ea_ep_diff_ew,
                    'ea_ep_diff_vw': ea_ep_diff_vw,
                    'n_obs': n_obs,
                    'sharpe': sharpe,
                }

                # Add rating if present in config
                if 'rating' in config:
                    row['rating'] = config['rating']

                rows.append(row)

            self._summary_cache = pd.DataFrame(rows)

        result = self._summary_cache

        # Handle aggregation
        if aggregate_by is not None:
            if isinstance(aggregate_by, str):
                aggregate_by = [aggregate_by]

            # Validate aggregate_by columns
            valid_cols = {'filter_type', 'signal', 'hp', 'rating', 'location'}
            for col in aggregate_by:
                if col not in valid_cols:
                    raise ValueError(f"Invalid aggregate_by column: '{col}'. Valid: {valid_cols}")
                if col not in result.columns:
                    raise ValueError(f"Column '{col}' not in summary. Available: {list(result.columns)}")

            # Columns to average
            numeric_cols = [
                'ew_ea_mean', 'vw_ea_mean', 'ew_ep_mean', 'vw_ep_mean',
                'ea_ep_diff_ew', 'ea_ep_diff_vw', 'sharpe'
            ]
            # Filter to columns that exist in result
            numeric_cols = [c for c in numeric_cols if c in result.columns]

            result = result.groupby(aggregate_by, dropna=False)[numeric_cols].mean().reset_index()

        return result

    def average_by_filter(self) -> pd.DataFrame:
        """
        Compute average statistics by (signal, hp, filter_type).

        This is a convenience method equivalent to:
        ``summary(aggregate_by=['signal', 'hp', 'filter_type'])``

        Returns
        -------
        pd.DataFrame
            Averaged statistics with one row per (signal, hp, filter_type).
            Useful for comparing baseline vs different filter types.

        Examples
        --------
        >>> results = DataUncertaintyAnalysis(...).fit()
        >>> avg = results.average_by_filter()
        >>> print(avg[['signal', 'hp', 'filter_type', 'ew_ea_mean', 'vw_ea_mean']])
        """
        groupby_cols = ['signal', 'hp', 'filter_type']
        if 'rating' in self._summary_cache.columns if self._summary_cache is not None else 'rating' in self._configs.columns:
            groupby_cols.append('rating')

        return self.summary(aggregate_by=groupby_cols)

    # Sentinel value to distinguish "not provided" from "filter for None"
    _NOT_PROVIDED = object()

    def filter(
        self,
        signal: Optional[Union[str, List[str]]] = None,
        hp: Optional[Union[int, List[int]]] = None,
        filter_type: Optional[Union[str, List[str]]] = None,
        location: Optional[Union[str, List[str]]] = None,
        rating: Optional[Union[str, List[str]]] = _NOT_PROVIDED
    ) -> 'DataUncertaintyResults':
        """
        Filter to subset of configurations.

        Parameters
        ----------
        signal : str or list, optional
            Filter by signal name(s)
        hp : int or list, optional
            Filter by holding period(s)
        filter_type : str or list, optional
            Filter by type(s): 'baseline', 'trim', 'price', 'bounce', 'wins'
        location : str or list, optional
            Filter by tail location(s): 'left', 'right', 'both'
        rating : str, list, or None, optional
            Filter by rating category(s): 'IG', 'NIG', or None (for all bonds)
            Use rating=None to filter for configs with no rating restriction.

        Returns
        -------
        DataUncertaintyResults
            New results object with filtered configurations
        """
        mask = pd.Series([True] * len(self._configs))

        if signal is not None:
            if isinstance(signal, str):
                signal = [signal]
            mask &= self._configs['signal'].isin(signal)

        if hp is not None:
            if isinstance(hp, int):
                hp = [hp]
            mask &= self._configs['hp'].isin(hp)

        if filter_type is not None:
            if isinstance(filter_type, str):
                filter_type = [filter_type]
            mask &= self._configs['filter_type'].isin(filter_type)

        if location is not None:
            if isinstance(location, str):
                location = [location]
            mask &= self._configs['location'].isin(location)

        # Rating filter: use sentinel to distinguish "not provided" from "filter for None"
        if rating is not self._NOT_PROVIDED and 'rating' in self._configs.columns:
            # Handle filtering by rating - need special handling for None values
            if not isinstance(rating, list):
                rating = [rating]

            # Build mask that handles None values correctly
            rating_mask = pd.Series([False] * len(self._configs))
            for r in rating:
                if r is None:
                    # Match None/NaN values in the rating column
                    rating_mask |= self._configs['rating'].isna()
                else:
                    rating_mask |= (self._configs['rating'] == r)
            mask &= rating_mask

        filtered_configs = self._configs[mask].reset_index(drop=True)
        cols = filtered_configs['column_name'].tolist()

        return DataUncertaintyResults(
            ew_ea=self._ew_ea[cols],
            vw_ea=self._vw_ea[cols],
            ew_ep=self._ew_ep[cols],
            vw_ep=self._vw_ep[cols],
            configs=filtered_configs
        )

    def to_excel(self, path: str):
        """
        Export results to Excel file.

        Creates sheets: Summary, EW_EA, VW_EA, EW_EP, VW_EP, Configs

        Parameters
        ----------
        path : str
            Output file path
        """
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            self.summary().to_excel(writer, sheet_name='Summary', index=False)
            self._ew_ea.to_excel(writer, sheet_name='EW_EA')
            self._vw_ea.to_excel(writer, sheet_name='VW_EA')
            self._ew_ep.to_excel(writer, sheet_name='EW_EP')
            self._vw_ep.to_excel(writer, sheet_name='VW_EP')
            self._configs.to_excel(writer, sheet_name='Configs', index=False)

    def __repr__(self) -> str:
        n_configs = len(self._configs)
        n_dates = len(self._ew_ea)
        signals = self._configs['signal'].unique().tolist()
        hps = sorted(self._configs['hp'].unique().tolist())
        return (
            f"DataUncertaintyResults(\n"
            f"  n_configs={n_configs},\n"
            f"  n_dates={n_dates},\n"
            f"  signals={signals},\n"
            f"  holding_periods={hps}\n"
            f")"
        )


# =============================================================================
# Worker Function for Parallel Processing
# =============================================================================

def _run_single_config(
    data: pd.DataFrame,
    signal: str,
    hp: int,
    filter_dict: Optional[Dict],
    num_portfolios: int,
    dynamic_weights: bool,
    rating: Optional[str],
    strategy_obj: Any,
    column_name: str,
    rebalance_frequency: Union[str, int] = 'monthly',
    rebalance_month: Union[int, List[int]] = 6,
) -> Dict:
    """
    Run a single configuration and return results.

    This is the worker function for parallel processing.
    """
    try:
        # Create strategy
        if strategy_obj is not None:
            # Clone strategy with correct holding period
            if isinstance(strategy_obj, Momentum):
                strategy = Momentum(
                    holding_period=hp,
                    lookback_period=strategy_obj.lookback_period,
                    skip=strategy_obj.skip,
                    num_portfolios=num_portfolios,
                    rebalance_frequency=rebalance_frequency,
                    rebalance_month=rebalance_month,
                    verbose=False
                )
            elif isinstance(strategy_obj, LTreversal):
                strategy = LTreversal(
                    holding_period=hp,
                    lookback_period=strategy_obj.lookback_period,
                    skip=strategy_obj.skip,
                    num_portfolios=num_portfolios,
                    rebalance_frequency=rebalance_frequency,
                    rebalance_month=rebalance_month,
                    verbose=False
                )
            else:
                # Generic strategy clone
                strategy = strategy_obj
        else:
            # Use SingleSort with signal column
            strategy = SingleSort(
                holding_period=hp,
                sort_var=signal,
                num_portfolios=num_portfolios,
                rebalance_frequency=rebalance_frequency,
                rebalance_month=rebalance_month,
                verbose=False
            )

        # Run strategy formation using legacy params (properly handles filters)
        # Catch RuntimeWarning about "Mean of empty slice" - indicates filter has no effect
        filter_warning = None
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", RuntimeWarning)
            sf = StrategyFormation(
                data=data,
                strategy=strategy,
                filters=filter_dict,
                rating=rating,
                turnover=False,
                chars=None,
                verbose=False,
                dynamic_weights=dynamic_weights
            )
            result = sf.fit()

            # Check if any RuntimeWarnings about empty slices were caught
            for w in caught_warnings:
                if issubclass(w.category, RuntimeWarning):
                    if 'empty slice' in str(w.message).lower() or 'mean of empty' in str(w.message).lower():
                        filter_warning = (
                            f"Filter has no effect - no observations meet the filter criteria. "
                            f"Filter: {filter_dict}"
                        )
                        break

        # Extract long-short returns
        ew_ea, vw_ea = result.get_long_short()

        # Get EP returns
        try:
            ew_ep, vw_ep = result.get_long_short_ex_post()
        except (ValueError, AttributeError):
            # EP not available (no filter applied or wins filter)
            ew_ep, vw_ep = ew_ea.copy(), vw_ea.copy()

        return {
            'column_name': column_name,
            'ew_ea': ew_ea,
            'vw_ea': vw_ea,
            'ew_ep': ew_ep,
            'vw_ep': vw_ep,
            'success': True,
            'error': None,
            'warning': filter_warning
        }

    except Exception as e:
        return {
            'column_name': column_name,
            'ew_ea': None,
            'vw_ea': None,
            'ew_ep': None,
            'vw_ep': None,
            'success': False,
            'error': str(e),
            'warning': None
        }


# =============================================================================
# Main Analysis Class
# =============================================================================

class DataUncertaintyAnalysis:
    """
    High-level wrapper for data uncertainty analysis.

    Runs analysis across multiple holding periods and filter configurations,
    returning long-short factors for EW/VW and EA/EP combinations.

    Parameters
    ----------
    data : pd.DataFrame
        Bond panel data with columns: date, ID, ret, VW, signal(s)
    signals : list of str, optional
        Column name(s) for pre-computed signals
    strategy : Strategy, optional
        Strategy object (Momentum, LTreversal) for derived signals
    holding_periods : list of int
        Holding periods to test (default: [1, 3, 6])
    num_portfolios : int
        Number of quantile buckets (default: 5)
    dynamic_weights : bool
        VW from d-1 (True) or formation date (False)
    filters : dict, optional
        Filter configurations (see examples)
    include_baseline : bool
        Always include no-filter baseline (default: True)
    rating : str, optional
        Rating filter: 'IG', 'NIG', or None
    rebalance_frequency : str or int, default='monthly'
        Rebalancing frequency:
        - 'monthly': Rebalance every month (staggered portfolios)
        - 'quarterly' or 3: Rebalance every 3 months
        - 'semi-annual' or 6: Rebalance every 6 months
        - 'annual' or 12: Rebalance every 12 months
    rebalance_month : int or list of int, default=6
        Month(s) when rebalancing occurs (1=Jan, 6=Jun, 12=Dec)
    columns : dict, optional
        Column name mapping from PyBondLab expected names to your data's names.
        Keys are PyBondLab names: 'date', 'ID', 'ret', 'VW', 'RATING_NUM', 'PRICE'
        Values are the corresponding column names in your data.
        Only specify columns that have different names in your data.
        Example: {'ID': 'cusip_id', 'VW': 'mcap_e', 'PRICE': 'prc_eom'}
    n_jobs : int
        Number of parallel workers (default: 1)
    verbose : bool
        Show progress output (default: True)

    Examples
    --------
    >>> # With default column names
    >>> results = DataUncertaintyAnalysis(
    ...     data=data,
    ...     signals=['momentum'],
    ...     holding_periods=[1, 3],
    ...     filters={'trim': [0.2, 0.5]},
    ... ).fit()

    >>> # With custom column names
    >>> results = DataUncertaintyAnalysis(
    ...     data=data,
    ...     signals=['var_90'],
    ...     holding_periods=[1, 3],
    ...     filters={'trim': [0.2], 'price': [50, 200]},
    ...     columns={
    ...         'ID': 'cusip_id',
    ...         'VW': 'mcap_e',
    ...         'RATING_NUM': 'spc_rat',
    ...         'PRICE': 'prc_eom',
    ...     },
    ... ).fit()
    """

    def __init__(
        self,
        data: pd.DataFrame,
        signals: Optional[List[str]] = None,
        strategy: Optional[Any] = None,
        holding_periods: Optional[List[int]] = None,
        num_portfolios: int = 5,
        dynamic_weights: bool = True,
        filters: Optional[Dict[str, List]] = None,
        include_baseline: bool = True,
        rating: Optional[Union[str, Tuple[int, int]]] = None,
        ratings: Optional[List[Optional[Union[str, Tuple[int, int]]]]] = None,
        subset_filter: Optional[SubsetFilter] = None,
        rebalance_frequency: Union[str, int] = 'monthly',
        rebalance_month: Union[int, List[int]] = 6,
        columns: Optional[Dict[str, str]] = None,
        n_jobs: int = 1,
        verbose: bool = True,
        use_fast_path: bool = True
    ):
        # Validate inputs
        if signals is None and strategy is None:
            raise ValueError("Either 'signals' or 'strategy' must be provided")
        if signals is not None and strategy is not None:
            raise ValueError("Cannot specify both 'signals' and 'strategy'")

        # Handle rating vs ratings parameter
        # If both specified, ratings takes precedence
        # If neither, default to [None] (all bonds)
        if ratings is not None:
            self.ratings = ratings
        elif rating is not None:
            self.ratings = [rating]
        else:
            self.ratings = [None]  # Default: all bonds

        # Validate ratings - accepts 'IG', 'NIG', None, or tuple (min, max)
        valid_rating_strings = {'IG', 'NIG', None}
        for r in self.ratings:
            if r not in valid_rating_strings and not isinstance(r, tuple):
                raise ValueError(f"Invalid rating '{r}'. Must be 'IG', 'NIG', None, or tuple (min, max)")
            if isinstance(r, tuple):
                if len(r) != 2:
                    raise ValueError(f"Rating tuple must have 2 elements (min, max), got {len(r)}")
                if not all(isinstance(x, (int, float)) for x in r):
                    raise ValueError(f"Rating tuple elements must be numeric, got {r}")

        self.data_raw = data
        self.signals = signals if signals is not None else [None]
        self.strategy = strategy
        self.holding_periods = holding_periods or [1, 3, 6]
        self.num_portfolios = num_portfolios
        self.dynamic_weights = dynamic_weights
        self.filters = filters or {}
        self.include_baseline = include_baseline
        self.rating = rating  # Keep for backward compatibility (slow path)
        self.subset_filter = subset_filter
        self.rebalance_frequency = rebalance_frequency
        self.rebalance_month = rebalance_month
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.use_fast_path = use_fast_path

        # Determine if this is non-staggered rebalancing
        self._is_nonstaggered = self._check_nonstaggered()

        # Build column mapping (merge defaults with user-provided)
        self.columns = DEFAULT_COLUMNS.copy()
        if columns is not None:
            self.columns.update(columns)

        # Parse filter configurations
        self._filter_configs = self._parse_filters()

        # Prepare data (rename columns, subset to required columns only)
        self.data = self._prepare_data()

    def _check_nonstaggered(self) -> bool:
        """Check if this is non-staggered (non-monthly) rebalancing."""
        freq = self.rebalance_frequency
        if isinstance(freq, str):
            return freq != 'monthly'
        elif isinstance(freq, int):
            return freq > 1
        return False

    def _parse_filters(self) -> List[FilterConfig]:
        """Parse filter dict into FilterConfig objects."""
        configs = []

        # Add baseline if requested
        if self.include_baseline:
            configs.append(FilterConfig(
                filter_type='baseline',
                level=None,
                location=None
            ))

        # Parse each filter type
        for filter_type, levels in self.filters.items():
            if filter_type not in ['trim', 'price', 'bounce', 'wins']:
                warnings.warn(f"Unknown filter type: {filter_type}")
                continue

            if filter_type == 'price':
                # Price filter: requires nested format [[left_levels], [right_levels]]
                # e.g., [[1,2,5,10], [125,150,200]]
                if not (isinstance(levels, (list, tuple)) and len(levels) == 2 and
                        isinstance(levels[0], (list, tuple)) and isinstance(levels[1], (list, tuple))):
                    raise ValueError(
                        f"Price filter requires nested format: [[left_levels], [right_levels]]\n"
                        f"Example: 'price': [[1, 5, 10], [125, 150, 200]]\n"
                        f"Got: {levels}"
                    )

                left_levels = levels[0]   # Exclude price < threshold
                right_levels = levels[1]  # Exclude price > threshold

                # Left tail configs
                for lvl in left_levels:
                    configs.append(FilterConfig(
                        filter_type='price',
                        level=lvl,
                        location='left'
                    ))

                # Right tail configs
                for lvl in right_levels:
                    configs.append(FilterConfig(
                        filter_type='price',
                        level=lvl,
                        location='right'
                    ))

                # Both configs (all combinations)
                for left_lvl in left_levels:
                    for right_lvl in right_levels:
                        configs.append(FilterConfig(
                            filter_type='price',
                            level=[left_lvl, right_lvl],
                            location='both'
                        ))

            elif filter_type == 'wins':
                # Wins: tuple of (percentile, location)
                for level in levels:
                    if isinstance(level, (list, tuple)) and len(level) == 2:
                        configs.append(FilterConfig(
                            filter_type='wins',
                            level=level[0],
                            location=level[1]
                        ))
                    else:
                        warnings.warn(f"Invalid wins config: {level}")
            else:
                # Trim, bounce: infer location from level sign
                for level in levels:
                    if isinstance(level, (list, tuple)):
                        location = 'both'
                    elif isinstance(level, (int, float)):
                        if level > 0:
                            location = 'right'
                        else:
                            location = 'left'
                    else:
                        location = None

                    configs.append(FilterConfig(
                        filter_type=filter_type,
                        level=level,
                        location=location
                    ))

        return configs

    def _prepare_data(self) -> pd.DataFrame:
        """
        Prepare data by renaming columns and subsetting to required columns only.

        This method:
        1. Validates that required columns are present (using user's column names)
        2. Renames columns from user names to PyBondLab expected names
        3. Subsets to only required columns + signals to minimize memory usage

        Returns
        -------
        pd.DataFrame
            Prepared data with standardized column names
        """
        # Build reverse mapping: user_col_name -> pbl_name
        user_to_pbl = {v: k for k, v in self.columns.items()}

        # Determine which columns are needed
        needed_pbl_cols = set(REQUIRED_COLUMNS)

        # Add PRICE if price filter is used
        if 'price' in self.filters:
            needed_pbl_cols.add('PRICE')

        # Get user column names for required columns
        needed_user_cols = []
        missing_cols = []
        rename_map = {}

        for pbl_name in needed_pbl_cols:
            user_name = self.columns.get(pbl_name, pbl_name)
            if user_name in self.data_raw.columns:
                needed_user_cols.append(user_name)
                if user_name != pbl_name:
                    rename_map[user_name] = pbl_name
            elif pbl_name in REQUIRED_COLUMNS:
                missing_cols.append(f"{pbl_name} (expected: '{user_name}')")
            # Optional columns (PRICE) - warn but don't fail
            elif pbl_name == 'PRICE' and 'price' in self.filters:
                warnings.warn(f"Price filter requested but '{user_name}' column not found. "
                             "Price filters will be skipped.")
                # Remove price filters
                self.filters = {k: v for k, v in self.filters.items() if k != 'price'}
                self._filter_configs = [fc for fc in self._filter_configs
                                       if fc.filter_type != 'price']

        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Use the 'columns' parameter to map your column names. "
                f"Example: columns={{'ID': 'your_id_col', 'VW': 'your_vw_col'}}"
            )

        # Add signal columns (keep original names)
        if self.signals[0] is not None:
            for sig in self.signals:
                if sig in self.data_raw.columns:
                    if sig not in needed_user_cols:
                        needed_user_cols.append(sig)
                else:
                    raise ValueError(f"Signal column '{sig}' not found in data")

        # Add subset_filter columns (keep original names)
        if self.subset_filter is not None:
            for col in self.subset_filter.keys():
                if col in self.data_raw.columns:
                    if col not in needed_user_cols:
                        needed_user_cols.append(col)
                else:
                    raise ValueError(f"Subset filter column '{col}' not found in data")

        # Subset and rename
        data = self.data_raw[needed_user_cols].copy()
        if rename_map:
            data = data.rename(columns=rename_map)

        if self.verbose:
            original_cols = len(self.data_raw.columns)
            subset_cols = len(data.columns)
            original_mem = self.data_raw.memory_usage(deep=True).sum() / 1e6
            subset_mem = data.memory_usage(deep=True).sum() / 1e6
            print(f"Data prepared: {subset_cols}/{original_cols} columns, "
                  f"{subset_mem:.1f}MB/{original_mem:.1f}MB ({100*subset_mem/original_mem:.0f}%)")

        return data

    def _generate_analysis_configs(self) -> List[AnalysisConfig]:
        """Generate all analysis configurations."""
        configs = []

        for signal in self.signals:
            signal_name = signal if signal else 'strategy'

            for hp in self.holding_periods:
                for fc in self._filter_configs:
                    column_name = f"{signal_name}_hp{hp}_{fc.get_column_suffix()}"
                    configs.append(AnalysisConfig(
                        signal=signal_name,
                        hp=hp,
                        filter_config=fc,
                        column_name=column_name
                    ))

        return configs

    def fit(self) -> DataUncertaintyResults:
        """
        Run the data uncertainty analysis.

        Returns
        -------
        DataUncertaintyResults
            Results container with factor returns and summary statistics
        """
        # Check if fast path can be used
        if self._can_use_fast_path():
            return self._fit_fast_all_signals()

        # Fall back to slow path
        t0 = time.time()

        # Generate all configurations
        analysis_configs = self._generate_analysis_configs()
        n_configs = len(analysis_configs)

        if self.verbose:
            print(f"DataUncertaintyAnalysis SLOW PATH: Running {n_configs} configurations")
            print(f"  Signals: {self.signals if self.signals[0] else ['strategy']}")
            print(f"  Holding periods: {self.holding_periods}")
            print(f"  Filter types: {list(self.filters.keys()) + (['baseline'] if self.include_baseline else [])}")
            print(f"  n_jobs: {self.n_jobs}")

        # Prepare results storage
        results_list = []

        if self.n_jobs == 1:
            # Sequential processing
            for i, ac in enumerate(analysis_configs):
                if self.verbose:
                    print(f"  [{i+1}/{n_configs}] {ac.column_name}...", end=' ')

                result = _run_single_config(
                    data=self.data,
                    signal=ac.signal if ac.signal != 'strategy' else None,
                    hp=ac.hp,
                    filter_dict=ac.filter_config.to_pbl_filter(),
                    num_portfolios=self.num_portfolios,
                    dynamic_weights=self.dynamic_weights,
                    rating=self.rating,
                    strategy_obj=self.strategy,
                    column_name=ac.column_name,
                    rebalance_frequency=self.rebalance_frequency,
                    rebalance_month=self.rebalance_month,
                )

                if self.verbose:
                    if result['success']:
                        if result.get('warning'):
                            print(f"OK (WARNING: {result['warning']})")
                        else:
                            print("OK")
                    else:
                        print(f"FAILED: {result['error']}")

                results_list.append(result)

        else:
            # Parallel processing
            # Determine start method based on platform
            start_method = 'fork' if platform.system() != 'Windows' else 'spawn'

            # Prepare worker arguments
            worker_args = []
            for ac in analysis_configs:
                worker_args.append((
                    self.data,
                    ac.signal if ac.signal != 'strategy' else None,
                    ac.hp,
                    ac.filter_config.to_pbl_filter(),
                    self.num_portfolios,
                    self.dynamic_weights,
                    self.rating,
                    self.strategy,
                    ac.column_name,
                    self.rebalance_frequency,
                    self.rebalance_month,
                ))

            import multiprocessing as mp
            ctx = mp.get_context(start_method)

            with ProcessPoolExecutor(max_workers=self.n_jobs, mp_context=ctx) as executor:
                futures = {
                    executor.submit(_run_single_config, *args): args[-1]
                    for args in worker_args
                }

                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    col_name = futures[future]
                    result = future.result()
                    results_list.append(result)

                    if self.verbose:
                        if result['success']:
                            if result.get('warning'):
                                status = f"OK (WARNING: {result['warning']})"
                            else:
                                status = "OK"
                        else:
                            status = f"FAILED: {result['error']}"
                        print(f"  [{completed}/{n_configs}] {col_name}... {status}")

        # Build output DataFrames
        ew_ea_dict = {}
        vw_ea_dict = {}
        ew_ep_dict = {}
        vw_ep_dict = {}
        config_rows = []

        for result, ac in zip(results_list, analysis_configs):
            col = result['column_name']

            if result['success']:
                ew_ea_dict[col] = result['ew_ea']
                vw_ea_dict[col] = result['vw_ea']
                ew_ep_dict[col] = result['ew_ep']
                vw_ep_dict[col] = result['vw_ep']
            else:
                # Create NaN series for failed configs
                # Use first successful result to get date index
                ew_ea_dict[col] = pd.Series(dtype=float)
                vw_ea_dict[col] = pd.Series(dtype=float)
                ew_ep_dict[col] = pd.Series(dtype=float)
                vw_ep_dict[col] = pd.Series(dtype=float)

            config_rows.append({
                'column_name': col,
                'signal': ac.signal,
                'hp': ac.hp,
                'filter_type': ac.filter_config.filter_type,
                'level': ac.filter_config.level,
                'location': ac.filter_config.location
            })

        # Create DataFrames
        ew_ea = pd.DataFrame(ew_ea_dict)
        vw_ea = pd.DataFrame(vw_ea_dict)
        ew_ep = pd.DataFrame(ew_ep_dict)
        vw_ep = pd.DataFrame(vw_ep_dict)
        configs = pd.DataFrame(config_rows)

        elapsed = time.time() - t0
        if self.verbose:
            print(f"Completed in {elapsed:.1f}s")

        return DataUncertaintyResults(
            ew_ea=ew_ea,
            vw_ea=vw_ea,
            ew_ep=ew_ep,
            vw_ep=vw_ep,
            configs=configs
        )

    # =========================================================================
    # FAST PATH - Level 1 Optimization
    # =========================================================================

    def _can_use_fast_path(self) -> bool:
        """
        Check if fast path can be used.

        Fast path requires:
        - use_fast_path=True (default)
        - Pre-computed signals (not strategy-based)
        - No strategy object (Momentum/LTreversal compute signal from returns)
        - Monthly rebalancing (non-staggered not yet supported in fast path)

        Multiple signals are supported - we loop over them in the fast path.
        """
        if not self.use_fast_path:
            return False
        # Non-staggered rebalancing not yet supported in fast path
        # (TODO: Phase 15b - Add fast non-staggered path for DataUncertaintyAnalysis)
        if self._is_nonstaggered:
            if self.verbose:
                print("Fast path disabled: non-staggered rebalancing (using slow path)")
            return False
        if self.strategy is not None:
            # Check if strategy can use fast path
            return self._can_use_fast_strategy_path()
        if self.signals[0] is None:
            return False
        return True

    def _can_use_fast_strategy_path(self) -> bool:
        """
        Check if fast strategy path can be used for Momentum/LTreversal.

        Fast strategy path requires:
        - Strategy is Momentum or LTreversal
        - no_gap=False, fill_na=False, drop_na=False (default NaN handling)

        These restrictions exist because the numba kernels only implement
        the default behavior (NaN propagates in rolling window).
        """
        if self.strategy is None:
            return False

        # Check if strategy is Momentum or LTreversal
        if not isinstance(self.strategy, (Momentum, LTreversal)):
            return False

        # Check for non-default NaN handling options
        # The numba kernels only implement default behavior
        if getattr(self.strategy, 'no_gap', False):
            if self.verbose:
                print("Fast strategy path disabled: no_gap=True not supported")
            return False
        if getattr(self.strategy, 'fill_na', False):
            if self.verbose:
                print("Fast strategy path disabled: fill_na=True not supported")
            return False
        if getattr(self.strategy, 'drop_na', False):
            if self.verbose:
                print("Fast strategy path disabled: drop_na=True not supported")
            return False

        return True

    def _compute_ex_ante_wins(
        self,
        ret: np.ndarray,
        date_idx: np.ndarray,
        n_dates: int,
        level: float,
        location: str
    ) -> np.ndarray:
        """
        Compute ex-ante winsorized returns using historical thresholds.

        For each date t, thresholds are computed from all returns BEFORE date t.
        This matches the slow path behavior in FilterClass._winsorizing().

        OPTIMIZED: Uses numba kernels for parallel processing.

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
        location : str
            'both', 'left', or 'right'

        Returns
        -------
        np.ndarray
            Ex-ante winsorized returns
        """
        from .numba_core import compute_ex_ante_thresholds_fast, apply_winsorization_fast

        # Step 1: Compute thresholds for all dates (efficient - sorted once)
        thresholds = compute_ex_ante_thresholds_fast(ret, date_idx, n_dates, level)

        # Step 2: Apply thresholds in parallel using numba
        loc_code = 0 if location == 'both' else (1 if location == 'right' else 2)
        wins_ret = apply_winsorization_fast(ret, date_idx, thresholds, loc_code)

        return wins_ret

    def _apply_filters_vectorized(
        self,
        ret: np.ndarray,
        price: Optional[np.ndarray],
        ret_lag: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Apply all filters to returns in one vectorized pass.

        Returns
        -------
        Tuple[np.ndarray, List[str]]
            (filtered_returns, filter_names)
            filtered_returns shape: (n_obs, n_filters)
        """
        n_obs = len(ret)
        n_filters = len(self._filter_configs)

        # Pre-allocate output
        filtered_returns = np.empty((n_obs, n_filters), dtype=np.float64)
        filter_names = []

        for f_idx, fc in enumerate(self._filter_configs):
            filter_names.append(fc.get_column_suffix())

            if fc.filter_type == 'baseline':
                # Baseline: use original returns
                filtered_returns[:, f_idx] = ret

            elif fc.filter_type == 'trim':
                # Trim: set to NaN where ret exceeds threshold
                level = fc.level
                if isinstance(level, (list, tuple)):
                    lower, upper = level
                    mask = (ret > upper) | (ret < lower)
                elif level >= 0:
                    mask = ret > level
                else:
                    mask = ret < level
                filtered_returns[:, f_idx] = np.where(mask, np.nan, ret)

            elif fc.filter_type == 'price':
                # Price: set to NaN where price exceeds threshold
                # location='left': exclude price < level (low prices)
                # location='right': exclude price > level (high prices)
                # location='both': exclude price < lower OR price > upper
                if price is None:
                    filtered_returns[:, f_idx] = ret  # No price column, use original
                else:
                    level = fc.level
                    location = fc.location
                    if location == 'both' and isinstance(level, (list, tuple)):
                        lower, upper = level
                        mask = (price < lower) | (price > upper)
                    elif location == 'left':
                        mask = price < level
                    elif location == 'right':
                        mask = price > level
                    else:
                        mask = np.zeros(len(price), dtype=bool)
                    filtered_returns[:, f_idx] = np.where(mask, np.nan, ret)

            elif fc.filter_type == 'bounce':
                # Bounce: set to NaN where ret * ret_lag exceeds threshold
                if ret_lag is None:
                    filtered_returns[:, f_idx] = ret  # No lag, use original
                else:
                    bounce = ret * ret_lag
                    level = fc.level
                    if isinstance(level, (list, tuple)):
                        lower, upper = level
                        mask = (bounce > upper) | (bounce < lower)
                    elif level >= 0:
                        mask = bounce > level
                    else:
                        mask = bounce < level
                    filtered_returns[:, f_idx] = np.where(mask, np.nan, ret)

            elif fc.filter_type == 'wins':
                # Winsorize: clip returns to percentile bounds (ex-post)
                level = fc.level  # percentile (e.g., 99)
                location = fc.location
                lb = np.nanpercentile(ret, 100 - level)
                ub = np.nanpercentile(ret, level)

                if location == 'both':
                    filtered_returns[:, f_idx] = np.clip(ret, lb, ub)
                elif location == 'right':
                    filtered_returns[:, f_idx] = np.where(ret > ub, ub, ret)
                elif location == 'left':
                    filtered_returns[:, f_idx] = np.where(ret < lb, lb, ret)
                else:
                    filtered_returns[:, f_idx] = ret

            else:
                # Unknown filter, use original
                filtered_returns[:, f_idx] = ret

        return filtered_returns, filter_names

    def _fit_fast_all_signals(self) -> DataUncertaintyResults:
        """
        BLAZING FAST path wrapper: loop over signals and ratings.

        Combines results from all (signal × rating) combinations into
        a single DataUncertaintyResults object.

        Routes to _fit_fast_strategy() for Momentum/LTreversal strategies,
        or _fit_fast_single() for pre-computed signals.
        """
        t0 = time.time()

        n_filters = len(self._filter_configs)
        n_ratings = len(self.ratings)
        n_hps = len(self.holding_periods)

        # Check if using strategy-based signals
        use_strategy_path = self.strategy is not None

        if use_strategy_path:
            strategy_name = self.strategy.__strategy_name__
            if self.verbose:
                print(f"DataUncertaintyAnalysis FAST STRATEGY PATH: {strategy_name}")
                print(f"  Lookback: {self.strategy.lookback_period}, Skip: {self.strategy.skip}")
                print(f"  Ratings: {self.ratings}")
                print(f"  Filters: {n_filters}")
                print(f"  Holding periods: {self.holding_periods}")
        else:
            n_signals = len(self.signals)
            if self.verbose:
                print(f"DataUncertaintyAnalysis FAST PATH: {n_signals} signals × {n_ratings} ratings × {n_filters} filters × {n_hps} HPs")
                print(f"  Signals: {self.signals}")
                print(f"  Ratings: {self.ratings}")
                print(f"  Holding periods: {self.holding_periods}")

        # Accumulate results from all (signal × rating) combinations
        all_ew_ea = {}
        all_vw_ea = {}
        all_ew_ep = {}
        all_vw_ep = {}
        all_config_rows = []

        if use_strategy_path:
            # Route to strategy path
            for rating_cat in self.ratings:
                result = self._fit_fast_strategy(rating_cat)

                # Merge results
                all_ew_ea.update(result['ew_ea'])
                all_vw_ea.update(result['vw_ea'])
                all_ew_ep.update(result['ew_ep'])
                all_vw_ep.update(result['vw_ep'])
                all_config_rows.extend(result['configs'])
        else:
            # Route to pre-computed signal path
            for signal_col in self.signals:
                for rating_cat in self.ratings:
                    result = self._fit_fast_single(signal_col, rating_cat)

                    # Merge results
                    all_ew_ea.update(result['ew_ea'])
                    all_vw_ea.update(result['vw_ea'])
                    all_ew_ep.update(result['ew_ep'])
                    all_vw_ep.update(result['vw_ep'])
                    all_config_rows.extend(result['configs'])

        # Create DataFrames
        dates = self.data['date'].unique()
        dates = np.sort(dates)

        ew_ea = pd.DataFrame(all_ew_ea, index=dates)
        vw_ea = pd.DataFrame(all_vw_ea, index=dates)
        ew_ep = pd.DataFrame(all_ew_ep, index=dates)
        vw_ep = pd.DataFrame(all_vw_ep, index=dates)
        configs = pd.DataFrame(all_config_rows)

        elapsed = time.time() - t0
        if self.verbose:
            print(f"BLAZING FAST PATH completed in {elapsed:.1f}s")

        return DataUncertaintyResults(
            ew_ea=ew_ea,
            vw_ea=vw_ea,
            ew_ep=ew_ep,
            vw_ep=vw_ep,
            configs=configs
        )

    def _fit_fast_single(
        self,
        signal_col: str,
        rating_cat: Optional[str]
    ) -> Dict[str, Any]:
        """
        BLAZING FAST path for a single signal and rating category.

        Parameters
        ----------
        signal_col : str
            Signal column name
        rating_cat : str or None
            Rating category: 'IG', 'NIG', or None (all bonds)

        Returns
        -------
        Dict with keys: 'ew_ea', 'vw_ea', 'ew_ep', 'vw_ep', 'configs'
            Each value is a dict mapping column_name to Series/list
        """
        from .numba_core import (
            compute_ranks_all_filters,
            build_rank_lookups_all_filters,
            compute_ls_returns_all_filters_hp1,
            compute_ls_returns_all_filters_staggered
        )

        n_filters = len(self._filter_configs)

        # Rating suffix for column names
        rating_suffix = f"_{rating_cat}" if rating_cat is not None else ""

        # =====================================================================
        # Step 1: Extract numpy arrays from DataFrame (ONCE)
        # =====================================================================
        data = self.data

        # Build date and ID mappings
        dates = data['date'].unique()
        dates = np.sort(dates)
        date_to_idx = {d: i for i, d in enumerate(dates)}
        n_dates = len(dates)

        ids = data['ID'].unique()
        id_to_idx = {bond_id: i for i, bond_id in enumerate(ids)}
        n_ids = len(ids)

        # Extract arrays
        date_idx = data['date'].map(date_to_idx).values.astype(np.int64)
        id_idx = data['ID'].map(id_to_idx).values.astype(np.int64)
        signal = data[signal_col].values.astype(np.float64)
        ret = data['ret'].values.astype(np.float64)
        vw = data['VW'].values.astype(np.float64)

        # Rating array for rating mask
        rating_num = data['RATING_NUM'].values.astype(np.float64)

        # Optional arrays for filters
        price = data['PRICE'].values.astype(np.float64) if 'PRICE' in data.columns else None

        # Compute ret_lag for bounce filters
        has_bounce = any(fc.filter_type == 'bounce' for fc in self._filter_configs)
        if has_bounce:
            # Compute lagged returns per bond
            ret_lag = np.full(len(ret), np.nan, dtype=np.float64)
            # Sort by (ID, date) to compute lag correctly
            sort_idx = np.lexsort((date_idx, id_idx))
            for i in range(1, len(sort_idx)):
                curr = sort_idx[i]
                prev = sort_idx[i - 1]
                if id_idx[curr] == id_idx[prev]:
                    ret_lag[curr] = ret[prev]
        else:
            ret_lag = None

        # =====================================================================
        # Step 2: Apply all filters to returns (vectorized)
        # =====================================================================
        t_filter = time.time()
        filtered_returns, filter_names = self._apply_filters_vectorized(ret, price, ret_lag)
        if self.verbose:
            print(f"    Filters applied in {time.time() - t_filter:.2f}s")

        # =====================================================================
        # Step 3: Build rating mask (formation-date eligibility)
        # =====================================================================
        # Rating mask determines which bonds are eligible for portfolio formation
        # at each observation's date. This is AND-ed with filter masks.
        # IG: RATING_NUM 1-10, NIG: RATING_NUM 11-22
        # Tuple: (min, max) custom range
        if rating_cat == 'IG':
            rating_mask = (rating_num >= 1) & (rating_num <= 10)
        elif rating_cat == 'NIG':
            rating_mask = (rating_num >= 11) & (rating_num <= 22)
        elif isinstance(rating_cat, tuple):
            min_r, max_r = rating_cat
            rating_mask = (rating_num >= min_r) & (rating_num <= max_r)
        else:
            # None = all bonds
            rating_mask = np.ones(len(ret), dtype=np.bool_)

        # Apply subset_filter to rating mask (AND with rating mask)
        if self.subset_filter is not None:
            for col, (min_val, max_val) in self.subset_filter.items():
                col_vals = data[col].values.astype(np.float64)
                rating_mask = rating_mask & (col_vals >= min_val) & (col_vals <= max_val)

        # =====================================================================
        # Step 4: Build filter masks for ALL filters (combined with rating mask)
        # =====================================================================
        # filter_masks[i, f] = True if observation i should be included in ranking for filter f
        filter_masks = np.zeros((len(ret), n_filters), dtype=np.bool_)
        wins_filter_indices = []

        for f_idx, fc in enumerate(self._filter_configs):
            if fc.filter_type in ('baseline', 'wins'):
                # Baseline and wins: include all observations with valid signal AND valid rating
                filter_masks[:, f_idx] = ~np.isnan(signal) & rating_mask
            else:
                # Trim/price/bounce: exclude observations with NaN filtered returns AND apply rating
                filter_masks[:, f_idx] = ~np.isnan(filtered_returns[:, f_idx]) & rating_mask

            if fc.filter_type == 'wins':
                wins_filter_indices.append(f_idx)

        # =====================================================================
        # Step 5: Compute ranks for ALL filters at once (PARALLEL)
        # =====================================================================
        t_ranks = time.time()
        nport = self.num_portfolios
        ranks_all = compute_ranks_all_filters(
            date_idx, signal, filter_masks, n_dates, nport, n_filters
        )
        if self.verbose:
            print(f"    Ranks computed in {time.time() - t_ranks:.2f}s")

        # =====================================================================
        # Step 6: Build rank lookup tables for ALL filters
        # =====================================================================
        t_lookup = time.time()
        rank_lookups = build_rank_lookups_all_filters(
            date_idx, id_idx, ranks_all, n_dates, n_ids, n_filters
        )
        if self.verbose:
            print(f"    Rank lookups built in {time.time() - t_lookup:.2f}s")

        # =====================================================================
        # Step 7: Build VW lookup table (for HP > 1 and VW from d-1)
        # =====================================================================
        vw_lookup = np.full(n_dates * n_ids, np.nan, dtype=np.float64)
        for i in range(len(date_idx)):
            d = date_idx[i]
            bond_id = id_idx[i]
            if d >= 0 and d < n_dates and bond_id >= 0 and bond_id < n_ids:
                vw_lookup[d * n_ids + bond_id] = vw[i]

        # Build VW from d-1 for HP=1 case
        vw_d_minus_1 = np.full(len(ret), np.nan, dtype=np.float64)
        for i in range(len(date_idx)):
            d = date_idx[i]
            bond_id = id_idx[i]
            if d > 0:
                lookup_idx = (d - 1) * n_ids + bond_id
                if lookup_idx >= 0 and lookup_idx < len(vw_lookup):
                    vw_d_minus_1[i] = vw_lookup[lookup_idx]

        # =====================================================================
        # Step 8: Compute returns for ALL (HP, filter) combinations (PARALLEL)
        # =====================================================================
        ew_ea_dict = {}
        vw_ea_dict = {}
        ew_ep_dict = {}
        vw_ep_dict = {}
        config_rows = []

        for hp in self.holding_periods:
            t_hp = time.time()

            if hp == 1:
                # HP=1: Use optimized kernel for HP=1
                ew_ea_ls, vw_ea_ls, ew_ep_ls, vw_ep_ls = compute_ls_returns_all_filters_hp1(
                    date_idx, id_idx, ret, filtered_returns, vw_d_minus_1,
                    rank_lookups, n_dates, n_ids, nport, n_filters
                )
            else:
                # HP>1: Use staggered rebalancing kernel
                ew_ea_ls, vw_ea_ls, ew_ep_ls, vw_ep_ls = compute_ls_returns_all_filters_staggered(
                    date_idx, id_idx, ret, filtered_returns, vw_lookup,
                    rank_lookups, n_dates, n_ids, nport, n_filters, hp, self.dynamic_weights
                )

            # Extract results for each filter
            for f_idx, fc in enumerate(self._filter_configs):
                # Column name includes signal, hp, filter, and rating suffix
                col_name = f"{signal_col}_hp{hp}_{fc.get_column_suffix()}{rating_suffix}"
                is_wins = fc.filter_type == 'wins'

                if is_wins:
                    # For wins filter:
                    # - EA is NaN (ranking unchanged from baseline, would be identical)
                    # - EP uses winsorized returns (returns ARE affected by winsorization)
                    ew_ea_dict[col_name] = np.full(n_dates, np.nan)
                    vw_ea_dict[col_name] = np.full(n_dates, np.nan)
                    ew_ep_dict[col_name] = ew_ep_ls[:, f_idx]
                    vw_ep_dict[col_name] = vw_ep_ls[:, f_idx]
                else:
                    # For other filters: both EA and EP have computed values
                    ew_ea_dict[col_name] = ew_ea_ls[:, f_idx]
                    vw_ea_dict[col_name] = vw_ea_ls[:, f_idx]
                    ew_ep_dict[col_name] = ew_ep_ls[:, f_idx]
                    vw_ep_dict[col_name] = vw_ep_ls[:, f_idx]

                config_rows.append({
                    'column_name': col_name,
                    'signal': signal_col,
                    'hp': hp,
                    'rating': rating_cat,
                    'filter_type': fc.filter_type,
                    'level': fc.level,
                    'location': fc.location
                })

            if self.verbose:
                print(f"    HP={hp}: Processed {n_filters} filters in {time.time() - t_hp:.2f}s")

        # Return dict of results (will be combined in _fit_fast_all_signals)
        return {
            'ew_ea': ew_ea_dict,
            'vw_ea': vw_ea_dict,
            'ew_ep': ew_ep_dict,
            'vw_ep': vw_ep_dict,
            'configs': config_rows
        }

    # =========================================================================
    # FAST STRATEGY PATH - Momentum/LTreversal with numba signal computation
    # =========================================================================

    def _fit_fast_strategy(
        self,
        rating_cat: Optional[str]
    ) -> Dict[str, Any]:
        """
        FAST path for Momentum/LTreversal strategies.

        Computes signals for ALL filters at once using panel-based numba kernels,
        then uses the existing fast portfolio formation code.

        Parameters
        ----------
        rating_cat : str or None
            Rating category: 'IG', 'NIG', or None (all bonds)

        Returns
        -------
        Dict with keys: 'ew_ea', 'vw_ea', 'ew_ep', 'vw_ep', 'configs'
        """
        from .numba_core import (
            compute_momentum_signals_panel,
            compute_ltreversal_signals_panel,
            get_bond_boundaries,
            compute_ranks_all_filters,
            build_rank_lookups_all_filters,
            compute_ls_returns_all_filters_hp1,
            compute_ls_returns_all_filters_staggered
        )

        n_filters = len(self._filter_configs)
        strategy = self.strategy
        lookback = strategy.lookback_period
        skip = strategy.skip
        is_momentum = isinstance(strategy, Momentum)
        strategy_name = 'momentum' if is_momentum else 'ltreversal'

        # Rating suffix for column names
        rating_suffix = f"_{rating_cat}" if rating_cat is not None else ""

        # =====================================================================
        # Step 1: Extract numpy arrays from DataFrame (ONCE)
        # =====================================================================
        t_extract = time.time()
        data = self.data

        # Build date and ID mappings - use categorical codes if available for speed
        if hasattr(data['date'].dtype, 'categories'):
            # Fast path for categorical dates
            dates = data['date'].cat.categories.values
            date_idx = data['date'].cat.codes.values.astype(np.int64)
            date_to_idx = {d: i for i, d in enumerate(dates)}
        else:
            dates = data['date'].unique()
            dates = np.sort(dates)
            date_to_idx = {d: i for i, d in enumerate(dates)}
            date_idx = data['date'].map(date_to_idx).values.astype(np.int64)
        n_dates = len(dates)

        if hasattr(data['ID'].dtype, 'categories'):
            # Fast path for categorical IDs
            ids = data['ID'].cat.categories.values
            id_idx = data['ID'].cat.codes.values.astype(np.int64)
            id_to_idx = {bond_id: i for i, bond_id in enumerate(ids)}
        else:
            ids = data['ID'].unique()
            id_to_idx = {bond_id: i for i, bond_id in enumerate(ids)}
            id_idx = data['ID'].map(id_to_idx).values.astype(np.int64)
        n_ids = len(ids)

        # Extract arrays - use .to_numpy() for potential speedup
        ret = data['ret'].to_numpy().astype(np.float64)
        vw = data['VW'].to_numpy().astype(np.float64)

        # Rating array for rating mask
        rating_num = data['RATING_NUM'].to_numpy().astype(np.float64)

        # Optional arrays for filters
        price = data['PRICE'].to_numpy().astype(np.float64) if 'PRICE' in data.columns else None

        # Compute ret_lag for bounce filters
        has_bounce = any(fc.filter_type == 'bounce' for fc in self._filter_configs)
        if has_bounce:
            ret_lag = np.full(len(ret), np.nan, dtype=np.float64)
            sort_idx = np.lexsort((date_idx, id_idx))
            for i in range(1, len(sort_idx)):
                curr = sort_idx[i]
                prev = sort_idx[i - 1]
                if id_idx[curr] == id_idx[prev]:
                    ret_lag[curr] = ret[prev]
        else:
            ret_lag = None

        if self.verbose:
            print(f"    Data extracted in {time.time() - t_extract:.2f}s")

        # =====================================================================
        # Step 2: Apply all filters to returns (vectorized) - for EP returns
        # =====================================================================
        # Note: This uses GLOBAL thresholds for wins, which is correct for EP
        t_filter = time.time()
        filtered_returns, filter_names = self._apply_filters_vectorized(ret, price, ret_lag)
        if self.verbose:
            print(f"    Filters applied in {time.time() - t_filter:.2f}s")

        # =====================================================================
        # Step 2b: Compute EX-ANTE winsorized returns for wins signal computation
        # =====================================================================
        # For wins filters, signal computation needs ex-ante (historical) thresholds,
        # not global thresholds. This matches the slow path behavior.
        t_wins_ea = time.time()
        wins_ea_returns = {}  # f_idx -> ex-ante winsorized returns
        for f_idx, fc in enumerate(self._filter_configs):
            if fc.filter_type == 'wins':
                level = fc.level
                location = fc.location
                # Compute ex-ante winsorized returns using historical thresholds
                wins_ea_ret = self._compute_ex_ante_wins(
                    ret, date_idx, n_dates, level, location
                )
                wins_ea_returns[f_idx] = wins_ea_ret
        if self.verbose and wins_ea_returns:
            print(f"    Ex-ante wins computed in {time.time() - t_wins_ea:.2f}s")

        # =====================================================================
        # Step 3: Compute signals - different behavior for different filter types
        # =====================================================================
        # IMPORTANT: For Momentum/LTreversal with filters:
        # - Baseline: Signal from original returns
        # - Trim/price/bounce: Signal from ORIGINAL returns (filter just excludes bonds)
        # - Wins: Signal from EX-ANTE WINSORIZED returns (historical thresholds)
        t_signal = time.time()

        # Sort data by (ID, date) for bond-wise processing
        sort_idx = np.lexsort((date_idx, id_idx))
        id_sorted = id_idx[sort_idx]
        ret_sorted = ret[sort_idx]
        filtered_sorted = filtered_returns[sort_idx, :]

        # Get bond boundaries
        bond_starts = get_bond_boundaries(id_sorted)

        # Compute baseline signal from ORIGINAL returns (used for most filters)
        logret_baseline = np.log(ret_sorted + 1.0).reshape(-1, 1)
        if is_momentum:
            baseline_signal_sorted = compute_momentum_signals_panel(
                logret_baseline, bond_starts, lookback, skip
            )[:, 0]
        else:
            baseline_signal_sorted = compute_ltreversal_signals_panel(
                logret_baseline, bond_starts, lookback, skip
            )[:, 0]

        # Un-sort baseline signal
        unsort_idx = np.argsort(sort_idx)
        baseline_signal = baseline_signal_sorted[unsort_idx]

        # Build signals_all: (n_obs, n_filters)
        # Most filters use baseline signal, wins filters get their own signal from EA winsorized returns
        signals_all = np.empty((len(ret), n_filters), dtype=np.float64)
        wins_filter_indices = []

        for f_idx, fc in enumerate(self._filter_configs):
            if fc.filter_type == 'wins':
                wins_filter_indices.append(f_idx)
                # Wins: compute signal from EX-ANTE winsorized returns (historical thresholds)
                wins_ea_ret = wins_ea_returns[f_idx]
                wins_ea_ret_sorted = wins_ea_ret[sort_idx]
                logret_wins = np.log(wins_ea_ret_sorted + 1.0).reshape(-1, 1)
                if is_momentum:
                    wins_signal_sorted = compute_momentum_signals_panel(
                        logret_wins, bond_starts, lookback, skip
                    )[:, 0]
                else:
                    wins_signal_sorted = compute_ltreversal_signals_panel(
                        logret_wins, bond_starts, lookback, skip
                    )[:, 0]
                signals_all[:, f_idx] = wins_signal_sorted[unsort_idx]
            else:
                # Baseline, trim, price, bounce: use baseline signal
                signals_all[:, f_idx] = baseline_signal

        if self.verbose:
            print(f"    Signals computed in {time.time() - t_signal:.2f}s")

        # =====================================================================
        # Step 4: Build rating mask (formation-date eligibility)
        # =====================================================================
        # IG: RATING_NUM 1-10, NIG: RATING_NUM 11-22, Tuple: (min, max) custom range
        if rating_cat == 'IG':
            rating_mask = (rating_num >= 1) & (rating_num <= 10)
        elif rating_cat == 'NIG':
            rating_mask = (rating_num >= 11) & (rating_num <= 22)
        elif isinstance(rating_cat, tuple):
            min_r, max_r = rating_cat
            rating_mask = (rating_num >= min_r) & (rating_num <= max_r)
        else:
            rating_mask = np.ones(len(ret), dtype=np.bool_)

        # Apply subset_filter to rating mask (AND with rating mask)
        if self.subset_filter is not None:
            for col, (min_val, max_val) in self.subset_filter.items():
                col_vals = data[col].values.astype(np.float64)
                rating_mask = rating_mask & (col_vals >= min_val) & (col_vals <= max_val)

        # =====================================================================
        # Step 5: Build filter masks for ALL filters (combined with rating mask)
        # =====================================================================
        # For strategy-based signals:
        # - Baseline/wins: include bonds with valid signal
        # - Trim/price/bounce: also require valid filtered return (exclude extreme bonds)
        filter_masks = np.zeros((len(ret), n_filters), dtype=np.bool_)

        for f_idx, fc in enumerate(self._filter_configs):
            if fc.filter_type in ('baseline', 'wins'):
                # Include all bonds with valid signal
                filter_masks[:, f_idx] = ~np.isnan(signals_all[:, f_idx]) & rating_mask
            else:
                # Trim/price/bounce: also require valid filtered return
                filter_masks[:, f_idx] = (~np.isnan(signals_all[:, f_idx]) &
                                          ~np.isnan(filtered_returns[:, f_idx]) &
                                          rating_mask)

        # =====================================================================
        # Step 6: Compute ranks for ALL filters at once (PARALLEL)
        # =====================================================================
        t_ranks = time.time()
        nport = self.num_portfolios

        # Each filter uses its signal for ranking (baseline signal for most, wins signal for wins)
        ranks_all = np.full((len(ret), n_filters), np.nan, dtype=np.float64)

        for f_idx in range(n_filters):
            signal_f = signals_all[:, f_idx]
            mask_f = filter_masks[:, f_idx]

            # Compute ranks per date for this filter
            single_filter_mask = np.zeros((len(ret), 1), dtype=np.bool_)
            single_filter_mask[:, 0] = mask_f

            ranks_single = compute_ranks_all_filters(
                date_idx, signal_f, single_filter_mask, n_dates, nport, 1
            )
            ranks_all[:, f_idx] = ranks_single[:, 0]

        if self.verbose:
            print(f"    Ranks computed in {time.time() - t_ranks:.2f}s")

        # =====================================================================
        # Step 8: Build rank lookup tables for ALL filters
        # =====================================================================
        t_lookup = time.time()
        rank_lookups = build_rank_lookups_all_filters(
            date_idx, id_idx, ranks_all, n_dates, n_ids, n_filters
        )
        if self.verbose:
            print(f"    Rank lookups built in {time.time() - t_lookup:.2f}s")

        # =====================================================================
        # Step 9: Build VW lookup table (for HP > 1 and VW from d-1)
        # =====================================================================
        vw_lookup = np.full(n_dates * n_ids, np.nan, dtype=np.float64)
        for i in range(len(date_idx)):
            d = date_idx[i]
            bond_id = id_idx[i]
            if d >= 0 and d < n_dates and bond_id >= 0 and bond_id < n_ids:
                vw_lookup[d * n_ids + bond_id] = vw[i]

        # Build VW from d-1 for HP=1 case
        vw_d_minus_1 = np.full(len(ret), np.nan, dtype=np.float64)
        for i in range(len(date_idx)):
            d = date_idx[i]
            bond_id = id_idx[i]
            if d > 0:
                lookup_idx = (d - 1) * n_ids + bond_id
                if lookup_idx >= 0 and lookup_idx < len(vw_lookup):
                    vw_d_minus_1[i] = vw_lookup[lookup_idx]

        # =====================================================================
        # Step 10: Compute returns for ALL (HP, filter) combinations (PARALLEL)
        # =====================================================================
        ew_ea_dict = {}
        vw_ea_dict = {}
        ew_ep_dict = {}
        vw_ep_dict = {}
        config_rows = []

        for hp in self.holding_periods:
            t_hp = time.time()

            if hp == 1:
                # HP=1: Use optimized kernel
                ew_ea_ls, vw_ea_ls, ew_ep_ls, vw_ep_ls = compute_ls_returns_all_filters_hp1(
                    date_idx, id_idx, ret, filtered_returns, vw_d_minus_1,
                    rank_lookups, n_dates, n_ids, nport, n_filters
                )
            else:
                # HP>1: Use staggered rebalancing kernel
                ew_ea_ls, vw_ea_ls, ew_ep_ls, vw_ep_ls = compute_ls_returns_all_filters_staggered(
                    date_idx, id_idx, ret, filtered_returns, vw_lookup,
                    rank_lookups, n_dates, n_ids, nport, n_filters, hp, self.dynamic_weights
                )

            # Extract results for each filter
            for f_idx, fc in enumerate(self._filter_configs):
                # Column name includes strategy, hp, filter, and rating suffix
                col_name = f"{strategy_name}_hp{hp}_{fc.get_column_suffix()}{rating_suffix}"

                # For strategy-based signals, ALL filters (including wins) have EA values
                # because each filter produces a DIFFERENT signal from filtered returns
                ew_ea_dict[col_name] = ew_ea_ls[:, f_idx]
                vw_ea_dict[col_name] = vw_ea_ls[:, f_idx]
                ew_ep_dict[col_name] = ew_ep_ls[:, f_idx]
                vw_ep_dict[col_name] = vw_ep_ls[:, f_idx]

                config_rows.append({
                    'column_name': col_name,
                    'signal': strategy_name,
                    'hp': hp,
                    'rating': rating_cat,
                    'filter_type': fc.filter_type,
                    'level': fc.level,
                    'location': fc.location
                })

            if self.verbose:
                print(f"    HP={hp}: Processed {n_filters} filters in {time.time() - t_hp:.2f}s")

        # Return dict of results
        return {
            'ew_ea': ew_ea_dict,
            'vw_ea': vw_ea_dict,
            'ew_ep': ew_ep_dict,
            'vw_ep': vw_ep_dict,
            'configs': config_rows
        }
