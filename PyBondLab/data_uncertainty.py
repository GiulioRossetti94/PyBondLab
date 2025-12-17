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

    def summary(self) -> pd.DataFrame:
        """
        Compute summary statistics for all configurations.

        Returns DataFrame with:
        - signal, hp, filter_type, level, location
        - ew_ea_mean, ew_ea_tstat (Newey-West)
        - vw_ea_mean, vw_ea_tstat
        - ew_ep_mean, ew_ep_tstat
        - vw_ep_mean, vw_ep_tstat
        - n_obs, sharpe (annualized EW EA Sharpe)

        All means are in % (×100).
        """
        if self._summary_cache is not None:
            return self._summary_cache

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

            rows.append({
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
                'n_obs': n_obs,
                'sharpe': sharpe,
            })

        self._summary_cache = pd.DataFrame(rows)
        return self._summary_cache

    def filter(
        self,
        signal: Optional[Union[str, List[str]]] = None,
        hp: Optional[Union[int, List[int]]] = None,
        filter_type: Optional[Union[str, List[str]]] = None,
        location: Optional[Union[str, List[str]]] = None
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
    column_name: str
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
                    verbose=False
                )
            elif isinstance(strategy_obj, LTreversal):
                strategy = LTreversal(
                    holding_period=hp,
                    lookback_period=strategy_obj.lookback_period,
                    skip=strategy_obj.skip,
                    num_portfolios=num_portfolios,
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
        rating: Optional[str] = None,
        columns: Optional[Dict[str, str]] = None,
        n_jobs: int = 1,
        verbose: bool = True
    ):
        # Validate inputs
        if signals is None and strategy is None:
            raise ValueError("Either 'signals' or 'strategy' must be provided")
        if signals is not None and strategy is not None:
            raise ValueError("Cannot specify both 'signals' and 'strategy'")

        self.data_raw = data
        self.signals = signals if signals is not None else [None]
        self.strategy = strategy
        self.holding_periods = holding_periods or [1, 3, 6]
        self.num_portfolios = num_portfolios
        self.dynamic_weights = dynamic_weights
        self.filters = filters or {}
        self.include_baseline = include_baseline
        self.rating = rating
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Build column mapping (merge defaults with user-provided)
        self.columns = DEFAULT_COLUMNS.copy()
        if columns is not None:
            self.columns.update(columns)

        # Parse filter configurations
        self._filter_configs = self._parse_filters()

        # Prepare data (rename columns, subset to required columns only)
        self.data = self._prepare_data()

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

            for level in levels:
                if filter_type == 'wins':
                    # Wins: tuple of (percentile, location)
                    if isinstance(level, (list, tuple)) and len(level) == 2:
                        configs.append(FilterConfig(
                            filter_type='wins',
                            level=level[0],
                            location=level[1]
                        ))
                    else:
                        warnings.warn(f"Invalid wins config: {level}")
                else:
                    # Trim, price, bounce: infer location from level
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
        t0 = time.time()

        # Generate all configurations
        analysis_configs = self._generate_analysis_configs()
        n_configs = len(analysis_configs)

        if self.verbose:
            print(f"DataUncertaintyAnalysis: Running {n_configs} configurations")
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
                    column_name=ac.column_name
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
                    ac.column_name
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
