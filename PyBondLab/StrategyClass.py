# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 13:09:08 2024

@authors: Giulio Rossetti & Alex Dickerson
"""

import numpy as np
import logging
from typing import Optional, List, Union, Callable
from abc import ABC, abstractmethod
import warnings


#==============================================================================
#   Abstract Strategy Class
#==============================================================================
class Strategy(ABC):
    """Abstract base class for investment strategies."""

    def __init__(
        self,
        holding_period: Optional[int] = None,
        num_portfolios: Optional[int] = None,
        lookback_period: Optional[int] = None,
        skip: Optional[int] = None,
        rebalance_frequency: Union[str, int] = 'monthly',
        rebalance_month: Union[int, List[int]] = 6,
        verbose: bool = False
    ):
        """
        Initialize strategy with consistent parameter ordering.

        Parameters
        ----------
        holding_period : int, optional
            Holding period for the strategy.
            Required when using with StrategyFormation.
            Optional when using with DataUncertaintyAnalysis (overridden by holding_periods parameter).
        num_portfolios : int, optional
            Number of portfolios to create.
            Required when using with StrategyFormation.
            Optional when using with DataUncertaintyAnalysis (overridden by num_portfolios parameter).
        lookback_period : int, optional
            Lookback/formation period
        skip : int, optional
            Skip period between formation and holding
                    rebalance_frequency : str or int, default 'monthly'
        Rebalancing frequency:
            - 'monthly': Rebalance every month (staggered overlapping portfolios)
            - 'quarterly': Rebalance every 3 months
            - 'semi-annual': Rebalance every 6 months
            - 'annual': Rebalance every 12 months
            - int: Custom frequency in months (e.g., 4 for every 4 months)
        rebalance_month : int or list of int, default 6
            Month(s) when rebalancing occurs:
            - For 'annual': single month (e.g., 6 for June)
            - For 'semi-annual': list of 2 months (e.g., [6, 12] for June and December)
            - For 'quarterly': list of 4 months (e.g., [3, 6, 9, 12])
            - For 'monthly': ignored
            - For custom int frequency: starting month (e.g., 3 to start in March)
        verbose : bool, default False
            Enable verbose output. Default is False since StrategyFormation
            provides more comprehensive output.
        """
        self.holding_period = holding_period
        self.num_portfolios = num_portfolios
        self.lookback_period = lookback_period
        self.skip = skip
        self.rebalance_frequency = rebalance_frequency
        self.rebalance_month = rebalance_month
        self.verbose = verbose

        # Validate rebalancing paramteres
        self._validate_rebalancing_params()
        
        # Strategy identification
        self.__strategy_name__ = self.__class__.__name__
        self.str_name = self._build_name_string()
        
        # Double sorting attributes (set by subclasses if needed)
        self.double_sort = 0
        
        if self.verbose:
            self._print_initialization()

    def _validate_rebalancing_params(self):
        """Validate rebalancing frequency and month parameters."""
        valid_frequencies = ['monthly', 'quarterly', 'semi-annual', 'annual']
        
        if isinstance(self.rebalance_frequency, str):
            if self.rebalance_frequency not in valid_frequencies:
                raise ValueError(
                    f"rebalance_frequency must be one of {valid_frequencies} or an integer. "
                    f"Got '{self.rebalance_frequency}'"
                )
        elif isinstance(self.rebalance_frequency, int):
            if self.rebalance_frequency < 1:
                raise ValueError("Custom rebalance_frequency must be >= 1 month")
            if self.rebalance_frequency > self.holding_period:
                warnings.warn(
                    f"rebalance_frequency ({self.rebalance_frequency}) > holding_period ({self.holding_period}). "
                    "Portfolios will have gaps between rebalances."
                )
        else:
            raise ValueError("rebalance_frequency must be a string or integer")
        
        # Validate rebalance_month
        if isinstance(self.rebalance_month, int):
            if not 1 <= self.rebalance_month <= 12:
                raise ValueError(f"rebalance_month must be between 1 and 12, got {self.rebalance_month}")
        elif isinstance(self.rebalance_month, list):
            if not all(1 <= m <= 12 for m in self.rebalance_month):
                raise ValueError("All months in rebalance_month must be between 1 and 12")
            if len(set(self.rebalance_month)) != len(self.rebalance_month):
                raise ValueError("rebalance_month contains duplicate months")
        else:
            raise ValueError("rebalance_month must be an int or list of ints")


    def _build_name_string(self) -> str:
        """Build a descriptive string for the strategy."""
        name = f"{self.holding_period}"
        if self.lookback_period is not None:
            name += f"_{self.lookback_period}"
        if self.skip is not None:
            name += f"_{self.skip}"
        if self.rebalance_frequency != 'monthly':
            if isinstance(self.rebalance_frequency, str):
                name += f"_{self.rebalance_frequency}"
            else:
                name += f"_rebal{self.rebalance_frequency}m"
        return name
    
    def _print_initialization(self):
        """Print initialization details."""
        print("-" * 50)
        print(f"Initializing {self.__strategy_name__} Strategy")
        print(f"Holding period: {self.holding_period}")
        print(f"Number of portfolios: {self.num_portfolios}")
        if self.lookback_period is not None:
            print(f"Lookback period: {self.lookback_period}")
        if self.skip is not None:
            print(f"Skip period: {self.skip}")
        print(f"Rebalancing frequency: {self.rebalance_frequency}")
        if self.rebalance_frequency != 'monthly':
            print(f"Rebalancing month(s): {self.rebalance_month}")
        print("-" * 50)

    @abstractmethod
    def compute_signal(self, data):
        """Compute the trading signal from data."""
        pass
        
    @abstractmethod
    def get_sort_var(self, adj=None):
        """Get the variable name used for sorting."""
        pass
    
    @property
    def strategy_name(self):
        """Get the strategy name."""
        return self.__strategy_name__

    # Legacy property names for backward compatibility: to be removed when everything is updated
    @property
    def K(self):
        """Legacy property for holding_period."""
        return self.holding_period
    
    @property
    def nport(self):
        """Legacy property for num_portfolios."""
        return self.num_portfolios
    
    @property
    def J(self):
        """Legacy property for lookback_period."""
        return self.lookback_period 


#==============================================================================
#   SINGLE SORTING
#==============================================================================
class SingleSort(Strategy):
    """Single sorting strategy class."""
    def __init__(
        self,
        sort_var: str,
        holding_period: Optional[int] = None,
        num_portfolios: Optional[int] = None,
        breakpoints: Optional[List[float]] = None,
        lookback_period: Optional[int] = None,
        skip: Optional[int] = None,
        rebalance_frequency: Union[str, int] = 'monthly',
        rebalance_month: Union[int, List[int]] = 6,
        breakpoint_universe_func: Optional[Union[str, Callable]] = None, # pass a function to filter data for breakpoints (eg NYSE only)
        verbose: bool = False
    ):
        """
        Initialize SingleSort strategy.

        Parameters
        ----------
        sort_var : str
            Primary sorting variable (column name)
        holding_period : int, optional
            Holding period for the strategy. For non-staggered rebalancing (quarterly,
            semi-annual, annual), this defaults to 1 as staggered portfolios are not used.
            For monthly rebalancing, this is required.
        num_portfolios : int, optional
            Number of portfolios (inferred from breakpoints if not provided)
        breakpoints : list of float, optional
            Custom breakpoints as percentiles (e.g. [30, 70] for 3 portfolios)
        lookback_period : int, optional
            Lookback period for signal calculation
        skip : int, optional
            Skip period between formation and holding
        rebalance_frequency : str or int, default 'monthly'
            Rebalancing frequency ('monthly', 'quarterly', 'semi-annual', 'annual', or int)
        rebalance_month : int or list of int, default 6
            Month(s) when rebalancing occurs
        break_point_universe_func: str or callable, optional
            Define subset of data for computing breakpoints:
            - str: column name (e.g., 'nyse') - uses data[column] == 1 as filter
            - callable: function(data) -> pd.Series (boolean mask)
            Examples:
              breakpoint_universe_func='nyse'  # Use NYSE stocks for breakpoints
              breakpoint_universe_func=lambda df: df['RATING_NUM'] <= 10  # Use IG bonds
              breakpoint_universe_func=lambda df: df['size'] > df['size'].median()  # Large stocks
        verbose : bool, default True
            Print initialization details
        """
        # Determine if non-staggered rebalancing
        is_nonstaggered = rebalance_frequency != 'monthly'

        # Handle holding_period defaults and validation
        if holding_period is None:
            if is_nonstaggered:
                # For non-staggered, default to 1 (staggered portfolios not used)
                holding_period = 1
            else:
                raise ValueError(
                    "holding_period is required for monthly (staggered) rebalancing. "
                    "Example: SingleSort(sort_var='signal', holding_period=3, num_portfolios=5)"
                )
        elif is_nonstaggered and holding_period != 1:
            raise ValueError(
                f"For non-staggered rebalancing (rebalance_frequency='{rebalance_frequency}'), "
                f"holding_period must be 1 (got {holding_period}). "
                f"The actual holding period is determined by rebalance_frequency."
            )

        # Validate parameter types (catch common mistakes early)
        if not isinstance(holding_period, (int, np.integer)):
            raise TypeError(
                f"holding_period must be an integer, got {type(holding_period).__name__}: {holding_period!r}"
            )
        if not isinstance(sort_var, str):
            raise TypeError(
                f"sort_var must be a string (column name), got {type(sort_var).__name__}: {sort_var!r}. "
                f"Use keyword arguments: SingleSort(holding_period=1, sort_var='column_name', num_portfolios=5)"
            )
        if num_portfolios is not None and not isinstance(num_portfolios, (int, np.integer)):
            raise TypeError(
                f"num_portfolios must be an integer, got {type(num_portfolios).__name__}: {num_portfolios!r}"
            )

        # Validate and set num_portfolios/breakpoints
        num_portfolios = self._validate_portfolios_and_breakpoints(num_portfolios, breakpoints)
        # call parent constructor
        super().__init__(
            holding_period=holding_period, 
            num_portfolios=num_portfolios, 
            lookback_period=lookback_period, 
            skip=skip, 
            rebalance_frequency=rebalance_frequency,
            rebalance_month=rebalance_month,
            verbose=verbose
        )
        self.__strategy_name__ = "Single Sorting"
        self.sort_var = sort_var
        self.breakpoints = breakpoints
        self.breakpoint_universe_func = breakpoint_universe_func
        
        if verbose:
            self._print_sort_details()

    @staticmethod
    def _validate_portfolios_and_breakpoints(num_portfolios, breakpoints):
        """Validate num_portfolios and breakpoints consistency."""
        if breakpoints is not None:
            if num_portfolios is None:
                return len(breakpoints) + 1
            elif num_portfolios != len(breakpoints) + 1:
                raise ValueError(
                    f"If breakpoints provided, num_portfolios must equal len(breakpoints)+1. "
                    f"Got num_portfolios={num_portfolios}, len(breakpoints)+1={len(breakpoints)+1}"
                )
            return num_portfolios
        
        if num_portfolios is None:
            raise ValueError("You must provide either num_portfolios or breakpoints")
        
        return num_portfolios

    def _print_sort_details(self):
        """Print sorting-specific details."""
        print(f"Sorting variable: {self.sort_var}")
        if self.breakpoints:
            print(f"Custom breakpoints: {self.breakpoints}")
        
    def compute_signal(self, data):
        """For single sort, return data as-is."""
        return data
    
    def get_sort_var(self, adj=None):
        """Return the sorting variable name."""
        return self.sort_var
    
    def set_sort_var(self, sort_var: str):
        """Set the sorting variable."""
        if not isinstance(sort_var, str):
            raise ValueError("sort_var must be a string")
        self.sort_var = sort_var


#==============================================================================
#   DOUBLE SORTING
#==============================================================================
class DoubleSort(Strategy):
    """Double sorting strategy implementation."""

    def __init__(
        self,
        holding_period: int,
        sort_var: str,
        sort_var2: str,
        num_portfolios: Optional[int] = None,
        num_portfolios2: Optional[int] = None,
        breakpoints: Optional[List[float]] = None,
        breakpoints2: Optional[List[float]] = None,
        how: str = 'unconditional',
        lookback_period: Optional[int] = None,
        skip: Optional[int] = None,
        rebalance_frequency: Union[str, int] = 'monthly',
        rebalance_month: Union[int, List[int]] = 6,
        breakpoint_universe_func: Optional[Union[str, Callable]] = None,
        breakpoint_universe_func2: Optional[Union[str, Callable]] = None,
        auto_match_signals: bool = False,
        verbose: bool = False
    ):
        """
        Initialize DoubleSort strategy.

        Parameters
        ----------
        holding_period : int
            Holding period for the strategy
        sort_var : str
            Primary sorting variable
        sort_var2 : str
            Secondary sorting variable
        num_portfolios : int, optional
            Number of primary portfolios
        num_portfolios2 : int, optional
            Number of secondary portfolios
        breakpoints : list of float, optional
            Custom primary breakpoints
        breakpoints2 : list of float, optional
            Custom secondary breakpoints
        how : str, default 'unconditional'
            Type of double sort: 'unconditional' or 'conditional'
        lookback_period : int, optional
            Lookback period for signal calculation
        skip : int, optional
            Skip period between formation and holding
        rebalance_frequency : str or int, default 'monthly'
            Rebalancing frequency ('monthly', 'quarterly', 'semi-annual', 'annual', or int)
        rebalance_month : int or list of int, default 6
            Month(s) when rebalancing occurs
        breakpoint_universe_func : str or callable, optional
            Subset for computing primary sort breakpoints
        breakpoint_universe_func2 : str or callable, optional
            Subset for computing secondary sort breakpoints
        auto_match_signals : bool, default False
            If True, automatically truncate data to ensure both sorting variables
            have the same date coverage. If False (default), raises an error when
            signals have mismatched date ranges, requiring manual data preparation.
        verbose : bool, default True
            Print initialization details
        """

        # Validate sort_var and sort_var2 are strings (common mistake: swapping with num_portfolios)
        if not isinstance(sort_var, str):
            raise TypeError(
                f"sort_var must be a string (column name), got {type(sort_var).__name__}: {sort_var!r}. "
                f"Did you pass arguments in the wrong order? Use keyword arguments: "
                f"DoubleSort(holding_period=..., sort_var='col1', sort_var2='col2', "
                f"num_portfolios=5, num_portfolios2=5, how='unconditional')"
            )
        if not isinstance(sort_var2, str):
            raise TypeError(
                f"sort_var2 must be a string (column name), got {type(sort_var2).__name__}: {sort_var2!r}. "
                f"Did you pass arguments in the wrong order? Use keyword arguments: "
                f"DoubleSort(holding_period=..., sort_var='col1', sort_var2='col2', "
                f"num_portfolios=5, num_portfolios2=5, how='unconditional')"
            )

        # Validate num_portfolios types if provided (common mistake: passing string column name)
        if num_portfolios is not None and not isinstance(num_portfolios, (int, np.integer)):
            raise TypeError(
                f"num_portfolios must be an integer, got {type(num_portfolios).__name__}: {num_portfolios!r}. "
                f"Did you pass a column name where num_portfolios was expected?"
            )
        if num_portfolios2 is not None and not isinstance(num_portfolios2, (int, np.integer)):
            raise TypeError(
                f"num_portfolios2 must be an integer, got {type(num_portfolios2).__name__}: {num_portfolios2!r}. "
                f"Did you pass a column name where num_portfolios2 was expected?"
            )

        # Validate sorting method
        if how not in ['unconditional', 'conditional']:
            raise ValueError(f"how must be 'unconditional' or 'conditional', got '{how}'")

        # Validate and set portfolio numbers
        num_portfolios = self._validate_portfolios_and_breakpoints(num_portfolios, breakpoints)
        num_portfolios2 = self._validate_portfolios_and_breakpoints(num_portfolios2, breakpoints2)

        # if num_portfolios is None or num_portfolios2 is None:
        #     raise ValueError("You must provide num_portfolios/num_portfolios2 or breakpoints/breakpoints2")

        # call parent constructor
        super().__init__(
            holding_period=holding_period, 
            num_portfolios=num_portfolios, 
            lookback_period=lookback_period, 
            skip=skip, 
            rebalance_frequency=rebalance_frequency,
            rebalance_month=rebalance_month,
            verbose=verbose
        )

        # Set double sort attributes
        self.__strategy_name__ = "Double Sorting"
        self.DoubleSort = 1
        self.sort_var = sort_var
        self.sort_var2 = sort_var2
        self.num_portfolios2 = num_portfolios2
        self.breakpoints = breakpoints
        self.breakpoints2 = breakpoints2
        self.breakpoint_universe_func = breakpoint_universe_func
        self.breakpoint_universe_func2 = breakpoint_universe_func2
        self.auto_match_signals = auto_match_signals
        self.how = how
        
        if verbose:
            self._print_double_sort_details()

    @staticmethod
    def _validate_portfolios_and_breakpoints(num_portfolios, breakpoints):
        """Validate num_portfolios and breakpoints consistency."""
        if breakpoints is not None:
            if num_portfolios is None:
                return len(breakpoints) + 1
            elif num_portfolios != len(breakpoints) + 1:
                raise ValueError(
                    f"If breakpoints provided, num_portfolios must equal len(breakpoints)+1"
                )
            return num_portfolios
        
        if num_portfolios is None:
            raise ValueError("You must provide num_portfolios or breakpoints")
        
        return num_portfolios
    
    def _print_double_sort_details(self):
        """Print double sorting details."""
        print(f"Double sort type: {self.how}")
        print(f"Portfolios: {self.num_portfolios} x {self.num_portfolios2}")
        print(f"Primary variable: {self.sort_var}")
        print(f"Secondary variable: {self.sort_var2}")
        if self.breakpoints:
            print(f"Primary breakpoints: {self.breakpoints}")
        if self.breakpoints2:
            print(f"Secondary breakpoints: {self.breakpoints2}")
        
    def compute_signal(self, data):
        """For double sort, return data as-is."""
        return data
    
    def get_sort_var(self, adj=None):
        """Return the primary sorting variable name."""
        return self.sort_var
    
    def get_sort_var2(self, adj=None):
        """Return the secondary sorting variable name."""
        return self.sort_var2
    
    def set_sort_var(self, sort_var: str):
        """Set the primary sorting variable."""
        if not isinstance(sort_var, str):
            raise ValueError("sort_var must be a string")
        self.sort_var = sort_var
        
    def set_sort_var2(self, sort_var2: str):
        """Set the secondary sorting variable."""
        if not isinstance(sort_var2, str):
            raise ValueError("sort_var2 must be a string")
        self.sort_var2 = sort_var2
        
    def set_num_portfolios2(self, num_portfolios2: int):
        """Set the number of portfolios for secondary sorting."""
        if not isinstance(num_portfolios2, int) or num_portfolios2 <= 0:
            raise ValueError("num_portfolios2 must be a positive integer")
        self.num_portfolios2 = num_portfolios2
    
    # Legacy property names for backward compatibility (to be removed when everything is updated)
    @property
    def nport2(self):
        """Legacy property for num_portfolios2."""
        return self.num_portfolios2
        
#==============================================================================
#   MOMENTUM: SingleSorting
#==============================================================================
class Momentum(Strategy):
    """Momentum strategy."""
    def __init__(
        self,
        lookback_period: Optional[int] = None,
        skip: int = 1,
        no_gap: bool = False,
        fill_na: bool = False,
        drop_na: bool = False,
        holding_period: Optional[int] = None,
        num_portfolios: Optional[int] = None,
        rebalance_frequency: Union[str, int] = 'monthly',
        rebalance_month: Union[int, List[int]] = 6,
        verbose: bool = False,
        # Legacy parameter names for backward compatibility
        K: Optional[int] = None,
        nport: Optional[int] = None,
        J: Optional[int] = None
    ):
        """
        Initialize Momentum strategy.

        Parameters
        ----------
        lookback_period : int
            Formation period (lookback window). Required.
        skip : int, default 1
            Skip period between formation and holding
        no_gap : bool, default False
            If True, require consecutive calendar months in the formation window.
            If a month row is missing (gap in data), signal is set to NaN.
            If False, rolling window operates on rows regardless of calendar gaps.
        fill_na : bool, default False
            If True, treat NaN returns as 0% return. Uses fixed window of
            lookback_period rows, but NaN values contribute 0 to cumulative return.
            If False, any NaN in the window results in NaN signal.
        drop_na : bool, default False
            If True, skip NaN returns and look back further to find lookback_period
            valid (non-NaN) returns. The window size varies to accumulate exactly
            lookback_period valid observations.
            If False, window is fixed at lookback_period rows.
            Note: drop_na=True and fill_na=True cannot both be True.
        holding_period : int, optional
            Holding period for the strategy.
            Required when using with StrategyFormation.
            Not needed when using with DataUncertaintyAnalysis.
        num_portfolios : int, optional
            Number of portfolios to create.
            Required when using with StrategyFormation.
            Not needed when using with DataUncertaintyAnalysis.
        rebalance_frequency : str or int, default 'monthly'
            Rebalancing frequency ('monthly', 'quarterly', 'semi-annual', 'annual', or int)
        rebalance_month : int or list of int, default 6
            Month(s) when rebalancing occurs
        verbose : bool, default True
            Print initialization details
        K : int, optional
            Legacy parameter name for holding_period
        nport : int, optional
            Legacy parameter name for num_portfolios
        J : int, optional
            Legacy parameter name for lookback_period

        Notes
        -----
        Behavior for NaN handling options:

        | Option   | NaN in window (row exists)           | Description                          |
        |----------|--------------------------------------|--------------------------------------|
        | Default  | Signal=NaN                           | Standard rolling, NaN propagates     |
        | fill_na  | NaN treated as 0%, signal computed   | Fixed J-row window, NaN → 0% return  |
        | drop_na  | NaN skipped, uses J valid returns    | Variable window to get J valid obs   |

        The no_gap parameter can be combined with any of the above:
        - no_gap=True: Also requires consecutive calendar months (no missing rows)

        Examples
        --------
        # For DataUncertaintyAnalysis (hp and nport not needed):
        >>> mom = Momentum(lookback_period=3, skip=1)

        # For StrategyFormation (hp and nport required):
        >>> mom = Momentum(lookback_period=3, skip=1, holding_period=6, num_portfolios=5)
        """
        # Handle backward compatibility: map old parameter names to new ones
        if K is not None:
            holding_period = K
        if nport is not None:
            num_portfolios = nport
        if J is not None:
            lookback_period = J

        # Validate required parameters
        if lookback_period is None:
            raise ValueError("Momentum strategy requires lookback_period (or J)")

        # Validate mutually exclusive options
        if fill_na and drop_na:
            raise ValueError("fill_na and drop_na cannot both be True. Choose one.")

        super().__init__(
            holding_period=holding_period,
            num_portfolios=num_portfolios,
            lookback_period=lookback_period,
            skip=skip,
            rebalance_frequency=rebalance_frequency,
            rebalance_month=rebalance_month,
            verbose=verbose
        )

        self.__strategy_name__ = "MOMENTUM"
        # no breakpoints
        self.breakpoints = None
        self.no_gap = no_gap
        self.fill_na = fill_na
        self.drop_na = drop_na

        if verbose:
            print(f"Formation period: {self.lookback_period} months")
            print(f"Skip period: {self.skip} months")
            if no_gap:
                print("Gap control: Enabled (require consecutive calendar months)")
            if fill_na:
                print("Fill NA: Enabled (NaN returns treated as 0%)")
            if drop_na:
                print("Drop NA: Enabled (skip NaN, use J valid returns)")
            print("Sorting on: past returns")
    
    def compute_signal(self, data):
        """
        Compute momentum signal from past returns.

        The signal is the cumulative return over the lookback_period, shifted
        by the skip period. Handles gaps and NaN values based on no_gap,
        fill_na, and drop_na parameters.
        """
        if 'ret' not in data.columns:
            raise ValueError("Data must contain 'ret' column")

        data = data.copy()
        data = data.sort_values(['ID', 'date'], ignore_index=True)

        # Create month index for gap detection
        if self.no_gap:
            data['month_idx'] = (
                data['date'].dt.year * 12 + data['date'].dt.month
            )

        J = self.lookback_period

        if self.drop_na:
            # Option B: Variable window to accumulate J valid (non-NaN) returns
            # For each row, look back to find exactly J valid returns
            data['logret'] = np.log(data['ret'] + 1)

            # Cumulative sum of log returns (only for valid values)
            # and cumulative count of valid observations
            data['cumlogret'] = data.groupby('ID')['logret'].cumsum()
            data['cumvalid'] = data.groupby('ID')['ret'].transform(
                lambda x: x.notna().cumsum()
            )

            # For each row, we need the cumlogret at the position where cumvalid was (current_cumvalid - J)
            # This requires finding the row where cumvalid == current_cumvalid - J

            def compute_drop_na_signal(group):
                """Compute signal using exactly J valid returns."""
                group = group.copy()
                n = len(group)
                signal = np.full(n, np.nan)

                cumlogret = group['cumlogret'].values
                cumvalid = group['cumvalid'].values
                logret = group['logret'].values

                for i in range(n):
                    if np.isnan(cumvalid[i]) or cumvalid[i] < J:
                        continue

                    # Find the position where cumvalid == cumvalid[i] - J
                    target_cumvalid = cumvalid[i] - J

                    if target_cumvalid == 0:
                        # Use all valid returns from start
                        signal[i] = cumlogret[i]
                    else:
                        # Find the last position where cumvalid <= target_cumvalid
                        # This is the position just before our J-valid window starts
                        for j in range(i - 1, -1, -1):
                            if cumvalid[j] == target_cumvalid:
                                signal[i] = cumlogret[i] - cumlogret[j]
                                break

                group['signal'] = signal
                return group

            data = data.groupby('ID', group_keys=False).apply(compute_drop_na_signal)
            data['signal'] = np.exp(data['signal']) - 1

            # Clean up
            data.drop(columns=['cumlogret', 'cumvalid', 'logret'], inplace=True)

        elif self.fill_na:
            # Option A: Fixed window of J rows, NaN returns treated as 0%
            ret_col = data['ret'].fillna(0)
            data['logret'] = np.log(ret_col + 1)

            data['signal'] = (
                data.groupby(['ID'], group_keys=False)['logret']
                .rolling(J, min_periods=J)
                .sum()
                .values
            )
            data['signal'] = np.exp(data['signal']) - 1
            data.drop(columns=['logret'], inplace=True)

        else:
            # Default: Standard rolling, NaN propagates
            data['logret'] = np.log(data['ret'] + 1)

            data['signal'] = (
                data.groupby(['ID'], group_keys=False)['logret']
                .rolling(J, min_periods=J)
                .sum()
                .values
            )
            data['signal'] = np.exp(data['signal']) - 1
            data.drop(columns=['logret'], inplace=True)

        # Apply no_gap check: invalidate signal if months are not consecutive
        if self.no_gap:
            # Get lagged month_idx (J - 1 periods ago)
            data['lmonth_idx'] = data.groupby('ID')['month_idx'].shift(J - 1)
            data['month_diff'] = data['month_idx'] - data['lmonth_idx']
            # If month_diff != J - 1, there's a gap in the data rows
            data.loc[data['month_diff'] != (J - 1), 'signal'] = np.nan
            # Clean up temporary columns
            data.drop(columns=['month_idx', 'lmonth_idx', 'month_diff'], inplace=True)

        # Apply skip period
        data['signal'] = data.groupby("ID")['signal'].shift(self.skip)

        return data
        
    def get_sort_var(self, adj=None):
        """Return the sorting variable name and update self.sort_var."""
        self.sort_var = f'signal_{adj}' if adj else 'signal'
        return self.sort_var
    
#==============================================================================
#   Long Term Reversal: SingleSorting
#==============================================================================

class LTreversal(Strategy):
    """Long-term reversal strategy."""
    def __init__(
        self,
        lookback_period: Optional[int] = None,
        skip: Optional[int] = None,
        no_gap: bool = False,
        fill_na: bool = False,
        drop_na: bool = False,
        holding_period: Optional[int] = None,
        num_portfolios: Optional[int] = None,
        rebalance_frequency: Union[str, int] = 'monthly',
        rebalance_month: Union[int, List[int]] = 6,
        verbose: bool = False
    ):
        """
        Initialize Long-term reversal strategy.

        The signal is: cumulative return over lookback_period minus cumulative
        return over skip period (i.e., long-term return excluding recent return).

        Parameters
        ----------
        lookback_period : int
            Formation period (total lookback window). Required.
        skip : int
            Recent period to exclude from signal. Required.
        no_gap : bool, default False
            If True, require consecutive calendar months in the formation window.
            If a month row is missing (gap in data), signal is set to NaN.
            If False, rolling window operates on rows regardless of calendar gaps.
        fill_na : bool, default False
            If True, treat NaN returns as 0% return. Uses fixed window of
            lookback_period rows, but NaN values contribute 0 to cumulative return.
            If False, any NaN in the window results in NaN signal.
        drop_na : bool, default False
            If True, skip NaN returns and look back further to find lookback_period
            valid (non-NaN) returns. The window size varies to accumulate exactly
            lookback_period valid observations.
            If False, window is fixed at lookback_period rows.
            Note: drop_na=True and fill_na=True cannot both be True.
        holding_period : int, optional
            Holding period for the strategy.
            Required when using with StrategyFormation.
            Not needed when using with DataUncertaintyAnalysis.
        num_portfolios : int, optional
            Number of portfolios to create.
            Required when using with StrategyFormation.
            Not needed when using with DataUncertaintyAnalysis.
        rebalance_frequency : str or int, default 'monthly'
            Rebalancing frequency ('monthly', 'quarterly', 'semi-annual', 'annual', or int)
        rebalance_month : int or list of int, default 6
            Month(s) when rebalancing occurs
        verbose : bool, default True
            Print initialization details

        Notes
        -----
        Behavior for NaN handling options:

        | Option   | NaN in window (row exists)           | Description                          |
        |----------|--------------------------------------|--------------------------------------|
        | Default  | Signal=NaN                           | Standard rolling, NaN propagates     |
        | fill_na  | NaN treated as 0%, signal computed   | Fixed J-row window, NaN → 0% return  |
        | drop_na  | NaN skipped, uses J valid returns    | Variable window to get J valid obs   |

        The no_gap parameter can be combined with any of the above:
        - no_gap=True: Also requires consecutive calendar months (no missing rows)

        Examples
        --------
        # For DataUncertaintyAnalysis (hp and nport not needed):
        >>> ltr = LTreversal(lookback_period=60, skip=12)

        # For StrategyFormation (hp and nport required):
        >>> ltr = LTreversal(lookback_period=60, skip=12, holding_period=6, num_portfolios=5)
        """
        if lookback_period is None or skip is None:
            raise ValueError("LT reversal strategy requires both lookback_period and skip periods")

        # Validate mutually exclusive options
        if fill_na and drop_na:
            raise ValueError("fill_na and drop_na cannot both be True. Choose one.")

        super().__init__(
            holding_period=holding_period,
            num_portfolios=num_portfolios,
            lookback_period=lookback_period,
            skip=skip,
            rebalance_frequency=rebalance_frequency,
            rebalance_month=rebalance_month,
            verbose=verbose
        )

        self.__strategy_name__ = "LT-REVERSAL"
        # no breakpoints
        self.breakpoints = None
        self.no_gap = no_gap
        self.fill_na = fill_na
        self.drop_na = drop_na

        if verbose:
            print(f"Formation period: {self.lookback_period} months")
            print(f"Skip period: {self.skip} months")
            if no_gap:
                print("Gap control: Enabled (require consecutive calendar months)")
            if fill_na:
                print("Fill NA: Enabled (NaN returns treated as 0%)")
            if drop_na:
                print("Drop NA: Enabled (skip NaN, use J valid returns)")
            print("Sorting on: past returns")

    def compute_signal(self, data):
        """
        Compute long-term reversal signal.

        The signal is the cumulative return over lookback_period minus the
        cumulative return over skip period. Handles gaps and NaN values based
        on no_gap, fill_na, and drop_na parameters.
        """
        if 'ret' not in data.columns:
            raise ValueError("Data must contain 'ret' column")

        data = data.copy()
        data = data.sort_values(['ID', 'date'], ignore_index=True)

        J = self.lookback_period
        skip = self.skip

        # Create month index for gap detection
        if self.no_gap:
            data['month_idx'] = (
                data['date'].dt.year * 12 + data['date'].dt.month
            )

        if self.drop_na:
            # Option B: Variable window to accumulate valid (non-NaN) returns
            # For LT reversal: compound return over J periods excluding recent skip periods
            # signal = exp(long_log - recent_log) - 1

            data['logret'] = np.log(data['ret'] + 1)
            data['cumlogret'] = data.groupby('ID')['logret'].cumsum()
            data['cumvalid'] = data.groupby('ID')['ret'].transform(
                lambda x: x.notna().cumsum()
            )
            data['is_valid'] = data['ret'].notna()

            def compute_drop_na_signal(group):
                """Compute LT reversal signal using valid returns."""
                group = group.copy()
                n = len(group)
                signal = np.full(n, np.nan)
                cumlogret = group['cumlogret'].values
                cumvalid = group['cumvalid'].values
                is_valid = group['is_valid'].values

                for i in range(n):
                    if not is_valid[i] or cumvalid[i] < J:
                        continue

                    # Long-term: sum of log returns for J valid returns ending at i
                    target_lt = cumvalid[i] - J
                    # Recent: sum of log returns for skip valid returns ending at i
                    target_recent = cumvalid[i] - skip

                    lt_logsum = None
                    recent_logsum = None

                    if target_lt == 0:
                        lt_logsum = cumlogret[i]
                    else:
                        # Find the last VALID row where cumvalid == target_lt
                        for j in range(i - 1, -1, -1):
                            if is_valid[j] and cumvalid[j] == target_lt:
                                lt_logsum = cumlogret[i] - cumlogret[j]
                                break

                    if target_recent == 0:
                        recent_logsum = cumlogret[i]
                    else:
                        # Find the last VALID row where cumvalid == target_recent
                        for j in range(i - 1, -1, -1):
                            if is_valid[j] and cumvalid[j] == target_recent:
                                recent_logsum = cumlogret[i] - cumlogret[j]
                                break

                    if lt_logsum is not None and recent_logsum is not None:
                        signal[i] = lt_logsum - recent_logsum

                group['signal'] = signal
                return group

            data = data.groupby('ID', group_keys=False).apply(compute_drop_na_signal)
            data['signal'] = np.exp(data['signal']) - 1
            data.drop(columns=['cumlogret', 'cumvalid', 'logret', 'is_valid'], inplace=True)

        elif self.fill_na:
            # Option A: Fixed window, NaN returns treated as 0%
            ret_col = data['ret'].fillna(0)
            data['logret'] = np.log(ret_col + 1)

            long_log = (
                data.groupby(['ID'], group_keys=False)['logret']
                .rolling(window=J, min_periods=J)
                .sum()
            )
            recent_log = (
                data.groupby(['ID'], group_keys=False)['logret']
                .rolling(window=skip, min_periods=skip)
                .sum()
            )
            data['signal'] = np.exp(long_log.values - recent_log.values) - 1
            data.drop(columns=['logret'], inplace=True)

        else:
            # Default: Standard rolling, NaN propagates
            # signal = exp(long_log - recent_log) - 1
            data['logret'] = np.log(data['ret'] + 1)

            long_log = (
                data.groupby(['ID'], group_keys=False)['logret']
                .rolling(window=J, min_periods=J)
                .sum()
            )
            recent_log = (
                data.groupby(['ID'], group_keys=False)['logret']
                .rolling(window=skip, min_periods=skip)
                .sum()
            )
            data['signal'] = np.exp(long_log.values - recent_log.values) - 1
            data.drop(columns=['logret'], inplace=True)

        # Apply no_gap check: invalidate signal if months are not consecutive
        if self.no_gap:
            # Check for gaps in the full lookback window (J months)
            data['lmonth_idx'] = data.groupby('ID')['month_idx'].shift(J - 1)
            data['month_diff'] = data['month_idx'] - data['lmonth_idx']
            data.loc[data['month_diff'] != (J - 1), 'signal'] = np.nan
            data.drop(columns=['month_idx', 'lmonth_idx', 'month_diff'], inplace=True)

        return data

    def get_sort_var(self, adj=None):
        """Return the sorting variable name and update self.sort_var."""
        self.sort_var = f'signal_{adj}' if adj else 'signal'
        return self.sort_var


#==============================================================================
#   Within-Firm Factor: Divergence in Firm Performance Signals
#==============================================================================
class WithinFirmSort(Strategy):
    """
    Within-firm sorting strategy for constructing factors.

    This strategy implements a within-firm high-low methodology:
    1. Groups bonds by date, rating tercile, and firm
    2. Within each group, sorts bonds into high/low by signal (33rd/67th percentiles)
    3. Computes value-weighted returns for high and low portfolios within each firm
    4. Aggregates across firms (firm cap-weighted) within each rating group
    5. Averages factor returns across rating groups

    The key difference from standard sorting: portfolio formation happens within each firm,
    isolating within-firm bond dispersion from cross-firm differences.

    Parameters
    ----------
    sort_var : str
        Sorting variable (e.g., 'eff_yld', 'oas', 'CS')
    holding_period : int, default 1
        Holding period for the strategy. Currently only holding_period=1 is supported.
    firm_id_col : str, default 'PERMNO'
        Column name for firm identifier
    min_bonds_per_firm : int, default 2
        Minimum bonds required per firm-date-rating group
    rating_bins : list of float, optional
        Custom bins for rating terciles. Default: [-inf, 7, 10, inf]
        Creates: IG+ (1-7), IG- (8-10), SG (11+)
    num_portfolios : int, default 2
        Number of portfolios (should be 2 for high/low)
    skip : int, optional
        Skip period between formation and holding
    rebalance_frequency : str or int, default 'monthly'
        Rebalancing frequency
    rebalance_month : int or list of int, default 6
        Month(s) when rebalancing occurs
    verbose : bool, default True
        Print initialization details

    Notes
    -----
    - This strategy returns 2 portfolios: Low (Q1) and High (Q2)
    - The high-low factor is Q2 - Q1 (high minus low)
    - Turnover is computed using PyBondLab's standard turnover machinery
    """

    def __init__(
        self,
        sort_var: str,
        holding_period: int = 1,
        firm_id_col: str = 'PERMNO',
        min_bonds_per_firm: int = 2,
        rating_bins: Optional[List[float]] = None,
        num_portfolios: int = 2,
        skip: Optional[int] = None,
        rebalance_frequency: Union[str, int] = 'monthly',
        rebalance_month: Union[int, List[int]] = 6,
        verbose: bool = False
    ):
        """Initialize WithinFirmSort strategy."""

        # Validate parameter types (catch common mistakes early)
        if not isinstance(sort_var, str):
            raise TypeError(
                f"sort_var must be a string (column name), got {type(sort_var).__name__}: {sort_var!r}. "
                f"Use keyword arguments: WithinFirmSort(sort_var='column_name', firm_id_col='PERMNO')"
            )
        if not isinstance(holding_period, (int, np.integer)):
            raise TypeError(
                f"holding_period must be an integer, got {type(holding_period).__name__}: {holding_period!r}"
            )
        if not isinstance(firm_id_col, str):
            raise TypeError(
                f"firm_id_col must be a string (column name), got {type(firm_id_col).__name__}: {firm_id_col!r}"
            )
        if not isinstance(min_bonds_per_firm, (int, np.integer)):
            raise TypeError(
                f"min_bonds_per_firm must be an integer, got {type(min_bonds_per_firm).__name__}: {min_bonds_per_firm!r}"
            )

        # Validate num_portfolios
        if num_portfolios != 2:
            warnings.warn(
                f"WithinFirmSort is designed for 2 portfolios (high/low). "
                f"Got num_portfolios={num_portfolios}. Using 2."
            )
            num_portfolios = 2

        # Validate holding_period - HP>1 is currently not supported (cohort averaging bug)
        if holding_period > 1:
            raise ValueError(
                f"WithinFirmSort currently only supports holding_period=1. "
                f"Got holding_period={holding_period}. "
                f"HP>1 staggered rebalancing has known bugs and is disabled."
            )

        # Call parent constructor
        super().__init__(
            holding_period=holding_period,
            num_portfolios=num_portfolios,
            lookback_period=None,
            skip=skip,
            rebalance_frequency=rebalance_frequency,
            rebalance_month=rebalance_month,
            verbose=verbose
        )

        self.__strategy_name__ = "Within-Firm Sort"
        self.sort_var = sort_var
        self.firm_id_col = firm_id_col
        self.min_bonds_per_firm = min_bonds_per_firm

        # Set rating bins
        if rating_bins is None:
            self.rating_bins = [-np.inf, 7, 10, np.inf]
        else:
            self.rating_bins = rating_bins

        # No custom breakpoints for this strategy
        self.breakpoints = None

        # Store numba function reference
        try:
            from numba import njit
            self._numba_available = True
        except ImportError:
            self._numba_available = False
            if verbose:
                print("Warning: numba not available. Within-firm sorting will be slower.")

        if verbose:
            self._print_within_firm_details()

    def _print_within_firm_details(self):
        """Print within-firm sorting details."""
        print(f"Sorting variable: {self.sort_var}")
        print(f"Firm ID column: {self.firm_id_col}")
        print(f"Min bonds per firm: {self.min_bonds_per_firm}")
        print(f"Rating bins: {self.rating_bins}")
        print("Portfolio formation: Within-firm high-low")

    def compute_signal(self, data):
        """
        For within-firm sorting, we don't compute a signal in the traditional sense.
        The sorting happens during portfolio formation based on the sort_var.
        This method just returns data as-is and validates required columns.
        """
        # Validate that firm_id_col exists
        if self.firm_id_col not in data.columns:
            raise ValueError(
                f"Firm ID column '{self.firm_id_col}' not found in data. "
                f"Available columns: {list(data.columns)}"
            )

        # Validate that sort_var exists
        if self.sort_var not in data.columns:
            raise ValueError(
                f"Sort variable '{self.sort_var}' not found in data. "
                f"Available columns: {list(data.columns)}"
            )

        return data

    def get_sort_var(self, adj=None):
        """Return the sorting variable name."""
        return self.sort_var

    def set_sort_var(self, sort_var: str):
        """Set the sorting variable."""
        if not isinstance(sort_var, str):
            raise ValueError("sort_var must be a string")
        self.sort_var = sort_var