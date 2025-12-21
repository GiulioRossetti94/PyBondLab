"""
Naming configuration for factor outputs.

This module provides a configuration system for generating consistent,
readable factor names across PyBondLab outputs.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class NamingConfig:
    """Configuration for factor naming conventions.

    Attributes
    ----------
    lowercase : bool, default=True
        Use lowercase names (e.g., 'cs' instead of 'CS').
    sign_correct : bool, default=False
        Flip negative factors and add '*' suffix.
        Applied independently for EW and VW.
    use_signal_name : bool, default=True
        Use signal column name as base (e.g., 'cs' instead of 'factor').
    weighting_prefix : bool, default=False
        Add 'ew_' or 'vw_' prefix to factor names.
    include_rating_suffix : bool, default=True
        Add '_ig' or '_hy' suffix for rating-filtered strategies.
    include_wf_suffix : bool, default=True
        Add '_wf' suffix for WithinFirmSort strategies.
    doublesort_sep : str, default='_'
        Separator for DoubleSort factor names (e.g., 'cs_duration').

    Examples
    --------
    Default config (recommended):

    >>> cfg = NamingConfig()
    >>> # SingleSort: 'cs', 'cs_ig', 'cs*'
    >>> # DoubleSort: 'cs_duration'
    >>> # WithinFirmSort: 'cs_wf', portfolios: 'cs_low', 'cs_high'

    With weighting prefix:

    >>> cfg = NamingConfig(weighting_prefix=True)
    >>> # 'ew_cs', 'vw_cs', 'ew_cs_ig', 'vw_cs*'

    Without signal name (legacy style):

    >>> cfg = NamingConfig(use_signal_name=False)
    >>> # 'factor', 'factor_ig', 'factor*'
    """
    lowercase: bool = True
    sign_correct: bool = False
    use_signal_name: bool = True
    weighting_prefix: bool = False
    include_rating_suffix: bool = True
    include_wf_suffix: bool = True
    doublesort_sep: str = '_'


def make_factor_name(
    signal_name: str,
    config: NamingConfig,
    *,
    weighting: Optional[str] = None,
    rating: Optional[str] = None,
    is_within_firm: bool = False,
    sign_corrected: bool = False,
    second_signal: Optional[str] = None,
) -> str:
    """Generate factor name based on config.

    Parameters
    ----------
    signal_name : str
        Base signal name (e.g., 'CS', 'duration').
    config : NamingConfig
        Naming configuration.
    weighting : str, optional
        'ew' or 'vw' for weighting prefix.
    rating : str, optional
        'ig' or 'hy' for rating suffix.
    is_within_firm : bool
        Add '_wf' suffix for WithinFirmSort.
    sign_corrected : bool
        Add '*' suffix if sign was flipped.
    second_signal : str, optional
        Second signal for DoubleSort (e.g., 'duration').

    Returns
    -------
    str
        Formatted factor name.

    Examples
    --------
    >>> cfg = NamingConfig()
    >>> make_factor_name('CS', cfg)
    'cs'
    >>> make_factor_name('CS', cfg, rating='ig')
    'cs_ig'
    >>> make_factor_name('CS', cfg, is_within_firm=True)
    'cs_wf'
    >>> make_factor_name('CS', cfg, sign_corrected=True)
    'cs*'
    >>> make_factor_name('CS', cfg, second_signal='duration')
    'cs_duration'
    >>> make_factor_name('CS', NamingConfig(weighting_prefix=True), weighting='ew')
    'ew_cs'
    """
    # Build base name
    if config.use_signal_name:
        name = signal_name
    else:
        name = 'factor'

    # Apply case
    if config.lowercase:
        name = name.lower()
        if second_signal:
            second_signal = second_signal.lower()

    # DoubleSort: add second signal
    if second_signal:
        name = f"{name}{config.doublesort_sep}{second_signal}"

    # WithinFirmSort suffix
    if is_within_firm and config.include_wf_suffix:
        name = f"{name}_wf"

    # Rating suffix
    if rating and config.include_rating_suffix:
        name = f"{name}_{rating}"

    # Weighting prefix
    if weighting and config.weighting_prefix:
        name = f"{weighting}_{name}"

    # Sign correction suffix (added last)
    if sign_corrected:
        name = f"{name}*"

    return name


def make_portfolio_name(
    signal_name: str,
    portfolio_num: int,
    num_portfolios: int,
    config: NamingConfig,
    *,
    second_signal: Optional[str] = None,
    second_portfolio_num: Optional[int] = None,
    is_within_firm: bool = False,
) -> str:
    """Generate portfolio name (e.g., 'cs1', 'cs5', 'cs_low', 'cs_high').

    Parameters
    ----------
    signal_name : str
        Base signal name (e.g., 'CS').
    portfolio_num : int
        Portfolio number (1 to num_portfolios).
        For WithinFirmSort: 1 = LOW, 2 = HIGH.
    num_portfolios : int
        Total number of portfolios.
    config : NamingConfig
        Naming configuration.
    second_signal : str, optional
        Second signal for DoubleSort.
    second_portfolio_num : int, optional
        Second portfolio number for DoubleSort.
    is_within_firm : bool
        Use _low/_high naming for WithinFirmSort.

    Returns
    -------
    str
        Formatted portfolio name.

    Examples
    --------
    >>> cfg = NamingConfig()
    >>> make_portfolio_name('CS', 1, 5, cfg)
    'cs1'
    >>> make_portfolio_name('CS', 5, 5, cfg)
    'cs5'
    >>> make_portfolio_name('CS', 1, 2, cfg, is_within_firm=True)
    'cs_low'
    >>> make_portfolio_name('CS', 2, 2, cfg, is_within_firm=True)
    'cs_high'
    >>> make_portfolio_name('CS', 1, 5, cfg, second_signal='dur', second_portfolio_num=3)
    'cs1_dur3'
    """
    base = signal_name.lower() if config.lowercase else signal_name

    if is_within_firm:
        # WithinFirmSort: use _low, _high
        suffix = '_low' if portfolio_num == 1 else '_high'
        return f"{base}{suffix}"

    if second_signal:
        # DoubleSort: cs1_dur1, cs5_dur5
        sec = second_signal.lower() if config.lowercase else second_signal
        return f"{base}{portfolio_num}{config.doublesort_sep}{sec}{second_portfolio_num}"

    # SingleSort: cs1, cs2, ..., cs5
    return f"{base}{portfolio_num}"


def rating_to_suffix(rating) -> Optional[str]:
    """Convert rating parameter to suffix string.

    Parameters
    ----------
    rating : str, tuple, or None
        Rating specification: 'IG', 'NIG', (min, max) tuple, or None.

    Returns
    -------
    str or None
        'ig' for investment grade, 'hy' for high yield, None otherwise.
    """
    if rating is None:
        return None
    if isinstance(rating, str):
        if rating.upper() == 'IG':
            return 'ig'
        elif rating.upper() in ('NIG', 'HY'):
            return 'hy'
    elif isinstance(rating, tuple):
        # Tuple (min, max) - check if it's IG range or HY range
        min_rating, max_rating = rating
        if max_rating <= 10:
            return 'ig'
        elif min_rating > 10:
            return 'hy'
    return None
