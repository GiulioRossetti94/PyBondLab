# -*- coding: utf-8 -*-
"""
Created: 03-11-2025
Last modified: 03-11-2025

@authors: Giulio Rossetti

Constants and configuration values for PyBondLab.

This module centralizes all magic numbers, default values, and configuration
constants used throughout the PyBondLaB.

"""

from typing import List


# =============================================================================
# Rating Constants
# =============================================================================
class RatingBounds:
    """Bond rating numeric boundaries."""
    
    # Investment Grade
    IG_MIN = 1
    IG_MAX = 10
    
    # Non-Investment Grade
    NIG_MIN = 11
    NIG_MAX = 22
    
    # Valid rating categories
    VALID_CATEGORIES = ["IG", "NIG", None]


# =============================================================================
# Default Values
# =============================================================================
class Defaults:
    """Default parameter values.
        Staggered rebalancing is default
    """
    
    # Rebalancing
    REBALANCE_FREQUENCY = 'monthly'
    REBALANCE_MONTH = 6
    
    # Strategy parameters
    SKIP_PERIOD = 1
    NUM_PORTFOLIOS = 10
    
    # Price filtering
    PRICE_THRESHOLD = 25
    
    # Data columns
    REQUIRED_COLUMNS = ["date", "ID", "ret", "RATING_NUM", "VW"]
    
    # Display
    VERBOSE = True


# =============================================================================
# Rebalancing Frequencies
# =============================================================================
class RebalanceFrequency:
    """Valid rebalancing frequency options."""
    
    MONTHLY = 'monthly'
    QUARTERLY = 'quarterly'
    SEMI_ANNUAL = 'semi-annual'
    ANNUAL = 'annual'
    
    VALID_FREQUENCIES = [MONTHLY, QUARTERLY, SEMI_ANNUAL, ANNUAL]
    
    @classmethod
    def is_valid(cls, freq: str) -> bool:
        """Check if frequency is valid."""
        return freq in cls.VALID_FREQUENCIES


# =============================================================================
# Filter Types
# =============================================================================
class FilterType:
    """Valid filter/adjustment types."""
    
    TRIM = 'trim'
    WINS = 'wins'
    PRICE = 'price'
    BOUNCE = 'bounce'
    
    VALID_TYPES = [TRIM, WINS, PRICE, BOUNCE]
    
    @classmethod
    def is_valid(cls, filter_type: str) -> bool:
        """Check if filter type is valid."""
        return filter_type in cls.VALID_TYPES


# =============================================================================
# Double Sort Methods
# =============================================================================
class DoubleSortMethod:
    """Valid double sorting methods."""
    
    CONDITIONAL = 'conditional'
    UNCONDITIONAL = 'unconditional'
    
    VALID_METHODS = [CONDITIONAL, UNCONDITIONAL]
    
    @classmethod
    def is_valid(cls, method: str) -> bool:
        """Check if method is valid."""
        return method in cls.VALID_METHODS


# =============================================================================
# Column Names
# =============================================================================
class ColumnNames:
    """Standard column names used throughout PyBondLab."""
    
    # Core columns
    INDEX = 'index'
    DATE = 'date'
    ID = 'ID'
    RETURN = 'ret'
    RATING = 'RATING_NUM'
    VALUE_WEIGHT = 'VW'
    PRICE = 'PRICE'
    
    # Derived columns
    PORTFOLIO_RANK = 'ptf_rank'
    WEIGHTS = 'weights'
    EQ_WEIGHTS = 'eweights'
    VAL_WEIGHTS = 'vweights'
    COUNT = 'count'
    SIGNAL = 'signal'
    LOG_RETURN = 'logret'
    
    # Required core columns
    REQUIRED = [DATE, ID, RETURN, RATING, VALUE_WEIGHT]


# =============================================================================
# Result Suffixes
# =============================================================================
class ResultSuffix:
    """Suffixes for different result types."""
    
    # Strategy types
    EX_ANTE = '_ea'
    EX_POST = '_ep'
    
    # Weighting types
    EQUAL_WEIGHT = 'ew'
    VALUE_WEIGHT = 'vw'
    
    # Components
    LONG = '_long'
    SHORT = '_short'
    LONG_SHORT = 'ls'


# =============================================================================
# Numeric Constants
# =============================================================================
class NumericConstants:
    """Numeric constants used in calculations."""
    
    # Portfolio numbering
    MIN_PORTFOLIO_RANK = 1
    
    # Percentiles
    PERCENTILE_MIN = 0
    PERCENTILE_MAX = 100
    
    # Infinity values for thresholds
    NEG_INF = float('-inf')
    POS_INF = float('inf')


# =============================================================================
# Validation Messages
# =============================================================================
class ValidationMessages:
    """Standard validation error messages."""
    
    INVALID_RATING = (
        "Invalid rating: {rating}. "
        "Valid options are None, 'IG', 'NIG', or a 2-tuple (min, max)."
    )
    
    INVALID_BOUNDS = (
        "Invalid bounds for column '{col}': "
        "lower bound ({low}) must be <= upper bound ({high})"
    )
    
    INVALID_FILTER = (
        "Invalid filtering option: {adj}. "
        "Valid options are {valid_options}"
    )
    
    INVALID_REBALANCE_FREQ = (
        "rebalance_frequency must be one of {valid_frequencies} or an integer. "
        "Got '{freq}'"
    )
    
    MISSING_COLUMN = (
        "Column '{col}' not found in data. "
        "Available columns: {available}"
    )
    
    EMPTY_DATA = (
        "No bonds matched between time {time_t} and {time_t1}. "
        "Setting return to nan and going to next period."
    )


# =============================================================================
# File Naming Patterns
# =============================================================================
class FilePatterns:
    """Patterns for result file naming."""
    
    # Portfolio labels
    PORTFOLIO_PREFIX = 'Q'  # Q1, Q2, Q3, etc.
    
    # Long/short labels
    LONG_LABEL = 'LONG_{weight}{strategy}_{name}'
    SHORT_LABEL = 'SHORT_{weight}{strategy}_{name}'
    LS_LABEL = '{weight}{strategy}_{name}'


# =============================================================================
# Performance Constants
# =============================================================================
class Performance:
    """Performance-related constants."""
    
    # Caching
    MAX_CACHE_SIZE = 128
    
    # Chunking for large operations
    CHUNK_SIZE = 10000
    
    # Minimum rows for pd.eval optimization
    MIN_ROWS_FOR_EVAL = 100000


# =============================================================================
# Helper Functions for Constants
# =============================================================================
def get_rating_bounds(rating: str) -> tuple:
    """
    Get numeric bounds for a rating category.
    
    Parameters
    ----------
    rating : str
        Rating category ('IG', 'NIG', or None)
    
    Returns
    -------
    tuple
        (min_rating, max_rating)
    
    Examples
    --------
    >>> get_rating_bounds('IG')
    (1, 10)
    >>> get_rating_bounds('NIG')
    (11, 22)
    """
    if rating == "IG":
        return (RatingBounds.IG_MIN, RatingBounds.IG_MAX)
    elif rating == "NIG":
        return (RatingBounds.NIG_MIN, RatingBounds.NIG_MAX)
    elif rating is None:
        return (NumericConstants.NEG_INF, NumericConstants.POS_INF)
    else:
        raise ValueError(ValidationMessages.INVALID_RATING.format(rating=rating))


def get_portfolio_labels(num_portfolios: int) -> List[str]:
    """
    Generate portfolio labels (Q1, Q2, ..., Qn).
    
    Parameters
    ----------
    num_portfolios : int
        Number of portfolios
    
    Returns
    -------
    list of str
        Portfolio labels
    
    Examples
    --------
    >>> get_portfolio_labels(5)
    ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    """
    return [f"{FilePatterns.PORTFOLIO_PREFIX}{i}" 
            for i in range(1, num_portfolios + 1)]


def get_signal_based_labels(
    signal_name: str,
    num_portfolios: int,
    signal_name2: str = None,
    num_portfolios2: int = None
) -> List[str]:
    """
    Generate signal-based portfolio labels.
    
    For single sorts: ['SIGNAL1', 'SIGNAL2', ..., 'SIGNALn']
    For double sorts: ['SIG1_1_SIG2_1', 'SIG1_1_SIG2_2', ..., 'SIG1_n_SIG2_m']
    
    Parameters
    ----------
    signal_name : str
        Primary signal variable name (e.g., 'ret_mom')
    num_portfolios : int
        Number of portfolios for primary sort
    signal_name2 : str, optional
        Secondary signal variable name for double sorts
    num_portfolios2 : int, optional
        Number of portfolios for secondary sort
    
    Returns
    -------
    list of str
        Portfolio labels with signal names
    
    Examples
    --------
    >>> get_signal_based_labels('ret_mom', 5)
    ['RET_MOM1', 'RET_MOM2', 'RET_MOM3', 'RET_MOM4', 'RET_MOM5']
    
    >>> get_signal_based_labels('ret_mom', 3, 'size', 5)
    ['RET_MOM1_SIZE1', 'RET_MOM1_SIZE2', ..., 'RET_MOM3_SIZE5']
    """
    # Convert signal names to uppercase
    sig1 = signal_name.upper()
    
    if signal_name2 is not None and num_portfolios2 is not None:
        # Double sort
        sig2 = signal_name2.upper()
        labels = [
            f'{sig1}{i}_{sig2}{j}'
            for i in range(1, num_portfolios + 1)
            for j in range(1, num_portfolios2 + 1)
        ]
    else:
        # Single sort
        labels = [f'{sig1}{i}' for i in range(1, num_portfolios + 1)]
    
    return labels


def get_adjusted_return_column(base_column: str, adjustment: str) -> str:
    """
    Get the adjusted return column name.
    
    Parameters
    ----------
    base_column : str
        Base column name (e.g., 'ret')
    adjustment : str
        Adjustment type (e.g., 'trim', 'wins')
    
    Returns
    -------
    str
        Adjusted column name (e.g., 'ret_trim')
    
    Examples
    --------
    >>> get_adjusted_return_column('ret', 'trim')
    'ret_trim'
    """
    return f"{base_column}_{adjustment}"