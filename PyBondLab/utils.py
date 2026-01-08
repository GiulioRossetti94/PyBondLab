import numpy as np
import pandas as pd
import warnings
from typing import Tuple, Union, Optional, List, Callable, Literal


# =============================================================================
# Panel Data Validation
# =============================================================================

def validate_panel(
    data: pd.DataFrame,
    id_col: str = 'ID',
    date_col: str = 'date',
    handle_duplicates: Literal['error', 'warn', 'drop'] = 'warn',
    keep: Literal['first', 'last'] = 'first',
    verbose: bool = True,
    # Aliases for consistency with StrategyFormation.fit()
    IDvar: Optional[str] = None,
    DATEvar: Optional[str] = None,
) -> pd.DataFrame:
    """
    Validate panel data for duplicate ID-date pairs.

    Duplicate ID-date pairs cause errors in portfolio formation because
    they create non-unique indices when mapping ranks and weights.
    This function detects duplicates and optionally removes them.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data to validate
    id_col : str, default 'ID'
        Column name for entity identifier (e.g., bond CUSIP).
        Alias: IDvar (for consistency with StrategyFormation.fit())
    date_col : str, default 'date'
        Column name for date identifier.
        Alias: DATEvar (for consistency with StrategyFormation.fit())
    handle_duplicates : str, default 'warn'
        How to handle duplicates:
        - 'error': Raise ValueError if duplicates found
        - 'warn': Print warning and return original data
        - 'drop': Remove duplicates and return cleaned data
    keep : str, default 'first'
        When dropping duplicates, which occurrence to keep:
        - 'first': Keep first occurrence
        - 'last': Keep last occurrence
    verbose : bool, default True
        If True, print summary of validation results
    IDvar : str, optional
        Alias for id_col (for consistency with StrategyFormation.fit())
    DATEvar : str, optional
        Alias for date_col (for consistency with StrategyFormation.fit())

    Returns
    -------
    pd.DataFrame
        The validated data (cleaned if handle_duplicates='drop')

    Raises
    ------
    ValueError
        If handle_duplicates='error' and duplicates are found
        If required columns are missing

    Examples
    --------
    >>> from PyBondLab import validate_panel
    >>>
    >>> # Check data and warn about duplicates (using default column names)
    >>> df = validate_panel(bond_data)
    >>>
    >>> # Check with custom column names
    >>> df = validate_panel(bond_data, id_col='cusip', date_col='trd_date')
    >>>
    >>> # Or using the same parameter names as StrategyFormation.fit()
    >>> df = validate_panel(bond_data, IDvar='cusip', DATEvar='trd_date')
    >>>
    >>> # Check and automatically drop duplicates
    >>> df_clean = validate_panel(bond_data, handle_duplicates='drop')
    >>>
    >>> # Strict mode - raise error if duplicates found
    >>> df = validate_panel(bond_data, handle_duplicates='error')
    >>>
    >>> # Typical workflow with custom column names
    >>> df_clean = validate_panel(data, IDvar='cusip', DATEvar='trd_date',
    ...                           handle_duplicates='drop')
    >>> sf = StrategyFormation(data=df_clean, strategy=strategy)
    >>> sf.fit(IDvar='cusip', DATEvar='trd_date')

    Notes
    -----
    Performance: Checking for duplicates is O(n) and takes approximately:
    - 100K rows: ~4ms
    - 1M rows: ~17ms
    - 5M rows: ~145ms

    For loops over multiple strategies, call this function ONCE before the
    loop to avoid repeated validation overhead.
    """
    # Handle alias parameters (IDvar/DATEvar take precedence if provided)
    if IDvar is not None:
        id_col = IDvar
    if DATEvar is not None:
        date_col = DATEvar

    # Validate inputs
    if id_col not in data.columns:
        raise ValueError(
            f"ID column '{id_col}' not found in data. "
            f"Available columns: {list(data.columns)}"
        )
    if date_col not in data.columns:
        raise ValueError(
            f"Date column '{date_col}' not found in data. "
            f"Available columns: {list(data.columns)}"
        )

    n_rows_original = len(data)

    # Check for duplicates
    n_duplicates = data.duplicated(subset=[id_col, date_col]).sum()

    # No duplicates - return original data
    if n_duplicates == 0:
        if verbose:
            print(f"Panel validation: OK ({n_rows_original:,} rows, no duplicates)")
        return data

    # Duplicates found - get examples for error messages
    dup_mask = data.duplicated(subset=[id_col, date_col], keep=False)
    dup_pairs = data[dup_mask][[id_col, date_col]].drop_duplicates()
    examples = list(dup_pairs.head(5).itertuples(index=False, name=None))

    if handle_duplicates == 'error':
        raise ValueError(
            f"Panel data contains {n_duplicates:,} duplicate {id_col}-{date_col} pairs. "
            f"Examples: {examples}. "
            f"Use handle_duplicates='drop' to remove them automatically."
        )

    elif handle_duplicates == 'warn':
        warnings.warn(
            f"Panel data contains {n_duplicates:,} duplicate {id_col}-{date_col} pairs. "
            f"This may cause errors in portfolio formation. "
            f"Use validate_panel(data, handle_duplicates='drop') to fix.",
            UserWarning
        )
        if verbose:
            print(f"Panel validation: WARNING")
            print(f"  - {n_rows_original:,} total rows")
            print(f"  - {n_duplicates:,} duplicate {id_col}-{date_col} pairs found")
            print(f"  - Example duplicates: {examples[:3]}")
        return data

    elif handle_duplicates == 'drop':
        result_data = data.drop_duplicates(subset=[id_col, date_col], keep=keep)
        n_rows_after = len(result_data)

        if verbose:
            print(f"Panel validation: FIXED")
            print(f"  - {n_rows_original:,} original rows")
            print(f"  - {n_duplicates:,} duplicate rows removed (kept='{keep}')")
            print(f"  - {n_rows_after:,} rows remaining")
        return result_data

    else:
        raise ValueError(
            f"Invalid handle_duplicates='{handle_duplicates}'. "
            f"Must be one of: 'error', 'warn', 'drop'"
        )


def check_duplicates(
    data: pd.DataFrame,
    id_col: str = 'ID',
    date_col: str = 'date',
    # Aliases for consistency with StrategyFormation.fit()
    IDvar: Optional[str] = None,
    DATEvar: Optional[str] = None,
) -> Tuple[bool, int]:
    """
    Quick check for duplicate ID-date pairs.

    This is a lightweight alternative to validate_panel() that just
    returns whether duplicates exist and how many.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data to check
    id_col : str, default 'ID'
        Column name for entity identifier.
        Alias: IDvar (for consistency with StrategyFormation.fit())
    date_col : str, default 'date'
        Column name for date identifier.
        Alias: DATEvar (for consistency with StrategyFormation.fit())
    IDvar : str, optional
        Alias for id_col (for consistency with StrategyFormation.fit())
    DATEvar : str, optional
        Alias for date_col (for consistency with StrategyFormation.fit())

    Returns
    -------
    tuple
        (has_duplicates: bool, n_duplicates: int)

    Examples
    --------
    >>> has_dups, n_dups = check_duplicates(data)
    >>> if has_dups:
    ...     print(f"Warning: {n_dups} duplicates found!")
    >>>
    >>> # With custom column names
    >>> has_dups, n_dups = check_duplicates(data, IDvar='cusip', DATEvar='trd_date')
    """
    # Handle alias parameters (IDvar/DATEvar take precedence if provided)
    if IDvar is not None:
        id_col = IDvar
    if DATEvar is not None:
        date_col = DATEvar
    n_dups = data.duplicated(subset=[id_col, date_col]).sum()
    return n_dups > 0, n_dups


#---- Portfolio sorting utilities
def compute_thresholds(
    data: pd.DataFrame,
    sig: str,
    breakpoints: Union[int, List[float]] = 10,
    subset: Optional[pd.Series] = None
) -> np.ndarray:
    """
    Compute threshold edges for portfolio sorting.

    Parameters:
    - data : pd.DataFrame - asset universe at t
    - sig : str - signal column name
    - breakpoints : int or list of float -
        int = number of portfolios (even percentiles)
        list = custom percentiles (e.g. [30, 70])
    - subset : optional pd.Series bool mask - restrict data used to compute breakpoints for example just NYSE stocks

    Returns:
    - np.ndarray of threshold edges
    """
    if subset is not None:
        data = data.loc[subset]

    if isinstance(breakpoints, int):
        percentiles = np.linspace(0, 100, breakpoints + 1)
    else:
        percentiles = [0] + breakpoints + [100]

    # Filter out NaN and inf values before computing percentiles
    # This prevents np.percentile from returning nan when data contains inf
    sig_values = data[sig].values
    valid_mask = ~np.isnan(sig_values) & ~np.isinf(sig_values)
    valid_values = sig_values[valid_mask]

    if len(valid_values) == 0:
        # Return array of NaN if no valid values
        return np.full(len(percentiles), np.nan)

    thres = np.percentile(valid_values, percentiles)
    thres[0] = -np.inf
    return thres

def intersect_id(
        It0: pd.DataFrame,
        It1: pd.DataFrame, 
        It1m: pd.DataFrame, 
        dynamic_weights: bool
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Find common IDs across time periods, filter dataframes.

    Parameters:
    - It0 : pd.DataFrame
    - It1 : pd.DataFrame
    - It1m : pd.DataFrame
    - dynamic_weights : bool

    Returns:
    - It0_filtered, It1_filtered, It1m_filtered : pd.DataFrames
    """
    id0 = It0['ID']
    id1 = It1['ID']
    id2 = It1m['ID']

    ids_0_1 = id0[id0.isin(id1)]
    ids_0_1_1m = ids_0_1[ids_0_1.isin(id2)]

    if dynamic_weights:
        final_ids = ids_0_1_1m
    else:
        final_ids = ids_0_1

    It0f = It0[It0['ID'].isin(final_ids)].copy()
    It1f = It1[It1['ID'].isin(final_ids)].copy()
    It1mf = It1m[It1m['ID'].isin(final_ids)].copy()

    return It0f, It1f, It1mf

def assign_bond_bins(
    sortvar: Union[np.ndarray, pd.Series],
    thres: Union[np.ndarray, list],
    nport: int
) -> np.ndarray:
    """
    Assign assets to bins based on thresholds.
    
    Parameters:
    - sortvar: array-like, values to sort on
    - thres: array-like, threshold edges
    - nport: int, number of portfolios

    Returns:
    - np.ndarray of bin assignments (1-based)
    """
    idx = np.full(sortvar.shape, np.nan)
    for p in range(nport):
        f = (sortvar > thres[p]) & (sortvar <= thres[p + 1])
        idx[f] = p + 1
    return idx

#---- custom breakpoints universe
def create_subset_mask(data: pd.DataFrame, 
                       subset_function: Optional[Union[str, Callable]]) -> Optional[pd.Series]:
    """
    Create a boolean mask for subsetting data based on subset_function.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to create mask from
    subset_function : str, callable, or None
        - str: column name, returns data[column] == 1
        - callable: function(data) -> boolean Series
        - None: returns None (no subsetting)
    
    Returns
    -------
    pd.Series (boolean) or None
    """
    if subset_function is None:
        return None
    
    if isinstance(subset_function, str):
        # Column name: use data[column] == 1 as filter
        if subset_function not in data.columns:
            raise ValueError(f"Column '{subset_function}' not found in data. "
                           f"Available columns: {list(data.columns)}")
        return data[subset_function] == 1
    
    elif callable(subset_function):
        # Custom function
        mask = subset_function(data)
        if not isinstance(mask, pd.Series):
            mask = pd.Series(mask, index=data.index)
        if not pd.api.types.is_bool_dtype(mask):
            raise ValueError("subset_function must return a boolean Series")
        return mask
    
    else:
        raise ValueError("subset_function must be a string (column name) or callable")


#---- double sorting
def double_sort_uncond(
    idx1: Union[np.ndarray, pd.Series],
    idx2: Union[np.ndarray, pd.Series],
    n1: int,
    n2: int
) -> np.ndarray:
    """
    Create combined rank from two independent sorts.

    Returns:
    - np.ndarray of combined rank
    """
    idx = np.full(idx1.shape, np.nan)
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            idx[(idx1 == i) & (idx2 == j)] = (i - 1) * n2 + j
    return idx

def double_sort_cond(
    sortvar2: Union[np.ndarray, pd.Series],
    idx1: Union[np.ndarray, pd.Series],
    n1: int,
    n2: int
) -> np.ndarray:
    """
    Conditional double sort: sort second signal within first signal bins.
    
    Returns:
    - np.ndarray of combined rank
    """
    idx = np.zeros_like(sortvar2)  # initialize the array 
    
    # loop over the number of portfolios and assign the rank within sorted bins
    for i in range(1, n1 + 1):
        temp = sortvar2[idx1 == i]  # get sortvar2 values for stocks in bin i
        # if temp is empty then skip. This means that there are no assets in bin i
        if len(temp) == 0:
            continue

        # sort values within the bin i
        thres2 = np.percentile(temp, np.linspace(0, 100, n2 + 1))
        thres2[0] = -np.inf

        # assign a rank based on the break points in thres2
        id2 = assign_bond_bins(temp, thres2, n2)
        nmax = np.max(id2)

        # assign bonds to idx from 1 to n1*n2
        idx[idx1 == i] = id2 + nmax * (i - 1)
    return idx

#---- Non-staggered rebalancing utilities

def _get_rebalancing_dates(datelist: List[pd.Timestamp], 
                          rebal_freq: Union[str, int], 
                          rebal_month: Union[int, List[int]]) -> List[int]:
    """
    Determine which date indices trigger portfolio rebalancing.
    
    Parameters
    ----------
    datelist : list of pd.Timestamp
        List of all dates in the dataset
    rebal_freq : str or int
        Rebalancing frequency
    rebal_month : int or list of int
        Month(s) for rebalancing
    
    Returns
    -------
    list of int
        Indices in datelist where rebalancing occurs
    """
    TM = len(datelist)
    
    if rebal_freq == 'monthly':
        # All dates are rebalancing dates (staggered overlapping)
        return list(range(TM))
    
    elif rebal_freq == 'quarterly':
        # Rebalance every 3 months
        if isinstance(rebal_month, int):
            rebal_months = [(rebal_month + 3 * i - 1) % 12 + 1 for i in range(4)]
        else:
            rebal_months = rebal_month
        return [i for i, date in enumerate(datelist) if date.month in rebal_months]
    
    elif rebal_freq == 'semi-annual':
        # Rebalance every 6 months
        if isinstance(rebal_month, int):
            rebal_months = [rebal_month, (rebal_month + 6 - 1) % 12 + 1]
        else:
            rebal_months = rebal_month
        return [i for i, date in enumerate(datelist) if date.month in rebal_months]
    
    elif rebal_freq == 'annual':
        # Rebalance once per year
        if isinstance(rebal_month, list):
            rebal_months = rebal_month
        else:
            rebal_months = [rebal_month]
        return [i for i, date in enumerate(datelist) if date.month in rebal_months]
    
    elif isinstance(rebal_freq, int):
        # Custom frequency: rebalance every N months
        start_month = rebal_month if isinstance(rebal_month, int) else rebal_month[0]
        rebal_dates = []
        
        # Find first occurrence of start_month
        first_idx = None
        for i, date in enumerate(datelist):
            if date.month == start_month:
                first_idx = i
                break
        
        if first_idx is None:
            # Start month not found, use first date
            first_idx = 0
        
        # Add rebalancing dates every N months
        rebal_dates.append(first_idx)
        current_date = datelist[first_idx]
        
        for i in range(first_idx + 1, TM):
            months_diff = (datelist[i].year - current_date.year) * 12 + \
                         (datelist[i].month - current_date.month)
            
            if months_diff >= rebal_freq:
                rebal_dates.append(i)
                current_date = datelist[i]
        
        return rebal_dates
    
    else:
        raise ValueError(f"Unsupported rebalance_frequency: {rebal_freq}")

#---- Assaying utilities

def summarize_ranks(dfs_by_date: dict) -> pd.DataFrame:
    dates      = []
    cnt1       = []
    cnt_max    = []
    
    for date, df in dfs_by_date.items():
        arr = df['ptf_rank'].values               # 1. pull out raw numpy array
        mx  = arr.max()                           # 2. compute the max in C
        cnt1.append(np.count_nonzero(arr == 1))   # 3. count “==1” in C
        cnt_max.append(np.count_nonzero(arr == mx))
        dates.append(date)
    
    # 4. build exactly one DataFrame at the end
    result = pd.DataFrame({
        'nbonds_s':      cnt1,
        'nbonds_l':   cnt_max,
    }, index=pd.to_datetime(dates))

    result['nbonds_ls'] = result['nbonds_s'] + result['nbonds_l']

    return result.sort_index()