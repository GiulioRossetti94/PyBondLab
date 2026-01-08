import numpy as np
import pandas as pd
from typing import Tuple, Union, Optional, List, Callable

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

    thres = np.percentile(data[sig], percentiles)
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