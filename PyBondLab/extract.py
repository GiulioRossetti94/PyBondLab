# -*- coding: utf-8 -*-
"""
Panel extraction utilities for PyBondLab batch results.

This module provides functions to extract unified panel DataFrames
from BatchStrategyFormation and BatchWithinFirmSortFormation results.

Authors: Giulio Rossetti & Alex Dickerson
"""

from __future__ import annotations

from typing import Optional, Union, List, Dict, Any
import numpy as np
import pandas as pd

from PyBondLab.naming import NamingConfig, make_factor_name


__all__ = ["extract_panel"]


def extract_panel(
    results,
    naming: Optional[NamingConfig] = None,
) -> pd.DataFrame:
    """
    Extract unified panel DataFrame from batch results.

    Parameters
    ----------
    results : BatchResults or BatchWithinFirmResults
        Batch processing results from BatchStrategyFormation or
        BatchWithinFirmSortFormation.
    naming : NamingConfig, optional
        Naming configuration. If sign_correct=True, factors with
        negative mean are flipped and legs are swapped.
        Default: NamingConfig() (lowercase, no sign correction)

    Returns
    -------
    pd.DataFrame
        Panel with columns:
        - date: Observation date
        - factor: Factor name (with * suffix if sign-corrected)
        - freq: Holding period (1=monthly, 3=quarterly, etc.)
        - leg: 'ls', 'l', or 's'
        - weighting: 'ew' or 'vw'
        - return: Portfolio return
        - turnover: Turnover (if computed, else NaN)
        - count: Number of bonds in portfolio (if computed, else NaN)
        - {char_name}: Characteristic values (if computed, else not present)

    Notes
    -----
    - For sign-corrected factors, 'l' and 's' legs are swapped
    - Turnover for 'ls' leg is factor turnover: (L + S) / 2 (average of both legs)
    - Count for 'ls' leg is total bonds: L + S (sum of both legs)
    - Chars for 'ls' leg is L - S spread
    - WithinFirmSort 'high' maps to 'l', 'low' maps to 's'

    Examples
    --------
    >>> from PyBondLab import BatchStrategyFormation, extract_panel, NamingConfig
    >>> batch = BatchStrategyFormation(data, signals=['cs', 'value'], ...)
    >>> results = batch.fit()
    >>> panel = extract_panel(results, naming=NamingConfig(sign_correct=True))
    >>> print(panel.head())
    """
    if naming is None:
        naming = NamingConfig()

    # Check if any results exist
    if len(results) == 0:
        raise ValueError("No results in batch - nothing to extract")

    # Get first result to detect metadata
    first_signal = next(iter(results))
    first_result = results[first_signal]

    # Determine if this is a FormationResults (has .ea) or direct StrategyResults
    if hasattr(first_result, 'ea'):
        # FormationResults - access via .ea
        first_sr = first_result.ea
    else:
        # Direct StrategyResults
        first_sr = first_result

    # Check if this is a fast batch result (lacks full portfolio data)
    # Fast batch results only have long-short returns, not individual legs
    is_fast_batch = not hasattr(first_sr, 'has_turnover')
    if is_fast_batch:
        raise ValueError(
            "extract_panel requires full portfolio results with individual leg returns. "
            "The batch results appear to be from the fast path (turnover=False, chars=None). "
            "Re-run BatchStrategyFormation with turnover=True to get full results."
        )

    # Detect available features
    has_turnover = first_sr.has_turnover
    has_chars = first_sr.has_characteristics
    has_counts = first_sr.has_bond_counts
    char_names = []
    if has_chars:
        char_names = first_sr.characteristics.available_characteristics

    # Get holding period from config
    holding_period = results.config.get('holding_period', 1)

    # Detect if WithinFirmSort
    is_within_firm = results.config.get('is_within_firm', False)
    if not is_within_firm:
        # Check from first result metadata
        is_within_firm = getattr(first_sr, 'is_within_firm', False)

    # Build panel rows
    rows = []

    for signal_name in results:
        result = results[signal_name]

        # Get StrategyResults (either directly or via .ea)
        if hasattr(result, 'ea'):
            sr = result.ea
        else:
            sr = result

        # Get returns for all legs
        ew_ls = sr.returns.get_long_short("ew")
        vw_ls = sr.returns.get_long_short("vw")
        ew_long = sr.returns.get_long_leg("ew")
        vw_long = sr.returns.get_long_leg("vw")
        ew_short = sr.returns.get_short_leg("ew")
        vw_short = sr.returns.get_short_leg("vw")

        # Check sign correction needed (independent for EW and VW)
        ew_flip = naming.sign_correct and ew_ls.mean() < 0
        vw_flip = naming.sign_correct and vw_ls.mean() < 0

        # Generate factor names
        ew_factor_name = make_factor_name(
            signal_name, naming, sign_corrected=ew_flip,
            is_within_firm=is_within_firm,
            rating=getattr(sr, 'rating_str', None),
            second_signal=getattr(sr, 'second_signal', None),
        )
        vw_factor_name = make_factor_name(
            signal_name, naming, sign_corrected=vw_flip,
            is_within_firm=is_within_firm,
            rating=getattr(sr, 'rating_str', None),
            second_signal=getattr(sr, 'second_signal', None),
        )

        # Get turnover if available
        ew_factor_turn = None
        vw_factor_turn = None
        ew_long_turn = None
        vw_long_turn = None
        ew_short_turn = None
        vw_short_turn = None

        if has_turnover:
            ew_turn_df, vw_turn_df = sr.get_turnover(level='portfolio')
            nport = ew_turn_df.shape[1]

            # Check for DoubleSort (average across conditioning groups)
            second_signal = getattr(sr, 'second_signal', None)
            num_portfolios = getattr(sr, 'num_portfolios', None)

            if second_signal is not None and num_portfolios is not None:
                # DoubleSort: average across conditioning groups
                n1 = num_portfolios
                n2 = nport // n1
                short_cols = list(range(n2))
                long_cols = list(range((n1 - 1) * n2, n1 * n2))

                ew_long_turn = ew_turn_df.iloc[:, long_cols].mean(axis=1)
                ew_short_turn = ew_turn_df.iloc[:, short_cols].mean(axis=1)
                vw_long_turn = vw_turn_df.iloc[:, long_cols].mean(axis=1)
                vw_short_turn = vw_turn_df.iloc[:, short_cols].mean(axis=1)
            else:
                # SingleSort or WithinFirmSort
                ew_long_turn = ew_turn_df.iloc[:, nport - 1]
                ew_short_turn = ew_turn_df.iloc[:, 0]
                vw_long_turn = vw_turn_df.iloc[:, nport - 1]
                vw_short_turn = vw_turn_df.iloc[:, 0]

            # Factor turnover = average of long and short legs
            # This represents turnover as a fraction of total portfolio capital
            # (since L-S has 100% long and 100% short = 200% total)
            ew_factor_turn = (ew_long_turn + ew_short_turn) / 2
            vw_factor_turn = (vw_long_turn + vw_short_turn) / 2

        # Get bond counts if available
        # Note: bond counts are the same for EW and VW (just count of bonds, not weighted)
        long_count = None
        short_count = None
        factor_count = None

        if has_counts:
            count_df = sr.get_bond_count()
            nport_count = count_df.shape[1]

            # Check for DoubleSort (average across conditioning groups)
            if second_signal is not None and num_portfolios is not None:
                # DoubleSort: average across conditioning groups
                n1 = num_portfolios
                n2 = nport_count // n1
                short_cols = list(range(n2))
                long_cols = list(range((n1 - 1) * n2, n1 * n2))

                long_count = count_df.iloc[:, long_cols].mean(axis=1)
                short_count = count_df.iloc[:, short_cols].mean(axis=1)
            else:
                # SingleSort or WithinFirmSort
                long_count = count_df.iloc[:, nport_count - 1]
                short_count = count_df.iloc[:, 0]

            # Factor count = sum of long and short (total bonds in factor)
            factor_count = long_count + short_count

        # Get characteristics if available
        ew_chars_data = {}  # {char_name: {'long': Series, 'short': Series, 'ls': Series}}
        vw_chars_data = {}

        if has_chars:
            ew_chars_dict, vw_chars_dict = sr.get_characteristics()

            for char_name in char_names:
                if char_name in ew_chars_dict:
                    ew_char_df = ew_chars_dict[char_name]
                    vw_char_df = vw_chars_dict[char_name]
                    nport_char = ew_char_df.shape[1]

                    # Check for DoubleSort
                    if second_signal is not None and num_portfolios is not None:
                        n1 = num_portfolios
                        n2 = nport_char // n1
                        short_cols = list(range(n2))
                        long_cols = list(range((n1 - 1) * n2, n1 * n2))

                        ew_long_char = ew_char_df.iloc[:, long_cols].mean(axis=1)
                        ew_short_char = ew_char_df.iloc[:, short_cols].mean(axis=1)
                        vw_long_char = vw_char_df.iloc[:, long_cols].mean(axis=1)
                        vw_short_char = vw_char_df.iloc[:, short_cols].mean(axis=1)
                    else:
                        # SingleSort or WithinFirmSort
                        ew_long_char = ew_char_df.iloc[:, nport_char - 1]
                        ew_short_char = ew_char_df.iloc[:, 0]
                        vw_long_char = vw_char_df.iloc[:, nport_char - 1]
                        vw_short_char = vw_char_df.iloc[:, 0]

                    # L-S spread
                    ew_ls_char = ew_long_char - ew_short_char
                    vw_ls_char = vw_long_char - vw_short_char

                    ew_chars_data[char_name] = {
                        'long': ew_long_char,
                        'short': ew_short_char,
                        'ls': ew_ls_char,
                    }
                    vw_chars_data[char_name] = {
                        'long': vw_long_char,
                        'short': vw_short_char,
                        'ls': vw_ls_char,
                    }

        # Helper function for safe series access (handles missing dates)
        def safe_get(series, date):
            """Safely get value from series, returning NaN if date not found."""
            if series is None:
                return np.nan
            try:
                return series[date]
            except KeyError:
                return np.nan

        # Build rows for each date
        dates = ew_ls.index

        for date in dates:
            # === EW rows ===
            # Long-short leg
            row_ew_ls = {
                'date': date,
                'factor': ew_factor_name,
                'freq': holding_period,
                'leg': 'ls',
                'weighting': 'ew',
                'return': -ew_ls[date] if ew_flip else ew_ls[date],
            }
            if has_turnover:
                row_ew_ls['turnover'] = safe_get(ew_factor_turn, date)
            if has_counts:
                row_ew_ls['count'] = safe_get(factor_count, date)
            if has_chars:
                for char_name in char_names:
                    if char_name in ew_chars_data:
                        ls_char_val = safe_get(ew_chars_data[char_name]['ls'], date)
                        # Negate L-S spread when sign-corrected (L and S are swapped)
                        row_ew_ls[char_name] = -ls_char_val if ew_flip else ls_char_val
            rows.append(row_ew_ls)

            # Long leg (swap to 's' if flipped)
            row_ew_l = {
                'date': date,
                'factor': ew_factor_name,
                'freq': holding_period,
                'leg': 's' if ew_flip else 'l',
                'weighting': 'ew',
                'return': ew_long[date],
            }
            if has_turnover:
                row_ew_l['turnover'] = safe_get(ew_long_turn, date)
            if has_counts:
                row_ew_l['count'] = safe_get(long_count, date)
            if has_chars:
                for char_name in char_names:
                    if char_name in ew_chars_data:
                        row_ew_l[char_name] = safe_get(ew_chars_data[char_name]['long'], date)
            rows.append(row_ew_l)

            # Short leg (swap to 'l' if flipped)
            row_ew_s = {
                'date': date,
                'factor': ew_factor_name,
                'freq': holding_period,
                'leg': 'l' if ew_flip else 's',
                'weighting': 'ew',
                'return': ew_short[date],
            }
            if has_turnover:
                row_ew_s['turnover'] = safe_get(ew_short_turn, date)
            if has_counts:
                row_ew_s['count'] = safe_get(short_count, date)
            if has_chars:
                for char_name in char_names:
                    if char_name in ew_chars_data:
                        row_ew_s[char_name] = safe_get(ew_chars_data[char_name]['short'], date)
            rows.append(row_ew_s)

            # === VW rows ===
            # Long-short leg
            row_vw_ls = {
                'date': date,
                'factor': vw_factor_name,
                'freq': holding_period,
                'leg': 'ls',
                'weighting': 'vw',
                'return': -vw_ls[date] if vw_flip else vw_ls[date],
            }
            if has_turnover:
                row_vw_ls['turnover'] = safe_get(vw_factor_turn, date)
            if has_counts:
                row_vw_ls['count'] = safe_get(factor_count, date)
            if has_chars:
                for char_name in char_names:
                    if char_name in vw_chars_data:
                        ls_char_val = safe_get(vw_chars_data[char_name]['ls'], date)
                        # Negate L-S spread when sign-corrected (L and S are swapped)
                        row_vw_ls[char_name] = -ls_char_val if vw_flip else ls_char_val
            rows.append(row_vw_ls)

            # Long leg (swap to 's' if flipped)
            row_vw_l = {
                'date': date,
                'factor': vw_factor_name,
                'freq': holding_period,
                'leg': 's' if vw_flip else 'l',
                'weighting': 'vw',
                'return': vw_long[date],
            }
            if has_turnover:
                row_vw_l['turnover'] = safe_get(vw_long_turn, date)
            if has_counts:
                row_vw_l['count'] = safe_get(long_count, date)
            if has_chars:
                for char_name in char_names:
                    if char_name in vw_chars_data:
                        row_vw_l[char_name] = safe_get(vw_chars_data[char_name]['long'], date)
            rows.append(row_vw_l)

            # Short leg (swap to 'l' if flipped)
            row_vw_s = {
                'date': date,
                'factor': vw_factor_name,
                'freq': holding_period,
                'leg': 'l' if vw_flip else 's',
                'weighting': 'vw',
                'return': vw_short[date],
            }
            if has_turnover:
                row_vw_s['turnover'] = safe_get(vw_short_turn, date)
            if has_counts:
                row_vw_s['count'] = safe_get(short_count, date)
            if has_chars:
                for char_name in char_names:
                    if char_name in vw_chars_data:
                        row_vw_s[char_name] = safe_get(vw_chars_data[char_name]['short'], date)
            rows.append(row_vw_s)

    # Build DataFrame
    panel = pd.DataFrame(rows)

    # Ensure column order
    base_cols = ['date', 'factor', 'freq', 'leg', 'weighting', 'return']
    if has_turnover:
        base_cols.append('turnover')
    if has_counts:
        base_cols.append('count')
    if has_chars:
        base_cols.extend(char_names)

    # Reorder columns
    panel = panel[base_cols]

    return panel
