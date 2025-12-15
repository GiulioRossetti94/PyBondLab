"""
Data Loading Utility for Real Data Examples

Provides a standard interface for loading bond data from CSV files.
Handles data formatting, column renaming, and filtering.

Author: Giulio Rossetti
"""

import pandas as pd
import numpy as np


def load_bond_data(data_path=None, start_date="2002-08-31", verbose=True):
    """
    Load and format bond data for PyBondLab.

    Parameters
    ----------
    data_path : str, optional
        Path to CSV file. If None, uses default OSBAP data path.
    start_date : str, optional
        Filter data to this start date and later
    verbose : bool, optional
        Print loading information

    Returns
    -------
    pd.DataFrame
        Formatted bond data ready for StrategyFormation
    """

    # Default data path (OSBAP or similar)
    # TODO: Update this default path to point to your local OSBAP data file
    # Download from: https://openbondassetpricing.com/
    if data_path is None:
        data_path = "path/to/your/WRDS_MMN_Corrected_Data_2024_July.csv"

    if verbose:
        print(f"Loading data from: {data_path}")

    try:
        # Load data
        df = pd.read_csv(data_path)

        if verbose:
            print(f"  Loaded {len(df):,} observations")

        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])

        # Filter to start date
        if start_date is not None:
            df = df[df['date'] >= start_date]
            if verbose:
                print(f"  Filtered to {start_date} onwards: {len(df):,} observations")

        # Create value weight if needed
        if 'VW' not in df.columns:
            df['VW'] = df.get('BOND_VALUE', df.get('bond_size', 1.0))

        # Rename columns to PyBondLab conventions
        rename_map = {
            "cusip": "ID",
            "CUSIP": "ID",
            "BONDPRC": "PRICE",
            "bond_ret": "ret",
            "rating": "RATING_NUM"
        }
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

        # Ensure ID column exists
        if 'ID' not in df.columns:
            raise ValueError("No bond identifier column found (cusip, CUSIP, or ID)")

        # Sort by ID and date (REQUIRED)
        df = df.sort_values(['ID', 'date'])

        if verbose:
            print(f"  Unique bonds: {df['ID'].nunique():,}")
            print(f"  Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
            print(f"  Data shape: {df.shape}")

        return df

    except FileNotFoundError:
        print(f"\nERROR: Data file not found at: {data_path}")
        print("\nPlease update the data_path parameter or set the default path in data_loader.py")
        print("\nData sources:")
        print("  - OSBAP: https://openbondassetpricing.com/")
        print("  - WRDS Enhanced TRACE (institutional access)")
        raise


def get_sample_period(df, start_year=None, end_year=None):
    """
    Filter data to specific sample period.

    Parameters
    ----------
    df : pd.DataFrame
        Bond data
    start_year : int, optional
        Start year (inclusive)
    end_year : int, optional
        End year (inclusive)

    Returns
    -------
    pd.DataFrame
        Filtered data
    """
    result = df.copy()

    if start_year is not None:
        result = result[result['date'].dt.year >= start_year]

    if end_year is not None:
        result = result[result['date'].dt.year <= end_year]

    return result


def apply_data_filters(df,
                       rating_range=None,
                       min_price=None,
                       max_price=None,
                       require_columns=None):
    """
    Apply common data filters.

    Parameters
    ----------
    df : pd.DataFrame
        Bond data
    rating_range : tuple, optional
        (min_rating, max_rating) for numeric ratings
    min_price : float, optional
        Minimum bond price
    max_price : float, optional
        Maximum bond price
    require_columns : list, optional
        List of required column names

    Returns
    -------
    pd.DataFrame
        Filtered data
    """
    result = df.copy()
    n_start = len(result)

    # Rating filter
    if rating_range is not None and 'RATING_NUM' in result.columns:
        min_rat, max_rat = rating_range
        result = result[(result['RATING_NUM'] >= min_rat) &
                       (result['RATING_NUM'] <= max_rat)]
        print(f"  Rating filter ({min_rat}-{max_rat}): {len(result):,} obs ({len(result)/n_start*100:.1f}%)")

    # Price filters
    if min_price is not None and 'PRICE' in result.columns:
        result = result[result['PRICE'] >= min_price]
        print(f"  Min price filter (>=${min_price}): {len(result):,} obs")

    if max_price is not None and 'PRICE' in result.columns:
        result = result[result['PRICE'] <= max_price]
        print(f"  Max price filter (<=${max_price}): {len(result):,} obs")

    # Required columns
    if require_columns is not None:
        missing = [col for col in require_columns if col not in result.columns]
        if missing:
            print(f"  WARNING: Missing required columns: {missing}")

    return result
