"""
Shared configuration for PyBondLab examples.

Data files are expected in the ``data/`` subdirectory:
  - bond_panel.parquet   (bond-level panel)
  - factors.parquet      (risk-free rate)
  - bbw_factors.parquet  (bond market factor MKTB)
"""

from pathlib import Path
import numpy as np
import pandas as pd

# -- Paths --------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
PANEL_PATH = DATA_DIR / "bond_panel.parquet"
FACTORS_PATH = DATA_DIR / "factors.parquet"
MKTB_PATH = DATA_DIR / "bbw_factors.parquet"

# -- TRACE -> PyBondLab standard column renaming ------------------------------
# PyBondLab standard names: date, ID, ret, VW, RATING_NUM
RENAME_MAP = {
    'cusip': 'ID',
    'ret_vw': 'ret',
    'mcap_e': 'VW',
    'spc_rat': 'RATING_NUM',
}

# Column kwargs for classes that accept explicit column mapping
# (AssayAnomaly, AssayAnomalyRunner use Wvar; BatchAssayAnomaly uses VWvar)
# AssayAnomaly/AssayAnomalyRunner use 'Wvar' for the weight column
ASSAY_COLUMNS = {
    'IDvar': 'ID',
    'DATEvar': 'date',
    'RETvar': 'ret',
    'Wvar': 'VW',
    'RATINGvar': 'RATING_NUM',
}
# BatchAssayAnomaly and DUA use 'VWvar' instead (different kwarg name)
BATCH_ASSAY_COLUMNS = {
    'IDvar': 'ID',
    'DATEvar': 'date',
    'RETvar': 'ret',
    'VWvar': 'VW',
    'RATINGvar': 'RATING_NUM',
}

# -- Sample period -----------------------------------------------------------
DATE_START = '1973-01-31'
DATE_END   = '2024-12-31'

# -- Representative signals (10 across signal groups) ------------------------
SIGNALS = [
    'mom6_1',       # momentum
    'str',          # short-term reversal
    'cs',           # credit spread
    'val_ipr',      # value (implied-price ratio)
    'ami',          # Amihud illiquidity
    'dvol',         # downside volatility
    'b_mktb',       # market beta
    'b_dvix',       # VIX beta
    'sze',          # bond size
    'bbtm',         # book-to-market
]


def load_panel(excess_returns=True, date_start=DATE_START, date_end=DATE_END):
    """Load the bond panel, rename to PyBondLab standard columns, convert to excess returns."""
    data = pd.read_parquet(PANEL_PATH)
    data['date'] = pd.to_datetime(data['date'])

    if excess_returns:
        factors = pd.read_parquet(FACTORS_PATH)
        factors['date'] = pd.to_datetime(factors['date'])
        rf = factors[['date', 'rf']].rename(columns={'rf': 'rfret'})
        if 'rfret' in data.columns:
            data.drop(columns=['rfret'], inplace=True)
        data = data.merge(rf, on='date', how='left')
        data['ret_vw'] = data['ret_vw'] - data['rfret']

    # Rename TRACE columns to PyBondLab standard names
    data = data.rename(columns=RENAME_MAP)

    # Convert categorical/nullable types to standard numpy types (avoids pandas/numba issues)
    if hasattr(data['ID'].dtype, 'categories'):
        data['ID'] = data['ID'].astype(str)
    for col in data.columns:
        if isinstance(data[col].dtype, pd.core.arrays.integer.IntegerDtype):
            data[col] = data[col].astype('float64')

    data = data[(data['date'] >= date_start) & (data['date'] <= date_end)]
    data = data.sort_values(['ID', 'date']).reset_index(drop=True)
    return data


def load_mktb():
    """Load the bond market factor (MKTB) for risk adjustment."""
    f = pd.read_parquet(MKTB_PATH)
    f.index = pd.to_datetime(f.index)
    s = f['MKTB']
    if hasattr(s.dtype, 'numpy_dtype'):
        s = s.astype('float64')
    return s
