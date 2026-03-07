"""
09 - Beta-Sorted Portfolios with Characteristics Tracking
==========================================================
Demonstrates how to combine RollingBeta with StrategyFormation to:

  1. Estimate rolling factor betas for individual bonds
  2. Sort bonds into quintile portfolios by estimated beta
  3. Track portfolio-level characteristics (betas, rating, maturity, etc.)
  4. Compute Newey-West t-statistics for long-short spreads

This is a general workflow pattern for any factor-beta-based strategy.
The example uses CPI inflation betas following Lu, Nozawa & Song (2025).

PyBondLab features used:
  - RollingBeta          (rolling OLS beta estimation)
  - StrategyFormation    (quintile portfolio formation)
  - SingleSort           (sort by estimated beta)
  - chars                (track characteristics within portfolios)
  - get_characteristics  (extract portfolio-level characteristics)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

import PyBondLab as pbl
from _config import load_panel

# ============================================================================
# Configuration
# ============================================================================

# Path to monthly CPI growth data
INF_DATA_DIR = "/Users/u1972481/Dropbox/1-research/Ra4-Andriollo-Rossetti-Ridge/Tail-Inflation/bond_inflation_project/data"
CPI_PATH = os.path.join(INF_DATA_DIR, "CUSR0000SA0_1m.csv")

BETA_WINDOW = 36      # Rolling window (months)
MIN_PERIODS = 24      # Minimum observations for beta estimation
N_PORTFOLIOS = 5      # Quintile sorts
NW_LAGS = 36          # Newey-West lags

SAMPLE_START = '2004-07-31'
SAMPLE_END = '2022-01-31'


def nw_tstat(y, lags=NW_LAGS):
    """Newey-West t-statistic for H0: mean(y) = 0."""
    y = pd.Series(y).dropna().astype(float)
    if len(y) <= lags + 1:
        return np.nan, np.nan
    X = np.ones((len(y), 1))
    res = sm.OLS(y.values, X, hasconst=False).fit(
        cov_type='HAC', cov_kwds={'maxlags': lags}
    )
    return float(res.params[0]), float(res.tvalues[0])


def stars(t):
    at = abs(t)
    if at >= 2.576:   return '***'
    elif at >= 1.960: return '**'
    elif at >= 1.645: return '*'
    return ''

# ============================================================================
# 1. Load bond panel
# ============================================================================

print("=" * 70)
print("BETA-SORTED PORTFOLIOS WITH CHARACTERISTICS TRACKING")
print("=" * 70)

data = load_panel(excess_returns=True, date_start=SAMPLE_START, date_end=SAMPLE_END)

# Return decomposition: credit = total excess - duration component
data['rx_dur'] = data['tret'] - data['rfret']
data['rx_credit'] = data['ret'] - data['rx_dur']

print(f"Panel: {len(data):,} obs, {data['ID'].nunique():,} bonds, "
      f"{data['date'].nunique()} months")

# ============================================================================
# 2. Prepare factor data (CPI inflation innovations)
# ============================================================================

print("\n-- Step 1: Prepare factor data --")

cpi_raw = pd.read_csv(CPI_PATH, index_col=0, parse_dates=True, dayfirst=True)
cpi_raw.index = (cpi_raw.index - pd.offsets.MonthBegin(1)).to_period('M').to_timestamp('M')
cpi = cpi_raw / 100  # percent to decimal

bond_dates = sorted(data['date'].unique())
cpi_panel = cpi[cpi.index.isin(bond_dates)].sort_index()

# ARMA(1,1) innovations
model = ARIMA(cpi_panel, order=(1, 0, 1)).fit()
eps = model.resid

# Build factors DataFrame: contemporaneous + lagged innovation
eps_df = eps.to_frame('eps')
eps_df['eps_lag'] = eps_df['eps'].shift(1)
eps_df = eps_df.dropna().reset_index()
eps_df.columns = ['date'] + list(eps_df.columns[1:])

print(f"  CPI innovations: {len(eps_df)} months")

# ============================================================================
# 3. Estimate rolling betas using RollingBeta
# ============================================================================

print("\n-- Step 2: Estimate rolling inflation betas --")

# RollingBeta estimates beta for each bond-month using a rolling window.
# With two factors (eps + eps_lag), inflation beta = sum of both slopes.

RETURN_TYPES = [
    ('ret',       'exret',     'Excess return'),
    ('rx_credit', 'rx_credit', 'Credit component'),
    ('rx_dur',    'rx_dur',    'Duration component'),
]

for ret_col, ret_label, desc in RETURN_TYPES:
    beta_col = f'inf_beta_{ret_label}'
    print(f"  {beta_col} ({desc}) ...", end=' ', flush=True)

    beta_est = pbl.RollingBeta(
        factors=eps_df,
        window=BETA_WINDOW,
        min_periods=MIN_PERIODS,
        add_constant=True,
        engine='auto',
        verbose=False,
    )

    data = beta_est.compute(
        data=data, date_col='date', id_col='ID', ret_cols=ret_col
    )

    # Sum contemporaneous + lagged betas
    b1 = f'eps_beta_{ret_col}'
    b2 = f'eps_lag_beta_{ret_col}'
    data[beta_col] = data[b1] + data[b2]

    n_valid = data[beta_col].notna().sum()
    print(f"{n_valid:,} obs, mean={data[beta_col].mean():.3f}")

    data.drop(columns=[b1, b2], inplace=True, errors='ignore')

# Clean up auxiliary columns
drop_cols = [c for c in data.columns if c.startswith('sigma_') or c.startswith('adj_r2_')]
data.drop(columns=drop_cols, inplace=True, errors='ignore')

# ============================================================================
# 4. Form beta-sorted portfolios with characteristic tracking
# ============================================================================

print("\n-- Step 3: Form quintile portfolios sorted by inflation beta --")

sort_col = 'inf_beta_exret'

# Track these characteristics within each quintile
char_cols = ['inf_beta_exret', 'inf_beta_rx_credit', 'inf_beta_rx_dur',
             'RATING_NUM', 'tmat']

sub = data.dropna(subset=char_cols).copy()
print(f"  Observations with valid data: {len(sub):,}")

strategy = pbl.SingleSort(
    sort_var=sort_col,
    holding_period=1,
    num_portfolios=N_PORTFOLIOS,
)

result = pbl.StrategyFormation(
    data=sub,
    strategy=strategy,
    rating=None,
    dynamic_weights=True,
    chars=char_cols,
).fit()

# ============================================================================
# 5. Extract and display results
# ============================================================================

print("\n-- Step 4: Extract portfolio characteristics --")

# get_characteristics() returns (ew_chars_dict, vw_chars_dict)
# Each dict maps char_name -> DataFrame with columns P1..P5
chars_ew, chars_vw = result.get_characteristics()

print(f"  Available characteristics: {list(chars_vw.keys())}")

# Display beta-sorted portfolio summary
col_labels = {
    'inf_beta_exret': 'beta(rx)',
    'inf_beta_rx_credit': 'beta(rx_cr)',
    'inf_beta_rx_dur': 'beta(rx_dur)',
    'RATING_NUM': 'Rating',
    'tmat': 'Maturity',
}

hdrs = list(col_labels.values())
print(f"\n  {'Ptf':<8}" + "".join(f"{h:>16}" for h in hdrs))
print("  " + "-" * (8 + 16 * len(hdrs)))

ts_store = {c: {} for c in char_cols}

for p in range(1, N_PORTFOLIOS + 1):
    vals = []
    for c in char_cols:
        vw_df = chars_vw[c]
        cn = [x for x in vw_df.columns if str(p) in x]
        if cn:
            ts = vw_df[cn[0]].dropna()
            ts_store[c][p] = ts
            vals.append(ts.mean())
        else:
            vals.append(np.nan)
    print(f"  {p:<8}" + "".join(f"{v:>16.3f}" for v in vals))

# High-Low spread with NW t-stats
hl_vals = []
for c in char_cols:
    if N_PORTFOLIOS in ts_store[c] and 1 in ts_store[c]:
        hl_ts = ts_store[c][N_PORTFOLIOS] - ts_store[c][1]
        m, t = nw_tstat(hl_ts)
        hl_vals.append((m, t, stars(t)))
    else:
        hl_vals.append((np.nan, np.nan, ''))

print(f"  {'H-L':<8}" + "".join(f"{f'{h[0]:.3f}{h[2]}':>16}" for h in hl_vals))
print(f"  {'':>8}" + "".join(f"{'(' + f'{h[1]:.3f}' + ')':>16}" for h in hl_vals))

# Also show VW portfolio returns
print("\n-- Portfolio VW excess returns --")
ew_ptf, vw_ptf = result.get_ptf()
print(f"  Monthly VW returns: {vw_ptf.shape[0]} months x {vw_ptf.shape[1]} portfolios")

for col in sorted(vw_ptf.columns, key=lambda c: int(''.join(filter(str.isdigit, c)) or 0)):
    m, t = nw_tstat(vw_ptf[col] * 100)
    print(f"    {col}: {m:.3f}%/mo  t={t:.2f}")

print("\n[Done] Script 09 complete.")
