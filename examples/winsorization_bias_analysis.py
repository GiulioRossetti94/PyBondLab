"""
Winsorization Bias Analysis

Compares winsorized vs actual factor returns to quantify the bias
introduced by asymmetric return winsorization.

Outputs:
- Long leg (P_N): μ̃, μ, Bias, Bias_t, %Bias
- Short leg (P_1): μ̃, μ, Bias, Bias_t, %Bias
- Long-Short: μ̃, μ, Bias, Bias_t, %Bias, Sharpe

Where:
- μ̃ = Winsorized mean (ex-post with wins filter)
- μ = Actual mean (baseline, no filter)
- Bias = μ̃ - μ
- Bias_t = NW t-stat on the bias series
- %Bias = Bias / |μ| × 100
- Sharpe = Annualized Sharpe ratio (baseline)
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import warnings

from PyBondLab import DataUncertaintyAnalysis

# Try to import statsmodels for NW t-stats
try:
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not available, t-stats will use simple formula")


# =============================================================================
# Configuration
# =============================================================================

# Signal groups
LEFT_TAIL_SIGNALS = [
    'b_dunc', 'b_dunc3', 'b_unc',
    'ltr48_12', 'ltr30_6',
    'ivol_bbw', 'ivol_vp',
    'b_dvix_vp', 'b_psb_m', 'b_amd_m',
    'var_95', 'es_95'
]

RIGHT_TAIL_SIGNALS = [
    'mom3_1', 'mom6_1', 'mom12_1'
]

# Column mapping (adjust to your data)
COLUMN_MAPPING = {
    'ID': 'cusip',
    'VW': 'mcap_e',
    'RATING_NUM': 'spc_rat',
    'ret': 'ret_vw'
}

# Analysis settings
HOLDING_PERIOD = 1
NUM_PORTFOLIOS = 10


# =============================================================================
# Helper Functions
# =============================================================================

def compute_nw_tstat(series: pd.Series) -> float:
    """
    Compute Newey-West t-statistic for a series.

    Uses HAC standard errors with lag = int(T^0.25).
    Tests H0: mean = 0.
    """
    series = series.dropna()
    if len(series) < 10:
        return np.nan

    T = len(series)
    lag = int(T ** 0.25)
    mean = series.mean()

    if HAS_STATSMODELS:
        # Use statsmodels for proper NW standard errors
        y = series.values
        X = add_constant(np.ones(len(y)))
        try:
            model = OLS(y, X[:, 0]).fit(cov_type='HAC', cov_kwds={'maxlags': lag})
            tstat = model.tvalues[0]
            return tstat
        except:
            pass

    # Fallback: simple t-stat (not NW adjusted)
    se = series.std() / np.sqrt(T)
    if se > 0:
        return mean / se
    return np.nan


def compute_annualized_sharpe(series: pd.Series) -> float:
    """Compute annualized Sharpe ratio (assuming monthly data)."""
    series = series.dropna()
    if len(series) < 12:
        return np.nan

    mean = series.mean()
    std = series.std()

    if std > 0:
        return (mean / std) * np.sqrt(12)
    return np.nan


def run_analysis(
    data: pd.DataFrame,
    signals: list,
    wins_location: str,
    wins_level: float = 99.5,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run winsorization bias analysis for a set of signals.

    Parameters
    ----------
    data : pd.DataFrame
        Bond panel data
    signals : list
        Signal column names
    wins_location : str
        'left' or 'right' for tail to winsorize
    wins_level : float
        Winsorization percentile (e.g., 99.5 for 0.5% tail)
    verbose : bool
        Print progress

    Returns
    -------
    pd.DataFrame
        Analysis results with columns for each metric
    """
    # Filter to signals that exist in data
    available_signals = [s for s in signals if s in data.columns]
    missing_signals = [s for s in signals if s not in data.columns]

    if missing_signals:
        print(f"  Warning: Missing signals: {missing_signals}")

    if not available_signals:
        print("  Error: No signals available!")
        return pd.DataFrame()

    if verbose:
        print(f"  Running DataUncertaintyAnalysis for {len(available_signals)} signals...")
        print(f"  Winsorization: {wins_level}% {wins_location}-tail")

    # Run analysis with baseline + wins filter
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        results = DataUncertaintyAnalysis(
            data=data,
            signals=available_signals,
            holding_periods=[HOLDING_PERIOD],
            filters={'wins': [(wins_level, wins_location)]},
            include_baseline=True,
            num_portfolios=NUM_PORTFOLIOS,
            dynamic_weights=True,
            verbose=False,
        ).fit()

    if verbose:
        print(f"  Analysis complete. Extracting results...")

    # Build results table
    rows = []

    for signal in available_signals:
        # Filter to this signal
        sig_results = results.filter(signal=signal)

        # Get baseline and wins columns
        baseline_col = f"{signal}_hp{HOLDING_PERIOD}_baseline"
        wins_col = f"{signal}_hp{HOLDING_PERIOD}_wins_{wins_level}_{wins_location}"

        # Check columns exist
        if baseline_col not in sig_results.vw_ex_ante.columns:
            print(f"  Warning: Baseline column not found for {signal}")
            continue
        if wins_col not in sig_results.vw_ex_post.columns:
            print(f"  Warning: Wins column not found for {signal}")
            continue

        # Extract series for each leg
        # Baseline (actual) - use ex_ante (same as ex_post for baseline)
        vw_ls_base = sig_results.vw_ex_ante[baseline_col]
        vw_long_base = sig_results.vw_long_ex_ante[baseline_col]
        vw_short_base = sig_results.vw_short_ex_ante[baseline_col]

        # Wins (winsorized) - use ex_post
        vw_ls_wins = sig_results.vw_ex_post[wins_col]
        vw_long_wins = sig_results.vw_long_ex_post[wins_col]
        vw_short_wins = sig_results.vw_short_ex_post[wins_col]

        # Compute bias series
        bias_ls = vw_ls_wins - vw_ls_base
        bias_long = vw_long_wins - vw_long_base
        bias_short = vw_short_wins - vw_short_base

        # Compute statistics
        row = {'Factor': signal}

        # Long leg
        row['L_wins'] = vw_long_wins.mean() * 100  # Convert to %
        row['L_base'] = vw_long_base.mean() * 100
        row['L_bias'] = bias_long.mean() * 100
        row['L_bias_t'] = compute_nw_tstat(bias_long)
        row['L_pct_bias'] = (bias_long.mean() / abs(vw_long_base.mean()) * 100) if vw_long_base.mean() != 0 else np.nan

        # Short leg
        row['S_wins'] = vw_short_wins.mean() * 100
        row['S_base'] = vw_short_base.mean() * 100
        row['S_bias'] = bias_short.mean() * 100
        row['S_bias_t'] = compute_nw_tstat(bias_short)
        row['S_pct_bias'] = (bias_short.mean() / abs(vw_short_base.mean()) * 100) if vw_short_base.mean() != 0 else np.nan

        # Long-Short
        row['LS_wins'] = vw_ls_wins.mean() * 100
        row['LS_base'] = vw_ls_base.mean() * 100
        row['LS_bias'] = bias_ls.mean() * 100
        row['LS_bias_t'] = compute_nw_tstat(bias_ls)
        row['LS_pct_bias'] = (bias_ls.mean() / abs(vw_ls_base.mean()) * 100) if vw_ls_base.mean() != 0 else np.nan
        row['LS_sharpe'] = compute_annualized_sharpe(vw_ls_base)

        # T-stats for means (for reference)
        row['L_base_t'] = compute_nw_tstat(vw_long_base)
        row['S_base_t'] = compute_nw_tstat(vw_short_base)
        row['LS_base_t'] = compute_nw_tstat(vw_ls_base)

        rows.append(row)

    return pd.DataFrame(rows)


def format_table(df: pd.DataFrame, title: str) -> pd.DataFrame:
    """
    Format the results DataFrame for display.

    Returns a formatted DataFrame with proper column ordering and rounding.
    """
    if df.empty:
        return df

    # Column order for display
    display_cols = [
        'Factor',
        # Long
        'L_wins', 'L_base', 'L_bias', 'L_bias_t', 'L_pct_bias',
        # Short
        'S_wins', 'S_base', 'S_bias', 'S_bias_t', 'S_pct_bias',
        # Long-Short
        'LS_wins', 'LS_base', 'LS_bias', 'LS_bias_t', 'LS_pct_bias', 'LS_sharpe',
    ]

    # Select and order columns
    df_display = df[[c for c in display_cols if c in df.columns]].copy()

    # Round numeric columns
    for col in df_display.columns:
        if col == 'Factor':
            continue
        elif 'pct_bias' in col:
            df_display[col] = df_display[col].round(1)
        elif '_t' in col:
            df_display[col] = df_display[col].round(2)
        elif 'sharpe' in col:
            df_display[col] = df_display[col].round(2)
        else:
            df_display[col] = df_display[col].round(3)

    return df_display


def print_table(df: pd.DataFrame, title: str):
    """Print formatted table to console."""
    print()
    print("=" * 120)
    print(title)
    print("=" * 120)
    print()

    if df.empty:
        print("  No results to display.")
        return

    # Column renaming for display
    col_rename = {
        'L_wins': 'μ̃_L', 'L_base': 'μ_L', 'L_bias': 'Bias_L',
        'L_bias_t': 't(Bias)_L', 'L_pct_bias': '%Bias_L',
        'S_wins': 'μ̃_S', 'S_base': 'μ_S', 'S_bias': 'Bias_S',
        'S_bias_t': 't(Bias)_S', 'S_pct_bias': '%Bias_S',
        'LS_wins': 'μ̃_LS', 'LS_base': 'μ_LS', 'LS_bias': 'Bias_LS',
        'LS_bias_t': 't(Bias)_LS', 'LS_pct_bias': '%Bias_LS', 'LS_sharpe': 'Sharpe',
    }

    df_print = df.rename(columns=col_rename)

    # Print header info
    print("  Columns:")
    print("    μ̃ = Winsorized mean (%), μ = Baseline mean (%)")
    print("    Bias = μ̃ - μ (%), t(Bias) = NW t-stat on bias")
    print("    %Bias = Bias/|μ| × 100, Sharpe = Annualized Sharpe (baseline)")
    print()

    # Print sections
    print("  " + "-" * 50 + " Long " + "-" * 50)
    long_cols = ['Factor', 'μ̃_L', 'μ_L', 'Bias_L', 't(Bias)_L', '%Bias_L']
    long_cols = [c for c in long_cols if c in df_print.columns]
    print(df_print[long_cols].to_string(index=False))
    print()

    print("  " + "-" * 50 + " Short " + "-" * 49)
    short_cols = ['Factor', 'μ̃_S', 'μ_S', 'Bias_S', 't(Bias)_S', '%Bias_S']
    short_cols = [c for c in short_cols if c in df_print.columns]
    print(df_print[short_cols].to_string(index=False))
    print()

    print("  " + "-" * 48 + " Long-Short " + "-" * 47)
    ls_cols = ['Factor', 'μ̃_LS', 'μ_LS', 'Bias_LS', 't(Bias)_LS', '%Bias_LS', 'Sharpe']
    ls_cols = [c for c in ls_cols if c in df_print.columns]
    print(df_print[ls_cols].to_string(index=False))
    print()


def main(data: pd.DataFrame):
    """
    Main analysis function.

    Parameters
    ----------
    data : pd.DataFrame
        Bond panel data with required columns

    Returns
    -------
    tuple of pd.DataFrame
        (left_tail_results, right_tail_results)
    """
    print()
    print("=" * 120)
    print("WINSORIZATION BIAS ANALYSIS")
    print("=" * 120)
    print()
    print(f"Configuration:")
    print(f"  Holding Period: {HOLDING_PERIOD}")
    print(f"  Num Portfolios: {NUM_PORTFOLIOS}")
    print(f"  Weighting: VW only")
    print()

    # Apply column mapping
    data_mapped = data.copy()
    reverse_mapping = {v: k for k, v in COLUMN_MAPPING.items()}
    cols_to_rename = {col: reverse_mapping[col] for col in data.columns if col in reverse_mapping}
    if cols_to_rename:
        data_mapped = data_mapped.rename(columns=cols_to_rename)
        print(f"Column mapping applied: {cols_to_rename}")
        print()

    # Run left-tail analysis
    print("LEFT-TAIL WINSORIZATION (0.50%)")
    print("-" * 40)
    df_left = run_analysis(
        data_mapped,
        LEFT_TAIL_SIGNALS,
        wins_location='left',
        wins_level=99.5,  # 99.5% left = 0.5% left tail
        verbose=True
    )
    df_left_fmt = format_table(df_left, "Left-Tail Winsorization")
    print_table(df_left_fmt, "LEFT-TAIL ASYMMETRIC RETURN WINSORIZATION (0.50%)")

    # Run right-tail analysis
    print()
    print("RIGHT-TAIL WINSORIZATION (99.50%)")
    print("-" * 40)
    df_right = run_analysis(
        data_mapped,
        RIGHT_TAIL_SIGNALS,
        wins_location='right',
        wins_level=99.5,  # 99.5% right = 99.5% right tail
        verbose=True
    )
    df_right_fmt = format_table(df_right, "Right-Tail Winsorization")
    print_table(df_right_fmt, "RIGHT-TAIL ASYMMETRIC RETURN WINSORIZATION (99.50%)")

    return df_left_fmt, df_right_fmt


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    print("\nThis script requires your data to be loaded.")
    print("Usage:")
    print("  from examples.winsorization_bias_analysis import main")
    print("  df_left, df_right = main(your_data)")
    print()
    print("Or modify the script to load your data directly.")

    # Example with synthetic data for testing
    print("\n" + "=" * 60)
    print("RUNNING WITH SYNTHETIC DATA FOR TESTING")
    print("=" * 60)

    np.random.seed(42)
    n_dates, n_bonds = 120, 500
    dates = pd.date_range('2010-01-31', periods=n_dates, freq='ME')
    bond_ids = [f'BOND_{i:04d}' for i in range(n_bonds)]

    rows = []
    for d in dates:
        for b in bond_ids:
            rows.append({
                'date': d,
                'ID': b,
                'ret': np.random.normal(0.005, 0.05),
                'VW': np.random.uniform(100, 10000),
                'RATING_NUM': np.random.randint(1, 23),
                # Synthetic signals
                'b_dunc': np.random.normal(0, 1),
                'mom3_1': np.random.normal(0, 1),
                'mom6_1': np.random.normal(0, 1),
                'mom12_1': np.random.normal(0, 1),
            })

    test_data = pd.DataFrame(rows)

    # Override signal lists for test
    LEFT_TAIL_SIGNALS = ['b_dunc']
    RIGHT_TAIL_SIGNALS = ['mom3_1', 'mom6_1', 'mom12_1']

    df_left, df_right = main(test_data)

    print("\n" + "=" * 60)
    print("RAW DATAFRAMES (for inspection)")
    print("=" * 60)
    print("\nLeft-tail results:")
    print(df_left)
    print("\nRight-tail results:")
    print(df_right)
