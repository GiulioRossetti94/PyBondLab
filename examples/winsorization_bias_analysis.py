"""
Winsorization Bias Analysis

Compares winsorized vs actual factor returns to quantify the bias
introduced by asymmetric return winsorization.

Output format:
                 ──────── Long ────────   ──────── Short ────────   ────── Long-Short ──────
Factor           μ̃_L    μ_L   Bias_L      μ̃_S    μ_S   Bias_S      μ̃_LS   μ_LS   Bias_LS
b_dunc           0.58   0.51   0.07       0.50   0.45   0.05        0.08   0.06    0.02
                (3.20) (2.80) (1.50)     (2.10) (1.90) (0.80)      (1.50) (1.20)  (0.50)

Where:
- μ̃ = Winsorized mean (ex-post with wins filter)
- μ = Actual mean (baseline, no filter)
- Bias = μ̃ - μ
- Values in parentheses = NW t-statistics
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


def run_analysis(
    data: pd.DataFrame,
    signals: list,
    wins_location: str,
    wins_level: float = 99.5,
    verbose: bool = True
) -> dict:
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
    dict
        Dictionary with 'means' and 'tstats' DataFrames
    """
    # Filter to signals that exist in data
    available_signals = [s for s in signals if s in data.columns]
    missing_signals = [s for s in signals if s not in data.columns]

    if missing_signals:
        print(f"  Warning: Missing signals: {missing_signals}")

    if not available_signals:
        print("  Error: No signals available!")
        return {'means': pd.DataFrame(), 'tstats': pd.DataFrame()}

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

    # Build results tables (means and t-stats separately)
    mean_rows = []
    tstat_rows = []

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

        # Means row
        mean_row = {
            'Factor': signal,
            'μ̃_L': round(vw_long_wins.mean() * 100, 2),
            'μ_L': round(vw_long_base.mean() * 100, 2),
            'Bias_L': round(bias_long.mean() * 100, 2),
            'μ̃_S': round(vw_short_wins.mean() * 100, 2),
            'μ_S': round(vw_short_base.mean() * 100, 2),
            'Bias_S': round(bias_short.mean() * 100, 2),
            'μ̃_LS': round(vw_ls_wins.mean() * 100, 2),
            'μ_LS': round(vw_ls_base.mean() * 100, 2),
            'Bias_LS': round(bias_ls.mean() * 100, 2),
        }
        mean_rows.append(mean_row)

        # T-stats row (in parentheses format)
        tstat_row = {
            'Factor': '',  # Empty for alignment
            'μ̃_L': f"({compute_nw_tstat(vw_long_wins):.2f})",
            'μ_L': f"({compute_nw_tstat(vw_long_base):.2f})",
            'Bias_L': f"({compute_nw_tstat(bias_long):.2f})",
            'μ̃_S': f"({compute_nw_tstat(vw_short_wins):.2f})",
            'μ_S': f"({compute_nw_tstat(vw_short_base):.2f})",
            'Bias_S': f"({compute_nw_tstat(bias_short):.2f})",
            'μ̃_LS': f"({compute_nw_tstat(vw_ls_wins):.2f})",
            'μ_LS': f"({compute_nw_tstat(vw_ls_base):.2f})",
            'Bias_LS': f"({compute_nw_tstat(bias_ls):.2f})",
        }
        tstat_rows.append(tstat_row)

    return {
        'means': pd.DataFrame(mean_rows),
        'tstats': pd.DataFrame(tstat_rows)
    }


def build_display_table(results: dict) -> pd.DataFrame:
    """
    Build a display table with means and t-stats interleaved.

    Each factor has two rows:
    - Row 1: means (numeric)
    - Row 2: t-stats in parentheses (string)
    """
    if results['means'].empty:
        return pd.DataFrame()

    means_df = results['means']
    tstats_df = results['tstats']

    # Interleave rows
    rows = []
    for i in range(len(means_df)):
        # Add means row
        mean_row = means_df.iloc[i].to_dict()
        rows.append(mean_row)

        # Add t-stats row
        tstat_row = tstats_df.iloc[i].to_dict()
        rows.append(tstat_row)

    return pd.DataFrame(rows)


def print_table(results: dict, title: str):
    """Print formatted table to console."""
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)
    print()

    if results['means'].empty:
        print("  No results to display.")
        return

    print("  Legend: μ̃ = Winsorized mean (%), μ = Baseline mean (%), Bias = μ̃ - μ (%)")
    print("          Values in parentheses are NW t-statistics")
    print()

    # Column order
    cols = ['Factor', 'μ̃_L', 'μ_L', 'Bias_L', 'μ̃_S', 'μ_S', 'Bias_S', 'μ̃_LS', 'μ_LS', 'Bias_LS']

    # Build display table
    display_df = build_display_table(results)

    # Print header
    header = "  {:12s}  {:>8s} {:>8s} {:>8s}   {:>8s} {:>8s} {:>8s}   {:>8s} {:>8s} {:>8s}".format(
        '', '──Long──', '', '', '──Short─', '', '', '───L-S──', '', ''
    )
    print(header)

    col_header = "  {:12s}  {:>8s} {:>8s} {:>8s}   {:>8s} {:>8s} {:>8s}   {:>8s} {:>8s} {:>8s}".format(
        'Factor', 'μ̃_L', 'μ_L', 'Bias_L', 'μ̃_S', 'μ_S', 'Bias_S', 'μ̃_LS', 'μ_LS', 'Bias_LS'
    )
    print(col_header)
    print("  " + "-" * 97)

    # Print rows
    for i, row in display_df.iterrows():
        factor = row['Factor'] if row['Factor'] else ''
        values = []
        for col in cols[1:]:  # Skip Factor
            val = row[col]
            if isinstance(val, (int, float)):
                values.append(f"{val:8.2f}")
            else:
                values.append(f"{val:>8s}")

        line = "  {:12s}  {:>8s} {:>8s} {:>8s}   {:>8s} {:>8s} {:>8s}   {:>8s} {:>8s} {:>8s}".format(
            factor, *values
        )
        print(line)

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
        (left_tail_results, right_tail_results) - each is the display table
    """
    print()
    print("=" * 100)
    print("WINSORIZATION BIAS ANALYSIS")
    print("=" * 100)
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
    results_left = run_analysis(
        data_mapped,
        LEFT_TAIL_SIGNALS,
        wins_location='left',
        wins_level=99.5,  # 99.5% left = 0.5% left tail
        verbose=True
    )
    print_table(results_left, "LEFT-TAIL ASYMMETRIC RETURN WINSORIZATION (0.50%)")
    df_left = build_display_table(results_left)

    # Run right-tail analysis
    print()
    print("RIGHT-TAIL WINSORIZATION (99.50%)")
    print("-" * 40)
    results_right = run_analysis(
        data_mapped,
        RIGHT_TAIL_SIGNALS,
        wins_location='right',
        wins_level=99.5,  # 99.5% right = 99.5% right tail
        verbose=True
    )
    print_table(results_right, "RIGHT-TAIL ASYMMETRIC RETURN WINSORIZATION (99.50%)")
    df_right = build_display_table(results_right)

    return df_left, df_right


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
    print(df_left.to_string(index=False))
    print("\nRight-tail results:")
    print(df_right.to_string(index=False))
