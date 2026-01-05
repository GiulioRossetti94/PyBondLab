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

Alpha tables show the same structure but with regression alphas (intercepts from
regressing factor returns on mktb).
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import warnings
from pathlib import Path

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

# --- Paths ---
BASE_REPO = Path(r"C:\Users\ASUS\Documents\GitHub\trace-data-pipeline-private")
BASE_STAGE2 = BASE_REPO / "stage2"
STAGE0_DATE_STAMP = "20251126"

# Signal groups
LEFT_TAIL_SIGNALS = [
    'b_dunc', 'b_dunc3', 'b_unc',
    'ltr48_12', 'ltr30_6',
    'ivol_bbw', 'ivol_vp',
    'b_dvix_vp', 'b_psb_m', 'b_amd_m',
    'var_95', 'es_90'
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
# Factor Loading
# =============================================================================

def load_mktb_factor() -> pd.Series:
    """
    Load the market factor (mktb) from the BBW factors parquet file.

    Returns
    -------
    pd.Series
        mktb factor with date index
    """
    factor_path = BASE_STAGE2 / "data" / f"bbw_factors_{STAGE0_DATE_STAMP}.parquet"

    if not factor_path.exists():
        print(f"  Warning: Factor file not found: {factor_path}")
        return pd.Series(dtype=float)

    dff = pd.read_parquet(factor_path).reset_index()
    dff["date"] = pd.to_datetime(dff["date"])
    dff = dff.set_index("date").sort_index()
    mktb = dff['MKTB'].copy()
    mktb.name = 'mktb'

    return mktb


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


def compute_alpha(y: pd.Series, mktb: pd.Series) -> tuple:
    """
    Compute alpha (intercept) from regressing y on mktb.

    Uses Newey-West HAC standard errors.

    Parameters
    ----------
    y : pd.Series
        Dependent variable (factor returns)
    mktb : pd.Series
        Market factor

    Returns
    -------
    tuple
        (alpha, t_stat) - both as floats
    """
    if not HAS_STATSMODELS:
        return np.nan, np.nan

    # Align series by date
    df = pd.DataFrame({'y': y, 'mktb': mktb}).dropna()

    if len(df) < 10:
        return np.nan, np.nan

    T = len(df)
    lag = int(T ** 0.25)

    try:
        X = add_constant(df['mktb'].values)
        model = OLS(df['y'].values, X).fit(cov_type='HAC', cov_kwds={'maxlags': lag})
        alpha = model.params[0]
        t_stat = model.tvalues[0]
        return alpha, t_stat
    except Exception:
        return np.nan, np.nan


def run_analysis(
    data: pd.DataFrame,
    signals: list,
    wins_location: str,
    wins_level: float = 99.5,
    rating: str = None,
    mktb: pd.Series = None,
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
    rating : str, optional
        Rating filter: 'IG', 'NIG', or None for all bonds
    mktb : pd.Series, optional
        Market factor for alpha computation. If None, alpha tables not computed.
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Dictionary with:
        - 'means', 'tstats': DataFrames for return summary tables
        - 'alpha_means', 'alpha_tstats': DataFrames for alpha summary tables (if mktb provided)
        - 'ts_long_wins', 'ts_long_base': Time series DataFrames for long leg
        - 'ts_short_wins', 'ts_short_base': Time series DataFrames for short leg
        - 'ts_ls_wins', 'ts_ls_base': Time series DataFrames for long-short
        - 'ts_bias_long', 'ts_bias_short', 'ts_bias_ls': Time series DataFrames for bias
    """
    # Filter to signals that exist in data
    available_signals = [s for s in signals if s in data.columns]
    missing_signals = [s for s in signals if s not in data.columns]

    if missing_signals:
        print(f"  Warning: Missing signals: {missing_signals}")

    if not available_signals:
        print("  Error: No signals available!")
        return {'means': pd.DataFrame(), 'tstats': pd.DataFrame()}

    rating_str = rating if rating else "All"
    if verbose:
        print(f"  Running DataUncertaintyAnalysis for {len(available_signals)} signals...")
        print(f"  Winsorization: {wins_level}% {wins_location}-tail")
        print(f"  Rating: {rating_str}")

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
            rating=rating,
            verbose=False,
        ).fit()

    if verbose:
        print(f"  Analysis complete. Extracting results...")

    # Build results tables (means and t-stats separately)
    mean_rows = []
    tstat_rows = []
    alpha_mean_rows = []
    alpha_tstat_rows = []

    # Time series storage (using signal name as column - homogeneous naming)
    ts_long_wins = {}
    ts_long_base = {}
    ts_short_wins = {}
    ts_short_base = {}
    ts_ls_wins = {}
    ts_ls_base = {}
    ts_bias_long = {}
    ts_bias_short = {}
    ts_bias_ls = {}

    # Rating suffix for column names
    rating_suffix = f"_{rating}" if rating else ""

    for signal in available_signals:
        # Filter to this signal
        sig_results = results.filter(signal=signal)

        # Get baseline and wins columns (with rating suffix if applicable)
        baseline_col = f"{signal}_hp{HOLDING_PERIOD}_baseline{rating_suffix}"
        wins_col = f"{signal}_hp{HOLDING_PERIOD}_wins_{wins_level}_{wins_location}{rating_suffix}"

        # Check columns exist
        if baseline_col not in sig_results.vw_ex_ante.columns:
            print(f"  Warning: Baseline column not found for {signal}: {baseline_col}")
            print(f"  Available columns: {list(sig_results.vw_ex_ante.columns)[:5]}...")
            continue
        if wins_col not in sig_results.vw_ex_post.columns:
            print(f"  Warning: Wins column not found for {signal}: {wins_col}")
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

        # Store time series with homogeneous naming (just signal name)
        ts_long_wins[signal] = vw_long_wins
        ts_long_base[signal] = vw_long_base
        ts_short_wins[signal] = vw_short_wins
        ts_short_base[signal] = vw_short_base
        ts_ls_wins[signal] = vw_ls_wins
        ts_ls_base[signal] = vw_ls_base
        ts_bias_long[signal] = bias_long
        ts_bias_short[signal] = bias_short
        ts_bias_ls[signal] = bias_ls

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

        # Alpha computation (if mktb provided)
        if mktb is not None and len(mktb) > 0:
            # Compute alphas for each leg
            alpha_long_wins, t_long_wins = compute_alpha(vw_long_wins, mktb)
            alpha_long_base, t_long_base = compute_alpha(vw_long_base, mktb)
            alpha_short_wins, t_short_wins = compute_alpha(vw_short_wins, mktb)
            alpha_short_base, t_short_base = compute_alpha(vw_short_base, mktb)
            alpha_ls_wins, t_ls_wins = compute_alpha(vw_ls_wins, mktb)
            alpha_ls_base, t_ls_base = compute_alpha(vw_ls_base, mktb)

            # Alpha bias = alpha_wins - alpha_base
            alpha_bias_long = alpha_long_wins - alpha_long_base if not (np.isnan(alpha_long_wins) or np.isnan(alpha_long_base)) else np.nan
            alpha_bias_short = alpha_short_wins - alpha_short_base if not (np.isnan(alpha_short_wins) or np.isnan(alpha_short_base)) else np.nan
            alpha_bias_ls = alpha_ls_wins - alpha_ls_base if not (np.isnan(alpha_ls_wins) or np.isnan(alpha_ls_base)) else np.nan

            # For bias t-stat, compute from bias series regression
            _, t_bias_long = compute_alpha(bias_long, mktb)
            _, t_bias_short = compute_alpha(bias_short, mktb)
            _, t_bias_ls = compute_alpha(bias_ls, mktb)

            # Alpha means row
            alpha_mean_row = {
                'Factor': signal,
                'α̃_L': round(alpha_long_wins * 100, 2) if not np.isnan(alpha_long_wins) else np.nan,
                'α_L': round(alpha_long_base * 100, 2) if not np.isnan(alpha_long_base) else np.nan,
                'Bias_L': round(alpha_bias_long * 100, 2) if not np.isnan(alpha_bias_long) else np.nan,
                'α̃_S': round(alpha_short_wins * 100, 2) if not np.isnan(alpha_short_wins) else np.nan,
                'α_S': round(alpha_short_base * 100, 2) if not np.isnan(alpha_short_base) else np.nan,
                'Bias_S': round(alpha_bias_short * 100, 2) if not np.isnan(alpha_bias_short) else np.nan,
                'α̃_LS': round(alpha_ls_wins * 100, 2) if not np.isnan(alpha_ls_wins) else np.nan,
                'α_LS': round(alpha_ls_base * 100, 2) if not np.isnan(alpha_ls_base) else np.nan,
                'Bias_LS': round(alpha_bias_ls * 100, 2) if not np.isnan(alpha_bias_ls) else np.nan,
            }
            alpha_mean_rows.append(alpha_mean_row)

            # Alpha t-stats row
            alpha_tstat_row = {
                'Factor': '',
                'α̃_L': f"({t_long_wins:.2f})" if not np.isnan(t_long_wins) else "(nan)",
                'α_L': f"({t_long_base:.2f})" if not np.isnan(t_long_base) else "(nan)",
                'Bias_L': f"({t_bias_long:.2f})" if not np.isnan(t_bias_long) else "(nan)",
                'α̃_S': f"({t_short_wins:.2f})" if not np.isnan(t_short_wins) else "(nan)",
                'α_S': f"({t_short_base:.2f})" if not np.isnan(t_short_base) else "(nan)",
                'Bias_S': f"({t_bias_short:.2f})" if not np.isnan(t_bias_short) else "(nan)",
                'α̃_LS': f"({t_ls_wins:.2f})" if not np.isnan(t_ls_wins) else "(nan)",
                'α_LS': f"({t_ls_base:.2f})" if not np.isnan(t_ls_base) else "(nan)",
                'Bias_LS': f"({t_bias_ls:.2f})" if not np.isnan(t_bias_ls) else "(nan)",
            }
            alpha_tstat_rows.append(alpha_tstat_row)

    # Build result dict
    result = {
        'means': pd.DataFrame(mean_rows),
        'tstats': pd.DataFrame(tstat_rows),
        # Time series DataFrames (columns = factor names, index = dates)
        'ts_long_wins': pd.DataFrame(ts_long_wins),
        'ts_long_base': pd.DataFrame(ts_long_base),
        'ts_short_wins': pd.DataFrame(ts_short_wins),
        'ts_short_base': pd.DataFrame(ts_short_base),
        'ts_ls_wins': pd.DataFrame(ts_ls_wins),
        'ts_ls_base': pd.DataFrame(ts_ls_base),
        'ts_bias_long': pd.DataFrame(ts_bias_long),
        'ts_bias_short': pd.DataFrame(ts_bias_short),
        'ts_bias_ls': pd.DataFrame(ts_bias_ls),
    }

    if mktb is not None and len(alpha_mean_rows) > 0:
        result['alpha_means'] = pd.DataFrame(alpha_mean_rows)
        result['alpha_tstats'] = pd.DataFrame(alpha_tstat_rows)

    return result


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
    """Print formatted return table to console."""
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


def build_alpha_display_table(results: dict) -> pd.DataFrame:
    """
    Build a display table for alpha results with means and t-stats interleaved.
    """
    if 'alpha_means' not in results or results['alpha_means'].empty:
        return pd.DataFrame()

    means_df = results['alpha_means']
    tstats_df = results['alpha_tstats']

    # Interleave rows
    rows = []
    for i in range(len(means_df)):
        mean_row = means_df.iloc[i].to_dict()
        rows.append(mean_row)
        tstat_row = tstats_df.iloc[i].to_dict()
        rows.append(tstat_row)

    return pd.DataFrame(rows)


def print_alpha_table(results: dict, title: str):
    """Print formatted alpha table to console."""
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)
    print()

    if 'alpha_means' not in results or results['alpha_means'].empty:
        print("  No alpha results to display (mktb factor not available).")
        return

    print("  Legend: α̃ = Winsorized alpha (%), α = Baseline alpha (%), Bias = α̃ - α (%)")
    print("          Alpha = intercept from regressing factor returns on mktb")
    print("          Values in parentheses are NW t-statistics")
    print()

    # Column order for alpha
    cols = ['Factor', 'α̃_L', 'α_L', 'Bias_L', 'α̃_S', 'α_S', 'Bias_S', 'α̃_LS', 'α_LS', 'Bias_LS']

    # Build display table
    display_df = build_alpha_display_table(results)

    # Print header
    header = "  {:12s}  {:>8s} {:>8s} {:>8s}   {:>8s} {:>8s} {:>8s}   {:>8s} {:>8s} {:>8s}".format(
        '', '──Long──', '', '', '──Short─', '', '', '───L-S──', '', ''
    )
    print(header)

    col_header = "  {:12s}  {:>8s} {:>8s} {:>8s}   {:>8s} {:>8s} {:>8s}   {:>8s} {:>8s} {:>8s}".format(
        'Factor', 'α̃_L', 'α_L', 'Bias_L', 'α̃_S', 'α_S', 'Bias_S', 'α̃_LS', 'α_LS', 'Bias_LS'
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
                if pd.isna(val):
                    values.append("     nan")
                else:
                    values.append(f"{val:8.2f}")
            else:
                values.append(f"{val:>8s}")

        line = "  {:12s}  {:>8s} {:>8s} {:>8s}   {:>8s} {:>8s} {:>8s}   {:>8s} {:>8s} {:>8s}".format(
            factor, *values
        )
        print(line)

    print()


def main(data: pd.DataFrame, mktb: pd.Series = None):
    """
    Main analysis function.

    Parameters
    ----------
    data : pd.DataFrame
        Bond panel data with required columns
    mktb : pd.Series, optional
        Market factor for alpha computation. If None, will try to load from file.

    Returns
    -------
    dict
        Dictionary with:
        - Display tables: 'left_All', 'left_IG', 'left_NIG', 'right_All', etc.
        - Alpha tables: 'left_All_alpha', etc. (if mktb available)
        - Time series DataFrames for each (tail, rating):
            'ts_left_All': dict with keys 'long_wins', 'long_base', 'short_wins',
                           'short_base', 'ls_wins', 'ls_base', 'bias_long',
                           'bias_short', 'bias_ls'
            Each value is a DataFrame with columns = factor names, index = dates
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
    print(f"  Ratings: All, IG, NIG")
    print()

    # Load mktb factor if not provided
    if mktb is None:
        print("Loading mktb factor...")
        mktb = load_mktb_factor()
        if len(mktb) > 0:
            print(f"  Loaded mktb factor: {len(mktb)} observations ({mktb.index.min()} to {mktb.index.max()})")
        else:
            print("  Warning: mktb factor not available. Alpha tables will be skipped.")
        print()

    # Apply column mapping
    data_mapped = data.copy()
    reverse_mapping = {v: k for k, v in COLUMN_MAPPING.items()}
    cols_to_rename = {col: reverse_mapping[col] for col in data.columns if col in reverse_mapping}
    if cols_to_rename:
        data_mapped = data_mapped.rename(columns=cols_to_rename)
        print(f"Column mapping applied: {cols_to_rename}")
        print()

    # Rating categories to analyze
    ratings = [None, 'IG', 'NIG']  # None = All bonds
    rating_labels = {None: 'All', 'IG': 'IG', 'NIG': 'NIG'}

    all_results = {}

    # =========================================================================
    # RETURN TABLES
    # =========================================================================
    print()
    print("#" * 100)
    print("# PART 1: RETURN BIAS TABLES")
    print("#" * 100)

    # Run left-tail analysis for each rating
    for rating in ratings:
        rating_label = rating_labels[rating]
        print()
        print(f"LEFT-TAIL WINSORIZATION (0.50%) - {rating_label} Bonds")
        print("-" * 50)
        results_left = run_analysis(
            data_mapped,
            LEFT_TAIL_SIGNALS,
            wins_location='left',
            wins_level=99.5,  # 99.5% left = 0.5% left tail
            rating=rating,
            mktb=mktb,
            verbose=True
        )
        title = f"LEFT-TAIL ASYMMETRIC RETURN WINSORIZATION (0.50%) - {rating_label} Bonds"
        print_table(results_left, title)
        all_results[f'left_{rating_label}'] = build_display_table(results_left)
        # Store raw results for alpha tables
        all_results[f'left_{rating_label}_raw'] = results_left

    # Run right-tail analysis for each rating
    for rating in ratings:
        rating_label = rating_labels[rating]
        print()
        print(f"RIGHT-TAIL WINSORIZATION (99.50%) - {rating_label} Bonds")
        print("-" * 50)
        results_right = run_analysis(
            data_mapped,
            RIGHT_TAIL_SIGNALS,
            wins_location='right',
            wins_level=99.5,  # 99.5% right = 99.5% right tail
            rating=rating,
            mktb=mktb,
            verbose=True
        )
        title = f"RIGHT-TAIL ASYMMETRIC RETURN WINSORIZATION (99.50%) - {rating_label} Bonds"
        print_table(results_right, title)
        all_results[f'right_{rating_label}'] = build_display_table(results_right)
        # Store raw results for alpha tables
        all_results[f'right_{rating_label}_raw'] = results_right

    # =========================================================================
    # ALPHA TABLES
    # =========================================================================
    if mktb is not None and len(mktb) > 0:
        print()
        print("#" * 100)
        print("# PART 2: ALPHA BIAS TABLES")
        print("#" * 100)

        # Left-tail alpha tables
        for rating in ratings:
            rating_label = rating_labels[rating]
            results_left = all_results[f'left_{rating_label}_raw']
            title = f"LEFT-TAIL ALPHA BIAS (0.50%) - {rating_label} Bonds"
            print_alpha_table(results_left, title)
            all_results[f'left_{rating_label}_alpha'] = build_alpha_display_table(results_left)

        # Right-tail alpha tables
        for rating in ratings:
            rating_label = rating_labels[rating]
            results_right = all_results[f'right_{rating_label}_raw']
            title = f"RIGHT-TAIL ALPHA BIAS (99.50%) - {rating_label} Bonds"
            print_alpha_table(results_right, title)
            all_results[f'right_{rating_label}_alpha'] = build_alpha_display_table(results_right)

    # =========================================================================
    # EXTRACT TIME SERIES DATAFRAMES
    # =========================================================================
    # Collect time series from each (tail, rating) combination
    # Keys: 'ts_{tail}_{rating}' -> dict of DataFrames
    ts_keys = ['ts_long_wins', 'ts_long_base', 'ts_short_wins', 'ts_short_base',
               'ts_ls_wins', 'ts_ls_base', 'ts_bias_long', 'ts_bias_short', 'ts_bias_ls']

    for tail in ['left', 'right']:
        for rating in ratings:
            rating_label = rating_labels[rating]
            raw_key = f'{tail}_{rating_label}_raw'

            if raw_key in all_results:
                raw = all_results[raw_key]
                # Store each time series type under a descriptive key
                ts_dict = {}
                for ts_key in ts_keys:
                    if ts_key in raw and not raw[ts_key].empty:
                        # Rename key to shorter form: 'ts_long_wins' -> 'long_wins'
                        short_key = ts_key.replace('ts_', '')
                        ts_dict[short_key] = raw[ts_key]

                all_results[f'ts_{tail}_{rating_label}'] = ts_dict

    # Clean up raw results from output dict (keep ts_ keys)
    keys_to_remove = [k for k in all_results if k.endswith('_raw')]
    for k in keys_to_remove:
        del all_results[k]

    return all_results


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    print("\nThis script requires your data to be loaded.")
    print("Usage:")
    print("  from examples.winsorization_bias_analysis import main")
    print("  all_results = main(your_data)")
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

    # Create synthetic mktb factor for testing
    mktb_test = pd.Series(
        np.random.normal(0.005, 0.03, n_dates),
        index=dates,
        name='mktb'
    )

    # Override signal lists for test
    LEFT_TAIL_SIGNALS = ['b_dunc']
    RIGHT_TAIL_SIGNALS = ['mom3_1', 'mom6_1', 'mom12_1']

    all_results = main(test_data, mktb=mktb_test)

    print("\n" + "=" * 60)
    print("DISPLAY TABLES (for inspection)")
    print("=" * 60)
    for key, val in all_results.items():
        if key.startswith('ts_'):
            continue  # Skip time series for this section
        if isinstance(val, pd.DataFrame) and not val.empty:
            print(f"\n{key}:")
            print(val.to_string(index=False))

    print("\n" + "=" * 60)
    print("TIME SERIES DATAFRAMES (for inspection)")
    print("=" * 60)
    for key, val in all_results.items():
        if key.startswith('ts_'):
            print(f"\n{key}:")
            if isinstance(val, dict):
                for ts_name, ts_df in val.items():
                    if isinstance(ts_df, pd.DataFrame) and not ts_df.empty:
                        print(f"  {ts_name}: shape={ts_df.shape}, columns={list(ts_df.columns)}")
                        print(f"    Date range: {ts_df.index.min()} to {ts_df.index.max()}")
                        print(f"    Sample (first 3 rows):")
                        print(ts_df.head(3).to_string())
                        print()

    print("\n" + "=" * 60)
    print("USAGE EXAMPLE: Computing bias from time series")
    print("=" * 60)
    print("""
    # Access time series for left-tail, All bonds
    ts = all_results['ts_left_All']

    # Get long-short returns
    ls_wins = ts['ls_wins']      # Winsorized L-S returns (DataFrame)
    ls_base = ts['ls_base']      # Baseline L-S returns (DataFrame)

    # Bias = winsorized - baseline
    ls_bias = ls_wins - ls_base  # Same as ts['bias_ls']

    # Or access individual factor
    b_dunc_wins = ts['ls_wins']['b_dunc']
    b_dunc_base = ts['ls_base']['b_dunc']
    b_dunc_bias = b_dunc_wins - b_dunc_base
    """)
