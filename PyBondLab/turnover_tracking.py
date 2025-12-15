"""
Turnover Case Tracking System

Tracks and classifies turnover calculations to identify edge cases
(e.g., exactly zero turnover, empty portfolios, first observations).

Authors: Giulio Rossetti & Alex Dickerson
"""

import numpy as np
import pandas as pd
from enum import Enum, auto
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, List
import warnings

# =============================================================================
# Case Classification Enum
# =============================================================================

class TurnoverCase(Enum):
    """Classification of turnover calculation scenarios."""

    # ===== NORMAL CASES =====
    NORMAL_REBALANCING = "Normal rebalancing activity"
    HEAVY_REBALANCING = "Heavy turnover (> 100%)"
    MINIMAL_REBALANCING = "Very low turnover (< 1%)"

    # ===== ZERO/NEAR-ZERO CASES =====
    ZERO_NO_TRADING = "Zero turnover: portfolio held, no trades, all returns equal"
    ZERO_EMPTY_PORTFOLIO = "Zero turnover: portfolio empty at both periods"
    NEAR_ZERO_SUSPICIOUS = "Near-zero turnover: investigate (should have weight drift)"

    # ===== EDGE CASES =====
    FULL_LIQUIDATION = "Portfolio fully liquidated (100% turnover)"
    NEW_PORTFOLIO = "Portfolio newly formed (100% turnover)"
    FIRST_OBSERVATION = "First observation: no previous period"
    EMPTY_T = "Empty portfolio at current period"
    EMPTY_T_MINUS_1 = "Empty portfolio at previous period"
    BOTH_EMPTY = "Empty portfolio at both periods"

    # ===== DATA ISSUES =====
    MISSING_DATA = "Missing data prevented calculation"
    INVALID_WEIGHTS = "Invalid weight values encountered"
    CALCULATION_ERROR = "Error during turnover calculation"


# =============================================================================
# Diagnostic Data Structure
# =============================================================================

@dataclass
class TurnoverDiagnostic:
    """Diagnostic information for a turnover calculation."""

    # Identifiers
    date: str
    portfolio_id: str
    cohort_id: Optional[int] = None

    # Portfolio state
    n_bonds_t: int = 0
    n_bonds_t_minus_1: int = 0
    portfolio_empty_t: bool = False
    portfolio_empty_t_minus_1: bool = False

    # Weight information
    sum_weights_t: float = 0.0
    sum_weights_t_minus_1: float = 0.0
    max_weight_change: float = 0.0
    mean_weight_change: float = 0.0

    # Trading activity
    n_bonds_added: int = 0
    n_bonds_removed: int = 0
    n_bonds_held: int = 0

    # Returns impact
    portfolio_return: Optional[float] = None
    return_std: Optional[float] = None
    max_return: Optional[float] = None
    min_return: Optional[float] = None

    # Turnover calculation
    turnover_value: float = np.nan
    turnover_case: str = "UNKNOWN"

    # Flags for special cases
    zero_turnover_flag: bool = False
    exact_zero: bool = False
    all_returns_equal: bool = False
    missing_data_flag: bool = False

    # Diagnostic metrics
    weight_drift_expected: Optional[float] = None
    weight_drift_actual: Optional[float] = None
    unexplained_drift: Optional[float] = None

    # Additional info
    notes: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV export."""
        return asdict(self)


# =============================================================================
# Classification Logic
# =============================================================================

def classify_turnover_case(
    turnover: float,
    n_bonds_t: int,
    n_bonds_t_minus_1: int,
    n_bonds_added: int = 0,
    n_bonds_removed: int = 0,
    returns: Optional[np.ndarray] = None,
    weights_t: Optional[np.ndarray] = None,
    weights_t_minus_1: Optional[np.ndarray] = None,
    zero_tolerance: float = 1e-10,
    near_zero_threshold: float = 1e-4
) -> Tuple[TurnoverCase, str]:
    """
    Classify the turnover case based on portfolio state and values.

    Parameters
    ----------
    turnover : float
        Calculated turnover value
    n_bonds_t : int
        Number of bonds at time t
    n_bonds_t_minus_1 : int
        Number of bonds at time t-1
    n_bonds_added : int
        Number of new positions
    n_bonds_removed : int
        Number of liquidated positions
    returns : np.ndarray, optional
        Bond returns (for analyzing weight drift)
    weights_t : np.ndarray, optional
        Current weights
    weights_t_minus_1 : np.ndarray, optional
        Previous weights
    zero_tolerance : float
        Threshold for considering exactly zero
    near_zero_threshold : float
        Threshold for near-zero warning

    Returns
    -------
    case : TurnoverCase
        Classification of this turnover calculation
    note : str
        Additional explanation
    """

    # Handle NaN/invalid cases first
    if np.isnan(turnover):
        if n_bonds_t == 0 and n_bonds_t_minus_1 == 0:
            return TurnoverCase.BOTH_EMPTY, "Portfolio doesn't exist at either period"
        return TurnoverCase.MISSING_DATA, "Turnover is NaN"

    if not np.isfinite(turnover):
        return TurnoverCase.CALCULATION_ERROR, f"Turnover is infinite: {turnover}"

    # Empty portfolio cases
    if n_bonds_t == 0 and n_bonds_t_minus_1 == 0:
        return TurnoverCase.BOTH_EMPTY, "No bonds in portfolio at either time"

    if n_bonds_t_minus_1 == 0:
        if turnover > 0.99:
            return TurnoverCase.NEW_PORTFOLIO, f"Portfolio newly formed with {n_bonds_t} bonds"
        return TurnoverCase.FIRST_OBSERVATION, "First observation (no previous period)"

    if n_bonds_t == 0:
        if turnover > 0.99:
            return TurnoverCase.FULL_LIQUIDATION, f"All {n_bonds_t_minus_1} bonds liquidated"
        return TurnoverCase.EMPTY_T, "Portfolio empty at current period"

    # Exactly zero turnover (suspicious - should be rare!)
    if abs(turnover) < zero_tolerance:
        # Check if all returns were equal (only way to get true zero with trading)
        if returns is not None and len(returns) > 1:
            return_range = returns.max() - returns.min()
            if return_range < zero_tolerance:
                return TurnoverCase.ZERO_NO_TRADING, \
                    f"All {len(returns)} bonds had identical returns ({returns.mean():.6f})"

        # If both periods empty, already handled above
        if n_bonds_t == 0 and n_bonds_t_minus_1 == 0:
            return TurnoverCase.ZERO_EMPTY_PORTFOLIO, "Empty portfolio at both periods"

        # Otherwise, true zero is suspicious
        return TurnoverCase.ZERO_NO_TRADING, \
            f"Exactly zero with {n_bonds_t} bonds (rare - should have weight drift!)"

    # Near-zero turnover (warning - weights should drift from price changes)
    if abs(turnover) < near_zero_threshold:
        note = f"Very low turnover ({turnover:.6f}) with {n_bonds_t} bonds"
        if returns is not None and len(returns) > 1:
            return_std = returns.std()
            if return_std > 0.001:  # Non-trivial return dispersion
                note += f" despite return std={return_std:.4f} - investigate!"
        return TurnoverCase.NEAR_ZERO_SUSPICIOUS, note

    # Low turnover (< 1%)
    if turnover < 0.01:
        return TurnoverCase.MINIMAL_REBALANCING, \
            f"Low turnover: {turnover:.4f}, {n_bonds_added} added, {n_bonds_removed} removed"

    # Full/heavy rebalancing
    if turnover > 1.0:
        return TurnoverCase.HEAVY_REBALANCING, \
            f"Heavy turnover: {turnover:.2f} (>{100:.0f}%)"

    # Normal rebalancing
    return TurnoverCase.NORMAL_REBALANCING, \
        f"Normal: {turnover:.4f}, {n_bonds_added} added, {n_bonds_removed} removed"


# =============================================================================
# Diagnostic Calculation
# =============================================================================

def compute_turnover_diagnostic(
    date: str,
    portfolio_id: str,
    ids_t: np.ndarray,
    weights_t: np.ndarray,
    ids_t_minus_1: np.ndarray,
    weights_t_minus_1: np.ndarray,
    turnover_value: float,
    returns: Optional[np.ndarray] = None,
    cohort_id: Optional[int] = None
) -> TurnoverDiagnostic:
    """
    Compute diagnostic information for a turnover calculation.

    Parameters
    ----------
    date : str
        Current date
    portfolio_id : str
        Portfolio identifier
    ids_t : np.ndarray
        Bond IDs at time t
    weights_t : np.ndarray
        Weights at time t
    ids_t_minus_1 : np.ndarray
        Bond IDs at time t-1
    weights_t_minus_1 : np.ndarray
        Weights at time t-1
    turnover_value : float
        Calculated turnover
    returns : np.ndarray, optional
        Bond returns between periods
    cohort_id : int, optional
        Cohort identifier (for staggered strategies)

    Returns
    -------
    TurnoverDiagnostic
        Complete diagnostic information
    """

    # Portfolio state
    n_bonds_t = len(ids_t)
    n_bonds_t_minus_1 = len(ids_t_minus_1)
    portfolio_empty_t = n_bonds_t == 0
    portfolio_empty_t_minus_1 = n_bonds_t_minus_1 == 0

    # Weight sums
    sum_weights_t = weights_t.sum() if n_bonds_t > 0 else 0.0
    sum_weights_t_minus_1 = weights_t_minus_1.sum() if n_bonds_t_minus_1 > 0 else 0.0

    # Trading activity
    ids_set_t = set(ids_t)
    ids_set_t_minus_1 = set(ids_t_minus_1)

    n_bonds_added = len(ids_set_t - ids_set_t_minus_1)
    n_bonds_removed = len(ids_set_t_minus_1 - ids_set_t)
    n_bonds_held = len(ids_set_t & ids_set_t_minus_1)

    # Weight changes (for continuing bonds)
    max_weight_change = 0.0
    mean_weight_change = 0.0

    if n_bonds_held > 0:
        # Create dictionaries for easy lookup
        weights_t_dict = dict(zip(ids_t, weights_t))
        weights_t_minus_1_dict = dict(zip(ids_t_minus_1, weights_t_minus_1))

        # Calculate changes for held bonds
        weight_changes = []
        for bond_id in (ids_set_t & ids_set_t_minus_1):
            w_curr = weights_t_dict[bond_id]
            w_prev = weights_t_minus_1_dict[bond_id]
            weight_changes.append(abs(w_curr - w_prev))

        max_weight_change = max(weight_changes)
        mean_weight_change = np.mean(weight_changes)

    # Returns statistics
    portfolio_return = None
    return_std = None
    max_return = None
    min_return = None
    all_returns_equal = False

    if returns is not None and len(returns) > 0:
        portfolio_return = np.mean(returns)
        return_std = np.std(returns)
        max_return = np.max(returns)
        min_return = np.min(returns)

        # Check if all returns are equal (rare!)
        if len(returns) > 1:
            all_returns_equal = (max_return - min_return) < 1e-10

    # Weight drift analysis
    weight_drift_expected = None
    weight_drift_actual = None
    unexplained_drift = None

    if returns is not None and n_bonds_held > 0:
        # Expected drift: how much weights should change from returns alone
        # Actual drift: how much weights actually changed
        # Unexplained: difference = trading activity

        # This is a simplified approximation
        # True calculation would need individual bond returns
        if return_std is not None:
            weight_drift_expected = return_std * np.sqrt(n_bonds_held)  # Rough estimate
            weight_drift_actual = mean_weight_change * n_bonds_held
            unexplained_drift = abs(weight_drift_actual - weight_drift_expected)

    # Classify the case
    case, note = classify_turnover_case(
        turnover=turnover_value,
        n_bonds_t=n_bonds_t,
        n_bonds_t_minus_1=n_bonds_t_minus_1,
        n_bonds_added=n_bonds_added,
        n_bonds_removed=n_bonds_removed,
        returns=returns,
        weights_t=weights_t,
        weights_t_minus_1=weights_t_minus_1
    )

    # Create diagnostic object
    diag = TurnoverDiagnostic(
        date=date,
        portfolio_id=portfolio_id,
        cohort_id=cohort_id,
        n_bonds_t=n_bonds_t,
        n_bonds_t_minus_1=n_bonds_t_minus_1,
        portfolio_empty_t=portfolio_empty_t,
        portfolio_empty_t_minus_1=portfolio_empty_t_minus_1,
        sum_weights_t=float(sum_weights_t),
        sum_weights_t_minus_1=float(sum_weights_t_minus_1),
        max_weight_change=float(max_weight_change),
        mean_weight_change=float(mean_weight_change),
        n_bonds_added=n_bonds_added,
        n_bonds_removed=n_bonds_removed,
        n_bonds_held=n_bonds_held,
        portfolio_return=float(portfolio_return) if portfolio_return is not None else None,
        return_std=float(return_std) if return_std is not None else None,
        max_return=float(max_return) if max_return is not None else None,
        min_return=float(min_return) if min_return is not None else None,
        turnover_value=float(turnover_value),
        turnover_case=case.name,
        zero_turnover_flag=abs(turnover_value) < 1e-10,
        exact_zero=abs(turnover_value) < 1e-15,
        all_returns_equal=all_returns_equal,
        missing_data_flag=np.isnan(turnover_value),
        weight_drift_expected=float(weight_drift_expected) if weight_drift_expected is not None else None,
        weight_drift_actual=float(weight_drift_actual) if weight_drift_actual is not None else None,
        unexplained_drift=float(unexplained_drift) if unexplained_drift is not None else None,
        notes=note
    )

    return diag


# =============================================================================
# Logging Utilities
# =============================================================================

class TurnoverLogger:
    """Manages logging of turnover diagnostics."""

    def __init__(self, output_dir: str = "turnover_logs"):
        """
        Initialize logger.

        Parameters
        ----------
        output_dir : str
            Directory to save log files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.diagnostics: List[TurnoverDiagnostic] = []

    def log(self, diagnostic: TurnoverDiagnostic):
        """Add a diagnostic entry to the log."""
        self.diagnostics.append(diagnostic)

    def save(self, prefix: str = "turnover"):
        """
        Save logs to CSV files.

        Parameters
        ----------
        prefix : str
            Prefix for output filenames
        """
        if not self.diagnostics:
            warnings.warn("No diagnostics to save")
            return

        # Convert to DataFrame
        df = pd.DataFrame([d.to_dict() for d in self.diagnostics])

        # Save detailed log
        detailed_file = self.output_dir / f"{prefix}_detailed.csv"
        df.to_csv(detailed_file, index=False)
        print(f"Saved detailed log: {detailed_file}")

        # Save alerts (suspicious cases only)
        alert_cases = [
            TurnoverCase.ZERO_NO_TRADING.name,
            TurnoverCase.NEAR_ZERO_SUSPICIOUS.name,
            TurnoverCase.CALCULATION_ERROR.name,
            TurnoverCase.INVALID_WEIGHTS.name
        ]
        df_alerts = df[df['turnover_case'].isin(alert_cases)]

        if len(df_alerts) > 0:
            alert_file = self.output_dir / f"{prefix}_alerts.csv"
            df_alerts.to_csv(alert_file, index=False)
            print(f"Saved alerts log: {alert_file} ({len(df_alerts)} alerts)")

        # Save summary
        self._save_summary(df, prefix)

    def _save_summary(self, df: pd.DataFrame, prefix: str):
        """Generate and save summary statistics."""
        summary = {
            'total_observations': len(df),
            'total_portfolios': df['portfolio_id'].nunique(),
            'total_dates': df['date'].nunique(),
        }

        # Case distribution
        case_counts = df['turnover_case'].value_counts()
        for case, count in case_counts.items():
            summary[f'count_{case}'] = count
            summary[f'pct_{case}'] = 100 * count / len(df)

        # Flag statistics
        summary['n_zero_turnover'] = df['zero_turnover_flag'].sum()
        summary['n_exact_zero'] = df['exact_zero'].sum()
        summary['n_all_returns_equal'] = df['all_returns_equal'].sum()
        summary['n_missing_data'] = df['missing_data_flag'].sum()

        # Turnover statistics
        valid_turnover = df[~df['missing_data_flag']]['turnover_value']
        summary['turnover_mean'] = valid_turnover.mean()
        summary['turnover_median'] = valid_turnover.median()
        summary['turnover_std'] = valid_turnover.std()
        summary['turnover_min'] = valid_turnover.min()
        summary['turnover_max'] = valid_turnover.max()

        # Save summary
        summary_df = pd.DataFrame([summary])
        summary_file = self.output_dir / f"{prefix}_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"Saved summary log: {summary_file}")

        # Print key findings
        print(f"\n{'='*80}")
        print(f"TURNOVER TRACKING SUMMARY")
        print(f"{'='*80}")
        print(f"Total observations: {summary['total_observations']}")
        print(f"Zero turnover cases: {summary['n_zero_turnover']}")
        print(f"Exact zero cases: {summary['n_exact_zero']}")
        print(f"Near-zero suspicious: {case_counts.get(TurnoverCase.NEAR_ZERO_SUSPICIOUS.name, 0)}")
        print(f"\nTurnover distribution:")
        print(f"  Mean: {summary['turnover_mean']:.6f}")
        print(f"  Median: {summary['turnover_median']:.6f}")
        print(f"  Min: {summary['turnover_min']:.6f}")
        print(f"  Max: {summary['turnover_max']:.6f}")


from pathlib import Path

# Make logger available for import
__all__ = [
    'TurnoverCase',
    'TurnoverDiagnostic',
    'classify_turnover_case',
    'compute_turnover_diagnostic',
    'TurnoverLogger'
]
