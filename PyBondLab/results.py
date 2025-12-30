# -*- coding: utf-8 -*-
"""
Result container classes for PyBondLab.

This module provides structured containers for portfolio formation results.

Authors: Giulio Rossetti & Alex Dickerson
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
import pickle

from PyBondLab.naming import (
    NamingConfig,
    make_factor_name,
    make_portfolio_name,
    rating_to_suffix,
)


__all__ = [
    "PortfolioReturns",
    "TurnoverResults",
    "CharacteristicsResults",
    "StrategyResults",
    "FormationResults",
    "NamingConfig",
]


# =============================================================================
# Portfolio Returns Results
# =============================================================================
@dataclass
class PortfolioReturns:
    """
    Container for portfolio return arrays and DataFrames.

    Attributes
    ----------
    ewport_df : pd.DataFrame
        Equal-weighted portfolio returns (time × portfolios)
    vwport_df : pd.DataFrame
        Value-weighted portfolio returns (time × portfolios)
    ewls_df : pd.DataFrame, optional
        Equal-weighted long-short returns
    vwls_df : pd.DataFrame, optional
        Value-weighted long-short returns
    ewls_long_df : pd.DataFrame, optional
        Equal-weighted long portfolio
    vwls_long_df : pd.DataFrame, optional
        Value-weighted long portfolio
    ewls_short_df : pd.DataFrame, optional
        Equal-weighted short portfolio
    vwls_short_df : pd.DataFrame, optional
        Value-weighted short portfolio
    """

    # Required DataFrames
    ewport_df: pd.DataFrame
    vwport_df: pd.DataFrame

    # Optional long/short DataFrames
    ewls_df: Optional[pd.DataFrame] = None
    vwls_df: Optional[pd.DataFrame] = None
    ewls_long_df: Optional[pd.DataFrame] = None
    vwls_long_df: Optional[pd.DataFrame] = None
    ewls_short_df: Optional[pd.DataFrame] = None
    vwls_short_df: Optional[pd.DataFrame] = None

    def summary_stats(self, annualize: bool = True, periods_per_year: int = 12) -> pd.DataFrame:
        """
        Compute summary statistics for portfolio returns.

        Parameters
        ----------
        annualize : bool, default True
            If True, multiply means by periods_per_year and std by sqrt(periods_per_year).
        periods_per_year : int, default 12
            Frequency scaler (12 for monthly, 52 for weekly, 252 for daily).

        Returns
        -------
        pd.DataFrame
            Summary statistics with columns: Mean (EW), Mean (VW), Std (EW), Std (VW), 
            Sharpe (EW), Sharpe (VW)
        """
        factor = periods_per_year if annualize else 1
        sqrt_factor = np.sqrt(periods_per_year) if annualize else 1.0

        ew = self.ewport_df.values
        vw = self.vwport_df.values

        # Compute statistics
        ew_std = ew.std(axis=0, ddof=1)
        vw_std = vw.std(axis=0, ddof=1)
        ew_mean = ew.mean(axis=0)
        vw_mean = vw.mean(axis=0)

        # Guard against division by zero in Sharpe
        with np.errstate(divide="ignore", invalid="ignore"):
            sharpe_ew = np.where(ew_std == 0, np.nan, (ew_mean / ew_std) * sqrt_factor)
            sharpe_vw = np.where(vw_std == 0, np.nan, (vw_mean / vw_std) * sqrt_factor)

        stats = pd.DataFrame({
            "Mean (EW)": ew_mean * factor,
            "Mean (VW)": vw_mean * factor,
            "Std (EW)": ew_std * sqrt_factor,
            "Std (VW)": vw_std * sqrt_factor,
            "Sharpe (EW)": sharpe_ew,
            "Sharpe (VW)": sharpe_vw,
        }, index=self.ewport_df.columns)

        return stats

    def get_long_short(self, weight_type: str = "ew") -> pd.Series:
        """
        Get long-short returns as a Series.

        Parameters
        ----------
        weight_type : {'ew', 'vw'}
            Weight type

        Returns
        -------
        pd.Series
            Long-short returns

        Raises
        ------
        ValueError
            If long-short data not available
        """
        if weight_type == "ew":
            if self.ewls_df is None:
                raise ValueError("EW long-short DataFrame (ewls_df) not available.")
            return self.ewls_df.iloc[:, 0]
        elif weight_type == "vw":
            if self.vwls_df is None:
                raise ValueError("VW long-short DataFrame (vwls_df) not available.")
            return self.vwls_df.iloc[:, 0]
        else:
            raise ValueError("weight_type must be 'ew' or 'vw'.")
        
    def get_ptf(self, weight_type: str = "ew") -> pd.DataFrame:
        """
        Get portfolios returns as a Dataframe
        
        """
        if weight_type == "ew":
            if self.ewport_df is None:
                raise ValueError("Portfolios DataFrame not available.")
            return self.ewport_df
        elif weight_type == "vw":
            if self.vwport_df is None:
                raise ValueError("VW long DataFrame (vwls_long_df) not available.")
            return self.vwport_df
        else:
            raise ValueError("weight_type must be 'ew' or 'vw'.")



    def get_long_leg(self, weight_type: str = "ew") -> pd.Series:
        """
        Get long portfolio returns as a Series.

        Parameters
        ----------
        weight_type : {'ew', 'vw'}
            Weight type

        Returns
        -------
        pd.Series
            Long portfolio returns

        Raises
        ------
        ValueError
            If long portfolio data not available
        """
        if weight_type == "ew":
            if self.ewls_long_df is None:
                raise ValueError("EW long DataFrame (ewls_long_df) not available.")
            return self.ewls_long_df.iloc[:, 0]
        elif weight_type == "vw":
            if self.vwls_long_df is None:
                raise ValueError("VW long DataFrame (vwls_long_df) not available.")
            return self.vwls_long_df.iloc[:, 0]
        else:
            raise ValueError("weight_type must be 'ew' or 'vw'.")

    def get_short_leg(self, weight_type: str = "ew") -> pd.Series:
        """
        Get short portfolio returns as a Series.

        Parameters
        ----------
        weight_type : {'ew', 'vw'}
            Weight type

        Returns
        -------
        pd.Series
            Short portfolio returns

        Raises
        ------
        ValueError
            If short portfolio data not available
        """
        if weight_type == "ew":
            if self.ewls_short_df is None:
                raise ValueError("EW short DataFrame (ewls_short_df) not available.")
            return self.ewls_short_df.iloc[:, 0]
        elif weight_type == "vw":
            if self.vwls_short_df is None:
                raise ValueError("VW short DataFrame (vwls_short_df) not available.")
            return self.vwls_short_df.iloc[:, 0]
        else:
            raise ValueError("weight_type must be 'ew' or 'vw'.")


# =============================================================================
# Turnover Results
# =============================================================================
@dataclass
class TurnoverResults:
    """
    Container for turnover statistics.

    Attributes
    ----------
    ew_turnover_df : pd.DataFrame
        Equal-weighted turnover by portfolio (time × portfolios)
    vw_turnover_df : pd.DataFrame
        Value-weighted turnover by portfolio (time × portfolios)
    """

    ew_turnover_df: pd.DataFrame
    vw_turnover_df: pd.DataFrame

    def summary_stats(self) -> pd.DataFrame:
        """
        Compute summary statistics for turnover.

        Returns
        -------
        pd.DataFrame
            Mean, median, and std of turnover by portfolio
        """
        return pd.DataFrame({
            "Mean (EW)": self.ew_turnover_df.mean(),
            "Median (EW)": self.ew_turnover_df.median(),
            "Std (EW)": self.ew_turnover_df.std(),
            "Mean (VW)": self.vw_turnover_df.mean(),
            "Median (VW)": self.vw_turnover_df.median(),
            "Std (VW)": self.vw_turnover_df.std(),
        })

    def plot_turnover(self, weight_type: str = "ew", figsize=(12, 6)):
        """
        Plot turnover over time.

        Parameters
        ----------
        weight_type : {'ew', 'vw'}
            Weight type
        figsize : tuple
            Figure size

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        df = self.ew_turnover_df if weight_type == "ew" else self.vw_turnover_df

        fig, ax = plt.subplots(figsize=figsize)
        df.plot(ax=ax, alpha=0.7)
        ax.set_title(f"{weight_type.upper()} Turnover Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Turnover")
        ax.legend(title="Portfolio", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        return fig


# =============================================================================
# Characteristics Results
# =============================================================================
@dataclass
class CharacteristicsResults:
    """
    Container for portfolio characteristics.

    Attributes
    ----------
    ew_chars : Dict[str, pd.DataFrame]
        Equal-weighted characteristics: {char_name: DataFrame}
    vw_chars : Dict[str, pd.DataFrame]
        Value-weighted characteristics: {char_name: DataFrame}
    """

    ew_chars: Dict[str, pd.DataFrame]
    vw_chars: Dict[str, pd.DataFrame]

    def get_characteristic(self, char_name: str, weight_type: str = "ew") -> pd.DataFrame:
        """
        Get a specific characteristic DataFrame.

        Parameters
        ----------
        char_name : str
            Characteristic name
        weight_type : {'ew', 'vw'}
            Weight type

        Returns
        -------
        pd.DataFrame
            Characteristic values over time by portfolio
        """
        chars_dict = self.ew_chars if weight_type == "ew" else self.vw_chars
        if char_name not in chars_dict:
            available = list(chars_dict.keys())
            raise ValueError(
                f"Characteristic '{char_name}' not found. Available: {available}"
            )
        return chars_dict[char_name]

    def summary_stats(self, char_name: str) -> pd.DataFrame:
        """
        Compute summary statistics for a characteristic across portfolios.

        Parameters
        ----------
        char_name : str
            Characteristic name

        Returns
        -------
        pd.DataFrame
            Summary statistics
        """
        ew_df = self.get_characteristic(char_name, "ew")
        vw_df = self.get_characteristic(char_name, "vw")

        return pd.DataFrame({
            "Mean (EW)": ew_df.mean(),
            "Median (EW)": ew_df.median(),
            "Std (EW)": ew_df.std(),
            "Mean (VW)": vw_df.mean(),
            "Median (VW)": vw_df.median(),
            "Std (VW)": vw_df.std(),
        })

    @property
    def available_characteristics(self) -> list[str]:
        """List available characteristic names (EW set)."""
        return list(self.ew_chars.keys())

    def get_all_characteristics(self, weight_type: str = "ew") -> Dict[str, pd.DataFrame]:
        """
        Get all characteristics as a dictionary.

        Parameters
        ----------
        weight_type : {'ew', 'vw'}
            Weight type

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary mapping characteristic names to DataFrames
        """
        return self.ew_chars if weight_type == "ew" else self.vw_chars


# =============================================================================
# Strategy Results
# =============================================================================
@dataclass
class StrategyResults:
    """
    Complete results for a strategy (EA or EP).

    Attributes
    ----------
    returns : PortfolioReturns
        Portfolio returns
    turnover : TurnoverResults, optional
        Turnover statistics
    characteristics : CharacteristicsResults, optional
        Portfolio characteristics
    port_idx : dict, optional
        Portfolio membership indices: {date: DataFrame}
    signal_name : str, optional
        Name of the signal used for sorting (for naming)
    rating_str : str, optional
        Rating suffix ('ig', 'hy', or None)
    is_within_firm : bool
        Whether this is a WithinFirmSort strategy
    second_signal : str, optional
        Second signal name for DoubleSort
    num_portfolios : int, optional
        Number of portfolios
    """

    returns: PortfolioReturns
    turnover: Optional[TurnoverResults] = None
    characteristics: Optional[CharacteristicsResults] = None
    port_idx: Optional[Dict[pd.Timestamp, pd.DataFrame]] = None
    # Metadata for naming
    signal_name: Optional[str] = None
    rating_str: Optional[str] = None
    is_within_firm: bool = False
    second_signal: Optional[str] = None
    num_portfolios: Optional[int] = None

    @property
    def has_turnover(self) -> bool:
        """Check if turnover results are available."""
        return self.turnover is not None

    @property
    def has_characteristics(self) -> bool:
        """Check if characteristics are available."""
        return self.characteristics is not None

    @property
    def has_port_idx(self) -> bool:
        """Check if portfolio indices are saved."""
        return self.port_idx is not None

    def get_ptf(self)-> tuple[pd.DataFrame, pd.DataFrame]:
        """
        get ptfs
        """
        ew = self.returns.get_ptf("ew")
        vw = self.returns.get_ptf("vw")
        return ew, vw

    def get_long_short(
        self,
        naming: Optional[NamingConfig] = None,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Get long-short returns as (EW, VW) tuple.

        Parameters
        ----------
        naming : NamingConfig, optional
            If provided, rename output series using naming conventions.
            If None, use legacy names (backward compatible).

        Returns
        -------
        tuple of pd.Series
            (ew_long_short, vw_long_short)
        """
        ew = self.returns.get_long_short("ew").copy()
        vw = self.returns.get_long_short("vw").copy()

        if naming is not None:
            # Apply sign correction if enabled
            ew_sign_corrected = False
            vw_sign_corrected = False

            if naming.sign_correct:
                if ew.mean() < 0:
                    ew = -ew
                    ew_sign_corrected = True
                if vw.mean() < 0:
                    vw = -vw
                    vw_sign_corrected = True

            # Generate names
            signal = self.signal_name or 'factor'
            ew_name = make_factor_name(
                signal,
                naming,
                weighting='ew',
                rating=self.rating_str,
                is_within_firm=self.is_within_firm,
                sign_corrected=ew_sign_corrected,
                second_signal=self.second_signal,
            )
            vw_name = make_factor_name(
                signal,
                naming,
                weighting='vw',
                rating=self.rating_str,
                is_within_firm=self.is_within_firm,
                sign_corrected=vw_sign_corrected,
                second_signal=self.second_signal,
            )

            ew.name = ew_name
            vw.name = vw_name

        return ew, vw

    def get_long_leg(self) -> tuple[pd.Series, pd.Series]:
        """
        Get long portfolio returns as (EW, VW) tuple.

        Returns
        -------
        tuple of pd.Series
            (ew_long, vw_long)
        """
        ew = self.returns.get_long_leg("ew")
        vw = self.returns.get_long_leg("vw")
        return ew, vw

    def get_short_leg(self) -> tuple[pd.Series, pd.Series]:
        """
        Get short portfolio returns as (EW, VW) tuple.

        Returns
        -------
        tuple of pd.Series
            (ew_short, vw_short)
        """
        ew = self.returns.get_short_leg("ew")
        vw = self.returns.get_short_leg("vw")
        return ew, vw

    def get_turnover(
        self,
        level: str = 'portfolio',
        naming: Optional[NamingConfig] = None,
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.Series, pd.Series]]:
        """
        Get turnover statistics.

        Parameters
        ----------
        level : str, default='portfolio'
            'portfolio' - Return turnover per portfolio (DataFrame).
            'factor' - Return factor turnover: (P_N + P_1) / 2 (Series).
        naming : NamingConfig, optional
            If provided, rename output using naming conventions.

        Returns
        -------
        ew_turnover, vw_turnover : tuple
            Turnover statistics (DataFrames for 'portfolio', Series for 'factor').

        Raises
        ------
        ValueError
            If turnover results not available
        """
        if not self.has_turnover:
            raise ValueError(
                "Turnover results not available. "
                "Set turnover=True when initializing StrategyFormation."
            )

        ew_turn = self.turnover.ew_turnover_df.copy()
        vw_turn = self.turnover.vw_turnover_df.copy()

        if level == 'factor':
            # Factor turnover = average of long and short leg turnovers
            nport = ew_turn.shape[1]

            if self.second_signal is not None and self.num_portfolios is not None:
                # DoubleSort: average across conditioning groups
                # Portfolio layout: (1,1), (1,2), ..., (1,n2), (2,1), ..., (n1,n2)
                # where n1 = num_portfolios for main signal, n2 = nport // n1
                n1 = self.num_portfolios  # portfolios for main signal
                n2 = nport // n1  # portfolios for conditioning signal

                # Short leg: portfolios (1, 1), (1, 2), ..., (1, n2) = columns 0 to n2-1
                short_cols = list(range(n2))
                # Long leg: portfolios (n1, 1), (n1, 2), ..., (n1, n2) = columns (n1-1)*n2 to n1*n2-1
                long_cols = list(range((n1 - 1) * n2, n1 * n2))

                # Average turnover across conditioning groups
                ew_long_turn = ew_turn.iloc[:, long_cols].mean(axis=1)
                ew_short_turn = ew_turn.iloc[:, short_cols].mean(axis=1)
                vw_long_turn = vw_turn.iloc[:, long_cols].mean(axis=1)
                vw_short_turn = vw_turn.iloc[:, short_cols].mean(axis=1)

                ew_factor = (ew_long_turn + ew_short_turn) / 2
                vw_factor = (vw_long_turn + vw_short_turn) / 2
            else:
                # SingleSort: (P_N + P_1) / 2
                ew_factor = (ew_turn.iloc[:, 0] + ew_turn.iloc[:, nport - 1]) / 2
                vw_factor = (vw_turn.iloc[:, 0] + vw_turn.iloc[:, nport - 1]) / 2

            if naming is not None:
                signal = self.signal_name or 'factor'
                ew_name = make_factor_name(
                    signal, naming,
                    weighting='ew', rating=self.rating_str,
                    is_within_firm=self.is_within_firm,
                    second_signal=self.second_signal,
                )
                vw_name = make_factor_name(
                    signal, naming,
                    weighting='vw', rating=self.rating_str,
                    is_within_firm=self.is_within_firm,
                    second_signal=self.second_signal,
                )
                ew_factor.name = f"{ew_name}_turnover"
                vw_factor.name = f"{vw_name}_turnover"

            return ew_factor, vw_factor

        # Portfolio-level turnover (default)
        if naming is not None:
            signal = self.signal_name or 'factor'
            nport = len(ew_turn.columns)
            # Rename columns: 1, 2, ..., N -> cs1, cs2, ..., csN
            ew_cols = [
                make_portfolio_name(
                    signal, i + 1, nport, naming,
                    is_within_firm=self.is_within_firm,
                    second_signal=self.second_signal,
                )
                for i in range(nport)
            ]
            vw_cols = [
                make_portfolio_name(
                    signal, i + 1, nport, naming,
                    is_within_firm=self.is_within_firm,
                    second_signal=self.second_signal,
                )
                for i in range(nport)
            ]
            ew_turn.columns = ew_cols
            vw_turn.columns = vw_cols

        return ew_turn, vw_turn

    def get_characteristics(
        self,
        naming: Optional[NamingConfig] = None,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Get characteristics as (EW, VW) tuple of dictionaries.

        Parameters
        ----------
        naming : NamingConfig, optional
            If provided, rename portfolio columns using naming conventions.

        Returns
        -------
        tuple of Dict[str, pd.DataFrame]
            (ew_chars_dict, vw_chars_dict)

        Raises
        ------
        ValueError
            If characteristics results not available
        """
        if not self.has_characteristics:
            raise ValueError(
                "Characteristics results not available. "
                "Set chars parameter when initializing StrategyFormation."
            )

        ew_chars = {k: v.copy() for k, v in self.characteristics.ew_chars.items()}
        vw_chars = {k: v.copy() for k, v in self.characteristics.vw_chars.items()}

        if naming is not None:
            signal = self.signal_name or 'factor'
            for char_name in ew_chars:
                nport = len(ew_chars[char_name].columns)
                new_cols = [
                    make_portfolio_name(
                        signal, i + 1, nport, naming,
                        is_within_firm=self.is_within_firm,
                        second_signal=self.second_signal,
                    )
                    for i in range(nport)
                ]
                ew_chars[char_name].columns = new_cols
                vw_chars[char_name].columns = new_cols

        return ew_chars, vw_chars

    def summary(self, periods_per_year: int = 12) -> dict[str, Any]:
        """
        Summary dict for returns, turnover, and characteristics.

        Parameters
        ----------
        periods_per_year : int, default 12
            Periods per year for annualization

        Returns
        -------
        dict
            Summary statistics
        """
        out: dict[str, Any] = {
            "returns": self.returns.summary_stats(periods_per_year=periods_per_year)
        }

        if self.has_turnover:
            out["turnover"] = self.turnover.summary_stats()

        if self.has_characteristics:
            out["characteristics"] = {
                char: self.characteristics.summary_stats(char)
                for char in self.characteristics.available_characteristics
            }

        return out


# =============================================================================
# Complete Formation Results
# =============================================================================
@dataclass
class FormationResults:
    """
    Top-level container for strategy formation outputs.
    Holds EA (ex-ante) and optional EP (ex-post) results.

    Attributes
    ----------
    name : str
        Strategy name
    ea : StrategyResults
        Ex-ante results
    ep : StrategyResults, optional
        Ex-post results (if filters applied)
    config : dict, optional
        Configuration used for formation
    metadata : dict, optional
        Additional metadata
    port_idx : dict, optional
        Portfolio indices by date (for save_idx=True)
    """

    name: str
    ea: StrategyResults
    ep: Optional[StrategyResults] = None
    config: Optional[dict] = None
    metadata: Optional[dict] = None
    port_idx: Optional[dict] = None

    @property
    def has_ep(self) -> bool:
        """Check if ex-post results are available."""
        return self.ep is not None

    def _get_strategy_results(self, strategy: str) -> StrategyResults:
        """Get EA or EP results with helpful error message."""
        sr = self.ea if strategy == "ea" else self.ep
        if sr is None:
            if strategy == "ep":
                raise ValueError(
                    "Ex-post (EP) results not available. "
                    "EP results require filters to be applied. "
                    "Use filters={'adj': 'trim', 'level': 0.2} when creating StrategyFormation."
                )
            else:
                raise ValueError(f"Ex-ante (EA) results not available.")
        return sr

    # ------------------------------ Accessors ------------------------------ #
    def get_returns(self, strategy: str = "ea", weight_type: str = "ew") -> pd.DataFrame:
        """
        Get portfolio returns DataFrame for EA/EP and EW/VW.

        Parameters
        ----------
        strategy : {'ea', 'ep'}
            Strategy type
        weight_type : {'ew', 'vw'}
            Weight type

        Returns
        -------
        pd.DataFrame
            Portfolio returns
        """
        sr = self._get_strategy_results(strategy)

        if weight_type == "ew":
            return sr.returns.ewport_df
        elif weight_type == "vw":
            return sr.returns.vwport_df
        else:
            raise ValueError("weight_type must be 'ew' or 'vw'.")
        
    def get_ptf(self, strategy: str = "ea") -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get Portfolios returns as (EW, VW) tuple.
        
        """
        sr = self._get_strategy_results(strategy)
        return sr.get_ptf()
    
    def get_ptf_ex_post(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get ex-post long-short returns as (EW, VW) tuple.

        Convenience method for get_long_short(strategy='ep').

        Returns
        -------
        tuple of pd.Series
            (ew_long_short, vw_long_short)
        """
        return self.get_ptf(strategy='ep')

    def get_long_short(
        self,
        strategy: str = "ea",
        naming: Optional[NamingConfig] = None,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Get long-short returns as (EW, VW) tuple.

        Parameters
        ----------
        strategy : {'ea', 'ep'}
            Strategy type
        naming : NamingConfig, optional
            If provided, rename output series using naming conventions.
            If None, use legacy names (backward compatible).

        Returns
        -------
        tuple of pd.Series
            (ew_long_short, vw_long_short)
        """
        sr = self._get_strategy_results(strategy)
        return sr.get_long_short(naming=naming)

    def get_long_short_ex_post(
        self,
        naming: Optional[NamingConfig] = None,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Get ex-post long-short returns as (EW, VW) tuple.

        Convenience method for get_long_short(strategy='ep').

        Parameters
        ----------
        naming : NamingConfig, optional
            If provided, rename output series using naming conventions.

        Returns
        -------
        tuple of pd.Series
            (ew_long_short, vw_long_short)
        """
        return self.get_long_short(strategy='ep', naming=naming)

    def get_long_leg(self, strategy: str = "ea") -> tuple[pd.Series, pd.Series]:
        """
        Get long portfolio returns as (EW, VW) tuple.

        Parameters
        ----------
        strategy : {'ea', 'ep'}
            Strategy type

        Returns
        -------
        tuple of pd.Series
            (ew_long, vw_long)
        """
        sr = self._get_strategy_results(strategy)
        return sr.get_long_leg()

    def get_short_leg(self, strategy: str = "ea") -> tuple[pd.Series, pd.Series]:
        """
        Get short portfolio returns as (EW, VW) tuple.

        Parameters
        ----------
        strategy : {'ea', 'ep'}
            Strategy type

        Returns
        -------
        tuple of pd.Series
            (ew_short, vw_short)
        """
        sr = self._get_strategy_results(strategy)
        return sr.get_short_leg()
    
    def get_turnover(
        self,
        strategy: str = "ea",
        level: str = 'portfolio',
        naming: Optional[NamingConfig] = None,
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.Series, pd.Series]]:
        """
        Get turnover statistics.

        Parameters
        ----------
        strategy : {'ea', 'ep'}
            Strategy type
        level : str, default='portfolio'
            'portfolio' - Return turnover per portfolio (DataFrame).
            'factor' - Return factor turnover: (P_N + P_1) / 2 (Series).
        naming : NamingConfig, optional
            If provided, rename output using naming conventions.

        Returns
        -------
        ew_turnover, vw_turnover : tuple
            Turnover statistics (DataFrames for 'portfolio', Series for 'factor').

        Raises
        ------
        ValueError
            If strategy or turnover results not available
        """
        sr = self._get_strategy_results(strategy)
        return sr.get_turnover(level=level, naming=naming)

    def get_characteristics(
        self,
        strategy: str = "ea",
        naming: Optional[NamingConfig] = None,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Get characteristics as (EW, VW) tuple of dictionaries.

        Parameters
        ----------
        strategy : {'ea', 'ep'}
            Strategy type
        naming : NamingConfig, optional
            If provided, rename portfolio columns using naming conventions.

        Returns
        -------
        tuple of Dict[str, pd.DataFrame]
            (ew_chars_dict, vw_chars_dict)

        Raises
        ------
        ValueError
            If strategy or characteristics results not available
        """
        sr = self._get_strategy_results(strategy)
        return sr.get_characteristics(naming=naming)

    def get_ptf_bins(self) -> dict:
        """
        Get the portfolio bins (indices) for all formation dates.

        Returns
        -------
        dict
            Dictionary mapping dates to DataFrames with portfolio assignments

        Raises
        ------
        ValueError
            If save_idx was not set to True during formation
        """
        if self.port_idx is None:
            raise ValueError(
                "No portfolio bins available. Set save_idx=True when initializing StrategyFormation."
            )
        return self.port_idx

    def get_ptf_counts(self) -> pd.DataFrame:
        """
        Get the number of bonds in the short leg, long leg, and their sum
        for each portfolio formation date.

        Uses the summarize_ranks utility function to compute counts from port_idx.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by date with columns:
            - nbonds_s  : number of bonds in the short leg (rank == 1)
            - nbonds_l  : number of bonds in the long leg  (rank == max rank)
            - nbonds_ls : total number of bonds in long + short

        Raises
        ------
        ValueError
            If save_idx was not set to True during formation
        """
        if self.port_idx is None:
            raise ValueError(
                "No portfolio bins available. Set save_idx=True when initializing StrategyFormation."
            )

        # Import here to avoid circular imports
        from PyBondLab.utils import summarize_ranks

        return summarize_ranks(self.port_idx)

    def get_ptf_turnover(self, strategy: str = "ea") -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get turnover DataFrames as (EW, VW) tuple.

        Alias for get_turnover() for backward compatibility.

        Parameters
        ----------
        strategy : {'ea', 'ep'}
            Strategy type

        Returns
        -------
        tuple of pd.DataFrame
            (ew_turnover, vw_turnover)

        Raises
        ------
        ValueError
            If strategy or turnover results not available
        """
        return self.get_turnover(strategy=strategy)

    # ----------------------------- Persistence ----------------------------- #
    def save(self, filepath: str) -> None:
        """
        Save results to disk as a pickle. Appends '.pkl' if missing.

        Parameters
        ----------
        filepath : str
            Path to save file
        """
        if not filepath.endswith(".pkl"):
            filepath += ".pkl"
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "FormationResults":
        """
        Load results from disk.

        Parameters
        ----------
        filepath : str
            Path to saved file

        Returns
        -------
        FormationResults
            Loaded results
        """
        with open(filepath, "rb") as f:
            return pickle.load(f)

    # ------------------------------- Summary ------------------------------- #
    def summary(self, periods_per_year: int = 12) -> dict[str, Any]:
        """
        Summary dict for EA and optionally EP.

        Parameters
        ----------
        periods_per_year : int, default 12
            Periods per year for annualization

        Returns
        -------
        dict
            Summary statistics
        """
        out: dict[str, Any] = {
            "name": self.name,
            "ea": self.ea.summary(periods_per_year=periods_per_year),
        }
        if self.has_ep:
            out["ep"] = self.ep.summary(periods_per_year=periods_per_year)
        return out


# =============================================================================
# Builder Functions for PyBondLab
# =============================================================================
def build_strategy_results(
    ewport_df: pd.DataFrame,
    vwport_df: pd.DataFrame,
    ewls_df: Optional[pd.DataFrame] = None,
    vwls_df: Optional[pd.DataFrame] = None,
    ewls_long_df: Optional[pd.DataFrame] = None,
    vwls_long_df: Optional[pd.DataFrame] = None,
    ewls_short_df: Optional[pd.DataFrame] = None,
    vwls_short_df: Optional[pd.DataFrame] = None,
    turnover_ew_df: Optional[pd.DataFrame] = None,
    turnover_vw_df: Optional[pd.DataFrame] = None,
    chars_ew: Optional[Dict[str, pd.DataFrame]] = None,
    chars_vw: Optional[Dict[str, pd.DataFrame]] = None,
    port_idx: Optional[Dict[pd.Timestamp, pd.DataFrame]] = None,
    # Metadata for naming
    signal_name: Optional[str] = None,
    rating_str: Optional[str] = None,
    is_within_firm: bool = False,
    second_signal: Optional[str] = None,
    num_portfolios: Optional[int] = None,
) -> StrategyResults:
    """
    Build a StrategyResults object from DataFrames.

    This is the primary interface for PyBondLab to construct results.
    PyBondLab should compute all long-short returns before calling this.

    Parameters
    ----------
    ewport_df : pd.DataFrame
        Equal-weighted portfolio returns
    vwport_df : pd.DataFrame
        Value-weighted portfolio returns
    ewls_df : pd.DataFrame, optional
        Equal-weighted long-short returns
    vwls_df : pd.DataFrame, optional
        Value-weighted long-short returns
    ewls_long_df : pd.DataFrame, optional
        Equal-weighted long portfolio
    vwls_long_df : pd.DataFrame, optional
        Value-weighted long portfolio
    ewls_short_df : pd.DataFrame, optional
        Equal-weighted short portfolio
    vwls_short_df : pd.DataFrame, optional
        Value-weighted short portfolio
    turnover_ew_df : pd.DataFrame, optional
        Equal-weighted turnover
    turnover_vw_df : pd.DataFrame, optional
        Value-weighted turnover
    chars_ew : dict, optional
        Equal-weighted characteristics
    chars_vw : dict, optional
        Value-weighted characteristics
    port_idx : dict, optional
        Portfolio indices
    signal_name : str, optional
        Name of the signal used for sorting (for naming)
    rating_str : str, optional
        Rating suffix ('ig', 'hy', or None)
    is_within_firm : bool
        Whether this is a WithinFirmSort strategy
    second_signal : str, optional
        Second signal name for DoubleSort
    num_portfolios : int, optional
        Number of portfolios

    Returns
    -------
    StrategyResults
        Complete strategy results
    """
    # Build PortfolioReturns
    returns = PortfolioReturns(
        ewport_df=ewport_df,
        vwport_df=vwport_df,
        ewls_df=ewls_df,
        vwls_df=vwls_df,
        ewls_long_df=ewls_long_df,
        vwls_long_df=vwls_long_df,
        ewls_short_df=ewls_short_df,
        vwls_short_df=vwls_short_df,
    )

    # Build TurnoverResults if provided
    turnover = None
    if turnover_ew_df is not None or turnover_vw_df is not None:
        turnover = TurnoverResults(
            ew_turnover_df=turnover_ew_df if turnover_ew_df is not None
            else pd.DataFrame(index=ewport_df.index, columns=ewport_df.columns),
            vw_turnover_df=turnover_vw_df if turnover_vw_df is not None
            else pd.DataFrame(index=vwport_df.index, columns=vwport_df.columns),
        )

    # Build CharacteristicsResults if provided
    characteristics = None
    if chars_ew is not None or chars_vw is not None:
        characteristics = CharacteristicsResults(
            ew_chars=chars_ew or {},
            vw_chars=chars_vw or {}
        )

    return StrategyResults(
        returns=returns,
        turnover=turnover,
        characteristics=characteristics,
        port_idx=port_idx,
        signal_name=signal_name,
        rating_str=rating_str,
        is_within_firm=is_within_firm,
        second_signal=second_signal,
        num_portfolios=num_portfolios,
    )


def build_formation_results(
    name: str,
    ea_results: StrategyResults,
    ep_results: Optional[StrategyResults] = None,
    config: Optional[dict] = None,
    metadata: Optional[dict] = None,
    port_idx: Optional[dict] = None,
) -> FormationResults:
    """
    Build a FormationResults object.

    Parameters
    ----------
    name : str
        Strategy name
    ea_results : StrategyResults
        Ex-ante results
    ep_results : StrategyResults, optional
        Ex-post results
    config : dict, optional
        Configuration
    metadata : dict, optional
        Metadata
    port_idx : dict, optional
        Portfolio indices by date (if save_idx=True)

    Returns
    -------
    FormationResults
        Complete formation results
    """
    return FormationResults(
        name=name,
        ea=ea_results,
        ep=ep_results,
        config=config,
        metadata=metadata,
        port_idx=port_idx,
    )