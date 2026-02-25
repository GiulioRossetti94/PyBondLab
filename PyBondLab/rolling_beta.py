# -*- coding: utf-8 -*-
"""
Rolling Beta Estimation

Computes rolling-window factor betas on bond panels. Supports two engines:
'numba' (fast, single factor) and 'numpy' (multi-factor, controls).

Authors: Giulio Rossetti & Alex Dickerson
"""

import numpy as np
import pandas as pd
import warnings
from typing import Optional, List, Union, Dict, Literal

# Check if numba is available
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    njit = None


# ============================================================================
# Numba JIT-compiled kernels (only defined if numba is available)
# ============================================================================

if NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=True)
    def _panel_rolling_ols_k1(gid, y, x, window, min_obs):
        """
        Numba kernel for K=1 (single factor) rolling OLS.
        Uses closed-form moments for maximum speed.
        """
        N = y.shape[0]

        alpha = np.full(N, np.nan)
        beta = np.full(N, np.nan)
        sig_tot = np.full(N, np.nan)
        sig_idi = np.full(N, np.nan)
        adj_r2 = np.full(N, np.nan)

        # within-group cumulative sums
        cx = np.empty(N, dtype=np.float64)
        cy = np.empty(N, dtype=np.float64)
        cxx = np.empty(N, dtype=np.float64)
        cyy = np.empty(N, dtype=np.float64)
        cxy = np.empty(N, dtype=np.float64)

        prev = -1
        sx = sy = sxx = syy = sxy = 0.0
        for i in range(N):
            g = gid[i]
            if g != prev:
                prev = g
                sx = sy = sxx = syy = sxy = 0.0

            xi = x[i]
            yi = y[i]

            sx += xi
            sy += yi
            sxx += xi * xi
            syy += yi * yi
            sxy += xi * yi

            cx[i] = sx
            cy[i] = sy
            cxx[i] = sxx
            cyy[i] = syy
            cxy[i] = sxy

        gstart = 0
        for i in range(N):
            if i == 0 or gid[i] != gid[i - 1]:
                gstart = i

            left = i - window + 1
            if left < gstart:
                left = gstart

            m = i - left + 1
            if m < min_obs:
                continue

            if left == gstart:
                Sx, Sy, Sxx, Syy, Sxy = cx[i], cy[i], cxx[i], cyy[i], cxy[i]
            else:
                j = left - 1
                Sx = cx[i] - cx[j]
                Sy = cy[i] - cy[j]
                Sxx = cxx[i] - cxx[j]
                Syy = cyy[i] - cyy[j]
                Sxy = cxy[i] - cxy[j]

            mf = float(m)
            xbar = Sx / mf
            ybar = Sy / mf

            Sxxc = Sxx - (Sx * Sx) / mf
            Sxyc = Sxy - (Sx * Sy) / mf
            Syyc = Syy - (Sy * Sy) / mf  # SST

            if Sxxc <= 0.0 or Syyc <= 0.0:
                continue

            b = Sxyc / Sxxc
            a = ybar - b * xbar

            # SSE = SST - b*Sxyc
            SSE = Syyc - b * Sxyc
            if SSE < 0.0:
                SSE = 0.0

            if m > 1:
                sig_tot[i] = np.sqrt(Syyc / (mf - 1.0))

            # dof = m - (K+1) = m-2
            if m > 2:
                sig_idi[i] = np.sqrt(SSE / (mf - 2.0))
                R2 = 1.0 - SSE / Syyc
                adj_r2[i] = 1.0 - (1.0 - R2) * (mf - 1.0) / (mf - 2.0)

            alpha[i] = a
            beta[i] = b

        return alpha, beta, sig_tot, sig_idi, adj_r2

    @njit(cache=True, fastmath=True)
    def _outer_add(A, z):
        """A += z z'"""
        p = z.shape[0]
        for i in range(p):
            zi = z[i]
            for j in range(p):
                A[i, j] += zi * z[j]

    @njit(cache=True, fastmath=True)
    def _outer_sub(A, z):
        """A -= z z'"""
        p = z.shape[0]
        for i in range(p):
            zi = z[i]
            for j in range(p):
                A[i, j] -= zi * z[j]

    @njit(cache=True, fastmath=True)
    def _panel_rolling_ols_kgt1(gid, y, X, window, min_obs, ridge):
        """
        Numba kernel for K>=2 rolling OLS using rolling moments + ring buffer.

        Maintains (within each asset):
          ZZ = sum z z'      where z=[1, x]
          Zy = sum z y
          Sy = sum y
          Syy= sum y^2

        Then per t:
          beta = solve(ZZ + ridge*I, Zy)
          SST  = Syy - Sy^2/n
          SSE  = Syy - 2 beta'Zy + beta'ZZ beta
        """
        N, K = X.shape
        K1 = K + 1

        betas = np.full((N, K1), np.nan)
        sig_tot = np.full(N, np.nan)
        sig_idi = np.full(N, np.nan)
        adj_r2 = np.full(N, np.nan)

        # ring buffers (store last W rows for subtracting)
        buf_y = np.empty(window, dtype=np.float64)
        buf_x = np.empty((window, K), dtype=np.float64)

        # rolling sums/moments
        ZZ = np.zeros((K1, K1), dtype=np.float64)
        Zy = np.zeros(K1, dtype=np.float64)
        Sy = 0.0
        Syy = 0.0
        n = 0

        # augmented regressor
        z = np.empty(K1, dtype=np.float64)

        prev = -1
        head = 0  # ring index

        for i in range(N):
            g = gid[i]
            if g != prev:
                # reset for new asset
                prev = g
                ZZ[:, :] = 0.0
                Zy[:] = 0.0
                Sy = 0.0
                Syy = 0.0
                n = 0
                head = 0

            yi = y[i]

            # if window full, remove oldest
            if n >= window:
                yo = buf_y[head]

                z[0] = 1.0
                for k in range(K):
                    z[k + 1] = buf_x[head, k]

                _outer_sub(ZZ, z)
                for k1 in range(K1):
                    Zy[k1] -= z[k1] * yo
                Sy -= yo
                Syy -= yo * yo

                n -= 1  # will add new one below

            # add new obs into ring position
            buf_y[head] = yi
            for k in range(K):
                buf_x[head, k] = X[i, k]

            z[0] = 1.0
            for k in range(K):
                z[k + 1] = X[i, k]

            _outer_add(ZZ, z)
            for k1 in range(K1):
                Zy[k1] += z[k1] * yi
            Sy += yi
            Syy += yi * yi
            n += 1

            # advance ring head
            head += 1
            if head == window:
                head = 0

            if n < min_obs:
                continue

            mf = float(n)
            SST = Syy - (Sy * Sy) / mf
            if SST <= 0.0:
                continue

            # Solve (ZZ + ridge I) beta = Zy
            A = ZZ.copy()
            for d in range(K1):
                A[d, d] += ridge

            # if singular, solve may fail; ridge usually prevents that
            try:
                b = np.linalg.solve(A, Zy)
            except:
                continue

            betas[i, :] = b

            # total vol
            if n > 1:
                sig_tot[i] = np.sqrt(SST / (mf - 1.0))

            dof = n - K1
            if dof <= 0:
                continue

            # SSE via quadratic form using unregularized ZZ (standard OLS identity)
            # SSE = y'y - 2 b'Zy + b'ZZ b
            quad = 0.0
            for a in range(K1):
                tmp = 0.0
                for c in range(K1):
                    tmp += ZZ[a, c] * b[c]
                quad += b[a] * tmp

            SSE = Syy - 2.0 * np.dot(b, Zy) + quad
            if SSE < 0.0:
                SSE = 0.0

            sig_idi[i] = np.sqrt(SSE / float(dof))

            R2 = 1.0 - SSE / SST
            adj_r2[i] = 1.0 - (1.0 - R2) * (mf - 1.0) / float(dof)

        return betas, sig_tot, sig_idi, adj_r2


# ============================================================================
# Main RollingBeta Class
# ============================================================================

class RollingBeta:
    """
    Rolling factor beta estimation for bond panels.

    Computes rolling-window regressions of bond returns on factors,
    adding beta columns to the panel for use with sorting strategies.

    Parameters
    ----------
    factors : pd.DataFrame
        Factor time series with 'date' column and factor columns.
        Example: DataFrame with columns ['date', 'MKT', 'SMB', 'HML']
    controls : pd.DataFrame, optional
        Control variables (included in regression but betas not primary output).
        Only supported with engine='numpy'.
    window : int, default 36
        Rolling window size (number of valid observations).
        Window expands from min_periods to window, then rolls forward.
    min_periods : int, default 24
        Minimum number of valid observations before beta estimation starts.
    add_constant : bool, default True
        Include intercept in regression.
    no_gap : bool, default False
        Require consecutive calendar months in window. If True, betas are
        set to NaN when the window contains non-consecutive months.
        Only supported with engine='numpy'.
    compute_volatility : bool, default True
        Compute total and idiosyncratic volatility.
    compute_r2 : bool, default True
        Save adjusted R-squared per observation.
    engine : str, default 'auto'
        Computation engine to use:
        - 'auto': Use numba if available and no advanced features needed
        - 'numba': Force fast Numba engine (~30x faster)
        - 'numpy': Force pure-numpy engine (supports controls, no_gap)
    ridge : float, default 1e-12
        Ridge regularization for multi-factor regression (numba engine only).
    verbose : bool, default True
        Print progress information.

    Attributes
    ----------
    factor_names : list of str
        Names of factor columns (excluding 'date').
    engine : str
        The resolved computation engine ('numba' or 'numpy').

    Examples
    --------
    >>> from PyBondLab import RollingBeta, SingleSort, StrategyFormation
    >>>
    >>> # Load factors
    >>> factors = pd.DataFrame({
    ...     'date': pd.date_range('2010-01', periods=120, freq='ME'),
    ...     'MKT': np.random.randn(120) * 0.05,
    ... })
    >>>
    >>> # Estimate betas (uses fast numba engine by default)
    >>> beta_est = RollingBeta(factors=factors, window=36, min_periods=24)
    >>> panel = beta_est.compute(bond_data, ret_cols='ret')
    >>>
    >>> # Sort by market beta
    >>> strategy = SingleSort(holding_period=6, sort_var='MKT_beta_ret', num_portfolios=5)
    >>> sf = StrategyFormation(panel, strategy)
    >>> results = sf.fit()

    Notes
    -----
    Engine selection:
    - The 'numba' engine is ~30x faster but doesn't support controls or no_gap.
    - The 'numpy' engine supports all features but is slower.
    - With engine='auto', numba is used when possible, falling back to numpy
      when controls or no_gap are specified.
    """

    def __init__(
        self,
        factors: pd.DataFrame,
        controls: Optional[pd.DataFrame] = None,
        window: int = 36,
        min_periods: int = 24,
        add_constant: bool = True,
        no_gap: bool = False,
        compute_volatility: bool = True,
        compute_r2: bool = True,
        engine: Literal['auto', 'numba', 'numpy'] = 'auto',
        ridge: float = 1e-12,
        verbose: bool = True,
    ):
        # Validate factors
        self._validate_factors(factors)

        # Store parameters
        self.factors = factors.copy()
        self.controls = controls.copy() if controls is not None else None
        self.window = window
        self.min_periods = min_periods
        self.add_constant = add_constant
        self.no_gap = no_gap
        self.compute_volatility = compute_volatility
        self.compute_r2 = compute_r2
        self.ridge = ridge
        self.verbose = verbose

        # Extract factor names
        self.factor_names = [c for c in self.factors.columns if c != 'date']
        self.control_names = (
            [c for c in self.controls.columns if c != 'date']
            if self.controls is not None else []
        )

        # Validate parameters
        self._validate_parameters()

        # Resolve engine
        self.engine = self._resolve_engine(engine)

        if self.verbose:
            self._print_initialization()

    def _validate_factors(self, factors: pd.DataFrame) -> None:
        """Validate factor DataFrame structure."""
        if not isinstance(factors, pd.DataFrame):
            raise TypeError("factors must be a pandas DataFrame")
        if 'date' not in factors.columns:
            raise ValueError("factors must have a 'date' column")
        if len(factors.columns) < 2:
            raise ValueError("factors must have at least one factor column besides 'date'")

    def _validate_parameters(self) -> None:
        """Validate parameter values."""
        if self.window < 2:
            raise ValueError("window must be >= 2")
        if self.min_periods < 2:
            raise ValueError("min_periods must be >= 2")
        if self.min_periods > self.window:
            raise ValueError("min_periods cannot exceed window")

    def _resolve_engine(self, engine: str) -> str:
        """Resolve which computation engine to use."""
        needs_numpy = self.controls is not None or self.no_gap

        if engine == 'auto':
            if needs_numpy:
                if self.verbose:
                    reason = "controls" if self.controls is not None else "no_gap"
                    print(f"Using numpy engine (required for {reason})")
                return 'numpy'
            if NUMBA_AVAILABLE:
                return 'numba'
            else:
                if self.verbose:
                    print("Numba not available, using numpy engine")
                return 'numpy'

        elif engine == 'numba':
            if not NUMBA_AVAILABLE:
                raise ImportError(
                    "Numba is not installed. Install with: pip install numba"
                )
            if needs_numpy:
                feature = "controls" if self.controls is not None else "no_gap"
                raise ValueError(
                    f"engine='numba' does not support {feature}. "
                    f"Use engine='numpy' or engine='auto'."
                )
            return 'numba'

        elif engine == 'numpy':
            return 'numpy'

        else:
            raise ValueError(f"Unknown engine: {engine}. Use 'auto', 'numba', or 'numpy'.")

    def _print_initialization(self) -> None:
        """Print initialization summary."""
        print("-" * 50)
        print("RollingBeta Estimator Initialized")
        print(f"  Engine: {self.engine}")
        print(f"  Factors: {self.factor_names}")
        if self.control_names:
            print(f"  Controls: {self.control_names}")
        print(f"  Window: {self.window}")
        print(f"  Min periods: {self.min_periods}")
        print(f"  Add constant: {self.add_constant}")
        if self.no_gap:
            print(f"  No gap: {self.no_gap}")
        print(f"  Compute volatility: {self.compute_volatility}")
        print(f"  Compute R²: {self.compute_r2}")
        print("-" * 50)

    def compute(
        self,
        data: pd.DataFrame,
        ret_cols: Union[str, List[str]] = 'ret',
        id_col: str = 'ID',
        date_col: str = 'date',
    ) -> pd.DataFrame:
        """
        Compute rolling betas and add to panel.

        Parameters
        ----------
        data : pd.DataFrame
            Bond panel with ID, date, and return columns.
        ret_cols : str or list of str, default 'ret'
            Return column(s) to compute betas for.
        id_col : str, default 'ID'
            Column name for bond identifier.
        date_col : str, default 'date'
            Column name for date.

        Returns
        -------
        pd.DataFrame
            Original panel with added beta columns:
            - {factor}_beta_{ret_col} for each factor and return column
            - sigma_total_{ret_col} if compute_volatility=True
            - sigma_idio_{ret_col} if compute_volatility=True
            - adj_R2_{ret_col} if compute_r2=True
        """
        # Normalize ret_cols to list
        if isinstance(ret_cols, str):
            ret_cols = [ret_cols]

        # Validate input data
        self._validate_input_data(data, ret_cols, id_col, date_col)

        if self.verbose:
            print(f"Computing betas for {len(ret_cols)} return column(s) using {self.engine} engine...")

        # Dispatch to appropriate engine
        if self.engine == 'numba':
            return self._compute_numba(data, ret_cols, id_col, date_col)
        else:
            return self._compute_numpy(data, ret_cols, id_col, date_col)

    def _validate_input_data(
        self,
        data: pd.DataFrame,
        ret_cols: List[str],
        id_col: str,
        date_col: str,
    ) -> None:
        """Validate input data structure."""
        if id_col not in data.columns:
            raise ValueError(f"ID column '{id_col}' not found in data")
        if date_col not in data.columns:
            raise ValueError(f"Date column '{date_col}' not found in data")
        for col in ret_cols:
            if col not in data.columns:
                raise ValueError(f"Return column '{col}' not found in data")

    # ========================================================================
    # Numba Engine Implementation
    # ========================================================================

    def _compute_numba(
        self,
        data: pd.DataFrame,
        ret_cols: List[str],
        id_col: str,
        date_col: str,
    ) -> pd.DataFrame:
        """Compute betas using fast Numba engine."""
        output = data.copy()

        for ret_col in ret_cols:
            if self.verbose and len(ret_cols) > 1:
                print(f"  Processing {ret_col}...")

            result = self._compute_numba_single_ret(
                data, ret_col, id_col, date_col
            )

            # Merge results back
            output = self._merge_numba_results(output, result, ret_col, id_col, date_col)

        if self.verbose:
            print("Beta computation complete.")

        return output

    def _compute_numba_single_ret(
        self,
        data: pd.DataFrame,
        ret_col: str,
        id_col: str,
        date_col: str,
    ) -> pd.DataFrame:
        """Compute betas for single return column using Numba."""
        # Prepare factor data
        factor_df = self.factors.copy()
        factor_df['date'] = pd.to_datetime(factor_df['date'])

        # Prepare bond data
        rdf = data[[id_col, date_col, ret_col]].copy()
        rdf[date_col] = pd.to_datetime(rdf[date_col])

        # Merge factors
        merged = pd.merge(
            rdf,
            factor_df,
            left_on=date_col,
            right_on='date',
            how='left',
            sort=False,
        )

        # Drop NaN rows
        merged = merged.dropna(subset=[ret_col] + self.factor_names).reset_index(drop=True)

        if len(merged) == 0:
            return pd.DataFrame()

        # Factorize IDs
        gid, _ = pd.factorize(merged[id_col], sort=False)
        merged['_gid'] = gid.astype(np.int32)

        # Sort by (gid, date)
        merged = merged.sort_values(['_gid', date_col], kind='mergesort').reset_index(drop=True)

        gid_arr = merged['_gid'].to_numpy(dtype=np.int32)
        y = merged[ret_col].to_numpy(dtype=np.float64)

        K = len(self.factor_names)

        if K == 1:
            # Single factor - use fast K=1 kernel
            x = merged[self.factor_names[0]].to_numpy(dtype=np.float64)
            alpha, beta, sig_tot, sig_idi, adj_r2 = _panel_rolling_ols_k1(
                gid_arr, y, x, int(self.window), int(self.min_periods)
            )

            result = pd.DataFrame({
                id_col: merged[id_col].to_numpy(),
                date_col: merged[date_col].to_numpy(),
                f'{self.factor_names[0]}_beta_{ret_col}': beta,
            })

            if self.compute_volatility:
                result[f'sigma_total_{ret_col}'] = sig_tot
                result[f'sigma_idio_{ret_col}'] = sig_idi

            if self.compute_r2:
                result[f'adj_R2_{ret_col}'] = adj_r2

        else:
            # Multiple factors - use K>1 kernel
            X = merged[self.factor_names].to_numpy(dtype=np.float64)
            betas, sig_tot, sig_idi, adj_r2 = _panel_rolling_ols_kgt1(
                gid_arr, y, X, int(self.window), int(self.min_periods), float(self.ridge)
            )

            result = pd.DataFrame({
                id_col: merged[id_col].to_numpy(),
                date_col: merged[date_col].to_numpy(),
            })

            # Extract factor betas (skip intercept at index 0)
            for j, factor_name in enumerate(self.factor_names):
                result[f'{factor_name}_beta_{ret_col}'] = betas[:, j + 1]

            if self.compute_volatility:
                result[f'sigma_total_{ret_col}'] = sig_tot
                result[f'sigma_idio_{ret_col}'] = sig_idi

            if self.compute_r2:
                result[f'adj_R2_{ret_col}'] = adj_r2

        return result

    def _merge_numba_results(
        self,
        output: pd.DataFrame,
        result: pd.DataFrame,
        ret_col: str,
        id_col: str,
        date_col: str,
    ) -> pd.DataFrame:
        """Merge Numba results back to original panel."""
        if result.empty:
            # Add NaN columns
            for factor_name in self.factor_names:
                output[f'{factor_name}_beta_{ret_col}'] = np.nan
            if self.compute_volatility:
                output[f'sigma_total_{ret_col}'] = np.nan
                output[f'sigma_idio_{ret_col}'] = np.nan
            if self.compute_r2:
                output[f'adj_R2_{ret_col}'] = np.nan
            return output

        # Get columns to merge (exclude id and date)
        merge_cols = [c for c in result.columns if c not in [id_col, date_col]]

        # Normalize dates for merge
        output['_date_merge'] = pd.to_datetime(output[date_col])
        result['_date_merge'] = pd.to_datetime(result[date_col])

        # Merge on id + date
        output = output.merge(
            result[[id_col, '_date_merge'] + merge_cols],
            on=[id_col, '_date_merge'],
            how='left',
        )

        output = output.drop(columns=['_date_merge'])

        return output

    # ========================================================================
    # NumPy Engine Implementation (supports controls and no_gap)
    # ========================================================================

    def _compute_numpy(
        self,
        data: pd.DataFrame,
        ret_cols: List[str],
        id_col: str,
        date_col: str,
    ) -> pd.DataFrame:
        """Compute betas using pure NumPy engine."""
        # Merge factors into data
        merged = self._merge_factors(data.copy(), date_col)

        # Get all regressor column names
        all_regressor_cols = self.factor_names + self.control_names

        # Compute betas for each bond
        results = self._compute_all_bonds_numpy(
            merged, ret_cols, all_regressor_cols, id_col, date_col
        )

        # Merge results back to original data
        output = self._merge_results_numpy(data, results, ret_cols, id_col, date_col)

        if self.verbose:
            print("Beta computation complete.")

        return output

    def _merge_factors(self, data: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Merge factor data into bond panel with date validation."""
        # Normalize dates to month-end
        data[date_col] = pd.to_datetime(data[date_col])
        data['_date_normalized'] = data[date_col].dt.to_period('M').dt.to_timestamp('M')

        factor_df = self.factors.copy()
        factor_df['date'] = pd.to_datetime(factor_df['date'])
        factor_df['_date_normalized'] = factor_df['date'].dt.to_period('M').dt.to_timestamp('M')

        # Check date coverage
        bond_dates = set(data['_date_normalized'].unique())
        factor_dates = set(factor_df['_date_normalized'].unique())

        missing = bond_dates - factor_dates
        if missing:
            pct_missing = len(missing) / len(bond_dates) * 100
            if pct_missing > 10:
                raise ValueError(
                    f"Factor data missing for {pct_missing:.1f}% of bond dates. "
                    f"First missing: {min(missing)}, Last missing: {max(missing)}"
                )
            elif self.verbose:
                warnings.warn(
                    f"Factor data missing for {len(missing)} dates ({pct_missing:.1f}%). "
                    f"Bonds on these dates will have NaN betas."
                )

        # Merge factor data
        merged = data.merge(
            factor_df.drop(columns=['date']),
            on='_date_normalized',
            how='left',
            validate='m:1',
        )

        # Merge control data if provided
        if self.controls is not None:
            control_df = self.controls.copy()
            control_df['date'] = pd.to_datetime(control_df['date'])
            control_df['_date_normalized'] = control_df['date'].dt.to_period('M').dt.to_timestamp('M')

            merged = merged.merge(
                control_df.drop(columns=['date']),
                on='_date_normalized',
                how='left',
                validate='m:1',
            )

        return merged

    def _compute_all_bonds_numpy(
        self,
        data: pd.DataFrame,
        ret_cols: List[str],
        regressor_cols: List[str],
        id_col: str,
        date_col: str,
    ) -> pd.DataFrame:
        """Compute betas for all bonds using NumPy engine."""
        # Preserve original index before sorting
        data = data.copy()
        data['_original_idx'] = data.index

        # Sort data
        data = data.sort_values([id_col, date_col]).reset_index(drop=True)

        all_results = []

        # Group by bond
        grouped = data.groupby(id_col, sort=False)
        n_bonds = len(grouped)

        if self.verbose:
            print(f"Processing {n_bonds} bonds...")

        for bond_idx, (bond_id, bond_group) in enumerate(grouped):
            if self.verbose and (bond_idx + 1) % 1000 == 0:
                print(f"  Processed {bond_idx + 1}/{n_bonds} bonds...")

            # Compute betas for this bond
            bond_results = self._compute_single_bond_numpy(
                bond_id, bond_group, ret_cols, regressor_cols, date_col
            )
            if bond_results is not None:
                all_results.append(bond_results)

        if not all_results:
            return pd.DataFrame()

        return pd.concat(all_results, ignore_index=True)

    def _compute_single_bond_numpy(
        self,
        bond_id,
        bond_group: pd.DataFrame,
        ret_cols: List[str],
        regressor_cols: List[str],
        date_col: str,
    ) -> Optional[pd.DataFrame]:
        """Compute betas for a single bond using NumPy."""
        results = []

        for ret_col in ret_cols:
            ret_results = self._compute_single_bond_single_ret_numpy(
                bond_id, bond_group, ret_col, regressor_cols, date_col
            )
            if ret_results:
                results.extend(ret_results)

        if not results:
            return None

        return pd.DataFrame(results)

    def _compute_single_bond_single_ret_numpy(
        self,
        bond_id,
        bond_group: pd.DataFrame,
        ret_col: str,
        regressor_cols: List[str],
        date_col: str,
    ) -> List[Dict]:
        """Compute betas for single bond/return using NumPy with NaN-aware pre-filtering."""
        # Extract arrays
        y_all = bond_group[ret_col].values.astype(float)
        X_all = bond_group[regressor_cols].values.astype(float)
        dates_all = bond_group[date_col].values
        orig_indices = bond_group['_original_idx'].values

        # For no_gap check
        if self.no_gap:
            dates_dt = pd.to_datetime(dates_all)
            month_idx_all = dates_dt.year * 12 + dates_dt.month
        else:
            month_idx_all = None

        # Create validity mask
        valid_mask = ~np.isnan(y_all) & np.all(~np.isnan(X_all), axis=1)

        # Filter to valid observations
        y = y_all[valid_mask]
        X = X_all[valid_mask]
        dates = dates_all[valid_mask]
        orig_idx = orig_indices[valid_mask]
        month_idx = month_idx_all[valid_mask] if self.no_gap else None

        n = len(y)

        if n < self.min_periods:
            return []

        # Add constant if needed
        if self.add_constant:
            X = np.column_stack([np.ones(n), X])

        k = X.shape[1]

        # Compute cumulative sums
        cum_X = np.cumsum(X, axis=0)
        cum_XTX = np.cumsum(np.einsum('ni,nj->nij', X, X), axis=0)
        cum_Xy = np.cumsum(X * y[:, None], axis=0)
        cum_y = np.cumsum(y)
        cum_yy = np.cumsum(y * y)

        # Prepend zeros
        cum_X = np.vstack([np.zeros((1, k)), cum_X])
        cum_XTX = np.vstack([np.zeros((1, k, k)), cum_XTX])
        cum_Xy = np.vstack([np.zeros((1, k)), cum_Xy])
        cum_y = np.insert(cum_y, 0, 0.0)
        cum_yy = np.insert(cum_yy, 0, 0.0)

        if self.no_gap:
            month_idx_padded = np.insert(month_idx, 0, 0)

        results = []

        for i in range(self.min_periods, n + 1):
            start = max(0, i - self.window)
            win_size = i - start

            # Check for gaps
            if self.no_gap:
                first_month = month_idx_padded[start + 1]
                last_month = month_idx_padded[i]
                if last_month - first_month != win_size - 1:
                    continue

            # O(1) window extraction
            Sxx = cum_XTX[i] - cum_XTX[start]
            Sxy = cum_Xy[i] - cum_Xy[start]
            Sy = cum_y[i] - cum_y[start]
            Syy = cum_yy[i] - cum_yy[start]

            try:
                beta = np.linalg.solve(Sxx, Sxy)
            except np.linalg.LinAlgError:
                continue

            row = {
                '_bond_id': bond_id,
                '_date': dates[i - 1],
                '_orig_idx': orig_idx[i - 1],
                '_ret_col': ret_col,
            }

            # Extract betas
            if self.add_constant:
                factor_betas = beta[1:]
            else:
                factor_betas = beta

            for j, name in enumerate(self.factor_names):
                row[f'{name}_beta_{ret_col}'] = factor_betas[j]

            # Diagnostics
            if self.compute_volatility or self.compute_r2:
                sse = Syy - beta @ Sxy
                tss = Syy - Sy**2 / win_size

                if self.compute_volatility:
                    if win_size > 1:
                        row[f'sigma_total_{ret_col}'] = np.sqrt(tss / (win_size - 1)) if tss > 0 else np.nan
                    dof = win_size - k
                    if dof > 0:
                        row[f'sigma_idio_{ret_col}'] = np.sqrt(sse / dof) if sse > 0 else np.nan

                if self.compute_r2 and tss > 0:
                    R2 = 1.0 - sse / tss
                    dof = win_size - k
                    if dof > 0:
                        row[f'adj_R2_{ret_col}'] = 1.0 - (1.0 - R2) * (win_size - 1) / dof

            results.append(row)

        return results

    def _merge_results_numpy(
        self,
        original_data: pd.DataFrame,
        results: pd.DataFrame,
        ret_cols: List[str],
        id_col: str,
        date_col: str,
    ) -> pd.DataFrame:
        """Merge NumPy results back to original panel."""
        if results.empty:
            output = original_data.copy()
            for ret_col in ret_cols:
                for name in self.factor_names:
                    output[f'{name}_beta_{ret_col}'] = np.nan
                if self.compute_volatility:
                    output[f'sigma_total_{ret_col}'] = np.nan
                    output[f'sigma_idio_{ret_col}'] = np.nan
                if self.compute_r2:
                    output[f'adj_R2_{ret_col}'] = np.nan
            return output

        # _orig_idx values are 0..N-1 (from RangeIndex after factor merge),
        # so we must reset original_data's index to match.
        output = original_data.copy().reset_index(drop=True)

        for ret_col in ret_cols:
            ret_results = results[results['_ret_col'] == ret_col].copy()

            if ret_results.empty:
                for name in self.factor_names:
                    output[f'{name}_beta_{ret_col}'] = np.nan
                continue

            beta_cols = [c for c in ret_results.columns
                         if c.endswith(f'_{ret_col}') and c != '_ret_col']

            ret_results_indexed = ret_results.set_index('_orig_idx')[beta_cols]

            for col in beta_cols:
                output[col] = np.nan
                output.loc[ret_results_indexed.index, col] = ret_results_indexed[col].values

        return output
