"""
Optimized AnomalyAssayer with Parallelization Support

Key improvements:
1. Optional parallelization for grid search (not default)
2. Better interface for custom breakpoints and functions
3. Correlation analysis function
4. Progress tracking
5. Result caching for faster reruns

@authors: Giulio Rossetti & Alex Dickerson (optimized version)
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Callable
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm

from PyBondLab.PyBondLab import StrategyFormation
from PyBondLab.StrategyClass import Strategy, SingleSort
from PyBondLab.visualization.plotting import PerformancePlotter
from PyBondLab.constants import get_rating_bounds

Number = Union[int, float]
SubsetFilter = Dict[str, List[Tuple[Number, Number]]]


# =============================================================================
# Optimized Runner with Parallelization Support
# =============================================================================
class AssayAnomalyRunner:
    """
    Grid runner with optional parallelization.

    New features:
    - Optional parallel execution (n_jobs parameter)
    - Better progress tracking
    - Custom breakpoints per portfolio size
    - Correlation analysis
    """

    # Default grids (can be overridden)
    HOLDING_PERIODS = [1, 3]
    NPORT = [5, 10]
    RATINGS = [None, "NIG", "IG"]

    def __init__(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        RETvar: Union[str, List[str], None] = None,
        IDvar: Optional[str] = None,
        DATEvar: Optional[str] = None,
        PRICEvar: Optional[str] = None,
        RATINGvar: Optional[str] = None,
        Wvar: Optional[str] = None,
        subset_filter: Optional[SubsetFilter] = None,
        holding_periods: Optional[List[int]] = None,
        nport: Optional[List[int]] = None,
        ratings: Optional[List[Any]] = None,
        breakpoints: Optional[Union[List[float], Dict[int, List[float]]]] = None,  # NEW: per-nport breakpoints
        breakpoint_universe_func: Optional[Callable] = None,
        dynamic_weights: bool = True,
        turnover: bool = True,
        save_idx: bool = True,
        n_jobs: int = 1,  # NEW: parallelization (default=1, no parallel)
        verbose: bool = True,
    ):
        self.strategy = strategy
        self.data = data.copy()
        self.subset_filter = subset_filter
        self.verbose = verbose
        self.n_jobs = n_jobs

        self.dynamic_weights = dynamic_weights
        self.turnover = turnover
        self.save_idx = save_idx

        # Handle breakpoints - can be list or dict{nport: breakpoints}
        self.breakpoints = breakpoints
        if isinstance(breakpoints, dict):
            self.breakpoints_dict = breakpoints
        elif breakpoints is not None:
            # Apply same breakpoints to all nport
            self.breakpoints_dict = {n: breakpoints for n in (nport or self.NPORT)}
        else:
            self.breakpoints_dict = {}

        # Breakpoint universe function
        self.breakpoint_universe_func = breakpoint_universe_func
        if breakpoint_universe_func is not None:
            self.strategy.breakpoint_universe_func = breakpoint_universe_func

        # Grids
        self.HOLDING_PERIODS = holding_periods or AssayAnomalyRunner.HOLDING_PERIODS
        self.NPORT = nport or AssayAnomalyRunner.NPORT
        self.RATINGS = ratings or AssayAnomalyRunner.RATINGS

        # RETvar(s)
        if isinstance(RETvar, list):
            self.retvars = RETvar
        elif RETvar is None:
            self.retvars = [None]
        else:
            self.retvars = [RETvar]

        # fit() kwargs
        self.fit_kwargs = {
            "IDvar": IDvar,
            "DATEvar": DATEvar,
            "PRICEvar": PRICEvar,
            "RATINGvar": RATINGvar,
            "VWvar": Wvar,
        }

        # Progress
        self._run_counter = 0
        self._run_total = 0
        self._start_time = None

    # ----------------------------- utilities ----------------------------------
    @staticmethod
    def _summarize_ranks(dfs_by_date: Dict[pd.Timestamp, pd.DataFrame]) -> pd.DataFrame:
        """Compute #bonds in S(=1), L(=max), and S+L."""
        dates, cnt1, cntmax = [], [], []

        for date, df in dfs_by_date.items():
            if df.empty or "ptf_rank" not in df:
                dates.append(date)
                cnt1.append(np.nan)
                cntmax.append(np.nan)
                continue
            arr = df["ptf_rank"].to_numpy()
            if arr.size == 0:
                dates.append(date)
                cnt1.append(np.nan)
                cntmax.append(np.nan)
                continue
            mx = np.nanmax(arr)
            cnt1.append(np.count_nonzero(arr == 1))
            cntmax.append(np.count_nonzero(arr == mx))
            dates.append(date)

        res = pd.DataFrame(
            {"nbonds_s": cnt1, "nbonds_l": cntmax},
            index=pd.to_datetime(dates),
        ).sort_index()
        res["nbonds_ls"] = res["nbonds_s"] + res["nbonds_l"]
        return res

    @staticmethod
    def _prep_panel(df: pd.DataFrame, weight: str, tp: str, retvar: str) -> pd.DataFrame:
        """Normalize TS DataFrame to standard schema."""
        if df.shape[1] < 1:
            raise ValueError("Expected DataFrame with at least one column of returns.")
        out = df.copy()
        out = out.rename(columns={out.columns[0]: retvar})
        return out.assign(weight=weight, type=tp)

    @staticmethod
    def _label_nport(nport: int) -> str:
        return {3: "T", 5: "Q", 10: "D"}.get(nport, str(nport))

    # ----------------------------- expansion ----------------------------------
    def _expand_subset_filters(self) -> List[Optional[Dict[str, Tuple[Number, Number]]]]:
        """Expand subset_filter dict (cartesian product)."""
        if not self.subset_filter:
            return [None]

        keys = list(self.subset_filter.keys())
        out: List[Dict[str, Tuple[Number, Number]]] = []

        def expand(idx=0, current=None):
            current = {} if current is None else current
            if idx == len(keys):
                out.append(current.copy())
                return
            k = keys[idx]
            vals = self.subset_filter[k]
            if isinstance(vals, list) and all(isinstance(v, tuple) and len(v) == 2 for v in vals):
                for v in vals:
                    current[k] = v
                    expand(idx + 1, current)
            else:
                current[k] = vals
                expand(idx + 1, current)

        expand()
        return out

    # ----------------------------- main loop ----------------------------------
    def run(self) -> "AnomalyResults":
        """
        Execute StrategyFormation.fit over all grid combinations.

        Uses parallelization if n_jobs > 1.
        """
        filter_sets = self._expand_subset_filters()

        n_hp = len(self.HOLDING_PERIODS)
        n_npf = len(self.NPORT)
        n_rat = len(self.RATINGS)
        n_ret = len(self.retvars)
        n_flt = len(filter_sets)
        self._run_total = n_hp * n_npf * n_rat * n_ret * n_flt

        print(f"\n{'='*70}")
        print(f"Assaying Anomaly: {self._run_total} total combinations")
        print(f"Parallelization: {f'{self.n_jobs} workers' if self.n_jobs > 1 else 'Sequential'}")
        print(f"Turnover: {self.turnover}, Save Index: {self.save_idx}")
        print(f"{'='*70}\n")

        self._start_time = time.time()

        # Build task list - sorted by (rating, filt, retcol) for batching optimization
        tasks = []
        for rat in self.RATINGS:
            for filt in filter_sets:
                for retcol in self.retvars:
                    for hp in self.HOLDING_PERIODS:
                        for npf in self.NPORT:
                            tasks.append((hp, npf, rat, retcol, filt))

        # Initialize cache for batching similar runs (Option 6 optimization)
        self._precomp_cache = {}

        # Execute tasks (parallel or sequential)
        if self.n_jobs > 1:
            runs = self._run_parallel(tasks)
        else:
            runs = self._run_sequential(tasks)

        # Clear cache after run
        self._precomp_cache = {}

        elapsed = time.time() - self._start_time
        print(f"\nAssaying complete in {elapsed:.2f}s ({elapsed/self._run_total:.3f}s per combination)")

        runs = pd.concat(runs, axis=0).sort_index()
        return AnomalyResults(
            runs,
            params={
                "sort_var": getattr(self.strategy, "sort_var", None),
                "IDvar": self.fit_kwargs.get("IDvar"),
                "RETvar": self.retvars if len(self.retvars) > 1 else self.retvars[0],
                "RATINGvar": self.fit_kwargs.get("RATINGvar"),
                "holding_periods": self.HOLDING_PERIODS,
                "nport": self.NPORT,
                "ratings": self.RATINGS,
                "subset_filter": self.subset_filter,
                "breakpoints": self.breakpoints,
                "n_jobs": self.n_jobs,
            },
        )

    def _make_batch_key(self, rating, filt, retcol) -> tuple:
        """Create a hashable cache key for batching similar runs."""
        filt_key = tuple(sorted(filt.items())) if filt else None
        return (rating, filt_key, retcol)

    def _run_sequential(self, tasks: List[Tuple]) -> List[pd.DataFrame]:
        """Run tasks sequentially with batching optimization."""
        runs = []
        for task in tasks:
            self._run_counter += 1
            runs.append(self._run_single(*task))
        return runs

    def _run_parallel(self, tasks: List[Tuple]) -> List[pd.DataFrame]:
        """Run tasks in parallel using ProcessPoolExecutor."""
        runs = []

        # Create partial function with fixed parameters
        run_func = partial(
            _run_single_static,
            data=self.data,
            strategy_info=self._get_strategy_info(),
            fit_kwargs=self.fit_kwargs,
            breakpoints_dict=self.breakpoints_dict,
            breakpoint_universe_func=self.breakpoint_universe_func,
            dynamic_weights=self.dynamic_weights,
            turnover=self.turnover,
            save_idx=self.save_idx,
        )

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(run_func, *task, i+1, self._run_total): task
                            for i, task in enumerate(tasks)}

            # Collect results as they complete
            for future in as_completed(future_to_task):
                self._run_counter += 1
                result = future.result()
                runs.append(result)

                if self.verbose and self._run_counter % 10 == 0:
                    elapsed = time.time() - self._start_time
                    rate = self._run_counter / elapsed
                    remaining = (self._run_total - self._run_counter) / rate
                    print(f"  [{self._run_counter}/{self._run_total}] "
                          f"{elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining")

        return runs

    def _get_strategy_info(self) -> Dict:
        """Extract strategy information for pickling."""
        return {
            "type": type(self.strategy).__name__,
            "sort_var": getattr(self.strategy, "sort_var", None),
            "sort_var2": getattr(self.strategy, "sort_var2", None),
            "lookback_period": getattr(self.strategy, "lookback_period", None),
            "skip": getattr(self.strategy, "skip", None),
        }

    def _run_single(
        self,
        hp: int,
        npf: int,
        rating: Any,
        retcol: Optional[str],
        filt: Optional[Dict[str, Tuple[Number, Number]]] = None,
    ) -> pd.DataFrame:
        """Run one (hp, npf, rating, retcol, filt) combination."""
        count_str = f"{self._run_counter}/{self._run_total}"

        # Clone strategy with custom breakpoints if provided
        breakpoints_for_npf = self.breakpoints_dict.get(npf, None)
        strat = self._clone_strategy_with(hp=hp, npf=npf, breakpoints=breakpoints_for_npf)

        if self.verbose:
            filt_str = (
                "ALL"
                if not filt
                else ";".join(f"{k}[{v[0]}-{('inf' if math.isinf(v[1]) else v[1])}]"
                             for k, v in filt.items())
            )
            rat_str = rating if rating is not None else "ALL"
            bp_str = f", BP={breakpoints_for_npf}" if breakpoints_for_npf else ""
            print(
                f"{count_str} HP={hp}, NPORT={npf}, RATING={rat_str}, "
                f"RET={retcol}, SUBSET={filt_str}{bp_str}"
            )

        # Option 6: Check cache for shared precomputed data
        batch_key = self._make_batch_key(rating, filt, retcol)
        cached_precomp = self._precomp_cache.get(batch_key)

        # Run StrategyFormation
        # Note: We pass the original data with a copy to ensure correctness.
        # The turnover/save_idx optimizations provide the main speedup.
        # Option 6: Pass cached precomputed data if available
        sf = (
            StrategyFormation(
                self.data.copy(),
                strategy=strat,
                subset_filter=filt,
                rating=rating,
                dynamic_weights=self.dynamic_weights,
                turnover=self.turnover,
                save_idx=self.save_idx,
                verbose=False,
                cached_precomp=cached_precomp,  # Option 6: Pass cached data
            )
            .fit(RETvar=retcol, **self.fit_kwargs)
        )

        # Option 6: Cache precomputed data for reuse by subsequent runs with same (rating, filt, retcol)
        if cached_precomp is None and hasattr(sf, '_shareable_precomp'):
            self._precomp_cache[batch_key] = sf._shareable_precomp

        # Extract and package results
        return self._package_results(sf, hp, npf, rating, retcol, filt)

    def _package_results(
        self,
        sf: StrategyFormation,
        hp: int,
        npf: int,
        rating: Any,
        retcol: Optional[str],
        filt: Optional[Dict[str, Tuple[Number, Number]]],
    ) -> pd.DataFrame:
        """Package StrategyFormation results into standard format."""
        # Unpack returns
        ew_ls, vw_ls = sf.get_long_short()
        ew_l, vw_l = sf.get_long_leg()
        ew_s, vw_s = sf.get_short_leg()

        # Turnover (only compute if enabled)
        if self.turnover:
            to_ew, to_vw = sf.get_ptf_turnover()
            ew_to = ((to_ew.iloc[:, -1] + to_ew.iloc[:, 0]) / 2).to_frame("TO")
            vw_to = ((to_vw.iloc[:, -1] + to_vw.iloc[:, 0]) / 2).to_frame("TO")
            ew_to_s = to_ew.iloc[:, 0].to_frame("TO")
            ew_to_l = to_ew.iloc[:, -1].to_frame("TO")
            vw_to_s = to_vw.iloc[:, 0].to_frame("TO")
            vw_to_l = to_vw.iloc[:, -1].to_frame("TO")
        else:
            # Create NaN turnover columns with same index as returns
            ew_to = pd.DataFrame({"TO": np.nan}, index=ew_ls.index)
            vw_to = pd.DataFrame({"TO": np.nan}, index=vw_ls.index)
            ew_to_s = ew_to.copy()
            ew_to_l = ew_to.copy()
            vw_to_s = vw_to.copy()
            vw_to_l = vw_to.copy()

        # Bond counts (only compute if save_idx enabled)
        if self.save_idx:
            bins_bonds = self._summarize_ranks(sf.get_ptf_bins())
            nbonds_s = bins_bonds["nbonds_s"].rename("nbonds")
            nbonds_l = bins_bonds["nbonds_l"].rename("nbonds")
            nbonds_ls = bins_bonds["nbonds_ls"].rename("nbonds")
        else:
            # Create NaN nbonds columns with same index as returns
            nbonds_s = pd.Series(np.nan, index=ew_ls.index, name="nbonds")
            nbonds_l = pd.Series(np.nan, index=ew_ls.index, name="nbonds")
            nbonds_ls = pd.Series(np.nan, index=ew_ls.index, name="nbonds")

        # Merge
        ew_ls_panel = pd.concat([ew_ls, ew_to, nbonds_ls], axis=1)
        vw_ls_panel = pd.concat([vw_ls, vw_to, nbonds_ls], axis=1)
        ew_l_panel = pd.concat([ew_l, ew_to_l, nbonds_l], axis=1)
        vw_l_panel = pd.concat([vw_l, vw_to_l, nbonds_l], axis=1)
        ew_s_panel = pd.concat([ew_s, ew_to_s, nbonds_s], axis=1)
        vw_s_panel = pd.concat([vw_s, vw_to_s, nbonds_s], axis=1)

        # Pack
        pack = pd.concat(
            [
                self._prep_panel(ew_ls_panel, "EW", "LS", "ret"),
                self._prep_panel(vw_ls_panel, "VW", "LS", "ret"),
                self._prep_panel(ew_l_panel, "EW", "L", "ret"),
                self._prep_panel(vw_l_panel, "VW", "L", "ret"),
                self._prep_panel(ew_s_panel, "EW", "S", "ret"),
                self._prep_panel(vw_s_panel, "VW", "S", "ret"),
            ],
            axis=0,
        )

        # Labels
        nport_label = self._label_nport(npf)
        rating_val = rating if rating is not None else "ALL"
        ret_lab = retcol if retcol is not None else "ret"

        if not filt:
            subset_label = "ALL"
        else:
            kv = next(iter(filt.items()))
            low, high = kv[1]
            low_str = "inf" if math.isinf(low) else low
            high_str = "inf" if math.isinf(high) else high
            subset_label = f"{kv[0]}_{low_str}-{high_str}"

        # Metadata
        base_id = f"HP_{hp}_NPF_{npf}_RAT_{rating_val}_{ret_lab}"
        pack["Holding"] = hp
        pack["Sort"] = nport_label
        pack["Rating"] = rating_val
        pack["Ret_string"] = ret_lab
        pack["Subset"] = subset_label
        pack["ID"] = (
            base_id + "_" + pack["weight"].astype(str) + "_" + pack["type"].astype(str)
        )

        return pack

    def _clone_strategy_with(
        self,
        hp: int,
        npf: int,
        breakpoints: Optional[List[float]] = None
    ) -> Strategy:
        """Clone strategy with new parameters."""
        if isinstance(self.strategy, SingleSort):
            return SingleSort(
                holding_period=hp,
                sort_var=self.strategy.sort_var,
                num_portfolios=npf,
                breakpoints=breakpoints or getattr(self.strategy, "breakpoints", None),
                lookback_period=getattr(self.strategy, "lookback_period", None),
                skip=getattr(self.strategy, "skip", None),
                breakpoint_universe_func=self.breakpoint_universe_func,
                verbose=False,
            )
        # Generic fallback
        s = self.strategy
        return type(s)(
            holding_period=hp,
            num_portfolios=npf,
            breakpoints=breakpoints,
            lookback_period=getattr(s, "lookback_period", None),
            skip=getattr(s, "skip", None),
            verbose=False,
            **{
                k: getattr(s, k)
                for k in ("sort_var", "sort_var2", "num_portfolios2", "breakpoints2", "how")
                if hasattr(s, k)
            },
        )


# =============================================================================
# Static function for parallel execution
# =============================================================================
def _run_single_static(
    hp: int,
    npf: int,
    rating: Any,
    retcol: Optional[str],
    filt: Optional[Dict],
    counter: int,
    total: int,
    data: pd.DataFrame,
    strategy_info: Dict,
    fit_kwargs: Dict,
    breakpoints_dict: Dict,
    breakpoint_universe_func: Optional[Callable],
    dynamic_weights: bool,
    turnover: bool,
    save_idx: bool,
) -> pd.DataFrame:
    """
    Static function for parallel execution.

    This needs to be top-level for pickling.
    """
    # Reconstruct strategy
    breakpoints_for_npf = breakpoints_dict.get(npf, None)

    if strategy_info["type"] == "SingleSort":
        strat = SingleSort(
            holding_period=hp,
            sort_var=strategy_info["sort_var"],
            num_portfolios=npf,
            breakpoints=breakpoints_for_npf,
            lookback_period=strategy_info.get("lookback_period"),
            skip=strategy_info.get("skip"),
            breakpoint_universe_func=breakpoint_universe_func,
            verbose=False,
        )
    else:
        raise NotImplementedError(f"Parallel execution not supported for {strategy_info['type']}")

    # Run StrategyFormation
    sf = (
        StrategyFormation(
            data.copy(),
            strategy=strat,
            subset_filter=filt,
            rating=rating,
            dynamic_weights=dynamic_weights,
            turnover=turnover,
            save_idx=save_idx,
            verbose=False,
        )
        .fit(RETvar=retcol, **fit_kwargs)
    )

    # Package results (inline to avoid pickling issues)
    ew_ls, vw_ls = sf.get_long_short()
    ew_l, vw_l = sf.get_long_leg()
    ew_s, vw_s = sf.get_short_leg()

    # Turnover (only compute if enabled)
    if turnover:
        to_ew, to_vw = sf.get_ptf_turnover()
        ew_to = ((to_ew.iloc[:, -1] + to_ew.iloc[:, 0]) / 2).to_frame("TO")
        vw_to = ((to_vw.iloc[:, -1] + to_vw.iloc[:, 0]) / 2).to_frame("TO")
        ew_to_s = to_ew.iloc[:, 0].to_frame("TO")
        ew_to_l = to_ew.iloc[:, -1].to_frame("TO")
        vw_to_s = to_vw.iloc[:, 0].to_frame("TO")
        vw_to_l = to_vw.iloc[:, -1].to_frame("TO")
    else:
        # Create NaN turnover columns with same index as returns
        ew_to = pd.DataFrame({"TO": np.nan}, index=ew_ls.index)
        vw_to = pd.DataFrame({"TO": np.nan}, index=vw_ls.index)
        ew_to_s = ew_to.copy()
        ew_to_l = ew_to.copy()
        vw_to_s = vw_to.copy()
        vw_to_l = vw_to.copy()

    # Bond counts (only compute if save_idx enabled)
    if save_idx:
        bins_bonds = AssayAnomalyRunner._summarize_ranks(sf.get_ptf_bins())
        nbonds_s = bins_bonds["nbonds_s"].rename("nbonds")
        nbonds_l = bins_bonds["nbonds_l"].rename("nbonds")
        nbonds_ls = bins_bonds["nbonds_ls"].rename("nbonds")
    else:
        # Create NaN nbonds columns with same index as returns
        nbonds_s = pd.Series(np.nan, index=ew_ls.index, name="nbonds")
        nbonds_l = pd.Series(np.nan, index=ew_ls.index, name="nbonds")
        nbonds_ls = pd.Series(np.nan, index=ew_ls.index, name="nbonds")

    ew_ls_panel = pd.concat([ew_ls, ew_to, nbonds_ls], axis=1)
    vw_ls_panel = pd.concat([vw_ls, vw_to, nbonds_ls], axis=1)
    ew_l_panel = pd.concat([ew_l, ew_to_l, nbonds_l], axis=1)
    vw_l_panel = pd.concat([vw_l, vw_to_l, nbonds_l], axis=1)
    ew_s_panel = pd.concat([ew_s, ew_to_s, nbonds_s], axis=1)
    vw_s_panel = pd.concat([vw_s, vw_to_s, nbonds_s], axis=1)

    def _prep(df, weight, tp):
        out = df.copy()
        out = out.rename(columns={out.columns[0]: "ret"})
        return out.assign(weight=weight, type=tp)

    pack = pd.concat(
        [
            _prep(ew_ls_panel, "EW", "LS"),
            _prep(vw_ls_panel, "VW", "LS"),
            _prep(ew_l_panel, "EW", "L"),
            _prep(vw_l_panel, "VW", "L"),
            _prep(ew_s_panel, "EW", "S"),
            _prep(vw_s_panel, "VW", "S"),
        ],
        axis=0,
    )

    # Labels
    nport_label = {3: "T", 5: "Q", 10: "D"}.get(npf, str(npf))
    rating_val = rating if rating is not None else "ALL"
    ret_lab = retcol if retcol is not None else "ret"

    if not filt:
        subset_label = "ALL"
    else:
        kv = next(iter(filt.items()))
        low, high = kv[1]
        low_str = "inf" if math.isinf(low) else low
        high_str = "inf" if math.isinf(high) else high
        subset_label = f"{kv[0]}_{low_str}-{high_str}"

    base_id = f"HP_{hp}_NPF_{npf}_RAT_{rating_val}_{ret_lab}"
    pack["Holding"] = hp
    pack["Sort"] = nport_label
    pack["Rating"] = rating_val
    pack["Ret_string"] = ret_lab
    pack["Subset"] = subset_label
    pack["ID"] = (
        base_id + "_" + pack["weight"].astype(str) + "_" + pack["type"].astype(str)
    )

    return pack


# =============================================================================
# Results class
# =============================================================================
class AnomalyResults:
    """
    Wraps the concatenated panel of runs and computes summary results.
    """

    def __init__(self, runs: pd.DataFrame, params: Dict[str, Any]):
        self.runs = runs.copy()
        self.params = params
        self.results_df: Optional[pd.DataFrame] = None

        self._runs_plotter = PerformancePlotter(self.runs)
        self._results_plotter: Optional[PerformancePlotter] = None

    # --------------------------- basic accessors ------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        return self.runs.copy()

    @property
    def df(self) -> pd.DataFrame:
        return self.to_dataframe()

    # ------------------------------ helpers ----------------------------------
    @staticmethod
    def _align_dates_for_reg(
        ret: pd.Series, f: Union[pd.Series, pd.DataFrame]
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Align return series and factor(s) on common dates; drop non-overlap.
        """
        ret = ret.rename("ret")
        ret.index = pd.to_datetime(ret.index)
        f_df = f.to_frame() if isinstance(f, pd.Series) else f.copy()
        f_df.index = pd.to_datetime(f_df.index)

        common = ret.index.intersection(f_df.index)
        if common.empty:
            # nothing in common – return empty slices to signal skip
            return ret.iloc[:0], f_df.iloc[:0]

        y = ret.loc[common].dropna()
        X = f_df.loc[y.index].dropna()
        # Re-sync after dropna on X
        common2 = y.index.intersection(X.index)
        return y.loc[common2], X.loc[common2]

    # ------------------------------ stats ------------------------------------
    def process_results(
        self,
        factor: Optional[Union[pd.Series, pd.DataFrame]] = None,
        nw_lag: int = 3,
    ) -> pd.DataFrame:
        """
        For each (weight,type,Holding,Sort,Rating,Ret_string,Subset) group:
          - avg return + HAC t/p, Sharpe
          - turnover avg, #bonds avg/median
          - optional alpha (HAC) & IR if factor given
        """
        panel = self.runs
        keys = ["weight", "type", "Holding", "Sort", "Rating", "Ret_string", "Subset"]
        records = []

        for key_vals, grp in panel.groupby(keys):
            kv = dict(zip(keys, key_vals))
            y_raw = grp["ret"].dropna()

            # if no data, skip
            if y_raw.empty:
                continue

            # Average (intercept only) with HAC
            X0 = np.ones(len(y_raw))
            mod_avg = sm.OLS(y_raw, X0, missing="drop").fit(
                cov_type="HAC", cov_kwds={"maxlags": nw_lag}
            )

            avg = float(mod_avg.params.iloc[0])
            avg_t = float(mod_avg.tvalues.iloc[0])
            avg_p = float(mod_avg.pvalues.iloc[0])
            std_y = float(y_raw.std())
            sr_ann = (avg / std_y * np.sqrt(12)) if std_y > 0 else np.nan

            row = {
                **kv,
                "n_obs": int(len(y_raw)),
                "avg": avg,
                "avg_t": avg_t,
                "avg_p": avg_p,
                "std": std_y,
                "sr_annual": sr_ann,
                "avg_bonds": float(grp["nbonds"].dropna().mean())
                if "nbonds" in grp
                else np.nan,
                "med_bonds": float(grp["nbonds"].dropna().median())
                if "nbonds" in grp
                else np.nan,
                "avg_to": float(grp["TO"].dropna().mean()) if "TO" in grp else np.nan,
            }

            # Factor adjustment (alpha / IR) if provided and overlap exists
            if factor is not None:
                y_al, X_al = self._align_dates_for_reg(y_raw, factor)
                if len(y_al) >= 3 and len(X_al) == len(y_al):
                    mod_alpha = sm.OLS(y_al, sm.add_constant(X_al), missing="drop").fit(
                        cov_type="HAC", cov_kwds={"maxlags": nw_lag}
                    )
                    alpha = float(mod_alpha.params.iloc[0])
                    alpha_t = float(mod_alpha.tvalues.iloc[0])
                    alpha_p = float(mod_alpha.pvalues.iloc[0])
                    resid_std = float(mod_alpha.resid.std())
                    ir_ann = (alpha / resid_std * np.sqrt(12)) if resid_std > 0 else np.nan

                    row.update(
                        {"alpha": alpha, "alpha_t": alpha_t, "alpha_p": alpha_p, "ir_annual": ir_ann}
                    )
                else:
                    row.update({"alpha": np.nan, "alpha_t": np.nan, "alpha_p": np.nan, "ir_annual": np.nan})

            records.append(row)

        self.results_df = pd.DataFrame.from_records(records)
        return self.results_df.sort_values(by=["avg"], ascending=True)

    def summary_results(
        self,
        factor: Optional[Union[pd.Series, pd.DataFrame]] = None,
        nw_lag: int = 3,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return (full_results_df, recap_table_by_type).
        Recap includes:
          - avg_of_avg, mean/min/max alpha
          - % significant (|avg_t|>1.96, |alpha_t|>1.96) when available
        """
        df_result = self.process_results(factor=factor, nw_lag=nw_lag)

        agg_kwargs = {"avg_of_avg": ("avg", "mean")}
        if "alpha" in df_result:
            agg_kwargs.update(
                {"mean_alpha": ("alpha", "mean"), "min_alpha": ("alpha", "min"), "max_alpha": ("alpha", "max")}
            )
        if "avg_t" in df_result:
            agg_kwargs["pct_sig_avg_t"] = ("avg_t", lambda x: 100.0 * (x.abs() > 1.96).mean())
        if "alpha_t" in df_result:
            agg_kwargs["pct_sig_alpha_t"] = ("alpha_t", lambda x: 100.0 * (x.abs() > 1.96).mean())

        recap = df_result.groupby("type").agg(**agg_kwargs)
        return df_result, recap

    # ------------------------------ plotting ---------------------------------
    def plot_kde(self, *args, **kwargs):
        if self.results_df is None:
            self.process_results()
        if self._results_plotter is None:
            self._results_plotter = PerformancePlotter(self.results_df)
        return self._results_plotter.plot_kde(*args, **kwargs)

    def plot_bar_tstats(self, *args, **kwargs):
        if self.results_df is None:
            self.process_results()
        if self._results_plotter is None:
            self._results_plotter = PerformancePlotter(self.results_df)
        return self._results_plotter.plot_bar_tstats(*args, **kwargs)

    def plot_boxplot(self, *args, **kwargs):
        if self.results_df is None:
            self.process_results()
        if self._results_plotter is None:
            self._results_plotter = PerformancePlotter(self.results_df)
        return self._results_plotter.plot_boxplot(*args, **kwargs)

    def plot_cumulative_paths(self, *args, **kwargs):
        return self._runs_plotter.plot_cumulative_paths(*args, **kwargs)

    # ------------------------------ I/O --------------------------------------
    def save_csv(self, filepath: Optional[str] = None, include_sort_var_column: bool = True) -> None:
        """
        Save raw runs to CSV, writing a small metadata header first.
        """
        default_name = f"{self.params.get('sort_var')}_anomaly_results.csv"
        filename = filepath or default_name

        df = self.to_dataframe()
        if include_sort_var_column:
            df = df.copy()
            df.insert(0, "sort_var", self.params.get("sort_var"))

        with open(filename, "w") as f:
            f.write("# anomaly_assayer_export\n")
            for k, v in self.params.items():
                f.write(f"# {k}: {v}\n")

        df.to_csv(filename, mode="a", index=True)
        print(f"Anomaly results saved to {filename}")

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        index_col: Union[str, List[str], None] = None,
        parse_dates: Union[bool, List[str]] = True,
    ) -> "AnomalyResults":
        """
        Load raw runs from CSV (metadata unavailable).
        """
        df = pd.read_csv(csv_path, index_col=index_col, parse_dates=parse_dates)
        inst = cls.__new__(cls)  # bypass __init__
        inst.runs = df
        inst.params = {}
        inst.results_df = None
        inst._runs_plotter = PerformancePlotter(inst.runs)
        inst._results_plotter = None
        return inst


# =============================================================================
# Convenience facade (backwards-friendly)
# =============================================================================
def AssayAnomaly(
    data: pd.DataFrame,
    sort_var: str,
    IDvar: Optional[str] = None,
    DATEvar: Optional[str] = None,
    RETvar: Union[str, List[str], None] = None,  # allow multiple
    PRICEvar: Optional[str] = None,
    RATINGvar: Optional[str] = None,
    Wvar: Optional[str] = None,
    subset_filter: Optional[SubsetFilter] = None,
    holding_periods: Optional[List[int]] = None,
    nport: Optional[List[int]] = None,
    ratings: Optional[List[Any]] = None,
    dynamic_weights: bool = True,
    turnover: bool = True,
    save_idx: bool = True,
    breakpoint_universe_func: Optional[Callable] = None,
    verbose: bool = True,
) -> AnomalyResults:
    """
    1) Build a SingleSort(strategy) with the first HP/NPORT as a seed
    2) Spin up the Runner
    3) Run the full grid and return AnomalyResults
    """
    init_hp = (holding_periods or AssayAnomalyRunner.HOLDING_PERIODS)[0]
    init_npf = (nport or AssayAnomalyRunner.NPORT)[0]

    base_strategy = SingleSort(
        holding_period=init_hp,
        sort_var=sort_var,
        num_portfolios=init_npf,
        breakpoint_universe_func=breakpoint_universe_func,
        verbose=False,
    )

    runner = AssayAnomalyRunner(
        strategy=base_strategy,
        data=data,
        IDvar=IDvar,
        DATEvar=DATEvar,
        RETvar=RETvar,
        PRICEvar=PRICEvar,
        RATINGvar=RATINGvar,
        Wvar=Wvar,
        subset_filter=subset_filter,
        holding_periods=holding_periods,
        nport=nport,
        ratings=ratings,
        dynamic_weights=dynamic_weights,
        turnover=turnover,
        save_idx=save_idx,
        breakpoint_universe_func=breakpoint_universe_func,
        verbose=verbose,
    )
    return runner.run()
