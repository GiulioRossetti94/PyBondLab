# API Semantic Notes

This page is a reference appendix for public API semantics that change with context. It is not a tutorial. The goal is to state where parameters are active, where they are ignored, where the package forces a behavior, and what that means for the returned objects.

The highest-risk caveats are surfaced in the main README and the workflow guide. This page is the compact contract sheet.

## Contract Matrix

| Parameter / Object | Active when | Ignored when | Forced when | Output consequence |
|---|---|---|---|---|
| `dynamic_weights` (`StrategyFormation`, `BatchStrategyFormation`, `DataUncertaintyAnalysis`) | VW portfolio formation with monthly staggered rebalancing and `holding_period > 1` | `holding_period == 1`; any EW result; characteristic averages that do not depend on VW reconstitution | Never forced by the package | Changes VW rebalancing semantics for overlapping cohorts: `True` uses buy-and-hold style VW updates from `t-1`; `False` uses formation-date VW weights across the holding window. No effect on HP=1 results. |
| `save_idx` (`StrategyFormation`) | When you need saved portfolio membership indices for later extraction or diagnostics | Fast-return-only paths where no portfolio indices are materialized; runs where you only need long-short returns | Forced to `True` for `WithinFirmSort` because hierarchical aggregation requires stored membership | Enables saved portfolio membership in the result object. Required for `extract_panel()` and any downstream operation that needs bond-level portfolio assignment after formation. |
| `turnover` / `compute_turnover` (`StrategyFormation`, `BatchStrategyFormation`) | When turnover diagnostics are requested | Never silently ignored at the formation layer, but irrelevant if you only inspect long-short returns | Not forced by default; some workflows achieve full-path behavior by setting `turnover=True` or requesting `chars=[...]` | Moves the run onto a full portfolio-construction path and returns turnover diagnostics. On batch runs, this is one way to ensure results contain enough structure for panel extraction instead of reduced fast-batch output. |
| `holding_period` (`SingleSort`, `DoubleSort`) | Monthly rebalancing (`rebalance_frequency='monthly'`) where it controls staggered overlapping cohorts | Non-monthly rebalancing as a timing control; in that case it does not determine the calendar holding interval | Forced to `1` by validation for non-monthly rebalancing; `WithinFirmSort` also restricts it to `1` | Under monthly rebalancing, HP>1 creates overlapping cohorts and makes `dynamic_weights` potentially relevant for VW returns. Under non-monthly rebalancing, the calendar interval is determined by `rebalance_frequency`, not by `holding_period`. |
| `rebalance_frequency` (`SingleSort`, `DoubleSort`, `WithinFirmSort`) | Any non-default rebalancing schedule: quarterly, semi-annual, annual, or integer month interval | Never ignored as a scheduling parameter; only ancillary controls such as `rebalance_month` become irrelevant under monthly rebalancing | Never forced globally; strategy validation may reject incompatible combinations such as non-monthly rebalancing with `holding_period != 1` | Changes the formation calendar from monthly staggered rebalancing to non-staggered periodic rebalancing. This changes the timing model, the interpretation of HP, and the set of compatible constructor arguments. |
| `rebalance_month` (`SingleSort`, `DoubleSort`, `WithinFirmSort`) | Non-monthly rebalancing where a specific calendar month anchor matters | `rebalance_frequency='monthly'` | Never forced | Determines which month is used for quarterly, semi-annual, or annual portfolio formation. No effect in monthly rebalancing. |
| Full results vs fast batch results (`BatchStrategyFormation`) | Full results are returned when the batch run needs turnover, characteristics, or saved portfolio structure | Reduced fast-batch results are returned when the run qualifies for the fast path and only long-short output is needed | The package implicitly selects the reduced fast-batch contract when fast-path eligibility holds and no full-path features are requested | Full results expose per-signal `FormationResults` accessors such as turnover, characteristics, and bond counts. Reduced fast-batch results expose long-short return series only and are marked with `is_fast_batch_result=True`. |

## Notes by Topic

### `dynamic_weights`

`dynamic_weights` is not a general "better vs worse" option. It only changes VW portfolio evolution when monthly staggered rebalancing produces overlapping holding cohorts. For HP=1, both settings collapse to the same VW timing convention and therefore produce identical results.

### `save_idx`

`save_idx` should be read as "persist portfolio membership," not as a generic diagnostics switch. It is separate from `turnover`. A run can compute turnover without necessarily exposing saved membership indices, and a run can save membership even if turnover is not requested.

### `turnover`

`turnover=True` is primarily a request for an additional diagnostic output, but it also matters operationally because it forces the package onto a full portfolio-construction path. That makes it relevant when users later want panel extraction or portfolio-level diagnostics rather than only factor returns.

### `holding_period` and `rebalance_frequency`

These two parameters should be interpreted together.

- Monthly rebalancing:
  `holding_period` controls the number of overlapping cohorts.
- Non-monthly rebalancing:
  `rebalance_frequency` controls the calendar spacing, and `holding_period` must be `1`.

This is one of the main semantic traps in the package because the same `holding_period` argument changes meaning once the rebalancing model changes.

### Fast vs full result tiers

The package has more than one result contract.

- Full formation results:
  used by `StrategyFormation` and by batch runs that require full diagnostics.
- Reduced fast-batch results:
  used when batch formation only needs long-short return output and the fast path is eligible.

Do not assume that a fast-batch result supports the same accessors as a full `FormationResults` object.
