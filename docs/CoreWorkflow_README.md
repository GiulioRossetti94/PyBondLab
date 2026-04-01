# Research Workflow Guide

This guide is a package-level overview for researchers who want to understand how PyBondLab is organized and how its public APIs relate to one another.

It does not replace the detailed feature documentation. Its purpose is different:

- define the package's public workflow
- clarify which APIs address which tasks
- explain the main semantic choices that affect outputs
- provide a consistent map from basic portfolio formation to larger empirical workflows

This guide is intentionally research-agnostic. It does not recommend any specific empirical design, signal definition, breakpoint convention, weighting scheme, subsample, or robustness protocol beyond describing what the package implements.

---

## 1. What PyBondLab Is

PyBondLab is a Python package for portfolio formation and related robustness analysis in panel data settings, with a focus on bond-level empirical asset pricing workflows.

At its core, the package does four things:

1. Defines sorting strategies such as single sorts, double sorts, momentum-based sorts, long-term reversal sorts, and within-firm sorts.
2. Forms portfolios through a common engine and computes equal-weighted and value-weighted returns.
3. Exposes optional diagnostics such as turnover, bond counts, and portfolio characteristics.
4. Extends the core workflow to batch processing, rolling beta estimation, data-cleaning robustness, and specification-curve style anomaly assaying.

---

## 2. The Main Public APIs

The public APIs are easiest to understand if they are grouped into layers.

### Core portfolio formation layer

- `SingleSort`
- `DoubleSort`
- `Momentum`
- `LTreversal`
- `WithinFirmSort`
- `StrategyFormation`
- result accessors such as `get_long_short()`, `get_ptf()`, `get_turnover()`, `get_bond_count()`, and `get_characteristics()`

This is the main portfolio-formation layer.

### Batch processing layer

- `BatchStrategyFormation`
- `BatchWithinFirmSortFormation`
- `extract_panel`

This layer is for running the same formation logic across many signals.

### Supporting analysis layer

- `PreAnalysisStats`
- `RollingBeta`
- `NamingConfig`

These tools support the core workflow but do not replace it.

### Robustness and multiverse layer

- `DataUncertaintyAnalysis`
- `assay_anomaly_fast`
- `BatchAssayAnomaly`
- `AssayAnomaly`

These tools sit on top of the core formation logic and address robustness questions rather than the basic formation problem itself.

---

## 3. A Consistent Order of Use

One consistent way to move through the package is:

1. Validate and inspect the panel.
2. Run a simple single-sort strategy through `StrategyFormation`.
3. Add turnover, characteristics, and bond counts only after the basic returns are understood.
4. Move to double sorts, non-monthly rebalancing, breakpoint universes, or within-firm sorts if needed.
5. Use batch APIs once the single-run workflow is clear.
6. Use `RollingBeta` only when the estimated beta itself is the input to a later sort.
7. Use `DataUncertaintyAnalysis` and anomaly-assay APIs only after the core formation semantics are clear.

This is not methodological advice. It is an interface-oriented ordering that makes the relationship between the APIs easier to see.

---

## 4. Required Input Structure

For the core portfolio formation engine, the panel currently needs these columns:

| Column | Required | Meaning |
|---|---|---|
| `date` | Yes | Observation date |
| `ID` | Yes | Bond identifier |
| `ret` | Yes | Return used for portfolio returns |
| `VW` | Yes | Value-weight input for VW portfolios |
| `RATING_NUM` | Yes | Numeric rating used by core validation and rating-aware workflows |
| `PRICE` | No | Only required for price-based filters |

If your columns use different names, map them in `.fit(...)`:

```python
result = pbl.StrategyFormation(data, strategy=strategy).fit(
    IDvar='cusip',
    RETvar='ret_vw',
    VWvar='mcap_e',
    RATINGvar='spc_rat',
    PRICEvar='prc_eom',
)
```

The package can also use additional columns:

- sort variables such as `cs`, `duration`, `momentum`, or any user-defined signal
- subset filter columns
- characteristic tracking columns passed through `chars=[...]`
- firm identifiers for `WithinFirmSort`
- factor columns used later by `RollingBeta`

This guide does not assume any particular signal naming convention or panel construction protocol.

---

## 5. Basic Formation Workflow

A basic formation workflow is a single sort with `StrategyFormation`.

```python
import PyBondLab as pbl

strategy = pbl.SingleSort(
    sort_var='cs',
    holding_period=1,
    num_portfolios=5,
)

result = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    turnover=True,
    chars=['tmat'],
).fit()

ew_ls, vw_ls = result.get_long_short()
ew_ptf, vw_ptf = result.get_ptf()
ew_turn, vw_turn = result.get_turnover()
counts = result.get_bond_count()
ew_chars, vw_chars = result.get_characteristics()
```

What each object is doing:

- `SingleSort(...)` defines the ranking rule.
- `StrategyFormation(...)` defines how the strategy is executed on the panel.
- `.fit()` performs column mapping, preparation, sorting, return formation, and optional diagnostics.
- `result` is the main object you inspect afterward.

This workflow is useful because many other parts of the package build on the same execution model.

---

## 6. How To Think About the Strategy Classes

The strategy classes define **what is being sorted** and some key formation semantics.

### `SingleSort`

Use when one characteristic determines the portfolio ranking.

Typical inputs:

- `sort_var`
- `holding_period`
- `num_portfolios`
- optional custom `breakpoints`
- optional `rebalance_frequency`
- optional breakpoint-universe functions

### `DoubleSort`

Use when two characteristics define the portfolio assignment.

### `Momentum` and `LTreversal`

Use when the signal itself is constructed from the return history rather than being pre-computed in the panel.

These strategies still run through `StrategyFormation`, but the signal is computed within the strategy rather than read directly from a pre-existing panel column.

### `WithinFirmSort`

Use when the sort is defined within issuer rather than across the full cross-section.
(See Dick-Nielsen et al. (2025))

---

## 7. The Central Execution Object: `StrategyFormation`

`StrategyFormation` is the execution engine. It is the main public class for translating a strategy definition plus panel data into portfolio outputs.

Conceptually it does five things:

1. validates and standardizes the input data
2. computes or reads the signal used for sorting
3. assigns portfolios at formation dates
4. computes returns and optional diagnostics
5. packages the outputs into a consistent result object

Most of the package's formation semantics live here, even when they are later exposed through batch or robustness APIs.

---

## 8. The Most Important Semantic Distinctions

These distinctions matter because they affect how the outputs should be interpreted.

### 8.1 Monthly `holding_period > 1` is staggered rebalancing

If `rebalance_frequency='monthly'` and `holding_period > 1`, PyBondLab uses overlapping cohorts.

This means:

- portfolio formation still occurs monthly
- multiple cohorts can be active at the same time
- reported returns are based on cohort aggregation

This is not the same thing as quarterly or annual rebalancing.

See:

- [SingleSort and DoubleSort documentation](SingleSort_DoubleSort_README.md)
- [Non-staggered rebalancing documentation](NonStaggeredRebalancing_README.md)

### 8.2 Non-monthly rebalancing is controlled by `rebalance_frequency`

If `rebalance_frequency` is quarterly, semi-annual, annual, or a custom integer frequency:

- the package uses non-monthly rebalancing logic
- in this mode, `holding_period` must be `1`
- the effective calendar holding interval is determined by `rebalance_frequency`

So there are two distinct ways to leave the simple monthly HP=1 case:

- staggered monthly multi-period holding
- non-monthly rebalancing

They are not interchangeable.

See:

- [Non-staggered rebalancing documentation](NonStaggeredRebalancing_README.md)
- [Batch strategy documentation](BatchStrategyFormation_README.md)

### 8.3 `dynamic_weights` only matters for staggered HP>1 workflows

For `holding_period == 1`, `dynamic_weights=True` and `False` do not change the VW result.

The parameter matters only when staggered multi-period holding is active.

See:

- [The `dynamic_weights` discussion in the standard sorting documentation](SingleSort_DoubleSort_README.md#the-dynamic_weights-parameter)
- [Non-staggered rebalancing note on `dynamic_weights`](NonStaggeredRebalancing_README.md#important-dynamic_weights-does-not-apply-to-non-staggered)

### 8.4 Fast batch results are not full formation results

When `BatchStrategyFormation` uses the fast path, the result is intentionally reduced.

That means:

- long-short returns are available
- full portfolio legs are not
- turnover is not
- bond counts are not
- characteristics are not
- `extract_panel()` is not appropriate for that reduced output

This is a result-tier distinction rather than an error condition.

See:

- [Batch strategy documentation](BatchStrategyFormation_README.md#unified-panel-extraction-with-extract_panel)
- [API semantic appendix](0_API_semantic_README.md)

### 8.5 `WithinFirmSort` is a distinct workflow

`WithinFirmSort` should not be treated as a cosmetic variant of a standard sort.

It has its own aggregation logic and currently supports `holding_period=1` only.

See:

- [WithinFirmSort documentation](WithinFirmSort_README.md)

---

## 9. Understanding Result Types

There are different result tiers in the package. Keeping them separate helps avoid confusion about which outputs are available in which workflow.

### Full formation results

Typical source:

- `StrategyFormation`
- batch runs with `turnover=True`
- batch runs with `chars=[...]`
- runs where banding or other options force the full path

Typical capabilities:

- `get_long_short()`
- `get_ptf()`
- `get_turnover()`
- `get_bond_count()`
- `get_characteristics()`

### Fast batch results

Typical source:

- `BatchStrategyFormation(..., turnover=False, chars=None, banding=None)`

Typical capability:

- `get_long_short()`

This is a reduced result tier intended for screening and large-scale fast runs.

For the detailed batch result contract and `extract_panel()` requirements, see [BatchStrategyFormation_README.md](BatchStrategyFormation_README.md).

### Robustness-analysis results

Typical source:

- `DataUncertaintyAnalysis`
- anomaly-assay APIs

These objects are not just lighter or heavier versions of `StrategyFormation` results. They expose aggregated robustness outputs for a different purpose.

---

## 10. When To Use Batch APIs

The batch APIs extend the same formation logic to many signals.

### `BatchStrategyFormation`

Typical use:

- you want to run many signals through the same formation settings
- you understand what the corresponding single-run call would be
- you know whether you need full results or long-short screening only

See [BatchStrategyFormation_README.md](BatchStrategyFormation_README.md) for the full result-tier distinction and panel-extraction workflow.

### `BatchWithinFirmSortFormation`

Typical use:

- the within-firm workflow itself is already clear
- you want to apply it across many signals

The batch APIs are scale-up interfaces for the same underlying formation logic.

See [WithinFirmSort_README.md](WithinFirmSort_README.md) for the underlying within-firm formation logic.

---

## 11. When To Use the Robustness APIs

PyBondLab has two major robustness layers.

### `DataUncertaintyAnalysis`

Use when the relevant question is:

- how sensitive are factor returns to data-cleaning choices?

This is the package's filter-robustness layer. It focuses on trimming, winsorization, price filters, bounce filters, and related return-cleaning choices.

See [DataUncertaintyAnalysis_README.md](DataUncertaintyAnalysis_README.md) for the supported dimensions and result structure.

### Anomaly-assay APIs

Use when the relevant question is:

- how sensitive are factor returns to specification choices?

The public entry points are:

- `assay_anomaly_fast` for one signal, speed-first
- `BatchAssayAnomaly` for many signals, speed-first
- `AssayAnomaly` for the slower, richer workflow

`AssayAnomalyRunner` exists for advanced/internal control and is not part of the default public path described in this guide.

These tools are best understood as extensions of the core formation workflow rather than substitutes for it.

See:

- [Anomaly assay documentation](AnomalyAssay_README.md)
- [Batch anomaly assay documentation](BatchAssayAnomaly_README.md)

---

## 12. Suggested Package Learning Path

For users who want a single package-level map, the following sequence is one consistent way to move through the package:

1. `PreAnalysisStats`
   Use it to inspect the panel before portfolio formation.

2. `SingleSort` + `StrategyFormation`
   Use this to learn the basic formation model and result accessors.

3. `DoubleSort`
   Use it once the single-sort workflow is stable.

4. Optional formation features
   Add turnover, characteristics, custom breakpoints, subset filters, and non-monthly rebalancing only after the basic workflow is already clear.

5. `BatchStrategyFormation`
   Use it to scale the same logic across many signals.

6. `RollingBeta`
   Use it only when beta estimation is part of the signal-generation pipeline.

7. `DataUncertaintyAnalysis`
   Use it for filter-robustness questions.

8. Anomaly-assay APIs
   Use them for specification-robustness questions.

9. `WithinFirmSort`
   Use it when the within-issuer design itself is part of the empirical question.

This is not a ranking of importance. It is a sequence intended to minimize interface ambiguity.

---

## 13. Common Sources of User Error

The most common sources of confusion are semantic rather than syntactic.

### Confusing staggered HP>1 with non-monthly rebalancing

These are different execution models.

Relevant detail:

- [Non-staggered rebalancing documentation](NonStaggeredRebalancing_README.md)
- [Standard sorting documentation](SingleSort_DoubleSort_README.md)

### Expecting fast batch results to behave like full results

They do not. Fast batch results are intentionally reduced.

Relevant detail:

- [Batch strategy documentation](BatchStrategyFormation_README.md)
- [API semantic appendix](0_API_semantic_README.md)

### Assuming every advanced module is a starting point

They are not. Most advanced modules are extensions of the core formation model.

Relevant detail:

- [Anomaly assay documentation](AnomalyAssay_README.md)
- [Data uncertainty documentation](DataUncertaintyAnalysis_README.md)

### Treating result objects from different layers as interchangeable

They are not. Core formation results, batch fast results, data-uncertainty results, and anomaly-assay results serve different purposes.

### Reading detailed feature docs before establishing the core model

The detailed docs are useful, but they are easier to navigate once the package-level workflow and result types are clear.

---

## 14. Where To Go Next

Once the workflow in this guide is clear, use the detailed docs for specific tasks:

- Standard sorting strategies:
  `docs/SingleSort_DoubleSort_README.md`

- Batch processing and result tiers:
  `docs/BatchStrategyFormation_README.md`

- Within-firm methodology:
  `docs/WithinFirmSort_README.md`

- Non-monthly rebalancing details:
  `docs/NonStaggeredRebalancing_README.md`

- Data-cleaning robustness:
  `docs/DataUncertaintyAnalysis_README.md`

- Specification-curve robustness:
  `docs/AnomalyAssay_README.md`

- Pre-analysis statistics:
  `docs/PreAnalysisStats_README.md`

- Rolling beta estimation:
  `docs/RollingBeta_README.md`

This guide is intended as a package-level map. The other docs provide feature-level detail.
