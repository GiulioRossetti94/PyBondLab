# API Semantic Notes

Some PyBondLab parameters are conditionally active, silently ignored, or behave differently depending on other settings. This page documents known interactions to help avoid surprises.

## Parameter Interactions

| Object | Parameter | Condition | Behavior |
|--------|-----------|-----------|----------|
| `StrategyFormation` / `BatchStrategyFormation` | `dynamic_weights` | `holding_period == 1` | No effect — VW results are identical for `True` and `False`. Only matters for `holding_period > 1`. |
| `SingleSort` / `DoubleSort` | `holding_period` | `rebalance_frequency != 'monthly'` | Omit it — defaults to `1`. The actual calendar holding period is determined by `rebalance_frequency` (e.g., quarterly = 3 months). Passing any value other than `1` raises `ValueError`. |
| `SingleSort` / `DoubleSort` | `rebalance_month` | `rebalance_frequency = 'monthly'` | Silently ignored. Only controls rebalancing months for quarterly, semi-annual, and annual frequencies. |
| `BatchStrategyFormation` | `banding` (int) | Always | Accepts an integer (e.g., `banding=1`) and converts internally to `banding_threshold = banding / num_portfolios`. |
| `StrategyFormation` | `banding_threshold` (float) | Always | Accepts a float directly (e.g., `banding_threshold=0.2`). Do not confuse with `BatchStrategyFormation.banding`. |
| `BatchStrategyFormation` | `turnover=False` | `extract_panel()` called | `extract_panel()` requires full portfolio results. Set `turnover=True` or `chars=[...]` before calling `extract_panel()`. |
| `WithinFirmSort` | `num_portfolios` | Any value other than `2` | Coerced to `2` with a warning. WithinFirmSort always creates exactly 2 portfolios (High/Low). |
| `StrategyFormation` + `WithinFirmSort` | `save_idx` | Always | Overridden to `True` regardless of what you pass. Required for hierarchical aggregation. |
| `DataUncertaintyAnalysis` | `rating` vs `ratings` | Both provided | `ratings` (list) takes precedence silently. Prefer `ratings=` for all new code. |
| `DataUncertaintyAnalysis` | `use_fast_path` | `rebalance_frequency != 'monthly'`, or `no_gap=True`, or `fill_na=True` | Silently falls back to the slow pandas path. |
| `DataUncertaintyResults` | `.summary()` EA columns | `wins` filter | EA statistics are `NaN` because winsorization does not change portfolio rankings — only EP returns reflect the winsorization effect. |

For the full audit, see `release_audit/api_semantics_analysis.md`.
