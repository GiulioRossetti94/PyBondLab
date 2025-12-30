# DataUncertaintyAnalysis UX Analysis and Improvement Plan

## Current State Assessment

### What Works Well
1. **Fast path performance**: 289x faster than slow path
2. **All filter types work**: baseline, trim, price, bounce, wins
3. **Rating filtering**: IG, NIG, tuple ranges all work
4. **Summary output**: Clear per-config Newey-West t-stats
5. **Filter method**: Easy subsetting by signal/hp/filter_type
6. **Column naming**: Verbose but unambiguous (`signal1_hp1_trim_0.2`)

### UX Issues Identified

| Priority | Issue | Impact | Effort |
|----------|-------|--------|--------|
| **P1** | No summary aggregation by filter_type | High | Low |
| **P2** | No EA-EP spread in summary | Medium | Low |
| **P3** | wins filter EA shows NaN (confusing) | Medium | Low |
| **P4** | No factor "averages" across filter levels | High | Medium |
| **P5** | Export format not configurable | Low | Medium |
| **P6** | Verbose output could be cleaner | Low | Low |

## Detailed Issue Analysis

### P1: No Summary Aggregation by Filter Type

**Current behavior:**
```python
summary = results.summary()
# Returns: signal, hp, filter_type, level, ew_ea_mean, ...
# Each row is one configuration (e.g., signal1_hp1_trim_0.2)
```

**User pain point:**
"I want to see: on average, does trimming help vs baseline?"

**Current workaround:**
```python
summary.groupby('filter_type')[['ew_ea_mean', 'vw_ea_mean']].mean()
```

**Proposed solution:**
```python
# Option 1: Add aggregate parameter to summary()
results.summary(aggregate_by='filter_type')
# Returns: filter_type, ew_ea_mean, vw_ea_mean, ...

# Option 2: Add separate method
results.summary_by_filter()
```

### P2: No EA-EP Spread in Summary

**Current behavior:**
- Summary shows: ew_ea_mean, vw_ea_mean, ew_ep_mean, vw_ep_mean
- No EA-EP comparison

**User pain point:**
"Does ex-post filtering (EP) improve returns vs ex-ante (EA)?"

**Proposed solution:**
Add columns to summary():
- `ea_ep_diff_ew`: (ew_ep_mean - ew_ea_mean)
- `ea_ep_diff_vw`: (vw_ep_mean - vw_ea_mean)

### P3: Wins Filter EA Shows NaN

**Current behavior:**
```
filter_type   ew_ea_mean   ew_ep_mean
wins          NaN          0.032
```

**Why it happens:**
- Winsorization clips extreme returns, doesn't exclude bonds
- Rankings are identical to baseline
- Fast path sets EA=NaN to avoid duplicating baseline

**User confusion:**
"Why is EA NaN for wins? Is it broken?"

**Proposed solutions:**
1. Show same value as baseline EA (since ranking is identical)
2. Add footnote in summary docstring explaining behavior
3. Add `filter_type_notes` column with explanations

### P4: No Factor Averages Across Filter Levels

**User pain point:**
"I ran trim=[0.2, 0.3, 0.4, 0.5]. What's the average effect of trimming?"

**Current workaround:**
```python
summary = results.summary()
trim_only = summary[summary['filter_type'] == 'trim']
trim_avg = trim_only.groupby(['signal', 'hp']).mean()
```

**Proposed solution:**
```python
# New method: average_by_filter()
avg = results.average_by_filter()
# Returns: signal, hp, filter_type, avg_ew_ea, avg_vw_ea, ...
# One row per (signal, hp, filter_type) combination
```

### P5: Export Format Not Configurable

**Current behavior:**
`to_excel()` creates sheets: Summary, EW_EA, VW_EA, EW_EP, VW_EP, Configs

**User request:**
- Long format (all in one sheet, factor stacked)
- Wide format (current default)

**Proposed solution:**
```python
results.to_excel('output.xlsx', format='wide')  # Default (current)
results.to_excel('output.xlsx', format='long')  # Stacked format
```

### P6: Verbose Output Formatting

**Current behavior:**
```
DataUncertaintyAnalysis FAST PATH: 1 signals × 1 ratings × 18 filters × 2 HPs
  Signals: ['signal1']
  Ratings: [None]
  Holding periods: [1, 3]
    Filters applied in 0.00s
    Ranks computed in 0.00s
    ...
```

**Assessment:**
- Output is informative and helpful
- Fast path is so fast (<1s) that progress bars aren't needed
- Slow path shows [1/N] which is sufficient

**Recommendation:** No change needed. Current output is good.

## Implementation Plan

### Phase 1: Quick Wins (P1, P2, P3)

**Effort:** ~30 minutes

1. **Add EA-EP diff columns to summary()**
   - File: `data_uncertainty.py` line ~295
   - Add: `ea_ep_diff_ew = ew_ep_mean - ew_ea_mean`
   - Add: `ea_ep_diff_vw = vw_ep_mean - vw_ea_mean`

2. **Add summary docstring note about wins EA**
   - Explain why wins EA is NaN
   - Document that wins rankings = baseline rankings

3. **Add summary(aggregate_by=) parameter**
   - If aggregate_by='filter_type': group by filter_type and average
   - Returns aggregated DataFrame

### Phase 2: Average Methods (P4)

**Effort:** ~1 hour

1. **Add average_by_filter() method**
```python
def average_by_filter(self) -> pd.DataFrame:
    """Average statistics by (signal, hp, filter_type)."""
    summary = self.summary()
    groupby_cols = ['signal', 'hp', 'filter_type']
    if 'rating' in summary.columns:
        groupby_cols.append('rating')

    numeric_cols = ['ew_ea_mean', 'vw_ea_mean', 'ew_ep_mean', 'vw_ep_mean',
                    'sharpe', 'n_obs']

    return summary.groupby(groupby_cols)[numeric_cols].mean().reset_index()
```

### Phase 3: Export Improvements (P5)

**Effort:** ~1 hour

1. **Add format parameter to to_excel()**
```python
def to_excel(self, path: str, format: str = 'wide'):
    if format == 'long':
        # Stack all panels into single DataFrame
        # Columns: date, factor, weighting, strategy (ea/ep), return
        ...
    else:
        # Current behavior (wide format)
        ...
```

## Recommendation

**Implement Phase 1 (Quick Wins) immediately:**
- Add EA-EP diff columns ✓
- Document wins EA behavior ✓
- Add aggregate_by parameter ✓

**Consider Phase 2 and 3 based on user feedback.**

## Validation

After implementation, run:
```bash
python examples/diagnose_data_uncertainty.py
```

All tests should still pass, plus new features should be visible in output.
