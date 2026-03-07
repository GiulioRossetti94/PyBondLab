"""
03 - Within-Firm Portfolio Formation
=====================================
Demonstrates WithinFirmSort and BatchWithinFirmSortFormation for
exploiting variation across bonds issued by the same firm.

PyBondLab features used:
  - StrategyFormation                (main engine)
  - WithinFirmSort                   (within-firm sorting strategy)
  - BatchWithinFirmSortFormation     (batch processing of multiple signals)
  - firm_id_col                      (issuer identifier)
  - min_bonds_per_firm               (minimum bonds per firm)
  - rating_bins                      (rating tercile boundaries)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from _config import load_panel

from PyBondLab import StrategyFormation, WithinFirmSort, BatchWithinFirmSortFormation

data = load_panel()

# WithinFirmSort requires a firm identifier column
# In TRACE data, 'ticker' or a PERMNO-equivalent groups bonds by issuer
if 'ticker' not in data.columns:
    data['ticker'] = data['ID'].str[:6]

print(f"Panel: {len(data):,} obs, {data['ticker'].nunique():,} issuers")

# =============================================================================
# 1. Basic within-firm sort on credit spread
# =============================================================================
result_wf = StrategyFormation(
    data=data,
    strategy=WithinFirmSort(sort_var='cs', holding_period=1, firm_id_col='ticker',
                            min_bonds_per_firm=2, num_portfolios=2),
    dynamic_weights=True,
).fit()
ew_wf, vw_wf = result_wf.get_long_short()

print("\n-- Within-firm: credit spread --")
print(f"EW L-S:  mean={ew_wf.mean()*100:.3f}%/mo  "
      f"t={ew_wf.mean()/ew_wf.std()*np.sqrt(len(ew_wf)):.2f}")
print(f"VW L-S:  mean={vw_wf.mean()*100:.3f}%/mo  "
      f"t={vw_wf.mean()/vw_wf.std()*np.sqrt(len(vw_wf)):.2f}")

# =============================================================================
# 2. Within-firm sort with custom rating bins
# =============================================================================
result_wfr = StrategyFormation(
    data=data,
    strategy=WithinFirmSort(sort_var='tmat', holding_period=1, firm_id_col='ticker',
                            min_bonds_per_firm=2, rating_bins=[0, 7, 14, 22],
                            num_portfolios=2),
    dynamic_weights=True,
).fit()
ew_wfr, _ = result_wfr.get_long_short()
print(f"\n-- Within-firm: maturity (custom rating bins) --")
print(f"EW L-S:  mean={ew_wfr.mean()*100:.3f}%/mo  "
      f"t={ew_wfr.mean()/ew_wfr.std()*np.sqrt(len(ew_wfr)):.2f}")

# =============================================================================
# 3. Within-firm sort with characteristics tracking
# =============================================================================
result_wfc = StrategyFormation(
    data=data,
    strategy=WithinFirmSort(sort_var='dvol', holding_period=1, firm_id_col='ticker',
                            min_bonds_per_firm=2, num_portfolios=2),
    chars=['tmat', 'cs', 'dvol'], dynamic_weights=True,
).fit()
ew_wfc, _ = result_wfc.get_long_short()
_, vw_ch = result_wfc.get_characteristics()
print(f"\n-- Within-firm: downside vol (with chars) --")
print(f"EW L-S:  mean={ew_wfc.mean()*100:.3f}%/mo  "
      f"t={ew_wfc.mean()/ew_wfc.std()*np.sqrt(len(ew_wfc)):.2f}")
print(f"Chars tracked: {list(vw_ch.keys())}")

# =============================================================================
# 4. Batch within-firm processing (multiple signals)
# =============================================================================
wf_signals = ['cs', 'tmat', 'dvol', 'ami', 'mom6_1']
batch_wf = BatchWithinFirmSortFormation(
    data=data,
    signals=wf_signals,
    firm_id_col='ticker',
    min_bonds_per_firm=2,
    turnover=False,
    verbose=True,
)
batch_results = batch_wf.fit()

print(f"\n-- Batch within-firm ({len(wf_signals)} signals) --")
for sig in batch_results.successful_signals:
    ew, vw = batch_results[sig].get_long_short()
    t_ew = ew.mean() / ew.std() * np.sqrt(len(ew))
    print(f"  {sig:10s}  EW={ew.mean()*100:+.3f}%/mo (t={t_ew:+.2f})")

print("\n[Done] Script 03 complete.")
