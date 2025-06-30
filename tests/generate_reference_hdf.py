import pandas as pd
import numpy as np
import PyBondLab as pbl
import os
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = Path(os.getenv("DATA_PATH", "."))
DATA_FILE = "WRDS_MMN_Corrected_Data_2024_July.csv"
CSV_PATH = DATA_DIR / DATA_FILE

# HDF5 reference store\REF_DIR = Path("tests/reference")
REF_DIR = Path("tests/reference")
REF_DIR.mkdir(parents=True, exist_ok=True)
HDF_PATH = REF_DIR / "strategy_references.h5"

# parameter sets to baseline:
PARAMS = [
    {"holding_period": 1,  "n_portf": 5},
    {"holding_period": 3,  "n_portf": 5},
    {"holding_period": 6,  "n_portf": 5},
    {"holding_period": 1,  "n_portf": 10},
    {"holding_period": 3,  "n_portf": 10},
    {"holding_period": 6,  "n_portf": 10},
]

SORT_VARS = [
    'RATING_NUM',
    'CS',
    'CS_6M_DELTA',
    'tmt',
    'DURATION',
    'BOND_RET'
]
WEIGHT_MODES = [
    {"dynamic_weights": False, "label": "static"},
    {"dynamic_weights": True,  "label": "dynamic"},
]
CHAR_LIST = ['bond_yield', 'cs']

# ─── Data loader ──────────────────────────────────────────────────────────────
def load_data():
    tbl = pd.read_csv(CSV_PATH)
    tbl = tbl.reset_index(drop=True)
    tbl['index'] = range(1, len(tbl) + 1)
    tbl['date'] = pd.to_datetime(tbl['date'])
    tbl = tbl.sort_values(['cusip', 'date'])
    tbl = tbl[tbl['date'] >= "2002-08-31"]
    tbl['VW'] = tbl['BOND_VALUE']
    tbl.rename(columns={
        "BONDPRC": "PRICE",
        "cusip": "ID",
        "rating": "RATING_NUM",
        "bond_ret": "ret"
    }, inplace=True)
    return tbl

# ─── Reference generator (HDF) ─────────────────────────────────────────────────
def generate_all_references_hdf():
    data = load_data()
    with pd.HDFStore(HDF_PATH, mode='w') as store:
        for p in PARAMS:
            hp = p['holding_period']
            npf = p['n_portf']

            for mode in WEIGHT_MODES:
                dyn = mode['dynamic_weights']
                label = mode['label']
                suffix = f"hp{hp}_npf{npf}_{label}"

                # containers for each sort var
                for var in SORT_VARS:
                    strat = pbl.SingleSort(hp, var, npf, skip=0)
                    res = pbl.StrategyFormation(
                        data.copy(),
                        strategy=strat,
                        rating=None,
                        dynamic_weights=dyn,
                        turnover=True,
                        chars=CHAR_LIST
                    ).fit()

                    # long-short, legs, full
                    df_ls = pd.concat(res.get_long_short(), axis=1)
                    df_ls.columns = [f"EW_{var}", f"VW_{var}"]
                    store.put(f"LS/{suffix}/{var}", df_ls)

                    df_l = pd.concat(res.get_long_leg(), axis=1)
                    df_l.columns = [f"EW_{var}_Long", f"VW_{var}_Long"]
                    store.put(f"Long/{suffix}/{var}", df_l)

                    df_s = pd.concat(res.get_short_leg(), axis=1)
                    df_s.columns = [f"EW_{var}_Short", f"VW_{var}_Short"]
                    store.put(f"Short/{suffix}/{var}", df_s)

                    # full portfolio: rename to ensure unique columns
                    # df_all = pd.concat(res.get_ptf(), axis=1)
                    # df_all.columns = [f"EW_{var}_All", f"VW_{var}_All"]
                    # store.put(f"All/{suffix}/{var}", df_all)

                    # turnover only if dynamic
                    ew_turn, vw_turn = res.get_ptf_turnover()
                    store.put(f"Turnover/EW/{suffix}/{var}", ew_turn)
                    store.put(f"Turnover/VW/{suffix}/{var}", vw_turn)

                    print(f"Stored references for {suffix}, var={var}")

                    # turnover only if dynamic


                print(f"Stored references for {suffix} in HDF5")

if __name__ == "__main__":
    generate_all_references_hdf()
