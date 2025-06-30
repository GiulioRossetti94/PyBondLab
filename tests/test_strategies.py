import pandas as pd
import numpy as np
import PyBondLab as pbl
import os
from pathlib import Path
import pytest

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_DIR = Path(os.getenv("DATA_PATH", "."))
DATA_FILE = "WRDS_MMN_Corrected_Data_2024_July.csv"
CSV_PATH = DATA_DIR / DATA_FILE
REF_DIR = Path("tests/reference")
HDF_PATH = REF_DIR / "strategy_references.h5"

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

# ─── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def bond_data():
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

@pytest.fixture(scope="session")
def reference_store():
    with pd.HDFStore(HDF_PATH, mode='r') as store:
        yield store

# ─── Helper ────────────────────────────────────────────────────────────────────
def compare_df(df1, df2, atol=1e-8, rtol=1e-5):
    pd.testing.assert_frame_equal(
        df1.sort_index(axis=1),
        df2.sort_index(axis=1),
        check_exact=False,
        rtol=rtol,
        atol=atol
    )

# ─── Parameterization ──────────────────────────────────────────────────────────
param_combos = []
for p in PARAMS:
    for mode in WEIGHT_MODES:
        for var in SORT_VARS:
            param_combos.append((
                p['holding_period'],
                p['n_portf'],
                mode['dynamic_weights'],
                mode['label'],
                var
            ))

@pytest.mark.parametrize(
    "holding_period,n_portf,dynamic_weights,label,sort_var",
    param_combos
)
def test_strategy_combination(
    bond_data, reference_store,
    holding_period, n_portf, dynamic_weights, label, sort_var
):
    suffix = f"hp{holding_period}_npf{n_portf}_{label}"

    strat = pbl.SingleSort(holding_period, sort_var, n_portf, skip=0)
    res = pbl.StrategyFormation(
        bond_data.copy(),
        strategy=strat,
        rating=None,
        dynamic_weights=dynamic_weights,
        turnover=True,
        chars=CHAR_LIST
    ).fit()

    # Long-short
    df_ls = pd.concat(res.get_long_short(), axis=1)
    df_ls.columns = [f"EW_{sort_var}", f"VW_{sort_var}"]
    ref_ls = reference_store.get(f"LS/{suffix}/{sort_var}")
    compare_df(df_ls, ref_ls)

    # Long leg
    df_l = pd.concat(res.get_long_leg(), axis=1)
    df_l.columns = [f"EW_{sort_var}_Long", f"VW_{sort_var}_Long"]
    ref_l = reference_store.get(f"Long/{suffix}/{sort_var}")
    compare_df(df_l, ref_l)

    # Short leg
    df_s = pd.concat(res.get_short_leg(), axis=1)
    df_s.columns = [f"EW_{sort_var}_Short", f"VW_{sort_var}_Short"]
    ref_s = reference_store.get(f"Short/{suffix}/{sort_var}")
    compare_df(df_s, ref_s)

    # Turnover
    ew_turn, vw_turn = res.get_ptf_turnover()
    ref_ew_turn = reference_store.get(f"Turnover/EW/{suffix}/{sort_var}")
    ref_vw_turn = reference_store.get(f"Turnover/VW/{suffix}/{sort_var}")
    compare_df(ew_turn, ref_ew_turn)
    compare_df(vw_turn, ref_vw_turn)
