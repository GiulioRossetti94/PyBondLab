# PyBondLab
PyBondLab is a Python module designed for portfolio sorting and other tools tailored for empirical asset pricing research, with a particular focus on corporate bonds. It is part of the [Open Source Bond Asset Pricing project](https://openbondassetpricing.com/).

## Overview
PyBondLab provides tools for computing and evaluating investment strategies. It features look-ahead bias free data cleaning procedures to ensure the integrity and reliability of empirical results.

## Installation
You can install the latest release using pip
```bash
pip install PyBondLab
```

Or you can install the master branch from GitHub by cloning the repo and running setup

```bash
git clone https://github.com/GiulioRossetti94/PyBondLab.git
cd PyBondLab
python setup.py install
```

## Usage & Examples
For a more complete overview, see notebooks and scripts in [example](https://github.com/GiulioRossetti94/PyBondLab/tree/main/examples)

### Portfolio sorting
This example demonstrates how to implement a long-short investment strategy using the `PyBondLab` module, based on quintile sorting of corporate bond on credit ratings.

At each month $t$, corporate bonds are sorted into five portfolios according to their credit rating (`RATING_NUM` column). 
The strategy involves:

1. **Long Position**: buy the bonds in the quintile portfolio containing bonds with the lowest credit rating (i.e., long bonds with higher default risk)
2. **Short Position**: sell (short-sell) the bonds in the quintile portfolio containing bonds with the highest credit rating (i.e., short/sell bonds with low default risk)

```python
import PyBondLab as pbl
import pandas as pd
import numpy as np
import wrds

# read bond dataset
# use a pandas DataFrame with columns:
# "RATING_NUM": the credit rating of bonds
# "ISSUE_ID": the identifier for each bond
# "date": the date
# "RET_L5M": bond returns used to compute portfolio returns
# "VW":  weights used to compute value-weighted performnace

# Read in data directly from WRDS
# Assumes you have a valid WRDS account and have set-up your cloud access #
# See:
# https://wrds-www.wharton.upenn.edu/pages/support/programming-wrds/programming-python/python-wrds-cloud/
wrds_username = '' # Input your WRDS username
db = wrds.Connection(wrds_username = wrds_username )

tbl1 = db.raw_sql("""SELECT  DATE, ISSUE_ID,CUSIP, RATING_NUM, RET_L5M,AMOUNT_OUTSTANDING,
                                TMT, N_SP, PRICE_L5M                         
                        FROM wrdsapps.bondret
                  """)
                  
# Required because the WRDS data comes with "duplicates" in the index
# does not affect data, but the "index" needs to be re-defined #                 
tbl1 = tbl1.reset_index()
tbl1['index'] = range(1,(len(tbl1)+1))

# Format the data
tbl1.columns = tbl1.columns.str.upper()
tbl1['date'] = pd.to_datetime(tbl1['DATE'])
tbl1['AMOUNT_OUTSTANDING'] = np.abs(tbl1['AMOUNT_OUTSTANDING'])
tbl1['PRICE_L5M'] = np.abs(tbl1['PRICE_L5M'])
tbl1 = tbl1.sort_values(['ISSUE_ID','DATE'])

# WRDS data "starts" officially on "2002-08-31"
tbl1 = tbl1[tbl1['date'] >= "2002-08-31"]

# Column used for value weigted returns
tbl1['VW'] = (tbl1['PRICE_L5M'] * tbl1['AMOUNT_OUTSTANDING'])/1000

holding_period = 1             # holding period returns
n_portf        = 5             # number of portfolios
sort_var1      = 'RATING_NUM'  # sorting characteristic/variable

# Initialize the single sort strategy
single_sort = pbl.SingleSort(holding_period, sort_var1, n_portf)

# Define a dictionary with strategy and optional parameters
params = {'strategy': single_sort,
          'rating':None,
  }

# Fit the strategy to the data. Specify ID identifier and column of returns 
res = pbl.StrategyFormation(tbl1, **params).fit(IDvar = "ISSUE_ID",RETvar = "RET_L5M")

# Get long-short portfolios (equal- and value-weighted)
ew,vw = res.get_long_short()
```
### Data cleaning / filtering options
Currently, the package provides functionality for four different data cleaning procedures routinely applied in corporate bonds research.
Data cleaning functionalities are passed with a dictionary `{'adj':"{adjustments}","level":{level}, **options}`

#### Return exclusion (trimming)
Filtering out bonds whose returns are above/below a certain threshold level:
- `{'adj':'trim,'level':0.2}` excludes returns > 20%
- `{'adj':'trim,'level':-0.2}` excludes returns < 20%
- `{'adj':'trim,'level':[-0.2,0.2]}` excludes returns < -20% and returns >20%

#### Price exclusion
Filtering out bonds whose prices are above/below a certain threshold level. When constructing the strategy, either rename a column to `"PRICE"` or specify `PRICEvar` in the `.fit()` method.

- `{'adj':'price,'level':150}` excludes prices > 150
- `{'adj':'price,'level':20}` excludes prices < 150
- `{'adj':'price,'level':[20, 150]}` excludes prices < 20 and prices > 150

By default, the threshold for below/above exclusion is set to 25 (by default `'price_threshold':25`).
This implies that if `'level':26` is passed, bonds whose price is above 26 are excluded. To specify a different threshold, include `'price_threshold'` in the dictionary.

- `{'adj':'price,'level':26}` excludes prices >26
- `{'adj':'price,'level':26,'price_threshold':30}` excludes prices <26

#### Return bounce-back exclusion
Filtering out bonds where the product of their monthly returns $R_t \times R_{t-1}$ meets a pre-defined threshold.

- `{'adj':'bounce,'level':0.01}` excludes if $R_t \times R_{t-1}$ > 0.01
- `{'adj':'bounce,'level':-0.01}` excludes if $R_t \times R_{t-1}$ < 0.01 
- `{'adj':'bounce,'level':[-0.01, 0.01]}` excludes if $R_t \times R_{t-1}$ <0.01 and  $R_t \times R_{t-1}$ >0.01

#### Return winsorization
**Ex Ante Winsorization**: This affects the formation of long-short portfolios only for strategies that sort bonds into different bins based on signals derived from past returns (e.g., momentum, short-term reversal, long-term reversal). These returns are winsorized up until the portfolio formation month $t$, and the corresponding signal is then computed with these winsorized returns.

**Ex Post Winsorization**: This impacts the performance of any characteristic-sorted portfolio, as it modifies the bond returns used to compute the returns of the different portfolios. This introduces a look-ahead bias in the portfolio performance and makes the performance unattainable or 'infeasible'. 

- `{'adj':'wins,'level':98,'location':'right'}`: winsorize the right tail of the distribution at the 98th percentile level.
- `{'adj':'wins,'level':98,'location':'left'}`: winsorize the left tail of the distribution at the 2nd percentile level.
- `{'adj':'wins,'level':98,'location':'both'}`: winsorize the left tail of the distribution at the 2nd percentile level and the right tail at 98th percentile level.

For ex ante winsorization, returns at time $t$ are winsorized based on the pooled distribution of returns from $t_0$ up until $t$. This means that for every $t$, percentile levels must be recomputed, which can impact the performance of the scripts. This is particularly relevant when running hundreds of different strategies (see Data Uncertainty section). To mitigate this issue, there is an option to pass a pandas DataFrame with precomputed rolling percentiles, thereby avoiding the need to compute the winsorization threshold at each iteration.
Example:
```python
BREAKPOINTS = pbl.load_breakpoints_WRDS()
BREAKPOINTS.index = pd.to_datetime(BREAKPOINTS.index)

# specify the parameters 
params = {'strategy':rating_single_sort,'rating':'NIG',
'filters': {'adj':'wins','level': 98,'location':'both','df_breakpoints':BREAKPOINTS}}

```
### Data Uncertainty
The scripts [MomentumDataUncertainty.py](examples/MomentumDataUncertainty.py) and [RatingDataUncertainty.py](examples/MomentumDataUncertainty.py) provide replications of Section X in Dickerson, Robotti, and Rossetti, [2024](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4575879).

These scripts allow comparison of the effects of ex-ante and ex-post data cleaning procedures on the expected returns of long-short portfolios sorted by specific variables/characteristics. They highlight the look-ahead bias that is incorrectly introduced when ex-post cleaning procedures are applied.

### Portfolio statistics
## Portfolio Turnover

Portfolio turnover plays a crucial role in unveiling potential anomalies. Ignoring transaction costs and portfolio turnover might render a possible strategy more appealing by showing larger gross alphas and average expected returns. However, these strategies might not improve an investor's opportunity set once transaction costs and portfolio turnover are considered.

To illustrate this, let's consider the following Python code using the PyBondLab library. Before running this snippet, ensure you have executed the previous code snippet that sets up the necessary data and imports.

### Step-by-Step Instructions

1. **Execute Initial Setup**:
    - Ensure you have run the initial setup code provided in an earlier part of this README, which includes importing libraries, defining the `single_sort` strategy, and loading the data into `tbl1`.

2. **Portfolio Turnover Calculation**:
    - The following code snippet demonstrates how to compute portfolio turnover, accounting for the given strategy parameters (see [example](https://github.com/GiulioRossetti94/PyBondLab/tree/main/examples) for the complete script). 

```python
# Define a dictionary with strategy and optional parameters.
params = {
    'strategy': single_sort,
    'rating': None,
    'turnover': True
}

# Fit the strategy to the data. Specify ID identifier and column of returns.
res = pbl.StrategyFormation(tbl1, **params).fit(IDvar="ISSUE_ID", RETvar="RET_L5M")

# Get long-short portfolios (equal- and value-weighted).
ew, vw = res.get_long_short()

# Compute portfolio turnover.
ew_turnover, vw_turnover = res.get_ptf_turnover()
```




## References
Dickerson, Robotti, and Rossetti, [2024](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4575879)

Novy-Marx, Velikov, [2023](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4338007)

[Open Source Bond Asset Pricing project](https://openbondassetpricing.com/)

## Requirements
The following requirements were used for testing. It is possible that older versions would work as well
- Python(3.11+)
- NumPy(1.26+)
- Pandas(2.2+)
- statsmodels(0.14+)
- matplotlib(3+)

## Contact
Giulio Rossetti giulio.rossetti.1@wbs.ac.uk

Alex Dickerson alexander.dickerson1@unsw.edu.au

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
