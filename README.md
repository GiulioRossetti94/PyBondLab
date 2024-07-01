# PyBondLab
PyBondLab is a Python module designed for portfolio sorting and other tools tailored for empirical asset pricing research, with a particular focus on corporate bonds. It is part of the [Open Source Bond Asset Pricing project](https://openbondassetpricing.com/).

## Overview
PyBondLab provides tools for computing and evaluating investment strategies. It features look-ahead bias free data cleaning procedures to ensure the integrity and reliability of empirical results.

## Installation




## Usage & Examples

### Portfolio sorting
This example demonstrates how to implement a long-short investment strategy using the `PyBondLab` module, based on quintile sorting of corporate bond on credit ratings.

At each month $t$, corporate bonds are sorted into five portfolios according to their credit rating (`RATING_NUM` column). 
The strategy involves:

1. **Long Position**: buy the portfolio containing bonds with the lowest credit rating
2. **Short Position**: sell the portfolio containing bonds with the highest credit rating

```python
import PyBondLab as pbl
import pandas as pd

# read bond dataset
# use a pandas DataFrame with columns:
# "RATING_NUM": the credit rating of bonds
# "ISSUE_ID": the identifier for each bond
# "date": the date
# "RET_L5M": bond returns used to compute portfolio returns
# "VW":  weights used to compute value-weighted performnace
data = pd.read_csv("bond_data.csv")

holding_period = 1             # holding period returns
n_portf        = 5             # number of portfolios
sort_var1      = 'RATING_NUM'  # sorting chararcteristic/variable

# Initialize the single sort strategy
single_sort = pbl.SingleSort(holding_period, sort_var1, n_portf)

# Define a dictionary with strategy and optional parameters
params = {'strategy': single_sort,
          'rating':None,
  }

# Fit the strategy to the data. Specify ID identifier and column of returns 
res = pbl.StrategyFormation(data, **params).fit(IDvar = "ISSUE_ID",RETvar = "RET_L5M")

# Get long-short portfolios (equal- and value-weighted)
ew,vw = res.get_long_short()


```
### Data Uncertainty
the scripts [MomentumDataUncertainty.py](examples/MomentumDataUncertainty.py) and [RatingDataUncertainty.py](examples/MomentumDataUncertainty.py) provide replications of Section X in Dickerson, Robotti, and Rossetti, [2024](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4575879).

These scripts allow comparison of the effects of ex-ante and ex-post data cleaning procedures on the expected returns of long-short portfolios sorted by specific variables/characteristics. They highlight the look-ahead bias that is incorrectly introduced when ex-post cleaning procedures are applied.



## References
Dickerson, Robotti, and Rossetti, [2024](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4575879)

[Open Source Bond Asset Pricing project](https://openbondassetpricing.com/)

## Requirements

## Contact
