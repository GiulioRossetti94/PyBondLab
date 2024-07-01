# PyBondLab
PyBondLab is a Python module designed for portfolio sorting and other tools tailored for empirical asset pricing research, with a particular focus on corporate bonds. 

## Overview
PyBondLab provides tools for computing and evaluating investment strategies. It features look-ahead bias free data cleaning procedures to ensure the integrity and reliability of empirical results.

## Documentation




## Usage & Examples

### Portfolio sorting
This example demonstrates how to implement a long-short investment strategy using the `PyBondLab` module, based on quintile sorting of corporate bond credit ratings.

At each month $t$, corporate bonds are sorted into five portfolios according to their credit rating (`RATING_NUM` column). 
The strategy involves:

1. **Long Position**: buy the portfolio containing bonds with the lowest credit rating
2. **Short Position**: sell the portfolio containing bonds with the highest credit rating

```python
import PyBondLab as pbl
import pandas as pd

# read bond dataset
data = pd.read_csv("bond_data.csv")

holding_period = 1             # holding period returns
n_portf        = 5             # number of portfolios
sort_var1    = 'RATING_NUM'    # sorting chararcteristic/ variable

# Initialize the single sort variable
single_sort = pbl.SingleSort(holding_period, sort_var1,n_portf)

# define a dictionary with strategy and optional parameters
params = {'strategy': single_sort,
          'rating':None,
  }

# fit the strategy to the data
res = pbl.StrategyFormation(data, **params).fit(IDvar = "ISSUE_ID",RETvar = "RET_L5M")

# get long-short portfolios (equal- and value-weighted)
ew,vw = res.get_long_short_ex_ante()


```
### Momentum in corporate bonds and Data Uncertainty

## References


## Requirements

## Contact