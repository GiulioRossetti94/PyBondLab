# PyBondLab
PyBondLab is a Python module for portfolio sorting and other tools for empirical asset pricing research with particular focus on corporate bonds. 

## Documentation




## Usage & Examples

### Portfolio sorting
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