from pandas import DataFrame
from PyBondLab.data.data_loading import data_load

def load() -> DataFrame:
    """
    Load the breakpoints (rollling percentiles) WRDS data
    """
    return data_load(__file__,'breakpoints_wrds.csv')