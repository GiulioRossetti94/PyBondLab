from pandas import DataFrame
from PyBondLab.data.data_loading import load_data

def load() -> DataFrame:
    """
    Load the breakpoints (rollling percentiles) WRDS data
    """
    return load_data(__file__,'breakpoints_wrds.csv')