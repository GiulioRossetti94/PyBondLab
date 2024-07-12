import pandas as pd
import os

def load_data(file_root:str,file_name: str) -> pd.DataFrame:
    """
    Load data from a csv file
    
    Parameters
    ----------
    file_root: str
        The root directory where the file is located
    file_name: str
        The name of the file to be loaded
        
    Returns
    -------
    pd.DataFrame
        The data loaded from the file
    """
    file_path = os.path.split(os.path.abspath(file_root))[0]
    data = pd.read_csv(os.path.join(file_path,file_name),index_col=0)
    data.index = pd.to_datetime(data.index)
    # if "date" in data.columns:
    #     data["date"] = pd.to_datetime(data["date"], dayfirst=True)
    #     data = data.set_index("date")

    # for col in data:
    #     data[col] = pd.to_numeric(data[col], errors='coerce')
    return data