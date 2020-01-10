"""
Functions for loading the data
"""
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

def load_bfw_datatable(fname, cols=None):
    """
    Load datatable of pairs, and often associated scores and metadata for each respective sample pair.

    cols allows for only the columns needed to be kept. Thus, column headers not in cols will be pruned out.

    Parameters
    ----------
    fname : str
        The name/path of the data file.
        
    cols: container (list or tuple): <optional> default=None
        List (or tuple) of columns headers to return; Note [] accessor is used, so typicall column keys are of type str.
        If element in cols does not exist, then it is simply ignored

    Returns
    -------
    data : pandas.DataFrame
        The data in a pandas.DataFrame with columns of at least: fold, p1, p2, label
        Note that columns are added in many steps, so scores, predicted, and others may also be columns.

    """
    assert Path(fname).is_file(), f"error: file of datatable does not exist {fname}"
    data = pd.read_pickle(fname)
    if cols:
        # only keep columns specified as input arg cols
        columns = data.columns.to_list()
        
        for column in columns:
            if column not in cols:
                del data[column]
        for col in cols:
            if col not in columns:
                warnings.warn(f'cols={col} was not found in datatable... will be ommitted')
                
    return data

