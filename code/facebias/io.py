"""
Functions for loading the data
"""
import warnings
from pathlib import Path

import pandas as pd


def _isfile(fpath):
    return Path(fpath).is_file()


def prune_dataframe(data, cols):
    # only keep columns specified as input arg cols
    columns = data.columns.to_list()

    for column in columns:
        if column not in cols:
            del data[column]
    for col in cols:
        if col not in columns:
            warnings.warn(f"cols={col} was not found in datatable... will be ommitted")

    return data


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
                warnings.warn(
                    f"cols={col} was not found in datatable... will be ommitted"
                )

    return data


def save_bfw_datatable(data, fpath="datatable.pkl", cols=None, append_cols=True):
    """
    Saves data table; if cols is set, only the respective cols included; if append=True, checks if the table exists;
    if so, load and include existing columns in file not in current data table (i.e., only update cols of data).

    If append is set False, data table will overwrite file (i.e., create new file, whether or not it exists).

    Parameters
    ----------
    data  : pd.DataFrame()
        Data table to save

    fpath : str: <optional> default = datatable.pkl
        The name/path of the data file.

    cols: container (list or tuple): <optional> default=None
        List (or tuple) of columns to save; Note [] accessor is used, so typically column keys are of type str.
        If element in cols does not exist, then it is simply ignored

    append_cols: bool: <optional> default=True
        If True, only update columns of data; else, overwrite or create new file.

    """
    if cols:
        data = prune_dataframe(data, cols)
    if append_cols:
        if _isfile(fpath):
            data_in = pd.read_pickle(fpath)

            if len(data) != len(data_in):
                warnings.warn(
                    "cannot append: sizes of tables are different\n terminating function call\nNothing saved"
                )
                return None

            for column in data.columns:
                data_in[column] = data["column"]
            data = data_in.copy()
            del data_in
    pd.to_pickle(data, fpath)
