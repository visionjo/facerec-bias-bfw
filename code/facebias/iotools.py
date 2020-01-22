"""
Functions for loading the data
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def _mkdir(din):
    """
    :param din directory to make
    """
    Path(din).absolute().mkdir(exist_ok=True, parents=True)


def makedir(din):
    if Path(din).absolute().is_dir():
        warnings.warn(f'Directory {din} exists')
    _mkdir(din)


def _isfile(fpath):
    return Path(fpath).is_file()


def load_features_from_image_list(
        li_images, dir_features, ext_img="jpg", ext_feat="pkl"
):
    """
    Provided a list of images and the directory holding features, load features into LUT with using the relative
    file path as the key paired with the respective feature as the value.

    NOTE that it is assumed that features are named and stored like the images, with the difference being dir_features
    points to the folder that the relative file path is rooted. Check that file extensions are properly set-- the only
    time this would make no difference is if the list is actually the relative file paths of the features, and, hence,
    no replacement of string will occur. Finally, the root directory should be considered separate from the relative
    file paths, as the LUT keys, otherwise, will contain the file path instead of the file path relative to the DB for
    which can be used to identify the source of each feature.

    Parameters
    ----------
    li_images       list of images
    dir_features    root directory of features
    ext_img         file extension of images as in list
    ext_feat        file extension of features as saved on disc

    Returns
    -------
    features: dict(file path, feature vector)

    """
    li_features = [
        dir_features + f.replace(f".{ext_img}", f".{ext_feat}") for f in
        li_images
    ]
    # read features as a dictionary, with keys set as the file path of the image with values set as the face encodings
    # features = {str(f.replace(dir_features, '')): pd.read_pickle(f) for f in li_features}
    # TODO some reason comprehension above does not work. Return to refactor later
    features = {}
    for feat in li_features:
        features[
            feat.replace(dir_features, "").replace(f".{ext_feat}",
                                                   f".{ext_img}")
        ] = np.load(feat)

    return features


def prune_dataframe(data, cols):
    # only keep columns specified as input arg cols
    columns = data.columns.to_list()

    for column in columns:
        if column not in cols:
            del data[column]
    for col in cols:
        if col not in columns:
            warnings.warn(
                f"cols={col} was not found in datatable... will be ommitted")

    return data


def load_bfw_datatable(fname, cols=None, default_score_col=None):
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
    assert Path(
        fname).exists(), f"error: file of datatable does not exist {fname}"
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
    if default_score_col and default_score_col in data:
        data['score'] = data[default_score_col]

    return data


def save_bfw_datatable(
        data, fpath="datatable.pkl", cols=None, append_cols=True,
        f_type="pickle"
):
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

    f_type: str: <optional> default = 'pickle'
        Specify the type of file to save ['pickle' or 'csv']. Note, if neither is set file will be dumped as pickle.

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

            for column in data.columns.to_list():
                data_in[column] = data[column]
            data = data_in.copy()
            del data_in
    if f_type.lower() == "csv":
        fpath = fpath[:-4] + ".csv"
        data.to_csv(fpath, index=False)
    else:
        pd.to_pickle(data, fpath)
