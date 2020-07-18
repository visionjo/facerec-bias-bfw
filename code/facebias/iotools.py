"""
Functions for loading the data
"""
import os
import pickle
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
        warnings.warn(f"Directory {din} exists")
    _mkdir(din)


def _isfile(fpath):
    return Path(fpath).is_file()


def _isdir(fpath):
    return Path(fpath).is_file()


def exists(path):
    return _isfile(path) or _isdir(path)


def load_features_from_image_list(
        li_images, dir_features, ext_img="jpg", ext_feat="pkl"
):
    """
    Provided a list of images and the directory holding features, load features
    into LUT with using the relative file path as the key paired with the
    respective feature as the value.

    NOTE that it is assumed that features are named and stored like the images,
    with the difference in dir_features points to the folder that the relative
    file path is rooted. Check that file extensions are properly set-- the only
    time this would make no difference is if the list is actually the relative
    file paths of the features, and, hence, no replacement of string will occur.

    Finally, the root directory should be considered separate from the relative
    file paths, as the LUT keys, otherwise, will contain the file path instead
    of the file path relative to the DB for which can be used to identify the
    source of each feature.

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
    return {
        f: np.load(os.path.join(dir_features, f.replace(f".{ext_img}", f".{ext_feat}")))
        for f in li_images
    }


def prune_df(data, columns_in):
    """
    Prune dataframe data so that only columns specified in cols remain. Delete
    the rest.

    Parameters
    ----------
    data : pandas.DataFrame
    columns_in : container (list or tuple)
                List (or tuple) of columns headers of data we want to keep

    Returns
    -------
    data : pandas.DataFrame
        The pruned dataframe that only contain columns in cols
    """
    columns = data.columns.to_list()

    for column in columns:
        if column not in columns_in:
            del data[column]
    for col in columns_in:
        if col not in columns:
            warnings.warn(f"cols={col} was not found in datatable... will be ommitted")

    return data


def load_bfw_datatable(f_name, cols=None, default_score_col=None):
    """
    Load datatable of pairs, and often associated scores and metadata for each
    respective sample pair.

    cols allows for only the columns needed to be kept. Thus, column headers not
    in cols will be pruned out.

    Parameters
    ----------
    f_name : str
        The name/path of the data file.

    cols: container (list or tuple): <optional> default=None
        List (or tuple) of columns headers to return; Note [] accessor is used,
        so typically column keys are of type str. If element in cols does not
        exist, then it is simply ignored
    default_score_col:  column to set as 'score' (some code assumes 'score' col)
    
    Returns
    -------
    data : pandas.DataFrame
        The data in a pandas.DataFrame with columns of at least:
            fold, p1, p2, label
        Note that columns are added in many steps, so scores, predicted, and
        others may also be columns.


    """
    assert Path(f_name).exists(), f"error: file of datatable does not exist {f_name}"
    set_score = False
    data = pd.read_pickle(f_name)

    if default_score_col and default_score_col in data:
        if cols:
            cols += [default_score_col]
        set_score = True
    if cols:
        data = prune_df(data, cols)
    if set_score:
        # set default score to column
        data["score"] = data[default_score_col]
    return data


def save_bfw_datatable(
        data, fpath="datatable.pkl", cols=None, append_cols=True, f_type="pickle"
):
    """
    Saves data table; if cols is set, only the respective cols included; if
    append=True, checks if the table exists;
    if so, load and include existing columns in file not in current data table
    (i.e., only update cols of data).

    If append is set False, data table will overwrite file (i.e., create new
    file, whether or not it exists).

    Parameters
    ----------
    data  : pd.DataFrame()
        Data table to save

    fpath : str: <optional> default = datatable.pkl
        The name/path of the data file.

    cols: container (list or tuple): <optional> default=None
        List (or tuple) of columns to save; Note [] accessor is used, so
        typically column keys are of type str. If element in cols does not
        exist, then it is simply ignored

    append_cols: bool: <optional> default=True
        If True, only update columns of data; else, overwrite or create file.

    f_type: str: <optional> default = 'pickle'
        Specify the type of file to save ['pickle' or 'csv']. Note, if neither
        is set file will be dumped as pickle.

    """
    if cols:
        data = prune_df(data, cols)
    if append_cols:
        if _isfile(fpath):
            data_in = pd.read_pickle(fpath)

            if len(data) != len(data_in):
                warnings.warn(
                    "cannot append: sizes of tables are different\n "
                    "terminating "
                    "function call\nNothing saved"
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


def reshape_arr(arr_in):
    shape = (arr_in.shape[0], arr_in[0].shape[0])
    if len(arr_in[0].shape) > 1:
        if arr_in[0].shape[1] == 512:
            shape = (arr_in.shape[0], arr_in[0].shape[1])

    tmp_arr = np.zeros((shape[0], 1, shape[1]))
    for i, vec in enumerate(arr_in):
        tmp_arr[i, 0, ...] = vec
    return tmp_arr


def split_bfw_features(path_data, dir_features, tags='gender', val_fold=1):
    """
    Split BFW data as 5-fold. Specifically, reserve val_fold as the held-out
    test fold, with the remaining four as train.
    """
    df = pd.read_csv(path_data)
    df['face'] = df.face.str.replace('.jpg', '.npy')

    fold_test = df.loc[df.fold == val_fold, :]
    features_val = fold_test.apply(
        lambda x: np.load(dir_features + '/' + x['face']), axis=1).to_numpy()
    features_val = reshape_arr(features_val)

    labels_val = fold_test[tags].to_numpy()
    refs_val = fold_test['face'].to_list()

    fold_train = df.loc[df.fold != val_fold, :]
    features_tr = fold_train.apply(
        lambda x: np.load(dir_features + '/' + x['face']), axis=1).to_numpy()
    features_tr = reshape_arr(features_tr)
    labels_tr = fold_train[tags].to_numpy()
    refs_tr = fold_train['face'].to_list()

    return refs_tr, features_tr, labels_tr, refs_val, features_val, labels_val


def prepare_training_features(path_data):
    with open(path_data, "rb") as f:
        train_data = pickle.load(f)

    li_features = list()
    li_feature_ref = list()
    li_label = list()
    for k, v in train_data.items():
        li_feature_ref.append(k)
        li_features.append(v[..., np.newaxis].T)
        li_label.append(1 if k.split("/")[0].split("_")[1][0] == "m" else 0)

    return li_feature_ref, np.array(li_features), np.array(li_label)


def load_utk_unlabeled_test(path_data, label_type='gender'):
    # Here are label representation as encapsulated in the file name (source
    # https://susanqq.github.io/UTKFace/): The labels of each face image is
    # embedded in the file name, formatted like
    # [age]_[gender]_[race]_[date&time].jpg
    #
    # [age] is an integer from 0 to 116, indicating the age [gender] is
    # either 0 (male) or 1 (female) [race] is an integer from 0 to 4,
    # denoting White, Black, Asian, Indian, and Others (like Hispanic,
    # Latino, Middle Eastern). [date&time] is in the format of
    # yyyymmddHHMMSSFFF, showing the date and time an image was collected to
    # UTKFace

    li_features = list()
    val_ref = list()
    li_label = list()

    for f in path_data.glob("*.npy"):
        vec = np.load(f)
        li_features.append(vec[..., np.newaxis].T)
        val_ref.append(f.name)
        if label_type == 'gender':
            # UTK-Face, the males are represented by 0's
            li_label.append(1 if str(f).split('_')[1] == '0' else 0)
        else:  # do ethnicity
            # UTK-Face, the males are represented by 0's
            li_label.append(int(str(f).split('_')[1]))

    return np.array(li_features), np.array(li_label)
