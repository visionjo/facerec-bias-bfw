"""
Functions to process the bfw data
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder


from facebias.iotools import load_features_from_image_list


def _label_encoder(labels):
    return LabelEncoder().fit_transform(labels)


def encode_gender_labels(file_paths):
    """
    Given a list of files, assuming directory separates different classes, i.e,

            <attribute>/<subject>/<face file>

    where <attribute> takes on the form <ethnicity>_<gender> (e.g., asian_male)
    :returns
        A unique integer value per gender.
    """
    tags = [Path(f).parent.parent.split("_")[1] for f in file_paths]

    return _label_encoder(tags)


def encode_ethnicity_labels(file_paths):
    """
    Given a list of files, assuming directory separates different classes, i.e,

        <attribute>/<subject>/<face file>

    where <attribute> takes on the form <ethnicity>_<gender> (e.g., asian_male)
    :returns
        A unique integer value per ethnicity.
    """

    tags = [str(Path(f).parent.parent).split("_")[0] for f in file_paths]

    return _label_encoder(tags)


def encode_attribute_labels(file_paths):
    """
    Given a list of files, assuming directory separates different classes, i.e,

        <attribute>/<subject>/<face file>

    :returns
        A unique integer value per attribute.
    """
    tags = [Path(f).parent.parent for f in file_paths]

    return _label_encoder(tags)


def encode_to_labels(file_paths):
    """
    Given a list of files, assuming directory separates different classes, i.e,
        <attribute>/<subject>/<face file>

    :returns
        A unique integer value per subject.
    """

    tags = [Path(f).parent for f in file_paths]
    return _label_encoder(tags)


def get_attribute_gender_ethnicity(data, path_col, col_suffix=""):
    """
    Given a panda.DataFrame that has one column which contains path to each gender-ethnicity image,
    extract four information about the image and append to different columns. These columns are:
    attribute column (att + col_suffix) - contains ethinicity_gender information
    ethnicity column (e + col_suffix) - contains ethnicity information
    gender column (g + col_suffix) - contains gender information
    label column (a + col_suffx) - contains ethnicity + gender abbreviation information

    Parameters
    ----------
    data        pandas.DataFrame that has a column 'path_col' that contains path to each file
    path_col    column name of the column that contains path to each image file
    col_suffix  suffix to be added to attribute, ethnicity, gender, and label columns

    Returns
    -------
    data:       pandas.DataFrame that is appended with attribute, ethnicity, gender, and label columns
    """
    attribute_col = f"att{col_suffix}"
    ethnicity_col = f"e{col_suffix}"
    gender_col = f"g{col_suffix}"
    label_col = f"a{col_suffix}"

    data[attribute_col] = data[path_col].apply(lambda x: x.split("/")[0])
    data[ethnicity_col] = data[attribute_col].apply(lambda x: x.split("_")[0][0].upper())
    data[gender_col] = data[attribute_col].apply(lambda x: x.split("_")[1][0].upper())
    data[label_col] = data[ethnicity_col] + data[gender_col]

    for col in [attribute_col, ethnicity_col, gender_col, label_col]:
        data[col] = data[col].astype("category")

    return data


def assign_person_unique_id(data):
    subject_names = list(set(
        ["/".join(p1.split('/')[:-1]) for p1 in data["p1"].unique()] +
        ["/".join(p2.split('/')[:-1]) for p2 in data["p2"].unique()]))
    ids = _label_encoder(subject_names)
    data["ids1"] = ids[:data.shape[0]]
    data["ids2"] = ids[data.shape[0]:]
    return data


def compute_score_into_table(data, dir_features):
    # create ali_images list of all faces (i.e., unique set)
    image_list = list(np.unique(data["p1"].to_list() + data["p2"].to_list()))
    # read features as a dictionary, with keys set as the filepath of the image with values set as the face encodings
    features = load_features_from_image_list(image_list, dir_features, ext_feat="npy")
    # score all feature pairs by calculating cosine similarity of the features
    data["score"] = data.apply(lambda x: cosine_similarity(features[x["p1"]], features[x["p2"]])[0][0], axis=1)
    return data


def load_image_pair_with_attribute_and_score(image_pair_path, dir_features):
    data = pd.read_csv(image_pair_path)
    data = get_attribute_gender_ethnicity(data, "p1", "1")
    data = get_attribute_gender_ethnicity(data, "p2", "2")
    data = assign_person_unique_id(data)
    data = compute_score_into_table(data, dir_features)
    return data
