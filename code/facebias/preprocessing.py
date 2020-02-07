"""
Functions to process the bfw data
"""

from pathlib import Path

import numpy as np
import pandas as pd
from facebias.iotools import load_features_from_image_list
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder


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
    data[ethnicity_col] = data[attribute_col].apply(
        lambda x: x.split("_")[0][0].upper()
    )
    data[gender_col] = data[attribute_col].apply(lambda x: x.split("_")[1][0].upper())
    data[label_col] = data[ethnicity_col] + data[gender_col]

    for col in [attribute_col, ethnicity_col, gender_col, label_col]:
        data[col] = data[col].astype("category")

    return data


def assign_person_unique_id(data):
    """
    Assign unique ids to images in 'p1' and 'p2' column of 'data' and put them in column 'ids1' and 'ids2'

    parameters
    ----------
    data: pandas.DataFrame that contains column 'p1' and 'p2' which specify the relative path to the pair of images

    returns
    -------
    data:   pandas.DateFrame that have extra columns 'ids1' and 'ids2' which are the unique ids of images 'p1' and 'p2'
        respectively
    """
    label_encoder = LabelEncoder()

    subject_names = list(
        set(
            ["/".join(p1.split("/")[:-1]) for p1 in data["p1"].unique()]
            + ["/".join(p2.split("/")[:-1]) for p2 in data["p2"].unique()]
        )
    )
    label_encoder.fit(subject_names)

    data["ids1"] = label_encoder.transform(
        data["p1"].apply(lambda x: "/".join(x.split("/")[:-1]))
    )
    data["ids2"] = label_encoder.transform(
        data["p2"].apply(lambda x: "/".join(x.split("/")[:-1]))
    )

    return data


def compute_score_into_table(data, embedding_dir_path):
    """
    compute cosine similarity scores of each pair of images specified in columns 'p1' and 'p2' in 'data' using the
    embeddings from embedding_dir_path

    parameters
    ----------
    data:               pandas.DataFrame that contains paths to pair of images. The path to the first image is in
        column 'p1' while the path to the second image is in column 'p2'
    embedding_dir_path: path to the root directory that contains all the embeddings. in the root directories must
        exist subdirectories with name {ethnicity}_{gender}s, each of which contain person id subdirectories

    returns
    -------
    data:   pandas.DataFrame that have one more column: 'score', which contains the cosine similarity score between
        the embeddings of the pair of images specified in column 'p1' and 'p2'
    """
    # create ali_images list of all faces (i.e., unique set)
    image_list = list(np.unique(data["p1"].to_list() + data["p2"].to_list()))
    # read features as a dictionary, with keys set as the filepath of the image with values set as the face encodings
    features = load_features_from_image_list(
        image_list, embedding_dir_path, ext_feat="npy"
    )
    # score all feature pairs by calculating cosine similarity of the features
    data["score"] = data.apply(
        lambda x: cosine_similarity(features[x["p1"]], features[x["p2"]])[0][0], axis=1
    )
    return data


def load_image_pair_with_attribute_and_score(image_pair_path, embedding_dir_path):
    """
    load pandas.DataFrame that contains pairs of images, their attributes (gender, ethnicity, etc.) and the cosine
    similarity score between the embeddings of the pair of images

    parameters
    ----------
    image_pair_path:    path to the csv file that contain all image pairs of interest. The csv must contain columns
        'p1', 'p2', and 'label'
    embedding_dir_path: path to the root directory that contains all the embeddings. in the root directories must
        exist subdirectories with name {ethnicity}_{gender}s, each of which contain person id subdirectories

    returns
    -------
    data:   pandas.DataFrame that contains columns 'p1', 'p2', 'label', 'g1', 'g2', 'e1', 'e2', 'a1', 'a2', 'ids1',
        'ids2', 'score'. 'p1' and 'p2' are the paths to each pair of images. 'label' indicates whether these two
        images are the same person. 'g1', 'e1', 'a1', 'ids1' are the gender, ethnicity, abbreviated ethnicity-gender,
        and unique id of 'p1' respectively. The same goes for 'g2', 'e2', 'a2', and 'ids2'.
        'score' is the cosine similarity of the embeddings in 'embedding_dir_path' that correspond to 'p1' and 'p2'
    """
    data = pd.read_csv(image_pair_path)
    data = get_attribute_gender_ethnicity(data, "p1", "1")
    data = get_attribute_gender_ethnicity(data, "p2", "2")
    data = assign_person_unique_id(data)
    data = compute_score_into_table(data, embedding_dir_path)
    return data
