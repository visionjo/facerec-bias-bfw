"""
Functions to process the bfw data
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

from code.facebias.io import load_features_from_image_list

def add_gender_ethnicity_attribute(data):
    data['att1'] = data.p1.apply(lambda x: x.split('/')[0]).astype('category')
    data['att2'] = data.p2.apply(lambda x: x.split('/')[0]).astype('category')

    data['e1'] = data.att1.apply(lambda x: x.split('_')[0][0].upper())
    data['e2'] = data.att2.apply(lambda x: x.split('_')[0][0].upper())

    data['g1'] = data.att1.apply(lambda x: x.split('_')[1][0].upper())
    data['g2'] = data.att2.apply(lambda x: x.split('_')[1][0].upper())

    data['a1'] = (data['e1'] + data['g1']).astype('category')
    data['a2'] = (data['e2'] + data['g2']).astype('category')

    data['e1'] = data['e1'].astype('category')
    data['e2'] = data['e2'].astype('category')
    data['g1'] = data['g1'].astype('category')
    data['g2'] = data['g2'].astype('category')

    return data

def assign_person_unique_id(data):
    le = LabelEncoder()

    subject_names = list(set(
        ["/".join(p1.split('/')[:-1]) for p1 in data['p1'].unique()] +
        ["/".join(p2.split('/')[:-1]) for p2 in data['p2'].unique()]))
    le.fit(subject_names)

    data['ids1'] = le.transform(data['p1'].apply(lambda x: "/".join(x.split('/')[:-1])))
    data['ids2'] = le.transform(data['p2'].apply(lambda x: "/".join(x.split('/')[:-1])))

    return data

def compute_score_into_table(data, dir_features):
    # create ali_images list of all faces (i.e., unique set)
    image_list = list(np.unique(data["p1"].to_list() + data["p2"].to_list()))

    # read features as a dictionary, with keys set as the filepath of the image with values set as the face encodings
    features = load_features_from_image_list(image_list, dir_features, ext_feat='npy')

    # score all feature pairs by calculating cosine similarity of the features
    data['score'] = data.apply(lambda x: cosine_similarity(features[x["p1"]], features[x["p2"]]), axis=1)

    return data

def load_image_pair_with_attribute_and_score(image_pair_path, dir_features):
    data = pd.read_csv(image_pair_path)
    data = add_gender_ethnicity_attribute(data)
    data = assign_person_unique_id(data)
    data = compute_score_into_table(data, dir_features)

    return data

if __name__=="__main__":
    final_data = load_image_pair_with_attribute_and_score("../../data/bfw-pairs.csv", "../../data/features/senet50")
    print(final_data.head())