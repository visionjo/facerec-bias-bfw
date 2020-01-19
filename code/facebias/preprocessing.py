from pathlib import Path

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
