from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.preprocessing._encoders import _BaseEncoder


def process_data(
        input_df: pd.DataFrame,
        categorical_features: List[str] = [],
        label: str = None,
        training: bool = True,
        encoder: _BaseEncoder = None,
        label_binarizer: BaseEstimator = None):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and
    a label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in
    functionality that scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in
        `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be
        returned for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    label_binarizer : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the
        encoder passed in.
    labelbinarzer : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the
        binarizer passed in.
    """

    if label is not None:
        y = input_df[label]
        input_df = input_df.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = input_df[categorical_features].values
    X_continuous = input_df.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        label_binarizer = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = label_binarizer.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = label_binarizer.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    input_df = np.concatenate([X_continuous, X_categorical], axis=1)
    return input_df, y, encoder, label_binarizer
