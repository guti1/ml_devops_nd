from typing import Tuple

import lightgbm as lgbm
import numpy as np
import optuna
import optuna.integration.lightgbm as lgb
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold


# Optional: implement hyperparameter tuning.
def train_model(train_X: np.array, train_y: np.array,
                hyperparam_tuning: bool = False) -> lgbm.LGBMModel:
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    train_X: np.array
        Training data.
    train_y : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    seed = 1301

    params = {
            "objective": "binary",
            "metric": "binary_error",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "seed": seed,
        }
    dtrain = lgb.Dataset(data=train_X, label=train_y)
    if hyperparam_tuning:

        skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
        study_tuner = optuna.create_study(direction='minimize')

        optuna.logging.set_verbosity(optuna.logging.ERROR)

        tuner = lgb.LightGBMTunerCV(params,
                                    dtrain,
                                    study=study_tuner,
                                    early_stopping_rounds=250,
                                    time_budget=1800,
                                    seed=42,
                                    folds=skf,
                                    num_boost_round=10000,)
        tuner.run()
        model = lgbm.train(params=tuner.best_params, train_set=dtrain,
                           num_boost_round=10000)

    else:
        base_params = {'objective': 'binary',
                       'metric': 'binary_error',
                       'verbosity': -1,
                       'boosting_type': 'gbdt',
                       'seed': seed,
                       'feature_pre_filter': False,
                       'lambda_l1': 0.044275861757513656,
                       'lambda_l2': 4.078523034198104e-05,
                       'num_leaves': 31,
                       'feature_fraction': 0.41600000000000004,
                       'bagging_fraction': 0.9003385691538116,
                       'bagging_freq': 3,
                       'min_child_samples': 20}

        model = lgbm.train(base_params, dtrain, 10000)

    return model


def compute_model_metrics(y: np.array, preds: np.array) -> Tuple[float,
                                                                 float,
                                                                 float]:
    """
    Validates the trained machine learning model using precision, recall,
    and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model: lgbm.LGBMModel, X: np.array) -> np.array:
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : lgbm.LGBMModel
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    pred = model.predict(X)

    return pred
