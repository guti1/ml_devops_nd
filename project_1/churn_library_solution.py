"""
A library to process the churn-dataset, fit models on it and save the resulting
artefacts as a rather minimal example for an end-to-end ML pipeline example.


Author: JG
2022-01
"""
import logging
from pathlib import Path
from typing import List, Tuple
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, RocCurveDisplay
from sklearn.model_selection import GridSearchCV, train_test_split


DATE_FORMAT = "%y-%b-%d %H:%M:%S"
logging.basicConfig(
    filename="./logs/churn_library.log",
    level=logging.INFO,
    filemode="w",
    datefmt=DATE_FORMAT,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class EmptyDataFrameException(Exception):
    """
    A template case for the exception stemming from empty DataFrame
    """


def import_data(path: str) -> pd.DataFrame:
    """
    returns dataframe for the csv found at path

    input:
            path: a path to the csv
    output:
            df: pandas dataframe
    """

    # Validate input path and import data
    import_file_path = Path(path)
    try:
        logging.info(
            f"Importing data from {import_file_path}".format(
                import_file_path=import_file_path
            )
        )
        abs_path = import_file_path.resolve(strict=True)
        imported_df = pd.read_csv(abs_path, index_col=0)
        imported_df["Churn"] = imported_df["Attrition_Flag"].apply(
            lambda val: 0 if val == "Existing Customer" else 1
        )
        shape = imported_df.shape
        logging.info(
            f"SUCCESS: Imported data from {import_file_path} with shape of {shape}".format(
                import_file_path=import_file_path, shape=shape
            )
        )
        return imported_df

    except FileNotFoundError as err:
        logging.error(
            f"ERROR: The file wasn't found at {import_file_path}".format(
                import_file_path=import_file_path
            )
        )
        raise err


def perform_eda(dataframe: pd.DataFrame) -> None:
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    """
    sns.set()
    if dataframe.empty:
        logging.error("ERROR: The input DataFrame is empty")
        raise EmptyDataFrameException("The input DataFrame is empty")

    dataframe["Churn"] = dataframe["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    # Plot histograms
    cols_to_plot_as_hist = ["Churn", "Customer_Age"]
    for col in cols_to_plot_as_hist:
        plt.figure(figsize=(20, 10))
        dataframe[col].hist()
        plt.savefig(f"./images/eda/{col}_hist.png")
        plt.clf()

    # Plot marital status value counts
    plt.figure(figsize=(20, 10))
    dataframe.Marital_Status.value_counts("normalize").plot(kind="bar")
    plt.savefig("./images/eda/marital_status_counts_barplot.png")
    plt.clf()

    # Plot distribution of total number of transactions
    plt.figure(figsize=(20, 10))
    sns.displot(dataframe["Total_Trans_Ct"])
    plt.savefig("./images/eda/total_transaction_counts_dist.png")
    plt.clf()

    # Plot correlation matrix
    plt.figure(figsize=(20, 10))
    sns.heatmap(dataframe.corr(), annot=False, cmap="Dark2_r", linewidths=2)
    plt.savefig("./images/eda/correlation_matrix_heatmap.png")
    plt.clf()


def encoder_helper(
    dataframe: pd.DataFrame, category_lst: List[str], response: str
) -> pd.DataFrame:
    """
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used
            for naming variables or index y
            column]

    output:
            df: pandas dataframe with new columns for
    """

    if dataframe.empty:
        logging.error("ERROR: The input DataFrame is empty")
        raise EmptyDataFrameException("The input DataFrame is empty")

    for col in category_lst:
        list_of_means = []
        grouped_df = dataframe.groupby(col).mean()[response]
        for val in dataframe[col]:
            list_of_means.append(grouped_df.loc[val])

        new_variable_name = col + "_" + response
        dataframe[new_variable_name] = list_of_means

    return dataframe


def perform_feature_engineering(
    dataframe: pd.DataFrame, response: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Helper function to split data into train and test sets by feautures
    and the target variable(response)
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used
              for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    if dataframe.empty:
        logging.error("ERROR: The input DataFrame is empty")
        raise EmptyDataFrameException("The input DataFrame is empty")

    keep_cols = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
        "Gender_Churn",
        "Education_Level_Churn",
        "Marital_Status_Churn",
        "Income_Category_Churn",
        "Card_Category_Churn",
    ]

    X_df = dataframe[keep_cols]
    y_df = dataframe[[response]]
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_df, test_size=0.3, random_state=42
    )

    return X_train, X_test, y_train, y_test


def classification_report_image(
    y_train,
    y_test,
    y_train_preds_lr,
    y_train_preds_rf,
    y_test_preds_lr,
    y_test_preds_rf,
) -> None:
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    plt.rc("figure", figsize=(5, 5))
    plt.text(
        0.01,
        1.25,
        str("Random Forest Train"),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.05,
        str(classification_report(y_test, y_test_preds_rf)),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.6,
        str("Random Forest Test"),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.7,
        str(classification_report(y_train, y_train_preds_rf)),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.axis("off")
    plt.savefig("./images/results/classification_result_rf.png")
    plt.clf()

    plt.rc("figure", figsize=(5, 5))
    plt.text(
        0.01,
        1.25,
        str("Logistic Regression Train"),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.05,
        str(classification_report(y_train, y_train_preds_lr)),
        {"fontsize": 10},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.text(
        0.01,
        0.6,
        str("Logistic Regression Test"),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.7,
        str(classification_report(y_test, y_test_preds_lr)),
        {"fontsize": 10},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.axis("off")
    plt.savefig("./images/results/classification_result_lr.png")
    plt.clf()


def feature_importance_plot(
    model: ClassifierMixin, X_data: pd.DataFrame, output_pth: str
) -> None:
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    shap.summary_plot(shap_values, X_data, plot_type="bar")
    plt.savefig(output_pth)
    plt.clf()


def train_models(X_train, X_test, y_train, y_test) -> None:
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(max_iter=500, random_state=42)

    param_grid = {
        "n_estimators": [200, 500],
        "max_features": ["auto", "sqrt"],
        "max_depth": [4, 5, 100],
        "criterion": ["gini", "entropy"],
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train.values.ravel())

    lrc.fit(X_train, y_train.values.ravel())

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
    )

    feature_importance_plot(
        cv_rfc.best_estimator_, X_test, "./images/results/importance.png"
    )

    plt.figure(figsize=(20, 10))
    axis = plt.gca()
    RocCurveDisplay.from_estimator(lrc, X_test, y_test, ax=axis)
    RocCurveDisplay.from_estimator(cv_rfc.best_estimator_, X_test, y_test, ax=axis)
    plt.show()
    plt.savefig("./images/results/classification_result_roc.png")
    plt.clf()

    joblib.dump(cv_rfc.best_estimator_, "models/rfc_model.pkl")
    joblib.dump(lrc, "models/lr_model.pkl")


if __name__ == "__main__":
    data_df = import_data("data/bank_data.csv")
    perform_eda(data_df)
    encoded_data_df = encoder_helper(
        data_df,
        [
            "Gender",
            "Education_Level",
            "Marital_Status",
            "Income_Category",
            "Card_Category",
        ],
        "Churn",
    )
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        encoded_data_df, "Churn"
    )
    train_models(X_train, X_test, y_train, y_test)
