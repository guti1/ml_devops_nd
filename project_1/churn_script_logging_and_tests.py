import os
import joblib
import logging
import shutil
from sklearn.exceptions import NotFittedError
import churn_library_solution as cls

DATE_FORMAT = "%y-%b-%d %H:%M:%S"
logging.basicConfig(
    filename="./logs/test_churn_library.log",
    level=logging.INFO,
    filemode="w",
    datefmt=DATE_FORMAT,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def test_import():
    """
    test data import - this example is completed for you to assist with the other test functions
    """

    try:
        df = cls.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns"
        )
        raise err


def test_eda():
    """
    Test perform_eda function in pipeline, first we clean up the existing plots, than we regenerate them
    """
    names_of_plots_to_check = [
        "Churn_hist.png",
        "Customer_Age_hist.png",
        "correlation_matrix_heatmap.png",
        "marital_status_counts_barplot.png",
        "total_transaction_counts_dist.png",
    ]
    directory_to_cleanup = "./images/eda"
    for filename in os.listdir(directory_to_cleanup):
        file_path = os.path.join(directory_to_cleanup, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))
    try:
        cls.perform_eda(cls.import_data("./data/bank_data.csv"))
        for plot in names_of_plots_to_check:
            try:
                with open("images/eda/%s" % plot, "r"):
                    logging.info("Testing perform_eda %s plot: SUCCESS" % plot)
            except FileNotFoundError as err:
                logging.error("Testing perform_eda: image missing")
                raise err
    except AssertionError as err:
        logging.error("Testing perform_eda: ")
        raise err


def test_encoder_helper():
    """
    test encoder helper - testing the target encoding step
    """

    columns_to_encode = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]
    encoded_columns = [col + "_Churn" for col in columns_to_encode]

    try:
        imported_df = cls.import_data("./data/bank_data.csv")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        encoded_df = cls.encoder_helper(
            imported_df,
            columns_to_encode,
            "Churn",
        )
    except AssertionError as err:
        logging.error("Testing encoder_helper: ERROR")
        raise err

    try:
        assert encoded_df.empty is not True
    except AssertionError as err:
        logging.error("Testing encoder_helper: The resulting df seems empty")
        raise err

    try:
        for col in encoded_columns:
            assert col in encoded_df
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The dataframe doesn't have all the encoded columns"
        )
        raise err


def test_perform_feature_engineering():
    """
    test perform_feature_engineering
    """

    columns_to_encode = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]
    try:
        imported_df = cls.import_data("./data/bank_data.csv")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        encoded_df = cls.encoder_helper(imported_df, columns_to_encode, "Churn")
        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
            encoded_df, "Churn"
        )

        assert x_train.shape[0] == y_train.shape[0]
        assert x_test.shape[0] == y_test.shape[0]
        logging.info("Feature generation: SUCCESS")
    except AssertionError as err:
        logging.error("Feature generation: ERROR")
        raise err


def test_train_models():
    """
    test train_models
    """
    models_to_load = ["rfc_model.pkl", "lr_model.pkl"]
    columns_to_encode = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]
    try:
        imported_df = cls.import_data("./data/bank_data.csv")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        encoded_df = cls.encoder_helper(imported_df, columns_to_encode, "Churn")
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
            encoded_df, "Churn"
        )
        cls.train_models(X_train, X_test, y_train, y_test)
        logging.info("Model fitting: SUCCESS")
    except BaseException as err:
        logging.error("Testing train_models: ERROR")
        raise err
    for model in models_to_load:
        try:
            model = joblib.load("models/%s" % model)
            logging.info("Testing training_models: SUCCESS")
        except FileNotFoundError as err:
            logging.error("Testing train_models: The files weren't found")
            raise err
        try:
            model.predict(X_test)
        except NotFittedError as err:
            logging.error("Testing trained models: The model is not fitted - ERROR")
            raise err


if __name__ == "__main__":
    pass
