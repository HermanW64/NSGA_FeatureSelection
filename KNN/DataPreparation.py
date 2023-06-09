"""
the module reads the dataset, do the basic data-cleaning,
and then split the data into training set (70%) and test set (30%)

*required libraries: sklearn, pandas
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
logging.basicConfig(level="INFO")


def data_preparation(data_file=None):

    # 1. Read the data with .data format
    data_path = "./data/" + str(data_file)

    # 1. Read the data with .data format
    data = pd.read_csv(data_path)

    # 3. Check if there are any missing values
    missing_values = data.isnull().any().any()
    if missing_values:
        logging.info("There are missing values in the data.")
    else:
        logging.info("No missing values found in the data.")

    # 4. Remove the rows with missing values
    valid_data = data.dropna()

    # 4.1 Remove irrelevant columns for musk
    if data_file == "musk.csv":
        valid_data = valid_data.drop(valid_data.columns[:3], axis=1)
        logging.info("Unnecessary columns dropped!")

    elif data_file == "arrhythmia.data":

        # Identify columns with "?" values
        columns_with_question_marks = valid_data.columns[valid_data.isin(["?"]).any()]

        # Remove columns with "?" values
        valid_data = valid_data.drop(columns_with_question_marks, axis=1)

    # 2. Show the first few lines with column titles
    # logging.info("cleaned data: ")
    # logging.info(valid_data.head(3))

    # 5. Split the data into X and Y
    X = valid_data.iloc[:, :-1]  # Input features (all columns except the last one)
    Y = valid_data.iloc[:, -1].astype(str)  # Target variable (last column)

    # Get the number of features
    num_features = X.shape[1]
    logging.info("number of features in the dataset: " + str(num_features))

    # Generate a vector of ones with the same length as the number of features
    feature_selection = np.ones(num_features)

    # Check if there is enough data for splitting
    if len(X) == 0:
        logging.info("Insufficient data for splitting.")
    else:
        # Split the data into training set (70%) and test set (30%)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.1429, random_state=42)
        logging.info("size of X_train: " + str(X_train.shape))
        logging.info("size of Y_train: " + str(Y_train.shape))
        logging.info("size of X_valid: " + str(X_valid.shape))
        logging.info("size of Y_valid: " + str(Y_valid.shape))
        logging.info("size of X_test: " + str(X_test.shape))
        logging.info("size of Y_test: " + str(Y_test.shape))
        logging.info("Data split completed! And data is prepared! \n")

    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test, num_features


