"""
build a KNN classifier based on the training data,
and show the classification error
"""
import logging
logging.basicConfig(level="INFO")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def train_KNN(X_train=None, Y_train=None, X_test=None, Y_test=None):
    """
    :param X_train: all features data on training set in pandas format
    :param Y_train: the class label on training set in pandas format
    # :param binary_vector: feature selection vector, 1 means chosen and 0 means not chosen
    :return: classification error on training set, the lower, the better
    """
    # 1. Get the selected feature names
    # selected_features = X_train.columns[binary_vector == 1]

    # Subset the dataset with selected features
    # X_train_selected = X_train.loc[:, selected_features]
    # logging.info("feature selected!")
    logging.info("size of X_train: " + str(X_train.shape))
    logging.info("size of Y_train: " + str(Y_train.shape))
    logging.info("size of X_test: " + str(X_test.shape))
    logging.info("size of Y_test: " + str(X_test.shape))

    # 2. Train a KNN classifier with k = 5
    k = 5
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)

    # 3. Show the classification error
    Y_pred = knn.predict(X_test)
    logging.info("size of Y_pred: " + str(Y_pred.shape))
    classification_error = round((1 - accuracy_score(Y_test, Y_pred)), 4)

    logging.info("KNN Classification error on test set with all features: {:.4f}\n".format(classification_error))

    return classification_error