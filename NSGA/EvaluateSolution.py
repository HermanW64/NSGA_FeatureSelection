"""
evaluate the best solution from NGSA-II
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import logging
logging.basicConfig(level="INFO")


def EvaluateSolution(X_train=None, Y_train=None, X_test=None, Y_test=None,
                     best_selection=None):

    logging.info("start to evaluate the best solution ...")
    # -- use the corresponding features on test set and get the classification error
    # get the selected feature names
    selected_features_train = X_train.columns[best_selection == 1]
    selected_features_test = X_test.columns[best_selection == 1]

    # generate the dataset with selected features
    X_train_selected = X_train.loc[:, selected_features_train]
    X_test_selected = X_test.loc[:, selected_features_test]

    # logging.info("dataset prepared!")
    # calculate classification score on the X_train_selected
    k = 5
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_selected, Y_train)

    # show the classification error
    Y_pred = knn.predict(X_test_selected)
    classification_error = 1 - accuracy_score(Y_test, Y_pred)
    clf_error_test = round(classification_error, 4)
    logging.info("The error on validation set with the best solution: " + str(clf_error_test))

    return clf_error_test
