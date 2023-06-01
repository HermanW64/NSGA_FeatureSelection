import numpy as np
from pymoo.core.problem import ElementwiseProblem
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import logging
logging.basicConfig(level="INFO")


class MyProblem(ElementwiseProblem):
    """
    define the optimization problem, including:
    # of variables, # of objectives,
    lower and upper of variables [x1, x2]
    x1: total number of features selected 1 to feature_num
    x2: classification error with selected features 0 to 100
    """

    def __init__(self, number_features=None, X_train=None, Y_train=None):

        super().__init__(n_var=number_features,
                         n_obj=2,
                         n_ieq_constr=0,
                         # the xl and xu should be the same size of n_var
                         xl=np.zeros(number_features),
                         xu=np.ones(number_features))

        self.X_train = X_train.copy()
        self.Y_train = Y_train.copy()

    def _evaluate(self, x, out, *args, **kwargs):
        # logging.info("problem defining starts ...")
        # x is one-dimensional vector

        # Apply the binary threshold to x
        x_binary = np.where(x > 0.5, 1, 0)

        # f are objective functions
        # --f2: total number of selected features (warning: f2 must be greater than 0)
        f2 = np.sum(x_binary)
        if f2 < 1:
            random_index = np.random.randint(0, len(x_binary))
            x_binary[random_index] = 1

        # --f1: KNN classification error on the training set with the selected features
        # get the selected feature names
        selected_features = self.X_train.columns[x_binary == 1]

        # generate the dataset with selected features
        X_train_selected = self.X_train.loc[:, selected_features]
        Y_train = self.Y_train

        # calculate classification score on the X_train_selected
        k = 5
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_selected, Y_train)

        # show the classification error
        Y_pred = knn.predict(X_train_selected)
        classification_error = 1 - accuracy_score(self.Y_train, Y_pred)

        f1 = classification_error

        # merge f and g respectively
        out["F"] = [f1, f2]

        # logging.info("problem defining finished!")

# problem = MyProblem()

