import numpy as np
from pymoo.core.problem import ElementwiseProblem
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class MyProblem(ElementwiseProblem):
    """
    define the optimization problem, including:
    # of variables, # of objectives,
    lower and upper of variables [x1, x2]
    x1: total number of features selected 1 to feature_num
    x2: classification error with selected features 0 to 100
    """

    def __init__(self, feature_selection=None, X_train=None, Y_train=None):

        super().__init__(n_var=len(feature_selection),
                         n_obj=2,
                         n_ieq_constr=2,
                         xl=np.array([0, 0]),
                         xu=np.array([1, 1]))

        self.X_train = X_train
        self.Y_train = Y_train

    def _evaluate(self, x, out, *args, **kwargs):
        # x is one-dimensional vector
        # Apply the binary threshold to x
        x_binary = np.where(x > 0.5, 1, 0)

        # f are objective functions
        # --f1: total number of selected features
        f1 = np.sum(x_binary)

        # --f2: KNN classification error on the training set with the selected features
        # get the selected feature names
        selected_features = self.X_train.columns[x_binary == 1]

        # generate the dataset with selected features
        X_train_selected = self.X_train.loc[:, selected_features]

        # calculate classification score on the X_train_selected
        k = 5
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_selected, self.Y_train)

        # show the classification error
        Y_pred = knn.predict(X_train_selected)
        classification_error = 1 - accuracy_score(self.Y_train, Y_pred)

        f2 = classification_error

        # merge f and g respectively
        out["F"] = [f1, f2]

# problem = MyProblem()

