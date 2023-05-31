import numpy as np
from pymoo.core.problem import ElementwiseProblem


class MyProblem(ElementwiseProblem):
    """
    define the optimization problem, including:
    # of variables, # of objectives,
    lower and upper of variables [x1, x2]
    x1: total number of features selected 1 to feature_num
    x2: classification error with selected features 0 to 100
    """

    def __init__(self, feature_selection=None, xl=None, xu=None):
        if xl is None:
            xl = [-2, -2]
            xu = [2, 2]

        super().__init__(n_var=len(feature_selection),
                         n_obj=2,
                         n_ieq_constr=2,
                         xl=np.array(xl),
                         xu=np.array(xu))

    def _evaluate(self, x, out, *args, **kwargs):
        # x is one-dimensional vector
        # Apply the binary threshold to x
        x_binary = np.where(x > 0.5, 1, 0)

        # f are objective functions
        # --f1: total number of selected features
        f1 = np.sum(x_binary)

        # --f2: KNN classification error on the training set with the selected features

        f2 = (x[0] - 1) ** 2 + x[1] ** 2

        # merge f and g respectively
        out["F"] = [f1, f2]

# problem = MyProblem()

