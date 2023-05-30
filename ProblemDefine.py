import numpy as np
from pymoo.core.problem import ElementwiseProblem


class MyProblem(ElementwiseProblem):
    """
    define the optimization problem, including:
    # of variables, # of objectives,
    lower and upper of variables [x1, x2]
    x1: total number of features selected 1- feature_num
    x2: classification error with selected features 0-100
    """

    def __init__(self, xl=None, xu=None):
        if xl is None:
            xl = [-2, -2]
            xu = [2, 2]

        super().__init__(n_var=2,
                         n_obj=2,
                         n_ieq_constr=2,
                         xl=np.array(xl),
                         xu=np.array(xu))

    def _evaluate(self, x, out, *args, **kwargs):
        # f are objective functions
        f1 = 100 * (x[0] ** 2 + x[1] ** 2)
        f2 = (x[0] - 1) ** 2 + x[1] ** 2

        # g are constraints
        g1 = 2 * (x[0] - 0.1) * (x[0] - 0.9) / 0.18
        g2 = - 20 * (x[0] - 0.4) * (x[0] - 0.6) / 4.8

        # merge f and g respectively
        out["F"] = [f1, f2]
        out["G"] = [g1, g2]


# problem = MyProblem()

