from pymoo.optimize import minimize
from NSGA.ProblemDefine import MyProblem
from NSGA.NSGAInitialize import NSGA2
from NSGA.TerminationCriteria import termination_criteria
import matplotlib.pyplot as plt


def Run_Optimization(problem=None, algorithm=None, termination=None):

    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   save_history=True,
                   verbose=True)

    # show results
    X = res.X
    print("result of X:\n", X)
    F = res.F
    print("result of F:\n", F)

    # visualization
    xl, xu = problem.bounds()
    plt.figure(figsize=(7, 5))
    plt.scatter(X[:, 0], X[:, 1], s=30, facecolors='none', edgecolors='r')
    plt.xlim(xl[0], xu[0])
    plt.ylim(xl[1], xu[1])
    plt.title("Decision Space for x1 and x2")
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
    plt.title("Pareto Front for obj1 and obj2")
    plt.show()





