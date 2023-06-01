"""
run the optimization process,
from problem defining to visualization
"""
from NSGA.ProblemDefine import MyProblem
from NSGA.TerminationCriteria import termination_criteria
from NSGA.NSGAInitialize import set_NSGA
from NSGA.OptimizeProcess import optimize
from NSGA.EvaluateSolution import EvaluateSolution
import logging

logging.basicConfig(level="INFO")


def run_optimization(num_features=None, X_train=None, Y_train=None,
                     X_test=None, Y_test=None, plot_name=None, verbose=False):
    """
    :param num_features:
    :param X_train:
    :param Y_train:
    :param X_test:
    :param Y_test:
    :return: the following 4 lists
    """

    # prepare container to record important data in each run (totally 15)
    min_mce_train_list = []
    min_mce_solution_binary_list = []
    hv_list = []
    clf_error_test_list = []

    # 1. problem define:
    problem = MyProblem(number_features=num_features, X_train=X_train, Y_train=Y_train)

    # 2. termination criteria
    termination = termination_criteria(max_gen=100)

    # 3. set hyperparameters for NSGA (all parameters are already set)
    algorithm = set_NSGA()

    # 4. run and visualize result of NSGA
    run_time = 1
    while run_time <= 15:
        logging.info("\n----the run {0} begins----".format(run_time))

        min_mce_train, min_mce_solution_binary, hv_value = optimize(problem=problem,
                                                                    termination=termination,
                                                                    algorithm=algorithm,
                                                                    total_num_features=num_features,
                                                                    run_time=run_time,
                                                                    plot_name=plot_name,
                                                                    verbose=verbose
                                                                    )

        # record the data
        min_mce_train_list.append(min_mce_train)
        min_mce_solution_binary_list.append(min_mce_solution_binary)
        hv_list.append(hv_value)

        run_time += 1

    # after 15 runs, get the lowest MCE from min_mce_train
    best_index = min_mce_train_list.index(min(min_mce_train_list))
    best_mce_train = min_mce_train_list[best_index]
    best_solution = min_mce_solution_binary_list[best_index]

    clf_error_test = EvaluateSolution(X_train=X_train, Y_train=Y_train,
                                      X_test=X_test, Y_test=Y_test,
                                      best_selection=best_solution)

    return min_mce_train_list, min_mce_solution_binary_list, hv_list, \
        clf_error_test, best_solution, best_mce_train
