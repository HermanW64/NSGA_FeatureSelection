import numpy as np

from KNN.DataPreparation import data_preparation
from KNN.KNN_Classification import train_KNN
from NSGA.ProblemDefine import MyProblem
from NSGA.TerminationCriteria import termination_criteria
from NSGA.NSGAInitialize import set_NSGA
from NSGA.OptimizeProcess import Run_Optimization
from NSGA.EvaluateSolution import EvaluateSolution
import logging
logging.basicConfig(level="INFO")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # I. ML part in the beginning
    X_train, X_test, Y_train, Y_test, num_features = data_preparation(data_file="sonar.data")
    train_KNN(X_train=X_train, Y_train=Y_train)

    # II. prepare container to record important data in each run (totally 15)
    min_mce_train_list = []
    min_mce_solution_binary_list = []
    hv_list = []
    clf_error_test_list = []

    # Run NSGA process once (one run)

    # 1. problem define:
    problem = MyProblem(number_features=num_features, X_train=X_train, Y_train=Y_train)

    # 2. termination criteria
    termination = termination_criteria(max_gen=100)

    # 3. set hyperparameters for NSGA (all parameters are already set)
    algorithm = set_NSGA()

    # 4. run and visualize result of NSGA
    run_time = 1
    while run_time <= 15:
        min_mce_train, min_mce_solution_binary, hv_value = Run_Optimization(problem=problem,
                                                                 termination=termination,
                                                                 algorithm=algorithm,
                                                                 total_num_features=num_features,
                                                                 run_time=run_time,
                                                                 )

        clf_error_test = EvaluateSolution(X_train=X_train, Y_train=Y_train,
                                      X_test=X_test, Y_test=Y_test,
                                      best_selection=min_mce_solution_binary)

        # add data
        min_mce_train_list.append(min_mce_train)
        min_mce_solution_binary_list.append(min_mce_solution_binary)
        hv_list.append(hv_value)
        clf_error_test_list.append(clf_error_test)

        run_time += 1

    logging.info("\n =====Summary of 15 Runs=====")
    logging.info("min mce on training set: \n" + str(min_mce_train_list))
    # logging.info("min mce solutions: \n" + str(min_mce_solution_binary_list))
    logging.info("HV for each run: \n" + str(hv_list))
    logging.info("clf error for each run:\n" + str(clf_error_test_list) + "\n")

    # choose the lowest error during 15 runs, and the corresponding solutions
    best_index = clf_error_test_list.index(min(clf_error_test_list))
    best_mce = clf_error_test_list[best_index]
    best_solution = min_mce_solution_binary_list[best_index]
    logging.info("the best MCE: " + str(best_mce))
    logging.info("the best solution: \n" + str(best_solution))


