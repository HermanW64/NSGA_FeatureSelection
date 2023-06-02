from pymoo.optimize import minimize
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
import numpy as np
np.set_printoptions(suppress=True)
import logging
logging.basicConfig(level="INFO")


def optimize(problem=None, algorithm=None, termination=None, total_num_features=None,
             run_time=None, plot_name=None, verbose=False):

    # 1. run the predefined problem
    # logging.info("optimization begins ...")
    res = minimize(problem,
                   algorithm,
                   termination,
                   # test_set=test_set,
                   # test_label=test_label,
                   # seed=1,
                   save_history=True,
                   verbose=verbose)

    # logging.info("optimization ends ...")

    # 2. save initial pareto and final pareto front
    # -- get the objectives f1 and f2 of the first and last generation
    initial_gen_F = res.history[0].pop.get("F")
    last_gen_F = res.history[-1].pop.get("F")

    # -- visualization (only plot for one run)
    if run_time == 1:
        # show pareto front in the end
        plt.figure(figsize=(7, 5))
        plt.scatter(initial_gen_F[:, 0], initial_gen_F[:, 1], s=30, facecolors='none', edgecolors='blue')
        plt.title("Initial Pareto Front on training set --" + str(plot_name))
        plt.xlabel("classification error")
        plt.xlim(0, 1)
        plt.ylabel("total number of features")
        plt.ylim(0, total_num_features)
        plt.savefig("./pareto_images/clf_initial_pareto_" + str(plot_name) + ".png")
        logging.info("initial pareto front plotted!")

        # show pareto front in the end
        plt.figure(figsize=(7, 5))
        plt.scatter(last_gen_F[:, 0], last_gen_F[:, 1], s=30, facecolors='none', edgecolors='blue')
        plt.title("Final Pareto Front on training set --" + str(plot_name))
        plt.xlabel("classification error")
        plt.xlim(0, 1)
        plt.ylabel("total number of features")
        plt.ylim(0, total_num_features)
        plt.savefig("./pareto_images/clf_final_pareto_" + str(plot_name) + ".png")
        logging.info("last pareto front plotted!")

    # 3. calculate HV
    # reference point should be the worst point among the population (1, total number of features)
    worst_obj1 = np.max(initial_gen_F[:, 0]).copy()
    worst_obj2 = np.max(initial_gen_F[:, 1]).copy()
    ref_point = np.array([worst_obj1, worst_obj2])

    # get all points of the best pareto front in the last generation
    ind = HV(ref_point=ref_point)
    hv_value = round(ind(initial_gen_F), 4)
    logging.info("HV value: " + str(hv_value))

    # 4. record the best solutions and minimum classification error
    # -- get the result data
    X = res.X
    F = np.round(res.F, 4)

    # -- find out the lowest classification error from the result
    min_mce_index = np.argmin(F[:, 0])
    min_mce_valid = F[min_mce_index, 0]

    # -- find out the corresponding feature selection
    min_mce_solution = X[min_mce_index, :]
    min_mce_solution_binary = np.where(min_mce_solution > 0.5, 1, 0)
    logging.info("minimum classification error on validation set: " + str(min_mce_valid))
    logging.info("number of the corresponding selected features: " + str(sum(min_mce_solution_binary)))
    logging.info("the corresponding best solution: \n" + str(min_mce_solution_binary))

    return min_mce_valid, min_mce_solution_binary, hv_value





