"""
Show summary information for 15 runs,
and store the information into a txt file
"""
import numpy as np
import logging
logging.basicConfig(level="INFO")


def summary(min_mce_valid_list=None, hv_list=None, clf_error_test=None,
            clf_error_test_allFeatures=None, best_solution=None, best_mce_train=None, file_name=None):

    # 1. display summary data
    logging.info("\n =====Summary of 15 Runs=====")
    logging.info("MCE on training set: \n" + str(np.min(min_mce_valid_list)))
    # logging.info("lowest MCE error on training set:\n" + str(best_mce_train) + "\n")
    logging.info("average HV for each run: " + str(np.average(hv_list)))

    logging.info("MCE on validation set: " + str(clf_error_test))
    logging.info("the corresponding best solution: \n" + str(best_solution))
    logging.info("the number of the selected features: " + str(np.sum(best_solution)))

    # store summary data into txt file
    hv_data = "HV data for 15 runs: " + str(hv_list)
    hv_line = "average HV for each run: " + str(np.average(hv_list))

    mce_training_set_line = "MCE on validation set for 15 runs: " + str(min_mce_valid_list)
    mce_training_set_line_lowest = "The lowest MCE on validation set for 15 runs: " + str(best_mce_train)

    best_solution_line = "best solution: \n" + str(best_solution)
    num_best_features_line = "the number of selected features: " + str(np.sum(best_solution))
    clf_error_test_set_line = "MCE on test data with selected features: " + str(clf_error_test)
    clf_error_test_allFeatures_line = "classification error on test data with all features: " + str(clf_error_test_allFeatures)

    data_write = hv_data + "\n" + hv_line + "\n" + mce_training_set_line + "\n" \
                 + mce_training_set_line_lowest + "\n" + best_solution_line + "\n" \
                 + num_best_features_line + "\n" + clf_error_test_set_line + "\n" + clf_error_test_allFeatures_line

    with open("./summary_data/summary_" + str(file_name) + ".txt", "w") as file:
        file.write(data_write)

    logging.info("data saved!")