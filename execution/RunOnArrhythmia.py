"""
run the optimization process on dataset sonar
15 runs in total
"""
from KNN.DataPreparation import data_preparation
from KNN.KNN_Classification import train_KNN
from NSGA.RunOptimization import run_15times
from NSGA.ShowSummary import summary


def run_on_arrhythmia():

    # I. ML part in the beginning
    X_train, X_valid, X_test, Y_train, Y_valid, Y_test, num_features = data_preparation(data_file="arrhythmia.data")
    clf_error_test_allFeatures= train_KNN(X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)

    # II. Run Optimization
    min_mce_valid_list, min_mce_solution_binary_list, hv_list, \
        clf_error_test, best_solution, best_mce_train = run_15times(num_features=num_features,
                                                                    X_train=X_train, Y_train=Y_train,
                                                                    X_valid=X_valid, Y_valid=Y_valid,
                                                                    X_test=X_test, Y_test=Y_test,
                                                                    plot_name="arrhythmia", verbose=True
                                                                    )

    # IV. Summary of the 15 runs
    summary(min_mce_valid_list=min_mce_valid_list, hv_list=hv_list, clf_error_test=clf_error_test,
            clf_error_test_allFeatures=clf_error_test_allFeatures, best_solution=best_solution,
            best_mce_train=best_mce_train, file_name="arrhythmia")
