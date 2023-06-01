"""
run the optimization process on dataset sonar
15 runs in total
"""
from KNN.DataPreparation import data_preparation
from KNN.KNN_Classification import train_KNN
from NSGA.RunOptimization import run_optimization
from NSGA.ShowSummary import summary


def run_on_sonar():

    # I. ML part in the beginning
    X_train, X_test, Y_train, Y_test, num_features = data_preparation(data_file="sonar.data")
    train_KNN(X_train=X_train, Y_train=Y_train)

    # II. Run Optimization
    min_mce_train_list, min_mce_solution_binary_list, hv_list, \
        clf_error_test, best_solution, best_mce_train = run_optimization(num_features=num_features, X_train=X_train,
                                                                         Y_train=Y_train,X_test=X_test, Y_test=Y_test,
                                                                         plot_name="sonar", verbose=True
                                                                         )

    # IV. Summary of the 15 runs
    summary(min_mce_train_list=min_mce_train_list, hv_list=hv_list, clf_error_test=clf_error_test,
            best_solution=best_solution, best_mce_train=best_mce_train, file_name="sonar")
