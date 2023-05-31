from KNN.DataPreparation import data_preparation
from KNN.KNN_Classification import train_KNN
from NSGA.OptimizeProcess import Run_Optimization

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # ML part in the beginning
    # X_train, X_test, Y_train, Y_test, feature_selection = data_preparation(data_file="sonar.data")
    # train_KNN(X_train=X_train, Y_train=Y_train, binary_vector=feature_selection)

    # Run NSGA process

    Run_Optimization()


