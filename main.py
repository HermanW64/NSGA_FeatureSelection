from KNN.DataPreparation import data_preparation
from KNN.KNN_Classification import train_KNN
from NSGA.ProblemDefine import MyProblem
from NSGA.TerminationCriteria import termination_criteria
from NSGA.NSGAInitialize import set_NSGA
from NSGA.OptimizeProcess import Run_Optimization

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # ML part in the beginning
    X_train, X_test, Y_train, Y_test, feature_selection = data_preparation(data_file="sonar.data")
    train_KNN(X_train=X_train, Y_train=Y_train, binary_vector=feature_selection)

    # Run NSGA process

    # 1. problem define:
    problem = MyProblem(feature_selection=feature_selection, X_train=X_train)

    # 2. termination criteria
    termination = termination_criteria(max_gen=10000)

    # 3. set hyperparameters for NSGA (all parameters are already set)
    algorithm = set_NSGA()

    # 4. run and visualize result of NSGA
    Run_Optimization(problem=problem, termination=termination, algorithm=algorithm)



