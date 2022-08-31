from confusion_matrix import ConfusionMatrix
import k_fold as Kfold
import statistics
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn import preprocessing
import json


def split_data_X_y(data):
    data_X = data[:,:-1]
    data_y = data[:,-1]
    return data_X, data_y

def main():

    #TODO load dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    y = np.reshape(y, (-1, 1))
    data = np.concatenate((X, y), axis = 1)

    # setup vars
    models = [
        (LogisticRegression(), "Logistic Regression"),
        (KNeighborsClassifier(n_neighbors=3), "K-Neighbors 3"),
        (KNeighborsClassifier(n_neighbors=5), "K-Neighbors 5"),
        (KNeighborsClassifier(n_neighbors=7), "K-Neighbors 7"),
        (GaussianNB(), "Gaussian Naive Bayes"),
        #(DecisionTreeClassifier(), "Decision Tree"),
    ]
    folds = [5, 10]
    rng = np.random.default_rng(12345)

    # load data
    rng.shuffle(data)
    results = {}
    for num_folds in folds:
        print(f"Num Folds {num_folds}")
        k_fold = Kfold.KFold(data, num_folds, False)

        for model, model_name in models:
            print(f"\tModel {model_name}")
            model_name = f"{model_name} ({num_folds} folds)"
            results[model_name] = {}
            accuracy_list = []
            recall_macro_list = []
            precision_macro_list = []
            f1_measure_macro_list = []
            recall_micro_list = []
            precision_micro_list = []
            f1_measure_micro_list = []

            results[model_name]["Folds"] = {}
            for fold in range(num_folds):
                print(f"\t\tFold {fold}")
                results[model_name]["Folds"][fold] = {}

                # split train/test
                test, train = k_fold.test_train_split(fold)

                # train
                train_X, train_y = split_data_X_y(train)
                scaler = preprocessing.StandardScaler().fit(train_X)
                scaled_train_X = scaler.transform(train_X)
                model.fit(scaled_train_X, train_y)

                # test
                test_X, test_y = split_data_X_y(test)
                scaled_test_X = scaler.transform(test_X)
                pred_y = model.predict(scaled_test_X)

                # generate confusion matrix
                conf_matrix = ConfusionMatrix(test_y.astype(int), pred_y.astype(int))
                accuracy = conf_matrix.accuracy()
                recall_macro = conf_matrix.recall("macro")
                precision_macro = conf_matrix.precision("macro")
                f1_measure_macro = conf_matrix.f1_measure("macro") 
                recall_micro = conf_matrix.recall("micro")
                precision_micro = conf_matrix.precision("micro")
                f1_measure_micro = conf_matrix.f1_measure("micro") 


                print(f"\t\t\tAccuracy: {accuracy}")
                print(f"\t\t\tRecall Macro: {recall_macro}")
                print(f"\t\t\tPrecision Macro: {precision_macro}")
                print(f"\t\t\tF1-Measure Macro: {f1_measure_macro}")
                print(f"\t\t\tRecall Micro: {recall_micro}")
                print(f"\t\t\tPrecision Micro: {precision_micro}")
                print(f"\t\t\tF1-Measure Micro: {f1_measure_micro}")

                accuracy_list.append(accuracy)
                recall_macro_list.append(recall_macro)
                precision_macro_list.append(precision_macro)
                f1_measure_macro_list.append(f1_measure_macro)
                recall_micro_list.append(recall_micro)
                precision_micro_list.append(precision_micro)
                f1_measure_micro_list.append(f1_measure_micro)

                results[model_name]["Folds"][fold]["Accuracy"] = accuracy
                results[model_name]["Folds"][fold]["Recall Macro"] = recall_macro
                results[model_name]["Folds"][fold]["Precision Macro"] = precision_macro
                results[model_name]["Folds"][fold]["F1-Measure Macro"] = f1_measure_macro
                results[model_name]["Folds"][fold]["Recall Micro"] = recall_micro
                results[model_name]["Folds"][fold]["Precision Micro"] = precision_micro
                results[model_name]["Folds"][fold]["F1-Measure Micro"] = f1_measure_micro

            accuracy_mean = statistics.mean(accuracy_list)
            recall_macro_mean = statistics.mean(recall_macro_list)
            precision_macro_mean = statistics.mean(precision_macro_list)
            f1_measure_macro_mean = statistics.mean(f1_measure_macro_list)
            recall_micro_mean = statistics.mean(recall_micro_list)
            precision_micro_mean = statistics.mean(precision_micro_list)
            f1_measure_micro_mean = statistics.mean(f1_measure_micro_list)

            accuracy_stdev = statistics.stdev(accuracy_list)
            recall_macro_stdev = statistics.stdev(recall_macro_list)
            precision_macro_stdev = statistics.stdev(precision_macro_list)
            f1_measure_macro_stdev = statistics.stdev(f1_measure_macro_list)
            recall_micro_stdev = statistics.stdev(recall_micro_list)
            precision_micro_stdev = statistics.stdev(precision_micro_list)
            f1_measure_micro_stdev = statistics.stdev(f1_measure_micro_list)

            results[model_name]["Accuracy"] = {"Mean": accuracy_mean, "Standard Deviation": accuracy_stdev}
            results[model_name]["Recall Macro"] = {"Mean": recall_macro_mean, "Standard Deviation": recall_macro_stdev}
            results[model_name]["Precision Macro"] = {"Mean": precision_macro_mean, "Standard Deviation": precision_macro_stdev}
            results[model_name]["F1-Measure Macro"] = {"Mean": f1_measure_macro_mean, "Standard Deviation": f1_measure_macro_stdev}
            results[model_name]["Recall Micro"] = {"Mean": recall_micro_mean, "Standard Deviation": recall_micro_stdev}
            results[model_name]["Precision Micro"] = {"Mean": precision_micro_mean, "Standard Deviation": precision_micro_stdev}
            results[model_name]["F1-Measure Micro"] = {"Mean": f1_measure_micro_mean, "Standard Deviation": f1_measure_micro_stdev}
        
            print("\t\tMean and Standard Deviation")
            print(f"\t\t\tAccuracy: Mean: {accuracy_mean} Standard Deviation: {accuracy_stdev}")
            print(f"\t\t\tRecall Macro: Mean: {recall_macro_mean} Standard Deviation: {recall_macro_stdev}")
            print(f"\t\t\tPrecision Macro: Mean: {precision_macro_mean} Standard Deviation: {precision_macro_stdev}")
            print(f"\t\t\tF1-Measure Macro: Mean: {f1_measure_macro_mean} Standard Deviation: {f1_measure_macro_stdev}")
            print(f"\t\t\tRecall Micro: Mean: {recall_micro_mean} Standard Deviation: {recall_micro_stdev}")
            print(f"\t\t\tPrecision Micro: Mean: {precision_micro_mean} Standard Deviation: {precision_micro_stdev}")
            print(f"\t\t\tF1-Measure Micro: Mean: {f1_measure_micro_mean} Standard Deviation: {f1_measure_micro_stdev}")
    
    with open('result.json', 'w') as fp:
        json.dump(results, fp, indent=4)
        

main()
