from confusion_matrix import ConfusionMatrix
import k_fold as Kfold
import statistics
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn import preprocessing
import pandas as pd
import pathlib
import json
import argparse

def print_if_enabled(print_string):
    if enable_print:
        print(print_string)

def split_data_X_y(data):
    data_X = data[:,:-1]
    data_y = data[:,-1]
    return data_X, data_y

def main():
    # load dataset
    data_csv = pd.read_csv("dataset/smoke_detection_iot.csv")

    # remove index, UTC and CNT
    utc_column = data_csv.columns.get_loc("UTC")
    cnt_column = data_csv.columns.get_loc("CNT")
    columns_to_remove = set([0, cnt_column, utc_column])
    all_columns = set(range(data_csv.shape[1]))
    data_columns = list(all_columns.difference(columns_to_remove))
    data_csv = data_csv.iloc[:,data_columns]

    # drop duplicates after removing columns (only 2)
    data_csv = data_csv.drop_duplicates()

    # get data array
    data = data_csv.values

    # feature/ classes
    feature_names = data_csv.columns[:-1]
    target_names = data_csv['Fire Alarm'].unique().astype(str).tolist()

    # num classes
    num_classes = len(np.unique(data[:,-1].astype(int)))
    binary = num_classes == 2

    # setup models, folds, rng
    models = [
        (LogisticRegression(), "Logistic Regression"),
        (KNeighborsClassifier(n_neighbors=3), "K-Neighbors 3"),
        (KNeighborsClassifier(n_neighbors=5), "K-Neighbors 5"),
        (KNeighborsClassifier(n_neighbors=7), "K-Neighbors 7"),
        (GaussianNB(), "Gaussian Naive Bayes"),
        (DecisionTreeClassifier(random_state=0), "Decision Tree"),
    ]
    folds = [5, 10]
    rng = np.random.default_rng(12345)
    rng.shuffle(data)

    # init results
    results = {}
    for num_folds in folds:
        print_if_enabled(f"Num Folds {num_folds}")
        k_fold = Kfold.KFold(data, num_folds, False)

        for model, model_name in models:
            print_if_enabled(f"\tModel {model_name}")
            model_name_folds = f"{model_name} ({num_folds} folds)"
            results[model_name_folds] = {}

            # init metrics lists
            accuracy_list = []
            recall_macro_list = []
            precision_macro_list = []
            f1_measure_macro_list = []
            recall_micro_list = []
            precision_micro_list = []
            f1_measure_micro_list = []

            results[model_name_folds]["Folds"] = {}
            for fold in range(num_folds):
                print_if_enabled(f"\t\tFold {fold}")
                results[model_name_folds]["Folds"][fold] = {}

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
                conf_matrix = ConfusionMatrix(test_y.astype(int), pred_y.astype(int), num_classes)
                accuracy = conf_matrix.accuracy()
                recall_macro = conf_matrix.recall("macro")
                precision_macro = conf_matrix.precision("macro")
                f1_measure_macro = conf_matrix.f1_measure("macro") 
                recall_micro = conf_matrix.recall("micro")
                precision_micro = conf_matrix.precision("micro")
                f1_measure_micro = conf_matrix.f1_measure("micro")
                conf_matrix_string = f"{conf_matrix.matrix}"

                # print and add metrics to results
                print_if_enabled(f"\t\t\tAccuracy: {accuracy}")
                results[model_name_folds]["Folds"][fold]["Accuracy"] = accuracy
                if binary:
                    print_if_enabled(f"\t\t\tRecall: {recall_macro}")
                    print_if_enabled(f"\t\t\tPrecision: {precision_macro}")
                    print_if_enabled(f"\t\t\tF1-Measure: {f1_measure_macro}")
                    results[model_name_folds]["Folds"][fold]["Recall"] = recall_macro
                    results[model_name_folds]["Folds"][fold]["Precision"] = precision_macro
                    results[model_name_folds]["Folds"][fold]["F1-Measure"] = f1_measure_macro
                else:
                    print_if_enabled(f"\t\t\tRecall Macro: {recall_macro}")
                    print_if_enabled(f"\t\t\tPrecision Macro: {precision_macro}")
                    print_if_enabled(f"\t\t\tF1-Measure Macro: {f1_measure_macro}")
                    print_if_enabled(f"\t\t\tRecall Micro: {recall_micro}")
                    print_if_enabled(f"\t\t\tPrecision Micro: {precision_micro}")
                    print_if_enabled(f"\t\t\tF1-Measure Micro: {f1_measure_micro}")
                    results[model_name_folds]["Folds"][fold]["Recall Macro"] = recall_macro
                    results[model_name_folds]["Folds"][fold]["Precision Macro"] = precision_macro
                    results[model_name_folds]["Folds"][fold]["F1-Measure Macro"] = f1_measure_macro
                    results[model_name_folds]["Folds"][fold]["Recall Micro"] = recall_micro
                    results[model_name_folds]["Folds"][fold]["Precision Micro"] = precision_micro
                    results[model_name_folds]["Folds"][fold]["F1-Measure Micro"] = f1_measure_micro
                results[model_name_folds]["Folds"][fold]["Confusion Matrix"] = conf_matrix_string

                # append data to metrics lists
                accuracy_list.append(accuracy)
                recall_macro_list.append(recall_macro)
                precision_macro_list.append(precision_macro)
                f1_measure_macro_list.append(f1_measure_macro)
                recall_micro_list.append(recall_micro)
                precision_micro_list.append(precision_micro)
                f1_measure_micro_list.append(f1_measure_micro)
                
                # plot decision trees
                if model_name == "Decision Tree":
                    plt.figure(figsize=(15,15))
                    plot_tree(model, 
                              feature_names = feature_names, 
                              class_names = target_names, 
                              filled = True, 
                              rounded = True,
                              fontsize=6)

                    plt.savefig(f"results/{model_name_folds} {fold}.png") 

            # calculate mean and stdev for metrics
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


            print_if_enabled("\t\tMean and Standard Deviation")
            print_if_enabled(f"\t\t\tAccuracy: Mean: {accuracy_mean} Standard Deviation: {accuracy_stdev}")
            results[model_name_folds]["Accuracy"] = {"Mean": accuracy_mean, "Standard Deviation": accuracy_stdev}
            if binary:
                print_if_enabled(f"\t\t\tRecall: Mean: {recall_macro_mean} Standard Deviation: {recall_macro_stdev}")
                print_if_enabled(f"\t\t\tPrecision: Mean: {precision_macro_mean} Standard Deviation: {precision_macro_stdev}")
                print_if_enabled(f"\t\t\tF1-Measure: Mean: {f1_measure_macro_mean} Standard Deviation: {f1_measure_macro_stdev}")
                results[model_name_folds]["Recall"] = {"Mean": recall_macro_mean, "Standard Deviation": recall_macro_stdev}
                results[model_name_folds]["Precision"] = {"Mean": precision_macro_mean, "Standard Deviation": precision_macro_stdev}
                results[model_name_folds]["F1-Measure"] = {"Mean": f1_measure_macro_mean, "Standard Deviation": f1_measure_macro_stdev}
            else:
                print_if_enabled(f"\t\t\tRecall Macro: Mean: {recall_macro_mean} Standard Deviation: {recall_macro_stdev}")
                print_if_enabled(f"\t\t\tPrecision Macro: Mean: {precision_macro_mean} Standard Deviation: {precision_macro_stdev}")
                print_if_enabled(f"\t\t\tF1-Measure Macro: Mean: {f1_measure_macro_mean} Standard Deviation: {f1_measure_macro_stdev}")
                print_if_enabled(f"\t\t\tRecall Micro: Mean: {recall_micro_mean} Standard Deviation: {recall_micro_stdev}")
                print_if_enabled(f"\t\t\tPrecision Micro: Mean: {precision_micro_mean} Standard Deviation: {precision_micro_stdev}")
                print_if_enabled(f"\t\t\tF1-Measure Micro: Mean: {f1_measure_micro_mean} Standard Deviation: {f1_measure_micro_stdev}")
                results[model_name_folds]["Recall Macro"] = {"Mean": recall_macro_mean, "Standard Deviation": recall_macro_stdev}
                results[model_name_folds]["Precision Macro"] = {"Mean": precision_macro_mean, "Standard Deviation": precision_macro_stdev}
                results[model_name_folds]["F1-Measure Macro"] = {"Mean": f1_measure_macro_mean, "Standard Deviation": f1_measure_macro_stdev}
                results[model_name_folds]["Recall Micro"] = {"Mean": recall_micro_mean, "Standard Deviation": recall_micro_stdev}
                results[model_name_folds]["Precision Micro"] = {"Mean": precision_micro_mean, "Standard Deviation": precision_micro_stdev}
                results[model_name_folds]["F1-Measure Micro"] = {"Mean": f1_measure_micro_mean, "Standard Deviation": f1_measure_micro_stdev}

    # save results
    with open('results/result.json', 'w') as fp:
        json.dump(results, fp, indent=4)
        
# create results dir
pathlib.Path("results").mkdir(exist_ok=True)

# enable printing
parser = argparse.ArgumentParser()
parser.add_argument('-print', action='store_true')    
enable_print = parser.parse_args().print

main()
