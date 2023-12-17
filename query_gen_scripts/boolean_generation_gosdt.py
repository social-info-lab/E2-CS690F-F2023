#########
# File containing functions related to creating a decision tree to find good boolean expansion queries
#########

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree._tree import TREE_LEAF
import pandas as pd
import time
import pathlib
from sklearn.ensemble import GradientBoostingClassifier
from gosdt_model.threshold_guess import compute_thresholds
from utils.corpus import create_train_val_test_clean
from gosdt_model.gosdt import GOSDT
from query_gen_scripts.boolean_generation import import_attr_lists, convert_docs_to_input
from query_gen_scripts.query_formation_gosdt import form_queries
from query_gen_scripts.query_testing_gosdt import generate_results


# Local testing (small dataset etc.)
local_test = True
# Limit the dataset on cluster to reduce memory usage
low_memory = False

def main():
    # Load the dataset
    generation_mode = 'tfidf'
    data_path_comb = f'./data/query-data/decahouse_2020-10_stephen_combined.csv'
    attr_path = f'./data/attr-lists/2020-10-attr-lists-50000'
    regularization = 0.0001
    rows_subset_perc=0.03
    num_attr = 2000
    if local_test:
        num_attr = 10000
        rows_subset_perc=0.05
        regularization = [0.01,0.05,0.001,0.005,0.0001,0.0005,0.00001,0.00005]

    if low_memory:
        num_attr = 1000
        # Percentage of the dataset to use for training
        rows_subset_perc=0.05 # 1% of the dataset ~ 10000 rows
        # Regularization parameter for GOSDT
        regularization = [0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005]
        # Uncomment to convert from scientific notation to decimal
        # regularization = np.format_float_positional(regularization)
    # GOSDT output paths
    gosdt_folder = f'./data/decision-trees/gosdt/gosdt-guesses/'
    
    corpus_train, labels_train, corpus_test, labels_test, corpus_val, labels_val = create_train_val_test_clean(
        data_path_comb, perc_train=0.7, perc_val=0.2, subset_perc=rows_subset_perc)
    
    attr_lists = import_attr_lists(attr_path)
    
    # GOSDT - Sparse Decision Tree Classifier
    for attr_list in attr_lists:
        if low_memory or local_test:
            attr_list = attr_list[:num_attr]
        
        # remove the column 'class' if present as this causes an internal error in GOSDT
        if 'class' in attr_list:
            attr_list.remove('class')
        input_arrs = convert_docs_to_input(attr_list, corpus_train)
        np.set_printoptions(threshold=np.inf)
        # convert the train and test data into dataframes for GOSDT
        X_train = pd.DataFrame(input_arrs, columns=attr_list)
        y_train = pd.DataFrame(labels_train, columns=['relevant'])
        # Check memory usage of train data
        mem_str = f'Memory size train set: \n  dense: {X_train.memory_usage().sum()/1000000} mb'
        # Convert the DataFrame to sparse with boolean data type
        X_train = X_train.astype(pd.SparseDtype(bool, fill_value=False))
        mem_str = mem_str + f'\n  Sparse: {X_train.memory_usage().sum()/1000000} mb'

        # Guesses
        # GBDT parameters for threshold and lower bound guesses
        n_est = 40
        max_depth = 1
        # guess thresholds
        h = X_train.columns
        print("X:", X_train.shape)
        print("y:",y_train.shape)
        X_train, thresholds, header, threshold_guess_time = compute_thresholds(X_train, y_train, n_est, max_depth)

        # guess lower bound
        start_time = time.perf_counter()
        clf = GradientBoostingClassifier(n_estimators=n_est, max_depth=max_depth, random_state=42)
        clf.fit(X_train, y_train.values.flatten())
        warm_labels = clf.predict(X_train)
        elapsed_time = time.perf_counter() - start_time
        lb_time = elapsed_time

        # save the labels from lower bound guesses as a tmp file and return the path to it.
        labelsdir = pathlib.Path('/tmp/warm_lb_labels')
        labelsdir.mkdir(exist_ok=True, parents=True)
        labelpath = labelsdir / 'warm_label.tmp'
        labelpath = str(labelpath)
        pd.DataFrame(warm_labels, columns=["class_labels"]).to_csv(labelpath, header="class_labels",index=None)

        # set GOSDT configuration
        for reg in regularization:
            gosdt_output_path = f'{gosdt_folder}2020-10-{generation_mode}-{reg}-{num_attr}-gosdt-tree.txt'
            gosdt_model_path = f'{gosdt_folder}2020-10-{generation_mode}-{reg}-{num_attr}-gosdt-model.json'
            gosdt_profile_path = f'{gosdt_folder}2020-10-{generation_mode}-{reg}-gosdt-profile.csv'
    
            config = {
                        # Params for GOSDT
                        "regularization": reg, #0.0001, #0.002, # regularization penalizes the tree with more leaves.  
                        #"depth_budget": 5, # limits the max depth of the tree
                        "time_limit": 900, # training time limit in seconds
                        # "uncertainty_tolerance": 0.2, # threshold for early stopping with near-optimal model        
                        "warm_LB": True,
                        "path_to_labels": labelpath,
                        "worker_limit": 1, # nr of parallel workers (when 0, a single thread is created for each core)
                        "stack_limit": 0, # max bytes considered for use when allocating local buffers for worker threads.
                                        # Special Cases: When set to 0, all local buffers will be allocated from the heap.
                        # Output flags
                        "verbose": True,
                        "allow_small_reg":True,
                        "diagnostics": True,
                        # Output paths
                        "model": gosdt_model_path,
                        # "tree": gosdt_tree_path,
                        # "trace": gosdt_trace_path,
                        "profile": gosdt_profile_path
                    }
            model = GOSDT(config)
            # train GOSDT 
            model.fit(X_train, y_train)
            print(model.tree)

            # Uncomment to train a sklearn decision tree for comparison
            # tree_clf = DecisionTreeClassifier(criterion='entropy', max_depth=15, min_samples_leaf=5)
            # tree_clf.fit(input_arrs, labels_train)

            # Save gosdt output to file
            with open(gosdt_output_path, 'w') as f:
                rows, cols = X_train.shape
                f.write(f'Training set size: \n  rows:{rows}, cols:{cols}\n')
                f.write(f'\n num_attr: {num_attr}, roes_subset_prec: {rows_subset_perc}\n')
                f.write(mem_str)
                f.write(f'\n\nmodel.config:\n {str(model.configuration)}\n')
                f.write(f'\nmodel.tree:\n {str(model.tree)}')

            form_queries(generation_mode, reg,num_attr)
            generate_results(generation_mode, corpus_test, labels_test, corpus_val, labels_val,
                            regularization=reg, num_attr=num_attr)


if __name__ == "__main__":
    main()
