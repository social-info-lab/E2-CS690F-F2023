#########
# File containing functions related to creating a decision tree to find good boolean expansion queries
#########

import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree._tree import TREE_LEAF
import matplotlib.pyplot as plt
import pandas as pd
from utils.cleaning import remove_stopwords_from_list, run_cleaning
from utils.corpus import create_train_test_clean_modified, create_training, create_training_clean

def import_attr_lists(path):
    '''
    Import the attr list files

    path: the path to the file. Don't include file extension.
    '''
    with open(f"{path}.json", 'r') as f:
        attr_list = json.load(f)
        return attr_list
    
def convert_docs_to_input(attr_list, corpus):
    '''
    Convert documents into input for the decision tree model

    attr_list: list of terms to search for in documents
    corpus: list of strings representing text documents
    '''
    def mask_func(word_index, input_index):
        if attr_list[word_index] in corpus[input_index]:
            return 1
        else:
            return 0
    input_arr = np.zeros((len(corpus), len(attr_list)))
    for i in list(range(len(corpus))):
        for j in list(range(len(attr_list))):
            input_arr[i][j] = mask_func(j, i)
    return input_arr

def get_paths(clf, node, current_path, paths):
    if node == -1:
        return
    
    current_path.append(node)
    
    # If the node is a leaf, add the current path to the list of paths
    if clf.tree_.children_left[node] == clf.tree_.children_right[node]:
        paths.append(list(current_path))
    else:
        get_paths(clf, clf.tree_.children_left[node], current_path, paths)
        get_paths(clf, clf.tree_.children_right[node], current_path, paths)
    
    current_path.pop()

# Code from https://stackoverflow.com/questions/51397109/prune-unnecessary-leaves-in-sklearn-decisiontreeclassifier
# by users Thomas and Mattias Blume

def is_leaf(inner_tree, index):
    # Check whether node is leaf node
    return (inner_tree.children_left[index] == TREE_LEAF and 
            inner_tree.children_right[index] == TREE_LEAF)

def prune_index(inner_tree, decisions, index=0):
    # Start pruning from the bottom - if we start from the top, we might miss
    # nodes that become leaves during pruning.
    # Do not use this directly - use prune_duplicate_leaves instead.
    if not is_leaf(inner_tree, inner_tree.children_left[index]):
        prune_index(inner_tree, decisions, inner_tree.children_left[index])
    if not is_leaf(inner_tree, inner_tree.children_right[index]):
        prune_index(inner_tree, decisions, inner_tree.children_right[index])

    # Prune children if both children are leaves now and make the same decision:     
    if (is_leaf(inner_tree, inner_tree.children_left[index]) and
        is_leaf(inner_tree, inner_tree.children_right[index]) and
        (decisions[index] == decisions[inner_tree.children_left[index]]) and 
        (decisions[index] == decisions[inner_tree.children_right[index]])):
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
        ##print("Pruned {}".format(index))

def prune_duplicate_leaves(mdl):
    # Remove leaves if both 
    decisions = mdl.tree_.value.argmax(axis=2).flatten().tolist() # Decision for each node
    prune_index(mdl.tree_, decisions)

def main():
    # Load the dataset
    generation_mode = 'tfidf'
    num_augment = 50000
    val_split = 0.5
    data_path = f'./data/query-data/vote-query-stephen - vote-query-stephen_cleanedV2.csv'
    augment_path = './data/query-data/decahouse_polls_2020-10_cleanedV2.csv'
    attr_path = f'./data/attr-lists/2020-10-attr-lists-{num_augment}'
    figure_output_path = f'./data/decision-trees/2020-10-{generation_mode}-{num_augment}.png'
    gen_query_path = f'./data/generated-queries/2020-10-{generation_mode}-{num_augment}.json'
    if augment_path is None:
        if 'clean' in data_path:
            corpus_train, labels_train = create_training_clean(data_path, per_class=200)
        else:
            corpus_train, labels_train = create_training(data_path, per_class=200)
    else:
        corpus_train, labels_train, corpus_test, labels_test, _, _ = create_train_test_clean_modified(data_path, augment_path, num_augment, val_split)
    label_classes = ['Not Relevant', 'Relevant']
    if generation_mode == 'None' or 'NoneNoStop':
        vectorizer = CountVectorizer()
        vectorizer.fit(corpus_train)
        attrs = vectorizer.get_feature_names_out().tolist()
        if generation_mode == 'NoneNoStop':
            attrs = remove_stopwords_from_list(attrs)
        attr_lists = [attrs]
    else:
        attr_lists = import_attr_lists(attr_path)
    
    # DecisionTreeClassifier
    for attr_list in attr_lists:
        input_arrs = convert_docs_to_input(attr_list, corpus_train)
        np.set_printoptions(threshold=np.inf)
        tree_clf = DecisionTreeClassifier(criterion='entropy', max_depth=15, min_samples_leaf=50)
        tree_clf.fit(input_arrs, labels_train)
        prune_duplicate_leaves(tree_clf)
        fig = plt.figure(figsize=(100, 100))
        _ = tree.plot_tree(tree_clf, 
                        feature_names=attr_list,  
                        class_names=label_classes,
                        filled=True)
        fig.savefig(figure_output_path)
        paths = []
        get_paths(tree_clf, 0, [], paths)
        def node_to_feature_name(tree, node, next_node):
            attr = tree_clf.tree_.feature[node]
            if tree.tree_.children_left[node] == next_node: # Check if left child, if it is, negate the attribute
                return f"!{attr_list[attr]}"
            return attr_list[attr]
        boolean_queries = []
        for path in paths:
            boolean_query = []
            for i in range(len(path)):
                if i != len(path) - 1:
                    boolean_query.append(node_to_feature_name(tree_clf, path[i], path[i+1]))
                else:
                    boolean_query.append((label_classes[tree_clf.tree_.value[path[i]].argmax()], tree_clf.tree_.value[path[i]].tolist()))
            boolean_queries.append(boolean_query)
        # Now save queries as json to generated-queries folder
        with open(gen_query_path, 'w') as f:
            json.dump(boolean_queries, f, indent=2)

if __name__ == "__main__":
    main()