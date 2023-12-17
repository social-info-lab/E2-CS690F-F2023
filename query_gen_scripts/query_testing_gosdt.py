#########
# File containing functions related to testing generated queries for performance on the test data
#########

import json
import numpy as np
from utils.corpus import create_train_test_clean_modified, create_test, create_test_clean
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

def import_gen_queries(path):
    '''
    Import the generated queries

    path: the path to the query file (should not include json extension)
    '''
    with open(f"{path}.json", 'r') as f:
        gen_queries = json.load(f)
        return gen_queries

def query_weight_score(not_relevant, relevant):
    '''
    Score function that weights relevancy with the number of relevant items

    not_relevant: number of items that are not relevant
    relevant: number of items that are relevant
    '''
    weight_relevance = 0.7  # Weight for relevance percentage
    weight_count = 0.3  # Weight for the total number of relevant items

    # Calculate the score
    score = (weight_relevance * relevant / (not_relevant + relevant)) + (weight_count * relevant)

    return score

def evaluate_results(pred, actual):
    '''
    Evaluate the results of the model

    pred: predicted labels
    actual: actual labels
    '''
    accuracy = accuracy_score(actual, pred)
    f1 = f1_score(actual, pred)
    precision = precision_score(actual, pred,zero_division=0 )
    recall = recall_score(actual, pred)
    cm = confusion_matrix(actual,pred)
    return accuracy, f1, precision, recall, cm

def process_queries(gen_queries):
    '''
    Given the generated queries, process them into a list of tuples of the form:
    (query: string, (num_rel: float, num_irrel: float))

    gen_queries: the generated queries
    '''
    processed_queries = []
    for query in gen_queries:
        text = query[:-1]
        rel_tuple = query[-1]
        # We only want relevant generated queries
        if rel_tuple[0] == 'Relevant':
            query_text = " AND ".join(text)
            processed_queries.append((query_text, 0))
    return sorted(processed_queries, key=lambda x: x[1], reverse=True)

def generate_results(generation_mode, corpus_test, labels_test, corpus_val, labels_val, regularization=0.05, num_attr=200):
    # generation_mode = 'tfidf'
    # num_augment = 50000
    val_split = 0.5
    testing_mode = 'val'
    output_path = f'./data/query-results/gosdt-guesses/2020-10-{generation_mode}-{regularization}-{num_attr}.txt'
    gen_query_path = f'./data/generated-queries/gosdt-guesses/2020-10-{generation_mode}-{regularization}-{num_attr}-gosdt'
    
    # import generated queries
    gen_queries = import_gen_queries(gen_query_path)
    k = 50 if len(gen_queries) >= 50 else len(gen_queries)
    # Take generated queries, and grab the top k queries from the model
    processed_queries = process_queries(gen_queries)[:k]
    # Run the queries on the test data
    output_string = ''
    testing_set = corpus_val if testing_mode == 'val' else corpus_test
    testing_labels = labels_val if testing_mode == 'val' else labels_test
    for query in processed_queries:
        print(f"Query: {query[0]}")
        output_string += f"Query: {query[0]}\n"
        preds = []
        for i in range(len(testing_set)):
            terms = query[0].split(' AND ')
            pred = 1
            for term in terms:
                if term[0] == '!':
                    if term[1:] in testing_set[i]:
                        pred = 0
                        break
                elif term not in testing_set[i]:
                    pred = 0
                    break
            preds.append(pred)
        # Evaluate the results
        accuracy, f1, precision, recall, cm = evaluate_results(preds, testing_labels)
        output_string += f"Accuracy: {accuracy}\n"
        print(f"Accuracy: {accuracy}")
        output_string += f"F1: {f1}\n"
        print(f"F1: {f1}")
        output_string += f"Precision: {precision}\n"
        print(f"Precision: {precision}")
        output_string += f"Recall: {recall}\n"
        print(f"Recall: {recall}")
        output_string += f'Confusion Matrix: \n{cm}\n'
        print(f"Confusion Matrix: \n{cm}")
        with open(output_path, 'w') as f:
            f.write(output_string)
            

            


