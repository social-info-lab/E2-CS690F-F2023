import pandas as pd
from utils.cleaning import run_cleaning
import csv
import os

def clean_corpus(path):
    '''
    Clean the corpus. Returns a list of strings.

    path: path to the file to load data from. Should be a csv.
    '''
    cleaned_corpus_path = path[:-4] + "_cleaned.csv"
    csv_fields = ['tweet_id', 'text', 'options', 'relevant']
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset='tweet_id', keep='first')
    corpus = df['text'].tolist()
    lines_processed = 0
    try:
        with open(cleaned_corpus_path) as f:
            reader = csv.reader(f)
            lines_processed = sum(1 for _ in reader) - 1
            if lines_processed < 0:
                lines_processed = 0
    except:
        lines_processed = 0

    print('Writing to file:', cleaned_corpus_path)
    print('Lines processed so far:', lines_processed)

    with open(cleaned_corpus_path, 'a+') as csvfile:  
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)  
        
        if os.stat(cleaned_corpus_path).st_size == 0:
            # writing the fields  
            csvwriter.writerow(csv_fields)  
        
        for i in range(lines_processed, len(corpus)):
            text = corpus[i]
            cleaned_text = run_cleaning(str(text), rm_stopwords=False)
            cur_row = [df['tweet_id'].iloc[i], cleaned_text, df['options'].iloc[i], df['relevant'].iloc[i]]
            csvwriter.writerow(cur_row) 

def clean_corpus_parallel(df, cleaned_path):
    '''
    Clean the corpus. Returns a list of strings.

    df: dataframe to load data from. Should be a subset of a dataframe
    path: path to the file to load data from. Should be a csv.
    '''
    csv_fields = ['tweet_id', 'text', 'options', 'relevant']
    corpus = df['text'].tolist()
    lines_processed = 0
    try:
        with open(cleaned_path) as f:
            reader = csv.reader(f)
            lines_processed = sum(1 for _ in reader) - 1
            if lines_processed < 0:
                lines_processed = 0
    except:
        lines_processed = 0

    print('Writing to file:', cleaned_path)
    print('Lines processed so far:', lines_processed)
    print('Total lines to process:', len(corpus))

    with open(cleaned_path, 'a+') as csvfile:  
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)  
        
        if os.stat(cleaned_path).st_size == 0:
            # writing the fields  
            csvwriter.writerow(csv_fields)  
        
        for i in range(lines_processed, len(corpus)):
            text = corpus[i]
            cleaned_text = run_cleaning(str(text), rm_stopwords=False)
            cur_row = [df['tweet_id'].iloc[i], cleaned_text, df['options'].iloc[i], df['relevant'].iloc[i]]
            csvwriter.writerow(cur_row) 

def create_training(path, per_class=100):
    '''
    Create training data for the decision tree model. Returns a list of strings and a list of labels.

    path: path to the file to load data from. Should be a csv.
    per_class: number of samples to take from each class
    '''
    df = pd.read_csv(path).dropna()
    relevant = df['relevant'] == 1
    rel_input = df['text'][relevant][:per_class] # get first 100 relevant queries
    rel_labels = df['relevant'][relevant][:per_class]
    irrelevant = df['relevant'] == 0
    irrel_input = df['text'][irrelevant][:per_class] # get first 100 irrelevant queries
    irrel_labels = df['relevant'][irrelevant][:per_class]
    corpus = rel_input.tolist() + irrel_input.tolist()
    corpus = [run_cleaning(text, rm_stopwords=False) for text in corpus]
    labels = rel_labels.tolist() + irrel_labels.tolist()
    return corpus, labels

def create_training_clean(path, per_class=100):
    '''
    Create training data for the decision tree model. Returns a list of strings and a list of labels.

    path: path to the file to load data from. Should be a csv and cleaned version. cleaned version is defined as having run
          run_cleaning on the text column
    per_class: number of samples to take from each class
    '''
    df = pd.read_csv(path).dropna()
    relevant = df['relevant'] == 1
    rel_input = df['text'][relevant][:per_class] # get first 100 relevant queries
    rel_labels = df['relevant'][relevant][:per_class]
    irrelevant = df['relevant'] == 0
    irrel_input = df['text'][irrelevant][:per_class] # get first 100 irrelevant queries
    irrel_labels = df['relevant'][irrelevant][:per_class]
    corpus = rel_input.tolist() + irrel_input.tolist()
    labels = rel_labels.tolist() + irrel_labels.tolist()
    return corpus, labels

def create_test(path, per_class_training=100, per_class_test=100):
    '''
    Create test data for the decision tree model. Returns a list of strings and a list of labels.

    path: path to the file to load data from. Should be a csv.
    per_class_training: number of samples to skip from each class (reserved for training set)
    per_class_test: number of samples to take from each class for testing
    '''
    df = pd.read_csv(path).dropna()
    relevant = df['relevant'] == 1
    irrelevant = df['relevant'] == 0
    rel_input = df['text'][relevant][per_class_training:per_class_training+per_class_test]
    rel_labels = df['relevant'][relevant][per_class_training:per_class_training+per_class_test]
    irrel_input = df['text'][irrelevant][per_class_training:per_class_training+per_class_test]
    irrel_labels = df['relevant'][irrelevant][per_class_training:per_class_training+per_class_test]
    corpus = rel_input.tolist() + irrel_input.tolist()
    corpus = [run_cleaning(text, rm_stopwords=False) for text in corpus]
    labels = rel_labels.tolist() + irrel_labels.tolist()
    return corpus, labels

def create_test_clean(path, per_class_training=100, per_class_test=100):
    '''
    Create test data for the decision tree model. Returns a list of strings and a list of labels.

    path: path to the file to load data from. Should be a csv. cleaned version is defined as having run
            run_cleaning on the text column
    per_class_training: number of samples to skip from each class (reserved for training set)
    per_class_test: number of samples to take from each class for testing
    '''
    df = pd.read_csv(path).dropna()
    relevant = df['relevant'] == 1
    irrelevant = df['relevant'] == 0
    rel_input = df['text'][relevant][per_class_training:per_class_training+per_class_test]
    rel_labels = df['relevant'][relevant][per_class_training:per_class_training+per_class_test]
    irrel_input = df['text'][irrelevant][per_class_training:per_class_training+per_class_test]
    irrel_labels = df['relevant'][irrelevant][per_class_training:per_class_training+per_class_test]
    corpus = rel_input.tolist() + irrel_input.tolist()
    labels = rel_labels.tolist() + irrel_labels.tolist()
    return corpus, labels

def create_train_test_clean_modified(train_path, test_and_aug_path, augment_amt, validation_percent):
    '''
    A custom function to handle our modified ds. We take the human labeled data and augment it with a machine labeled one
    for non-relevant values. Since we don't care about equal per_class size, we just leave those be and keep track of
    how much non-relevant data from the machine labeled dataset we use (so we don't use the same data twice).

    train_path: path to the human labeled data
    test_and_aug_path: path to the machine labeled data
    augment_amt: number of non-relevant data to augment with
    validation_percent: percentage of test data to use for validation
    '''
    df_train = pd.read_csv(train_path).dropna()
    df_aug_all = pd.read_csv(test_and_aug_path).dropna()
    df_aug_irrel = df_aug_all[df_aug_all['relevant'] == 0]
    df_train = pd.concat([df_train, df_aug_irrel[:augment_amt]])
    df_test = df_aug_all.drop(df_aug_irrel.index[:augment_amt])
    # create train rel and irrel labels and corpus
    relevant = df_train['relevant'] == 1
    irrelevant = df_train['relevant'] == 0
    rel_input = df_train['text'][relevant]
    rel_labels = df_train['relevant'][relevant]
    irrel_input = df_train['text'][irrelevant]
    irrel_labels = df_train['relevant'][irrelevant]
    corpus_train = rel_input.tolist() + irrel_input.tolist()
    labels = rel_labels.tolist() + irrel_labels.tolist()
    # create test rel and irrel labels and corpus
    relevant = df_test['relevant'] == 1
    irrelevant = df_test['relevant'] == 0
    rel_input = df_test['text'][relevant]
    rel_labels = df_test['relevant'][relevant]
    irrel_input = df_test['text'][irrelevant]
    irrel_labels = df_test['relevant'][irrelevant]
    num_rel = len(rel_input)
    num_irrel = len(irrel_input)
    validation_amt_rel = int(validation_percent * num_rel)
    validation_amt_irrel = int(validation_percent * num_irrel)
    corpus_val = rel_input.tolist()[:validation_amt_rel] + irrel_input.tolist()[:validation_amt_irrel]
    labels_val = rel_labels.tolist()[:validation_amt_rel] + irrel_labels.tolist()[:validation_amt_irrel]
    corpus_test = rel_input.tolist()[validation_amt_rel:] + irrel_input.tolist()[validation_amt_irrel:]
    labels_test = rel_labels.tolist()[validation_amt_rel:] + irrel_labels.tolist()[validation_amt_irrel:]
    return corpus_train, labels, corpus_test, labels_test, corpus_val, labels_val

def create_train_val_test_clean(data_path, perc_train, perc_val, subset_perc=None):
    '''
    Create the train/val/test split for the data. Returns a list of strings and a list of labels.

    data_path: path to the file to load data from. Should be a csv. cleaned version is defined as having run
               run_cleaning on the text column
    perc_train: percentage of data to use for training, note we don't care about perc_test since we assume that's the
                entire dataset minus the training set
    perc_val: percentage of data to use for validation
    subset_size: if not None, only use a subset % of the data
    '''
    df = pd.read_csv(data_path).dropna()
    relevant = df['relevant'] == 1
    irrelevant = df['relevant'] == 0
    rel_input = df['text'][relevant]
    rel_labels = df['relevant'][relevant]
    irrel_input = df['text'][irrelevant]
    irrel_labels = df['relevant'][irrelevant]
    num_rel = len(rel_input)
    num_irrel = len(irrel_input)
    if subset_perc is not None:
        num_rel = int(subset_perc * num_rel)
        num_irrel = int(subset_perc * num_irrel)
    train_amt_rel = int(perc_train * num_rel)
    train_amt_irrel = int(perc_train * num_irrel)
    val_amt_rel = int(perc_val * num_rel)
    val_amt_irrel = int(perc_val * num_irrel)
    corpus_train = rel_input.tolist()[:train_amt_rel] + irrel_input.tolist()[:train_amt_irrel]
    labels_train = rel_labels.tolist()[:train_amt_rel] + irrel_labels.tolist()[:train_amt_irrel]
    corpus_val = rel_input.tolist()[train_amt_rel:train_amt_rel+val_amt_rel] + irrel_input.tolist()[train_amt_irrel:train_amt_irrel+val_amt_irrel]
    labels_val = rel_labels.tolist()[train_amt_rel:train_amt_rel+val_amt_rel] + irrel_labels.tolist()[train_amt_irrel:train_amt_irrel+val_amt_irrel]
    corpus_test = rel_input.tolist()[train_amt_rel+val_amt_rel:] + irrel_input.tolist()[train_amt_irrel+val_amt_irrel:]
    labels_test = rel_labels.tolist()[train_amt_rel+val_amt_rel:] + irrel_labels.tolist()[train_amt_irrel+val_amt_irrel:]
    return corpus_train, labels_train, corpus_test, labels_test, corpus_val, labels_val
