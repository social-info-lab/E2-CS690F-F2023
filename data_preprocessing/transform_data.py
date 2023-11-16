import pandas as pd
import string
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

def tokenization(text):
    text = re.split('\W+', text)
    return text

def remove_stopwords(text):
    stopword = nltk.corpus.stopwords.words('english')
    text = [word for word in text if word not in stopword]
    return text

def lemmatizer(text):
    wn = nltk.WordNetLemmatizer()
    text = [wn.lemmatize(word) for word in text]
    return text

def sentence(x):
    #print(x)
    return ' '.join(x)

def tfidf(x):
    vec = TfidfVectorizer()
    x = vec.fit_transform(x)
    arr = x.toarray()
    features = vec.get_feature_names_out()
    return pd.DataFrame(arr, columns=features)

def top_k_features(x, k=100):
    top_k = x.sum().sort_values(ascending=False)[:k]
    return x[top_k.axes[0]]

def transform_dataset():
    # Read in the data
    df_s = pd.read_csv("datasets/vote-query-stephen.csv")
    # rename the label column
    df_s = df_s.rename(columns={"relvant (1 == yes)": "relevant"})
    # Read in the first 10k lines    
    df_big = pd.read_csv("datasets/decahose_polls_2021-08.csv", nrows=10000)
    # merge the rows of the two datasets
    df = pd.concat([df_s, df_big], ignore_index=True)

    # Apply preprocessing steps to the text
    nltk.download('stopwords')
    nltk.download('wordnet')
    df['text'] = df['text'].apply(lambda x: remove_punct(x))
    df['text'] = df['text'].apply(lambda x: tokenization(x))
    df['text'] = df['text'].apply(lambda x: remove_stopwords(x))
    df['text'] = df['text'].apply(lambda x: lemmatizer(x))
    df['text'] = df['text'].apply(lambda x: sentence(x))

    # Apply tfidf to the text and 
    df_tfidf = tfidf(df['text'])
    # get the top k words with the highest tfidf sum as features 
    df_top = top_k_features(df_tfidf)
    # binarize
    df_top[df_top > 0] = 1
    # Add the label column
    df_top['label'] = df["relevant"]
    # Save the dataframe
    df_top.to_csv("datasets/polls-13k.csv", index=False)

if __name__ == "__main__":
    transform_dataset()