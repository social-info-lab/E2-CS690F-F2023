#########
# File containing document cleaning functions
#
# Code modified from: https://www.geeksforgeeks.org/python-efficient-text-data-cleaning/#
#########

import html
import re
from autocorrect import Speller
from nltk.corpus import stopwords

def remove_html(text):
    '''
    Remove any html tags in text. Returns modified text.

    text: input string
    '''
    return html.unescape(text)

def remove_links(text):
    '''
    Remove any links (http) found in text. Returns modified text.

    text: input string
    '''
    return re.sub(r'https?:\/\/.\S+', "", text)

def remove_symbol(symbols, text):
    '''
    Remove any symbols from the input text. Returns modified text.
    
    symbols: a list of symbols to remove
    text: input string
    '''
    return text.translate({ord(x): '' for x in symbols})

# def expand_contractions(text):
#     '''
#     Expand any contractions like n't to not. Returns modified text.

#     text: input string
#     '''
#     #dictionary consisting of the contraction and the actual value
#     Apos_dict={"'s":" is","n't":" not","'m":" am","'ll":" will",
#             "'d":" would","'ve":" have","'re":" are"}
#     #replace the contractions
#     for key,value in Apos_dict.items():
#         if key in text:
#             text=text.replace(key,value)
#     return text

# def word_separation(text):
#     '''
#     Split any words w/o spaces when separated by caps. DonaldTrump --> Donald Trump. Returns modified text.

#     text: input string
#     '''
#     text = " ".join([s for s in re.split("([A-Z][a-z]+[^A-Z]*)", text) if s])
#     return text

def lowercase_text(text):
    '''
    Lowercase the text so it is consistent. Returns modified text.

    text: input string
    '''
    return text.lower()

# def spellcheck_text(text):
#     '''
#     Run a spellchecker on the text. Returns modified text.

#     text: input string
#     '''
#     spell = Speller(lang='en')
#     return spell(text)

def remove_stopwords(text):
    '''
    Remove stopwords from text. Returns a list of words to be tokenized.

    text: input string
    '''
    stopwords_eng = stopwords.words('english')
    text_tokens=text.split()
    text_list=[]
    #remove stopwords
    for word in text_tokens:
        if word not in stopwords_eng:
            text_list.append(word)
    
    return text_list

common_symbols = [
    '!', '"', '#', '$', '%', '&', '(', ')', '*', '+', ',', 
    '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[',
    '\'',']', '^', '_', '`', '{', '|', '}', '~', '\'']

def run_cleaning(text, symbols=common_symbols, rm_stopwords=True):
    '''
    Run the functions above on the symbols list. Returns a list of words to be tokenized if rm_stopwords \
    is True, otherwise returns text.

    text: input text
    symbols: a list of symbols to remove
    '''
    no_html = remove_html(text)
    no_links = remove_links(no_html)
    no_symbols = remove_symbol(symbols, no_links)
    # no_contr = expand_contractions(no_symbols)
    # fix_spaces = word_separation(no_contr)
    lowercased = lowercase_text(no_symbols)
    # spellchecked = spellcheck_text(lowercased)
    if rm_stopwords:
        clean_stopwords = remove_stopwords(lowercased)
        return clean_stopwords
    else:
        return lowercased

def remove_stopwords_from_list(input_list):
    stopwords_eng = stopwords.words('english')
    return_list = []
    for word in input_list:
        if ' ' in word:
            word_arr = word.split()
            no_match = False
            for w in word_arr:
                if w in stopwords_eng:
                    break
                no_match = True
            if no_match:
                return_list.append(word)
        else:
            if word not in stopwords_eng:
                return_list.append(word)
    return return_list