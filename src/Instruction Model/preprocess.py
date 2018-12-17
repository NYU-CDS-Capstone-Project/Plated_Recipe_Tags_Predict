import pandas as pd
import numpy as np
import string
import pickle as pkl
import random
from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize

# lowercase and remove punctuation
def tokenizer(sent):
    #print(sent)
    if pd.isnull(sent):
        words = []
    else:
        table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        sent = sent.translate(table)
        tokens = word_tokenize(sent)
        # convert to lower case
        tokens = [w.lower() for w in tokens]
        # remove punctuation from each word
        #table = str.maketrans('', '', string.punctuation)
        #stripped = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        words = [word for word in tokens if word.isalpha()]
        #re.findall(r'\d+', 'sdfa')
    return words

def tokenize_dataset(step_n):
    """returns tokenization for each step, training set tokenizatoin"""
    token_dataset = []
    for sample in step_n:
        tokens = tokenizer(sample)
        token_dataset.append(tokens)
    return token_dataset

def all_tokens_list(train_data):
    """returns all tokens of instruction (all steps) for creating vocabulary"""
    all_tokens = []
    for columns in train_data.columns:
        for sample in train_data[columns]:
            all_tokens += sample[:] 
    return all_tokens

# save index 0 for unk and 1 for pad
def build_vocab(all_tokens, max_vocab_size):
    # Returns:
    # id2token: list of tokens, where id2token[i] returns token that corresponds to token i
    # token2id: dictionary where keys represent tokens and corresponding values represent indices
    PAD_IDX = 0
    UNK_IDX = 1
    token_counter = Counter(all_tokens)
    vocab, count = zip(*token_counter.most_common(max_vocab_size))
    id2token = list(vocab)
    token2id = dict(zip(vocab, range(2,2+len(vocab)))) 
    id2token = ['<pad>', '<unk>'] + id2token
    token2id['<pad>'] = PAD_IDX 
    token2id['<unk>'] = UNK_IDX
    return token2id, id2token

# convert token to id in the dataset
def token2index_dataset(tokens_data, token2id):
    """returns [[[step1 indices],[step2 indices],...,[step6 indices]],[],[],...]"""
    recipie_indices_data = []
    UNK_IDX = 1
    for recipie in tokens_data.iterrows():
        step_indices_data = []
        for step in recipie[1]:
            index_list = [token2id[token] if token in token2id else UNK_IDX for token in step]
            step_indices_data.append(index_list)
        recipie_indices_data.append(step_indices_data)
    return recipie_indices_data

