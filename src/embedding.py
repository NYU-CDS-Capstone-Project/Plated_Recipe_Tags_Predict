import numpy as np

def load_emb_vectors(fname):
    data = {}
    with open(fname, 'r') as f:
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            data[word] = embedding
    return data

def build_emb_weight(words_emb_dict, id2token):
    vocab_size = len(id2token)
    emb_dim = len(words_emb_dict['a'])
    emb_weight = np.zeros([vocab_size, emb_dim])
    for i in range(2,vocab_size):
        emb = words_emb_dict.get(id2token[i], None)
        if emb is not None:
            emb_weight[i] = emb
    return emb_weight