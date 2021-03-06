# use package to train word embeedding with recipe data

import word2vec
word2vec.word2vec(train='tokenized_instructions_train_with_plated.txt', 
                  output='vocab.bin', 
                  size=50,
                  window=10,
                  sample="1e-3",
                  hs=1,
                  negative=0,
                  threads=20,
                  iter_=10,
                  min_count=10,
                  alpha=0.025,
                  debug=2,
                  binary=1,
                  cbow=0,
                  save_vocab=None,
                  read_vocab=None,
                  verbose=True)
