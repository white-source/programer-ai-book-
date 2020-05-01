import gensim
word2vec = gensim.models.KeyedVectors.load_word2vec_format(embedding_path,binary=True)
word2vec