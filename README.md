# Doc2vec: Gensim Doc2Vec or Average of Word Vectors

## NOT USABLE, UNDER HEAVY DEVELOPMENT


Goal of this repository is to build a tool to easily generate document/paragraph/sentence vectors for further machine learning models.



We do such transformation by the following two ways. 

1. Use Gensim Doc2Vec

2. Use Average of Word Vectors

Either way we can train again with our own data, or directly use the pre-trained doc2vec/word2vec models.

## Training (Optional):

* input: Large Corpus
* output: Doc2Vec/Word2Vec models

## Inference:

* input: Single Document; Pre-trained Doc2Vec/Word2Vec model
* output: Vector of dimention N



## Reference

https://github.com/jhlau/doc2vec

https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-IMDB.ipynb

https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-lee.ipynb

https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-wikipedia.ipynb