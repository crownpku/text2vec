# Text2Vec 
### Easily generate document/paragraph/sentence vectors and calculate similarity. 

## NOT USABLE, UNDER HEAVY DEVELOPMENT

Goal of this repository is to build a tool to easily generate document/paragraph/sentence vectors for similarity calculation and as input for further machine learning models.

We do such transformation by the following ways. 

1. Use Bag of Words (TODO)

2. Use Average of Word Embeddings (TODO)

3. Use Weighted Word Embeddings wrt TFIDF (DONE)

4. Use Gensim Doc2Vec (TODO)

Either way we can train again with our own data, or directly use the pre-trained doc2vec/word2vec models.

## Training (Optional):

* Initialize: Pre-trained Word Embeddings (Optional)
* Input: Large Corpus
* Output: Doc2Vec/Word2Vec models

## Inference:

* Initialize: Pre-trained Doc2Vec/Word2Vec model
* input: Single Document
* output: Vector of dimention N



## Reference

https://github.com/jhlau/doc2vec

https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/

https://github.com/sdimi/average-word2vec

https://github.com/bnjmacdonald/text2vec

https://github.com/taki0112/Vector_Similarity



