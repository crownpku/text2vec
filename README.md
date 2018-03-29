# Text2Vec 
### Easily generate document/paragraph/sentence vectors and calculate similarity. 


Goal of this repository is to build a tool to easily generate document/paragraph/sentence vectors for similarity calculation and as input for further machine learning models.

## Usage

We do such transformation by the following ways. 
doc_list is a list of documents/paragraphs/sentences.

1. Use TFIDF
```
import text2vec
docs_tfidf = text2vec.get_tfidf(doc_list)
```

2. Use Average of Word Embeddings
```
import text2vec
docs_avgw2v = text2vec.avg_wv(doc_list)
```

3. Use Weighted Word Embeddings wrt TFIDF
```
import text2vec
docs_emb = text2vec.tfidf_weighted_wv(doc_list)
```

4. Use Gensim Doc2Vec (TODO)

Either way we can train again with our own data, or directly use the pre-trained doc2vec/word2vec models.



### Training (Optional) (TODO)

* Initialize: Pre-trained Word Embeddings (Optional)
* Input: Large Corpus
* Output: Doc2Vec/Word2Vec models

### Inference:

* Initialize: Pre-trained Doc2Vec/Word2Vec model
* input: List of Documents
* output: List of Vectors of dimention N



## Reference

https://github.com/jhlau/doc2vec

https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/

https://github.com/sdimi/average-word2vec

https://github.com/bnjmacdonald/text2vec

https://github.com/taki0112/Vector_Similarity



