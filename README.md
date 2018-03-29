# Text2Vec 
### Easily generate document/paragraph/sentence vectors and calculate similarity. 


Goal of this repository is to build a tool to easily generate document/paragraph/sentence vectors for similarity calculation and as input for further machine learning models.

## Usage

We do such transformation by the following ways. 

doc_list is a list of documents/paragraphs/sentences.

* Initialize: Pre-trained Doc2Vec/Word2Vec model
* input: List of Documents
* output: List of Vectors of dimention N


#### Use TFIDF
```
import text2vec
docs_tfidf = text2vec.get_tfidf(doc_list)
```

#### Use Latent Semantic Indexing(LSI)
```
import text2vec
docs_lsi = text2vec.get_lsi(doc_list)
```

#### Use Random Projections(RP)
```
import text2vec
docs_rp = text2vec.get_rp(doc_list)
```

#### Use Latent Dirichlet Allocation(LDA)
```
import text2vec
docs_lda = text2vec.get_lda(doc_list)
```

#### Use Hierarchical Dirichlet Process(HDP)
```
import text2vec
docs_hdp = text2vec.get_hdp(doc_list)
```

#### Use Average of Word Embeddings
```
import text2vec
docs_avgw2v = text2vec.avg_wv(doc_list)
```

#### Use Weighted Word Embeddings wrt. TFIDF
```
import text2vec
docs_emb = text2vec.tfidf_weighted_wv(doc_list)
```

### Similarity Calculation
Coming up soon


## Reference

https://radimrehurek.com/gensim/tut2.html

https://github.com/jhlau/doc2vec

https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/

https://github.com/sdimi/average-word2vec

https://github.com/bnjmacdonald/text2vec

https://github.com/taki0112/Vector_Similarity



