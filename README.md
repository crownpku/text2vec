# Text2Vec 
### Easily generate document/paragraph/sentence vectors and calculate similarity. 

### [中文Blog](http://www.crownpku.com/2018/03/30/Text2Vec-%E7%AE%80%E5%8D%95%E7%9A%84%E6%96%87%E6%9C%AC%E5%90%91%E9%87%8F%E5%8C%96%E5%B7%A5%E5%85%B7.html)

Goal of this repository is to build a tool to easily generate document/paragraph/sentence vectors for similarity calculation and as input for further machine learning models.

## Requirements
* spacy2.0 (with English model downloaded and installed)
* gensim
* numpy

## Usage of Text to Vector (text2vec)

* Initialize: Pre-trained Doc2Vec/Word2Vec model
```
import text2vec
```

* input: List of Documents, doc_list is a list of documents/paragraphs/sentences.
```
t2v = text2vec.text2vec(doc_list)
```

* output: List of Vectors of dimention N

We do such transformation by the following ways. 

```
# Use TFIDF
docs_tfidf = t2v.get_tfidf()

# Use Latent Semantic Indexing(LSI)
docs_lsi = t2v.get_lsi()

# Use Random Projections(RP)
docs_rp = t2v.get_rp()

# Use Latent Dirichlet Allocation(LDA)
docs_lda = t2v.get_lda()

# Use Hierarchical Dirichlet Process(HDP)
docs_hdp = t2v.get_hdp()

# Use Average of Word Embeddings
docs_avgw2v = t2v.avg_wv()

# Use Weighted Word Embeddings wrt. TFIDF
docs_emb = t2v.tfidf_weighted_wv()
```

For a more detailed introduction of using Weighted Word Embeddings wrt. TFIDF, please read [here](https://github.com/crownpku/text2vec/blob/master/wv_wrt_tfidf.md).


## Usage of Similarity Calculation (simical)

For example, we want to calculate the similarity/distance between the first two sentences in the docs_emb we just computed.

Note that cosine similarity is between 0-1 (1 is most similar while 0 is least similar).
For the other similarity measurements the results are actually **distance** (the larget the less similar). It's better to calculate distance for all possible pairs and then rank.

```
# Initialize
import text2vec
sc = text2vec.simical(docs_emb[0], docs_emb[1])

# Use Cosine
simi_cos = sc.Cosine()

# Use Euclidean
simi_euc = sc.Euclidean()

# Use Triangle's Area Similarity (TS)
simi_ts = sc.Triangle()

# Use Sector's Area Similairity (SS)
simi_ss = sc.Sector()

# Use TS-SS
simi_ts_ss = sc.TS_SS()
```

## Reference

https://radimrehurek.com/gensim/tut2.html

https://github.com/sdimi/average-word2vec

https://github.com/taki0112/Vector_Similarity



