# Weighted Word Vector w.r.t TF-IDF


This document introduces the method of calculating document vector by using *weighted word vector w.r.t TF-IDF*

The idea is that we want to combine TF-IDF weights and the pretrained word embeddings for each word in the document to generate a more sophisticated document vector (compared to simple TF-IDF weights or average of all word embeddings within the document).

Let's say we have 1000 documents with a vocabulary of 3380 unique words. We get those 3380 unique words after some preprocessing like:

1. Get rid of space, punct, top or num
```
 def _keep_token(self, t):
     return (t.is_alpha and 
         not (t.is_space or t.is_punct or 
         t.is_stop or t.like_num))
```

2. Get rid of extreme rare words (less than 5 here) or extreme common words (more than 20% of total words)
```
docs_dict.filter_extremes(no_below=5, no_above=0.2)
```

3. Lemmatization to get the unique word for its different variants
```
def _lemmatize_doc(self, doc):
    return [ t.lemma_ for t in doc if self._keep_token(t)]
```

### Calculate TF-IDF weights

We calculate the TF-IDF weights for each of the documents in the document list:
```
def _get_tfidf(self, docs, docs_dict):
    docs_corpus = [docs_dict.doc2bow(doc) for doc in docs]
    model_tfidf = TfidfModel(docs_corpus, id2word=docs_dict)
    docs_tfidf  = model_tfidf[docs_corpus]
    docs_vecs   = np.vstack([sparse2full(c, len(docs_dict)) for c in docs_tfidf])
    return docs_vecs
    
#tf-idf
docs_tfidf   = self._get_tfidf(self.docs, self.docs_dict)
```

*Now `docs_tfidf` is a matrix of shape 1000x3380. Each row is the TF-IDF vector with length 3380, with each column the TF-IDF weight for the corresponding weight for a unique word.*

### Get Word Embeddings for words in vocabulary

We have a pre-trained word embedding model. Each word in our model has a word vector of length 384.

Remember that for our current case, we have a vocabulary of size 3380. For each word out of the vocabulary, we get their word embeddings, each of length 384:

```
#Load glove embedding vector for each TF-IDF term
tfidf_emb_vecs = np.vstack([self.nlp(self.docs_dict[i]).vector for i in range(len(self.docs_dict))])
```

*Now we get the `tfidf_emb_vecs` which is a matrix of shapre 3380x384. Each row is a word in the vocabulary with its 384 dimension of pre-trained word vector.*

### Get Weighted Word Vector w.r.t TF-IDF

We have the `docs_tfidf` with shape 1000x3380, and `tfidf_emb_vecs` with shape 3380x384. To get the Weighted Word Vector w.r.t TF-IDF, we simply need to multiply the two matrices. Please carefully re-visit the meaning of these two matrices if you feel confused.

```
docs_emb = np.dot(docs_tfidf, tfidf_emb_vecs)
```

Now we get 'docs_emb' which is a matrix of size 1000x384. Each row is a document, with its *Weighted Word Vector w.r.t TF-IDF* of dimension 384.



To wrap-up, here is the part of code in text2vec.py:
```
def tfidf_weighted_wv(self):
        #tf-idf
        docs_vecs   = self._get_tfidf(self.docs, self.docs_dict)

        #Load glove embedding vector for each TF-IDF term
        tfidf_emb_vecs = np.vstack([self.nlp(self.docs_dict[i]).vector for i in range(len(self.docs_dict))])

        #To get a TF-IDF weighted Glove vector summary of each document, 
        #we just need to matrix multiply docs_vecs with tfidf_emb_vecs
        docs_emb = np.dot(docs_vecs, tfidf_emb_vecs)

        return docs_emb
```