import spacy
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.matutils import sparse2full
import numpy as np



#### Functions to lemmatise docs
def keep_token(t):
    return (t.is_alpha and 
            not (t.is_space or t.is_punct or 
                 t.is_stop or t.like_num))
def lemmatize_doc(doc):
    return [ t.lemma_ for t in doc if keep_token(t)]


#Gensim to create a dictionary and filter out stop and infrequent words (lemmas).
def get_docs_dict(docs):
    docs_dict = Dictionary(docs)
    docs_dict.filter_extremes(no_below=20, no_above=0.2)
    docs_dict.compactify()
    return docs_dict


# Gensim can again be used to create a bag-of-words representation of each document,
# build the TF-IDF model, 
# and compute the TF-IDF vector for each document.
def get_tfidf(docs, docs_dict):
    docs_corpus = [docs_dict.doc2bow(doc) for doc in docs]
    model_tfidf = TfidfModel(docs_corpus, id2word=docs_dict)
    docs_tfidf  = model_tfidf[docs_corpus]
    docs_vecs   = np.vstack([sparse2full(c, len(docs_dict)) for c in docs_tfidf])
    return docs_vecs



# Get a TF-IDF weighted Glove vector summary of each document
# Input: a list of documents, Output: Matrix of vector for all the documents
def tfidf_weighted_wv(doc_list):
    #####Initialize######
    #Load spacy model
    nlp  = spacy.load('en')

    #lemmatise docs
    docs = [lemmatize_doc(nlp(doc)) for doc in doc_list]
    
    #Get docs dictionary
    docs_dict = get_docs_dict(docs)

    #tf-idf
    docs_vecs   = get_tfidf(docs, docs_dict)

    #Load glove embedding vector for each TF-IDF term
    tfidf_emb_vecs = np.vstack([nlp(docs_dict[i]).vector for i in range(len(docs_dict))])

    #To get a TF-IDF weighted Glove vector summary of each document, 
    #we just need to matrix multiply docs_vecs with tfidf_emb_vecs
    docs_emb = np.dot(docs_vecs, tfidf_emb_vecs)
    
    return docs_emb
