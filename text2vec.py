import spacy
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim import corpora, models, similarities
from gensim.matutils import sparse2full
import numpy as np



# Functions to lemmatise docs
def _keep_token(t):
    return (t.is_alpha and 
            not (t.is_space or t.is_punct or 
                 t.is_stop or t.like_num))
def _lemmatize_doc(doc):
    return [ t.lemma_ for t in doc if _keep_token(t)]



    


#Gensim to create a dictionary and filter out stop and infrequent words (lemmas).
def _get_docs_dict(docs):
    docs_dict = Dictionary(docs)
    docs_dict.filter_extremes(no_below=5, no_above=0.2)
    docs_dict.compactify()
    return docs_dict

# Preprocess docs
def _preprocess(doc_list):
    #Load spacy model
    nlp  = spacy.load('en')
    #lemmatise docs
    docs = [_lemmatize_doc(nlp(doc)) for doc in doc_list] 
    #Get docs dictionary
    docs_dict = _get_docs_dict(docs)
    return nlp, docs, docs_dict


# Gensim can again be used to create a bag-of-words representation of each document,
# build the TF-IDF model, 
# and compute the TF-IDF vector for each document.
def _get_tfidf(docs, docs_dict):
    docs_corpus = [docs_dict.doc2bow(doc) for doc in docs]
    model_tfidf = TfidfModel(docs_corpus, id2word=docs_dict)
    docs_tfidf  = model_tfidf[docs_corpus]
    docs_vecs   = np.vstack([sparse2full(c, len(docs_dict)) for c in docs_tfidf])
    return docs_vecs


#Get avg w2v for one document
def _document_vector(doc, docs_dict, nlp):
    # remove out-of-vocabulary words
    doc_vector = [nlp(word).vector for word in doc if word in docs_dict.token2id]
    return np.mean(doc_vector, axis=0)


# Get a TF-IDF weighted Glove vector summary for document list
# Input: a list of documents, Output: Matrix of vector for all the documents
def tfidf_weighted_wv(doc_list):
    #Initialize
    nlp, docs, docs_dict = _preprocess(doc_list)
    
    #tf-idf
    docs_vecs   = _get_tfidf(docs, docs_dict)

    #Load glove embedding vector for each TF-IDF term
    tfidf_emb_vecs = np.vstack([nlp(docs_dict[i]).vector for i in range(len(docs_dict))])

    #To get a TF-IDF weighted Glove vector summary of each document, 
    #we just need to matrix multiply docs_vecs with tfidf_emb_vecs
    docs_emb = np.dot(docs_vecs, tfidf_emb_vecs)
    
    return docs_emb

# Get average vector for document list
def avg_wv(doc_list):
    #Initialize
    nlp, docs, docs_dict = _preprocess(doc_list)
    
    docs_vecs = np.vstack([_document_vector(doc, docs_dict, nlp) for doc in docs])
    return docs_vecs

# Get TF-IDF vector for document list
def get_tfidf(doc_list):
    #Initialize
    nlp, docs, docs_dict = _preprocess(doc_list)
    
    docs_corpus = [docs_dict.doc2bow(doc) for doc in docs]
    model_tfidf = TfidfModel(docs_corpus, id2word=docs_dict)
    docs_tfidf  = model_tfidf[docs_corpus]
    docs_vecs   = np.vstack([sparse2full(c, len(docs_dict)) for c in docs_tfidf])
    return docs_vecs


# Get Latent Semantic Indexing(LSI) vector for document list
def get_lsi(doc_list, num_topics=300):
    #Initialize
    nlp, docs, docs_dict = _preprocess(doc_list)
    
    docs_corpus = [docs_dict.doc2bow(doc) for doc in docs]
    model_lsi = models.LsiModel(docs_corpus, num_topics, id2word=docs_dict)
    docs_lsi  = model_lsi[docs_corpus]
    docs_vecs   = np.vstack([sparse2full(c, len(docs_dict)) for c in docs_lsi])
    return docs_vecs

# Get Random Projections(RP) vector for document list
def get_rp(doc_list):
    #Initialize
    nlp, docs, docs_dict = _preprocess(doc_list)
    
    docs_corpus = [docs_dict.doc2bow(doc) for doc in docs]
    model_rp = models.RpModel(docs_corpus, id2word=docs_dict)
    docs_rp  = model_rp[docs_corpus]
    docs_vecs   = np.vstack([sparse2full(c, len(docs_dict)) for c in docs_rp])
    return docs_vecs

# Get Latent Dirichlet Allocation(LDA) vector for document list
def get_lda(doc_list, num_topics=100):
    #Initialize
    nlp, docs, docs_dict = _preprocess(doc_list)
    
    docs_corpus = [docs_dict.doc2bow(doc) for doc in docs]
    model_lda = models.LdaModel(docs_corpus, num_topics, id2word=docs_dict)
    docs_lda  = model_lda[docs_corpus]
    docs_vecs   = np.vstack([sparse2full(c, len(docs_dict)) for c in docs_lda])
    return docs_vecs

# Get Hierarchical Dirichlet Process(HDP) vector for document list
def get_hdp(doc_list):
    #Initialize
    nlp, docs, docs_dict = _preprocess(doc_list)
    
    docs_corpus = [docs_dict.doc2bow(doc) for doc in docs]
    model_hdp = models.HdpModel(docs_corpus, id2word=docs_dict)
    docs_hdp  = model_hdp[docs_corpus]
    docs_vecs   = np.vstack([sparse2full(c, len(docs_dict)) for c in docs_hdp])
    return docs_vecs