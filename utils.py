import pandas as pd
import numpy as np
import scipy as sp
import networkx as nx

import time
import random
from random import shuffle

from sklearn.preprocessing import normalize,StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score

from nltk.tokenize import RegexpTokenizer


#__authors__ = "Adrien Guille, Antoine Gourru"
#__email__ = "adrien.guille@univ-lyon2.fr, antoine.gourru@univ-lyon2.fr"


def compute_M(A):
    A = sp.sparse.csr_matrix(normalize(A, norm='l1', axis=1), dtype=np.float32)
    A2 = A @ A
    return A + A2

def evaluate(embeddings, labels, ratio, C, verbose=True):
    d = embeddings.shape[1]
    scores = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=ratio, random_state=i)
        classifier = LinearSVC(penalty='l2', loss='squared_hinge', dual=False, tol=0.0001, C=C, multi_class='ovr', fit_intercept=True, class_weight=None, random_state=i, max_iter=4000)
        classifier.fit(X_train, y_train)
        y_pred = []
        for x in X_test:
            x = x.reshape((1,d))
            y_pred.append(classifier.predict(x))
        y_pred = np.asarray(y_pred)
        scores.append(accuracy_score(y_test, y_pred)*100)

    accuracy_mean = np.mean(scores)
    accuracy_std = np.std(scores)
   
    if verbose:
        print('Accuracy (%.1f): %.3f (std: %.3f)' % (ratio, accuracy_mean, accuracy_std))
    return accuracy_mean, accuracy_std


def find_optimal_C(embeddings, labels):
    Cs = [1, 2, 4, 8]
    scores = [evaluate(embeddings, labels, 0.5, C, verbose=False) for C in Cs]
    accuracy_scores = [accuracy_mean for accuracy_mean, accuracy_std in scores]
    return Cs[accuracy_scores.index(max(accuracy_scores))]


def read_data(dataset):
    # features
    if dataset in ["nyt", "dblp","cora2"]:
        content = pd.read_csv("data/"+dataset+"/features.txt", sep="\t", header=None, quoting=3)
        vectorizer = TfidfVectorizer(lowercase=True, analyzer="word", stop_words="english", max_df=0.25, min_df=4, norm='l2', use_idf=True)
        features = vectorizer.fit_transform(content[1].values)
        vectorizerTF = TfidfVectorizer(lowercase=True, analyzer="word", stop_words="english", max_df=0.25, min_df=4, norm=None, use_idf=False)
        tf = vectorizerTF.fit_transform(content[1].values)
        tokenizer = RegexpTokenizer(r'\w+')
        raw = [tokenizer.tokenize(i.lower()) for i in content[1].values]
        en_stop = vectorizer.get_stop_words()
        for i in range(len(raw)):
            raw[i] = [word for word in raw[i] if not word in en_stop]
        voc = vectorizer.get_feature_names()
    else:
        print("Unknown dataset: %s" % dataset)
        return None
    # graph
    graph = nx.read_adjlist("data/"+dataset+"/graph.txt", nodetype=int)
    A = nx.to_scipy_sparse_matrix(graph, nodelist=range(features.shape[0]), format="csr")
    # labels
    groups = np.loadtxt("data/"+dataset+"/group.txt", delimiter="\t", dtype=int)
    groups = groups[:, 1]
    return features, groups, A, graph,voc,raw,tf
