#!/usr/bin/env python
# coding: utf-8

# __authors__ = "Adrien Guille, Antoine Gourru"
# __email__ = "adrien.guille@univ-lyon2.fr, antoine.gourru@univ-lyon2.fr"


import pandas as pd
import numpy as np
import scipy as sp

from utils import read_data,evaluate, find_optimal_C

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from gensim.models import Word2Vec

from models import train_rle,train_w2v,train_geld,prepare_data

ratios = 0.5
methods = ["RLE"]  #"GELD"
dataset= "nyt"
d = 160

print("\nLoading %s..." % dataset)

tf_idf, groups, A, graph,voc,raw,tf = read_data(dataset)
n = A.shape[0]
density = 100 * 2 * len(list(graph.edges())) / (n * (n - 1))
degrees = np.array([degree for _, degree in graph.degree()])
mean_degree = degrees.mean()
print("%d nodes, %d features, %d edges (density: %.3f%%, mean degree: %.1f)" % (tf_idf.shape[0], 
                                                                                tf_idf.shape[1], 
                                                                                len(list(graph.edges())), 
                                                                                density,
                                                                                mean_degree))
embeddings = {}

print("\nTraining embeddings in dimension %d..." % d)

# -------------------------
# Learn the representations
# -------------------------
data_graph,data_text,sigma_init,D_init,U = prepare_data(d,tf,A,voc,raw)

if "GELD" in methods:
    optimal_alpha = {"cora2": 0.99, "dblp": 0.8,"nyt":0.95}
    D,D_norm,sigma = train_geld(d,data_graph,data_text,U,D_init,sigma_init,n_epoch=40,lamb=None,alpha=optimal_alpha[dataset])   
    embeddings["GELD"] = D_norm
    
if "RLE" in methods:
    optimal_window = {"cora2": 15, "dblp": 5,"nyt":10}
    embeddings["RLE"], _ = train_rle(A,tf,U,d,0.7,verbose=True)


# ------------------------------------------------------------------------------
# Train and evaluate the logistic regression for test/train ratios in [0.5, 0.9]
# ------------------------------------------------------------------------------

print("\nEvaluating embeddings...")
ratios = [0.9,0.5]
results = pd.DataFrame(ratios, columns=["ratio"])
for method in methods:
    optimal_C = find_optimal_C(embeddings[method], groups)
    print(optimal_C)
    results[method] = results["ratio"].apply(lambda ratio: evaluate(embeddings[method], 
                                                                    groups, 
                                                                    ratio, 
                                                                    C=optimal_C, 
                                                                    verbose=False))
                                      
results2 = pd.DataFrame(ratios, columns=["ratio"])
         
for method in methods:
    print(method+"%.1f",results[method].tolist()[0])
    results2[[method+"_accuracy_mean", method+"_accuracy_std"]] = pd.DataFrame(results[method].tolist(), index=results.index)


results2.to_csv(dataset+"_classification.csv")

