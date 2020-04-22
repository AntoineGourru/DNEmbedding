#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy as sp

from utils import read_data,evaluate, find_optimal_C

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from gensim.models import Word2Vec

def train_rle(A,T,d, lamb,raw,voc,window,negative,verbose=True):
    if verbose:
        print('RLE - d=%d' % d)
        
    #voc = [*voc]
    
    w2v = Word2Vec(raw,size=d,window = window,min_count=1,iter = 200,negative = negative)
    
    U = np.zeros((len(voc), d))
    for i in range(len(voc)):
        U[i,:] = w2v[voc[i]]        
    
    #G = normalize(adj, norm='l1', axis=1) 
    start = time.time()
    
    G = sp.sparse.csr_matrix(normalize(compute_M(A), norm='l1', axis=1), dtype=np.float32)
    T = sp.sparse.csr_matrix(normalize(T, norm='l1', axis=1), dtype=np.float32)
    
    Tprime = G @ T

    O = T * (1 - lamb) + Tprime * lamb

    rle_embeddings = O @ U

    rle_embeddings = normalize(rle_embeddings,axis=0)

    training_time = time.time() - start
    if verbose:
        print('Training time: %.1f' % training_time)
    
    return rle_embeddings, training_time


def train_w2v(d,voc,raw,window,negative):

    w2v = Word2Vec(raw,size=d,window=window,iter = 200,min_count=1,negative=negative)

    U = np.zeros((len(voc), d))
    for i in range(len(voc)):
        U[i,:] = w2v[voc[i]]  
        
    return U

def prepare_data(d,tf,A,voc,raw,window=15,negative=5):
    
    U = train_w2v(d,voc,raw,window,negative)
    print("U done")
    
    N = A.shape[0]
         
    ind = np.argwhere(tf > 0)
    data_text = [[] for _ in range(N)]
    for dat in ind:
        data_text[dat[0]].append([dat[1],tf[dat[0],dat[1]]]) 
    print("text done")

    ind = np.argwhere(A > 0)
    data_graph = [[] for _ in range(N)]
    for dat in ind:
        data_graph[dat[0]].append([dat[1],A[dat[0],dat[1]],1]) 
        data_graph[dat[1]].append([dat[0],A[dat[0],dat[1]],-1]) 
    print("graph done")
    
    sigma = np.zeros((N,d))

    sig_def = np.std(U, axis=0)
    for i in range(N):
        nz = tf[i].nonzero()[1]
        if len(nz) < 2:
            sigma[i] = sig_def * sig_def
        else:
            sig = np.std(U[tf[i].nonzero()[1]], axis=0)
            sigma[i] = sig * sig
               
    T = sp.sparse.csr_matrix(normalize(tf, norm='l1', axis=1), dtype=np.float32)
    D = T @ U
    
    D
    
    return data_graph,data_text,sigma,D,U
    
def train_geld(d,data_graph,data_text,U,D_init,sigma_init,n_epoch=20,lamb=None,alpha=0.99,groups = None, test = False):
    
    if lamb == None:
        lamb = np.power(range(1,n_epoch+1),-0.2)*0.1
    
    N = D_init.shape[0]
    
    D = D_init.copy()
    sigma = sigma_init.copy()
    
    for epoch in range(n_epoch):
        
        aine = np.random.choice(N, N, replace=False)
        for i in aine:
            mu_opt = np.zeros(d)
            denom = 0
            dat = data_graph[i]
            l_graph = len(dat)
            if(l_graph != 0):
                for obs in dat:

                    indicateur_p = ( obs[2] + 1 ) / 2
                    indicateur_n = 1 - indicateur_p
                    temp = ( indicateur_p * sigma[i] + indicateur_n * sigma[obs[0]] )
                    pond = obs[1] / temp

                    #print(pond)
                    mu_opt +=  alpha * (D[obs[0]] * pond)
                    denom += alpha *  pond

            dat = data_text[i]
            l_text = len(dat)
            if(l_text != 0) :
                for obs in dat:

                    mu_opt +=  (1 - alpha) * (U[obs[0]] * obs[1])/sigma[i]
                    denom += (1 - alpha) *  (obs[1]/sigma[i])

            if (l_text + l_graph)!=0:
                D[i] =  (1- lamb[epoch]) * D[i] + (lamb[epoch]) *  (mu_opt / denom)

        aine = np.random.choice(N, N, replace=False)
        for i in aine:
            sig_opt = np.zeros(d)
            denom = 0

            dat = data_graph[i]
            l_graph = len(dat)
            if(l_graph != 0):
                for obs in dat:
                    if obs[2] == 1:
                        dist = D[i] - D[obs[0]]
                        d2 = dist * dist
                        sig_opt += alpha * obs[1] * d2
                        denom += alpha *  obs[1]

            dat = data_text[i]
            l_text = len(dat)
            if(l_text != 0) :
                for obs in dat:

                    dist = D[i] - U[obs[0]]
                    d2 = dist * dist
                    sig_opt += (1-alpha) * obs[1] * d2
                    denom += (1-alpha) *  obs[1]
            if (l_text + l_graph)!=0:
                sigma[i] =  (1- lamb[epoch]) * sigma[i] + (lamb[epoch]) *  (sig_opt / denom)
    
        if test:
            D_norm = normalize(D, axis=1) 
            optimal_C = find_optimal_C(D_norm, groups)
            print(evaluate(D_norm,groups,0.5,C=optimal_C,verbose=False)[0])
            
    D_norm = normalize(D, axis=1)
       
    return D,D_norm,sigma
