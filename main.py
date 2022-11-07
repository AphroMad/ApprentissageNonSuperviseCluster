#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 16:20:59 2022

@author: marsaa
"""


from sklearn import metrics
import kmedoids
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
import time 
import matplotlib.pyplot as plt
from scipy.io import arff
import numpy as np 
from sklearn import cluster, metrics
import scipy.cluster.hierarchy as shc
import hdbscan
import uuid
import pandas as pd
  
def show_plot(f0,f1,c, title):
    plt.scatter ( f0 , f1 , c=c, s = 8 )
    plt.title (title)
    plt.show ()

def save_plot(f0,f1,c, title):
    plt.scatter ( f0 , f1 , c=c, s = 8 )
    plt.title (title)
    plt.savefig(out+title+'.png')
    plt.close()
    
def kmeans(data,k):
    model = cluster.KMeans(n_clusters=k, init="k-means++")
    model.fit(data)
    return (model)

def kmenoids(data,k):
    distmatrix = euclidean_distances(data)
    fp = kmedoids.fasterpam(distmatrix,k)
    return (fp) 

def agglomerative(data,seuil,linkage):
    model = cluster.AgglomerativeClustering ( distance_threshold = seuil,linkage = linkage , n_clusters = None )
    model.fit(data)
    return model

def dbscan(data,eps,min_sample):
    model = cluster.DBSCAN(eps=eps,min_samples=min_sample)
    model.fit(data)
    return model

def hdbscanf(data,min_cluster_size):
    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    model.fit(data)
    return model

def bouldin_score_kmeans(data, f0, f1, k_list):
    best_score_kmeans= 100000
    best_k_kmeans = 1
    best_score_time_kmeans = 0
    best_result = None
    
    for k in k_list:
        
        # kmeans
        tp1 = time.time()
        result = kmeans(data,k)
        tp2 = time.time()
        score = metrics.davies_bouldin_score(data,result.labels_)
        if (score<best_score_kmeans):
            best_score_kmeans = score
            best_k_kmeans = k
            best_score_time_kmeans = tp2-tp1
            best_result = result

    text = ("KMEANS [bouldin]\n best_k="+str(best_k_kmeans)+" (score = "+str(round(best_score_kmeans,3))+") ["+str(round(best_score_time_kmeans*1000,3))+" ms]")
    print(text)
    save_plot(f0, f1, c=best_result.labels_, title=text) 

def bouldin_score_kmenoids(data, f0, f1, k_list):
    
    best_score_kmenoids= 100000
    best_k_kmenoids = 1
    best_score_time_kmenoids= 0
    best_result = None
    
    for k in k_list:
        # kmenoids
        tp1 = time.time()
        result = kmenoids(data,k)
        tp2 = time.time()
        score = metrics.davies_bouldin_score(data,result.labels)
        if (score<best_score_kmenoids):
            best_score_kmenoids = score
            best_k_kmenoids = k
            best_score_time_kmenoids = tp2-tp1
            best_result = result
    text = ("KMENOIDS [bouldin]\n best_k="+str(best_k_kmenoids)+" (score = "+str(round(best_score_kmenoids,3))+") ["+str(round(best_score_time_kmenoids*1000,3))+" ms]")
    print(text)
    save_plot(f0, f1, c=best_result.labels, title=text) 
    
def bouldin_score_agglo(data, f0, f1, seuil_list, linkage):
    best_score = 100000000
    best_k = 0
    best_seuil = 0
    best_score_time = 0
    best_result = None
    
    for seuil in seuil_list:
        tp1 = time.time()
        result = agglomerative(data,seuil,linkage)
        tp2 = time.time()
        k = result.n_clusters_
        if(k != None and k>1 and k<1000):
            score = metrics.davies_bouldin_score(data,result.labels_)
            if (score<best_score):
                best_k = k
                best_score = score
                best_seuil = seuil
                best_result = result
                best_score_time = tp2-tp1
    text = ("AGGLOMERATIVE("+linkage+")[bouldin]\n best_k="+str(best_k)+" best_seuil="+str(best_seuil)+" (score = "+str(round(best_score,3))+") ["+str(round(best_score_time*1000,3))+" ms]")
    print(text)
    save_plot(f0, f1, c=best_result.labels_, title=text)    

def bouldin_score_dbs(data, f0, f1, eps_list,min_samples_list):
    best_score_dbs= 100
    best_eps_dbs = 0
    best_mins_dbs = 0
    best_score_time_dbs = 0
    best_k_dbs = 0
    best_result = None
    
    #DBSCAN
    for eps in eps_list:
        for min_samples in min_samples_list:
            tp1 = time.time()
            result = dbscan(data,eps,min_samples)
            #print(result)
            tp2 = time.time()
            k = len(set(result.labels_))
            if (k>1 and k<1000):
                score = metrics.davies_bouldin_score(data,result.labels_)
                #print("ok")
                if (score<best_score_dbs):
                    best_score_dbs = score
                    best_eps_dbs = eps
                    best_mins_dbs = min_samples
                    best_score_time_dbs = tp2-tp1
                    best_k_dbs = k 
                    best_result = result
            
    text = ("DBSCAN [bouldin]\nbest_K="+str(best_k_dbs)+" (best_eps="+str(best_eps_dbs)+"  best_mins="+str(best_mins_dbs)+" score = "+str(round(best_score_dbs,3))+") ["+str(round(best_score_time_dbs*1000,3))+" ms]")
    print(text)
    save_plot(f0, f1, c=best_result.labels_, title=text)   

def bouldin_score_hdbs(data, f0, f1, minc_list):
    best_score_hdbs= 100
    best_minc_hdbs = 0
    best_score_time_hdbs = 0
    best_k_hdbs = 0
    best_result = None
    
    #hdbsCAN
    
    for min_cluster_size in minc_list:
        tp1 = time.time()
        result = hdbscanf(data,min_cluster_size)
        #print(result)
        tp2 = time.time()
        k = len(set(result.labels_))
        if (k>1 and k<1000):
            score = metrics.davies_bouldin_score(data,result.labels_)
            #print("ok")
            if (score<best_score_hdbs):
                best_score_hdbs = score
                best_minc_hdbs = min_cluster_size
                best_score_time_hdbs = tp2-tp1
                best_k_hdbs = k 
                best_result = result
            
    text = ("HDBSCAN [bouldin]\nbest_K="+str(best_k_hdbs)+"  (best_minc="+str(best_minc_hdbs)+" score = "+str(round(best_score_hdbs,3))+") ["+str(round(best_score_time_hdbs*1000,3))+" ms]")
    print(text)
    save_plot(f0, f1, c=best_result.labels_, title=text)   

def silhouette_score_kmeans(data, f0, f1, k_list):
    best_score_kmeans= 0
    best_k_kmeans = 1
    best_score_time_kmeans = 0
    best_result = None
    
    for k in k_list:
        
        # kmeans
        tp1 = time.time()
        result = kmeans(data,k)
        tp2 = time.time()
        score = metrics.silhouette_score(data,result.labels_)
        if (score>best_score_kmeans):
            best_score_kmeans = score
            best_k_kmeans = k
            best_score_time_kmeans = tp2-tp1
            best_result = result

    text = ("KMEANS [silhouette]\n best_k="+str(best_k_kmeans)+" (score = "+str(round(best_score_kmeans,3))+") ["+str(round(best_score_time_kmeans*1000,3))+" ms]")
    print(text)
    save_plot(f0, f1, c=best_result.labels_, title=text) 

def silhouette_score_kmenoids(data, f0, f1, k_list):
    
    best_score_kmenoids= 0
    best_k_kmenoids = 1
    best_score_time_kmenoids= 0
    best_result = None
    for k in k_list:
        # kmenoids
        tp1 = time.time()
        result = kmenoids(data,k)
        tp2 = time.time()
        score = metrics.silhouette_score(data,result.labels)
        if (score>best_score_kmenoids):
            best_score_kmenoids = score
            best_k_kmenoids = k
            best_score_time_kmenoids = tp2-tp1
            best_result = result
    text = ("KMENOIDS [silhouette]\n best_k="+str(best_k_kmenoids)+" (score = "+str(round(best_score_kmenoids,3))+") ["+str(round(best_score_time_kmenoids*1000,3))+" ms]")
    print(text)
    save_plot(f0, f1, c=best_result.labels, title=text) 
    
def silhouette_score_agglo(data, f0, f1, seuil_list, linkage):
    best_score = 0
    best_k = 0
    best_seuil = 0
    best_score_time = 0
    best_result = None
    
    for seuil in seuil_list:
        tp1 = time.time()
        result = agglomerative(data,seuil,linkage)
        tp2 = time.time()
        k = result.n_clusters_
        if(k != None and k>1 ):
            score = metrics.silhouette_score(data,result.labels_)
            if (score>best_score):
                best_k = k
                best_score = score
                best_seuil = seuil
                best_result = result
                best_score_time = tp2-tp1
    text = ("AGGLOMERATIVE("+linkage+")[silhouette]\n best_k="+str(best_k)+" best_seuil="+str(best_seuil)+" (score = "+str(round(best_score,3))+") ["+str(round(best_score_time*1000,3))+" ms]")
    print(text)
    save_plot(f0, f1, c=best_result.labels_, title=text)   

def silhouette_score_dbs(data, f0, f1, eps_list,min_samples_list):
    best_score_dbs= 0
    best_eps_dbs = 0
    best_mins_dbs = 0
    best_score_time_dbs = 0
    best_k_dbs = 0
    best_result = None
    
    #DBSCAN
    for eps in eps_list:
        for min_samples in min_samples_list:
            tp1 = time.time()
            result = dbscan(data,eps,min_samples)
            #print(result)
            tp2 = time.time()
            k = len(set(result.labels_))
            if (k>1 and k<1000):
                score = metrics.silhouette_score(data,result.labels_)
                #print("ok")
                if (score>best_score_dbs):
                    best_score_dbs = score
                    best_eps_dbs = eps
                    best_mins_dbs = min_samples
                    best_score_time_dbs = tp2-tp1
                    best_k_dbs = k 
                    best_result = result
            
    text = ("DBSCAN [silhouette]\nbest_K="+str(best_k_dbs)+" (best_eps="+str(best_eps_dbs)+"  best_mins="+str(best_mins_dbs)+" score = "+str(round(best_score_dbs,3))+") ["+str(round(best_score_time_dbs*1000,3))+" ms]")
    print(text)
    save_plot(f0, f1, c=best_result.labels_, title=text)   

def silhouette_score_hdbs(data, f0, f1, minc_list):
    best_score_hdbs= 0
    best_minc_hdbs = 0
    best_score_time_hdbs = 0
    best_k_hdbs = 0
    best_result = None
    
    #hdbsCAN
    
    for min_cluster_size in minc_list:
        tp1 = time.time()
        result = hdbscanf(data,min_cluster_size)
        #print(result)
        tp2 = time.time()
        k = len(set(result.labels_))
        if (k>1 and k<1000):
            score = metrics.silhouette_score(data,result.labels_)
            #print("ok")
            if (score>best_score_hdbs):
                best_score_hdbs = score
                best_minc_hdbs = min_cluster_size
                best_score_time_hdbs = tp2-tp1
                best_k_hdbs = k 
                best_result = result
            
    text = ("HDBSCAN [silhouette]\nbest_K="+str(best_k_hdbs)+"  (best_minc="+str(best_minc_hdbs)+" score = "+str(round(best_score_hdbs,3))+") ["+str(round(best_score_time_hdbs*1000,3))+" ms]")
    print(text)
    save_plot(f0, f1, c=best_result.labels_, title=text)   

def process_file(file):
    databrut = pd.read_csv(path+file, sep=" ", encoding="ISO-8859-1", skipinitialspace=True).to_numpy()
    print(databrut)
    data = [[x[0]/1000000, x[1]/1000000] for x in databrut]
    print(data)
    f0 = [f[0] for f in data]
    f1 = [f[1] for f in data]
    
    show_plot(f0, f1, None, "test")
    
    #show_plot(f0, f1, None, "Plot initial : "+file)
    #bouldin_score_hdbs(data, f0, f1, range(1,100))
    
    for linkage in ["single","complete","ward","average"]:
        silhouette_score_agglo(data, f0, f1,seuil_list , linkage)
        bouldin_score_agglo(data, f0, f1, seuil_list, linkage)
    '''
    bouldin_score_kmeans(data, f0, f1, k_list)
    silhouette_score_kmeans(data, f0, f1, k_list)
    bouldin_score_kmenoids(data, f0, f1, k_list)
    silhouette_score_kmenoids(data, f0, f1, k_list)
    bouldin_score_hdbs(data, f0, f1, minc_list)
    silhouette_score_hdbs(data, f0, f1, minc_list)
    bouldin_score_dbs(data, f0, f1, eps_list, min_samples_list)
    silhouette_score_dbs(data, f0, f1, eps_list, min_samples_list)      
    '''
    
path = './dataset-rapport/'  
out = './zz1/'
filelist =  ["zz1.txt"]
thre_list =  []
k_list = range(6,20)
seuil_list = [i/100 for i in range(2,800,10)]
minc_list = range(2,100)
eps_list = [i/100 for i in range(2,100,10)]
min_samples_list = range(1,100,10)
def main():
    for file in filelist:
        process_file(file)
        
main()
