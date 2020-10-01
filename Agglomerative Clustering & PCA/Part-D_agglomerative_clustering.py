## Importing Python Libraries


import numpy as np
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
from numpy.linalg import norm
import warnings
warnings.filterwarnings("ignore")

## Importing dataset

data=pd.read_csv("../data/AllBooks_baseline_DTM_Labelled.csv")
data=data.drop(['Unnamed: 0'], axis=1)
data=data.drop([13], axis=0)


# ### Applying PCA through scikit-learn to reduced the data to 100 components

from sklearn.decomposition import PCA
pca = PCA(n_components=100)
principalComponents = pca.fit_transform(data)
reduced_data = pd.DataFrame(data = principalComponents)
df_reduced=reduced_data
reduced_data=np.array(reduced_data)            #dataframe converted to array


## Agglomerative clustering on reduced dataset using Part-B

## Cosine similarity function to estimate extent of similarity between 2 data points

def cosine_similarity(data1,data2):
    similarity=np.dot(data1, data2)/(norm(data1)*norm(data2))     
    return(np.exp(-1*similarity))

## Single linkage function to calculate minimum similarity between 2 data points

def single_linkage(clusters,similarity_matrix,iter1,iter2):
    
    min_distance=99999
    for i in clusters[iter1]:
        for j in clusters[iter2]: 
            if(similarity_matrix[i][j] < min_distance):   #finds minimum distance between two clusters using each data pt
                min_distance = similarity_matrix[i][j]
            else:
                min_distance=similarity_matrix[j][i]
    return(min_distance)

## Function to join two most similar clusters

def join_clusters(clusters,similarity_matrix,number_of_clusters):
    
    if(len(clusters) <= number_of_clusters):                 #base case
        return
    
    min_distance=999999
    join_cluster=[0,1]
    
    for i in range(len(clusters)):
        for j in range(i+1,len(clusters)):
            min_single_linkage=single_linkage(clusters,similarity_matrix,i,j)         #calculate single linkage distances
            if(min_single_linkage < min_distance):
                min_distance = min_single_linkage
                join_cluster[0]=i
                join_cluster[1]=j
    clusters[join_cluster[0]]=clusters[join_cluster[0]]+clusters[join_cluster[1]]    #combines two most similar clusters

    del(clusters[join_cluster[1]])
    
    join_clusters(clusters,similarity_matrix,number_of_clusters)    #recursion to join clusters till base case is reached
    return

## Function to find the distance matrix between two data points

def create_matrix(reduced_data):
    
    similarity_matrix=[[0]*len(reduced_data)]*len(reduced_data)     #matrix of zeros of shape (len(data) * len(data))
    
    # similarity matrix to store similarity distance (e^(-z)) between two data points
    for i in range(len(reduced_data)):
        for j in range(i+1):
            similarity_matrix[i][j]=similarity_matrix[j][i]=cosine_similarity(reduced_data[i],reduced_data[j])
            
    return(similarity_matrix)

## Utility function to implement Agglomerative Clustering

def agglomerative_Clustering(reduced_data,number_of_clusters):
    
    similarity_matrix=create_matrix(reduced_data)
                    
    clusters=[]            #stores the clusters formed
    
    for i in range(len(reduced_data)):
        clusters.append([i])
        
    join_clusters(clusters,similarity_matrix,number_of_clusters)   # function to merge two most similar clusters
    
    agglomerative_clusters=sorted(clusters)
    
    for i in range(len(agglomerative_clusters)):
        agglomerative_clusters[i]=sorted(agglomerative_clusters[i])
    
    return(agglomerative_clusters)

agglomerative_clusters=agglomerative_Clustering(reduced_data,8)       #function calling with required parameters

## Text file output of agglomerative clustering on reduced dataset

file=open('../clusters/agglomerative_reduced.txt','w')
for i in agglomerative_clusters:
    for j in range(len(i)):
        file.write(str(i[j]))
        if(j!=len(i)-1):
            file.write(',')
    file.write('\n')
file.close()



