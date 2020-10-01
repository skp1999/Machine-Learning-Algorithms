## Importing necessary libraries


import numpy as np
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
from numpy.linalg import norm
import warnings
warnings.filterwarnings("ignore")

## Importing dataset and adding a new column of cluster number for each data pt

data=pd.read_csv('../data/AllBooks_baseline_DTM_Labelled.csv')
data=data.drop([13], axis=0)
data=data.drop(['Unnamed: 0'], axis=1)   #Dropping the class (Religious text)
arr=np.array(data.columns)     #arr is the array containing columns name
data=np.array(data)            #dataframe converted to array

#creating new dataframe with a new column as cluster number
data_with_clusters=pd.read_csv('../data/AllBooks_baseline_DTM_Labelled.csv')
data_with_clusters=data_with_clusters.drop([13], axis=0).reset_index()
new_column=np.zeros(len(data))
data_with_clusters['Cluster_number']=new_column
data_with_clusters=data_with_clusters.drop(['Unnamed: 0'], axis=1)

## k-means clustering algorithm from basic


number_of_clusters=8              #no of clusters =8 (specified in the problem)
num_iter=50                       #number of iterations

centroid=np.random.randn(number_of_clusters,data.shape[1])            #random initialization of centroids

for iter in range(num_iter):           #iterations loop
    
    clusters=[[],[],[],[],[],[],[],[]]          #stores the data points of the 8 clusters formed
    
    for i in range(0, len(data)):
        
        min_cosine_similarity=999999
        
        for k in range(0,number_of_clusters):         #loop assigns cluster number for each data point
            cosine_similarity=np.exp(-1*(np.dot(centroid[k], data[i])/(norm(centroid[k])*norm(data[i]))))
            if(cosine_similarity < min_cosine_similarity):
                min_cosine_similarity = cosine_similarity;
                cluster_number=k;
        
        data_with_clusters.at[i,'Cluster_number']=cluster_number
        clusters[cluster_number].append(i)
         
    for k in range(number_of_clusters):         #loop updates the centroid values after each iteration
        if(len(clusters[k]) != 0):
            subset = data_with_clusters[data_with_clusters["Cluster_number"] == k]
            sum_array=np.array(subset.sum())
            
            for m in range(1,len(sum_array)-1):
                centroid[k][m-1]=sum_array[m]/(len(subset))
                
kmeans_clusters=sorted(clusters)             #sorts the clusters as per the first element of each cluster (output desired)
    
for i in range(len(kmeans_clusters)):        #sorts the data points in each cluster
    kmeans_clusters[i]=sorted(kmeans_clusters[i])

## Output to the text file "kmeans.txt"


file=open('../clusters/kmeans.txt','w')
for i in kmeans_clusters:
    for j in range(len(i)):
        file.write(str(i[j]))
        if(j!=len(i)-1):
            file.write(',')
    file.write('\n')
file.close()