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

### Applying PCA through scikit-learn to reduced the data to 100 components

from sklearn.decomposition import PCA
pca = PCA(n_components=100)
principalComponents = pca.fit_transform(data)
reduced_data = pd.DataFrame(data = principalComponents)
df_reduced=reduced_data
reduced_data=np.array(reduced_data)            #dataframe converted to array

## k-means clustering on reduced dataset (using Part-C)

#creating new dataframe with a new column as cluster number
data_with_clusters=df_reduced
new_column=np.zeros(len(reduced_data))

arr=np.array(data.columns)     #arr is the array containing columns name

number_of_clusters=8
num_iter=100

centroid=np.random.randn(number_of_clusters,reduced_data.shape[1])

for iter in range(num_iter):
    
    clusters=[[],[],[],[],[],[],[],[]]
    
    for i in range(0, len(reduced_data)):
        
        max_cosine_similarity=-1
        
        for k in range(0,number_of_clusters):
            cosine_similarity=np.exp(-1*(np.dot(centroid[k], reduced_data[i])/(norm(centroid[k])*norm(reduced_data[i]))))
            if(cosine_similarity > max_cosine_similarity):
                max_cosine_similarity = cosine_similarity;
                cluster_number=k;
        
        data_with_clusters.at[i,'Cluster_number']=cluster_number
        clusters[cluster_number].append(i)
         
    for k in range(number_of_clusters):
        if(len(clusters[k]) != 0):
            subset = data_with_clusters[data_with_clusters["Cluster_number"] == k]
            sum_array=np.array(subset.sum())
            
            for m in range(len(sum_array)-1):
                centroid[k][m]=sum_array[m]/len(clusters[k])

kmeans_clusters=sorted(clusters)
    
for i in range(len(kmeans_clusters)):
    kmeans_clusters[i]=sorted(kmeans_clusters[i])

### Text file output of k-means clustering on reduced dataset


file=open('../clusters/kmeans_reduced.txt','w')
for i in kmeans_clusters:
    for j in range(len(i)):
        file.write(str(i[j]))
        if(j!=len(i)-1):
            file.write(',')
    file.write('\n')
file.close()
