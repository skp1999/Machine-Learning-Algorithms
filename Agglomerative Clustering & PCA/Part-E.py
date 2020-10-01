## Importing necessary libraries

import numpy as np
import pandas as pd
import os
from numpy import linalg as np_la
import warnings
warnings.filterwarnings("ignore")

## Importing dataset and preprocessing it

data=pd.read_csv('../data/AllBooks_baseline_DTM_Labelled.csv')            #Import the "Religious text" dataset
data=data.drop([13], axis=0)                   # Removing the 13th row
data_labels=np.array(data['Unnamed: 0'])   

#loop to remove characters after '_' in the class label
i=0;
for item in data['Unnamed: 0']:
    x=item.find("_")
    data_labels[i]=item[0:x]
    i=i+1

class_labels=['Buddhism','TaoTeChing','Upanishad','YogaSutra','BookOfProverb', 'BookOfEcclesiastes', 'BookOfEccleasiasticus',
       'BookOfWisdom']
for i in range(len(data_labels)):
    for j in range(len(class_labels)):
        if(data_labels[i]==class_labels[j]):
            data_labels[i]=j
        

#replace the class labels with the new one (data_labels)
data=data.drop(['Unnamed: 0'], axis=1)
data.insert(0, 'Unnamed: 0', data_labels)


# ## Assigning class label to each doucment 

classes=[[],[],[],[],[],[],[],[]]     # stores document number in each class
document_labels=np.array(data['Unnamed: 0'])


for i in range(len(document_labels)):
        classes[document_labels[i]].append(i)


## Importing text file and extracting information of clusters from "agglomerative.txt"

file = open("../clusters/agglomerative.txt","r")
clusters=[[],[],[],[],[],[],[],[]]
index=0

for line in file:
  
    #Let's split the line into an array called "fields" using the "," as a separator:
    fields = line.split(",")
    
    for i in range(len(fields)):
        clusters[index].append(fields[i])
        
    index=index+1
    
file.close()


for i in range(len(clusters)):
    if(len(clusters[i])!=0):
        item=clusters[i][-1]
        clusters[i][-1]=item[0:len(item)-1]
    
    for j in range(len(clusters[i])):
        clusters[i][j]=int(clusters[i][j])


## H(Y) - Entropy of class labels

HY=0
for i in range(len(classes)):
    if(len(classes[i])!=0):
        p=len(classes[i])/len(data)
        HY=HY+(p*np.log2(p))

HY=-HY


## H(C) - Entropy of cluster labels

HC=0
for i in range(len(clusters)):
    if(len(clusters[i])!=0):
        p=len(clusters[i])/len(data)
        HC=HC+(p*np.log2(p))

HC=-HC

# Calculation of H(Y|C)

HYC=0

for i in range(len(clusters)):
    count=0
    for j in range(len(clusters[i])):
        if(clusters[i][j] in classes[i]):
            count=count+1
    if(count!=0):
        p=count/len(clusters[i])
        HYC=HYC+p*np.log2(p)

    HYC=HYC/len(clusters)

HYC=-HYC

IYC=HY-HYC


## NMI(Normalized Mutual Information) = (2*I(Y;C))/(H(Y)+H(C))

NMI=(2*IYC)/(HY+HC)

print("Normalized Mutual Information (NMI) for agglomerative.txt : ",NMI)

## Importing text file and extracting information of clusters from "kmeans.txt"

file = open("../clusters/kmeans.txt","r")
clusters=[[],[],[],[],[],[],[],[]]
index=0

for line in file:
  
    #Let's split the line into an array called "fields" using the "," as a separator:
    fields = line.split(",")
    
    for i in range(len(fields)):
        clusters[index].append(fields[i])
        
    index=index+1
    
file.close()


for i in range(len(clusters)):
    if(len(clusters[i])!=0):
        item=clusters[i][-1]
        clusters[i][-1]=item[0:len(item)-1]
    
    for j in range(len(clusters[i])):
        clusters[i][j]=int(clusters[i][j])


## H(Y) - Entropy of class labels

HY=0
for i in range(len(classes)):
    if(len(classes[i])!=0):
        p=len(classes[i])/len(data)
        HY=HY+(p*np.log2(p))

HY=-HY


## H(C) - Entropy of cluster labels

HC=0
for i in range(len(clusters)):
    if(len(clusters[i])!=0):
        p=len(clusters[i])/len(data)
        HC=HC+(p*np.log2(p))

HC=-HC

# Calculation of H(Y|C)

HYC=0

for i in range(len(clusters)):
    count=0
    for j in range(len(clusters[i])):
        if(clusters[i][j] in classes[i]):
            count=count+1
    if(count!=0):
        p=count/len(clusters[i])
        HYC=HYC+p*np.log2(p)

    HYC=HYC/len(clusters)

HYC=-HYC

IYC=HY-HYC


## NMI(Normalized Mutual Information) = (2*I(Y;C))/(H(Y)+H(C))

NMI=(2*IYC)/(HY+HC)

print("Normalized Mutual Information (NMI) for kmeans.txt : ",NMI)




## NMI for reduced datasets

data=data.drop(['Unnamed: 0'],axis=1)

from sklearn.decomposition import PCA
pca = PCA(n_components=100)
principalComponents = pca.fit_transform(data)
reduced_data = pd.DataFrame(data = principalComponents)

## NMI score calculation for 'agglomerative_reduced.txt'

file = open("../clusters/agglomerative_reduced.txt","r")
clusters=[[],[],[],[],[],[],[],[]]
index=0

#Repeat for each song in the text file

for line in file:
  
    #Let's split the line into an array called "fields" using the ";" as a separator:
    fields = line.split(",")
    
    for i in range(len(fields)):
        clusters[index].append(fields[i])
        
    index=index+1
    
file.close()


for i in range(len(clusters)):
    if(len(clusters[i])!=0):
        item=clusters[i][-1]
        clusters[i][-1]=item[0:len(item)-1]
    
    for j in range(len(clusters[i])):
        clusters[i][j]=int(clusters[i][j])

## H(Y) - Entropy of class labels

HY=0
for i in range(len(classes)):
    p=len(classes[i])/len(reduced_data)
    HY=HY+(p*np.log2(p))

HY=-HY

## H(C) - Entropy of cluster labels

HC=0
for i in range(len(clusters)):
    if(len(clusters[i])!=0):
        p=len(clusters[i])/len(reduced_data)
        HC=HC+(p*np.log2(p))

HC=-HC

## I(Y;C) - Mutual Information between Y and C { I(Y;C)=H(Y)-H(Y|C)}

# Calculation of H(Y|C)

HYC=0

for i in range(len(clusters)):
    count=0
    for j in range(len(clusters[i])):
        if(clusters[i][j] in classes[i]):
            count=count+1
    if(count!=0):
        p=count/len(clusters[i])
        HYC=HYC+p*np.log2(p)

    HYC=HYC/len(clusters)

HYC=-HYC

IYC=HY-HYC

## NMI(Normalized Mutual Information) = (2*I(Y;C))/(H(Y)+H(C))

NMI=(2*IYC)/(HY+HC)

print('Normalized Mutual Information (NMI) for agglomerative_reduced.txt : ',NMI)


## NMI score calculation for 'agglomerative_reduced.txt'

file = open("../clusters/kmeans_reduced.txt","r")
clusters=[[],[],[],[],[],[],[],[]]
index=0

#Repeat for each song in the text file

for line in file:
  
    #Let's split the line into an array called "fields" using the ";" as a separator:
    fields = line.split(",")
    
    for i in range(len(fields)):
        clusters[index].append(fields[i])
        
    index=index+1
    
file.close()


for i in range(len(clusters)):
    if(len(clusters[i])!=0):
        item=clusters[i][-1]
        clusters[i][-1]=item[0:len(item)-1]
    
    for j in range(len(clusters[i])):
        clusters[i][j]=int(clusters[i][j])

## H(Y) - Entropy of class labels

HY=0
for i in range(len(classes)):
    p=len(classes[i])/len(reduced_data)
    HY=HY+(p*np.log2(p))

HY=-HY

## H(C) - Entropy of cluster labels

HC=0
for i in range(len(clusters)):
    if(len(clusters[i])!=0):
        p=len(clusters[i])/len(reduced_data)
        HC=HC+(p*np.log2(p))

HC=-HC

## I(Y;C) - Mutual Information between Y and C { I(Y;C)=H(Y)-H(Y|C)}

# Calculation of H(Y|C)

HYC=0

for i in range(len(clusters)):
    count=0
    for j in range(len(clusters[i])):
        if(clusters[i][j] in classes[i]):
            count=count+1
    if(count!=0):
        p=count/len(clusters[i])
        HYC=HYC+p*np.log2(p)

    HYC=HYC/len(clusters)

HYC=-HYC

IYC=HY-HYC

## NMI(Normalized Mutual Information) = (2*I(Y;C))/(H(Y)+H(C))

NMI=(2*IYC)/(HY+HC)

print('Normalized Mutual Information (NMI) for kmeans_reduced.txt : ',NMI)

# #       NMI for "kmeans.txt"                     =   0.928316112327191
# #       NMI for "kmeans_reduced.txt"             =   1.9043026895357538
# #       NMI for "agglomerative.txt"              =   1.4235014030385384
# #       NMI for "agglomerative_reduced.txt"      =   1.8938037432836978
