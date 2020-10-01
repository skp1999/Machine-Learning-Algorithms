
## Import necessary libraries and modules

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import os
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

## Importing the Dataset and performing pre-processing to use it for training neural networks


## Read the data from 'data' folder
data=pd.read_csv('../data/seeds_dataset.txt',sep='\t',header=None)

def data_preprocessing(data):
    
    for i in range(0,data.shape[1]-1):
        for j in range(len(data)):
            data[i][j]=(data[i][j]-data[i].mean())/data[i].std()       ## Perform Z-score normalization on the data
    
    x = data.iloc[:,0:7]
    y = data.iloc[:,7]

    ## Split the data into training and testing dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)   
    
    ## Converting y_train dataframe to numpy array for further use
    y_target_train=[]
    for i in range(len(y_train)):
        y_target_train.append(str(y_train.iloc[i]))
    
    ## Converting y_test dataframe to numpy array for further use    
    y_target_test=[]
    for i in range(len(y_test)):
        y_target_test.append(str(y_train.iloc[i]))
    
    return (x_train, x_test, y_target_train,y_target_test)


## Get the data to be used for training the neural network
x_train, x_test, y_target_train, y_target_test = data_preprocessing(data)



## Neural network implementation using Scikit Learn MLP Classifier


# Part 1(A)

mlb = MultiLabelBinarizer()    ## create an object of MultiLabelBinarizer() class to one-hot encode the target output train and test data
y_train=mlb.fit_transform(y_target_train)
y_test=mlb.fit_transform(y_target_test)

## Using MLP Classifier to train the neural network with given specifications in Part-1(A)
mlp = MLPClassifier(hidden_layer_sizes=(32), alpha = 0.01, activation='logistic', 
                    solver='sgd', max_iter=200, batch_size=32)
mlp.fit(x_train,y_train)


## Predict using the trained neural network
predict_train = mlp.predict(x_train)
predict_test = mlp.predict(x_test)

## Printing the final train and test accuracy
print("Part-2 Specification 1A - ")
print("Final Train accuracy : ", accuracy_score(y_train, predict_train))
print("Final Test accuracy : ", accuracy_score(y_test,predict_test))



# Part 1(B)

mlb = MultiLabelBinarizer()   ## create an object of MultiLabelBinarizer() class to one-hot encode the target output train and test data
y_train=mlb.fit_transform(y_target_train)
y_test=mlb.fit_transform(y_target_test)

## Using MLP Classifier to train the neural network with given specifications in Part-1(B)
mlp = MLPClassifier(hidden_layer_sizes=(64,32), alpha = 0.01, activation='relu', 
                    solver='sgd', max_iter=200, batch_size=32)
mlp.fit(x_train,y_train)

## Predict using the trained neural network
predict_train = mlp.predict(x_train)
predict_test = mlp.predict(x_test)

## Printing the final train and test accuracy
print("Part-2 Specification 1B - ")
print("Final Train accuracy : ", accuracy_score(y_train, predict_train))
print("Final Test accuracy : ", accuracy_score(y_test,predict_test))