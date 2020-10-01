
## Import necessary libraries and modules

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

## Importing data and performing pre-processing to get the training and the testing set

data=pd.read_csv('../data/seeds_dataset.txt',sep='\t',header=None)


def data_preprocessing(data):
    for i in range(0,data.shape[1]-1):
        for j in range(len(data)):
            data[i][j]=(data[i][j]-data[i].mean())/data[i].std()
    
    output=pd.DataFrame()
    for i in range(len(data)):
        if(data[data.shape[1]-1][i]==1):
            a_row = pd.Series([1, 0, 0])
            row_df = pd.DataFrame([a_row])
            output = pd.concat([row_df, output], ignore_index=True)
        elif(data[data.shape[1]-1][i]==2):
            a_row = pd.Series([0, 1, 0])
            row_df = pd.DataFrame([a_row])
            output = pd.concat([row_df, output], ignore_index=True)
        else:
            a_row = pd.Series([0, 0, 1])
            row_df = pd.DataFrame([a_row])
            output = pd.concat([row_df, output], ignore_index=True)
        
    data.drop(data.columns[[7]], axis = 1, inplace = True) 
    
    x = data.iloc[:,0:7]
    y = output
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
    
    
    return (x_train, y_train, x_test, y_test)

## get the train and test data to be used later
train_x, train_y, test_x, test_y = data_preprocessing(data)


## Loading the dataset and creating mini-batches of size 32

def create_mini_batches(x_train, y_train, mini_batch_size):
    mini_batches = [] 
    for i in range(0, x_train.shape[0]//32):
        x_train_mini = x_train[i:i+mini_batch_size]
        y_train_mini = y_train[i:i+mini_batch_size]
        mini_batches.append((x_train_mini,y_train_mini))
        
    return(mini_batches)



## Class Neural Network to train the neural network as per the specifications

class Neural_Network:
    
    ## Constructor to initilize all the values to their default
    def __init__(self):

        self.layers_size = []
        self.L=0
        self.parameters = {}
        self.n = 0
        self.acc_train=[]
        self.acc_test=[]
        self.iterations = 0
        self.activation = ''
    
    ## Function to add layers and the number of nodes in each layer   
    def add_layers(self, layers_size):

        self.layers_size = layers_size
        self.L = len(self.layers_size)
        return
    
    ## Function to calculate sigmoid
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    ## Function to calculate derivative of sigmoid
    def sigmoid_derivative(self, Z):
        s = 1 / (1 + np.exp(-Z))
        return s * (1 - s)
    
    ## Function to apply softmax activation
    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)

    ## Function to apply relu activation
    def relu(self, Z):
        R = np.maximum(0, Z)
        return R

    ## Function to apply derivative of relu
    def relu_derivative(self, Z):
        Z[Z >= 0] = 1
        Z[Z < 0]  = 0
        return Z
    
    ## Function to initialize weights and bias
    def initialize_parameters(self):
 
        for l in range(1, len(self.layers_size)):
            self.parameters["W" + str(l)] = np.random.randn(self.layers_size[l], self.layers_size[l - 1]) / np.sqrt(
                self.layers_size[l - 1])
            self.parameters["b" + str(l)] = np.zeros((self.layers_size[l], 1))
 
    ## Function to move forward in the Neural Network cosidering activations and bias
    def forward_propagation(self, X):

        store = {}
        A = X.T
        for i in range(self.L - 1):
            Z = self.parameters["W" + str(i+1)].dot(A) + self.parameters["b" + str(i+1)]
            if(self.activation=='relu'):
                A = self.relu(Z)
            else:
                A = self.sigmoid(Z) 
            
            store["A" + str(i+1)] = A
            store["W" + str(i+1)] = self.parameters["W" + str(i+1)]
            store["Z" + str(i+1)] = Z
 
        Z = self.parameters["W" + str(self.L)].dot(A) + self.parameters["b" + str(self.L)]
        A = self.softmax(Z)
        store["A" + str(self.L)] = A
        store["W" + str(self.L)] = self.parameters["W" + str(self.L)]
        store["Z" + str(self.L)] = Z
        
        return A, store
 
    ## Function to back-propagate in the neural network to update the parameters
    def backward_propagation(self, X, Y, store):
 
        derivatives = {}
 
        store["A0"] = X.T
 
        A = store["A" + str(self.L)]
        dZ = A - Y.T
 
        dW = dZ.dot(store["A" + str(self.L - 1)].T) / self.n
        db = np.sum(dZ, axis=1, keepdims=True) / self.n
        dAPrev = store["W" + str(self.L)].T.dot(dZ)
 
        derivatives["dW" + str(self.L)] = dW
        derivatives["db" + str(self.L)] = db
 
        for i in range(self.L-1, 0, -1):
            if(self.activation=='relu'):
                dZ = dAPrev * self.relu_derivative(store["Z" + str(i)])
            else:
                dZ = dAPrev * self.sigmoid_derivative(store["Z" + str(i)])
            dW = 1. / self.n * dZ.dot(store["A" + str(i - 1)].T)
            db = 1. / self.n * np.sum(dZ, axis=1, keepdims=True)
            if i > 1:
                dAPrev = store["W" + str(i)].T.dot(dZ)
 
            derivatives["dW" + str(i)] = dW
            derivatives["db" + str(i)] = db
 
        return derivatives
 
    ## Perform mini batch Stochastic Gradient to train the neural network
    def mini_batch_SGD(self, X, Y, X_test, Y_test, activation, learning_rate=0.01, num_iter=200):
 
        self.n = X.shape[0]
        self.layers_size.insert(0, X.shape[1])
        self.iterations = num_iter
        self.initialize_parameters()
        self.activation=activation
        
        for loop in range(self.iterations):
            ## create mini-batch of size 32 (last 8 rows are ignored)
            mini_batches = create_mini_batches(X,Y,32)
            for mini_batch in mini_batches:
                x_mini, y_mini = mini_batch
                A, store = self.forward_propagation(x_mini)
                cost = -np.mean(y_mini * np.log(A.T+ 1e-8))
                derivatives = self.backward_propagation(x_mini, y_mini, store)

                ## update the parameters
                for l in range(1, self.L + 1):
                    self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - learning_rate * derivatives[
                        "dW" + str(l)]
                    self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - learning_rate * derivatives[
                        "db" + str(l)]
            
            ## Store the train and test scores after every 10 epochs
            if(loop % 10)==0:
                self.acc_train.append(self.predict(X,Y))
                self.acc_test.append(self.predict(X_test, Y_test))
            
    ## Function to predict the trained neural network
    def predict(self, X, Y):
        A, cache = self.forward_propagation(X)
        y_hat = np.argmax(A, axis=0)
        Y = np.argmax(Y, axis=1)
        accuracy = (y_hat == Y).mean()
        return accuracy * 100
    
    ## Function to plot the train and test accuracy   
    def plot_accuracy(self):
        plt.figure()
        fig=plt.subplot()
        fig.plot(np.arange(0,200,10), self.acc_train, label='Train accuracy')
        fig.plot(np.arange(0,200,10), self.acc_test, label = 'Test accuracy')
        plt.xlabel("epochs")
        plt.ylabel("Accuracy")
        fig.legend()
        plt.show()
        
## End of class


## Creating an object and creating a neural network with specifications in Part-1(A)
layers_dims = [7,32,3]
nn=Neural_Network()      # object of class Neural_Network
nn.add_layers(layers_dims)
## Create mini batches of size 32
mini_batches = create_mini_batches(train_x,train_y,32)
## Apply mini batch SGD to train the neural network
nn.mini_batch_SGD(np.array(train_x),np.array(train_y), np.array(test_x), np.array(test_y), activation = 'sigmoid',
        learning_rate=0.01, num_iter=200)
nn.plot_accuracy()
print("Part 1A - ")
print("Final Train Accuracy:", nn.predict(np.array(train_x), np.array(train_y)))
print("Final Test Accuracy:", nn.predict(np.array(test_x), np.array(test_y)))


## Creating an object and creating a neural network with specifications in Part-1(B)
layers_dims = [7,64,32,3]
nn=Neural_Network()              # object of class Neural_Network
nn.add_layers(layers_dims)
## Create mini batches of size 32
mini_batches = create_mini_batches(train_x,train_y,32)
## Apply mini batch SGD to train the neural network
nn.mini_batch_SGD(np.array(train_x),np.array(train_y), np.array(test_x), np.array(test_y), activation = 'relu',
        learning_rate=0.01, num_iter=200)
nn.plot_accuracy()
print("Part 1B - ")
print("Final Train Accuracy:", nn.predict(np.array(train_x), np.array(train_y)))
print("Final Test Accuracy:", nn.predict(np.array(test_x), np.array(test_y)))
