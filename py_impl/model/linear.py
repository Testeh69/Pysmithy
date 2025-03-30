import sys
import os
sys.path.insert(0, os.path.abspath("C:/Users/Orefice/OneDrive/Bureau/IT/Pysmithy"))

import numpy as np
from py.ops.activation import sigmoid



"""Régression"""
class LinearRegression:


    def __init__(self, resolution:int = -1):
        self.methode_resolution = resolution
        self.weights = None
        self.bias = 0
        self.X_train = None
        self.y_train = None
        self.gradient = []

    def fit(self,X_train,y_train, learning_rate:float = 0.0001, epochs:int = 1000, gradient_limit = 2.0, sample:int  = - 1):
        n_samples, n_features = X_train.shape
        self.X_train = X_train
        self.y_train = y_train
        self.weights = np.random.randn(n_features,1) *0.1
        self.bias = np.random.rand()
        result = X_train.T@X_train
        if self.methode_resolution == -1:
                self.weights = np.linalg.pinv(result)@np.transpose(X_train)@y_train
        else:
            for _ in range(epochs):
                if sample == -1:
                    self.backward_propagation(learning_rate, sample = 10, gradient_limit=gradient_limit)
                else:
                    self.backward_propagation(learning_rate, sample = sample, gradient_limit=gradient_limit)
        return self.weights, self.bias



    def predict(self,X_test):
        return X_test@self.weights + self.bias


    def backward_propagation(self,learning_rate, sample, gradient_limit = 2.0):
        len_batches = int(self.X_train.shape[0]//sample)
        
        for batch in range(1,sample+1):
            idx_begin = (batch-1)*len_batches
            idx_end = batch*len_batches
            X_batch = self.X_train[ idx_begin:idx_end]
            y_batch = self.y_train[idx_begin:idx_end]
            gradient =  np.clip((2/sample) *X_batch.T@(X_batch @ self.weights + self.bias - y_batch.reshape(-1,1)),-gradient_limit,gradient_limit)
            gradient_bias = np.clip((2/sample) * np.sum(X_batch @ self.weights + self.bias - y_batch.reshape(-1,1)),-gradient_limit,gradient_limit)
            self.weights -= learning_rate * gradient
            self.bias -= learning_rate * gradient_bias
            self.gradient.append(gradient)





""" Classification"""



class LogisticRegression(LinearRegression):
     

    def __init__(self):
        super().__init__()
    
    
    def fit(self,X_train,y_train, learning_rate = 0.001, epochs = 10000):
        n_samples, n_features = X_train.shape
        self.weights = np.random.randn( n_features,1) *0.01
        self.bias = np.random.randn(1)*0.01
        for _ in range(epochs):
            self.backward_propagation(X_train,y_train,learning_rate=learning_rate)
        return self.weights, self.bias


    def predict(self,X_test):
        result_linear = super().predict(X_test=X_test)
        print(result_linear)
        return sigmoid(float(result_linear))

          
    def backward_propagation(self, X_train, y_train, learning_rate):
        n_samples = X_train.shape[0]
        y_pred = self.predict(X_train) - y_train
        gradient = 1/X_train.shape[0]* (X_train.T*(y_pred))
        gradient_bias = (2/n_samples) * np.sum(X_train @ self.weights + self.bias - y_train)
        self.weights -= learning_rate * gradient
        self.bias -= learning_rate * gradient_bias