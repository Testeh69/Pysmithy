import sys
import os
sys.path.insert(0, os.path.abspath("C:/Users/Orefice/OneDrive/Bureau/IT/Pysmithy"))

import numpy as np
from py.ops.activation import sigmoid


#svd
"""Régression"""
class LinearRegression:


    def __init__(self, resolution = -1):
        self.methode_resolution = resolution
        self.weights = None
        self.bias = 0

    def fit(self,X_train,y_train, learning_rate = 0.1, epochs = 10000):
        n_samples, n_features = X_train.shape
        self.weights = np.random.randn(n_features,1) *0.1
        self.bias = np.random.randint(1)* 0.01

        result = X_train.T@X_train
        if self.methode_resolution == -1:
                self.weights = np.linalg.pinv(result)@np.transpose(X_train)@y_train
        else:
            for _ in range(epochs):
                self.backward_propagation(X_train,y_train,learning_rate)

        return self.weights
    


    def predict(self,X_test):
        return X_test@self.weights + self.bias
    

    def backward_propagation(self,X_train,y_train, learning_rate):
        n_samples = X_train.shape[0]
        gradient =  (2/X_train.shape[0]) * np.transpose(X_train)@(X_train @ self.weights + self.bias - y_train)
        gradient_bias = (2/n_samples) * np.sum(X_train @ self.weights + self.bias - y_train)
        self.weights -= learning_rate * gradient
        self.bias -= learning_rate * gradient_bias





""" Classification"""
class LogisticRegression(LinearRegression):
     

    def __init__(self):
        super().__init__()
    
    
    def fit(self,X_train,y_train, learning_rate = 0.001, epochs = 10000):
        n_samples, n_features = X_train.shape
        self.weights = np.random.randn( n_features,1) *0.01
        self.biais = np.random.randn(1)*0.01
        for _ in range(epochs):
            self.backward_propagation(X_train,y_train,learning_rate=learning_rate)
        return self.weights, self.biais


    def predict(self,X_test):
        result_linear = super().predict(X_test=X_test)
        return sigmoid(result_linear)

          
    def backward_propagation(self, X_train, y_train, learning_rate):
        y_pred = self.predict(X_train) - y_train
        gradient = 1/X_train.shape[0]* (X_train.T*(y_pred))
        self.weights -= learning_rate * gradient