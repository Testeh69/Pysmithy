import numpy as np




class LinearRegression:


    def __init__(self, resolution = -1):
        self.methode_resolution = resolution
        self.weight = None

    def fit(self,X_train,y_train, learning_rate = 0.001, epochs = 10000):
        shape_vector_weight = X_train.shape[1]
        y_train = y_train.reshape(-1,1)
        n_samples, n_features = X_train.shape
        self.weight = np.random.randn( n_features,1) *0.01
        result = X_train.T@X_train
        if self.methode_resolution == -1:
                self.weight = np.linalg.pinv(result)@np.transpose(X_train)@y_train
        else:
            for _ in range(epochs):
                self.backward_propagation(X_train,y_train,learning_rate)

        return self.weight
    


    def predict(self,X_test):
        return X_test@self.weight
    

    def backward_propagation(self,X_train,y_train, learning_rate):
            gradient =  (2/X_train.shape[0]) * np.transpose(X_train)@(X_train@self.weight - y_train)
            self.weight -= learning_rate * gradient



 

X = np.array([[1,2,3],[7,8,9]])
y = np.array([1,2])
test = LinearRegression()
result = test.fit(X,y)
print(result)
lol = X@result
print(lol)