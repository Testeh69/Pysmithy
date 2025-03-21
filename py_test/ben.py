from py_impl.model.linear import LogisticRegression
import numpy as np




X_train = np.random.rand(100, 2)  # 100 exemples, 2 features
y_train = (np.random.rand(100, 1) > 0.5).astype(int)  # Labels binaires
X_test = np.random.rand(10, 2)  # 10 nouveaux exemples

model = LogisticRegression()
model.fit(X_train,y_train)
print(model.predict(X_test))