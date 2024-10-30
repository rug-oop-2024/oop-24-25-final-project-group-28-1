import numpy as np
from autoop.core.ml.model.regression import MultipleLinearRegression
from sklearn.datasets import make_regression

X,y = make_regression(n_samples=100, n_features=4, n_informative=3, noise=20, random_state=99)
#X = np.array([[1, 2], [2, 3], [4, 5], [3, 2], [5, 6]])
#y = np.array([5, 7, 10, 8, 12])

# Multiple linear regression test
# Custom implementation
model = MultipleLinearRegression()
model.fit(X, y)
predictions = model.predict(X)
print("Custom Model Predictions:", predictions)
print(model.coef, model.intercept)