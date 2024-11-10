import numpy as np
from sklearn.linear_model import Ridge
from autoop.core.ml.model import Model


class RidgeRegressionModel(Model):
    def __init__(self, parameters=None):
        super().__init__(model_type="regression", parameters=parameters)
        self.model = Ridge(**(parameters or {}))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)