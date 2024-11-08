import numpy as np
from autoop.core.ml.model import Model
from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel(Model):
    def __init__(self, parameters=None):
        super().__init__(model_type="classification", parameters=parameters)
        self.model = LogisticRegression(**(parameters or {}))

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
