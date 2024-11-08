import numpy as np
from autoop.core.ml.model import Model
from sklearn.ensemble import RandomForestClassifier


class RandomForestClassifierModel(Model):
    def __init__(self, parameters=None):
        super().__init__(model_type="classification", parameters=parameters)
        self.model = RandomForestClassifier(**(parameters or {}))

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
