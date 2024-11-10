import numpy as np
from autoop.core.ml.model import Model
from sklearn.ensemble import RandomForestClassifier
from typing import Optional, Dict


class RandomForestClassifierModel(Model):
    def __init__(
        self,
        name: str,
        asset_path: str,
        parameters: Optional[Dict] = None
    ):
        super().__init__(model_type="classification", name=name, asset_path=asset_path, parameters=parameters)
        self.model = RandomForestClassifier(**(parameters or {}))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trained:
            raise ValueError("Model must be trained before making predictions.")
        return self.model.predict(X)
