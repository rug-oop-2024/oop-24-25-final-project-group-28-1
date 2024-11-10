import numpy as np
from autoop.core.ml.model import Model
from sklearn.ensemble import RandomForestClassifier
from typing import Optional, Dict


class RandomForestClassifierModel(Model):
    def __init__(
        self,
        name: str = "RandomForestClassifier",
        model_type: str = "classification",
        asset_path: str = "",
        parameters: Optional[Dict] = None
    ):
        super().__init__(
            name=name,
            model_type=model_type,
            parameters=parameters
        )
        self.model = RandomForestClassifier(**(parameters or {}))
        self._model_type = model_type

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trained:
            raise ValueError("""Model must be trained before making
                             predictions.""")
        return self.model.predict(X)
