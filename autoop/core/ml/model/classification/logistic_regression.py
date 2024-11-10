import numpy as np
from autoop.core.ml.model import Model
from sklearn.linear_model import LogisticRegression
from typing import Optional, Dict


class LogisticRegressionModel(Model):
    def __init__(
        self,
        name: str = "LogisticRegression",
        model_type: str = "classification",
        asset_path: str = "",
        parameters: Optional[Dict] = None
    ):
        super().__init__(
            name=name,
            model_type=model_type,
            parameters=parameters
        )
        self.model = LogisticRegression(**(parameters or {}))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
