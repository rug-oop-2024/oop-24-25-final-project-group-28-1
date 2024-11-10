from sklearn.linear_model import LinearRegression
import numpy as np
from typing import Optional, Dict


class MultipleLinearRegression():
    def __init__(
        self,
        name: str = "MultipleLinearRegression",
        model_type: str = "regression",
        asset_path: str = "",
        parameters: Optional[Dict] = None
    ):
        super().__init__(
            name=name,
            model_type=model_type,
            parameters=parameters
        )
        self.model = LinearRegression(**(parameters or {}))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
