from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.metric import get_metric
import numpy as np
from copy import deepcopy
from typing import Dict, Any, Optional


class Model(ABC):
    def __init__(
        self,
        name: str,
        model_type: str,
        parameters: Optional[Dict[str, Any]] = None
    ):
        # Use Artifact as attribute
        self._artifact = Artifact(
            name=name,
            tags="No Tags",
            type=model_type,
            asset_path="",
            data=b"",
            features=""
        )
        self._model_type = model_type
        self._parameters = parameters or {}
        self.trained = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Abstract method to train the model on given data.
        Args:
            X (np.ndarray): Features for training.
            y (np.ndarray): Target labels or values.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Abstract method to make predictions based on input features.
        Args:
            X (np.ndarray): Features for prediction.
        Returns:
            np.ndarray: Predicted values or labels.
        """
        pass

    def evaluate(
        self, X: np.ndarray,
        y: np.ndarray,
        metric_name: str
    ) -> float:
        """
        Evaluates the model on the given data using a specified metric.
        Args:
            X (np.ndarray): Features for evaluation.
            y (np.ndarray): Ground truth values or labels.
            metric_name (str): Name of the metric to use for evaluation.
        Returns:
            float: Calculated metric value.
        """
        predictions = self.predict(X)
        metric = get_metric(metric_name)
        return metric(y, predictions)

    def save_model(self, path: str) -> None:
        """
        Saves model parameters or state to a file.
        """
        self._artifact.metadata["parameters"] = deepcopy(self._parameters)
        self._artifact.metadata["trained"] = self.trained
        self.asset_path = path

    def load_model(self, path: str) -> None:
        """
        Loads model parameters or state from a file.
        """
        self.asset_path = path
        self._parameters = deepcopy(
            self._artifact.metadata.get("parameters", {})
        )
        self.trained = self._artifact.metadata.get("trained", False)

    @property
    def parameters(self) -> Dict[str, Any]:
        """
        Gets the current model parameters.
        """
        return deepcopy(self._parameters)

    @parameters.setter
    def parameters(self, params: Dict[str, Any]) -> None:
        """
        Sets or updates model parameters.
        """
        if isinstance(params, dict):
            self._parameters.update(params)
        else:
            raise ValueError("Parameters must be provided as a dictionary.")
