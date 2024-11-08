from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.metric import get_metric
import numpy as np
from copy import deepcopy
from typing import Literal, Dict, Any


class Model(Artifact, ABC):
    """
    Base class for all machine learning models.
    Attributes:
        model_type (str): Type of model, "classification" or "regression".
        parameters (dict): Hyperparameters or model parameters for training.
    """

    model_type: Literal["classification", "regression"]
    _parameters: Dict[str, Any]
    trained: bool = False

    def __init__(
        self,
        model_type: Literal["classification", "regression"],
        parameters: Dict[str, Any] = None
    ):
        super().__init__(asset_path="", data=b"", metadata={})
        self.model_type = model_type
        self._parameters = parameters if parameters is not None else {}

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
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
        self.metadata["parameters"] = deepcopy(self._parameters)
        self.metadata["trained"] = self.trained
        self.asset_path = path

    def load_model(self, path: str) -> None:
        """
        Loads model parameters or state from a file.
        """
        self.asset_path = path
        self._parameters = deepcopy(self.metadata.get("parameters", {}))
        self.trained = self.metadata.get("trained", False)

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
