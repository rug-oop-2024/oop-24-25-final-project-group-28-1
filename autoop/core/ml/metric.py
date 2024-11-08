from abc import ABC, abstractmethod
import numpy as np


METRICS = [
    "mean_squared_error",
    "accuracy",
]  # add the names (in strings) of the metrics you implement


def get_metric(name: str):
    """
    Factory function to get a metric by name.
    Return a metric instance given its str name.
    """
    if name == "mean_squared_error":
        return MeanSquaredError()
    elif name == "accuracy":
        return Accuracy()
    else:
        raise ValueError(f"Unknown metric name: {name}")


class Metric(ABC):
    """
    Base class for all metrics.
    """
    # remember: metrics take ground truth and prediction as input
    # and return a real number
    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the metric given ground truth and predictions.

        :param y_true: numpy array of shape (n_samples,) - ground truth values.
        :param y_pred: numpy array of shape (n_samples,) - predicted values.
        :return: A real number representing the computed metric.
        """
        pass


class MeanSquaredError(Metric):
    """
    Mean Squared Error metric implementation.
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates the Mean Squared Error between ground truth and predictions.
        """
        return np.mean((y_true - y_pred) ** 2)


class Accuracy(Metric):
    """
    Accuracy metric implementation for classification.
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates the accuracy score by comparing ground truth and
        predictions.
        """
        return np.mean(y_true == y_pred)
