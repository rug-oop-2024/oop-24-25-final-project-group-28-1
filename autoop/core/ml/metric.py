from abc import ABC, abstractmethod
import numpy as np


METRICS = [
    "mean_squared_error",
    "accuracy",
    "mean_absolute_error",
    "r2_score",
    "precision",
    "recall",
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
    elif name == "mean_absolute_error":
        return MeanAbsoluteError()
    elif name == "r2_score":
        return R2Score()
    elif name == "precision":
        return Precision()
    elif name == "recall":
        return Recall()
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


# Regression Metrics
class MeanSquaredError(Metric):
    """
    Mean Squared Error metric implementation.
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates the Mean Squared Error between ground truth and predictions.
        """
        return np.mean((y_true - y_pred) ** 2)


# Regression Metrics
class MeanAbsoluteError(Metric):
    """
    Calculates Mean Absolute Error between ground truth and predictions.
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y_true - y_pred))


# Regression Metrics
class R2Score(Metric):
    """
    Calculates R-squared for regression. (1 - residual sum of squares/total
    sum of squares)
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        total_variance = np.sum((y_true - np.mean(y_true)) ** 2)
        residual_variance = np.sum((y_true - y_pred) ** 2)
        return 1 - (residual_variance / total_variance)


# Classification Metrics
class Accuracy(Metric):
    """
    Accuracy metric implementation for classification by comparing ground
    truth and predictions.
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(y_true == y_pred)


# Classification Metrics
class Precision(Metric):
    """
    Calculates Precision for binary classification.
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        true_positives = np.sum((y_pred == 1) & (y_true == 1))
        predicted_positives = np.sum(y_pred == 1)
        return true_positives / predicted_positives if (
            predicted_positives > 0
        ) else 0.0


# Classification Metrics
class Recall(Metric):
    """
    Calculates Recall for binary classification.
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        true_positives = np.sum((y_pred == 1) & (y_true == 1))
        actual_positives = np.sum(y_true == 1)
        return true_positives / actual_positives if (
            actual_positives > 0
        ) else 0.0
