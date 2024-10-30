from sklearn.linear_model import LinearRegression 
import numpy as np

class MultipleLinearRegression():
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        if self.alpha <= 0:
            raise ValueError("alpha must be a positive non-zero integer.")
        self.parameters = {
             "fit_intercept": True,
             "precompute": False,
             "copy_X": True,
             "max_iter": 1000,
             "tol": 0.0001,
             "warm_start": False,
             "positive": False,
             "random_stat": None,
             "selection": "cyclic"}
        self.MultipleLinearRegression = LinearRegression()
        

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Utilizes LinearRegression.fit() to train the model given observations (X) and
        ground truth (y).

        :param X: numpy array of shape (n_samples, n_features) -
        training data.
        :param y: numpy array of shape (n_samples,) - the target values.
        :return: None.
        """
        # Validation to ensure inputs are right types and dimension
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("Both X and y should be numpy arrays.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("""The number of samples in X and y should
                             be equal.""")
        f = self.MultipleLinearRegression.fit(X, y)
        self.coef = f.coef_
        self.intercept = f.intercept_
        self.f = f

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Utilizes Lasso to predict the target values based on
        the observations (X).

        :param X: numpy array of shape (n_samples, n_features)
        - the input data for predictions.
        :return: numpy array of shape (n_samples,) - the predicted values.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be a numpy array.")
        return self.f.predict(X)

    def set_params(self, d: dict) -> None:
        """
        Sets parameters for training the model.

        :param d: parameter dictionary
        :return: None
        """
        self.MultipleLinearRegression.set_params(**d)


