import numpy as np
from numpy.typing import NDArray


class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        return np.round(X @ weights, 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        pred = np.asarray(model_prediction).reshape(-1)
        truth = np.asarray(ground_truth).reshape(-1)
        return round(float(np.mean((pred - truth) ** 2)), 5)
