import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        epochs: int,
        lr: float,
    ) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        w = np.zeros(X.shape[1], dtype=np.float64)
        b = 0.0
        n = X.shape[0]

        for _ in range(epochs):
            y_hat = X @ w + b
            dldw = (2 / n) * (X.T @ (y_hat - y))
            dldb = (2 / n) * np.sum(y_hat - y)
            w = w - lr * dldw
            b = b - lr * dldb

        return np.round(w, 5), round(float(b), 5)
