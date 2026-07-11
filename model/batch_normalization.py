import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(
        self,
        x: List[List[float]],
        gamma: List[float],
        beta: List[float],
        running_mean: List[float],
        running_var: List[float],
        momentum: float,
        eps: float,
        training: bool,
    ) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats
        # Affine: y = gamma * x_hat + beta
        x = np.array(x, dtype=np.float64)
        gamma = np.array(gamma, dtype=np.float64)
        beta = np.array(beta, dtype=np.float64)
        running_mean = np.array(running_mean, dtype=np.float64)
        running_var = np.array(running_var, dtype=np.float64)

        if training:
            mean_batch = np.mean(x, axis=0)
            var_batch = np.var(x, axis=0)
            x_hat = (x - mean_batch) / np.sqrt(var_batch + eps)
            running_mean = (1 - momentum) * running_mean + momentum * mean_batch
            running_var = (1 - momentum) * running_var + momentum * var_batch
        else:
            x_hat = (x - running_mean) / np.sqrt(running_var + eps)

        y = gamma * x_hat + beta

        return (
            np.round(y, 4).tolist(),
            np.round(running_mean, 4).tolist(),
            np.round(running_var, 4).tolist(),
        )
