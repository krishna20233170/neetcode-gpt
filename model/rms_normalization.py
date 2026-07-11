import math
from typing import List

import numpy as np


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # RMS Normalization (LayerNorm without mean centering or beta)
        # Normalize x, then scale by gamma
        x_arr = np.array(x, dtype=np.float64)
        gamma_arr = np.array(gamma, dtype=np.float64)

        rms = math.sqrt(float(np.mean(x_arr ** 2) + eps))
        x_hat = x_arr / rms
        return np.round(gamma_arr * x_hat, 4).tolist()
