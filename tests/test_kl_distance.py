import numpy as np
import pandas as pd

from lnl_surrogate.kl_distance_computer import kl_distance


def test_kl_distance():
    # Create a dataframe
    x1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 1000)
    x2 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 1100)
    x3 = np.random.multivariate_normal([1, 1], [[1, 0], [0, 1]], 1000)

    kl1 = kl_distance(x1, x2)
    kl2 = kl_distance(x1, x3)

    # Check that the KL distance is 0
    assert kl1 < kl2
