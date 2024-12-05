import numpy as np
import pandas as pd

from lnl_surrogate.kl_distance import compute_statistics


def generate_posterior(n_samples: int, n_params: int) -> pd.DataFrame:
    mean = np.zeros(n_params)
    cov = np.eye(n_params)
    post = np.random.multivariate_normal(mean, cov, n_samples)
    labels = [f"param_{i}" for i in range(n_params)]
    return pd.DataFrame(post, columns=labels)

def test_kl_distance():
    # Create a dataframe
    x1 = pd.DataFrame(np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 1000))
    x2 = pd.DataFrame(np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 1100))
    x3 = pd.DataFrame(np.random.multivariate_normal([1, 1], [[1, 0], [0, 1]], 1000))

    kl1 = compute_statistics(x1, x2)
    kl2 = compute_statistics(x1, x3)

    # Check that the KL distance is 0
    assert kl1 < kl2


