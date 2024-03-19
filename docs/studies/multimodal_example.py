from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.integrate import quad
from scipy.stats import gaussian_kde


def generate_data(mu: List[float], sigma: List[float], N: int):
    """Generate some data."""
    D = len(mu)
    df = pd.DataFrame(
        {f"x[{i}]": np.random.normal(mu[i], sigma[i], N) for i in range(D)}
    )
    x = np.random.choice([0, 1], N)
    df["x"] = df["x[0]"] * (1 - x) + df["x[1]"] * x
    return df


def plot_data(data: pd.DataFrame):
    """Plot the data."""
    fig, ax = plt.subplots()
    # scipy KDE
    kde = gaussian_kde(data["x"])
    x = np.linspace(-2, 3, 1000)
    ax.plot(x, kde(x), label="scipy")
    ax.legend()
    # KDE area from [-2, -0.1] vs area from [-0.1, 2]
    area1, _ = quad(kde, -2, -0.1)
    area2, _ = quad(kde, -0.1, 3)
    assert np.isclose(area1, area2, rtol=0.2)

    plt.show()


if __name__ == "__main__":
    data = generate_data([-1, 1], [0.1, 0.4], 10000)
    plot_data(data)
    # area from [-2, -0.1] vs area from [-0.1, 2]
    # should be the same
