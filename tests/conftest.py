import os

import pytest
from lnl_computer.mock_data import MockData, generate_mock_data
import numpy as np
from scipy.stats import multivariate_normal



TEST_DIR = "out_test"


@pytest.fixture
def tmpdir() -> str:
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR, exist_ok=True)
    return TEST_DIR


@pytest.fixture
def mock_data() -> MockData:
    return generate_mock_data(outdir=TEST_DIR)


class FakeData:
    def __init__(self, inputs, model, search_space):
        self.inputs = inputs
        self.model = model
        self.outputs = model.predict(inputs)
        self.search_space = search_space
        self.truth = dict(x1=0.5, x2=-0.2)


class DummyModel:
    def __init__(self, f):
        self.f = f

    def predict(self, x):
        return self.f(x)

def _gaus2d(xy):
    """normalized 2D gaussian"""
    norm = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])
    res = -1 *  norm.pdf(xy)
    return res, res


@pytest.fixture
def mock_inout_data() -> FakeData:
    radial = np.linspace(0, 2 * np.pi, 20)
    from trieste.space import Box
    return FakeData(
        inputs=np.array([np.cos(radial), np.sin(radial)]).T,
        model=DummyModel(_gaus2d),
        search_space=Box((-1, -1), (1, 1)),
    )
