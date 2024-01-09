import os

import pytest
from lnl_computer.mock_data import MockData, generate_mock_data


TEST_DIR = "out_test"


@pytest.fixture
def tmpdir() -> str:
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR, exist_ok=True)
    return TEST_DIR



@pytest.fixture
def mock_data() -> MockData:
    return generate_mock_data(outdir=TEST_DIR)
