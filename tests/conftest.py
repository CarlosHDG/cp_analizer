import os
import sys

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from data_analizer import ProcessCapabilityAnalizer


@pytest.fixture
def sample_data():
    np.random.seed(42)
    return np.random.normal(loc=100.0, scale=2.0, size=(10, 5))


@pytest.fixture
def analyzer(sample_data):
    return ProcessCapabilityAnalizer(
        data_subgroups=sample_data,
        usl=105.0,
        lsl=95.0,
        target_mean=100.0,
    )
