import jax.numpy as np
import numpy as onp
import pytest
from jax.config import config

config.update("jax_enable_x64", True)

from jax_sindy.differentiation import finite_difference


@pytest.fixture
def data_derivative_1d():
    x = 2 * np.linspace(1, 100, 100)
    x_dot = 2 * np.ones(100).reshape(-1, 1)
    return x, x_dot


@pytest.fixture
def data_derivative_2d():
    x = np.zeros((100, 2))
    x = x.at[:, 0].set(2 * np.linspace(1, 100, 100))
    x = x.at[:, 1].set(-10 * np.linspace(1, 100, 100))

    x_dot = np.ones((100, 2))
    x_dot = x_dot.at[:, 0].mul(2)
    x_dot = x_dot.at[:, 1].mul(-10)
    return x, x_dot


@pytest.mark.parametrize(
    "data, order",
    [
        (pytest.lazy_fixture("data_derivative_1d"), 1),
        (pytest.lazy_fixture("data_derivative_2d"), 1),
    ],
)
def test_finite_difference(data, order):
    x, x_dot = data
    diff = finite_difference(order=order)
    t = np.array([1])
    onp.testing.assert_allclose(diff(x, t), x_dot)
