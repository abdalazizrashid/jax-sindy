"""Calculate derivatives."""

from typing import Callable

import jax.numpy as np
from jax import jit


@jit
def finite_difference(order: int = 1) -> Callable:
    """Finite difference derivatives.

    Differentiate using finite difference method.
    Forward difference for first order and centered difference for the second
    order

    Args:
        order (int): 1 or 2 (default=2)

    Returns:
        Callable: a function with the fallowing signature:
        .. code-block:: haskell

            f: x -> [t] -> dx

    Example:
        >>> t = np.linspace(0, 10)
        >>> x = np.sin(t)
        >>> diff = finite_difference(order=1)
        >>> diff(x, t)
        DeviceArray([[ 0.89619225],
                     [-0.01135296],
                     [-0.9062661 ],
                     [-0.7928059 ],
                     [ 0.20278412],
                     [ 0.97274274],
                     [ 0.6603614 ],
                     [-0.38678235],
                     [-1.0035661 ],
                     [-1.2048244 ]], dtype=float32)
    """

    if order == 1:
        return _forward_difference
    if order > 1:
        raise NotImplementedError

    # should never reach this return
    return lambda: None


# pylint: disable=C0103
def _forward_difference(
    x: np.ndarray,
    t: np.ndarray,
    drop_endpoints: bool = True,
) -> np.ndarray:
    """
    First order forward difference
    (and 2nd order backward difference for final point).
    """

    # maintain compatibility with scikit-learn
    x = np.expand_dims(x, -1) if np.ndim(x) == 1 else x
    x_dot = np.full_like(x, fill_value=np.nan)

    # Uniform timestep (assume t contains dt) if not calculate dt
    diff = lambda x, t: x[1:, ...] - x[:-1, ...] / t
    diff_endpoints = (
        lambda x, t: (3 * x[-1, ...] / 2 - 2 * x[-2, ...] + x[-3, ...] / 2) / t
    )

    if len(t) == 1:
        x_dot = x_dot.at[:-1, ...].set(diff(x, t))
        x_dot = (
            x_dot.at[-1, ...].set(diff_endpoints(x, t))
            if drop_endpoints
            else x_dot
        )

    else:
        t_diff: np.ndarray = t[1:] - t[:-1]
        x_dot = x_dot.at[:-1, ...].set(diff(x, np.expand_dims(t_diff, -1)))
        x_dot = (
            x_dot.at[-1, ...].set(diff_endpoints(x, t_diff[-1]))
            if drop_endpoints
            else x_dot
        )

    return x_dot
