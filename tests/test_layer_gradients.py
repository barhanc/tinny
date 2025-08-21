# pylint: disable=missing-function-docstring, missing-class-docstring, missing-module-docstring


import pytest
import numpy as np
import hypothesis.extra.numpy

from numpy.typing import NDArray
from hypothesis import given, assume
from hypothesis import strategies as st

from nncore import nn


def compute_grad_fd(
    f: nn.Layer,
    x: NDArray,
    i: tuple[int, ...],
    j: tuple[int, ...],
    eps: float = 1e-8,
) -> float:
    dx = np.zeros_like(x)
    dx[j] = eps
    dy = f.forward(x + dx, training=True) - f.forward(x - dx, training=True)
    return dy[i] / (2 * dx[j])


def compute_grad_bp(
    f: nn.Layer,
    x: NDArray,
    i: tuple[int, ...],
    j: tuple[int, ...],
) -> float:
    y = f.forward(x, training=True)
    grad_y = np.zeros_like(y)
    grad_y[i] = 1.0
    grad_x = f.backward(grad_y)
    return grad_x[j]


def some_shape(ndim: int, min_length: int = 1, max_length: int = 128):
    return st.tuples(*(st.integers(min_length, max_length) for _ in range(ndim)))


def some_tensor(*shape: int, dtype=np.float64, domain: tuple[float, float] = (-5.0, +5.0)):
    return hypothesis.extra.numpy.arrays(dtype, shape, elements=st.floats(*domain))


def some_multi_index(*shape: int):
    return st.tuples(*(st.integers(0, length - 1) for length in shape))


@pytest.mark.parametrize("activation_type", [nn.Sigmoid, nn.ReLU, nn.Tanh])
@given(some_shape(ndim=2), st.data())
def test_activation_grad_correctness(
    activation_type: type[nn.Activation],
    shape: tuple[int, int],
    data: st.DataObject,
):
    i = data.draw(some_multi_index(*shape), label="Multi-index")
    x = data.draw(some_tensor(*shape), label="Input tensor")
    f = activation_type()

    # Skip gradient check at ReLU non-differentiable point
    if activation_type is nn.ReLU:
        assume(not np.isclose(x[i], 0.0))

    grad_bp = compute_grad_bp(f, x, i, i)
    grad_fd = compute_grad_fd(f, x, i, i)

    assert np.isclose(grad_bp, grad_fd)
