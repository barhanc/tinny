# pylint: disable=missing-function-docstring, missing-class-docstring, missing-module-docstring

import logging

from typing import Literal
from itertools import pairwise

import pytest
import numpy as np
import hypothesis.extra.numpy

from numpy.typing import NDArray
from hypothesis import given, assume
from hypothesis import strategies as some

from nncore import nn

# #########################################################
# Helper functions
# #########################################################


def compute_grad_x_finite_diff(
    f: nn.Layer,
    x: NDArray,
    i: tuple[int, ...],
    j: tuple[int, ...],
    eps: float = 1e-8,
) -> float:
    dx = np.zeros_like(x)
    dx[i] = eps
    dy = f.forward(x + dx, training=True) - f.forward(x - dx, training=True)
    return dy[j] / (2 * dx[i])


def compute_grad_x_backprop(
    f: nn.Layer,
    x: NDArray,
    i: tuple[int, ...],
    j: tuple[int, ...],
) -> float:
    y = f.forward(x, training=True)
    grad_y = np.zeros_like(y)
    grad_y[j] = 1.0
    grad_x = f.backward(grad_y)
    return grad_x[i]


def some_shape(ndim: int, min_length: int, max_length):
    return some.tuples(*(some.integers(min_length, max_length) for _ in range(ndim)))


def some_tensor(*shape: int, dtype=np.float64, domain: tuple[float, float] = (-5.0, +5.0)):
    return hypothesis.extra.numpy.arrays(dtype, shape, elements=some.floats(*domain))


def some_multi_index(*shape: int):
    return some.tuples(*(some.integers(0, length - 1) for length in shape))


# #########################################################
# Tests
# #########################################################


@pytest.mark.parametrize("activation_type", [nn.Sigmoid, nn.ReLU, nn.Tanh])
@given(some_shape(ndim=2, min_length=1, max_length=256), some.data())
def test_activation_grad_x_correctness(
    activation_type: type[nn.Activation],
    shape: tuple[int, int],
    data: some.DataObject,
):
    i = data.draw(some_multi_index(*shape), label="Multi-index")
    x = data.draw(some_tensor(*shape), label="Input tensor")
    f = activation_type()

    # Skip gradient check at ReLU non-differentiable point
    if activation_type is nn.ReLU:
        assume(not np.isclose(x[i], 0.0))

    grad_bp = compute_grad_x_backprop(f, x, i, i)
    grad_fd = compute_grad_x_finite_diff(f, x, i, i)

    logging.info(
        "%s ∂Yj/∂Xi , %+12.6f , %+12.6f , %.6f",
        activation_type.__name__,
        grad_bp,
        grad_fd,
        abs(grad_bp - grad_fd),
    )

    assert np.isclose(grad_bp, grad_fd)


@given(
    some.integers(1, 64),
    some.integers(1, 64),
    some.integers(1, 64),
    some.one_of(some.just("He"), some.just("Xavier")),
    some.data(),
)
def test_linear_grad_x_correctness(
    batch_size: int,
    in_features: int,
    out_features: int,
    init_method: Literal["He", "Xavier"],
    data: some.DataObject,
):
    i = data.draw(some_multi_index(batch_size, in_features), label="Input multi-index")
    j = data.draw(some_multi_index(batch_size, out_features), label="Output multi-index")
    x = data.draw(some_tensor(batch_size, in_features), label="Input tensor")
    f = nn.Linear(in_features, out_features, init_method)

    grad_bp = compute_grad_x_backprop(f, x, i, j)
    grad_fd = compute_grad_x_finite_diff(f, x, i, j)

    logging.info(
        "Linear ∂Yj/∂Xi , %+12.6f , %+12.6f , %.6f",
        grad_bp,
        grad_fd,
        abs(grad_bp - grad_fd),
    )

    assert np.isclose(grad_bp, grad_fd)


def test_linear_grad_params_correctness():
    assert False


@given(
    some_shape(ndim=4, min_length=1, max_length=8),
    some.integers(1, 16),
    some.tuples(some.integers(1, 5), some.integers(1, 5)),
    some.tuples(some.integers(1, 2), some.integers(1, 2)),
    some.tuples(some.integers(0, 2), some.integers(0, 2)),
    some.one_of(some.just("He"), some.just("Xavier")),
    some.data(),
)
def test_conv2d_grad_x_correctness(
    in_shape: tuple[int, int, int, int],
    out_channels: int,
    kernel_size: tuple[int, int],
    strides: tuple[int, int],
    padding: tuple[int, int],
    init_method: Literal["Xavier", "He"],
    data: some.DataObject,
):
    # Standard notation when describing convolutions, so pylint: disable=invalid-name
    B, in_channels, H, W = in_shape
    H_out = int(1 + (H + 2 * padding[0] - kernel_size[0]) / strides[0])
    W_out = int(1 + (W + 2 * padding[1] - kernel_size[1]) / strides[1])
    # pylint: enable=invalid-name

    assume(H_out >= 1 and W_out >= 1)

    i = data.draw(some_multi_index(*in_shape), "Input multi-index")
    j = data.draw(some_multi_index(B, out_channels, H_out, W_out), "Output multi-index")
    x = data.draw(some_tensor(*in_shape), "Input image")
    f = nn.Conv2D(in_channels, out_channels, kernel_size, strides, padding, init_method)

    grad_bp = compute_grad_x_backprop(f, x, i, j)
    grad_fd = compute_grad_x_finite_diff(f, x, i, j)

    logging.info(
        "Conv2D ∂Yj/∂Xi , %+12.6f , %+12.6f , %.6f",
        grad_bp,
        grad_fd,
        abs(grad_bp - grad_fd),
    )

    assert np.isclose(grad_bp, grad_fd)


def test_conv2d_grad_params_correctness():
    assert False


@given(
    some.integers(1, 64),
    some.lists(some.integers(1, 64), min_size=1, max_size=8),
    some.data(),
)
def test_sequential_grad_flow_correctness(
    batch_size: int,
    dims: list[int],
    data: some.DataObject,
):
    layers = []
    for dim_a, dim_b in pairwise(dims):
        layers.append(nn.Linear(dim_a, dim_b, init_method="Xavier"))
        layers.append(nn.Sigmoid())

    f = nn.Sequential(*layers)
    i = data.draw(some_multi_index(batch_size, dims[0]), label="Input multi-index")
    j = data.draw(some_multi_index(batch_size, dims[-1]), label="Output multi-index")
    x = data.draw(some_tensor(batch_size, dims[0]), label="Input tensor")

    grad_bp = compute_grad_x_backprop(f, x, i, j)
    grad_fd = compute_grad_x_finite_diff(f, x, i, j)

    logging.info(
        "Sequential ∂Yj/∂Xi , %+12.6f , %+12.6f , %.6f",
        grad_bp,
        grad_fd,
        abs(grad_bp - grad_fd),
    )

    assert np.isclose(grad_bp, grad_fd)
