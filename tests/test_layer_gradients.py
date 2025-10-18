# pylint: disable=missing-function-docstring, missing-class-docstring, missing-module-docstring
# pylint: disable=invalid-name

import logging

from copy import deepcopy
from typing import Literal
from itertools import pairwise

import pytest
import numpy as np
import hypothesis.extra.numpy

from numpy.typing import NDArray, DTypeLike
from hypothesis import given, assume
from hypothesis import strategies as some

from tinny import nn


def compute_grad_x_fd(
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


def compute_grad_x_bp(
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


def compute_grad_params_bp(
    f: nn.Layer,
    x: NDArray,
    idxs: list[tuple[int, ...]],
    j: tuple[int, ...],
) -> list[float]:
    y = f.forward(x, training=True)
    grad_y = np.zeros_like(y)
    grad_y[j] = 1.0
    f.backward(grad_y)

    grads = []
    for i, grad_p in zip(idxs, f.gradients()):
        assert grad_p is not None
        grads.append(grad_p[i])

    return grads


def compute_grad_params_fd(
    f: nn.Layer,
    x: NDArray,
    idxs: list[tuple[int, ...]],
    j: tuple[int, ...],
    eps: float = 1e-8,
) -> list[float]:
    grads = []

    for k, i in enumerate(idxs):
        f1 = deepcopy(f)
        f2 = deepcopy(f)

        f1.parameters()[k][i] += eps
        f2.parameters()[k][i] -= eps

        dy = f1.forward(x, training=True) - f2.forward(x, training=True)
        grads.append(dy[j] / (2 * eps))

    return grads


def some_shape(ndim: int, min_length: int, max_length: int):
    return some.tuples(*(some.integers(min_length, max_length) for _ in range(ndim)))


def some_tensor(*shape: int, dtype: DTypeLike = np.float64, domain: tuple[float, float] = (-5.0, +5.0)):
    return hypothesis.extra.numpy.arrays(dtype, shape, elements=some.floats(*domain))


def some_multi_index(*shape: int):
    return some.tuples(*(some.integers(0, length - 1) for length in shape))


@pytest.mark.parametrize("activation_type", [nn.Sigmoid, nn.ReLU, nn.Tanh, nn.GELU])
@given(some_shape(ndim=2, min_length=1, max_length=8), some.data())
def test_activation_grad_x_correctness(
    activation_type: type[nn.Activation],
    shape: tuple[int, int],
    data: some.DataObject,
):
    f = activation_type()
    i = data.draw(some_multi_index(*shape), "Input multi-index")
    j = data.draw(some_multi_index(*shape), "Output multi-index")
    x = data.draw(some_tensor(*shape), "Input tensor")

    # Skip gradient check at ReLU non-differentiable point
    if activation_type is nn.ReLU:
        assume(not np.isclose(x[i], 0.0))

    grad_bp = compute_grad_x_bp(deepcopy(f), x, i, j)
    grad_fd = compute_grad_x_fd(deepcopy(f), x, i, j)

    assert np.isclose(grad_fd, grad_bp)


@given(
    some.integers(1, 8),
    some.integers(1, 8),
    some.integers(1, 8),
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
    f = nn.Linear(in_features, out_features, init_method, dtype=np.float64)
    i = data.draw(some_multi_index(batch_size, in_features), "Input multi-index")
    j = data.draw(some_multi_index(batch_size, out_features), "Output multi-index")
    x = data.draw(some_tensor(batch_size, in_features), "Input tensor")

    grad_bp = compute_grad_x_bp(deepcopy(f), x, i, j)
    grad_fd = compute_grad_x_fd(deepcopy(f), x, i, j)

    assert np.isclose(grad_fd, grad_bp)


@given(
    some.integers(1, 8),
    some.integers(1, 8),
    some.integers(1, 8),
    some.one_of(some.just("He"), some.just("Xavier")),
    some.data(),
)
def test_linear_grad_params_correctness(
    batch_size: int,
    in_features: int,
    out_features: int,
    init_method: Literal["He", "Xavier"],
    data: some.DataObject,
):
    f = nn.Linear(in_features, out_features, init_method, dtype=np.float64)
    idxs = [
        data.draw(some_multi_index(in_features, out_features), "Weight multi-index"),
        data.draw(some_multi_index(out_features), "Bias multi-index"),
    ]
    j = data.draw(some_multi_index(batch_size, out_features), "Output multi-index")
    x = data.draw(some_tensor(batch_size, in_features), "Input tensor")

    grads_bp = compute_grad_params_bp(deepcopy(f), x, idxs, j)
    grads_fd = compute_grad_params_fd(deepcopy(f), x, idxs, j)

    assert np.allclose(grads_fd, grads_bp)


@given(
    some_shape(ndim=4, min_length=1, max_length=4),
    some.integers(1, 4),
    some.tuples(some.integers(1, 4), some.integers(1, 4)),
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
    B, in_channels, H, W = in_shape
    H_out = int(1 + (H + 2 * padding[0] - kernel_size[0]) / strides[0])
    W_out = int(1 + (W + 2 * padding[1] - kernel_size[1]) / strides[1])

    assume(H_out >= 1 and W_out >= 1)

    f = nn.Conv2D(
        in_channels,
        out_channels,
        kernel_size,
        strides,
        padding,
        init_method,
        dtype=np.float64,
    )
    i = data.draw(some_multi_index(*in_shape), "Input multi-index")
    j = data.draw(some_multi_index(B, out_channels, H_out, W_out), "Output multi-index")
    x = data.draw(some_tensor(*in_shape), "Input image")

    grad_bp = compute_grad_x_bp(deepcopy(f), x, i, j)
    grad_fd = compute_grad_x_fd(deepcopy(f), x, i, j)

    assert np.isclose(grad_fd, grad_bp)


@given(
    some_shape(ndim=4, min_length=1, max_length=4),
    some.integers(1, 4),
    some.tuples(some.integers(1, 4), some.integers(1, 4)),
    some.tuples(some.integers(1, 2), some.integers(1, 2)),
    some.tuples(some.integers(0, 2), some.integers(0, 2)),
    some.one_of(some.just("He"), some.just("Xavier")),
    some.data(),
)
def test_conv2d_grad_params_correctness(
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

    assume(H_out >= 1 and W_out >= 1)

    f = nn.Conv2D(
        in_channels,
        out_channels,
        kernel_size,
        strides,
        padding,
        init_method,
        dtype=np.float64,
    )
    idxs = [
        data.draw(
            some_multi_index(out_channels, in_channels * kernel_size[0] * kernel_size[1]),
            "Kernel multi-index",
        ),
        data.draw(some_multi_index(out_channels, 1), "Bias multi-index"),
    ]
    j = data.draw(some_multi_index(B, out_channels, H_out, W_out), "Output multi-index")
    x = data.draw(some_tensor(*in_shape), "Input image")

    grads_bp = compute_grad_params_bp(deepcopy(f), x, idxs, j)
    grads_fd = compute_grad_params_fd(deepcopy(f), x, idxs, j)

    assert np.allclose(grads_fd, grads_bp)


@given(
    some.integers(1, 8),
    some.lists(some.integers(1, 8), min_size=1, max_size=8),
    some.data(),
)
def test_sequential_grad_flow_correctness(batch_size: int, dims: list[int], data: some.DataObject):
    layers = []
    for dim_a, dim_b in pairwise(dims):
        layers.append(nn.Linear(dim_a, dim_b, init_method="Xavier", dtype=np.float64))
        layers.append(nn.Sigmoid())

    f = nn.Sequential(*layers)
    i = data.draw(some_multi_index(batch_size, dims[0]), "Input multi-index")
    j = data.draw(some_multi_index(batch_size, dims[-1]), "Output multi-index")
    x = data.draw(some_tensor(batch_size, dims[0]), "Input tensor")

    grad_bp = compute_grad_x_bp(deepcopy(f), x, i, j)
    grad_fd = compute_grad_x_fd(deepcopy(f), x, i, j)

    logging.info("Sequential ∂Yj/∂Xi | %+8.6f | %+8.6f", grad_bp, grad_fd)

    assert np.isclose(grad_fd, grad_bp)


@given(
    some.integers(1, 8),
    some.integers(1, 8),
    some.data(),
)
def test_residual_grad_flow_correctness(batch_size: int, dim: int, data: some.DataObject):
    layer = nn.Sequential(
        nn.Linear(dim, dim, dtype=np.float64),
        nn.Sigmoid(),
        nn.Linear(dim, dim, dtype=np.float64),
    )
    f = nn.Residual(layer)
    i = data.draw(some_multi_index(batch_size, dim), "Input multi-index")
    j = data.draw(some_multi_index(batch_size, dim), "Output multi-index")
    x = data.draw(some_tensor(batch_size, dim), "Input tensor")

    grad_bp = compute_grad_x_bp(deepcopy(f), x, i, j)
    grad_fd = compute_grad_x_fd(deepcopy(f), x, i, j)

    assert np.isclose(grad_fd, grad_bp)
