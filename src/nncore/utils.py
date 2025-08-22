# pylint: disable=missing-function-docstring, missing-class-docstring, missing-module-docstring

import numpy as np
import matplotlib.pyplot as plt

from numpy.typing import NDArray, DTypeLike


def zeros(*dims: int, dtype: DTypeLike = np.float32) -> NDArray:
    return np.zeros(shape=tuple(dims), dtype=dtype)


def ones(*dims: int, dtype: DTypeLike = np.float32) -> NDArray:
    return np.ones(shape=tuple(dims), dtype=dtype)


def rand(*dims: int, dtype: DTypeLike = np.float32) -> NDArray:
    return np.random.rand(*dims).astype(dtype)


def randn(*dims: int, dtype: DTypeLike = np.float32) -> NDArray:
    return np.random.randn(*dims).astype(dtype)


def chunks(arr: NDArray, size: int):
    for i in range(0, len(arr), size):
        yield arr[i : i + size]


def onehot(y: NDArray, dtype: DTypeLike = np.float32) -> NDArray:
    one_hot = zeros(y.shape[0], np.max(y) + 1)
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot.astype(dtype)


def tiles(imgs: NDArray):
    """Display 2D matrices as grayscale tiles from a 4D tensor (rows, cols, height, width)."""
    space = 2
    rows, cols, h, w = imgs.shape

    img_matrix = np.empty(shape=(rows * (h + space) - space, cols * (w + space) - space))
    img_matrix.fill(np.nan)

    for r in range(rows):
        for c in range(cols):
            x = r * (h + space)
            y = c * (w + space)
            _min = np.min(imgs[r, c])
            _max = np.max(imgs[r, c])
            img_matrix[x : x + h, y : y + w] = (imgs[r, c] - _min) / (_max - _min)

    plt.matshow(img_matrix, cmap="gray")
    plt.axis("off")
    plt.show()


def limit_weights(w: NDArray, limit: float) -> NDArray:
    """Apply the norm limit regularization to `w`."""
    if limit == 0:
        return w
    norm = np.linalg.norm(w, ord=2, axis=0)
    mask = norm > limit
    return w * (mask * (limit / norm) + (~mask) * 1.0)


def sigmoid(x: NDArray, sample: bool = False) -> NDArray:
    """
    If `sample=True` return a sample from the binomial distribution with parameter
    ```
        p = σ(x) = 1 / (1 + exp(-x))
    ```
    Otherwise return the value of the logistic sigmoid function
    ```
        σ(x) = 1 / (1 + exp(-x))
    ```
    """
    σ = 1.0 / (1.0 + np.exp(-x))  # Math notation, so pylint: disable=non-ascii-name
    if sample:
        return σ > rand(*σ.shape)
    return σ


def relu(x: NDArray, sample: bool = False) -> NDArray:
    """
    If `sample=True` return a sample from the noisy ReLU (N-ReLU) defined as
    ```
        N-ReLU(x) = max(0, x + z * sqrt(σ(x)))
    ```
    where `z` is a sample from standard normal distribution. Otherwise return the value of the ReLU
    function
    ```
        ReLU(x) = max(0, x)
    ```
    """
    if sample:
        return np.maximum(0, x + np.sqrt(sigmoid(x)) * randn(*x.shape))
    return np.maximum(0, x)


def softmax(x: NDArray) -> NDArray:
    """Return the value of the softmax function."""
    m = x.max(axis=1, keepdims=True)
    y: NDArray = np.exp(x - m)
    return y / y.sum(axis=1, keepdims=True)
