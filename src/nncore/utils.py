import numpy as np
import matplotlib.pyplot as plt

from numpy.typing import NDArray


def zeros(*dims: int) -> NDArray[np.float32]:
    """Return NDArray of the given shape and type float32 filled with 0s."""
    return np.zeros(shape=tuple(dims), dtype=np.float32)


def ones(*dims: int) -> NDArray[np.float32]:
    """Return NDArray of the given shape and type float32 filled with 1s."""
    return np.ones(shape=tuple(dims), dtype=np.float32)


def rand(*dims: int) -> NDArray[np.float32]:
    """Return a float32 array of given shape with samples from uniform [0, 1)."""
    return np.random.rand(*dims).astype(np.float32)


def randn(*dims: int) -> NDArray[np.float32]:
    """Return a float32 array of given shape with samples from a standard normal distribution."""
    return np.random.randn(*dims).astype(np.float32)


def chunks(arr: NDArray, size: int):
    """Yield successive chunks of the array with the given size."""
    for i in range(0, len(arr), size):
        yield arr[i : i + size]


def onehot(y: NDArray) -> NDArray[np.float32]:
    """Convert class labels to one-hot encoded format."""
    one_hot = zeros(y.shape[0], np.max(y) + 1)
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot.astype(np.float32)


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


def limit_weights(w: NDArray, limit: float) -> NDArray[np.float32]:
    """Apply the norm limit regularization to `w`."""
    if limit == 0:
        return w
    norm = np.linalg.norm(w, ord=2, axis=0)
    mask = norm > limit
    return (w * (mask * (limit / norm) + (~mask) * 1.0)).astype(np.float32)


def sigmoid(x: NDArray, sample: bool = False) -> NDArray[np.float32]:
    """
    Return the value of the sigmoid function.

    NOTE: If `sample=True` return a sample from the binomial distribution with parameter p = σ(x).
    """
    σ = 1.0 / (1.0 + np.exp(-x))  # Math notation, so pylint: disable=non-ascii-name
    if sample:
        return (σ > rand(*σ.shape)).astype(np.float32)
    return σ.astype(np.float32)


def relu(x: NDArray, sample: bool = False) -> NDArray[np.float32]:
    """
    Return the value of the ReLU function.

    NOTE: If `sample=True` return a sample from the noisy ReLU (N-ReLU) defined as `max(0, x + z *
    sqrt(σ(x)))` where `z` is a sample from standard normal distribution.
    """
    if sample:
        return np.maximum(0, x + np.sqrt(sigmoid(x)) * randn(*x.shape)).astype(np.float32)
    return np.maximum(0, x).astype(np.float32)


def softmax(x: NDArray) -> NDArray[np.float32]:
    """Return the value of the softmax function."""
    m = x.max(axis=1, keepdims=True)
    y: NDArray = np.exp(x - m)
    return (y / y.sum(axis=1, keepdims=True)).astype(np.float32)
