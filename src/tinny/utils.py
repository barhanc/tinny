# pylint: disable=missing-function-docstring, missing-class-docstring, missing-module-docstring
# pylint: disable=invalid-name
from tinny import xp
from tinny import NDArray, FloatType, IntpType, DTypeLike


def chunks(arr: NDArray, size: int):
    for i in range(0, len(arr), size):
        yield arr[i : i + size]


def onehot(y: NDArray[IntpType], dtype: DTypeLike = xp.float32) -> NDArray[FloatType]:
    one_hot = xp.zeros((y.shape[0], xp.max(y) + 1))
    one_hot[xp.arange(y.shape[0]), y] = 1
    return one_hot.astype(dtype)


def sigmoid(x: NDArray[FloatType], sample: bool = False) -> NDArray[FloatType]:
    σ = 1.0 / (1.0 + xp.exp(-x))
    if sample:
        return (σ > xp.random.rand(*σ.shape)).astype(dtype=x.dtype)
    return σ


def relu(x: NDArray[FloatType], sample: bool = False) -> NDArray[FloatType]:
    if sample:
        return xp.maximum(0, x + xp.sqrt(sigmoid(x)) * xp.random.randn(*x.shape))
    return xp.maximum(0, x)


def softmax(x: NDArray[FloatType]) -> NDArray[FloatType]:
    m = x.max(axis=1, keepdims=True)
    y: NDArray[FloatType] = xp.exp(x - m)
    return y / y.sum(axis=1, keepdims=True)
