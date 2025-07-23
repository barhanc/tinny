from itertools import chain
from abc import abstractmethod, ABC
from typing import Optional, Literal

import numpy as np

from numpy.typing import NDArray
from ..utils import zeros, rand


class Layer(ABC):
    """
    Interface for any differentiable parametrized tensor function with parameters `θ` that takes a
    single tensor `x` and returns a single tensor `y = Layer(x; θ)`.
    """

    # Reference to the inputs of the layer
    x: Optional[NDArray]
    # Reference to the outputs of the layer
    y: Optional[NDArray]

    def reset(self) -> None:
        """(Re)Initialize the layer."""
        self.x = None
        self.y = None

    def parameters(self) -> list[NDArray]:
        """Return a list of references to the parameters of the layer."""
        return []

    def gradients(self) -> list[Optional[NDArray]]:
        """
        Return a list of references to the gradients ∂Loss/∂θ of the loss function w.r.t. the
        parameters, in the same order as the `.parameters()` method.
        """
        return []

    @abstractmethod
    def forward(self, x: NDArray, training: bool) -> NDArray:
        """Propagate the input `x` forward through the layer and save the output in `self.y`."""
        raise NotImplementedError

    @abstractmethod
    def backward(self, grad_y: NDArray) -> NDArray:
        """
        Given ∂Loss/∂y (`grad_y`):
            * compute ∂Loss/∂x and ∂Loss/∂θ (where θ are the layer's params);
            * return ∂Loss/∂x;

        NOTE: We assume that the layers are connected in a simple path (i.e. the computation graph
        is linear) and thus we don't have to keep and accumulate the gradients ∂Loss/∂y in the layer
        itself, but can instead just dynamically pass ∂Loss/∂y while traversing this linear
        computation graph.
        """
        raise NotImplementedError


class Sequential(Layer):
    def __init__(self, *layers: Layer):
        self.layers: tuple[Layer, ...] = layers
        self.x: Optional[NDArray] = None
        self.y: Optional[NDArray] = None
        self.reset()

    def reset(self):
        self.x = None
        self.y = None
        for layer in self.layers:
            layer.reset()

    def parameters(self) -> list[NDArray]:
        return list(chain(*(layer.parameters() for layer in self.layers)))

    def gradients(self) -> list[Optional[NDArray]]:
        return list(chain(*(layer.gradients() for layer in self.layers)))

    def forward(self, x: NDArray, training: bool) -> NDArray:
        self.x = x
        for layer in self.layers:
            x = layer.forward(x, training)
        self.y = x
        return self.y

    def backward(self, grad_y: NDArray) -> NDArray:
        for i in reversed(range(len(self.layers))):
            # Propagate the ∂Loss/∂y backward through layer `i`
            grad_y = self.layers[i].backward(grad_y)
        return grad_y


class Sigmoid(Layer):
    def __init__(self):
        self.x: Optional[NDArray] = None
        self.y: Optional[NDArray] = None
        self.reset()

    def forward(self, x: NDArray, training: bool) -> NDArray:
        self.x = x
        self.y = 1.0 / (1.0 + np.exp(-x))
        return self.y

    def backward(self, grad_y: NDArray) -> NDArray:
        return grad_y * (self.y * (1.0 - self.y))


class ReLU(Layer):
    def __init__(self):
        self.x: Optional[NDArray] = None
        self.y: Optional[NDArray] = None
        self.reset()

    def forward(self, x: NDArray, training: bool) -> NDArray:
        self.x = x
        self.y = np.maximum(0, x)
        return self.y

    def backward(self, grad_y: NDArray) -> NDArray:
        return grad_y * (self.x > 0).astype(np.float32)


class Tanh(Layer):
    def __init__(self):
        self.x: Optional[NDArray] = None
        self.y: Optional[NDArray] = None
        self.reset()

    def forward(self, x: NDArray, training: bool) -> NDArray:
        self.x = x
        self.y = np.tanh(x)
        return self.y

    def backward(self, grad_y: NDArray) -> NDArray:
        return grad_y * (1 - self.y**2)


class Dropout(Layer):
    def __init__(self, p: float):
        assert 0 < p <= 1
        self.x: Optional[NDArray] = None
        self.y: Optional[NDArray] = None

        self.p: float = p
        self.mask: Optional[NDArray] = None

        self.reset()

    def reset(self):
        self.x = None
        self.y = None
        self.mask = None

    def forward(self, x: NDArray, training: bool) -> NDArray:
        self.x = x
        if training:
            # Save the dropout mask for backward pass
            self.mask = (rand(*x.shape) > self.p).astype(np.float32)
            self.y = x * self.mask
        else:
            self.y = x * (1 - self.p)
        return self.y

    def backward(self, grad_y: NDArray) -> NDArray:
        grad_x = grad_y * self.mask
        return grad_x


class Flatten(Layer):
    def __init__(self, start_dim: int = 1):
        self.start_dim: int = start_dim
        self.x: Optional[NDArray] = None
        self.y: Optional[NDArray] = None
        self.reset()

    def forward(self, x: NDArray, training: bool) -> NDArray:
        self.x = x
        self.y = x.reshape(*x.shape[: self.start_dim], -1)
        return self.y

    def backward(self, grad_y: NDArray) -> NDArray:
        grad_x = grad_y.reshape(self.x.shape)
        return grad_x


class Linear(Layer):
    def __init__(self, vsize: int, hsize: int, init_method: Literal["Xavier", "He"]):
        self.vsize: int = vsize
        self.hsize: int = hsize
        self.init_method: Literal["Xavier", "He"] = init_method

        self.x: Optional[NDArray] = None
        self.y: Optional[NDArray] = None

        # Parameters and gradients
        self.w: Optional[NDArray] = None
        self.b: Optional[NDArray] = None
        self.grad_w: Optional[NDArray] = None
        self.grad_b: Optional[NDArray] = None

        self.reset()

    def reset(self):
        self.x = None
        self.y = None

        # Weights initialization
        match self.init_method:
            case "Xavier":
                scale = np.sqrt(6 / (self.vsize + self.hsize))
                self.w = np.random.uniform(-scale, +scale, size=(self.vsize, self.hsize)).astype(np.float32)
            case "He":
                scale = np.sqrt(4 / (self.vsize + self.hsize))
                self.w = np.random.normal(0, scale, size=(self.vsize, self.hsize)).astype(np.float32)
            case _:
                raise ValueError(f"Unrecognised {self.init_method=}")

        # Bias initialization
        self.b = zeros(self.hsize)

        self.grad_w = None
        self.grad_b = None

    def parameters(self) -> list[NDArray]:
        return [self.w, self.b]

    def gradients(self) -> list[Optional[NDArray]]:
        return [self.grad_w, self.grad_b]

    def forward(self, x: NDArray, training: bool) -> NDArray:
        self.x = x
        self.y = self.b + x @ self.w
        return self.y

    def backward(self, grad_y: NDArray) -> NDArray:
        # Compute ∂Loss/∂x
        grad_x = grad_y @ self.w.T

        # Compute  ∂Loss/∂w and ∂Loss/∂b
        self.grad_w = self.x.T @ grad_y
        self.grad_b = grad_y.sum(axis=0)

        # Propagate ∂Loss/∂x backward
        return grad_x


class Conv2D(Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        strides: int | tuple[int, int],
        padding: int | tuple[int, int],
        init_method: Literal["Xavier", "He"],
    ):
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.kernel_size: tuple[int, int] = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.strides: tuple[int, int] = (strides, strides) if isinstance(strides, int) else strides
        self.padding: tuple[int, int] = (padding, padding) if isinstance(padding, int) else padding

        self.init_method: Literal["Xavier", "He"] = init_method

        self.x: Optional[NDArray] = None
        self.y: Optional[NDArray] = None

        # Parameters and gradients
        self.w: Optional[NDArray] = None
        self.b: Optional[NDArray] = None
        self.grad_w: Optional[NDArray] = None
        self.grad_b: Optional[NDArray] = None

        # Cached indices for the im2col transformation
        self.indices: Optional[NDArray] = None
        # Cached input after padding and im2col transformation
        self.xcol: Optional[NDArray] = None
        # Shape of the last propagated tensor
        self.dims: Optional[tuple[int, int, int, int]] = None

        self.reset()

    def reset(self):
        self.x = None
        self.y = None

        # Weights initialization
        nrow = self.out_channels
        ncol = self.in_channels * self.kernel_size[0] * self.kernel_size[1]

        match self.init_method:
            case "Xavier":
                scale = np.sqrt(6 / (nrow + ncol))
                self.w = np.random.uniform(-scale, +scale, size=(nrow, ncol)).astype(np.float32)
            case "He":
                scale = np.sqrt(4 / (nrow + ncol))
                self.w = np.random.normal(0, scale, size=(nrow, ncol)).astype(np.float32)
            case _:
                raise ValueError(f"Unrecognised {self.init_method=}")

        # Bias initialization
        eps = 0.01  # Initialize biases to small positive values (for ReLU)
        self.b = zeros(self.out_channels, 1) + eps

        self.grad_w = None
        self.grad_b = None

        self.xcol = None
        self.dims = None
        self.indices = None

    def parameters(self) -> list[NDArray]:
        return [self.w, self.b]

    def gradients(self) -> list[Optional[NDArray]]:
        return [self.grad_w, self.grad_b]

    def _pad(self, x: NDArray) -> NDArray:
        assert len(x.shape) == 4
        pad_h, pad_w = self.padding
        return np.pad(x, pad_width=[(0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)])

    def _unpad(self, x: NDArray) -> NDArray:
        assert len(x.shape) == 4
        *_, H, W = x.shape
        pad_h, pad_w = self.padding
        return x[:, :, pad_h : H - pad_h, pad_w : W - pad_w]

    def forward(self, x: NDArray, training: bool) -> NDArray:
        assert len(x.shape) == 4
        assert x.shape[1] == self.in_channels

        self.x = x

        B, C_in, H_in, W_in = x.shape
        H_out = int(1 + (H_in + 2 * self.padding[0] - self.kernel_size[0]) / self.strides[0])
        W_out = int(1 + (W_in + 2 * self.padding[1] - self.kernel_size[1]) / self.strides[1])

        if self.dims != x.shape:
            use_cached_indices = False
            self.dims = x.shape
        else:
            use_cached_indices = True

        # Apply padding transformation
        x = self._pad(x)  # (B, C_in, H_in + 2*pad_h, W_in + 2*pad_w)

        # Compute indices for im2col transformation
        if not use_cached_indices:
            idx_c, idx_h_ker, idx_w_ker = np.indices((C_in, *self.kernel_size)).reshape(3, -1)
            idx_h_out, idx_w_out = np.indices((H_out, W_out)).reshape(2, -1)

            idx_b = np.arange(B).reshape(-1, 1, 1)
            idx_c = idx_c.reshape(-1, 1)
            idx_h = idx_h_ker.reshape(-1, 1) + self.strides[0] * idx_h_out
            idx_w = idx_w_ker.reshape(-1, 1) + self.strides[1] * idx_w_out

            multi_index = (idx_b, idx_c, idx_h, idx_w)
            self.indices = np.ravel_multi_index(multi_index, dims=x.shape)  # (B, C_in * H_ker * W_ker, H_out * W_out)
            self.indices = self.indices.transpose(1, 0, 2)  # (C_in * H_ker * W_ker, B, H_out * W_out)
            self.indices = self.indices.reshape(-1, B * H_out * W_out)  # (C_in * H_ker * W_ker, B * H_out * W_out)

        # Apply im2col transformation
        x = self.xcol = np.take(x, self.indices)  # (C_in * H_ker * W_ker, B * H_out * W_out)

        # Apply affine transformation
        x = self.b + self.w @ x  # (C_out, B * H_out * W_ou)

        # Reshape
        x = x.reshape(-1, B, H_out, W_out)  # (C_out, B, H_out * W_out)
        x = x.transpose(1, 0, 2, 3)  # (B, C_out, H_out, W_out)
        self.y = x

        return self.y

    def backward(self, grad_y: NDArray) -> NDArray:
        assert len(grad_y.shape) == 4
        assert len(self.x.shape) == 4
        assert self.x.shape[1] == self.in_channels
        assert grad_y.shape[1] == self.out_channels

        B, C_in, H_in, W_in = self.x.shape
        H_out = int(1 + (H_in + 2 * self.padding[0] - self.kernel_size[0]) / self.strides[0])
        W_out = int(1 + (W_in + 2 * self.padding[1] - self.kernel_size[1]) / self.strides[1])

        # --- Compute ∂Loss/∂x, ∂Loss/∂w and ∂Loss/∂b

        # Backpropagate through reshape operations
        grad_y = grad_y.transpose(1, 0, 2, 3)  # (C_out, B, H_out, W_out)
        grad_y = grad_y.reshape(-1, B * H_out * W_out)  # (C_out, B * H_out * W_out)

        # Backpropagate through affine transformation
        x = self.xcol  # (C_in * H_ker * W_ker, B * H_out * W_out)

        self.grad_w = grad_y @ x.T
        self.grad_b = grad_y.sum(axis=1, keepdims=True)

        grad_y = self.w.T @ grad_y  # (C_in * H_ker * W_ker, B * H_out * W_out)

        # Backpropagate through im2col operation
        dims = (B, C_in, H_in + 2 * self.padding[0], W_in + 2 * self.padding[1])
        grad_x = zeros(*dims).reshape(-1)

        for idx, val in zip(self.indices, grad_y):
            grad_x[idx] += val

        grad_x = grad_x.reshape(dims)

        # Backpropagate through padding operation
        grad_x = self._unpad(grad_x)

        # Propagate ∂Loss/∂x backward
        return grad_x


__all__ = [
    "Layer",
    "Sequential",
    "Sigmoid",
    "ReLU",
    "Tanh",
    "Dropout",
    "Flatten",
    "Linear",
    "Conv2D",
]
