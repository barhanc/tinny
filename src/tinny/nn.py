# pylint: disable=missing-function-docstring, missing-class-docstring, missing-module-docstring
# pylint: disable=invalid-name

import os

from itertools import chain
from abc import abstractmethod, ABC
from typing import Optional, Literal

import numpy as np

from numpy.typing import NDArray, DTypeLike

type FloatType = np.float32 | np.float64
type BoolType = np.bool_
type IntpType = np.intp

# NOTE: This is a hack to allow GPU execution of NumPy code
# fmt:off
xp = np
if os.environ.get("GPU") == "1":
    try:
        import cupy as cp
        xp = cp
    except ImportError:
        print("Cupy is not available!")
# fmt:on


class Layer(ABC):
    """
    Interface for any differentiable parametrized NDArray function with
    parameters `θ` that takes a single NDArray `x` and returns a single NDArray `y
    = Layer(x; θ)`.
    """

    x: Optional[NDArray[FloatType]]  # Reference to the inputs of the layer
    y: Optional[NDArray[FloatType]]  # Reference to the outputs of the layer

    @abstractmethod
    def reset(self) -> None:
        """Initialize the layer."""
        raise NotImplementedError

    @abstractmethod
    def parameters(self) -> list[NDArray[FloatType]]:
        """Return a list of references to the parameters of the layer."""
        raise NotImplementedError

    @abstractmethod
    def gradients(self) -> list[Optional[NDArray[FloatType]]]:
        """
        Return a list of references to the gradients ∂Loss/∂θ of the loss
        function w.r.t. the parameters, in the same order as the `.parameters()`
        method.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, x: NDArray[FloatType], training: bool) -> NDArray[FloatType]:
        """
        Propagate the input `x` forward through the layer and return the output.
        Save the references to the input and output respectively in `self.x` and
        `self.y`.
        """
        raise NotImplementedError

    @abstractmethod
    def backward(self, grad_y: NDArray[FloatType]) -> NDArray[FloatType]:
        """
        Given ∂Loss/∂y (`grad_y`):
            * compute ∂Loss/∂x and ∂Loss/∂θ (where θ are the layer's params);
            * return ∂Loss/∂x;

        NOTE: We assume that the layers are connected in a simple path (i.e. the
        computation graph is linear) and thus we don't have to keep and
        accumulate the gradients ∂Loss/∂y in the layer itself, but can instead
        just dynamically pass ∂Loss/∂y while traversing this linear computation
        graph.
        """
        raise NotImplementedError


class Sequential(Layer):
    def __init__(self, *layers: Layer):
        self.layers: tuple[Layer, ...] = layers
        self.x: Optional[NDArray[FloatType]] = None
        self.y: Optional[NDArray[FloatType]] = None
        self.reset()

    def reset(self):
        self.x = None
        self.y = None
        for layer in self.layers:
            layer.reset()

    def parameters(self) -> list[NDArray[FloatType]]:
        return list(chain(*(layer.parameters() for layer in self.layers)))

    def gradients(self) -> list[Optional[NDArray[FloatType]]]:
        return list(chain(*(layer.gradients() for layer in self.layers)))

    def forward(self, x: NDArray[FloatType], training: bool) -> NDArray[FloatType]:
        self.x = x
        for layer in self.layers:
            x = layer.forward(x, training)
        self.y = x
        return self.y

    def backward(self, grad_y: NDArray[FloatType]) -> NDArray[FloatType]:
        for layer in reversed(self.layers):
            grad_y = layer.backward(grad_y)
        return grad_y


class Residual(Layer):
    def __init__(self, layer: Layer):
        self.layer: Layer = layer
        self.x: Optional[NDArray[FloatType]] = None
        self.y: Optional[NDArray[FloatType]] = None
        self.reset()

    def reset(self) -> None:
        self.x = None
        self.y = None
        self.layer.reset()

    def parameters(self) -> list[NDArray[FloatType]]:
        return self.layer.parameters()

    def gradients(self) -> list[Optional[NDArray[FloatType]]]:
        return self.layer.gradients()

    def forward(self, x: NDArray[FloatType], training: bool) -> NDArray[FloatType]:
        self.x = x
        self.y = self.layer.forward(x, training) + x
        return self.y

    def backward(self, grad_y: NDArray[FloatType]) -> NDArray[FloatType]:
        return self.layer.backward(grad_y) + grad_y


class Flatten(Layer):
    def __init__(self, start_dim: int = 1):
        self.start_dim: int = start_dim
        self.x: Optional[NDArray[FloatType]] = None
        self.y: Optional[NDArray[FloatType]] = None

    def reset(self):
        self.x = None
        self.y = None

    def parameters(self) -> list[NDArray[FloatType]]:
        return []

    def gradients(self) -> list[Optional[NDArray[FloatType]]]:
        return []

    def forward(self, x: NDArray[FloatType], training: bool) -> NDArray[FloatType]:
        self.x = x
        self.y = x.reshape(*x.shape[: self.start_dim], -1)
        return self.y

    def backward(self, grad_y: NDArray[FloatType]) -> NDArray[FloatType]:
        assert self.x is not None
        return grad_y.reshape(self.x.shape)


class Activation(Layer):
    def reset(self):
        self.x = None
        self.y = None

    def parameters(self) -> list[NDArray[FloatType]]:
        return []

    def gradients(self) -> list[Optional[NDArray[FloatType]]]:
        return []


class Sigmoid(Activation):
    def __init__(self):
        self.x: Optional[NDArray[FloatType]] = None
        self.y: Optional[NDArray[FloatType]] = None

    def forward(self, x: NDArray[FloatType], training: bool) -> NDArray[FloatType]:
        self.x = x
        self.y = 1.0 / (1.0 + xp.exp(-x))
        return self.y

    def backward(self, grad_y: NDArray[FloatType]) -> NDArray[FloatType]:
        assert self.y is not None
        return grad_y * (self.y * (1.0 - self.y))


class Tanh(Activation):
    def __init__(self):
        self.x: Optional[NDArray[FloatType]] = None
        self.y: Optional[NDArray[FloatType]] = None

    def forward(self, x: NDArray[FloatType], training: bool) -> NDArray[FloatType]:
        self.x = x
        self.y = xp.tanh(x)
        return self.y

    def backward(self, grad_y: NDArray[FloatType]) -> NDArray[FloatType]:
        assert self.y is not None
        return grad_y * (1 - self.y**2)


class ReLU(Activation):
    def __init__(self):
        self.x: Optional[NDArray[FloatType]] = None
        self.y: Optional[NDArray[FloatType]] = None

    def forward(self, x: NDArray[FloatType], training: bool) -> NDArray[FloatType]:
        self.x = x
        self.y = xp.maximum(0, x)
        return self.y

    def backward(self, grad_y: NDArray[FloatType]) -> NDArray[FloatType]:
        assert self.y is not None
        return grad_y * (self.y > 0)


class GELU(Activation):
    def __init__(self):
        self.x: Optional[NDArray[FloatType]] = None
        self.y: Optional[NDArray[FloatType]] = None

    def forward(self, x: NDArray[FloatType], training: bool) -> NDArray[FloatType]:
        self.x = x
        self.y = 0.5 * x * (1 + xp.tanh(0.8 * x))
        return self.y

    def backward(self, grad_y: NDArray[FloatType]) -> NDArray[FloatType]:
        assert self.y is not None
        assert self.x is not None
        a = 0.8
        return grad_y * ((1.0 + xp.tanh(a * self.x)) * (0.5 + a * (self.x - self.y)))


class Dropout(Layer):
    def __init__(self, p: float):
        assert 0 < p <= 1
        self.x: Optional[NDArray[FloatType]] = None
        self.y: Optional[NDArray[FloatType]] = None
        self.p: float = p
        self.mask: Optional[NDArray[BoolType]] = None

    def reset(self):
        self.x = None
        self.y = None
        self.mask = None

    def parameters(self) -> list[NDArray[FloatType]]:
        return []

    def gradients(self) -> list[Optional[NDArray[FloatType]]]:
        return []

    def forward(self, x: NDArray[FloatType], training: bool) -> NDArray[FloatType]:
        self.x = x
        if training:
            # Save the dropout mask for backward pass
            self.mask = xp.random.rand(*x.shape) > self.p
            self.y = x * self.mask
        else:
            self.y = x * (1 - self.p)
        return self.y

    def backward(self, grad_y: NDArray[FloatType]) -> NDArray[FloatType]:
        assert self.mask is not None
        return grad_y * self.mask


class Linear(Layer):
    def __init__(
        self,
        vsize: int,
        hsize: int,
        init_method: Literal["Xavier", "He"] = "He",
        dtype: DTypeLike = xp.float32,
    ):
        self.vsize: int = vsize
        self.hsize: int = hsize
        self.init_method: Literal["Xavier", "He"] = init_method
        self.dtype: DTypeLike = dtype

        # Input and output NDArrays
        self.x: Optional[NDArray[FloatType]]
        self.y: Optional[NDArray[FloatType]]

        # Parameters and gradients
        self.w: NDArray[FloatType]
        self.b: NDArray[FloatType]
        self.grad_w: Optional[NDArray[FloatType]]
        self.grad_b: Optional[NDArray[FloatType]]

        self.reset()

    def reset(self):
        # Input and output reset
        self.x = None
        self.y = None

        # Weights initialization
        match self.init_method:
            case "Xavier":
                scale = xp.sqrt(6 / (self.vsize + self.hsize))
                self.w = xp.random.uniform(-scale, +scale, size=(self.vsize, self.hsize)).astype(self.dtype)
            case "He":
                scale = xp.sqrt(4 / (self.vsize + self.hsize))
                self.w = xp.random.normal(0, scale, size=(self.vsize, self.hsize)).astype(self.dtype)
            case _:
                raise ValueError(f"Unrecognised {self.init_method=}")

        # Bias initialization
        self.b = xp.zeros(shape=(self.hsize,), dtype=self.dtype)

        # Gradients initialization
        self.grad_w = None
        self.grad_b = None

    def parameters(self) -> list[NDArray[FloatType]]:
        return [self.w, self.b]

    def gradients(self) -> list[Optional[NDArray[FloatType]]]:
        return [self.grad_w, self.grad_b]

    def forward(self, x: NDArray[FloatType], training: bool) -> NDArray[FloatType]:
        self.x = x
        self.y = self.b + x @ self.w
        return self.y

    def backward(self, grad_y: NDArray[FloatType]) -> NDArray[FloatType]:
        assert self.x is not None
        self.grad_w = self.x.T @ grad_y
        self.grad_b = grad_y.sum(axis=0)
        grad_x = grad_y @ self.w.T
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
        dtype: DTypeLike = xp.float32,
    ):
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.kernel_size: tuple[int, int] = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.strides: tuple[int, int] = (strides, strides) if isinstance(strides, int) else strides
        self.padding: tuple[int, int] = (padding, padding) if isinstance(padding, int) else padding

        self.init_method: Literal["Xavier", "He"] = init_method
        self.dtype: DTypeLike = dtype

        # Input and output tensors
        self.x: Optional[NDArray[FloatType]]
        self.y: Optional[NDArray[FloatType]]

        # Parameters and gradients
        self.w: NDArray[FloatType]
        self.b: NDArray[FloatType]
        self.grad_w: Optional[NDArray[FloatType]]
        self.grad_b: Optional[NDArray[FloatType]]

        # Cached indices for the im2col transformation
        self.indices: Optional[NDArray[IntpType]]
        # Cached input after padding and im2col transformation
        self.xcol: Optional[NDArray[FloatType]]
        # Shape of the last propagated tensor after padding
        self.dims: Optional[tuple[int, int, int, int]]

        self.reset()

    def reset(self):
        # Input and output reset
        self.x = None
        self.y = None

        # Weights initialization
        nrow = self.out_channels
        ncol = self.in_channels * self.kernel_size[0] * self.kernel_size[1]

        match self.init_method:
            case "Xavier":
                scale = xp.sqrt(6 / (nrow + ncol))
                self.w = xp.random.uniform(-scale, +scale, size=(nrow, ncol)).astype(self.dtype)
            case "He":
                scale = xp.sqrt(4 / (nrow + ncol))
                self.w = xp.random.normal(0, scale, size=(nrow, ncol)).astype(self.dtype)
            case _:
                raise ValueError(f"Unrecognised {self.init_method=}")

        # Bias initialization
        eps = 0.01  # Initialize biases to small positive values (for ReLU)
        self.b = xp.zeros(shape=(self.out_channels, 1), dtype=self.dtype) + eps

        # Gradients initialization
        self.grad_w = None
        self.grad_b = None

        # Other initialization
        self.xcol = None
        self.dims = None
        self.indices = None

    def parameters(self) -> list[NDArray[FloatType]]:
        return [self.w, self.b]

    def gradients(self) -> list[Optional[NDArray[FloatType]]]:
        return [self.grad_w, self.grad_b]

    def _pad(self, x: NDArray[FloatType]) -> NDArray[FloatType]:
        assert len(x.shape) == 4
        pad_h, pad_w = self.padding
        return xp.pad(x, pad_width=[(0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)])

    def _unpad(self, x: NDArray[FloatType]) -> NDArray[FloatType]:
        assert len(x.shape) == 4
        *_, H, W = x.shape  # Math notation, so pylint: disable=invalid-name
        pad_h, pad_w = self.padding
        return x[:, :, pad_h : H - pad_h, pad_w : W - pad_w]

    def forward(self, x: NDArray[FloatType], training: bool) -> NDArray[FloatType]:
        assert len(x.shape) == 4
        assert x.shape[1] == self.in_channels

        # --- Save the reference to the input ---
        self.x = x

        # --- Compute size of the output tensor ---
        B, C_in, H_in, W_in = self.x.shape
        H_out = int(1 + (H_in + 2 * self.padding[0] - self.kernel_size[0]) / self.strides[0])
        W_out = int(1 + (W_in + 2 * self.padding[1] - self.kernel_size[1]) / self.strides[1])

        # --- Apply padding transformation ---
        # Shape (B, C_in, H_in + 2*pad_h, W_in + 2*pad_w)
        x = self._pad(x)

        # --- Compute indices for im2col transformation ---
        if self.dims != x.shape:
            self.dims = x.shape

            idx_c, idx_h_ker, idx_w_ker = xp.indices((C_in, *self.kernel_size)).reshape(3, -1)
            idx_h_out, idx_w_out = xp.indices((H_out, W_out)).reshape(2, -1)

            idx_b = xp.arange(B).reshape(-1, 1, 1)
            idx_c = idx_c.reshape(-1, 1)
            idx_h = idx_h_ker.reshape(-1, 1) + self.strides[0] * idx_h_out
            idx_w = idx_w_ker.reshape(-1, 1) + self.strides[1] * idx_w_out

            multi_index = (idx_b, idx_c, idx_h, idx_w)
            # Shape (B, C_in * H_ker * W_ker, H_out * W_out)
            self.indices = xp.ravel_multi_index(multi_index, dims=x.shape)
            # Shape (C_in * H_ker * W_ker, B, H_out * W_out)
            self.indices = self.indices.transpose(1, 0, 2)
            # Shape (C_in * H_ker * W_ker, B * H_out * W_out)
            self.indices = self.indices.reshape(-1, B * H_out * W_out)

        # --- Apply im2col transformation ---
        assert self.indices is not None
        # Shape (C_in * H_ker * W_ker, B * H_out * W_out)
        x = self.xcol = x.take(self.indices)

        # --- Apply affine transformation ---
        # Shape (C_out, B * H_out * W_ou)
        x = self.b + self.w @ x

        # --- Reshape ---
        # Shape (C_out, B, H_out, W_out)
        x = x.reshape(-1, B, H_out, W_out)
        # Shape (B, C_out, H_out, W_out)
        x = x.transpose(1, 0, 2, 3)

        self.y = x
        return self.y

    def backward(self, grad_y: NDArray[FloatType]) -> NDArray[FloatType]:
        assert self.x is not None
        assert len(grad_y.shape) == 4
        assert len(self.x.shape) == 4
        assert self.x.shape[1] == self.in_channels
        assert grad_y.shape[1] == self.out_channels

        # --- Compute size of the output tensor ---
        B, C_in, H_in, W_in = self.x.shape
        H_out = int(1 + (H_in + 2 * self.padding[0] - self.kernel_size[0]) / self.strides[0])
        W_out = int(1 + (W_in + 2 * self.padding[1] - self.kernel_size[1]) / self.strides[1])

        # --- Backpropagate through reshape operations ---
        # Shape (C_out, B, H_out, W_out)
        grad_y = grad_y.transpose(1, 0, 2, 3)
        # Shape (C_out, B * H_out * W_out)
        grad_y = grad_y.reshape(-1, B * H_out * W_out)

        # --- Backpropagate through affine transformation ---
        assert self.xcol is not None

        # Shape (C_in * H_ker * W_ker, B * H_out * W_out)
        x = self.xcol
        self.grad_w = grad_y @ x.T
        self.grad_b = grad_y.sum(axis=1, keepdims=True)

        # Shape (C_in * H_ker * W_ker, B * H_out * W_out)
        grad_y = self.w.T @ grad_y

        # --- Backpropagate through im2col operation ---
        assert self.indices is not None

        dims = (B, C_in, H_in + 2 * self.padding[0], W_in + 2 * self.padding[1])
        grad_x = xp.zeros(dims, dtype=self.dtype).reshape(-1)

        for idx, val in zip(self.indices, grad_y):
            grad_x[idx] += val

        grad_x = grad_x.reshape(dims)

        # --- Backpropagate through padding operation ---
        grad_x = self._unpad(grad_x)

        # --- Propagate ∂Loss/∂x backward ---
        return grad_x


# =================================================================================================
# =================================================================================================


class Optimizer(ABC):
    # References to the parameters of the model
    parameters: list[NDArray]

    @abstractmethod
    def __init__(self, parameters: list[NDArray], *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def apply(self, gradients: list[NDArray]) -> None:
        """
        Given the list of gradfients ∂Loss/∂θ of the loss function w.r.t. the parameters in the
        same order as in the `self.parameters` list, apply the gradients and advance the
        optimizer.
        """
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(
        self,
        parameters: list[NDArray],
        lr: float,
        momentum: float,
        nesterov: bool = False,
        l2_penalty: Optional[float] = None,
        weight_limit: Optional[float] = None,
    ):
        self.lr: float = lr
        self.momentum: float = momentum
        self.nesterov: bool = nesterov
        self.l2_penalty: Optional[float] = l2_penalty
        self.weight_limit: Optional[float] = weight_limit

        self.parameters: list[NDArray] = parameters
        self.velocities: list[NDArray] = [xp.zeros(param.shape, param.dtype) for param in self.parameters]

    def apply(self, gradients: list[NDArray]):
        for p, v_p, grad_p in zip(self.parameters, self.velocities, gradients):
            # Apply L2 regularization
            if self.l2_penalty:
                grad_p += self.l2_penalty * p

            # Update velocities
            v_p *= self.momentum
            v_p -= self.lr * grad_p

            # Update parameters
            if self.nesterov:
                p += self.momentum * v_p - self.lr * grad_p
            else:
                p += v_p

            # Apply weight limit normalization
            if self.weight_limit:
                norm = xp.linalg.norm(p, ord=2, axis=0)
                mask = norm > self.weight_limit
                p *= mask * (self.weight_limit / norm) + (~mask) * 1.0


class Adam(Optimizer):
    def __init__(
        self,
        parameters: list[NDArray],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.lr: float = lr
        self.beta1: float = betas[0]
        self.beta2: float = betas[1]
        self.eps: float = eps
        self.t: int = 0

        self.parameters: list[NDArray] = parameters
        self.means: list[NDArray] = [xp.zeros(param.shape, param.dtype) for param in self.parameters]
        self.variances: list[NDArray] = [xp.zeros(param.shape, param.dtype) for param in self.parameters]

    def apply(self, gradients: list[NDArray]):
        # Update time step
        self.t += 1

        for p, m_p, v_p, grad_p in zip(self.parameters, self.means, self.variances, gradients):
            # Update means
            m_p *= self.beta1
            m_p += (1 - self.beta1) * grad_p

            # Update variances
            v_p *= self.beta2
            v_p += (1 - self.beta2) * grad_p**2

            # Compute unbiased estimators
            mhat_p = m_p / (1 - self.beta1**self.t)
            vhat_p = v_p / (1 - self.beta2**self.t)

            # Update parameters
            p -= self.lr * mhat_p / (self.eps + xp.sqrt(vhat_p))
