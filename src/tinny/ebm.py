# pylint: disable=missing-function-docstring, missing-class-docstring, missing-module-docstring
# pylint: disable=invalid-name
from itertools import pairwise
from typing import Optional, Literal, Callable

from tqdm import trange

from tinny import xp
from tinny import NDArray, FloatType, DTypeLike


def _limit_weights(w: NDArray[FloatType], limit: float) -> NDArray[FloatType]:
    if limit == 0:
        return w
    norm = xp.linalg.norm(w, ord=2, axis=0)
    mask = norm > limit
    return w * (mask * (limit / norm) + (~mask) * 1.0)


class RBM:
    def __init__(
        self,
        vsize: int,
        hsize: int,
        pc_size: Optional[int],
        v_activation: Callable[[NDArray[FloatType], bool], NDArray[FloatType]],
        h_activation: Callable[[NDArray[FloatType], bool], NDArray[FloatType]],
        lr: float,
        momentum: float,
        l1_penalty: Optional[float],
        l2_penalty: Optional[float],
        weight_limit: Optional[float],
        init_method: Literal["Xavier", "He"],
        dtype: DTypeLike = xp.float32,
    ):
        self.vsize: int = vsize
        self.hsize: int = hsize
        self.pc_size: Optional[int] = pc_size
        self.v_activation: Callable[[NDArray[FloatType], bool], NDArray[FloatType]] = v_activation
        self.h_activation: Callable[[NDArray[FloatType], bool], NDArray[FloatType]] = h_activation

        self.lr: float = lr
        self.momentum: float = momentum
        self.l1_penalty: Optional[float] = l1_penalty
        self.l2_penalty: Optional[float] = l2_penalty
        self.weight_limit: Optional[float] = weight_limit

        self.init_method: Literal["Xavier", "He"] = init_method
        self.dtype: DTypeLike = dtype
        self.reset()

    def reset(self):
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

        # Biases initialization
        self.b = xp.zeros(self.vsize, dtype=self.dtype)
        self.c = xp.zeros(self.hsize, dtype=self.dtype)

        # Momentum initialization
        self.m_w = xp.zeros((self.vsize, self.hsize), dtype=self.dtype)
        self.m_b = xp.zeros(self.vsize, dtype=self.dtype)
        self.m_c = xp.zeros(self.hsize, dtype=self.dtype)

        # Persistent chain initialization
        if self.pc_size:
            self.pc = xp.zeros((self.pc_size, self.hsize), dtype=self.dtype)

    def probas_v(self, h: NDArray[FloatType], sample: bool) -> NDArray[FloatType]:
        return self.v_activation(self.b + h @ self.w.T, sample)

    def probas_h(self, v: NDArray[FloatType], sample: bool) -> NDArray[FloatType]:
        return self.h_activation(self.c + v @ self.w, sample)

    def sample(self, v: NDArray[FloatType], steps: int, verbose: bool = False) -> NDArray[FloatType]:
        # Perform Gibbs sampling
        for k in trange(steps, desc="Sampling", disable=not verbose):
            h = self.probas_h(v, sample=True)
            v = self.probas_v(h, sample=k < steps - 1)
        return v


class DBN:
    def __init__(self, *rbms: RBM):
        for rbm1, rbm2 in pairwise(rbms):
            assert rbm1.hsize == rbm2.vsize

        self.layers: tuple[RBM, ...] = rbms
        self.reset()

    def reset(self):
        for rbm in self.layers:
            rbm.reset()

    def propagate_up(self, v: NDArray[FloatType], n_layers: int) -> NDArray[FloatType]:
        assert 0 <= n_layers < len(self.layers)
        for i in range(n_layers):
            v = self.layers[i].probas_h(v, sample=False)
        return v

    def propagate_dn(self, h: NDArray[FloatType], n_layers: int) -> NDArray[FloatType]:
        assert 0 <= n_layers < len(self.layers)
        for i in reversed(range(n_layers)):
            h = self.layers[i].probas_v(h, sample=False)
        return h

    def sample(self, v: NDArray[FloatType], steps: int, verbose: bool = False) -> NDArray[FloatType]:
        i = len(self.layers) - 1
        v = self.propagate_up(v, i)
        h = self.layers[i].sample(v, steps, verbose)
        v = self.propagate_dn(h, i)
        return v


def cdk(rbm: RBM, minibatch: NDArray[FloatType], k: int = 1):
    batch_size = minibatch.shape[0]
    v = minibatch

    # -------------------------
    # --- Compute gradients ---
    # -------------------------

    # --- Positive phase ---
    σ = rbm.probas_h(v, sample=False)

    grad_w = -1 / batch_size * (v.T @ σ)
    grad_b = -1 / batch_size * (v.sum(axis=0))
    grad_c = -1 / batch_size * (σ.sum(axis=0))

    # --- Negative phase ---

    # Perform Gibbs sampling
    h = rbm.probas_h(v, sample=True)
    v = rbm.probas_v(h, sample=True)
    for _ in range(k - 1):
        h = rbm.probas_h(v, sample=True)
        v = rbm.probas_v(h, sample=True)

    # Negative gradient estimation
    σ = rbm.probas_h(v, sample=False)

    grad_w += 1 / batch_size * (v.T @ σ)
    grad_b += 1 / batch_size * (v.sum(axis=0))
    grad_c += 1 / batch_size * (σ.sum(axis=0))

    # ---------------------
    # --- Update params ---
    # ---------------------

    if rbm.l1_penalty:
        grad_w += rbm.l1_penalty * xp.sign(rbm.w)

    if rbm.l2_penalty:
        grad_w += rbm.l2_penalty * rbm.w

    rbm.m_w = rbm.momentum * rbm.m_w - rbm.lr * grad_w
    rbm.m_b = rbm.momentum * rbm.m_b - rbm.lr * grad_b
    rbm.m_c = rbm.momentum * rbm.m_c - rbm.lr * grad_c

    rbm.w += rbm.m_w
    rbm.b += rbm.m_b
    rbm.c += rbm.m_c

    if rbm.weight_limit:
        rbm.w = _limit_weights(rbm.w, rbm.weight_limit)


def pcd(rbm: RBM, minibatch: NDArray[FloatType], k: int = 1):
    assert rbm.pc_size is not None
    batch_size = minibatch.shape[0]
    v = minibatch

    # -------------------------
    # --- Compute gradients ---
    # -------------------------

    # --- Positive phase ---
    σ = rbm.probas_h(v, sample=False)

    grad_w = -1 / batch_size * (v.T @ σ)
    grad_b = -1 / batch_size * (v.sum(axis=0))
    grad_c = -1 / batch_size * (σ.sum(axis=0))

    # --- Negative phase ---

    # Perform Gibbs sampling starting from persistent chain
    h = rbm.pc
    v = rbm.probas_v(h, sample=True)
    for _ in range(k - 1):
        h = rbm.probas_h(v, sample=True)
        v = rbm.probas_v(h, sample=True)

    # Negative gradient estimation
    σ = rbm.probas_h(v, sample=False)

    grad_w += 1 / rbm.pc_size * (v.T @ σ)
    grad_b += 1 / rbm.pc_size * (v.sum(axis=0))
    grad_c += 1 / rbm.pc_size * (σ.sum(axis=0))

    # Update persistent chain
    rbm.pc = rbm.probas_h(v, sample=True)

    # ---------------------
    # --- Update params ---
    # ---------------------

    if rbm.l1_penalty:
        grad_w += rbm.l1_penalty * xp.sign(rbm.w)

    if rbm.l2_penalty:
        grad_w += rbm.l2_penalty * rbm.w

    rbm.m_w = rbm.momentum * rbm.m_w - rbm.lr * grad_w
    rbm.m_b = rbm.momentum * rbm.m_b - rbm.lr * grad_b
    rbm.m_c = rbm.momentum * rbm.m_c - rbm.lr * grad_c

    rbm.w += rbm.m_w
    rbm.b += rbm.m_b
    rbm.c += rbm.m_c

    if rbm.weight_limit:
        rbm.w = _limit_weights(rbm.w, rbm.weight_limit)
