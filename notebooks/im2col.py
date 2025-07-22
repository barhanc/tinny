"""Simple and clean implementation of the im2col transformation"""

import numpy as np
from numpy.typing import NDArray


def im2col(
    x: NDArray,
    strides: tuple[int, int],
    dilation: tuple[int, int],
    kernel_size: tuple[int, int],
) -> tuple[NDArray, NDArray]:
    # Get input dimensions
    B, C_in, H_in, W_in = x.shape

    # Compute output size
    H_out = int(1 + (H_in - dilation[0] * (kernel_size[0] - 1) - 1) / strides[0])
    W_out = int(1 + (W_in - dilation[1] * (kernel_size[1] - 1) - 1) / strides[1])

    # The general idea is to first compute an index-array of shape `(B, C_in*H_ker*W_ker,
    # H_out*W_out)` with flat index (i.e. a number between 0 and prod(x.shape) - 1) of the correct
    # element to take from `x` and then use `np.take`.

    # First we compute 1-D index arrays for each dimension. Basically if we would concatenate these
    # arrays as columns then each row would be a unique multi-index to the array of appropriate
    # shape i.e. `(C_in, H_ker, W_ker)` or `(H_out, W_out)`.

    # `(C_in * H_ker * W_ker,)` each
    idx_c, idx_h_ker, idx_w_ker = np.indices((C_in, *kernel_size)).reshape(3, -1)

    # `(H_out * W_out,)` each
    idx_h_out, idx_w_out = np.indices((H_out, W_out)).reshape(2, -1)

    # Then we compute 4 index-arrays broadcastable to `(B, C_in * H_ker * W_ker, H_out * W_out)`
    # each, such that forall i,j,k: x[idx_b[i,j,k], idx_c[i,j,k], idx_h[i,j,k], idx_w[i,j,k]] is the
    # correct element to put under index (i,j,k) in the im2col matrix.

    # broadcastable to `(B, C_in * H_ker * W_ker, H_out * W_out)`
    idx_b = np.arange(B).reshape(-1, 1, 1)
    idx_c = idx_c.reshape(-1, 1)
    idx_h = dilation[0] * idx_h_ker.reshape(-1, 1) + strides[0] * idx_h_out
    idx_w = dilation[1] * idx_w_ker.reshape(-1, 1) + strides[1] * idx_w_out

    # Here we could just return x[idx_b, idx_c, idx_h, idx_w]. However for the backward pass it is
    # much more convenient to have these indices raveled i.e. instead of having 4 index-arrays with
    # indices along given (i.e.: 0,1,2,3) dimension we can transform them to a single index-array
    # with raveled index i.e. index to the flattened array `x`.

    # `(B, C_in * H_ker * W_ker, H_out * W_out)`
    idxs = np.ravel_multi_index((idx_b, idx_c, idx_h, idx_w), x.shape)

    # Finally we return the raveled index-array and the im2col array which can be constructed using
    # `np.take` as by default the flattened input array is used to extract the values.
    return idxs, np.take(x, idxs)
