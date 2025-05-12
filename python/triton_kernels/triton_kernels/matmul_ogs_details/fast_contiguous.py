import torch
from ._weight_transpose import _weight_transpose


def weight_transpose(X, Y, BLOCK_M=128, BLOCK_N=128, num_warps=16) -> None:
    if X.dtype.itemsize == 1:
        X = X.view(torch.int8)
    if Y.dtype.itemsize == 1:
        Y = Y.view(torch.int8)

    # check compatibility:
    assert X.shape == Y.shape
    assert X.dtype == Y.dtype

    # this doubles up as an assertion:
    is_3d = {3: True, 2: False}[len(X.shape)]

    M = X.shape[-2]
    N = X.shape[-1]
    E = X.shape[0] if is_3d else 1

    stride_xm = X.stride(-2)
    stride_xn = X.stride(-1)
    stride_xe = X.stride(0) if is_3d else 0

    stride_ym = Y.stride(-2)
    stride_yn = Y.stride(-1)
    stride_ye = Y.stride(0) if is_3d else 0

    grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N, E)

    _weight_transpose[grid](
        M,
        N,
        BLOCK_M,
        BLOCK_N,
        X,
        stride_xe,
        stride_xm,
        stride_xn,
        Y,
        stride_ye,
        stride_ym,
        stride_yn,
        num_warps=num_warps,
    )


def fast_contiguous(X):
    if X.is_contiguous():
        return X
    Y = torch.empty(X.shape, device=X.device, dtype=X.dtype)
    weight_transpose(X, Y)
    return Y
