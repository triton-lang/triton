import pytest
import torch

from triton_kernels.matmul_details._common import _matmul_flops_and_bytes_from_slices, matmul_launch_metadata
from triton_kernels.proton_opts import set_launch_metadata_allow_sync


class _Kernel:
    name = "_p_matmul_test"
    num_stages = 4


def _old_flops_and_bytes(args, M, N, K, X, Y, W, slice_sizes, nbits, batch_size):
    n_tokens = slice_sizes.sum()
    z = 1 if args["RAGGED_DIMENSION"] == "K" else batch_size
    fM = M if M is not None else n_tokens
    fK = K if K is not None else n_tokens
    flops = 2.0 * fM * N * fK * z

    if args["RAGGED_DIMENSION"] == "K":
        n_x_bytes = n_tokens * X.shape[-2] * X.element_size()
        n_y_bytes = Y.numel() * Y.element_size() * (2 if args["OutAcc"] is not None else 1)
        n_w_bytes = n_tokens * W.shape[-1] * W.element_size()
    else:
        n_x_bytes = n_tokens * X.shape[-1] * X.element_size()
        n_y_bytes = n_tokens * Y.shape[-1] * Y.element_size()
        n_w_bytes = (W.numel() * W.element_size() // slice_sizes.numel()) * (slice_sizes > 0).sum()

    return {f"flops{nbits}": flops.to(torch.float64), "bytes": n_x_bytes + n_y_bytes + n_w_bytes}


def _metadata_args(*, ragged_dimension, M, N, K, X, Y, W, slice_sizes, batch_size=1, out_acc=None):
    return {
        "M": M,
        "N": N,
        "K": K,
        "YPtr": Y,
        "XPtr": X,
        "WPtr": W,
        "XSliceSizes": slice_sizes,
        "X_EXPECTED_SLICE_SIZE": None,
        "RAGGED_DIMENSION": ragged_dimension,
        "OutAcc": out_acc,
        "batch_size": batch_size,
        "EPILOGUE_SUBTILE": None,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "case",
    [
        "ragged_m",
        "ragged_k",
        "ragged_k_out_acc",
    ],
)
def test_matmul_launch_metadata_nosync_matches_old_formula(case):
    device = torch.device("cuda")
    slice_sizes = torch.tensor([7, 0, 13, 4, 1], dtype=torch.int32, device=device)
    nbits = 16

    if case == "ragged_m":
        M, N, K = None, 16, 8
        batch_size = 2
        X = torch.empty((40, K), dtype=torch.float16, device=device)
        Y = torch.empty((40, N), dtype=torch.float16, device=device)
        W = torch.empty((slice_sizes.numel(), K, N), dtype=torch.float16, device=device)
        args = _metadata_args(
            ragged_dimension="M",
            M=M,
            N=N,
            K=K,
            X=X,
            Y=Y,
            W=W,
            slice_sizes=slice_sizes,
            batch_size=batch_size,
        )
    else:
        M, N, K = 8, 16, None
        out_acc = torch.empty((M, N), dtype=torch.float32, device=device) if case == "ragged_k_out_acc" else None
        X = torch.empty((M, 40), dtype=torch.float16, device=device)
        Y = torch.empty((M, N), dtype=torch.float16, device=device)
        W = torch.empty((40, N), dtype=torch.float16, device=device)
        args = _metadata_args(
            ragged_dimension="K",
            M=M,
            N=N,
            K=K,
            X=X,
            Y=Y,
            W=W,
            slice_sizes=slice_sizes,
            out_acc=out_acc,
        )

    expected = _old_flops_and_bytes(args, M, N, K, X, Y, W, slice_sizes, nbits, args["batch_size"])
    direct_actual = _matmul_flops_and_bytes_from_slices(args, M, N, K, X, Y, W, slice_sizes, nbits, args["batch_size"])

    try:
        set_launch_metadata_allow_sync(False)
        actual = matmul_launch_metadata(None, _Kernel(), args)
        torch.cuda.synchronize(device)
    finally:
        set_launch_metadata_allow_sync(True)

    assert actual["name"].startswith(_Kernel.name)
    assert actual[f"flops{nbits}"].dtype == torch.float64
    assert actual["bytes"].dtype == torch.int64
    torch.testing.assert_close(direct_actual[f"flops{nbits}"].cpu(), expected[f"flops{nbits}"].cpu(), rtol=0, atol=0)
    torch.testing.assert_close(direct_actual["bytes"].cpu(), expected["bytes"].to(torch.int64).cpu(), rtol=0, atol=0)
    torch.testing.assert_close(actual[f"flops{nbits}"].cpu(), expected[f"flops{nbits}"].cpu(), rtol=0, atol=0)
    torch.testing.assert_close(actual["bytes"].cpu(), expected["bytes"].to(torch.int64).cpu(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_matmul_flops_and_bytes_from_slices_handles_large_slice_count():
    device = torch.device("cuda")
    slice_sizes = torch.arange(1501, dtype=torch.int32, device=device) % 17
    n_tokens = int(slice_sizes.cpu().sum())
    nbits = 16
    M, N, K = None, 16, 8
    batch_size = 2
    X = torch.empty((n_tokens, K), dtype=torch.float16, device=device)
    Y = torch.empty((n_tokens, N), dtype=torch.float16, device=device)
    W = torch.empty((slice_sizes.numel(), K, N), dtype=torch.float16, device=device)
    args = _metadata_args(
        ragged_dimension="M",
        M=M,
        N=N,
        K=K,
        X=X,
        Y=Y,
        W=W,
        slice_sizes=slice_sizes,
        batch_size=batch_size,
    )

    expected = _old_flops_and_bytes(args, M, N, K, X, Y, W, slice_sizes, nbits, batch_size)
    actual = _matmul_flops_and_bytes_from_slices(args, M, N, K, X, Y, W, slice_sizes, nbits, batch_size)
    torch.cuda.synchronize(device)

    torch.testing.assert_close(actual[f"flops{nbits}"].cpu(), expected[f"flops{nbits}"].cpu(), rtol=0, atol=0)
    torch.testing.assert_close(actual["bytes"].cpu(), expected["bytes"].to(torch.int64).cpu(), rtol=0, atol=0)
