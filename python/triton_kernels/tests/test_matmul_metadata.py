import pytest
import torch

from triton_kernels.matmul_details._common import _matmul_flops_and_bytes_from_slices, matmul_launch_metadata
from triton_kernels.proton_opts import launch_metadata_allow_sync, set_launch_metadata_allow_sync


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


def _metadata_args(
    *,
    ragged_dimension,
    M,
    N,
    K,
    X,
    Y,
    W,
    slice_sizes,
    batch_size=1,
    out_acc=None,
    mx_block_size=32,
    x_mx_scale=None,
    w_mx_scale=None,
    x_tensor_scale=None,
    w_tensor_scale=None,
):
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
        "MX_BLOCK_SIZE": mx_block_size,
        "XMxScale": x_mx_scale,
        "WMxScale": w_mx_scale,
        "XTensorScale": x_tensor_scale,
        "WTensorScale": w_tensor_scale,
    }


@pytest.mark.parametrize(
    ("batch_size", "x_batched", "mx_block_size", "tensor_scale_dtype", "expected_bytes"),
    [
        (1, False, 32, None, 664),
        (1, False, 16, None, 688),
        (1, False, 16, torch.float32, 784),
        (2, True, 16, torch.float32, 1568),
        (2, False, 16, torch.float32, 1392),
    ],
)
def test_matmul_launch_metadata_fp4_x_fp4_uses_flops4_and_logical_bytes(batch_size, x_batched, mx_block_size,
                                                                        tensor_scale_dtype, expected_bytes):
    M, N, K = 8, 16, 32
    X = torch.empty((batch_size, M, K // 2) if x_batched else (M, K // 2), dtype=torch.uint8)
    Y = torch.empty((batch_size, M, N) if batch_size > 1 else (M, N), dtype=torch.bfloat16)
    W = torch.empty((batch_size, K // 2, N), dtype=torch.uint8)
    x_tensor_scale = None if tensor_scale_dtype is None else torch.empty((), dtype=tensor_scale_dtype)
    w_tensor_scale = None if tensor_scale_dtype is None else torch.empty((), dtype=tensor_scale_dtype)
    args = _metadata_args(
        ragged_dimension=None,
        M=M,
        N=N,
        K=K,
        X=X,
        Y=Y,
        W=W,
        slice_sizes=None,
        batch_size=batch_size,
        mx_block_size=mx_block_size,
        x_mx_scale=torch.empty((), dtype=torch.uint8),
        w_mx_scale=torch.empty((), dtype=torch.uint8),
        x_tensor_scale=x_tensor_scale,
        w_tensor_scale=w_tensor_scale,
    )

    actual = matmul_launch_metadata(None, _Kernel(), args)

    assert "flops8" not in actual
    assert actual["flops4"] == 8192 * batch_size
    assert actual["bytes"] == expected_bytes


def test_matmul_launch_metadata_fp4_x_fp4_uses_logical_rows_for_swizzled_storage():
    M, N, K = 1000, 16, 32
    X = torch.empty((512, K), dtype=torch.uint8)
    Y = torch.empty((M, N), dtype=torch.bfloat16)
    W = torch.empty((K // 2, N), dtype=torch.uint8)
    args = _metadata_args(
        ragged_dimension=None,
        M=M,
        N=N,
        K=K,
        X=X,
        Y=Y,
        W=W,
        slice_sizes=None,
        mx_block_size=32,
        x_mx_scale=torch.empty((), dtype=torch.uint8),
        w_mx_scale=torch.empty((), dtype=torch.uint8),
    )

    actual = matmul_launch_metadata(None, _Kernel(), args)

    bytes_per_k = 17
    assert actual["bytes"] == M * bytes_per_k + N * bytes_per_k + Y.numel() * Y.element_size()


def test_matmul_launch_metadata_fp4_x_fp4_handles_empty_k():
    M, N, K = 8, 16, 0
    X = torch.empty((M, K), dtype=torch.uint8)
    Y = torch.empty((M, N), dtype=torch.bfloat16)
    W = torch.empty((K, N), dtype=torch.uint8)
    args = _metadata_args(
        ragged_dimension=None,
        M=M,
        N=N,
        K=K,
        X=X,
        Y=Y,
        W=W,
        slice_sizes=None,
        mx_block_size=32,
        x_mx_scale=torch.empty((), dtype=torch.uint8),
        w_mx_scale=torch.empty((), dtype=torch.uint8),
    )

    actual = matmul_launch_metadata(None, _Kernel(), args)

    assert actual["flops4"] == 0
    assert actual["bytes"] == Y.numel() * Y.element_size()


def test_matmul_launch_metadata_mixed_fp4_fp8_keeps_legacy_metrics():
    M, N, K = 8, 16, 32
    X = torch.empty((M, K), dtype=torch.float8_e4m3fn)
    Y = torch.empty((M, N), dtype=torch.bfloat16)
    W = torch.empty((K // 2, N), dtype=torch.uint8)
    args = _metadata_args(
        ragged_dimension=None,
        M=M,
        N=N,
        K=K,
        X=X,
        Y=Y,
        W=W,
        slice_sizes=None,
    )

    actual = matmul_launch_metadata(None, _Kernel(), args)

    assert actual["flops8"] == 8192
    assert "flops4" not in actual
    assert actual["bytes"] == X.numel() + Y.numel() * 2 + W.numel()


def test_matmul_launch_metadata_fp4_x_fp4_ragged_m_counts_active_scale_bytes():
    N, K = 16, 32
    slice_sizes = torch.tensor([2, 0, 3], dtype=torch.int32)
    n_tokens = int(slice_sizes.sum())
    X = torch.empty((n_tokens, K // 2), dtype=torch.uint8)
    Y = torch.empty((n_tokens, N), dtype=torch.bfloat16)
    W = torch.empty((slice_sizes.numel(), K // 2, N), dtype=torch.uint8)
    args = _metadata_args(
        ragged_dimension="M",
        M=None,
        N=N,
        K=K,
        X=X,
        Y=Y,
        W=W,
        slice_sizes=slice_sizes,
        mx_block_size=16,
        x_mx_scale=torch.empty((), dtype=torch.uint8),
        w_mx_scale=torch.empty((), dtype=torch.uint8),
        x_tensor_scale=torch.empty((), dtype=torch.float32),
        w_tensor_scale=torch.empty((), dtype=torch.float32),
    )

    previous_allow_sync = launch_metadata_allow_sync()
    try:
        set_launch_metadata_allow_sync(True)
        actual = matmul_launch_metadata(None, _Kernel(), args)
    finally:
        set_launch_metadata_allow_sync(previous_allow_sync)

    assert "flops8" not in actual
    assert actual["flops4"] == 5120
    assert actual["bytes"] == 974


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

    previous_allow_sync = launch_metadata_allow_sync()
    try:
        set_launch_metadata_allow_sync(False)
        actual = matmul_launch_metadata(None, _Kernel(), args)
        torch.cuda.synchronize(device)
    finally:
        set_launch_metadata_allow_sync(previous_allow_sync)

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
