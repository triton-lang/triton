import pytest
import itertools
import triton
import torch


@pytest.mark.parametrize(
    "TM, TN, TK, SPLITK, NWARP, M, N, K, AT, BT, DTYPE",
    itertools.chain(
        *[
            [
                # 1 warp
                (16, 16, 16, 1, 1, None, None, None, AT, BT, DTYPE),
                (32, 16, 16, 1, 1, None, None, None, AT, BT, DTYPE),
                (16, 32, 16, 1, 1, None, None, None, AT, BT, DTYPE),
                (16, 16, 32, 1, 1, None, None, None, AT, BT, DTYPE),
                (32, 16, 32, 1, 1, None, None, None, AT, BT, DTYPE),
                (16, 32, 32, 1, 1, None, None, None, AT, BT, DTYPE),
                (16, 16, 64, 1, 1, None, None, None, AT, BT, DTYPE),
                (64, 16, 64, 1, 1, None, None, None, AT, BT, DTYPE),
                (16, 64, 64, 1, 1, None, None, None, AT, BT, DTYPE),
                # # 2 warp
                (64, 32, 64, 1, 2, None, None, None, AT, BT, DTYPE),
                (32, 64, 64, 1, 2, None, None, None, AT, BT, DTYPE),
                (64, 32, 16, 1, 2, None, None, None, AT, BT, DTYPE),
                (32, 64, 16, 1, 2, None, None, None, AT, BT, DTYPE),
                (128, 32, 32, 1, 2, None, None, None, AT, BT, DTYPE),
                (32, 128, 32, 1, 2, None, None, None, AT, BT, DTYPE),
                # # 4 warp
                (128, 64, 16, 1, 4, None, None, None, AT, BT, DTYPE),
                (64, 128, 16, 1, 4, None, None, None, AT, BT, DTYPE),
                (128, 32, 32, 1, 4, None, None, None, AT, BT, DTYPE),
                (32, 128, 32, 1, 4, None, None, None, AT, BT, DTYPE),
                (128, 32, 64, 1, 4, None, None, None, AT, BT, DTYPE),
                (32, 128, 64, 1, 4, None, None, None, AT, BT, DTYPE),
                # 8 warp
                # (128, 256, 16, 1, 8, None, None, None, AT, BT, DTYPE),
                # (256, 128, 16, 1, 8, None, None, None, AT, BT, DTYPE),
                # (256, 128, 32, 1, 8, None, None, None, AT, BT, DTYPE),
                # split-k
                (64, 64, 16, 2, 4, None, None, None, AT, BT, DTYPE),
                (64, 64, 16, 4, 4, None, None, None, AT, BT, DTYPE),
                (64, 64, 16, 8, 4, None, None, None, AT, BT, DTYPE),
                # variable input
                (128, 128, 32, 1, 4, 1024, 1024, 1024, AT, BT, DTYPE),
                (128, 128, 32, 1, 4, 384, 128, 640, AT, BT, DTYPE),
                # (128, 128, 32, 1, 4, 107, 233, 256, AT, BT, DTYPE),
                # (128, 128, 32, 1, 4, 107, 233, 311, AT, BT, DTYPE),
            ]
            for DTYPE in ["float16"]
            for AT in [False, True]
            for BT in [False, True]
        ]
    ),
)
def test_op(TM, TN, TK, SPLITK, NWARP, M, N, K, AT, BT, DTYPE):
    DTYPE = {"float16": torch.float16, "float32": torch.float32}[DTYPE]
    torch.manual_seed(0)
    triton.ops._matmul._kernels = dict()
    triton.ops._matmul._CONFIGS = [
        ({"TM": str(TM), "TN": str(TN), "TK": str(TK), "SPLITK": str(SPLITK)}, NWARP)
    ]
    if M is None:
        M = TM
    if N is None:
        N = TN
    if K is None:
        K = TK * SPLITK
    a = torch.randn((K, M) if AT else (M, K), device="cuda", dtype=DTYPE)
    b = torch.randn((N, K) if BT else (K, N), device="cuda", dtype=DTYPE)
    a = a.t() if AT else a
    b = b.t() if BT else b
    th_c = torch.matmul(a, b)
    tt_c = triton.ops.matmul(a, b)
    assert triton.testing.allclose(th_c, tt_c)
