import pytest
import itertools
import triton
import torch


@pytest.mark.parametrize(
    "BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, NWARP, M, N, K, AT, BT, DTYPE",
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
                # # # 2 warp
                (64, 32, 64, 1, 2, None, None, None, AT, BT, DTYPE),
                (32, 64, 64, 1, 2, None, None, None, AT, BT, DTYPE),
                (64, 32, 16, 1, 2, None, None, None, AT, BT, DTYPE),
                (32, 64, 16, 1, 2, None, None, None, AT, BT, DTYPE),
                (128, 32, 32, 1, 2, None, None, None, AT, BT, DTYPE),
                (32, 128, 32, 1, 2, None, None, None, AT, BT, DTYPE),
                # # # 4 warp
                (128, 64, 16, 1, 4, None, None, None, AT, BT, DTYPE),
                (64, 128, 16, 1, 4, None, None, None, AT, BT, DTYPE),
                (128, 32, 32, 1, 4, None, None, None, AT, BT, DTYPE),
                (32, 128, 32, 1, 4, None, None, None, AT, BT, DTYPE),
                (128, 32, 64, 1, 4, None, None, None, AT, BT, DTYPE),
                (32, 128, 64, 1, 4, None, None, None, AT, BT, DTYPE),
                # # 8 warp
                (128, 256, 16, 1, 8, None, None, None, AT, BT, DTYPE),
                (256, 128, 16, 1, 8, None, None, None, AT, BT, DTYPE),
                (256, 128, 32, 1, 8, None, None, None, AT, BT, DTYPE),
                # # split-k
                # (64, 64, 16, 2, 4, None, None, None, AT, BT, DTYPE),
                # (64, 64, 16, 4, 4, None, None, None, AT, BT, DTYPE),
                # (64, 64, 16, 8, 4, None, None, None, AT, BT, DTYPE),
                # # variable input
                # (128, 128, 32, 1, 4, 1024, 1024, 1024, AT, BT, DTYPE),
                # (128, 128, 32, 1, 4, 384, 128, 640, AT, BT, DTYPE),
                # (128, 128, 32, 1, 4, 107, 233, 256, AT, BT, DTYPE),
                # (128, 128, 32, 1, 4, 107, 233, 311, AT, BT, DTYPE),
            ] for DTYPE in ["float16"] for AT in [False, True] for BT in [False, True]
        ]
    ),
)
def test_op(BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, NWARP, M, N, K, AT, BT, DTYPE):
    torch.manual_seed(0)
    # nuke kernel decorators -- will set meta-parameters manually
    META = {'BLOCK_M': BLOCK_M, 'BLOCK_N': BLOCK_N, 'BLOCK_K': BLOCK_K, 'SPLIT_K': SPLIT_K, 'GROUP_M': 8}
    configs = [triton.Config(meta=META, num_warps=NWARP)]
    kernel = triton.ops._matmul.kernel
    decorators = kernel.kernel_decorators
    kernel.kernel_decorators = []
    triton.autotune(configs, [])(kernel)
    kernel.kernel_decorators += decorators[1:]
    # get matrix shape
    M = BLOCK_M if M is None else M
    N = BLOCK_N if N is None else N
    K = BLOCK_K * SPLIT_K if K is None else K
    # allocate/transpose inputs
    DTYPE = {"float16": torch.float16, "float32": torch.float32}[DTYPE]
    a = torch.randn((K, M) if AT else (M, K), device="cuda", dtype=DTYPE)
    b = torch.randn((N, K) if BT else (K, N), device="cuda", dtype=DTYPE)
    a = a.t() if AT else a
    b = b.t() if BT else b
    # run test
    th_c = torch.matmul(a, b)
    tt_c = triton.ops.matmul(a, b)
    assert triton.testing.allclose(th_c, tt_c)
