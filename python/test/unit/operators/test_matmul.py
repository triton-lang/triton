import itertools

import pytest
import torch

import triton
import triton.language as tl
import triton.ops


def f8_to_f16(x):

    @triton.jit
    def kernel(Y, X, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(X + offs, mask=mask)
        y = x.to(tl.float8e5)
        tl.store(Y + offs, y, mask=mask)

    ret = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    grid = lambda META: (triton.cdiv(x.numel(), META['BLOCK_SIZE']),)
    kernel[grid](ret, triton.reinterpret(x, tl.float8e5), ret.numel(), BLOCK_SIZE=1024)
    return ret


@pytest.mark.parametrize(
    "BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, NWARP, NSTAGE, M, N, K, AT, BT, ADTYPE, BDTYPE",
    itertools.chain(
        *[
            [
                # 1 warp
                (16, 16, 16, 1, 1, 2, None, None, None, AT, BT, DTYPE, DTYPE),
                (32, 16, 16, 1, 1, 2, None, None, None, AT, BT, DTYPE, DTYPE),
                (16, 32, 16, 1, 1, 2, None, None, None, AT, BT, DTYPE, DTYPE),
                (16, 16, 32, 1, 1, 2, None, None, None, AT, BT, DTYPE, DTYPE),
                (32, 16, 32, 1, 1, 2, None, None, None, AT, BT, DTYPE, DTYPE),
                (16, 32, 32, 1, 1, 2, None, None, None, AT, BT, DTYPE, DTYPE),
                (16, 16, 64, 1, 1, 2, None, None, None, AT, BT, DTYPE, DTYPE),
                (64, 16, 64, 1, 1, 2, None, None, None, AT, BT, DTYPE, DTYPE),
                (16, 64, 64, 1, 1, 2, None, None, None, AT, BT, DTYPE, DTYPE),
                # 2 warp
                (64, 32, 64, 1, 2, 2, None, None, None, AT, BT, DTYPE, DTYPE),
                (32, 64, 64, 1, 2, 2, None, None, None, AT, BT, DTYPE, DTYPE),
                (64, 32, 16, 1, 2, 2, None, None, None, AT, BT, DTYPE, DTYPE),
                (32, 64, 16, 1, 2, 2, None, None, None, AT, BT, DTYPE, DTYPE),
                (128, 32, 32, 1, 2, 2, None, None, None, AT, BT, DTYPE, DTYPE),
                (32, 128, 32, 1, 2, 2, None, None, None, AT, BT, DTYPE, DTYPE),
                # 4 warp
                (128, 64, 16, 1, 4, 2, None, None, None, AT, BT, DTYPE, DTYPE),
                (64, 128, 16, 1, 4, 2, None, None, None, AT, BT, DTYPE, DTYPE),
                (128, 32, 32, 1, 4, 2, None, None, None, AT, BT, DTYPE, DTYPE),
                (32, 128, 32, 1, 4, 2, None, None, None, AT, BT, DTYPE, DTYPE),
                (128, 32, 64, 1, 4, 2, None, None, None, AT, BT, DTYPE, DTYPE),
                (32, 128, 64, 1, 4, 2, None, None, None, AT, BT, DTYPE, DTYPE),
                # 8 warp
                (128, 256, 16, 1, 8, 2, None, None, None, AT, BT, DTYPE, DTYPE),
                (256, 128, 16, 1, 8, 2, None, None, None, AT, BT, DTYPE, DTYPE),
                (256, 128, 32, 1, 8, 2, None, None, None, AT, BT, DTYPE, DTYPE),
                # split-k
                (64, 64, 16, 2, 4, 2, None, None, None, AT, BT, DTYPE, DTYPE),
                (64, 64, 16, 4, 4, 2, None, None, None, AT, BT, DTYPE, DTYPE),
                (64, 64, 16, 8, 4, 2, None, None, None, AT, BT, DTYPE, DTYPE),
                # variable input
                (128, 128, 32, 1, 4, 2, 1024, 1024, 1024, AT, BT, DTYPE, DTYPE),
                (128, 128, 32, 1, 4, 2, 384, 128, 640, AT, BT, DTYPE, DTYPE),
                (128, 128, 32, 1, 4, 2, 107, 233, 256, AT, BT, DTYPE, DTYPE),
                (128, 128, 32, 1, 4, 2, 107, 233, 311, AT, BT, DTYPE, DTYPE),
                (128, 256, 64, 1, 8, 3, 1024, 1024, 1024, AT, BT, DTYPE, DTYPE),
            ] for DTYPE in ["float16", "bfloat16", "float32"] for AT in [False, True] for BT in [False, True]
        ],
        # n-stage
        *[
            [
                (16, 16, 16, 1, 1, STAGES, 1024, 1024, 1024, AT, BT, DTYPE, DTYPE),
                (64, 32, 64, 1, 2, STAGES, 1024, 1024, 1024, AT, BT, DTYPE, DTYPE),
                (128, 64, 16, 1, 4, STAGES, 1024, 1024, 1024, AT, BT, DTYPE, DTYPE),
                (256, 128, 32, 1, 8, STAGES, 1024, 1024, 1024, AT, BT, DTYPE, DTYPE),
                (128, 128, 32, 1, 4, STAGES, 384, 128, 640, AT, BT, DTYPE, DTYPE),
                # split-k
                (64, 64, 16, 8, 4, STAGES, 1024, 1024, 1024, AT, BT, DTYPE, DTYPE),
                (64, 64, 16, 8, 4, STAGES, 1024, 1024, 32, AT, BT, DTYPE, DTYPE),
            ] for DTYPE in ["float16", "bfloat16", "float32"] for AT in [False, True] for BT in [False, True] for STAGES in [2, 3, 4]
        ],
        # mixed-precision
        *[
            [
                (16, 16, 16, 1, 1, 2, None, None, None, AT, BT, ADTYPE, BDTYPE),
                (128, 32, 32, 1, 2, 2, None, None, None, AT, BT, ADTYPE, BDTYPE),
                (128, 256, 16, 1, 8, 2, None, None, None, AT, BT, ADTYPE, BDTYPE),
                (32, 64, 16, 1, 1, 2, 64, 128, 32, AT, BT, ADTYPE, BDTYPE),
                (128, 128, 32, 8, 4, 2, 1024, 1024, 1024, AT, BT, ADTYPE, BDTYPE),
            ] for ADTYPE, BDTYPE in [("float8", "float16"), ("float16", "float32"), ("float32", "float16"),
                                     ("bfloat16", "float32"), ("float32", "bfloat16")] for AT in [False, True] for BT in [False, True]
        ]
    ),
)
def test_op(BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, NWARP, NSTAGE, M, N, K, AT, BT, ADTYPE, BDTYPE):
    capability = torch.cuda.get_device_capability()
    if capability[0] < 7:
        pytest.skip("Only test tl.dot() on devices with sm >= 70")
    if capability[0] < 8 and (ADTYPE == "bfloat16" or BDTYPE == "bfloat16"):
        pytest.skip("Only test bfloat16 on devices with sm >= 80")
    if (ADTYPE == "bfloat16" or BDTYPE == "bfloat16") and SPLIT_K != 1:
        pytest.skip("bfloat16 matmuls don't allow split_k for now")
    torch.manual_seed(0)
    # nuke kernel decorators -- will set meta-parameters manually
    kwargs = {'BLOCK_M': BLOCK_M, 'BLOCK_N': BLOCK_N, 'BLOCK_K': BLOCK_K, 'SPLIT_K': SPLIT_K}
    pre_hook = None if SPLIT_K == 1 else lambda nargs: nargs['C'].zero_()
    configs = [triton.Config(kwargs=kwargs, num_warps=NWARP, num_stages=NSTAGE, pre_hook=pre_hook)]
    kernel = triton.ops._matmul.kernel
    kernel.configs = configs
    # kernel.run = kernel.run.run.run

    # get matrix shape
    M = BLOCK_M if M is None else M
    N = BLOCK_N if N is None else N
    K = BLOCK_K * SPLIT_K if K is None else K

    def get_input(n, m, t, dtype):
        if t:
            return get_input(m, n, False, dtype).t()
        if dtype == "float8":
            x = torch.randint(10, 50, (n, m), device="cuda", dtype=torch.int8)
            return f8_to_f16(x)
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
        return .1 * torch.randn((n, m), device="cuda", dtype=dtype)

    # allocate/transpose inputs
    a = get_input(M, K, AT, ADTYPE)
    b = get_input(K, N, BT, BDTYPE)
    # run test
    th_c = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    try:
        tt_c = triton.ops.matmul(a, b)
        atol, rtol = 1e-2, 0
        if ADTYPE == torch.bfloat16 or BDTYPE == torch.bfloat16:
            atol, rtol = 3.5e-2, 0
        torch.testing.assert_allclose(th_c, tt_c, atol=atol, rtol=rtol)
    except triton.OutOfResources as e:
        pytest.skip(str(e))
