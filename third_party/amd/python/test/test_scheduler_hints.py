import pytest
import torch
import triton
import triton.language as tl

from triton._internal_testing import is_hip

if not is_hip():
    pytest.skip(allow_module_level=True)


def test_schedule_hint(device):

    @triton.jit
    def kernel(X, Y, Z, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        off_m = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        off_k = tl.arange(0, BLOCK_K)
        Xs = X + off_m[:, None] * BLOCK_K + off_k[None, :] * 1
        Ys = Y + off_k[:, None] * 1 + off_n[None, :] * BLOCK_K
        z_offset = off_m[:, None] * BLOCK_N + off_n[None, :] * 1
        Zs = Z + z_offset
        x = tl.load(Xs)
        y = tl.load(Ys)
        z = tl.dot(x, y)
        # additional computations to give more diverse context to backend scheduler
        z += z_offset
        tl.store(Zs, z)

    M = 128
    N = 128
    K = 128

    pgm_default = kernel.warmup(torch.float32, torch.float32, torch.float32, M, N, K, grid=(1, ))
    pgm_ilp = kernel.warmup(torch.float32, torch.float32, torch.float32, M, N, K,
                            schedule_hint="memory-bound-attention", grid=(1, ))

    # check that option affects only llvm backend
    assert pgm_default.asm["llir"] == pgm_ilp.asm["llir"]
    assert pgm_default.asm["amdgcn"] != pgm_ilp.asm["amdgcn"]
