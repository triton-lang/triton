import torch

import triton
import pytest
import triton.language as tl
from triton.tools.disasm import get_sass


def test_disam_cubin():
    if not triton.runtime.driver.active.get_current_target().backend == "cuda":
        pytest.skip("Test requires CUDA.")

    @triton.jit
    def kernel(X, i: tl.constexpr):
        tl.store(X, i)

    x = torch.empty(1, dtype=torch.int32, device='cuda')
    h = kernel[(1, )](x, i=12)
    assert x[0] == 12
    sass = get_sass(h.asm["cubin"])
    # check that the sass has a store instruction.
    assert "STG.E" in sass
