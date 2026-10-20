import pytest
import torch
import triton
import triton.language as tl

from triton._internal_testing import is_hip


@pytest.mark.skipif(not is_hip(), reason="requires HIP")
@pytest.mark.parametrize("wgp_cu_mode", ["wgp", "cu"])
def test_reduce_cu_mode(wgp_cu_mode, device):
    arch = triton.runtime.driver.active.get_current_target().arch
    if not arch.startswith(("gfx11", "gfx120")):
        pytest.skip("target has no WGP/CU mode distinction")

    @triton.jit
    def kernel(src, dst, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        total = tl.sum(tl.load(src + offs))
        tl.store(dst + offs, tl.full((BLOCK, ), 0, tl.float32) + total)

    x = torch.rand(1024, device=device)
    y = torch.empty_like(x)
    handle = kernel[(1, )](x, y, BLOCK=1024, num_warps=8, wgp_cu_mode=wgp_cu_mode)
    assert ('"amdgpu-synchronize-as"' in handle.asm["llir"]) == (wgp_cu_mode == "cu")
    torch.testing.assert_close(y, x.sum().expand_as(y), rtol=1e-4, atol=1e-4)
