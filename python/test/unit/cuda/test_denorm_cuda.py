import pytest
import torch
import triton
import triton.language as tl

from triton._internal_testing import is_cuda
from triton.language import core


@core.extern
def _fma(x, scale, lib_path: tl.constexpr, _semantic=None):
    return core.extern_elementwise(
        "test_denorm",
        lib_path.value,
        [x, scale],
        {(core.float32, core.float32): ("fma", core.float32)},
        is_pure=True,
        _semantic=_semantic,
    )


@triton.jit
def _kernel(x, scale, lib_path: tl.constexpr):
    offsets = tl.arange(0, 2)
    tl.store(x + offsets, _fma(tl.load(x + offsets), scale, lib_path))


@pytest.mark.skipif(not is_cuda(), reason="CUDA only")
def test_linked_function_preserves_denorms(tmp_path):
    lib = tmp_path / "denorm.ll"
    lib.write_text(r'''
define float @fma(float %x, float %scale) #0 {
  %result = call float @llvm.fma.f32(float %x, float %scale, float 0.000000e+00)
  ret float %result
}
declare float @llvm.fma.f32(float, float, float)
attributes #0 = { cold denormal_fpenv(float: preservesign) "target-cpu"="sm_100" }
''')
    bits = [1, -2147483645]
    x = torch.tensor(bits, dtype=torch.int32, device="cuda").view(torch.float32)
    _kernel[(1, )](x, 1.0, str(lib), extern_libs={"test_denorm": str(lib)})
    assert x.view(torch.int32).tolist() == bits
