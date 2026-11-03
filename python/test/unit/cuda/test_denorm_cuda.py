import pytest
import torch
import triton
import triton.language as tl

from triton._internal_testing import is_cuda
from triton.language import core


@core.extern
def _sin_f32(x, lib_path: tl.constexpr, _semantic=None):
    return core.extern_elementwise(
        "test_denorm",
        lib_path.value,
        [x],
        {(core.float32, ): ("_sin_f32", core.float32)},
        is_pure=True,
        _semantic=_semantic,
    )


@triton.jit
def _kernel(x, lib_path: tl.constexpr):
    offsets = tl.arange(0, 1)
    tl.store(x + offsets, _sin_f32(tl.load(x + offsets), lib_path))


@pytest.mark.skipif(not is_cuda(), reason="CUDA only")
def test_linked_function_preserves_denorms(tmp_path):
    lib = tmp_path / "denorm.ll"
    lib.write_text(r'''
define float @_sin_f32(float %x) #0 {
  %scaled = fmul float %x, 0x3FE45F3060000000
  %quadrant = call i32 @llvm.nvvm.f2i.rn.ftz(float %scaled)
  %quadrant.f = sitofp i32 %quadrant to float
  %result = call float @llvm.fma.f32(float %quadrant.f, float 0xBFF921FB40000000, float %x)
  ret float %result
}
declare i32 @llvm.nvvm.f2i.rn.ftz(float)
declare float @llvm.fma.f32(float, float, float)
attributes #0 = { cold denormal_fpenv(float: preservesign) "target-cpu"="sm_100" "target-features"="+ptx87,+sm_100" }
''')
    bits = [1]
    x = torch.tensor(bits, dtype=torch.int32, device="cuda").view(torch.float32)
    _kernel[(1, )](x, str(lib), extern_libs={"test_denorm": str(lib)})
    assert x.view(torch.int32).tolist() == bits
