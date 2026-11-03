import ctypes

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import requires_tma
from triton.tools.tensor_descriptor import TensorDescriptor


@requires_tma
def test_specialization_after_host_tensordesc():

    @triton.jit
    def kernel(a, b):
        pass

    device = "cuda"
    A = torch.randn(1024, device=device)
    desc = TensorDescriptor.from_tensor(A, [128])
    h = kernel.warmup(desc, 16, grid=(1, ))
    assert "%a: !tt.tensordesc<128xf32>" in h.asm["ttir"]
    assert "%b: i32 {tt.divisibility = 16 : i32}" in h.asm["ttir"]


@requires_tma
def test_host_tensordesc_encoded_bytes():
    cuda = pytest.importorskip("cuda.bindings.driver")

    @triton.jit
    def load(desc, out):
        tl.store(out + tl.arange(0, 32), desc.load([32]))

    src = torch.arange(128, device="cuda", dtype=torch.int32)
    out = torch.empty_like(src[:32])

    def encode(desc, metadata):
        status, tensor_map = cuda.cuTensorMapEncodeTiled(cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT32, 1,
                                                         desc.base.data_ptr(), [cuda.cuuint64_t(desc.shape[0])],
                                                         [cuda.cuuint64_t(desc.base.nbytes)],
                                                         [cuda.cuuint32_t(metadata["block_size"][0])],
                                                         [cuda.cuuint32_t(1)],
                                                         cuda.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
                                                         cuda.CUtensorMapSwizzle(metadata["swizzle"]),
                                                         cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                                         cuda.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
        assert status == cuda.CUresult.CUDA_SUCCESS
        return ctypes.string_at(tensor_map.getPtr(), 128)

    desc = TensorDescriptor.from_tensor(src, [32])
    desc.encode_tma = encode
    load[(1, )](desc, out)
    torch.testing.assert_close(out, src[32:64])

    desc.encode_tma = lambda desc, metadata: bytes(127)
    with pytest.raises(TypeError, match="128 descriptor bytes"):
        load[(1, )](desc, out)
