import torch

import triton
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
    assert "%a: !tt.tensordesc<tensor<128xf32>>" in h.asm["ttir"]
    assert "%b: i32 {tt.divisibility = 16 : i32}" in h.asm["ttir"]
