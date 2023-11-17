# Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os

import pytest
import torch
from torch.testing import assert_close

import triton


@pytest.mark.parametrize('TTGIR,TRANS_A,TRANS_B', [
    # TODO: uncomment when it's done
    # ["wgmma_tma_64_64_16_f16.ttgir", False, True],
])
def test_tma_wgmma_64_64_16_f16(TTGIR, TRANS_A, TRANS_B):
    capability = torch.cuda.get_device_capability()
    if capability[0] < 9:
        pytest.skip("Only test wgmma on devices with sm >= 90")

    SIZE_M = 64
    SIZE_N = 64
    SIZE_K = 16
    if (TRANS_A):
        a = torch.randn((SIZE_K, SIZE_M), device='cuda', dtype=torch.float16).T
    else:
        a = torch.randn((SIZE_M, SIZE_K), device='cuda', dtype=torch.float16)

    if (TRANS_B):
        b = torch.randn((SIZE_N, SIZE_K), device='cuda', dtype=torch.float16).T
    else:
        b = torch.randn((SIZE_K, SIZE_N), device='cuda', dtype=torch.float16)

    c = torch.empty((SIZE_M, SIZE_N), device=a.device, dtype=torch.float32)

    ttgir_path = os.path.dirname(__file__) + "/" + TTGIR
    kernel = triton.compile(ttgir_path)
    kernel[(1, 1, 1)](  #
        a.data_ptr(), b.data_ptr(), c.data_ptr(),  #
        SIZE_M, SIZE_N, SIZE_K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0))

    golden = torch.matmul(a, b)
    torch.set_printoptions(profile="full", sci_mode=False)
    assert_close(c, golden, rtol=1e-2, atol=1e-3, check_dtype=False)
