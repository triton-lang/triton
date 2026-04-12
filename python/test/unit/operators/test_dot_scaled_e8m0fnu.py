import pytest
import torch
import triton
import triton.language as tl
import numpy as np

def test_dot_scaled_e8m0fnu():
    @triton.jit
    def kernel(lhs_ptr, lhs_scale_ptr, rhs_ptr, rhs_scale_ptr, out_ptr,
               M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        lhs = tl.load(lhs_ptr + tl.arange(0, M * K))
        lhs = tl.reshape(lhs, (M, K))
        lhs_scale = tl.load(lhs_scale_ptr + tl.arange(0, M * (K // 32)))
        lhs_scale = tl.reshape(lhs_scale, (M, K // 32))
        
        rhs = tl.load(rhs_ptr + tl.arange(0, K * N))
        rhs = tl.reshape(rhs, (K, N))
        rhs_scale = tl.load(rhs_scale_ptr + tl.arange(0, N * (K // 32)))
        rhs_scale = tl.reshape(rhs_scale, (N, K // 32))
        
        # Verify the fix: bitcast lhs_scale and rhs_scale to uint8 internally
        out = tl.dot_scaled(lhs, lhs_scale, "e4m3", rhs, rhs_scale, "e4m3", acc=None, out_dtype=tl.float32)
        tl.store(out_ptr + tl.arange(0, M * N), tl.reshape(out, (M * N,)))

    M, N, K = 128, 128, 128
    lhs_scale = torch.randint(0, 255, (M, K // 32), dtype=torch.uint8, device='cuda')
    rhs_scale = torch.randint(0, 255, (N, K // 32), dtype=torch.uint8, device='cuda')


