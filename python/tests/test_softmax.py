import torch
import triton

def test_op(M = 1024, N = 1024, dtype = torch.float32):
    x = torch.randn(M, N, dtype=dtype, device='cuda')
    th_y = torch.softmax(x, dim=-1)
    tt_y = triton.ops.softmax(x)
    assert torch.allclose(tt_y, th_y)