import torch
import triton


def test_op():
    torch.manual_seed(0)
    DTYPE = torch.float16
    N, H, W, CI, CO, R, S = 1, 56, 56, 1024, 1024, 3, 3
    pad, stride, = (1, 1), (1, 1)
    dilation = (1, 1)
    a = torch.rand((N , CI, H, W ), dtype=DTYPE, device='cuda')  / CI**.5
    b = torch.rand((CI, R , S, CO), dtype=DTYPE, device='cuda')  / CI**.5
    th_c = torch.nn.functional.conv2d(a, b.permute(3,0,1,2), None, stride, pad, dilation)
    tt_c = triton.ops.conv(a, b, pad, stride)
    rtol, atol = {torch.float32: (1e-4, 1e-5),
                  torch.float16: (1e-2, 1e-3)}[DTYPE]
    assert torch.allclose(tt_c, th_c, atol=atol, rtol=rtol)