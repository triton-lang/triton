import torch
import triton

N, C, K = 32, 32, 32
H, W = 32, 32
R, S = 3, 3
a = torch.randn(N, C, H, W).cuda()
b = torch.randn(C, R, S, K).cuda()
#c = torch.nn.functional.conv2d(a, b)
c = triton.ops.conv(a, b)
print(c)