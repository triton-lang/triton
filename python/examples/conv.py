import torch
import triton

N, C, K = 32, 8, 32
H, W = 16, 16
R, S = 3, 3
torch.manual_seed(0)
a = torch.randn(N, C, H, W).cuda()
b = torch.ones(C, R, S, K).cuda()

rc = torch.nn.functional.conv2d(a, b.permute(3, 0, 1, 2))
tc = triton.ops.conv(a, b)
print((rc - tc).abs().max())
#print((rc[:30,:30,:,:] - tc[:30, :30, :, :]).abs().max())
#print(tc[31, 31,:,:])