import torch
from torch.autograd import Variable

torch.ops.load_library("/home/philippe/Development/triton/build/examples/python/pytorch/libtorch_triton.so")

d = torch.empty(64, 64, 64, 64).uniform_(0, 1).cuda()
w = torch.empty(64, 3, 3, 64).uniform_(0, 1).cuda()
a = torch.ops.triton.conv_forward(d, w)
print(a)
