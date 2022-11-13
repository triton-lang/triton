import triton
import triton._C.libtriton.triton as _triton
import re
import torch

x = torch.randn((32, 32), device="cuda")
test = triton.compile("./reduce.ttgir")
test[(1,1,1)](x, x)

# triton.compile("./reduce.ttgir", signature=)