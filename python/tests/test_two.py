import pytest
import torch
from torch.testing import assert_close

import triton
import triton.language as tl

def patch_kernel(template, to_replace):
    kernel = triton.JITFunction(template.fn)
    for key, value in to_replace.items():
        kernel.src = kernel.src.replace(key, value)
    return kernel

torch_type = {
    "int32": torch.int32,
    "float32": torch.float32,
    "float64": torch.float64
}

torch_ops = {
    "log": "log",
    "cos": "cos",
    "sin": "sin",
    "sqrt": "sqrt",
    "abs": "abs",
    "exp": "exp",
    "sigmoid": "sigmoid",
    "umulhi": None,
    "cdiv": None,
    "fdiv": "div",
    "minimum": "minimum",
    "maximum": "maximum",
}

libdevice = '/usr/local/cuda/nvvm/libdevice/libdevice.10.bc'


def get_tensor(shape, data_type, b_positive=False):
    x = None
    if data_type.startswith('int'):
        x = torch.randint(2**31-1, shape, dtype=torch_type[data_type], device='cuda')
    else:
        x = torch.randn(shape, dtype=torch_type[data_type], device='cuda')
    
    if b_positive:
        x = torch.abs(x)
    
    return x

def test_two_input(expr, output_type, input0_type, input1_type):
    @triton.jit
    def kernel(X0, X1, Y, BLOCK: tl.constexpr):
        x0 = tl.load(X0 + tl.arange(0, BLOCK))
        x1 = tl.load(X1 + tl.arange(0, BLOCK))
        y = GENERATE_TEST_HERE
        tl.store(Y + tl.arange(0, BLOCK), y)
    
    shape = (128, )
    # limit the range of integers so that the sum does not overflow
    x0 = get_tensor(shape, input0_type)
    x1 = get_tensor(shape, input1_type)

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': "tl." + expr + "(x0, x1)"})
    
    # triton result
    y = torch.zeros(shape, dtype=x0.dtype, device="cuda")
    kernel[(1,)](x0, x1, y, BLOCK=shape[0], extern_libs={"libdevice": libdevice})
    # reference result
    
    if expr == "cdiv":
        y_ref = (x0 + x1 -1 ) // x1
    elif expr == "umulhi":
        y_ref = (x0 * x1) >> 32
    else:
        y_ref = getattr(torch, torch_ops[expr])(x0, x1)
    import pdb; pdb.set_trace()
    # compare
    assert_close(y, y_ref)
    print("success")

test_two_input("umulhi", "int32", "int32", "int32")
