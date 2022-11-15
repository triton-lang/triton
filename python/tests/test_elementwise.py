import tempfile
from inspect import Parameter, Signature

import _testcapi
import pytest
import torch
from torch.testing import assert_close

import triton
import triton.language as tl

torch_type = {
    "bool": torch.bool,
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
    "where": "where",
}

libdevice = '/usr/local/cuda/nvvm/libdevice/libdevice.10.bc'


def get_tensor(shape, data_type, b_positive=False):
    x = None
    if data_type.startswith('int'):
        x = torch.randint(2**31 - 1, shape, dtype=torch_type[data_type], device='cuda')
    elif data_type.startswith('bool'):
        x = torch.randint(1, shape, dtype=torch_type[data_type], device='cuda')
    else:
        x = torch.randn(shape, dtype=torch_type[data_type], device='cuda')

    if b_positive:
        x = torch.abs(x)

    return x


@pytest.mark.parametrize('expr, output_type, input0_type',
                         [('log', 'float32', 'float32'),
                          ('log', 'float64', 'float64'),
                             ('cos', 'float32', 'float32'),
                             ('cos', 'float64', 'float64'),
                             ('sin', 'float32', 'float32'),
                             ('sin', 'float64', 'float64'),
                             ('sqrt', 'float32', 'float32'),
                             ('sqrt', 'float64', 'float64'),
                             ('abs', 'float32', 'float32'),
                             ('exp', 'float32', 'float32'),
                             ('exp', 'float64', 'float64'),
                             ('sigmoid', 'float32', 'float32'),
                          ])
def test_single_input(expr, output_type, input0_type):
    src = f"""
def kernel(X, Y, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    y = tl.{expr}(x)
    tl.store(Y + tl.arange(0, BLOCK), y)
"""
    fp = tempfile.NamedTemporaryFile(mode='w', suffix=".py")
    fp.write(src)
    fp.flush()

    def kernel(X, Y, BLOCK: tl.constexpr):
        pass
    kernel.__code__ = _testcapi.code_newempty(fp.name, "kernel", 1)
    parameters = []
    parameters.append(Parameter("X", 1))
    parameters.append(Parameter("Y", 1))
    parameters.append(Parameter("BLOCK", 1))
    kernel.__signature__ = Signature(parameters=parameters)
    kernel = triton.jit(kernel)

    shape = (128, )
    # limit the range of integers so that the sum does not overflow
    x = get_tensor(shape, input0_type, expr == 'log' or expr == 'sqrt')
    # triton result
    y = torch.zeros(shape, dtype=torch_type[output_type], device="cuda")
    kernel[(1,)](x, y, BLOCK=shape[0], extern_libs={"libdevice": libdevice})
    # reference result
    y_ref = getattr(torch, torch_ops[expr])(x)
    # compare
    assert_close(y, y_ref)


@pytest.mark.parametrize('expr, output_type, input0_type, input1_type',
                         [('umulhi', 'int32', 'int32', 'int32'),
                          ('cdiv', 'int32', 'int32', 'int32'),
                             ('fdiv', 'float32', 'float32', 'float32'),
                             ('minimum', 'float32', 'float32', 'float32'),
                             ('maximum', 'float32', 'float32', 'float32'),
                          ])
def test_two_input(expr, output_type, input0_type, input1_type):
    src = f"""
def kernel(X0, X1, Y, BLOCK: tl.constexpr):
    x0 = tl.load(X0 + tl.arange(0, BLOCK))
    x1 = tl.load(X1 + tl.arange(0, BLOCK))
    y = tl.{expr}(x0, x1)
    tl.store(Y + tl.arange(0, BLOCK), y)
"""
    fp = tempfile.NamedTemporaryFile(mode='w', suffix=".py")
    fp.write(src)
    fp.flush()

    def kernel(X0, X1, Y, BLOCK: tl.constexpr):
        pass
    kernel.__code__ = _testcapi.code_newempty(fp.name, "kernel", 1)
    parameters = []
    parameters.append(Parameter("X0", 1))
    parameters.append(Parameter("X1", 1))
    parameters.append(Parameter("Y", 1))
    parameters.append(Parameter("BLOCK", 1))
    kernel.__signature__ = Signature(parameters=parameters)
    kernel = triton.jit(kernel)

    shape = (128, )
    # limit the range of integers so that the sum does not overflow
    x0 = get_tensor(shape, input0_type)
    x1 = get_tensor(shape, input1_type)

    # triton result
    y = torch.zeros(shape, dtype=torch_type[output_type], device="cuda")
    kernel[(1,)](x0, x1, y, BLOCK=shape[0], extern_libs={"libdevice": libdevice})
    # reference result

    if expr == "cdiv":
        y_ref = torch.div(x0 + x1 - 1, x1, rounding_mode='trunc')
    elif expr == "umulhi":
        y_ref = ((x0.to(torch.int64) * x1) >> 32).to(torch.int32)
    else:
        y_ref = getattr(torch, torch_ops[expr])(x0, x1)
    # compare
    assert_close(y, y_ref)


@pytest.mark.parametrize('expr, output_type, input0_type, input1_type, input2_type',
                         [('where', "int32", "bool", "int32", "int32"), ])
def test_three_input(expr, output_type, input0_type, input1_type, input2_type):
    src = f"""
def kernel(X0, X1, X2, Y, BLOCK: tl.constexpr):
    x0 = tl.load(X0 + tl.arange(0, BLOCK))
    x1 = tl.load(X1 + tl.arange(0, BLOCK))
    x2 = tl.load(X2 + tl.arange(0, BLOCK))
    y = tl.{expr}(x0, x1, x2)
    tl.store(Y + tl.arange(0, BLOCK), y)
"""
    fp = tempfile.NamedTemporaryFile(mode='w', suffix=".py")
    fp.write(src)
    fp.flush()

    def kernel(X0, X1, X2, Y, BLOCK: tl.constexpr):
        pass
    kernel.__code__ = _testcapi.code_newempty(fp.name, "kernel", 1)
    parameters = []
    parameters.append(Parameter("X0", 1))
    parameters.append(Parameter("X1", 1))
    parameters.append(Parameter("X2", 1))
    parameters.append(Parameter("Y", 1))
    parameters.append(Parameter("BLOCK", 1))
    kernel.__signature__ = Signature(parameters=parameters)
    kernel = triton.jit(kernel)

    shape = (128, )
    # limit the range of integers so that the sum does not overflow
    x0 = get_tensor(shape, input0_type)
    x1 = get_tensor(shape, input1_type)
    x2 = get_tensor(shape, input1_type)

    # triton result
    y = torch.zeros(shape, dtype=torch_type[output_type], device="cuda")
    kernel[(1,)](x0, x1, x2, y, BLOCK=shape[0], extern_libs={"libdevice": libdevice})
    # reference result

    y_ref = getattr(torch, torch_ops[expr])(x0, x1, x2)
    # compare
    assert_close(y, y_ref)
