import os
import re
import numpy as np
import torch
import triton
import triton.language as tl
import pytest

from numpy.random import RandomState
from typing import Optional, Union
from triton.runtime.jit import TensorWrapper, reinterpret

int_dtypes = ['int8', 'int16', 'int32', 'int64']
uint_dtypes = ['uint8', 'uint16', 'uint32', 'uint64']
integral_dtypes = int_dtypes + uint_dtypes
float_dtypes = ['float16', 'float32', 'float64']
float_dtypes_with_bfloat16 = float_dtypes + ['bfloat16']
dtypes = integral_dtypes + float_dtypes
dtypes_with_bfloat16 = dtypes + ['bfloat16']
torch_float8_dtypes = ['float8_e4m3fn', 'float8_e5m2']
torch_dtypes = ['bool'] + int_dtypes + ['uint8'] + float_dtypes + ['bfloat16']


def is_interpreter():
    return os.environ.get('TRITON_INTERPRET', '0') == '1'


def get_current_target():
    if is_interpreter():
        return None
    return triton.runtime.driver.active.get_current_target()


def is_cuda():
    target = get_current_target()
    return False if target is None else target.backend == "cuda"


def is_hopper():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_hip():
    target = get_current_target()
    return False if target is None else target.backend == "hip"


def is_hip_mi200():
    target = get_current_target()
    return target.backend == 'hip' and target.arch == 'gfx90a'


def is_hip_mi300():
    target = get_current_target()
    return target.backend == 'hip' and target.arch in ('gfx940', 'gfx941', 'gfx942')


def is_hip_cdna():
    return is_hip_mi200() or is_hip_mi300()


def is_xpu():
    target = get_current_target()
    return False if target is None else target.backend == "xpu"


def get_arch():
    target = get_current_target()
    return "" if target is None else str(target.arch)


def numpy_random(shape, dtype_str, rs: Optional[RandomState] = None, low=None, high=None):
    """
    Override `rs` if you're calling this function twice and don't want the same
    result for both calls.
    """
    if isinstance(shape, int):
        shape = (shape, )
    if rs is None:
        rs = RandomState(seed=17)
    if dtype_str in int_dtypes + uint_dtypes:
        iinfo = np.iinfo(getattr(np, dtype_str))
        low = iinfo.min if low is None else max(low, iinfo.min)
        high = iinfo.max if high is None else min(high, iinfo.max)
        dtype = getattr(np, dtype_str)
        x = rs.randint(low, high, shape, dtype=dtype)
        x[x == 0] = 1  # Workaround. Never return zero so tests of division don't error out.
        return x
    elif dtype_str and 'float8' in dtype_str:
        x = rs.randint(20, 40, shape, dtype=np.int8)
        return x
    elif dtype_str in float_dtypes:
        return rs.normal(0, 1, shape).astype(dtype_str)
    elif dtype_str == 'bfloat16':
        return (rs.normal(0, 1, shape).astype('float32').view('uint32') & np.uint32(0xffff0000)).view('float32')
    elif dtype_str in ['bool', 'int1', 'bool_']:
        return rs.normal(0, 1, shape) > 0.0
    else:
        raise RuntimeError(f'Unknown dtype {dtype_str}')


def to_triton(x: np.ndarray, device, dst_type=None) -> Union[TensorWrapper, torch.Tensor]:
    '''
    Note: We need dst_type because the type of x can be different from dst_type.
          For example: x is of type `float32`, dst_type is `bfloat16`.
          If dst_type is None, we infer dst_type from x.
    '''
    t = x.dtype.name
    if t in uint_dtypes:
        signed_type_name = t.lstrip('u')  # e.g. "uint16" -> "int16"
        x_signed = x.astype(getattr(np, signed_type_name))
        return reinterpret(torch.tensor(x_signed, device=device), getattr(tl, t))
    else:
        if dst_type and 'float8' in dst_type:
            return reinterpret(torch.tensor(x, device=device), getattr(tl, dst_type))
        if t == 'float32' and dst_type == 'bfloat16':
            return torch.tensor(x, device=device).bfloat16()
        return torch.tensor(x, device=device)


def torch_dtype_name(dtype) -> str:
    if isinstance(dtype, triton.language.dtype):
        return dtype.name
    elif isinstance(dtype, torch.dtype):
        # 'torch.int64' -> 'int64'
        m = re.match(r'^torch\.(\w+)$', str(dtype))
        return m.group(1)
    else:
        raise TypeError(f'not a triton or torch dtype: {type(dtype)}')


def to_numpy(x):
    if isinstance(x, TensorWrapper):
        return x.base.cpu().numpy().astype(getattr(np, torch_dtype_name(x.dtype)))
    elif isinstance(x, torch.Tensor):
        if x.dtype is torch.bfloat16:
            return x.cpu().float().numpy()
        return x.cpu().numpy()
    else:
        raise ValueError(f"Not a triton-compatible tensor: {x}")


def supports_tma():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


requires_tma = pytest.mark.skipif(not supports_tma(), reason="Requires TMA support (NVIDIA Hopper or higher)")
