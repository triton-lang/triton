import os
import re
import numpy as np
import torch
import triton
import triton.language as tl
from triton import knobs
from typing import Optional, Set, Union
import pytest

from numpy.random import RandomState
from triton.runtime.jit import TensorWrapper, reinterpret, type_canonicalisation_dict

int_dtypes = ['int8', 'int16', 'int32', 'int64']
uint_dtypes = ['uint8', 'uint16', 'uint32', 'uint64']
integral_dtypes = int_dtypes + uint_dtypes
float_dtypes = ['float16', 'float32', 'float64']
float_dtypes_with_bfloat16 = float_dtypes + ['bfloat16']
dtypes = integral_dtypes + float_dtypes
dtypes_with_bfloat16 = dtypes + ['bfloat16']
torch_float8_dtypes = ['float8_e4m3fn', 'float8_e5m2']
torch_dtypes = ['bool'] + int_dtypes + ['uint8'] + float_dtypes + ['bfloat16']
tma_dtypes = sorted(set(dtypes_with_bfloat16) - {"int64", "uint64", "float64"})


def is_interpreter():
    return os.environ.get('TRITON_INTERPRET', '0') == '1'


def get_current_target():
    if is_interpreter():
        return None
    return triton.runtime.driver.active.get_current_target()


def is_cuda():
    target = get_current_target()
    return False if target is None else target.backend == "cuda"


def is_ampere_or_newer():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 8


def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


def is_hopper_or_newer():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_hopper():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 9


def is_hip():
    target = get_current_target()
    return False if target is None else target.backend == "hip"


def is_hip_cdna2():
    target = get_current_target()
    return target is not None and target.backend == 'hip' and target.arch == 'gfx90a'


def is_hip_cdna3():
    target = get_current_target()
    return target is not None and target.backend == 'hip' and target.arch == 'gfx942'


def is_hip_cdna4():
    target = get_current_target()
    return target is not None and target.backend == 'hip' and target.arch == 'gfx950'


def is_hip_gfx11():
    target = get_current_target()
    return target is not None and target.backend == 'hip' and 'gfx11' in target.arch


def is_hip_gfx12():
    target = get_current_target()
    return target is not None and target.backend == 'hip' and 'gfx12' in target.arch


def is_hip_cdna():
    return is_hip_cdna2() or is_hip_cdna3() or is_hip_cdna4()


def get_hip_lds_size():
    return 163840 if is_hip_cdna4() else 65536


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


def str_to_triton_dtype(x: str) -> tl.dtype:
    return tl.str_to_ty(type_canonicalisation_dict[x], None)


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


def supports_tma(byval_only=False):
    if is_interpreter():
        return True
    if not is_cuda():
        return False
    cuda_version = knobs.nvidia.ptxas.version
    min_cuda_version = (12, 0) if byval_only else (12, 3)
    cuda_version_tuple = tuple(map(int, cuda_version.split(".")))
    assert len(cuda_version_tuple) == 2, cuda_version_tuple
    return torch.cuda.get_device_capability()[0] >= 9 and cuda_version_tuple >= min_cuda_version


def tma_skip_msg(byval_only=False):
    if byval_only:
        return "Requires __grid_constant__ TMA support (NVIDIA Hopper or higher, CUDA 12.0 or higher)"
    else:
        return "Requires advanced TMA support (NVIDIA Hopper or higher, CUDA 12.3 or higher)"


requires_tma = pytest.mark.skipif(not supports_tma(), reason=tma_skip_msg())


def default_alloc_fn(size: int, align: int, _):
    return torch.empty(size, dtype=torch.int8, device="cuda")


def unwrap_tensor(t: Union[torch.Tensor, triton.runtime.jit.TensorWrapper]) -> torch.Tensor:
    if isinstance(t, triton.runtime.jit.TensorWrapper):
        return t.base
    return t


def _fresh_knobs_impl(skipped_attr: Optional[Set[str]] = None):
    from triton import knobs

    if skipped_attr is None:
        skipped_attr = set()

    monkeypatch = pytest.MonkeyPatch()

    knobs_map = {
        name: knobset
        for name, knobset in knobs.__dict__.items()
        if isinstance(knobset, knobs.base_knobs) and knobset != knobs.base_knobs and name not in skipped_attr
    }

    # We store which variables we need to unset below in finally because
    # monkeypatch doesn't appear to reset variables that were never set
    # before the monkeypatch.delenv call below.
    env_to_unset = []
    prev_propagate_env = knobs.propagate_env

    def fresh_function():
        nonlocal env_to_unset
        for name, knobset in knobs_map.items():
            setattr(knobs, name, knobset.copy().reset())
            for knob in knobset.knob_descriptors.values():
                if knob.key in os.environ:
                    monkeypatch.delenv(knob.key, raising=False)
                else:
                    env_to_unset.append(knob.key)
        knobs.propagate_env = True
        return knobs

    def reset_function():
        for name, knobset in knobs_map.items():
            setattr(knobs, name, knobset)
        # `undo` should be placed before `del os.environ`
        # Otherwise, it may restore environment variables that monkeypatch deleted
        monkeypatch.undo()
        for k in env_to_unset:
            if k in os.environ:
                del os.environ[k]
        knobs.propagate_env = prev_propagate_env

    return fresh_function, reset_function
