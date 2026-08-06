import importlib
import multiprocessing
import os
import re
import tempfile
import numpy as np
import sys
import torch
import triton
import triton.language as tl
from triton import knobs
from typing import Optional, Set, Union
from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar
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
_COMPILE_WARMUP_ACTIVE = ContextVar("triton_compile_warmup_active", default=False)
_PROCESS_POOL = None


def is_interpreter():
    return os.environ.get('TRITON_INTERPRET', '0') == '1'


def is_compile_warmup():
    return _COMPILE_WARMUP_ACTIVE.get()


def random_int(low, high, *, warmup_value=None, **kwargs):
    if is_compile_warmup():
        return low if warmup_value is None else warmup_value
    return int(torch.randint(low, high, size=(), **kwargs).item())


def random_float(*, warmup_value=0.5, **kwargs):
    if is_compile_warmup():
        return warmup_value
    return float(torch.rand((), **kwargs).item())


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
    return is_cuda() and torch.cuda.get_device_capability()[0] in [10, 11]


def is_blackwell_ultra():
    return is_cuda() and torch.cuda.get_device_capability()[0:2] == (10, 3)


def is_rubin():
    return is_cuda() and torch.cuda.get_device_capability()[0:2] == (10, 7)


def is_hopper_or_newer():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_hopper():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 9


def is_sm12x():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 12


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


def is_hip_rdna3():
    target = get_current_target()
    return target is not None and target.backend == 'hip' and ('gfx110' in target.arch or 'gfx115' in target.arch)


def is_hip_rdna4m():
    target = get_current_target()
    return target is not None and target.backend == 'hip' and 'gfx117' in target.arch


def is_hip_rdna4():
    target = get_current_target()
    # check for gfx120 instead of gfx12, to avoid matching gfx1250
    return target is not None and target.backend == 'hip' and 'gfx120' in target.arch


def is_hip_gfx1250():
    target = get_current_target()
    return target is not None and target.backend == 'hip' and 'gfx1250' in target.arch


def is_hip_cdna3_or_newer():
    return is_hip_cdna3() or is_hip_cdna4()


def is_hip_cdna():
    return is_hip_cdna2() or is_hip_cdna3() or is_hip_cdna4()


def is_hip_rdna():
    return is_hip_rdna3() or is_hip_rdna4m() or is_hip_rdna4()


def get_hip_lds_size():
    return 163840 if is_hip_cdna4() else 65536


def is_xpu():
    target = get_current_target()
    return False if target is None else target.backend == "xpu"


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


def supports_ws():
    if is_interpreter():
        return True
    if not is_cuda():
        return False
    return torch.cuda.get_device_capability()[0] >= 9


def supports_clc():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 10


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


@dataclass
class ProcessResult:
    exc: None | BaseException
    driver_stderr_output: str


def _call_in_process(client_fn, args, kwargs, env, stderr_file, compilation_listener):
    if env is not None:
        os.environ.update(env)

    # Capture driver/runtime writes to stderr that bypass Python's file objects.
    with open(stderr_file, "w+b") as tmp_stderr:
        saved_stderr_fd = os.dup(2)
        os.dup2(tmp_stderr.fileno(), 2)
        exc = None

        previous_listener = knobs.compilation.listener
        knobs.compilation.listener = compilation_listener
        try:
            client_fn(*args, **kwargs)
            # Raise any CUDA errors
            torch.cuda.synchronize()
        except Exception as e:
            exc = e
        finally:
            knobs.compilation.listener = previous_listener
            sys.stderr.flush()
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stderr_fd)
    return exc


def _run_in_process_worker(client_fn, result_pipe, args, kwargs, env, stderr_file, compilation_listener):
    result_pipe.send(_call_in_process(client_fn, args, kwargs, env, stderr_file, compilation_listener))


def _run_in_replenishing_process_worker(task_pipe, preload_module, triton_key):
    importlib.import_module(preload_module)
    # The source tree is fixed for the pool lifetime; avoid rehashing it in each worker.
    triton.runtime.cache.triton_key = lambda: triton_key
    torch.cuda.synchronize()
    while True:
        try:
            client_fn, args, kwargs, env, stderr_file, compilation_listener = task_pipe.recv()
        except EOFError:
            return
        previous_environment = dict(os.environ)
        try:
            with knobs.compilation.scope(), knobs.runtime.scope(), knobs.cache.scope():
                exc = _call_in_process(client_fn, args, kwargs, env, stderr_file, compilation_listener)
        finally:
            os.environ.clear()
            os.environ.update(previous_environment)
        task_pipe.send(exc)
        if exc is not None:
            return


class ReplenishingProcessPool:

    def __init__(self, preload_module):
        self.preload_module = preload_module
        self.triton_key = triton.runtime.cache.triton_key()
        self.ctx = multiprocessing.get_context("forkserver")
        self.worker = None
        self.spare = None

    def _start_process(self):
        task_pipe, child_pipe = self.ctx.Pipe()
        process = self.ctx.Process(
            target=_run_in_replenishing_process_worker,
            args=(child_pipe, self.preload_module, self.triton_key),
            daemon=True,
        )
        process.start()
        child_pipe.close()
        return process, task_pipe

    def start(self):
        if self.worker is None:
            self.worker = self._start_process()
            self.spare = self._start_process()

    def close(self):
        for current in (self.worker, self.spare):
            if current is None:
                continue
            process, task_pipe = current
            task_pipe.close()
            process.terminate()
            process.join()
        self.worker = None
        self.spare = None

    def run(self, client_fn, args=(), kwargs=None, env=None):
        if kwargs is None:
            kwargs = {}
        self.start()
        process, task_pipe = self.worker
        with tempfile.TemporaryDirectory() as tmpdir:
            stderr_file = os.path.join(tmpdir, "err.log")
            task_pipe.send((client_fn, args, kwargs, env, stderr_file, knobs.compilation.listener))
            try:
                exc = task_pipe.recv()
            except EOFError:
                process.join()
                with open(stderr_file, "r") as f:
                    stderr = f.read()
                print(stderr, file=sys.stderr)
                raise RuntimeError(
                    f"child process exited with code {process.exitcode} without returning a result") from None
            with open(stderr_file, "r") as f:
                stderr = f.read()
        if exc is not None:
            process.join()
            task_pipe.close()
            self.worker = self.spare
            self.spare = self._start_process()
        return ProcessResult(exc, stderr)


@contextmanager
def use_process_pool(preload_module):
    global _PROCESS_POOL

    pool = ReplenishingProcessPool(preload_module)
    pool.start()
    _PROCESS_POOL = pool
    try:
        yield pool
    finally:
        pool.close()
        _PROCESS_POOL = None


def run_in_process(client_fn, args=(), kwargs=None, env=None):
    if is_compile_warmup() or os.environ.get("DISABLE_SUBPROCESS"):
        return client_fn(*args, **(kwargs or {}))
    if _PROCESS_POOL is not None:
        return _PROCESS_POOL.run(client_fn, args, kwargs, env)
    if kwargs is None:
        kwargs = {}

    ctx = multiprocessing.get_context("forkserver")
    result_pipe, child_pipe = ctx.Pipe(duplex=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        stderr_file = os.path.join(tmpdir, "err.log")
        process = ctx.Process(
            target=_run_in_process_worker,
            args=(client_fn, child_pipe, args, kwargs, env, stderr_file, knobs.compilation.listener),
        )
        process.start()
        child_pipe.close()
        timeout = os.environ.get("TRITON_TEST_PROCESS_TIMEOUT")
        timeout = None if timeout is None else float(timeout)
        if not result_pipe.poll(timeout):
            process.kill()
            process.join()
            result_pipe.close()
            raise TimeoutError(f"test subprocess {client_fn.__name__} exceeded its {timeout}-second timeout")
        try:
            exc = result_pipe.recv()
        except EOFError:
            process.join()
            with open(stderr_file, "r") as f:
                stderr = f.read()
            print(stderr, file=sys.stderr)
            raise RuntimeError(
                f"child process exited with code {process.exitcode} without returning a result") from None
        finally:
            result_pipe.close()
        process.join(timeout)
        if process.is_alive():
            process.kill()
            process.join()
            raise TimeoutError(f"test subprocess {client_fn.__name__} exceeded its {timeout}-second timeout")
        with open(stderr_file, "r") as f:
            stderr = f.read()
    return ProcessResult(exc, stderr)


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
