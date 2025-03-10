import importlib.util
import itertools
import os
import shutil
import pathlib

import pytest
import torch

import triton
import triton.language as tl
from triton.runtime.jit import JITFunction
from triton._internal_testing import is_hip


@triton.jit
def function_0(i):
    return i + 1


@triton.jit
def function_1(i):
    i = i + 1
    cond: tl.constexpr = True
    if cond:
        FN: tl.constexpr = function_2
    else:
        FN: tl.constexpr = function_0
    return FN(i)


@triton.jit
def function_2(i):
    i = i + 1
    return i


@triton.jit
def combine_fn(a, b):
    return COMBINE_OP  # noqa: F821


@triton.jit
def kernel(X, i, BLOCK: tl.constexpr):
    i = i + 1
    i = function_1(i)
    tl.store(X, i)


@triton.jit(do_not_specialize=["i"])
def kernel_nospec(X, i, BLOCK: tl.constexpr):
    i = i + 1
    i = function_1(i)
    tl.store(X, i)


@triton.jit(do_not_specialize_on_alignment=["i"])
def kernel_nospec_on_alignment(X, i, BLOCK: tl.constexpr):
    i = i + 1
    i = function_1(i)
    tl.store(X, i)


@triton.jit
def kernel_with_combine_fn(X, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    i = REDUCE_OR_SCAN(i, 0, combine_fn)  # noqa: F821
    tl.store(X, i)


def apply_src_change(target, old, new, to_modify):
    kernel.hash = None
    function_0.hash = None
    function_1.hash = None
    function_2.hash = None
    to_modify._unsafe_update_src(to_modify.src.replace(old, new))
    ret = target.cache_key
    to_modify._unsafe_update_src(to_modify.src.replace(new, old))
    return ret


def test_nochange():
    baseline = kernel.cache_key
    updated = apply_src_change(kernel, 'i + 1', 'i + 1', function_1)
    assert baseline == updated


def test_toplevel_change():
    baseline = kernel.cache_key
    updated = apply_src_change(kernel, 'i + 1', 'i + 2', function_1)
    assert baseline != updated


def test_nested1_change():
    baseline = kernel.cache_key
    updated = apply_src_change(kernel, 'i + 1', 'i + 2', function_2)
    assert baseline != updated


def test_nested2_change():
    baseline = kernel.cache_key
    updated = apply_src_change(kernel, 'i + 1', 'i + 2', function_0)
    assert baseline != updated


def test_combine_fn_change():
    # Test that tl.reduce and associative_scan calls include
    # the combine_fn in the hash

    orig_combine_fn_src = combine_fn.src
    orig_kernel_src = kernel_with_combine_fn.src
    seen_keys = set()

    for reduce_or_scan, combine_op in itertools.product(
        ["tl.reduce", "tl.associative_scan"],
        ["a + b", "a * b"],
    ):
        combine_fn._unsafe_update_src(orig_combine_fn_src.replace("COMBINE_OP", combine_op))
        kernel_with_combine_fn._unsafe_update_src(orig_kernel_src.replace("REDUCE_OR_SCAN", reduce_or_scan))
        try:
            key = kernel_with_combine_fn.cache_key
        finally:
            combine_fn._unsafe_update_src(orig_combine_fn_src)
            kernel_with_combine_fn._unsafe_update_src(orig_kernel_src)

        assert key not in seen_keys
        seen_keys.add(key)


def write_and_load_module(temp_file: pathlib.Path, code, num_extra_lines):
    temp_file.write_text(('# extra line\n' * num_extra_lines) + code)
    spec = importlib.util.spec_from_file_location("module.name", str(temp_file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_changed_line_numbers_invalidate_cache(tmp_path: pathlib.Path):
    from textwrap import dedent
    code = dedent("""
        import triton
        @triton.jit
        def test_kernel(i):
            i = i + 1
    """)
    temp_file0 = tmp_path / "test_changed_line_numbers_invalidate_cache0.py"
    orig_mod = write_and_load_module(temp_file0, code, 0)
    orig_cache_key = orig_mod.test_kernel.cache_key

    temp_file1 = tmp_path / "test_changed_line_numbers_invalidate_cache1.py"
    updated_mod = write_and_load_module(temp_file1, code, 1)
    updated_cache_key = updated_mod.test_kernel.cache_key
    assert orig_cache_key != updated_cache_key


def test_reuse(device, fresh_triton_cache):
    counter = 0

    def inc_counter(*args, **kwargs):
        nonlocal counter
        counter += 1

    JITFunction.cache_hook = inc_counter
    x = torch.empty(1, dtype=torch.int32, device=device)
    for i in range(10):
        kernel[(1, )](x, 1, BLOCK=1024)
    assert counter == 1


@pytest.mark.parametrize('mode', ['enable', 'disable', 'disable_on_alignment'])
def test_specialize(mode, device, fresh_triton_cache):
    counter = 0

    def inc_counter(*args, **kwargs):
        nonlocal counter
        counter += 1

    JITFunction.cache_hook = inc_counter
    x = torch.empty(1, dtype=torch.int32, device=device)
    function = {'enable': kernel, 'disable': kernel_nospec, 'disable_on_alignment': kernel_nospec_on_alignment}[mode]
    target = {'enable': 3, 'disable': 1, 'disable_on_alignment': 2}[mode]
    for i in [1, 2, 4, 8, 16, 32]:
        function[(1, )](x, i, BLOCK=512)
    assert counter == target


def test_annotation(device):

    @triton.jit
    def kernel(X, i: tl.int32):
        tl.store(X, i)

    x = torch.empty(1, dtype=torch.int32, device=device)

    device = getattr(torch, device).current_device()
    kernel[(1, )](x, 1)
    kernel[(1, )](x, 8)
    kernel[(1, )](x, 16)
    kernel[(1, )](x, 17)
    assert len(kernel.device_caches[device][0]) == 3


GLOBAL_DEFAULT_ARG = 1


def test_kernel_default_arg(device):
    global GLOBAL_DEFAULT_ARG

    @triton.jit
    def kernel(X, i: tl.constexpr = GLOBAL_DEFAULT_ARG):
        tl.store(X, i)

    x = torch.empty(1, dtype=torch.int32, device=device)
    kernel[(1, )](x)
    assert x == torch.ones_like(x)

    # Changing the global variable should not change the default argument in
    # `kernel`.  That value gets set at the time the function is declared.
    GLOBAL_DEFAULT_ARG = 2
    kernel[(1, )](x)
    assert x == torch.ones_like(x)

    device = getattr(torch, device).current_device()
    assert len(kernel.device_caches[device][0]) == 1


GLOBAL_VAR = tl.constexpr(1)


def test_kernel_global_var_change(device):
    global GLOBAL_VAR

    @triton.jit
    def kernel(X):
        tl.store(X, GLOBAL_VAR)

    x = torch.empty(1, dtype=torch.int32, device=device)
    kernel[(1, )](x)
    assert x == torch.ones_like(x)

    GLOBAL_VAR = 2
    with pytest.raises(RuntimeError) as e:
        kernel[(1, )](x)

    assert "global variable" in str(e.value).lower()


GLOBAL = 42  # noqa


def test_local_shadows_global():
    global GLOBAL

    @triton.jit
    def kernel():
        _, GLOBAL = 0, 0  # noqa
        a = GLOBAL  # noqa

    # No error because the `GLOBAL` we're modifying is not the same `GLOBAL` as
    # inside the kernel.
    GLOBAL = 42
    kernel[(1, )]()
    GLOBAL = 43
    kernel[(1, )]()


CONSTEXPR_GLOBAL = tl.constexpr(42)


def test_local_does_not_shadow_global():
    global CONSTEXPR_GLOBAL

    @triton.jit
    def kernel():
        a = CONSTEXPR_GLOBAL  # noqa
        _, CONSTEXPR_GLOBAL = 0, 0  # noqa

    CONSTEXPR_GLOBAL = tl.constexpr(42)
    kernel[(1, )]()
    CONSTEXPR_GLOBAL = tl.constexpr(43)

    # Error because the `CONSTEXPR_GLOBAL` we're modifying is the same
    # `CONSTEXPR_GLOBAL` that's read inside `kernel`.  (Alternatively, we could
    # make this kernel an error altogether, as it is if it's a pure Python
    # function -- the fact that we store to `CONSTEXPR_GLOBAL` inside the kernel
    # makes the first read a read of the local variable, which doesn't exist
    # yet.)
    with pytest.raises(RuntimeError):
        kernel[(1, )]()


CONFLICTING_GLOBAL = tl.constexpr(0)


@triton.jit
def conflicting_global_inner():
    a = CONFLICTING_GLOBAL  # noqa


def test_conflicting_global_in_inner_function():
    global CONFLICTING_GLOBAL

    @triton.jit
    def kernel1():
        a = CONFLICTING_GLOBAL  # noqa
        conflicting_global_inner()

    @triton.jit
    def kernel2():
        a = CONFLICTING_GLOBAL  #noqa
        conflicting_global_inner()

    kernel1[(1, )]()

    # This should be an error because kernel2 calls conflicting_global_inner,
    # which saw a value for 42 for the global when it was first compiled.
    CONFLICTING_GLOBAL = 1

    with pytest.raises(RuntimeError) as e:
        kernel2[(1, )]()

    assert "Global variable CONFLICTING_GLOBAL has value" in str(e.value)


def test_use_builtin():

    @triton.jit
    def kernel():
        a = float(0)  # noqa

    # No error about the value of `float` changing.
    kernel[(1, )]()
    kernel[(1, )]()


def test_no_cache_module_as_global():

    @triton.jit
    def kernel():
        tl.arange(0, 16)

    kernel[(1, )]()
    # `tl` should not be entered into used_global_vals
    assert not kernel.used_global_vals


BUILTIN_AS_GLOBAL = tl.int32


def test_cache_builtin_as_global():
    global BUILTIN_AS_GLOBAL

    @triton.jit
    def kernel():
        x = BUILTIN_AS_GLOBAL  # noqa

    kernel[(1, )]()

    BUILTIN_AS_GLOBAL = tl.int64
    with pytest.raises(RuntimeError) as e:
        kernel[(1, )]()

    assert "global variable" in str(e.value).lower()


@triton.jit
def no_cache_callable_inner():
    pass


def test_no_cache_callable():

    @triton.jit
    def kernel():
        no_cache_callable_inner()

    kernel[(1, )]()
    # `no_cache_callable_inner` should not be entered into used_global_vals.
    assert not kernel.used_global_vals


def test_jit_warmup_cache(device) -> None:

    @triton.jit
    def kernel_add(a, b, o, N: tl.constexpr):
        idx = tl.arange(0, N)
        tl.store(o + idx, tl.load(a + idx) + tl.load(b + idx))

    args = [
        torch.randn(32, dtype=torch.float32, device=device),
        torch.randn(32, dtype=torch.float32, device=device),
        torch.randn(32, dtype=torch.float32, device=device),
        32,
    ]
    device = getattr(torch, device).current_device()
    assert len(kernel_add.device_caches[device][0]) == 0
    kernel_add.warmup(torch.float32, torch.float32, torch.float32, 32, grid=(1, ))
    assert len(kernel_add.device_caches[device][0]) == 1
    kernel_add.warmup(*args, grid=(1, ))
    assert len(kernel_add.device_caches[device][0]) == 1
    kernel_add.warmup(*args, grid=(1, ))
    assert len(kernel_add.device_caches[device][0]) == 1


def test_jit_debug(device) -> None:

    @triton.jit
    def kernel(tmp):
        tl.device_assert(tl.load(tmp) == 1, "tmp == 1")

    device = getattr(torch, device).current_device()
    tmp = torch.tensor([1], dtype=torch.int32, device=device)
    assert len(kernel.device_caches[device][0]) == 0
    kernel[(1, )](tmp, debug=False)
    assert len(kernel.device_caches[device][0]) == 1
    kernel[(1, )](tmp, debug=True)
    assert len(kernel.device_caches[device][0]) == 2
    bins = list(kernel.device_caches[device][0].values())
    assert bins[0].asm['ttir'] != bins[1].asm['ttir']


@triton.jit
def add_fn(a, b, o, N: tl.constexpr):
    idx = tl.arange(0, N)
    tl.store(o + idx, tl.load(a + idx) + tl.load(b + idx))


def test_jit_noinline(device) -> None:

    @triton.jit
    def kernel_add_device(a, b, o, N: tl.constexpr):
        add_fn(a, b, o, N)

    device = getattr(torch, device).current_device()
    assert len(kernel_add_device.device_caches[device][0]) == 0
    kernel_add_device.warmup(torch.float32, torch.float32, torch.float32, 32, grid=(1, ))
    assert len(kernel_add_device.device_caches[device][0]) == 1
    bins = list(kernel_add_device.device_caches[device][0].values())
    inline_ttir = bins[0].asm['ttir']
    add_fn.noinline = True
    add_fn.hash = None
    kernel_add_device.hash = None
    kernel_add_device.device_caches[device][0].clear()
    kernel_add_device.warmup(torch.float32, torch.float32, torch.float32, 32, grid=(1, ))
    assert len(kernel_add_device.device_caches[device][0]) == 1
    bins = list(kernel_add_device.device_caches[device][0].values())
    noinline_ttir = bins[0].asm['ttir']
    assert inline_ttir != noinline_ttir


def test_memory_leak() -> None:

    @triton.jit
    def kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
        xnumel = 10
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)


def test_preload(device, fresh_triton_cache) -> None:

    @triton.jit
    def kernel_add(a, b, o, N: tl.constexpr, type: tl.constexpr):
        idx = tl.arange(0, N)
        tl.device_assert(idx < 32, "idx < 32")
        tl.store(o + idx, tl.load(a + idx) + tl.load(b + idx))

    @triton.jit
    def kernel_sub(a, b, o, N: tl.constexpr, type: tl.constexpr):
        idx = tl.arange(0, N)
        tl.device_assert(idx < 32, "idx < 32")
        tl.store(o + idx, tl.load(a + idx) - tl.load(b + idx))

    device = getattr(torch, device).current_device()

    # get the serialized specialization data
    specialization_data = None

    def cache_hook(*args, **kwargs):
        nonlocal specialization_data
        specialization_data = kwargs["compile"]["specialization_data"]

    JITFunction.cache_hook = cache_hook
    pre_compile = kernel_add.warmup(torch.float32, torch.float32, torch.float32, 32, tl.float32, grid=(1, ))
    hash = pre_compile.hash
    assert specialization_data is not None

    # clear the cache
    shutil.rmtree(fresh_triton_cache)
    kernel_add.device_caches[device][0].clear()

    # preload the kernel
    kernel_preload = kernel_add.preload(specialization_data)
    assert kernel_preload.hash == hash
    assert len(kernel_add.device_caches[device][0]) == 1

    # we should hit the cache and not compile anything
    counter = 0

    def inc_counter(*args, **kwargs):
        nonlocal counter
        counter += 1

    JITFunction.cache_hook = inc_counter
    final_kernel = kernel_add.warmup(torch.float32, torch.float32, torch.float32, 32, tl.float32, grid=(1, ))
    JITFunction.cache_hook = None
    assert counter == 0
    assert len(kernel_add.device_caches[device][0]) == 1
    assert final_kernel.hash == hash

    # test that we can't preload a mismatched kernel
    with pytest.raises(RuntimeError, match="Specialization data is for"):
        kernel_sub.preload(specialization_data)


def test_hooks(device, fresh_triton_cache) -> None:

    @triton.jit
    def kernel_add(a, b, o, N: tl.constexpr, type: tl.constexpr):
        idx = tl.arange(0, N)
        tl.device_assert(idx < 32, "idx < 32")
        tl.store(o + idx, tl.load(a + idx) + tl.load(b + idx))

    # get the serialized specialization data
    specialization_data = None
    is_warmup = False
    key = 0

    def cache_hook(*args, **kwargs):
        nonlocal specialization_data
        specialization_data = kwargs["compile"]["specialization_data"]
        nonlocal is_warmup
        is_warmup = kwargs["compile"]["is_warmup"]
        nonlocal key
        key = kwargs["compile"]["key"]

    specialization_data_compiled = None

    def compiled_hook(*args, **kwargs):
        nonlocal specialization_data_compiled
        specialization_data_compiled = kwargs["compile"]["specialization_data"]

    JITFunction.cache_hook = cache_hook
    JITFunction.compiled_hook = compiled_hook
    kernel_add.warmup(torch.float32, torch.float32, torch.float32, 32, tl.float32, grid=(1, ))
    assert specialization_data is not None and specialization_data_compiled == specialization_data
    assert is_warmup is True
    assert key in kernel_add.device_caches[getattr(torch, device).current_device()][0]


@pytest.mark.skipif(reason="within_2g is a HIP specific optimization", condition=not is_hip())
def test_within_2gb(device, fresh_triton_cache) -> None:
    default_buffer_ops = os.environ.get("AMDGCN_USE_BUFFER_OPS", "0")
    from triton.backends import backends

    amd_backend = backends["amd"]
    try:
        use_buffer_ops_opts = ["1", "0"]
        # The ranges should only be available when buffer ops are enabled
        pointer_ranges = [[(0, )], []]
        for use_buffer_ops, pointer_range in zip(use_buffer_ops_opts, pointer_ranges):
            # Set AMDGCN_USE_BUFFER_OPS
            amd_backend.compiler.use_buffer_ops.cache_clear()
            os.environ["AMDGCN_USE_BUFFER_OPS"] = use_buffer_ops

            @triton.jit
            def kernel_add(a):
                tl.load(a)

            # This is the attribute we want to test
            pointer_range_32 = None

            def cache_hook(*args, **kwargs):
                nonlocal pointer_range_32
                pointer_range_32 = [
                    k for k, v in kwargs["compile"]["configs"][0].items() if ["tt.pointer_range", 32] in v
                ]

            JITFunction.cache_hook = cache_hook
            # In warmup we assume that the pointer range is 32 bits
            kernel_add.warmup(torch.float32, grid=(1, ))
            assert pointer_range_32 == pointer_range
            # Torch tensor > 2GB
            kernel_add[(1, 0)](torch.empty(2**31, dtype=torch.int8, device=device))
            assert len(pointer_range_32) == 0
            # Torch tensor <= 2GB
            kernel_add[(1, 0)](torch.empty(2**31 - 1, dtype=torch.int8, device=device))
            assert pointer_range_32 == pointer_range
    finally:
        amd_backend.compiler.use_buffer_ops.cache_clear()
        os.environ["AMDGCN_USE_BUFFER_OPS"] = default_buffer_ops


def test_function_arguments(device):

    @triton.jit
    def func1():
        return 1

    @triton.jit
    def func2():
        return 2

    @triton.jit
    def func3(x):
        return x

    @triton.jit
    def func4(x, y):
        return x + y

    @triton.jit
    def kernel(Y, fn: tl.constexpr, fn_args):
        tl.store(Y, fn(*fn_args))

    JITFunction.cache_hook = None
    JITFunction.compiled_hook = None
    y = torch.zeros((5, ), dtype=torch.int32, device=device)
    kernel[(1, )](y[0], func1, tuple())
    kernel[(1, )](y[1], func2, tuple())
    kernel[(1, )](y[2], func3, (3, ))
    kernel[(1, )](y[3], func4, (3, 4))
    kernel[(1, )](y[4], func1, tuple())
    assert len(kernel.device_caches[0][0]) == 4
    assert y.tolist() == [1, 2, 3, 7, 1]
