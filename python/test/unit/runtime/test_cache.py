import expecttest
import importlib.util
import itertools
import json
import multiprocessing
import os
import re
import gc
import shutil
import pathlib
import subprocess
import sys
from collections import namedtuple
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from contextlib import ExitStack
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_hip
from triton.runtime.cache import FileCacheManager, RemoteCacheManager


@pytest.mark.parametrize("backend, options_type, arch", [
    ("amd", "HIPOptions", "gfx950"),
    ("nvidia", "CUDAOptions", "sm90"),
])
def test_extern_libs_hash_contents(backend, options_type, arch, tmp_path):
    compiler = importlib.import_module(f"triton.backends.{backend}.compiler")
    library = tmp_path / "custom.bc"
    library.write_bytes(b"original library")
    options = getattr(compiler, options_type)(arch=arch, extern_libs={"custom": str(library)})
    try:
        original = options.hash()
        assert options.hash() == original
        library.write_bytes(b"modified library")
        # File hashes are memoized per process. Start cold, as in a new invocation.
        compiler.file_hash.cache_clear()
        assert options.hash() != original
    finally:
        compiler.file_hash.cache_clear()


def test_file_cache_manager_get_group_rejects_missing_child(fresh_knobs, tmp_path):
    fresh_knobs.cache.dir = str(tmp_path)
    manager = FileCacheManager("key")
    metadata_path = manager.put("{}", "kernel.json", binary=False)
    artifact_path = manager.put("binary", "kernel.cubin", binary=False)

    manager.put_group("kernel.json", {
        "kernel.json": metadata_path,
        "kernel.cubin": artifact_path,
    })
    assert manager.get_group("kernel.json") == {
        "kernel.json": metadata_path,
        "kernel.cubin": artifact_path,
    }

    os.remove(artifact_path)
    assert manager.get_group("kernel.json") is None


def test_file_cache_manager_get_group_relocated_dir(fresh_knobs, tmp_path):
    original, copy = tmp_path / "original", tmp_path / "copy"
    fresh_knobs.cache.dir = str(original)
    manager = FileCacheManager("key")
    metadata_path = manager.put("{}", "kernel.json", binary=False)
    artifact_path = manager.put("binary", "kernel.cubin", binary=False)
    manager.put_group("kernel.json", {
        "kernel.json": metadata_path,
        "kernel.cubin": artifact_path,
    })

    shutil.copytree(original, copy)
    shutil.rmtree(original)
    fresh_knobs.cache.dir = str(copy)
    manager = FileCacheManager("key")
    assert manager.get_group("kernel.json") == {
        "kernel.json": os.path.join(manager.cache_dir, "kernel.json"),
        "kernel.cubin": os.path.join(manager.cache_dir, "kernel.cubin"),
    }


def test_remote_cache_manager_get_group_rejects_missing_child(fresh_knobs, tmp_path):

    class DictRemoteCacheBackend:
        data = {}

        def __init__(self, key):
            self.key = key

        def get(self, filenames):
            return {filename: self.data[filename] for filename in filenames if filename in self.data}

        def put(self, filename, data):
            self.data[filename] = data

    fresh_knobs.cache.dir = str(tmp_path)
    fresh_knobs.cache.remote_manager_class = DictRemoteCacheBackend
    DictRemoteCacheBackend.data = {}

    manager = RemoteCacheManager("key")
    manager.put("{}", "kernel.json", binary=False)
    manager.put(b"binary", "kernel.cubin")
    manager.put_group("kernel.json", {
        "kernel.json": "unused-local-path",
        "kernel.cubin": "unused-local-path",
    })

    group = manager.get_group("kernel.json")
    assert group is not None
    assert set(group) == {"kernel.json", "kernel.cubin"}

    del DictRemoteCacheBackend.data["kernel.cubin"]
    assert manager.get_group("kernel.json") is None


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
    with ExitStack() as cleanup:
        # Restoring a dependency's source also invalidates its callers' hashes.
        for function in (kernel, function_0, function_1, function_2, target):
            function.hash = None
            cleanup.callback(setattr, function, "hash", None)
        cleanup.callback(to_modify._unsafe_update_src, to_modify.src)
        to_modify._unsafe_update_src(to_modify.src.replace(old, new))
        return target.cache_key


def test_nochange():
    baseline = kernel.cache_key
    updated = apply_src_change(kernel, 'i + 1', 'i + 1', function_1)
    assert baseline == updated


def test_toplevel_change():
    baseline = kernel.cache_key
    updated = apply_src_change(kernel, 'i + 1', 'i + 2', function_1)
    assert baseline != updated
    assert kernel.cache_key == baseline


def test_keyword_only_default_dependency_change():

    @triton.jit
    def with_default(i, *, function_1: tl.constexpr = function_1):
        return function_1(i)

    baseline = with_default.cache_key
    updated = apply_src_change(with_default, 'i + 1', 'i + 2', function_1)
    assert baseline != updated
    assert with_default.cache_key == baseline


def test_nested1_change():
    baseline = kernel.cache_key
    updated = apply_src_change(kernel, 'i + 1', 'i + 2', function_2)
    assert baseline != updated
    assert kernel.cache_key == baseline


def test_nested2_change():
    baseline = kernel.cache_key
    updated = apply_src_change(kernel, 'i + 1', 'i + 2', function_0)
    assert baseline != updated
    assert kernel.cache_key == baseline


ORDER_DEPENDENT_CONSTEXPR = tl.constexpr(42)


@triton.jit
def order_dependent_inner():
    return ORDER_DEPENDENT_CONSTEXPR


@triton.jit
def order_dependent_outer():
    order_dependent_inner()


def test_cache_key_independent_of_dependency_hash_order():
    functions = (order_dependent_inner, order_dependent_outer)

    for function in functions:
        function.hash = None
        function.used_global_vals = {}
    cold_key = order_dependent_outer.cache_key

    for function in functions:
        function.hash = None
        function.used_global_vals = {}
    order_dependent_inner.cache_key
    prehashed_key = order_dependent_outer.cache_key

    assert prehashed_key == cold_key


def test_cache_key_independent_of_globals_dict_identity():

    def make_child(value):
        shared = tl.constexpr(value)

        @triton.jit
        def child():
            return shared

        return child

    child_a = make_child(1)
    child_b = make_child(2)

    @triton.jit
    def parent():
        child_a()
        child_b()

    functions = (child_a, child_b, parent)

    def key_after_prehashing(first, second):
        for function in functions:
            function.hash = None
            function.used_global_vals = {}
        first.cache_key
        second.cache_key
        return parent.cache_key

    assert key_after_prehashing(child_a, child_b) == key_after_prehashing(child_b, child_a)


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


@triton.constexpr_function
def constexpr_flag_fn():
    return False


@triton.jit
def constexpr_fn_user(out):
    a: tl.constexpr = constexpr_flag_fn()
    tl.store(out, a)


def test_constexpr_fn_change():
    baseline = constexpr_fn_user.cache_key

    orig_src = constexpr_flag_fn.src
    new_src = orig_src.replace("False", "True")
    constexpr_flag_fn._unsafe_update_src(new_src)
    constexpr_fn_user.hash = None
    updated = constexpr_fn_user.cache_key
    assert baseline != updated

    constexpr_flag_fn._unsafe_update_src(orig_src)
    constexpr_fn_user.hash = None
    assert constexpr_fn_user.cache_key == baseline


@triton.constexpr_function
def invalid_constexpr_fn():
    return torch.cuda.get_device_capability()


def test_invalid_constexpr_fn():
    with pytest.raises(RuntimeError):
        invalid_constexpr_fn.cache_key


def write_and_load_module(temp_file: pathlib.Path, code, num_extra_lines):
    temp_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file.write_text(('# extra line\n' * num_extra_lines) + code)
    spec = importlib.util.spec_from_file_location("module.name", str(temp_file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("change_line_info", [False, True])
@pytest.mark.parametrize("filename, extra_lines, decorators, indentation", [
    ("kernel.py", 0, 1, 4),
    ("other.py", 0, 1, 4),
    ("kernel.py", 1, 1, 4),
    ("kernel.py", 0, 2, 4),
    ("kernel.py", 0, 1, 8),
], ids=["unchanged", "file", "line", "decorator", "column"])
def test_source_locations_invalidate_cache(tmp_path: pathlib.Path, fresh_knobs, fresh_triton_cache, change_line_info,
                                           filename, extra_lines, decorators, indentation):
    from textwrap import dedent, indent

    fresh_knobs.compilation.disable_line_info = False

    def load_kernel(filename="kernel.py", extra_lines=0, decorators=1, indentation=4):
        code = dedent("""\
            import triton
            import triton.language as tl
            def identity(fn):
                return fn
            def make_kernel():
        """)
        body = "@identity\n" * decorators + dedent("""\
            @triton.jit
            def test_kernel(X):
                tl.store(X, 1)
            return test_kernel
        """)
        code += indent(body, " " * indentation) + "test_kernel = make_kernel()\n"
        return write_and_load_module(tmp_path / filename, code, extra_lines).test_kernel

    original = load_kernel()
    original_key = original.cache_key
    original_compiled = triton.compile(triton.compiler.ASTSource(original, {"X": "*i32"}))

    fresh_knobs.compilation.disable_line_info = change_line_info
    updated = load_kernel(filename, extra_lines, decorators, indentation)
    updated_compiled = triton.compile(triton.compiler.ASTSource(updated, {"X": "*i32"}))
    source_changed = (filename, extra_lines, decorators, indentation) != ("kernel.py", 0, 1, 4)
    assert original.src == updated.src
    assert (original_key != updated.cache_key) == source_changed
    # Column-only changes may produce identical binary bytes.
    assert (original_compiled.hash != updated_compiled.hash) == (source_changed or change_line_info)
    if change_line_info:
        assert original_compiled.kernel != updated_compiled.kernel


def test_reuse(device, fresh_triton_cache):
    counter = 0

    def inc_counter(*args, **kwargs):
        nonlocal counter
        counter += 1

    triton.knobs.runtime.jit_cache_hook = inc_counter
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

    triton.knobs.runtime.jit_cache_hook = inc_counter
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


class Wrapped(NamedTuple):
    value: object


@pytest.mark.parametrize("a, b", [(1, 1.0), (1, True), (0.0, -0.0), ((1, ), (1.0, )),
                                  (np.float64(0.0), np.float64(-0.0)), (Wrapped(1), Wrapped(1.0)),
                                  (tl.constexpr(1), tl.constexpr(1.0))])
def test_constexpr_cache_key_type(a, b, device):

    @triton.jit
    def kernel(X, F, Y, Z, C: tl.constexpr, IS_TUPLE: tl.constexpr):
        if IS_TUPLE:
            c = C[0]
        else:
            c = C
        tl.store(Y, tl.load(X) * c)
        tl.store(Z, 1.0 / (tl.load(F) * c))

    # x * 1 is exact in int32, x * 1.0 rounds to 2**30 in fp32; 1 / (1.0 * c) keeps the sign of zero
    x = torch.tensor([2**30 + 1], dtype=torch.int32, device=device)
    f = torch.ones(1, dtype=torch.float32, device=device)

    def run(c):
        y = torch.empty_like(x)
        z = torch.empty_like(f)
        kernel[(1, )](x, f, y, z, c, isinstance(c, tuple))
        return y.item(), z.item()

    device = getattr(torch, device).current_device()
    for first, second in [(a, b), (b, a)]:
        kernel.device_caches.clear()
        expected = run(second)
        kernel.device_caches.clear()
        run(first)
        assert run(second) == expected
        assert len(kernel.device_caches[device][0]) == 2


class OtherWrapped(NamedTuple):
    other: object


ConfigA = namedtuple("Config", ["value"])
ConfigB = namedtuple("Config", ["other"])


@triton.constexpr_function
def wrapped_value(c):
    return getattr(c, "value", 2)


@pytest.mark.parametrize("cls_a, cls_b", [(Wrapped, OtherWrapped), (ConfigA, ConfigB)])
def test_constexpr_cache_key_namedtuple_class(cls_a, cls_b, device):

    @triton.jit
    def kernel(X, C: tl.constexpr):
        tl.store(X, wrapped_value(C))

    x = torch.empty(1, dtype=torch.int32, device=device)
    for c, expected in [(cls_a(1), 1), (cls_b(1), 2), (cls_a(1), 1)]:
        kernel[(1, )](x, c)
        assert x.item() == expected


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


def test_keyword_only_shadows_global(monkeypatch):
    monkeypatch.setitem(globals(), "GLOBAL", 42)

    @triton.jit
    def kernel(*, GLOBAL: tl.constexpr):
        tl.static_assert(GLOBAL == 1)

    kernel[(1, )](GLOBAL=1)
    monkeypatch.setitem(globals(), "GLOBAL", 43)
    kernel[(1, )](GLOBAL=1)


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


def test_cache_closure():

    def make_closure(cst):

        @triton.jit
        def closure():
            tl.full((16, ), cst, dtype=tl.int32)

        return closure

    cst = tl.constexpr(42)
    closure = make_closure(cst)

    closure[(1, )]()
    cst.value = 43
    with pytest.raises(RuntimeError) as e:
        closure[(1, )]()

    assert "cst has changed since we compiled this kernel, from constexpr[42] to constexpr[43]" in str(e.value)


CLOSURE_SHADOW_GLOBAL = tl.constexpr(3)


def test_cache_closure_shadows_global():

    def make_closure(value):
        CLOSURE_SHADOW_GLOBAL = value

        @triton.jit
        def closure():
            return CLOSURE_SHADOW_GLOBAL

        return closure

    first = make_closure(tl.constexpr(7))
    second = make_closure(tl.constexpr(9))
    same_as_first = make_closure(tl.constexpr(7))
    captures_none = make_closure(None)

    first_key = first.cache_key
    second_key = second.cache_key
    same_as_first_key = same_as_first.cache_key
    captures_none.cache_key

    def tracked_values(fn):
        return {name: value for (name, _), (value, _) in fn.used_global_vals.items()}

    assert first_key != second_key
    assert first_key == same_as_first_key
    assert tracked_values(first)["CLOSURE_SHADOW_GLOBAL"].value == 7
    assert tracked_values(second)["CLOSURE_SHADOW_GLOBAL"].value == 9
    assert "CLOSURE_SHADOW_GLOBAL" not in tracked_values(captures_none)


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


def test_constexpr_cache_invalidation_recreated(device):

    def test_run(val):
        VAL = tl.constexpr(val)

        @triton.jit
        def kernel(out):
            tl.store(out, VAL)

        out = torch.zeros(1, device=device)
        kernel[(1, )](out)
        return out.item()

    assert test_run(123) == 123
    assert test_run(123) == 123
    assert test_run(1234) == 1234
    assert test_run(1234) == 1234


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


class _PreloadClass:

    def __init__(self, offset):
        self.offset = offset

    class Nested:

        def __init__(self, offset):
            self.offset = offset


@triton.aggregate
class _PreloadAggregate:
    offset: tl.constexpr


@triton.jit
def _preload_function():
    return 17


def _serialize_constants(constants):
    from triton.runtime.jit import serialize_specialization_data
    return serialize_specialization_data("reference_kernel", {}, constants, {}, SimpleNamespace(), "reference-key",
                                         SimpleNamespace(backend="cuda", arch=90, warp_size=32))


def _preload_constants(specialization_data, monkeypatch):
    jit = importlib.import_module("triton.runtime.jit")
    target = SimpleNamespace(backend="cuda", arch=90, warp_size=32)
    backend = SimpleNamespace(parse_options=lambda options: options)
    kernel = SimpleNamespace(
        _fn_name="reference_kernel",
        device_caches={0: (None, None, target, backend, None)},
        _do_compile=lambda key, signature, device, constexprs, options, attrs, warmup: constexprs,
    )
    monkeypatch.setattr(jit, "driver", SimpleNamespace(active=SimpleNamespace(get_current_device=lambda: 0)))
    return jit.JITFunction.preload(kernel, specialization_data)


@pytest.mark.parametrize("constant", [int, _PreloadClass, _PreloadClass.Nested, _PreloadAggregate, _preload_function])
def test_preload_reference_constants(constant, monkeypatch):
    specialization_data = _serialize_constants({(0, ): constant})
    constants = _preload_constants(specialization_data, monkeypatch)
    assert constants[(0, )] is constant


def test_preload_dtype_constant(monkeypatch):
    specialization_data = _serialize_constants({(0, ): tl.float32})
    assert _preload_constants(specialization_data, monkeypatch)[(0, )] == tl.float32


def test_preload_registered_local_function(monkeypatch):

    @triton.jit
    def local_function():
        return 17

    specialization_data = _serialize_constants({(0, ): local_function})
    assert _preload_constants(specialization_data, monkeypatch)[(0, )] is local_function


def test_preload_class_import_in_fresh_process(tmp_path, monkeypatch):
    module_name = "preload_class_module"
    (tmp_path / f"{module_name}.py").write_text("class Outer:\n    class Inner:\n        pass\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module(module_name)
    try:
        specialization_data = _serialize_constants({(0, ): module.Outer.Inner})
        code = """
import sys
import pytest
from test_cache import _preload_constants

assert 'preload_class_module' not in sys.modules
with pytest.MonkeyPatch.context() as patch:
    constants = _preload_constants(sys.stdin.read(), patch)
import preload_class_module
assert constants[(0,)] is preload_class_module.Outer.Inner
"""
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(pathlib.Path(__file__).parent), *sys.path]))
        subprocess.run([sys.executable, "-c", code], input=specialization_data, text=True, env=env, check=True,
                       timeout=60)
    finally:
        sys.modules.pop(module_name, None)


@pytest.mark.parametrize("class_key, message", [
    ("builtins:MissingClass", "Unable to resolve class"),
    ("builtins:len", "Expected class"),
])
def test_preload_invalid_class_reference(class_key, message, monkeypatch):
    specialization_data = json.loads(_serialize_constants({(0, ): int}))
    specialization_data["constant_vals"][0]["class"] = class_key
    with pytest.raises(RuntimeError, match=message):
        _preload_constants(json.dumps(specialization_data), monkeypatch)


def test_preload_local_class_reference(monkeypatch):

    class Local:
        pass

    specialization_data = _serialize_constants({(0, ): Local})
    with pytest.raises(RuntimeError, match="Unable to resolve class.*<locals>.Local"):
        _preload_constants(specialization_data, monkeypatch)


@pytest.mark.parametrize("cls", [_PreloadClass, _PreloadClass.Nested, _PreloadAggregate])
def test_preload_class_kernel(cls, device, fresh_triton_cache, fresh_knobs):

    @triton.jit
    def kernel(out, cls: tl.constexpr):
        tl.store(out, cls(17).offset)

    specializations = []

    def cache_hook(**kwargs):
        specializations.append(kwargs["compile"]["specialization_data"])

    fresh_knobs.runtime.jit_cache_hook = cache_hook
    fresh_knobs.runtime.jit_post_compile_hook = cache_hook
    output = torch.empty((), device=device, dtype=torch.int32)
    compiled = kernel[(1, )](output, cls)
    assert output.item() == 17
    assert len(specializations) == 2
    assert specializations[0] == specializations[1]

    shutil.rmtree(fresh_triton_cache)
    kernel.device_caches.clear()
    preloaded = kernel.preload(specializations[0])
    assert preloaded.hash == compiled.hash
    output.zero_()
    assert kernel[(1, )](output, cls) is preloaded
    assert output.item() == 17
    assert len(specializations) == 4


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

    triton.knobs.runtime.jit_cache_hook = cache_hook
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

    triton.knobs.runtime.jit_cache_hook = inc_counter
    final_kernel = kernel_add.warmup(torch.float32, torch.float32, torch.float32, 32, tl.float32, grid=(1, ))
    assert counter == 0
    assert len(kernel_add.device_caches[device][0]) == 1
    assert final_kernel.hash == hash

    # test that we can't preload a mismatched kernel
    with pytest.raises(RuntimeError, match="Specialization data is for"):
        kernel_sub.preload(specialization_data)

    specialization_data_unknown_target = re.sub(r'("target"\s*:\s*\{[^{}]*"backend"\s*:\s*)"(.*?)"',
                                                r'\1"unknown_target"', specialization_data, count=1)

    with pytest.raises(RuntimeError, match="Specialization data is for {'backend': 'unknown_target'"):
        kernel_add.preload(specialization_data_unknown_target)


@triton.jit
def sequence_offset(idx, offsets: tl.constexpr):
    tl.static_assert(len(offsets) == 2)
    tl.static_assert(len(offsets[0]) == 2)
    tl.static_assert(len(offsets[1]) == 1)
    return idx + offsets[0][0] + offsets[0][1] + offsets[1][0]


@triton.jit
def tuple_call_kernel(out_ptr, offsets: tl.constexpr):
    tl.static_assert(len(offsets) == 2)
    idx = tl.arange(0, 1)
    tl.store(out_ptr + idx, sequence_offset(idx, offsets))


def test_preload_constexpr_tuple_arg(device, fresh_triton_cache, fresh_knobs) -> None:
    device = getattr(torch, device).current_device()
    offsets = ((2, 3), (5, ))
    specialization_data = None

    def cache_hook(*args, **kwargs):
        nonlocal specialization_data
        specialization_data = kwargs["compile"]["specialization_data"]

    fresh_knobs.runtime.jit_cache_hook = cache_hook
    tuple_call_kernel.device_caches[device][0].clear()
    pre_compile = tuple_call_kernel.warmup(torch.int32, offsets, grid=(1, ))
    hash = pre_compile.hash
    assert specialization_data is not None

    shutil.rmtree(fresh_triton_cache)
    tuple_call_kernel.device_caches[device][0].clear()

    kernel_preload = tuple_call_kernel.preload(specialization_data)
    assert kernel_preload.hash == hash
    assert len(tuple_call_kernel.device_caches[device][0]) == 1

    counter = 0

    def inc_counter(*args, **kwargs):
        nonlocal counter
        counter += 1

    fresh_knobs.runtime.jit_cache_hook = inc_counter
    final_kernel = tuple_call_kernel.warmup(torch.int32, offsets, grid=(1, ))
    assert counter == 0
    assert len(tuple_call_kernel.device_caches[device][0]) == 1
    assert final_kernel.hash == hash


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
    name = None

    def cache_hook(*args, **kwargs):
        nonlocal specialization_data
        specialization_data = kwargs["compile"]["specialization_data"]
        nonlocal is_warmup
        is_warmup = kwargs["compile"]["is_warmup"]
        nonlocal key
        key = kwargs["compile"]["key"]
        nonlocal name
        name = kwargs["fn"].name

    specialization_data_compiled = None

    def compiled_hook(*args, **kwargs):
        nonlocal specialization_data_compiled
        specialization_data_compiled = kwargs["compile"]["specialization_data"]

    triton.knobs.runtime.jit_cache_hook = cache_hook
    triton.knobs.runtime.jit_post_compile_hook = compiled_hook
    kernel_add.warmup(torch.float32, torch.float32, torch.float32, 32, tl.float32, grid=(1, ))
    assert specialization_data is not None and specialization_data_compiled == specialization_data
    assert is_warmup is True
    assert key in kernel_add.device_caches[getattr(torch, device).current_device()][0]
    assert name == "test_hooks.<locals>.kernel_add"


@pytest.mark.skipif(reason="within_2g is a HIP specific optimization", condition=not is_hip())
def test_within_2gb(device, fresh_triton_cache) -> None:
    default_buffer_ops = os.environ.get("AMDGCN_USE_BUFFER_OPS", "0")
    try:
        use_buffer_ops_opts = ["1", "0"]
        # The ranges should only be available when buffer ops are enabled
        pointer_ranges = [[(0, )], []]
        for use_buffer_ops, pointer_range in zip(use_buffer_ops_opts, pointer_ranges):
            # Set AMDGCN_USE_BUFFER_OPS
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

            triton.knobs.runtime.jit_cache_hook = cache_hook
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

    y = torch.zeros((5, ), dtype=torch.int32, device=device)
    kernel[(1, )](y[0], func1, tuple())
    kernel[(1, )](y[1], func2, tuple())
    kernel[(1, )](y[2], func3, (3, ))
    kernel[(1, )](y[3], func4, (3, 4))
    kernel[(1, )](y[4], func1, tuple())

    device = getattr(torch, device).current_device()
    assert len(kernel.device_caches[device][0]) == 4
    assert y.tolist() == [1, 2, 3, 7, 1]


class MockThreadPool(Executor):

    def __init__(self):
        self.work_queue = []

    def submit(self, fn, *args, **kwargs):
        future = Future()

        def task():
            if not future.set_running_or_notify_cancel():
                return

            try:
                result = fn(*args, **kwargs)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)

        self.work_queue.append(task)
        return future

    def run_one(self):
        task = self.work_queue.pop(0)
        task()

    def run_all(self):
        while self.work_queue:
            self.run_one()

    def shutdown(self, wait=True, *, cancel_futures=False):
        self.run_all()


def test_async_compile_mock(device, fresh_triton_cache):

    @triton.jit
    def kernel(Y, a: tl.constexpr):
        tl.store(Y, a)

    with (
            MockThreadPool() as pool,
            triton.AsyncCompileMode(pool),
    ):
        a = torch.empty((16, 16), device=device)
        b = torch.empty((16, 16), dtype=torch.int32, device=device)
        kernel.warmup(a, 0, grid=(1, ))
        kernel.warmup(a, 1, grid=(1, ))
        kernel.warmup(b, 0, grid=(1, ))
        kernel.warmup(b, 1, grid=(1, ))

        device = getattr(torch, device).current_device()

        # Nothing has actually compiled yet
        assert len(kernel.device_caches[device][0]) == 4
        assert len(pool.work_queue) == 4

        # Duplicates are only submitted once
        kernel.warmup(a, 0, grid=(1, ))
        kernel.warmup(a, 1, grid=(1, ))
        assert len(kernel.device_caches[device][0]) == 4
        assert len(pool.work_queue) == 4

        pool.run_one()
        kernel[(1, )](a, 0)
        assert len(kernel.device_caches[device][0]) == 4
        assert a[0, 0] == 0.0

        pool.run_all()


def test_async_compile(device, fresh_triton_cache):

    @triton.jit
    def kernel(Y, a: tl.constexpr):
        tl.store(Y, a)

    with (
            ThreadPoolExecutor(2) as pool,
            triton.AsyncCompileMode(pool),
    ):
        a = torch.empty((16, 16), device=device)
        b = torch.empty((16, 16), dtype=torch.int32, device=device)
        kernel.warmup(a, 0, grid=(1, ))
        kernel.warmup(a, 1, grid=(1, ))
        kernel.warmup(b, 0, grid=(1, ))
        kernel.warmup(b, 1, grid=(1, ))

        device = getattr(torch, device).current_device()
        assert len(kernel.device_caches[device][0]) == 4

        kernel[(1, )](b, 1)
        assert b[0, 0] == 1
        kernel[(1, )](b, 0)
        assert b[0, 0] == 0
        kernel[(1, )](a, 0)
        assert a[0, 0] == 0
        kernel[(1, )](a, 1)
        assert a[0, 0] == 1
        kernel[(1, )](a, 2)
        assert a[0, 0] == 2


def test_async_compile_error(fresh_triton_cache):

    @triton.jit
    def fn(x: tl.constexpr):
        tl.static_assert(x == 2)

    with (
            ThreadPoolExecutor(2) as pool,
            triton.AsyncCompileMode(pool),
    ):
        assert triton.runtime._async_compile.active_mode.get() is not None
        fn.warmup(1, grid=(1, ))

        assert len(fn.device_caches[0][0]) == 1
        # Resolving the queued kernel reports the compilation failure. The
        # mode must not resolve the failed FutureKernel again on exit.
        with pytest.raises(triton.compiler.errors.CompileTimeAssertionFailure):
            fn[(1, )](1)

    # After the AsyncCompileMode context manager exits, the active mode should
    # be set to None again, even if there was an error.
    assert triton.runtime._async_compile.active_mode.get() is None
    # A failed Future would otherwise retain its exception traceback and the
    # compiler objects reachable through it.
    assert len(fn.device_caches[0][0]) == 0


def test_higher_order_kernel(device, fresh_triton_cache, capsys):

    @triton.jit
    def fn_a():
        tl.static_print("Compiling with fn_a")
        return 0

    @triton.jit
    def kernel(out_ptr, FUNC: tl.constexpr) -> None:
        val = FUNC()
        tl.store(out_ptr, val)

    output = torch.empty((), device=device, dtype=torch.int32)
    kernel[(1, )](output, fn_a)
    assert output.item() == 0

    # Test we can update src in-place
    orig_src = fn_a.src
    new_src = orig_src.replace("with fn_a", "with fn_a after modification")
    new_src = new_src.replace("0", "1")
    fn_a._unsafe_update_src(new_src)
    kernel[(1, )](output, fn_a)
    assert output.item() == 1

    # Test that the on disc cache works
    kernel.device_caches.clear()
    kernel[(1, )](output, fn_a)
    assert output.item() == 1

    fn_a._unsafe_update_src(orig_src)
    kernel[(1, )](output, fn_a)
    assert output.item() == 0

    expecttest.assert_expected_inline(capsys.readouterr().out, """\
Compiling with fn_a
Compiling with fn_a after modification
""")


def test_preload_higher_order_kernels(device, fresh_triton_cache) -> None:

    @triton.jit
    def fn_a():
        return 17

    @triton.jit
    def fn_b():
        return 31

    @triton.jit
    def kernel(out_ptr, FUNC: tl.constexpr) -> None:
        val = FUNC()
        tl.store(out_ptr, val)

    device = getattr(torch, device).current_device()

    # get the serialized specialization data
    specialization_data = None

    def cache_hook(*args, **kwargs):
        nonlocal specialization_data
        specialization_data = kwargs["compile"]["specialization_data"]

    triton.knobs.runtime.jit_cache_hook = cache_hook
    output = torch.empty((), device=device, dtype=torch.int32)
    compiled_kernel = kernel[(1, )](output, fn_a)
    assert output.item() == 17
    hash = compiled_kernel.hash
    assert specialization_data is not None

    # clear the cache
    shutil.rmtree(fresh_triton_cache)
    kernel.device_caches[device][0].clear()

    # preload the kernel
    kernel_preload = kernel.preload(specialization_data)
    assert kernel_preload.hash == hash
    assert len(kernel.device_caches[device][0]) == 1

    # we should hit the cache and not compile anything
    counter = 0

    def inc_counter(*args, **kwargs):
        nonlocal counter
        counter += 1

    triton.knobs.runtime.jit_cache_hook = inc_counter
    final_kernel = kernel[(1, )](output, fn_a)
    assert counter == 0
    assert len(kernel.device_caches[device][0]) == 1
    assert final_kernel.hash == hash

    # different function should compile and not hit the cache
    kernel[(1, )](output, fn_b)
    assert counter == 1
    assert output.item() == 31


def test_module_load_unload(device, fresh_knobs):

    @triton.jit
    def kernel(out_ptr, val) -> None:
        tl.store(out_ptr, val)

    # we should hit the kernel unload call to decrese the counter from 1 to 0
    counter = 1
    owner_pid = os.getpid()

    def kernel_unload(*args, **kwargs):
        nonlocal counter
        assert os.getpid() == owner_pid
        counter -= 1

    # turn off python garbage collector, so the callback is not called
    # in the garbage collector
    gc.disable()
    triton.knobs.runtime.kernel_unload_hook.add(kernel_unload)

    out = torch.randn(1, dtype=torch.float32, device=device)
    pre_compile = kernel.warmup(out, 1, grid=(1, ))
    pre_compile._init_handles()

    assert counter == 1
    assert pre_compile.module is not None

    if "fork" in multiprocessing.get_all_start_methods():
        child = multiprocessing.get_context("fork").Process(target=pre_compile.__del__)
        child.start()
        child.join()
        assert child.exitcode == 0
        assert counter == 1

    pre_compile.__del__()

    assert counter == 0
    assert pre_compile.module is None
    # turn on garbage collector
    gc.enable()
