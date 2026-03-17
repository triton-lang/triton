import importlib

import pytest
import torch
from triton_kernels.specialize import cacheable, specialize
import triton
import triton.language as tl
from triton.experimental import gluon
import triton.experimental.gluon.language as gl  # noqa: F401


@triton.jit
def identity(x):
    return x


@triton.jit
def template_kernel(o, fn: tl.constexpr):
    cst = 1.0
    cst = fn(cst)
    tl.store(o, cst)


gluon_identity = gluon.jit(identity.fn)
gluon_template_kernel = gluon.jit(template_kernel.fn)


def retrieve_fn(module, name):
    module = importlib.import_module(module)
    fn = getattr(module, name)
    return fn


_specialized_kernels = {}


def get_specialized_kernel(language):
    assert language in ["gluon", "triton"]
    if language in _specialized_kernels:
        return _specialized_kernels[language]
    import types
    if language == "gluon":
        _identity = gluon_identity
        _template = gluon_template_kernel
    else:
        _identity = identity
        _template = template_kernel
    spec_constants = {"fn": _identity}
    spec_tuples = {}
    module = types.ModuleType("specialized_kernel")
    module.specialized = specialize(_template, module, spec_constants, spec_tuples)
    _specialized_kernels[language] = module.specialized
    return _specialized_kernels[language]


@cacheable
def cacheable_kernel():
    return get_specialized_kernel("triton")


@cacheable
def cacheable_kernel_gluon():
    return get_specialized_kernel("gluon")


@pytest.mark.parametrize("is_gluon", [True, False])
def test_cacheable(device, fresh_triton_cache, monkeypatch, is_gluon):
    specialized_kernel = get_specialized_kernel("gluon" if is_gluon else "triton")
    expected_fn_name = "cacheable_kernel_gluon" if is_gluon else "cacheable_kernel"
    monkeypatch.setenv("TRITON_DISABLE_LINE_INFO", "0")

    specialization_data = None
    fn_name = None
    module_name = None

    def cache_hook(*args, **kwargs):
        nonlocal specialization_data
        nonlocal fn_name
        nonlocal module_name
        specialization_data = kwargs["compile"]["specialization_data"]
        fn_name = kwargs["fn"].name
        module_name = kwargs["fn"].module

    triton.knobs.runtime.jit_cache_hook = cache_hook
    o = torch.empty((1, ), dtype=torch.float32, device=device)
    k = specialized_kernel[(1, )](o, )
    hash = k.hash
    assert o.item() == 1.0
    assert module_name == "tests.test_specialize"
    assert fn_name == expected_fn_name

    ir_key = "source" if is_gluon else "ttir"
    ir_src = k.asm[ir_key]
    loc = None
    for line in ir_src.split("\n"):
        if loc and loc in line:
            assert "test_specialize.py" in line
            assert ":20:5" in line
        if "store" in line:
            loc = line.split("(", 1)[1].split(")", 1)[0]
    assert loc is not None, f"Expected to find a store instruction with location info, got: {ir_src}"

    compile_count = 0

    def count_hook(*args, **kwargs):
        nonlocal compile_count
        compile_count += 1

    triton.knobs.runtime.jit_cache_hook = count_hook
    # clear the cache
    specialized_kernel.device_caches.clear()

    # retrieve the kernel from name and preload it.
    fn = retrieve_fn(module_name, fn_name)
    assert fn == specialized_kernel
    preload = fn.preload(specialization_data)
    assert compile_count == 1
    assert preload.hash == hash

    # verify that we hit the cache.
    compile_count = 0
    specialized_kernel[(1, )](o, )
    assert compile_count == 0
