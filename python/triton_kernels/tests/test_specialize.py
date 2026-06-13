import importlib
import types

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import GluonJITFunction
import triton.experimental.gluon.language as gl
from triton_kernels.specialize import ClosureArg, FnSpecs, SpecializationModule, cacheable, specialize


@triton.jit
def identity(x):
    return x


@gluon.jit
def gluon_identity(x):
    return x


@triton.jit
def template_kernel(o, fn: tl.constexpr):
    cst = 1.0
    cst = fn(cst)
    tl.store(o, cst)


@gluon.jit
def gluon_template_kernel(o, fn: gl.constexpr, fn_args):
    cst = fn(1.0, *fn_args)
    gl.store(o, cst)


def test_specialization_module_preserves_gluon_dialect():
    specializations = SpecializationModule(
        "specialized_gluon_kernel",
        kernels=[("kernel", gluon_template_kernel)],
        closure_args={"fn": ClosureArg("fn", "fn_args")},
    )

    module = specializations.get(fn=FnSpecs("identity", gluon_identity))

    assert isinstance(module, types.ModuleType)
    assert isinstance(module.kernel, GluonJITFunction)
    assert module.kernel.is_gluon()
    assert "fn: gl.constexpr = gluon_identity" in module.kernel.src


def retrieve_fn(module, name):
    module = importlib.import_module(module)
    fn = getattr(module, name)
    return fn


_specialized_kernel = None


def get_specialized_kernel():
    global _specialized_kernel
    if _specialized_kernel is not None:
        return _specialized_kernel
    import types
    spec_constants = {"fn": identity}
    spec_tuples = {}
    module = types.ModuleType("specialized_kernel")
    module.specialized = specialize(template_kernel, module, spec_constants, spec_tuples)
    _specialized_kernel = module.specialized
    return _specialized_kernel


@cacheable
def cacheable_kernel():
    return get_specialized_kernel()


def test_cacheable(device, fresh_triton_cache, monkeypatch):
    specialized_kernel = get_specialized_kernel()
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
    assert fn_name == "cacheable_kernel"

    # check line info in ttir
    ttir = k.asm["ttir"]
    loc = None
    for line in ttir.split("\n"):
        if loc and loc in line:
            assert "test_specialize.py" in line
            assert ":18:5" in line
        if "store" in line:
            loc = line.split("(", 1)[1].split(")", 1)[0]
    assert loc is not None, f"Expected to find a store instruction with location info, got: {ttir}"

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
