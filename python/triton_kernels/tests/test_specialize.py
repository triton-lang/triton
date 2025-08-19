import torch
import importlib
from triton_kernels.specialize import cacheable, specialize
import triton
import triton.language as tl


@triton.jit
def template_kernel(o):
    cst = 1.0
    tl.store(o, cst)


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
    spec_constants = {}
    spec_tuples = {}
    module = types.ModuleType("specialized_kernel")
    module.specialized = specialize(template_kernel, module, spec_constants, spec_tuples)
    _specialized_kernel = module.specialized
    return _specialized_kernel


@cacheable
def cacheable_kernel():
    return get_specialized_kernel()


def test_cacheable(device, fresh_triton_cache):
    specialized_kernel = get_specialized_kernel()

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
