import triton.compiler.compiler as compiler
import hashlib
import json
from triton.runtime.cache import get_cache_key
from triton._internal_testing import is_cuda, is_hip
from pathlib import Path


target = compiler.GPUTarget("hip", "gfx942", 64) if is_hip() else compiler.GPUTarget("cuda", 80, 32)

# backend = compiler.make_backend(target)
# key = get_cache_key(kernel_src, backend, None, env_vars=None)
# hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
kernels = []
for i in range(50000):
    print(f"i = {i}")
    compiled_kernel = compiler.compile(str(Path(__file__).parent / "attn_fwd.ttir"), target, None, None)
    kernels.append(compiled_kernel)
