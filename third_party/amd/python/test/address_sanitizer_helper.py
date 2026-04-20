import os
import sys
import torch
import triton
import triton.language as tl

# Diagnostic prints so that, when ASan fails to emit its report at runtime,
# the test harness in test_address_sanitizer.py still receives enough context
# on stderr to diagnose the cause.
print("PYEXE:", sys.executable, file=sys.stderr)
_diag_env = {
    k: os.environ.get(k)
    for k in (
        "HSA_XNACK",
        "TRITON_ENABLE_ASAN",
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "AMDGCN_USE_BUFFER_OPS",
        "AMD_PYTORCH_NO_CUDA_MEMORY_CACHING",
        "PYTORCH_NO_HIP_MEMORY_CACHING",
        "HSA_DISABLE_FRAGMENT_ALLOCATOR",
        "TRITON_ALWAYS_COMPILE",
    )
}
print("ENV:", _diag_env, file=sys.stderr)
try:
    with open(f"/proc/{os.getpid()}/maps") as _f:
        _hits = [ln for ln in _f if ("libamdhip64" in ln or "libclang_rt.asan" in ln)]
    print("MAPS:\n" + "".join(_hits), file=sys.stderr)
except Exception as _e:
    print(f"MAPS: <unreadable: {_e}>", file=sys.stderr)

size = 4096
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')
output = torch.empty_like(x)
n_elements = output.numel()
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    #Set access to go out of bounds for ASAN test
    offsets = block_start + tl.arange(0, BLOCK_SIZE) + 1
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    output = x + y
    tl.store(output_ptr + offsets, output)


pgm = add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
amdgcn = pgm.asm['amdgcn']
print(amdgcn)
