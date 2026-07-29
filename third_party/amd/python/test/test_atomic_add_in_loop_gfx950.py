"""
Regression test for https://github.com/triton-lang/triton/issues/11037.
tl.atomic_add inside a while loop hangs on AMD gfx950 (MI350 class).
"""

import numpy as np
import pytest
import torch
import triton
import triton.language as tl
from triton.compiler import ASTSource, compile as triton_compile
from triton.backends.compiler import GPUTarget
from triton._internal_testing import is_hip_cdna4

TARGET = GPUTarget("hip", "gfx950", 64)


@triton.jit
def single(counter_ptr, out_ptr, N):
    idx = tl.atomic_add(counter_ptr, 1)
    if idx < N:
        tl.store(out_ptr + idx, idx + 1)


@triton.jit
def steal(counter_ptr, out_ptr, N):
    idx = tl.atomic_add(counter_ptr, 1)
    while idx < N:
        tl.store(out_ptr + idx, idx + 1)
        idx = tl.atomic_add(counter_ptr, 1)


def compile_steal():
    src = ASTSource(
        fn=steal,
        signature={"counter_ptr": "*i32", "out_ptr": "*i32", "N": "i32"},
        constexprs={},
    )
    compiled = triton_compile(src, target=TARGET)
    return compiled.asm["amdgcn"]


def test_fence_precedes_barrier_for_scalar_atomic_in_loop():
    """The scalar atomic's LDS-broadcast barrier must be preceded by the
    MachineSink compiler fence, just like the tensor-atomic case on gfx1250."""

    amdgcn = compile_steal()
    lines = amdgcn.splitlines()

    # The fence itself does not show up as a named instruction, so check
    # instruction order instead: condBr, then the atomic, then the barrier.
    barrier_idx = next((i for i, l in enumerate(lines) if "s_barrier" in l), None)
    assert barrier_idx is not None, "expected an s_barrier broadcasting the atomic result"

    cbranch_idx = next((i for i, l in enumerate(lines) if "s_cbranch" in l), None)
    assert cbranch_idx is not None, "expected an s_cbranch from the atomic's condBr thread masking"

    global_atomic_idx = next((i for i, l in enumerate(lines) if "global_atomic" in l), None)
    assert global_atomic_idx is not None, "expected a global_atomic instruction"

    # Ordering must be: condBr, atomic (inside the masked block), barrier.
    assert cbranch_idx < global_atomic_idx < barrier_idx, (
        "unexpected instruction ordering around the atomic's condBr/barrier; "
        f"s_cbranch={cbranch_idx}, global_atomic={global_atomic_idx}, s_barrier={barrier_idx}")


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 (MI350-class) hardware")
def test_gpu_work_stealing_terminates():
    """Exact repro from #11037: single-atomic control must pass, and the
    work-stealing while-loop kernel must terminate with correct output."""

    n_progs, N = 1024, 1024
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.zeros(N, dtype=torch.int32, device="cuda")
    single[(n_progs, )](counter, out, N)
    torch.cuda.synchronize()
    assert torch.equal(out, torch.arange(1, N + 1, dtype=torch.int32, device="cuda"))

    n_progs, N = 64, 50000
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.zeros(N, dtype=torch.int32, device="cuda")
    steal[(n_progs, )](counter, out, N)
    torch.cuda.synchronize()
    assert int(counter[0]) == N
    np.testing.assert_array_equal(out.cpu().numpy(), np.arange(1, N + 1, dtype=np.int32))
