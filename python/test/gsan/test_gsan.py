from __future__ import annotations

import pytest
import torch
import triton

from triton._internal_testing import is_cuda
from triton.experimental.gsan import create_mem_pool
from triton._C.libtriton.gsan_testing import ScalarClock, AtomicScope
from triton.experimental.gsan._testing_utils import (load_one_i32, shadow_cell_from_address, store_one_i32,
                                                     thread_state_from_smid)


@pytest.mark.skipif(not is_cuda(), reason="GSan requires CUDA")
def test_load_store_updates_shadow(fresh_knobs):
    pool = create_mem_pool()
    with torch.cuda.use_mem_pool(pool):
        target = torch.zeros(1, dtype=torch.int32, device="cuda")
        scratch = torch.zeros(1, dtype=torch.int32, device="cuda")

    triton.knobs.compilation.instrumentation_mode = "gsan"
    store_one_i32[(1, )](target, num_warps=1)
    cell0 = shadow_cell_from_address(target.data_ptr())

    tid = cell0.write_clock.thread_id
    vc = thread_state_from_smid(tid).vector_clock
    print(tid)
    print(vc)
    print(vc[tid])
    print(vc[100:150])
    epoch0 = thread_state_from_smid(tid).vector_clock[tid]

    assert cell0.write_clock.thread_id == tid
    assert cell0.write_clock.epoch == epoch0
    assert cell0.read_clocks[0].thread_id == 0
    assert cell0.read_clocks[0].epoch == 0
    assert cell0.num_reads == 0

    load_one_i32[(1, )](target, scratch, num_warps=1)
    cell1 = shadow_cell_from_address(target.data_ptr())
    epoch1 = thread_state_from_smid(tid).vector_clock[tid]

    assert epoch1 == epoch0 + 1
    assert cell1.write_clock == cell0.write_clock
    assert cell1.read_clocks[0] == ScalarClock(epoch1, tid, AtomicScope.NON_ATOMIC)
    # 1 read per thread. TODO: Add uniform path?
    assert cell1.num_reads == 32
