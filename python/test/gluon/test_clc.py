import torch
import pytest

from triton._internal_testing import is_blackwell
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import mbarrier
from triton.experimental.gluon.language.nvidia.blackwell import clc


@pytest.mark.parametrize("num_ctas", [1, 2])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_clc_basic(num_ctas):

    @gluon.jit
    def clc_kernel(WasLaunched, IsCancelled, ProgramId, smem_size: gl.constexpr):
        # Large shared memory allocation to force 1 block per SM
        cga_layout: gl.constexpr = [[0]] if gl.num_ctas() == 2 else []
        unswizzled_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0], cga_layout=cga_layout)
        dummy_alloc = gl.allocate_shared_memory(gl.int64, [smem_size // 8 - 32], unswizzled_layout)

        # Try to cancel another launch
        clc_result = gl.allocate_shared_memory(gl.int64, [2], unswizzled_layout)
        clc_mbar = mbarrier.allocate_mbarrier()
        mbarrier.init(clc_mbar, count=1)

        clc.try_cancel(clc_result, clc_mbar, multicast=True)
        mbarrier.expect(clc_mbar, 16)
        mbarrier.wait(clc_mbar, 0)

        clc_response = clc.load_result(clc_result)
        pid = gl.program_id(0)
        gl.store(WasLaunched + pid, True)
        gl.store(IsCancelled + pid, clc_response.is_canceled())
        gl.store(ProgramId + pid, clc_response.program_id(0))
        dummy_alloc._keep_alive()

    device = "cuda"
    dev_props = torch.cuda.get_device_properties(device)
    num_sms = dev_props.multi_processor_count
    smem_size = dev_props.shared_memory_per_block_optin // num_ctas

    grid = 2 * (num_sms // num_ctas)
    was_launched = torch.zeros([grid], dtype=torch.bool, device=device)
    is_cancelled = torch.zeros([grid], dtype=torch.bool, device=device)
    program_ids = torch.zeros([grid], dtype=torch.int32, device=device)
    clc_kernel[(grid, )](was_launched, is_cancelled, program_ids, smem_size, num_ctas=num_ctas)

    num_launched = torch.sum(was_launched).item()
    assert num_launched < grid

    num_cancelled = torch.sum(is_cancelled).item()
    assert num_launched + num_cancelled == grid

    for pid in range(grid):
        if is_cancelled[pid]:
            assert not was_launched[program_ids[pid]]
