import torch
import pytest

from triton._internal_testing import is_blackwell
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import mbarrier
from triton.experimental.gluon.language.nvidia.blackwell import clc


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_clc_basic():
    """Test CLC try_cancel, load_result, is_canceled, get_first_ctaid ops."""

    @gluon.jit
    def clc_kernel(Out):
        pid = gl.program_id(0)

        # Allocate shared memory for CLC result (128-bit) and mbarrier
        clc_result = gl.allocate_shared_memory(gl.int64, [2], mbarrier.MBarrierLayout())
        clc_mbar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
        mbarrier.init(clc_mbar, count=1)

        # Issue CLC try_cancel (will fail since no pending clusters, but tests the op)
        clc.try_cancel(clc_result, clc_mbar)
        mbarrier.expect(clc_mbar, 16)
        mbarrier.wait(clc_mbar, 0)

        # Load result into registers and query
        clc_response = clc.load_result(clc_result)
        is_canceled = clc_response.is_canceled()
        ctaid_x = clc_response.get_first_ctaid(0)

        # Write results
        out_ptr = Out + pid * 2
        gl.store(out_ptr, is_canceled)  # Store bool directly
        gl.store(out_ptr + 1, ctaid_x)

    out = torch.zeros(2, dtype=torch.int32, device="cuda")
    clc_kernel[(1,)](out)

    # is_canceled should be 0 (False) since there are no pending clusters
    assert out[0].item() == 0, f"Expected is_canceled=0, got {out[0].item()}"
