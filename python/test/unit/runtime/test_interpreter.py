import pytest
import torch

import triton
import triton.language as tl


@pytest.mark.interpreter
def test_zero_strided_interpreter_tensors(device):
    @triton.jit
    def _simple_add(
        X,
        stride_x_a,
        stride_x_b,
    ):
        pid_a = tl.program_id(0)
        pid_b = tl.program_id(1)

        # doesn't directly index c dim, so relies on 0-strided c dim to affect every element
        x_ptr = X + pid_a * stride_x_a + pid_b * stride_x_b

        tl.atomic_add(x_ptr, 1)

    x = torch.zeros((2, 2, 1), device=device)
    c_dim = 3
    x = x.expand((2, 2, c_dim))

    a, b, c = x.shape
    grid = (a, b, c)
    with torch.cuda.device(x.device.index):
        _simple_add[grid](
            x,
            x.stride(0),
            x.stride(1)
        )

    assert torch.allclose(x, torch.ones_like(x) * c_dim)
