import pytest
import torch
import triton
import triton.language as tl


@pytest.mark.interpreter
def test_complex_conditionals(device):
    # Example that will fail because of "or"
    # Guess it's related to order of evaluating/visiting nodes in Triton front-end.
    @triton.jit
    def test_if_or(x_ptr, out_ptr):
        x = tl.load(x_ptr)
        if (((x.dtype == tl.float32) or (x.dtype == tl.float16))
                and not (x.dtype == tl.float32)) and (x.dtype == tl.float16 and x.dtype == tl.float32):
            tl.store(out_ptr, 0)
        else:
            tl.store(out_ptr, 1)

    x_float = torch.randn((32, ), dtype=torch.float32, device=device)
    x_int = torch.randn((32, ), dtype=torch.float32, device=device).to(torch.int32)
    out_ptr = torch.empty((1, ), dtype=torch.int32, device=device)

    test_if_or[(8, 8)](x_float, out_ptr)
    assert out_ptr == 0, "Expected 0"
    test_if_or[(8, 8)](x_int, out_ptr)
    assert out_ptr == 1, "Expected 1"
