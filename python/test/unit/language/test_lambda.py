import pytest
import torch

import triton
import triton.language as tl
from triton.compiler.errors import UnsupportedLanguageConstruct


def test_associative_scan_lambda_cumsum(device):
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32, device=device)
    y = torch.empty_like(x)

    @triton.jit
    def kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
        offsets = tl.arange(0, BLOCK)
        mask = offsets < n_elements
        val = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        result = tl.associative_scan(val, axis=0, combine_fn=lambda a, b: a + b)
        tl.store(y_ptr + offsets, result, mask=mask)

    kernel[(1, )](x, y, x.numel(), BLOCK=8)

    torch.testing.assert_close(y, torch.cumsum(x, dim=0), rtol=1e-3, atol=1e-5)


def test_associative_scan_lambda_max(device):
    x = torch.tensor([1.0, 3.0, 2.0, 5.0, 4.0], dtype=torch.float32, device=device)
    y = torch.empty_like(x)

    @triton.jit
    def kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
        offsets = tl.arange(0, BLOCK)
        mask = offsets < n_elements
        val = tl.load(x_ptr + offsets, mask=mask, other=float("-inf"))
        result = tl.associative_scan(val, axis=0, combine_fn=lambda a, b: tl.maximum(a, b))
        tl.store(y_ptr + offsets, result, mask=mask)

    kernel[(1, )](x, y, x.numel(), BLOCK=8)

    torch.testing.assert_close(y, torch.cummax(x, dim=0).values, rtol=1e-3, atol=1e-5)


def test_reduce_lambda_product(device):
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device=device)
    y = torch.empty(1, dtype=torch.float32, device=device)

    @triton.jit
    def kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
        offsets = tl.arange(0, BLOCK)
        mask = offsets < n_elements
        val = tl.load(x_ptr + offsets, mask=mask, other=1.0)
        result = tl.reduce(val, axis=0, combine_fn=lambda a, b: a * b)
        tl.store(y_ptr, result)

    kernel[(1, )](x, y, x.numel(), BLOCK=4)

    expected = torch.tensor([24.0], dtype=torch.float32, device=device)
    torch.testing.assert_close(y, expected, rtol=1e-3, atol=1e-5)


def test_multiple_lambdas_in_same_kernel(device):
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
    y = torch.empty_like(x)

    @triton.jit
    def kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
        offsets = tl.arange(0, BLOCK)
        mask = offsets < n_elements
        val = tl.load(x_ptr + offsets, mask=mask, other=float("inf"))
        tl.reduce(val, axis=0, combine_fn=lambda a, b: a + b)
        scan_result = tl.associative_scan(val, axis=0, combine_fn=lambda a, b: tl.minimum(a, b))
        tl.store(y_ptr + offsets, scan_result, mask=mask)

    kernel[(1, )](x, y, x.numel(), BLOCK=4)

    expected = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
    torch.testing.assert_close(y, expected, rtol=1e-3, atol=1e-5)


def test_lambda_constexpr_capture(device):
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
    y = torch.empty_like(x)

    @triton.jit
    def kernel(x_ptr, y_ptr, n_elements, SCALE: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.arange(0, BLOCK)
        mask = offsets < n_elements
        val = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        result = tl.associative_scan(val, axis=0, combine_fn=lambda a, b: a + b + SCALE)
        tl.store(y_ptr + offsets, result, mask=mask)

    kernel[(1, )](x, y, x.numel(), SCALE=2.0, BLOCK=4)

    expected = torch.tensor([1.0, 5.0, 10.0], dtype=torch.float32, device=device)
    torch.testing.assert_close(y, expected, rtol=1e-3, atol=1e-5)


def test_reduce_lambda_constexpr_capture(device):
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
    y = torch.empty(1, dtype=torch.float32, device=device)

    @triton.jit
    def kernel(x_ptr, y_ptr, n_elements, SCALE: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.arange(0, BLOCK)
        mask = offsets < n_elements
        val = tl.load(x_ptr + offsets, mask=mask, other=-SCALE)
        result = tl.reduce(val, axis=0, combine_fn=lambda a, b: a + b + SCALE)
        tl.store(y_ptr, result)

    kernel[(1, )](x, y, x.numel(), SCALE=2.0, BLOCK=4)

    expected = torch.tensor([10.0], dtype=torch.float32, device=device)
    torch.testing.assert_close(y, expected, rtol=1e-3, atol=1e-5)


def test_lambda_tensor_capture_is_rejected(device):
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
    y = torch.empty_like(x)

    @triton.jit
    def kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
        offsets = tl.arange(0, BLOCK)
        mask = offsets < n_elements
        val = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        result = tl.associative_scan(val, axis=0, combine_fn=lambda a, b: a + b + val)
        tl.store(y_ptr + offsets, result, mask=mask)

    with pytest.raises(UnsupportedLanguageConstruct, match="lambda can only capture constexpr values"):
        kernel[(1, )](x, y, x.numel(), BLOCK=4)


def test_lambda_runtime_scalar_capture_is_rejected(device):
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
    y = torch.empty_like(x)

    @triton.jit
    def kernel(x_ptr, y_ptr, n_elements, runtime_scale, BLOCK: tl.constexpr):
        offsets = tl.arange(0, BLOCK)
        mask = offsets < n_elements
        val = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        result = tl.associative_scan(val, axis=0, combine_fn=lambda a, b: a + b + runtime_scale)
        tl.store(y_ptr + offsets, result, mask=mask)

    with pytest.raises(UnsupportedLanguageConstruct, match="lambda can only capture constexpr values"):
        kernel[(1, )](x, y, x.numel(), x.sum(), BLOCK=4)
