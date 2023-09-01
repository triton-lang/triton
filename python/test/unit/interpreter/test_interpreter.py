import random

import pytest
import torch

import triton
import triton.language as tl
from triton.interpreter.interpreter import program_ids_from_grid


def test_addition():

    @triton.jit(interpret=True)
    def add_kernel(
        x_ptr,
        y_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)

    a = torch.rand((128,), device="cuda")
    b = torch.rand((128,), device="cuda")
    expected = a + b
    output = torch.empty((128,), device="cuda")

    def grid(meta):
        return (triton.cdiv(128, meta["BLOCK_SIZE"]),)

    add_kernel[grid](a, b, output, 128, BLOCK_SIZE=32)

    assert torch.allclose(expected, output, atol=1e-2, rtol=0)


def test_program_ids_from_grid():
    random.seed(123)
    grid = (3, 4)
    expected_combinations = 3 * 4
    unique_combinations = set(program_ids_from_grid(grid))
    assert len(unique_combinations) == expected_combinations

    first_run = list(program_ids_from_grid(grid))
    second_run = list(program_ids_from_grid(grid))
    assert first_run != second_run


def test_atomic():
    @triton.jit(interpret=True)
    def atomic(
        x_ptr,
    ):
        pid = tl.program_id(axis=0)
        tl.atomic_add(x_ptr + pid, 1)
        t = tl.atomic_xchg(x_ptr + pid, 3)
        t += 1  # 2
        tl.atomic_cas(x_ptr + pid, 3, t)  # match
        tl.atomic_cas(x_ptr + pid, 40, 9)  # no match
    nb_dim = 16
    a = torch.zeros((nb_dim, ), dtype=torch.int32, device="cuda")

    atomic[(nb_dim, )](a)
    assert torch.allclose(a, torch.full_like(a, 2))

def test_args():

    @triton.jit(interpret=True)
    def assign_kernel(
        x_ptr,
        output_ptr,
        n_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, x, mask=mask)

    a = torch.rand((128,), device="cuda")
    expected = a
    output = torch.empty((128,), device="cuda")

    def grid(meta):
        return (triton.cdiv(128, meta["BLOCK_SIZE"]),)

    assign_kernel[grid](a, output, 128, BLOCK_SIZE=32)

    assert torch.allclose(expected, output, atol=1e-2, rtol=0)

    try:
        assign_kernel[grid](a, output, BLOCK_SIZE=32)
        pytest.fail("Should raise exception")
    except TypeError:
        pass

def test_slice():

    @triton.jit(interpret=True)
    def assign_kernel(
        x_ptr,
        output_ptr,
        n_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, x, mask=mask)

    a = torch.rand((128,), device="cuda")[64:]
    expected = a
    output = torch.empty((64,), device="cuda")

    def grid(meta):
        return (triton.cdiv(64, meta["BLOCK_SIZE"]),)

    assign_kernel[grid](a, output, 64, BLOCK_SIZE=32)

    assert torch.allclose(expected, output, atol=1e-2, rtol=0)

def test_constexpr_math():

    @triton.jit(interpret=True)
    def math_kernel(
        add_ptr,
        mul_ptr,
        div_ptr,
        value: tl.constexpr,
        n_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        values = tl.zeros((BLOCK_SIZE,), dtype=tl.float32) + (1 + value + 1)
        tl.store(add_ptr + offsets, values, mask=mask)
        values = tl.zeros((BLOCK_SIZE,), dtype=tl.float32) + (2 * value * 2)
        tl.store(mul_ptr + offsets, values, mask=mask)
        values = tl.zeros((BLOCK_SIZE,), dtype=tl.float32) + (2 / value / 2)
        tl.store(div_ptr + offsets, values, mask=mask)

    add_ptr = torch.empty((128,), device="cuda", dtype=torch.float32)
    mul_ptr = torch.empty((128,), device="cuda", dtype=torch.float32)
    div_ptr = torch.empty((128,), device="cuda", dtype=torch.float32)

    def grid(meta):
        return (triton.cdiv(128, meta["BLOCK_SIZE"]),)

    math_kernel[grid](add_ptr, mul_ptr, div_ptr, 0.5, 128, BLOCK_SIZE=32)

    assert torch.allclose(torch.full_like(add_ptr, 1 + 0.5 + 1), add_ptr, atol=1e-2, rtol=0)
    assert torch.allclose(torch.full_like(mul_ptr, 2 * 0.5 * 2), mul_ptr, atol=1e-2, rtol=0)
    assert torch.allclose(torch.full_like(div_ptr, 2 / 0.5 / 2), div_ptr, atol=1e-2, rtol=0)

def test_full():

    @triton.jit(interpret=True)
    def full_kernel(
        output_ptr,
        n_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        val = tl.full((BLOCK_SIZE,), 1., dtype=tl.float32)
        tl.store(output_ptr + offsets, val, mask=mask)

    expected = torch.full((128,), 1, device="cuda", dtype=torch.float32)
    output = torch.empty((128,), device="cuda")

    def grid(meta):
        return (triton.cdiv(128, meta["BLOCK_SIZE"]),)

    full_kernel[grid](output, 128, BLOCK_SIZE=32)

    assert torch.allclose(expected, output, atol=1e-2, rtol=0)

def test_float64():
    @triton.jit(interpret=True)
    def full_kernel(
        output_ptr,
        n_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        val = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
        tl.store(output_ptr + offsets, val, mask=mask)

    expected = torch.zeros((128,), device="cuda", dtype=torch.float64)
    output = torch.empty((128,), device="cuda", dtype=torch.float64)

    def grid(meta):
        return (triton.cdiv(128, meta["BLOCK_SIZE"]),)

    full_kernel[grid](output, 128, BLOCK_SIZE=32)

    assert torch.allclose(expected, output, atol=1e-2, rtol=0)
