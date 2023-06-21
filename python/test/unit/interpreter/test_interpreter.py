import random

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
