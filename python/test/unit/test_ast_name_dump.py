import torch

import pytest
import os
import re

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@pytest.mark.parametrize(None, [None])
@triton.jit
def kernel1(x_ptr, y_ptr, output_ptr, n_elements,
               BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    output = x + y
    tl.store(output_ptr + offsets, output)

@pytest.mark.parametrize(None, [None])
def run_kernel1(BLOCK_SIZE: int, device: str):
    size = 98432
    x = torch.rand(size, device=device)
    y = torch.rand(size, device=device)
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    k = kernel1[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return k

def test_dump_source_var_names(capfd, monkeypatch, device: str):
    check_strs = ['Triton Source Value Name:',
                  'pid',
                  'block_start',
                  'offsets',
                  'x',
                  'y',
                  'output']

    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    monkeypatch.setenv("TRITON_DUMP_SOURCE_VAR_NAME", '1')

    run_kernel1(BLOCK_SIZE=512, device=device)

    stdout, stderr = capfd.readouterr()
    for s in check_strs:
        if s not in stdout:
            print(f'{s} is not in stdout:\n{stdout}')
            assert False and 'Failed to file expected value names'

# @pytest.mark.parametrize(None, [None])
def test_use_name_loc_as_prefix(capfd, monkeypatch, device: str):
    check_strs = ['%pid = tt.get_program_id x',
                  '%block_start = arith.muli %pid',
                  'tt.addptr %arg0, %block_start',
                  '%x =',
                  '%y =',
                  '%output = arith.addf %x, %y']

    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    monkeypatch.setenv("USE_NAMELOC_AS_PREFIX", '1')

    k = run_kernel1(BLOCK_SIZE=1024, device=device)
    asm = k.asm['ttgir']
    for s in check_strs:
        if s not in asm:
            print(f'{s} is not in asm:\n{asm}')
            assert False and 'Failed to file expected value names'
