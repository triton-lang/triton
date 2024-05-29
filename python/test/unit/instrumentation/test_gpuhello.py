import torch

import pytest
import os

import triton
import triton.language as tl

test_stdout = 'Hello From First Instruction of GPU Kernel: kernel1\ttest_gpuhello.py:15:4\n\
Hello From First Instruction of GPU Kernel: kernel2\ttest_gpuhello.py:19:4\n\
Hello From First Instruction of GPU Kernel: kernel3\ttest_gpuhello.py:23:4\n'

@triton.jit
def kernel1(BLOCK_SIZE: tl.constexpr):
    return

@triton.jit
def kernel2(BLOCK_SIZE: tl.constexpr):
    return

@triton.jit
def kernel3(BLOCK_SIZE: tl.constexpr):
    return

def func(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    kernel1[grid](BLOCK_SIZE=1024)
    kernel2[grid](BLOCK_SIZE=1024)
    kernel3[grid](BLOCK_SIZE=1024)

def test_op(capfd):
    TRITON_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    os.environ["LLVM_PASS_PLUGIN_PATH"] = os.path.join(TRITON_PATH, "triton/_C/libGPUHello.so")
    size = 98432
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')
    func(x, y)
    stdout, stderr = capfd.readouterr()
    assert repr(stderr) == repr(test_stdout)
    os.environ["LLVM_PASS_PLUGIN_PATH"] = ""
