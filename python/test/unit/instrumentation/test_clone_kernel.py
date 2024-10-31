import numpy as np
import pytest
import torch
import os

import triton
import triton.language as tl
from triton.language.extra import libdevice

def test_clone_kernel(device):
    @triton.jit
    def _kernel(out_ptr, N: tl.constexpr, BLOCK_N: tl.constexpr):
        current_size = N - tl.program_id(0) * BLOCK_N
        tl.assume(current_size >= BLOCK_N)
        if current_size >= 128:
            tl.store(out_ptr + tl.program_id(0), current_size)
        else:
            tl.store(out_ptr + tl.program_id(0), current_size + 101024)

    output = torch.zeros(1024 // 128, device=device)
    pgm = _kernel[(1024 // 128, )](output, N=1024, BLOCK_N=128)
    kernel_signatures = []
    for line in pgm.asm['llir'].split("\n"):
        if "amdgpu_kernel" in line:
            kernel_signatures.append(line.strip())

    if 'LLVM_PASS_PLUGIN_PATH' in os.environ and os.environ['LLVM_PASS_PLUGIN_PATH'].split("/")[-1] == "libCloneKernelAndAugmentArgsLib.so":
        assert kernel_signatures == ['define amdgpu_kernel void @_kernel(ptr addrspace(1) inreg nocapture writeonly %0) local_unnamed_addr #0 !dbg !4 {', 
                                     'define amdgpu_kernel void @_kernelPv(ptr addrspace(1) inreg nocapture writeonly %0, ptr %1) local_unnamed_addr #0 !dbg !15 {']
    else:
        assert kernel_signatures == ["define amdgpu_kernel void @_kernel(ptr addrspace(1) inreg nocapture writeonly %0) local_unnamed_addr #0 !dbg !4 {"]
