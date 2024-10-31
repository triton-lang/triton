import torch
import os

import triton
import triton.language as tl


def test_clone_kernel(device):

    @triton.jit
    def _kernel(out_ptr, N: tl.constexpr, BLOCK_N: tl.constexpr):
        size = N - tl.program_id(0) * BLOCK_N
        tl.assume(size >= BLOCK_N)
        tl.store(out_ptr + tl.program_id(0), size)

    output = torch.zeros(1024 // 128, device=device)
    pgm = _kernel[(1024 // 128, )](output, N=1024, BLOCK_N=128)
    kernel_signatures = []
    for line in pgm.asm['llir'].split("\n"):
        if "define" in line:
            stack = []
            matches = []

            for i, char in enumerate(line):
                if char == '(':
                    stack.append(i)
                elif char == ')':
                    if stack:
                        start_index = stack.pop()
                        matches.append((start_index, i))
            kernel_signatures.append(line.strip()[matches[-1][0] + 1:matches[-1][1]])

    if 'LLVM_PASS_PLUGIN_PATH' in os.environ and "libCloneKernelAndAugmentArgsLib.so" in os.environ['LLVM_PASS_PLUGIN_PATH']:    
        if (triton.runtime.driver.active.get_current_target().backend == 'hip'):
            assert kernel_signatures == [
                'ptr addrspace(1) inreg nocapture writeonly %0', 'ptr addrspace(1) inreg nocapture writeonly %0, ptr %1'
            ]
        else:
            assert kernel_signatures == ['ptr addrspace(1) %0', 'ptr addrspace(1) %0, ptr %1']
    else:
        if (triton.runtime.driver.active.get_current_target().backend == 'hip'):
            assert kernel_signatures == ["ptr addrspace(1) inreg nocapture writeonly %0"]
        else:
            assert kernel_signatures == ["ptr addrspace(1) %0"]
