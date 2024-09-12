import torch
import triton
import triton.language as tl
import numpy as np
import triton.profiler as proton

enable_profiling = True
file_name = "add_kernel.ttgir"
BLOCK = 4096
SLOT = 32
WG = 1


@triton.jit(noinline=True)
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               profile_mem,  # *Pointer* to profile memory.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = (int(n_elements / BLOCK), 1, 1)

    # We preallocate the profile memory.
    pconfig = proton.IntraKernelConfig(slots=SLOT, header=3, wg_num=WG, word_per_slot=2)
    scratch = proton.intra_kernel_smem(pconfig)
    profile_mem = torch.empty((np.prod(grid) * scratch), device="cuda", dtype=torch.uint32)

    if not enable_profiling:
        add_kernel[grid](x, y, output, n_elements, BLOCK, profile_mem, num_warps=4, proton_slots=SLOT)
        for sig in add_kernel.cache[0]:
            kernel = add_kernel.cache[0][sig]
            # This gives the kernel's TTGIR source code so we can add proton_record to it.
            print(kernel.asm['ttgir'])
    else:
        exp_kernel = triton.compile(file_name)
        # Note that the BLOCK argument is constant-folded.
        # See visit_FunctionDef in python/triton/compiler/code_generator.py
        exp_kernel[grid](x, y, output, n_elements, profile_mem)
        proton.dump_chrome_trace(np.prod(grid), pconfig, profile_mem, "chrome_trace.json")
    return output


torch.manual_seed(0)
size = 2**18
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')

output_triton = add(x, y)
print(output_triton)
