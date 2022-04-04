import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,  # *Pointer* to first input vector
    y_ptr,  # *Pointer* to second input vector
    output_ptr,  # *Pointer* to output vector
    n_elements,  # Size of the vector
    K,
    stride
    # BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process
    #              # NOTE: `constexpr` so it can be used as a shape value
):
    # There are multiple 'program's processing different data. We identify which program
    # we are here
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0
    # This program will process inputs that are offset from the initial data.
    # for instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers
    block_start = pid * 256
    offsets = block_start + tl.arange(0, 256)
    # Create a mask to guard memory operations against out-of-bounds accesses
    mask = offsets < n_elements

    x_ptrs = x_ptr + offsets
    y_ptrs = y_ptr + offsets
    output = tl.zeros((256,), dtype=tl.float32)
    for k in range(0, K, 32):
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        y = tl.load(y_ptrs, mask=mask, other=0.0)
        output += x + y

        x_ptrs += stride
        y_ptrs += stride

    # Write x + y back to DRAM
    tl.store(output_ptr + offsets, output, mask=mask)

size = 1024
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')
z = torch.empty_like(x)
# add_kernel[(1,)](x, y, z, size, 256)
# print(add_kernel[(1,)].kernel.compile_to_ttir())
mod, ctx = add_kernel.compile_to_ttir(x, y, z, size, 128, 8, grid=(1,))
mod.get_context()
mod.dump()
# print(mod)
