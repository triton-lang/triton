import torch

import triton
import triton.language as tl

@triton.jit
def add(x, y):
    return x + y

@triton.jit
def sub(x, y):
    return x - y

@triton.jit
def binary_kernel(x_ptr,
               y_ptr,
               fn_name: tl.constexpr,
               output_ptr,
               n_elements,
               BLOCK_SIZE: tl.constexpr,
               ):
    
    if fn_name == "add":
        FN: tl.constexpr = add
    elif fn_name == "sub":
        FN: tl.constexpr = sub
    else:
        tl.static_assert(False, f"Invalid {fn_name=}")
        
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = FN(x, y)
    tl.store(output_ptr + offsets, output, mask=mask)

def binary(x: torch.Tensor, y: torch.Tensor, fn_name: str):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    binary_kernel[grid](x, y, fn_name, output, n_elements, BLOCK_SIZE=1024)
    return output

def main():
    torch.manual_seed(0)
    size = 1
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')
    output_torch = x + y
    output_triton = binary(x, y, "add")
    print("torch", output_torch)
    print("triton", output_triton)


if __name__ == "__main__":
    main()