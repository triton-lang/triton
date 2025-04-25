import torch
import triton
import argparse
import ctypes
import struct
import triton.language as tl
import triton.profiler as proton
import triton.profiler.language as pl
from triton.profiler.hooks import InstrumentationHook

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def write_tensor_to_file(num_blocks, metadata, tensor, filename):
    data_ptr = tensor.data_ptr()
    size = tensor.numel()
    dtype_size = tensor.element_size()
    total_bytes = size * dtype_size

    with open(filename, 'wb') as f:
        f.write(struct.pack('III', num_blocks, metadata.num_warps, metadata.profile_scratch_size))

        data_arr = ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_ubyte * total_bytes))

        f.write(bytes(data_arr.contents))


@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    pid = tl.program_id(axis=0)
    pl.enter_scope("r0")
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    pl.enter_scope("r1")
    y = tl.load(y_ptr + offsets, mask=mask)
    pl.enter_scope("r2")
    pl.exit_scope("r1")
    pl.exit_scope("r0")
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
    pl.exit_scope("r2")


def add(args):
    torch.manual_seed(0)
    size = 2048
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    proton.start("", backend="instrumentation")
    InstrumentationHook.enable_host_buffer = True
    pgm = add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    write_tensor_to_file(2, pgm.metadata, InstrumentationHook.host_buffer, args.trace_file)
    proton.finalize()


@triton.jit
def loop_kernel():
    for k in range(0, 20):
        pl.enter_scope("r0")
        pl.exit_scope("r0")


def loop(args):
    grid = (1, )
    proton.start("", backend="instrumentation", mode=proton.mode.Default(buffer_size=256))
    InstrumentationHook.enable_host_buffer = True
    pgm = loop_kernel[grid]()
    write_tensor_to_file(1, pgm.metadata, InstrumentationHook.host_buffer, args.trace_file)
    proton.finalize()


def main():
    parser = argparse.ArgumentParser(description='Proton intra kernel profiler trace generator')
    parser.add_argument('trace_file', type=str, help='Trace file path')
    parser.add_argument('--kernel', '-k', type=str, help='Kernel name')
    args = parser.parse_args()

    if args.kernel == "add":
        add(args)
    if args.kernel == "loop":
        loop(args)


if __name__ == '__main__':
    main()
