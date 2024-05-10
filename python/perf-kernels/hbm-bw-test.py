"""
Simple test to measure achieved HBM bandwidth.
This kernel moves N bytes of data from one region in HBM to another, using Triton.
"""

# %%
# Compute Kernel
# --------------

import argparse
import sys
import torch

import triton
import triton.language as tl


@triton.jit
def copy_kernel(
    input_ptr,  # *Pointer* to input vector.
    output_ptr,  # *Pointer* to output vector.
    NUM_ELEMENTS: tl.constexpr,  # Total elements to move.
    BLOCK_SIZE: tl.constexpr,  # Elements to load / store per iteration
    VECTOR_SIZE: tl.constexpr,  # Size of the entire vector being moved.
    READ_ONLY: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    # Offset at which to start for this WG.
    lo = pid * NUM_ELEMENTS
    # Offset until which to read for this WG.
    hi = lo + NUM_ELEMENTS
    # NUM_ITERS: tl.constexpr = triton.cdiv(NUM_ELEMENTS, BLOCK_SIZE)
    IRREGULAR_SIZE: tl.constexpr = NUM_ELEMENTS % BLOCK_SIZE
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    if IRREGULAR_SIZE:
        hi = hi - IRREGULAR_SIZE
    # Move buffer in chunks of block_size
    for idx in range(lo, hi, BLOCK_SIZE):
        offsets = idx + tl.arange(0, BLOCK_SIZE)
        in_vals = tl.load(input_ptr + offsets)
        acc += in_vals
        if not READ_ONLY:
            tl.store(output_ptr + offsets, in_vals)
    # Unroll last irregular iter in case the total sized moved by this WG
    # is not a multiple of block size.
    if IRREGULAR_SIZE:
        lo = hi
        hi = hi + IRREGULAR_SIZE
        offsets = lo + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hi
        in_vals = tl.load(input_ptr + offsets, mask=mask)
        if not READ_ONLY:
            tl.store(output_ptr + offsets, in_vals, mask=mask)

    if READ_ONLY:
        tl.store(output_ptr + tl.arange(0, BLOCK_SIZE), acc)


def copy(src: torch.Tensor, block_size, wgs, dst: torch.Tensor):
    assert src.is_cuda
    vector_size = src.numel()
    assert dst.numel() == vector_size or dst.numel() == block_size
    size_per_wg = vector_size / wgs
    assert size_per_wg >= block_size, \
           "Too many WGS. Please increase the size of the buffer using -size." \
           f" We want a buffer of size {wgs * block_size} f32 elements or larger."
    grid = (wgs, 1, 1)
    # Each WG will move these many elements
    n_elements = triton.cdiv(vector_size, wgs)
    # If we want to read only, we do a dummy write of a single block size back to HBM
    read_only = dst.numel() != src.numel()
    copy_kernel[grid](
        src,
        dst,
        NUM_ELEMENTS=n_elements,
        BLOCK_SIZE=block_size,
        VECTOR_SIZE=vector_size,
        READ_ONLY=read_only,
        num_warps=4,
    )


def get_reference(x, wgs, gbps):
    ms = triton.testing.do_bench(lambda: torch.clone(x))
    bw = gbps(ms)
    triton_output = torch.empty_like(x)
    copy(x, block_size=16384, wgs=wgs, dst=triton_output)
    err = triton_output - x
    if torch.count_nonzero(err):
        assert False, f"Torch and Triton do not match - max error is "\
                      f"{torch.max(torch.abs(err))}"
    return bw


def align_size_to_wgs(size, wgs):
    return (size // wgs) * wgs


def run_benchmark_suite(vector_size, block_size, num_cores, read_only):
    configs = []
    # Define WGs in powers of 2 from 1 - 2048.
    x_vals = [(2**i) for i in range(0, 12)]
    num_cu_aligned_wgs = [(num_cores * i) for i in range(1, 5)]
    import bisect
    for i in num_cu_aligned_wgs:
        bisect.insort(x_vals, i)
    configs.append(
        triton.testing.Benchmark(
            x_names=['wgs'],  # Argument names to use as an x-axis for the plot.
            x_vals=x_vals, x_log=True,  # x axis is logarithmic.
            line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
            line_vals=['triton'],  # Possible values for `line_arg`.
            line_names=['Triton'],  # Label name for the lines.
            styles=[('blue', '-'), ('green', '-')],  # Line styles.
            ylabel='GiB/s',  # Label name for the y-axis.
            plot_name=f'size={vector_size}',  # Name for the plot. Used also as a file name for saving the plot.
            args={'size': vector_size},  # Values for function arguments not in `x_names` and `y_name`.
        ))

    @triton.testing.perf_report(configs)
    def benchmark(size, provider, wgs):
        aligned_size = align_size_to_wgs(size, wgs)
        src_tensor = torch.randn(aligned_size, device='cuda')
        dst_tensor = torch.empty(block_size, device='cuda')
        if not read_only:
            dst_tensor = torch.empty_like(src_tensor)
        ms = triton.testing.do_bench(lambda: copy(src_tensor, block_size, wgs, dst_tensor))
        # 8 because 4 bytes from load, 4 from store.
        if read_only:
            gbps = lambda ms: 4 * size / ms * 1e3 / 1024**3
        else:
            gbps = lambda ms: 8 * size / ms * 1e3 / 1024**3
        return gbps(ms)

    benchmark.run(print_data=True, show_plots=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="HBM Bandwidth Benchmark",
        allow_abbrev=False,
    )
    parser.add_argument("-direction", type=str, default="read-only",
                        help="Data movement direction: read-only, read-write")
    parser.add_argument("-size", type=int, default=1024, help="Size of buffer moved, in MiB")
    parser.add_argument("-num_wgs", type=int, default=0, help="Number of workgroups to use")
    parser.add_argument("-block_size", type=int, default=16384, help="Block size per iteration to load / store")
    parser.add_argument("-run_sweep", action='store_true', default=False, help="Run sweep of B/W vs workgroups")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)
    num_cores = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    size = args.size
    rw = args.direction == "read_write"
    num_elements = size * 1024 * 1024 // 4
    if args.run_sweep:
        assert args.num_wgs == 0, "If running the benchmark suite, please do not specify the number of WGs to use."
        run_benchmark_suite(num_elements, args.block_size, num_cores, not rw)
        return
    if args.num_wgs == 0:
        # num_wgs not user specified - get from device properties
        num_wgs = num_cores
        print(f"Using {num_wgs} workgroups. It is recommended to "\
               "use -num_wgs to provide this number.")
    else:
        assert args.num_wgs > 0, "Please provide a positive, non-zero number of workgroups!"
        num_wgs = args.num_wgs
        if num_wgs % num_cores:
            print(f"Note! Your device has {num_cores} cores. It is recommended to use"\
                   " a number for workgroups that is a multiple of this number."\
                   f" You have currently chosen {num_wgs}.")
    num_elements_rounded = align_size_to_wgs(num_elements, num_wgs)
    if num_elements != num_elements_rounded:
        print(f"Removing last {num_elements - num_elements_rounded} elements to "\
            "get a tensor size aligned to multiple of number of workgroups.")
    num_elements = num_elements_rounded
    src_tensor = torch.randn(num_elements, device="cuda")
    if rw:
        # 8 because 4B for read. 4B for write.
        gbps = lambda ms: 8 * num_elements / ms * 1e3 / 1024**3
        ref_bw = get_reference(src_tensor, num_wgs, gbps)
        print(f"Reference PyTorch bandwidth = {ref_bw} GiB/s")
    else:
        gbps = lambda ms: 4 * num_elements / ms * 1e3 / 1024**3
    if size < 1024:
        print("Note! It is recommended to use a buffer larger than 1 GiB.")
    if num_elements % args.block_size:
        print("Note! This config is suboptimal. It is recommended to use a buffer that"\
               f" is a multiple of wgs x block size = {num_wgs * args.block_size} elements.")
    dst_tensor = torch.empty_like(src_tensor) if rw else torch.empty(args.block_size, device='cuda')
    triton_ms = triton.testing.do_bench(lambda: copy(src_tensor, args.block_size, num_wgs, dst=dst_tensor), warmup=1,
                                        rep=1)
    print(f"Triton bandwidth = {gbps(triton_ms)} GiB/s")


if __name__ == '__main__':
    sys.exit(main())
