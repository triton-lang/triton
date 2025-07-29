"""
Tensor Layouts
==============

Tensors in Gluon require layouts. Layouts specify how the elements of the tensor
are distributed among the threads a thread block. Tensors are distributed with
respect to the hierarchy of the GPU beginning with thread blocks, then warps,
lanes, and finally individual registers in each lane.

Tensors are evenly distributed across theads, meaning that all threads own the
same number of elements. Because Triton requires that all tile dimensions are
powers of 2, this means that the number of elements per thread is a power of 2.

A layout, in general, defines a mapping from a logical tensor index to the warp,
lane, and register the element lives in. `BlockedLayout` is the most common
kind of layout in Gluon. A `BlockedLayout` defines how elements are organized
in a "block" of the same rank as the tensor. If the block is smaller than the
tensor, it is tiled over the tensor along the registers, and if the block is
larger, the tensor is broadcasted to fit the block along the registers.

Consider the following example:

```python
gl.BlockedLayout(
    size_per_thread=[2, 4],
    threads_per_warp=[16, 2],
    warps_per_cta=[2, 2],
    order=[1, 0],
)
```

By multiplying `size_per_thread`, `threads_per_warp`, and `warps_per_cta`,
elementwise we obtain the block shape `[2 * 16 * 2, 4 * 2 * 2] = [64, 16]`.
Within this block, the layout describes a hierarchy of register, thread, and
warp tiling over the logical elements of the tensor. The `order` specifies the
order in which the dimensions of the tensor are tiled.

In this example, `size_per_thread=[2, 4]` indicates that within each block, each
thread owns a contiguous `2x4` subtile of the tensor, stored as registers in
that thread. `order=[1, 0]` indicates that the layout tiles the columns first
then the rows, i.e. column-major order. For a thread T, the tile looks like:

```
[[T:0, T:1, T:2, T:3],
 [T:4, T:5, T:6, T:7]]
```

Notice that the registers increment over the inner dimension.

If `order` was `[0, 1]` (row-major order), the tile would look like:

```
[[T:0, T:2, T:4, T:6],
 [T:1, T:3, T:5, T:7]]
```

Likewise, `threads_per_warp=[16, 2]` indicates how the tensor elements owned by
a single thread are tiled to obtain the elements owned by a single warp. For
`order=[1, 0]`, the warp tile of threads looks like:

```
[[ T0,  T1],
 [ T2,  T3],
 ...
 [T28, T29],
 [T30, T31]]
```

Note that the size of the warp tile must match the number of threads per warp,
which for NVIDIA hardware is 32. If we substitute each thread with its thread
tile, we obtain the warp tile over the elements of the tensor:

```
[[ T0:0,  T0:1,  T0:2,  T0:3,  T1:0,  T1:1,  T1:2,  T1:3],
 [ T0:4,  T0:5,  T0:6,  T0:7,  T1:4,  T1:5,  T1:6,  T1:7],
 [ T2:0,  T2:1,  T2:2,  T2:3,  T3:0,  T3:1,  T3:2,  T3:3],
 [ T2:4,  T2:5,  T2:6,  T2:7,  T3:4,  T3:5,  T3:6,  T3:7],
 ...
 [T28:0, T28:1, T28:2, T28:3, T29:0, T29:1, T29:2, T29:3],
 [T28:4, T28:5, T28:6, T28:7, T29:4, T29:5, T29:6, T29:7],
 [T30:0, T30:1, T30:2, T30:3, T31:0, T31:1, T31:2, T31:3],
 [T30:4, T30:5, T30:6, T30:7, T31:4, T31:5, T31:6, T31:7]]
```

We can again repeat this process for `warps_per_cta=[2, 2]` to obtain a full
mapping of tensor elements within a block to all the threads in a program.

If the tensor had the same dimensions as the block, then there would be no
implicit broadcasting needed to fit the block into the tensor. But consider a
`128x128xf32` tensor. Dividing the block shape into the tensor shape, we obtain
a `[2, 8]` tiling of the block. The block is tiled according to `order=[1, 0]`
by adding more registers to each thread:

```
[[B0, B1, B2, B3],
 [B4, B5, B6, B7]]
```

In each block, each thread owns 8 registers. Thus over the whole tensor, each
thread owns `8 * 8 = 64` registers. Knowing how many registers a tensor uses
is important for managing register pressure and budget in the kernel.

Consider a tensor smaller than the block, say `32x8xf32`. The number of tiles
at each level of the block does not change, thus even though the tensor has only
`32 * 8 = 256` elements, it will be stored as `64 * 16 = 1024` physical
registers in each program. The tensor is broadcasted along each dimension to fit
the block starting with warps, then threads, then registers.

Dividing the tensor shape into the block shape, we obtain `[2, 2]`. Since this
exactly matches `warps_per_cta=[2, 2]`, this means each warp has a full copy of
the tensor, mapped to its lanes in the same way. From the perspective of the
tensor, this looks like:

```
[[  T0:0| T32:0| T64:0| T96:0, ...,   T1:3| T33:3| T65:3| T97:3],
 [  T0:4| T32:4| T64:4| T96:4, ...,   T1:7| T33:7| T65:7| T97:7],
 ...
 [ T30:0| T62:0| T94:0|T126:0, ...,  T31:3| T63:3| T95:3|T127:3]
 [ T30:4| T62:4| T94:4|T126:4, ...,  T31:7| T63:7| T95:7|T127:7]]
```

There are many different kinds of layouts in Gluon. Many of them are specialized
layouts required for specific operations, like MMA instructions utilizing tensor
cores. Some of them are used to represent the results of manipulating the shape
of tensors via `expand_dims`, `broadcast`, `reshape`, `join`, `split`, etc.

Blocked layouts are typically the most common form of layouts in Gluon. They are
primarily used to represent coalesced layouts for global memory accesses and to
represent certain register layouts for tensors stored in Tensor Memory on
NVIDIA Blackwell GPUs.

Now that we have a basic understanding of what blocked layouts, let's look at an
example of how layouts can affect the performance of the kernel by expanding on
the `memcpy` example from the previous tutorial. Using a `BlockedLayout`, we
will have each program load and store a whole tile rather than one scalar.
"""

import pytest
import torch
import triton
from functools import partial
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

# %%
# This is a helper for toggling specific parts of the tutorial. Run the tutorial
# with `python 02-layouts.py` to run everything, but you can select specific
# parts with `python 02-layouts.py R_vs_throughput,LDG_STG_instructions`.


def _enabled(label):
    from sys import argv
    return len(argv) == 1 or label in argv[1].split(",")


# %%
# Parameterize the kernel over the layout so we can test different layouts. Each
# program copies a block of data, but this we will use the layout to distribute
# the work over all the threads insteead of looping.


@gluon.jit
def memcpy_1d_kernel(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr, layout: gl.constexpr):
    # Compute the start to the block the program will copy like before.
    pid = gl.program_id(0)
    start = pid * XBLOCK

    # Create a tensor with the values [0, XBLOCK) using the provided layout.
    # This tensor collectively represents the elements that will be copied by
    # this program.
    indices = gl.arange(0, XBLOCK, layout=layout)

    # Obtain the offsets of each element to be copied, which are distriuted
    # among the threads in the program according to the layout.
    offsets = start + indices

    # Adding the offsets to `in_ptr` yields a tensor of pointers to each element
    # the program will load.
    in_ptrs = in_ptr + offsets

    # Because some of the pointers may be out-of-bounds, we will need to mask
    # the global load. For each pointer, mask the load if the offset is greater
    # than the number of elements.
    mask = offsets < xnumel

    value = gl.load(in_ptrs, mask=mask)

    # We need to mask the store as well to prevent writing out-of-bounds.
    out_ptrs = out_ptr + offsets
    gl.store(out_ptrs, value, mask=mask)


def memcpy_1d_impl(input, output, XBLOCK, layout, num_warps):
    xnumel = input.numel()
    grid = (triton.cdiv(xnumel, XBLOCK), )
    # Return the compiled kernel so we can inspect it later.
    compiled_kernel = memcpy_1d_kernel[grid](input, output, xnumel, XBLOCK, layout, num_warps=num_warps)
    return compiled_kernel


# %%
# Let's benchmark the kernel with a variety of layouts. Recall that the best
# `XBLOCK` we obtained in the previous tutorial was 2048, which is what we will
# use to start.
#
# For a 1D tensor, there isn't much choice for the layout. If we use the default
# `num_warps=4`, the only valid layouts are
#
# ```python
# gl.BlockedLayout(
#     size_per_thread=[R],
#     threads_per_warp=[32],
#     warps_per_cta=[4],
#     order=[0],
# ```
#
# Where `R` must be a power of 2.


def get_throughput(input, ms):
    tbytes = (input.numel() * input.element_size() >> 30) / 1024
    return tbytes / (ms * 1e-3)


def bench_memcpy_impl(input, output, impl):
    compiled_kernel = impl(input, output)
    fn = lambda: impl(input, output)
    ms = triton.testing.do_bench(fn)
    return compiled_kernel, get_throughput(input, ms)


def bench_memcpy(impl):
    torch.manual_seed(0)
    xnumel = 2 << 30
    input = torch.randn(xnumel, device="cuda")
    output = torch.empty_like(input)

    return bench_memcpy_impl(input, output, impl)


@pytest.mark.parametrize("XBLOCK", [128, 256])
@pytest.mark.parametrize("xnumel", [200, 1000])
@pytest.mark.parametrize("num_warps", [4])
def test_memcpy_1d(XBLOCK, xnumel, num_warps):
    torch.manual_seed(0)
    input = torch.randn(xnumel, device="cuda")
    output = torch.empty_like(input)
    layout = gl.BlockedLayout([1], [32], [num_warps], [0])
    memcpy_1d_impl(input, output, XBLOCK, layout, num_warps=num_warps)
    torch.testing.assert_close(input, output, atol=0, rtol=0)


# %%
# By choosing `XBLOCK=2048`, the largest value we can pick for `R` without
# incurring redundant values via broadcasting is `R=16`. For smaller values,
# the layout is tiled to fill the tensor.

if __name__ == "__main__" and _enabled("R_vs_throughput"):
    print("R vs. Throughput")
    print("================")
    XBLOCK = 2048
    num_warps = 4
    kernel = partial(memcpy_1d_impl, XBLOCK=XBLOCK, num_warps=num_warps)
    compiled_kernels = []
    for i in range(0, 5):
        R = 2**i
        layout = gl.BlockedLayout([R], [32], [num_warps], [0])
        impl = partial(kernel, layout=layout)
        compiled_kernel, throughput = bench_memcpy(impl)
        compiled_kernels.append((R, compiled_kernel))
        print(f"R={R:<3} {throughput:.3f} TB/s")
    print()

# %%
# Running this on B200, we obtain
#
# ```
# R vs. Throughput
# ================
# R=1   3.287 TB/s
# R=2   3.238 TB/s
# R=4   3.237 TB/s
# R=8   3.251 TB/s
# R=16  3.107 TB/s
# ```
#
# We can see that the our selected layout does affect performance (and that this
# is 10x faster than copying one scalar at a time). Let's dig deeper into why
# the layout affects performance by examining the SASS for the kernels.

if __name__ == "__main__" and _enabled("LDG_STG_instructions"):
    print("LDG/STG instructions")
    print("====================")
    for R, compiled_kernel in compiled_kernels:
        print(f"\nR={R}")
        print("==========")
        sass = compiled_kernel.asm["sass"]
        for line in sass.split("\n"):
            if "LDG.E" in line or "STG.E" in line:
                print(line)
    print()

# %%
# The output shows that the layout affects the vectorization of the loads and
# striding when multiple loads are emitted. We can summarize this in a table:
#
# | R  | width | vec_len | n_loads | stride |
# |----|-------|---------|---------|--------|
# | 1  | 32    | 32      | 1       | 0x00   |
# | 2  | 64    | 64      | 1       | 0x00   |
# | 4  | 128   | 128     | 1       | 0x00   |
# | 8  | 256   | 128     | 2       | 0x10   |
# | 16 | 512   | 128     | 4       | 0x10   |
#
# Modern NVIDIA GPUs have 128-byte cache lines, divided into 32-byte sectors.
# The 32B sectors are the granularity at which global memory is accessed by a
# processor. Thus, the GPU hardware attempts to minimize the number of 32B
# sector accesses by grouping contiguous memory accesses within a warp.
#
# When R=1, each `LDG.E` at the warp level reads exactly 128 contiguous bytes of
# global memory, which fits into a cache line. Note that PyTorch allocates
# tensors aligned to 256 bytes by default.
#
# Increasing `R` to 2 or 4 widens each `LGD.E` instruction but slows down the
# kernel, despite the number of 32B sector reads remaining unchanged. This can
# be due to a variety of obscure hardware factors, but if you look at the
# annotations to the left of the printed instructions you can see one potential
# factor:
#
# ```
# 16:1:2:-:1	@!P0 LDG.E R0, desc[UR4][R8.64];
# --:-:3:-:1	@!P0 LDG.E R15, desc[UR4][R4.64];
# --:-:4:-:1	@!P0 LDG.E R17, desc[UR4][R4.64+0x200];
# ...
# 08:0:-:-:1	@!P0 STG.E desc[UR4][R6.64], R15;
# 16:0:-:-:1	@!P0 STG.E desc[UR4][R6.64+0x200], R17;
# 04:0:-:-:1	@!P0 STG.E desc[UR4][R6.64+0x400], R19;
# ```
#
# These annotations are
#
# ```
# wait_mask : read_barrier : write_barrier : yield : stall
# ```
#
# The load instructions set a `write_barrier` because they are writing to
# registers. Subsequent `STG.E` instructions have a `wait_mask` that block until
# the barrier is cleared. By issuing smaller granularity loads, the store
# instructions can start executing earlier.
#
# The fact that `R=8` is faster than `R=2` and `R=4` despite having strided
# loads is perhaps due two independent transactions, but it's hard to know for
# sure without using a profiler.

if __name__ == "__main__" and _enabled("XBLOCK_R_vs_throughput"):
    print("(XBLOCK, R) vs. Throughput")
    print("==========================")
    num_warps = 4

    print("XBLOCK   ", end=" ")
    for i in range(0, 5):
        print(f"R={2**i:<3}", end=" ")
    print()

    for j in range(10, 15):
        XBLOCK = 2**j
        print(f"{XBLOCK:<8}", end=" ")
        kernel = partial(memcpy_1d_impl, XBLOCK=XBLOCK, num_warps=num_warps)
        for i in range(0, 5):
            R = 2**i
            layout = gl.BlockedLayout([R], [32], [num_warps], [0])
            impl = partial(kernel, layout=layout)
            compiled_kernel, throughput = bench_memcpy(impl)
            print(f"{throughput:.3f}", end=" ")
        print()
    print()

# %%
# If we run this experiment with a variety of `XBLOCK`, we see that `R=8` is
# not always faster than `R=2` and `R=4`.
#
# ```
# (XBLOCK, R) vs. Throughput
# ==========================
# XBLOCK    R=1   R=2   R=4   R=8   R=16
# 1024     3.283 3.274 3.271 3.275 2.613
# 2048     3.286 3.237 3.237 3.252 3.109
# 4096     3.277 3.246 3.227 3.198 3.091
# 8192     3.303 3.266 3.241 3.239 3.088
# 16384    3.261 3.278 3.243 3.255 3.073
# ```
#
# We can also conclude that `R=1` is the best layout for 1D load, and that we
# should be using `XBLOCK=8192` instead.

# %%
# With higher-dimensional tensors, picking the right layout is a lot less
# forgiving, because the tensors can be accessed in non-contiguous ways. Let's
# look at a 2D memcpy kernel to illustrate this.
#
# In order to index into a strided 2D tensor, we need to generate 2D offsets by
# broadcasting two 1D offsets and adding them together. The final layout of the
# 2D offsets will be a `BlockedLayout`, but we need a `SliceLayout` to represent
# the layout of the 1D offsets:
#
# ```python
# gl.SliceLayout(dim=1, parent=layout)
# ```
#
# A slice layout is defined relative to a parent layout. It is the layout that
# results from dropping the `dim` dimension from the parent layout. For example,
# let's use the blocked layout we discussed at the beginning of this tutorial.
#
# ```python
# layout = gl.BlockedLayout(
#     size_per_thread=[2, 4],
#     threads_per_warp=[16, 2],
#     warps_per_cta=[2, 2],
#     order=[1, 0],
# )
# ```
#
# Recall the mapping of tensor elements to registers in a warp is:
#
# ```
# [[ T0:0,  T0:1,  T0:2,  T0:3,  T1:0,  T1:1,  T1:2,  T1:3],
#  [ T0:4,  T0:5,  T0:6,  T0:7,  T1:4,  T1:5,  T1:6,  T1:7],
#  [ T2:0,  T2:1,  T2:2,  T2:3,  T3:0,  T3:1,  T3:2,  T3:3],
#  [ T2:4,  T2:5,  T2:6,  T2:7,  T3:4,  T3:5,  T3:6,  T3:7],
#  ...
#  [T28:0, T28:1, T28:2, T28:3, T29:0, T29:1, T29:2, T29:3],
#  [T28:4, T28:5, T28:6, T28:7, T29:4, T29:5, T29:6, T29:7],
#  [T30:0, T30:1, T30:2, T30:3, T31:0, T31:1, T31:2, T31:3],
#  [T30:4, T30:5, T30:6, T30:7, T31:4, T31:5, T31:6, T31:7]]
# ```
#
# To form the slice layout along `dim=1`, first collapse the mappings in each
# row together:
#
# ```
# [  T0:0| T0:1| T0:2| T0:3| T1:0| T1:1| T1:2| T1:3,
#    T0:4| T0:5| T0:6| T0:7| T1:4| T1:5| T1:6| T1:7,
#    T2:0| T2:1| T2:2| T2:3| T3:0| T3:1| T3:2| T3:3,
#    T2:4| T2:5| T2:6| T2:7| T3:4| T3:5| T3:6| T3:7,
#  ...
#   T28:0|T28:1|T28:2|T28:3|T29:0|T29:1|T29:2|T29:3,
#   T28:4|T28:5|T28:6|T28:7|T29:4|T29:5|T29:6|T29:7,
#   T30:0|T30:1|T30:2|T30:3|T31:0|T31:1|T31:2|T31:3,
#   T30:4|T30:5|T30:6|T30:7|T31:4|T31:5|T31:6|T31:7]
# ```
#
# Then remove redundant register mappings within each thread:
#
# ```
# [  T0:0| T1:0,
#    T0:1| T1:1,
#    T2:0| T3:0,
#    T2:1| T3:1,
#  ...
#   T28:0|T29:0,
#   T28:1|T29:1,
#   T30:0|T31:0,
#   T30:1|T31:1]
# ```
#
# Such a layout would result from a reduction of a 2D tensor along `dim=1`. You
# can see that each element in the reduction result would be broadcasted to two
# threads pairwise.
#
# Likewise, to expand a 1D tensor to 2D, we start with the tensor in slice
# layout and perform the reverse transformation by broadcasting duplicate each
# element of the 1D tensor until it fills the rows to the desired size.
#
# You can see that broadcasting is a zero-cost operation because it does not
# compute any new values nor does it require any data movement: each thread
# already has the value that will be broadcasted, and the values are lazily
# materialized into physical registers by the compiler when needed.


@gluon.jit
def memcpy_2d_kernel(in_ptr, out_ptr,  #
                     xnumel, ynumel, xstride_in, ystride_in, xstride_out, ystride_out,  #
                     layout: gl.constexpr, XBLOCK: gl.constexpr, YBLOCK: gl.constexpr):
    pid_x = gl.program_id(0)
    pid_y = gl.program_id(1)

    start_x = pid_x * XBLOCK
    start_y = pid_y * YBLOCK
    indices_x = start_x + gl.arange(0, XBLOCK, layout=gl.SliceLayout(dim=1, parent=layout))
    indices_y = start_y + gl.arange(0, YBLOCK, layout=gl.SliceLayout(dim=0, parent=layout))

    # Multiply the x and y offsets by the strides of the input tensor in each
    # dimension to translate tensor indices to memory offsets.
    #
    # `indices_x[:, None]` expands the tensor by appending a size 1 dimension
    # to its shape, thus yielding a [XBLOCK, 1] tensor. When combined with a
    # [1, YBLOCK] tensor, each is broadcasted to a [XBLOCK, YBLOCK] tensor.
    in_offsets = xstride_in * indices_x[:, None] + ystride_in * indices_y[None, :]
    out_offsets = xstride_out * indices_x[:, None] + ystride_out * indices_y[None, :]

    # Compute the mask in the same way, by selecting for indices along each
    # dimension that are in bounds and broadcasting them together.
    mask = (indices_x[:, None] < xnumel) & (indices_y[None, :] < ynumel)

    value = gl.load(in_ptr + in_offsets, mask=mask)
    gl.store(out_ptr + out_offsets, value, mask=mask)


def memcpy_2d_impl(input, output, XBLOCK, YBLOCK, layout, num_warps):
    xnumel, ynumel = input.shape
    grid = (triton.cdiv(xnumel, XBLOCK), triton.cdiv(ynumel, YBLOCK))
    # Pass the strides of the input and output tensors into the kernel. The
    # compiler will specialize the kernel if any of the strides are 1, which is
    # common for the inner dimension of tensors.
    compiled_kernel = memcpy_2d_kernel[grid](  #
        input, output, xnumel, ynumel,  #
        *input.stride(), *output.stride(),  #
        layout, XBLOCK, YBLOCK, num_warps=num_warps)
    return compiled_kernel


# %%
# Let's test our kernel, focusing especially on cases where the tensor elements
# are not contiguous in memory.


@pytest.mark.parametrize("XBLOCK, YBLOCK", [(128, 256), (256, 128)])
@pytest.mark.parametrize("xnumel, ynumel", [(100, 2000), (1000, 200)])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("num_warps", [4])
def test_memcpy_2d(XBLOCK, YBLOCK, xnumel, ynumel, transpose, num_warps):
    torch.manual_seed(0)
    input = torch.randn((xnumel, ynumel), device="cuda")
    # Transposing the tensor makes it non-contiguous along the inner dimension.
    input = input.T if transpose else input
    output = torch.empty_like(input)
    layout = gl.BlockedLayout([1, 1], [1, 32], [1, num_warps], [1, 0])
    memcpy_2d_impl(input, output, XBLOCK, YBLOCK, layout, num_warps=num_warps)
    torch.testing.assert_close(input, output, atol=0, rtol=0)


# %%
# Instead of autotuning, we should just pick the layout we know will work based
# based on our findings in 1D. Given that a 2D tensor is just a contiguous
# memory block underneath, we can try to degenerate the 2D memcpy into the same
# kernel as the 1D memcpy.


def bench_memcpy_2d(impl, transposed=False):
    # 8 GB tensor, but spread across 2 dimensions.
    xnumel = 32 * 1024
    ynumel = 64 * 1024
    input = torch.randn((xnumel, ynumel), device="cuda")
    if transposed:
        input = input.T
    output = torch.empty_like(input)
    return bench_memcpy_impl(input, output, impl)


# %%
# Let's choose `XBLOCK=1`, which means each program will process a row vector,
# and we can pick a blocked layout that behaves the same over this row vector
# as the `R=1` does in 1D.

if __name__ == "__main__" and _enabled("memcpy_2d_layout"):
    print("Benchmarking 2D memcpy")
    print("======================")
    XBLOCK = 1
    YBLOCK = 2048
    layout = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    impl = partial(memcpy_2d_impl, XBLOCK=XBLOCK, YBLOCK=YBLOCK, layout=layout, num_warps=4)
    _, throughput = bench_memcpy_2d(impl)
    print(f"Throughput: {throughput:.3f} TB/s")

# %%
# Running this yields 3.130 TB/s, which is 5% slower than the 1D memcpy. There
# may be a variety of reasons, including the more complex 2D indexing
# arithmetic, but let's examine the performance in more detail first.
#
# Our 2D memcpy kernel has another problem: the optimal layout depends on the
# layout of the input tensor. Let's see what happens when the input tensor is
# transposed:

if __name__ == "__main__" and _enabled("memcpy_2d_layout"):
    _, throughput = bench_memcpy_2d(impl, transposed=True)
    print(f"Transposed throughput: {throughput:.3f} TB/s")

# %%
# Performance craters and we get 0.387 TB/s, which is almost as slow as our
# scalar memcpy from the last tutorial. The reason is obvious: the inner
# dimension is no longer contiguous. Simply swapping the block sizes and the
# transposing the layout restores performance:

if __name__ == "__main__" and _enabled("memcpy_2d_layout"):
    layout = gl.BlockedLayout([1, 1], [32, 1], [4, 1], [0, 1])
    impl = partial(memcpy_2d_impl, XBLOCK=2048, YBLOCK=1, layout=layout, num_warps=4)
    _, throughput = bench_memcpy_2d(impl, transposed=True)
    print(f"Fixed throughput: {throughput:.3f} TB/s")
    print()

# %%
# This yields 3.295 TB/s, which is actually slightly faster than the 1D memcpy.
# Each program accesses memory in the same way, which means the only variation
# could be where the programs are scheduled.
#
# While we know each program accesses unique memory above the cache line size,
# we can't completely rule out data locality as the GPU cache structure is very
# complex with many mechanisms to improve performance. For example, the GPU
# contains TLBs (Translation Lookaside Buffers) which cache virtual address
# translations. GPU pages are 64KB by default, but with our chosen block sizes,
# a program copies less than a whole page on its own. This can cause TLB misses
# when programs that access the same page are scheduled in different SMs.
#
# The L2 cache is also divided into a number of partitions (16 on H100 GPUs),
# which are mapped based on physical addresses. Scheduling can affect whether
# kernels that access the same L2 partition are scheduled near each other.
#
# In a subsequent tutorial, we will explore implementing persistent kernels and
# how they can be used to better control scheduling, among other benefits, to
# improve performance.
#
# One can conclude that the 1D memcpy provides more consistent performance than
# the 2D memcpy, but it only works if the input AND output tensors are views
# over a contiguous memory block. The 2D memcpy shines when either input or
# output has a more exotic layout.
#
# Consider a non-contiguous input tensor, which we can construct by taking a
# view of every other row of an 8 GB tensor. We can copy this into a contiguous
# output tensor, which is the same as performing `x.contiguous()` in PyTorch.

if __name__ == "__main__" and _enabled("memcpy_2d_contig"):
    print("Non-contiguous memcpy")
    print("=====================")
    # 8 GB tensor.
    xnumel = 32 * 1024
    ynumel = 64 * 1024
    input = torch.randn((xnumel, ynumel), device="cuda")
    # Take a view over every other row.
    input = input[::2]
    output = torch.empty_like(input)
    assert not input.is_contiguous() and output.is_contiguous()

    # Benchmark 2D memcpy.
    layout = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    impl = partial(memcpy_2d_impl, XBLOCK=1, YBLOCK=2048, layout=layout, num_warps=4)
    _, throughput = bench_memcpy_impl(input, output, impl)
    print(f"2D memcpy: {throughput:.3f} TB/s")

    # Benchmark PyTorch contiguous.
    fn = lambda: input.contiguous()
    ms = triton.testing.do_bench(fn)
    throughput = get_throughput(input, ms)
    print(f"torch.Tensor.contiguous: {throughput:.3f} TB/s")

    # We can eke out even more performance by using the transposed "trick".
    layout = gl.BlockedLayout([1, 1], [32, 1], [4, 1], [0, 1])
    impl = partial(memcpy_2d_impl, XBLOCK=2048, YBLOCK=1, layout=layout, num_warps=4)
    _, throughput = bench_memcpy_impl(input.T, output.T, impl)
    print(f"2D memcpy (transposed): {throughput:.3f} TB/s")

# %%
# The results from the benchmark:
#
# ```
# Non-contiguous memcpy
# =====================
# 2D memcpy: 3.129 TB/s
# torch.Tensor.contiguous: 1.473 TB/s
# 2D memcpy (transposed): 3.199 TB/s
# ```
#
# So our 2D memcpy provides similar performance even when the input tensor has
# an exotic layout. It's already over 2x faster than the PyTorch implementation
# of `Tensor.contiguous`. And we also see that applying the transposition
# "trick" we "learned" through experimentation results in slightly more
# performance.
