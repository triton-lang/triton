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
[[ T0,  T1,  T2, ..., T15],
 [T16, T17, T18, ..., T31]]
```

Note that the size of the warp tile must match the number of threads per warp,
which for NVIDIA hardware is 32. If we substitute each thread with its thread
tile, we obtain the warp tile over the elements of the tensor:

```
[[ T0:0,  T0:1,  T0:2,  T0:3, ..., T15:0, T15:1, T15:2, T15:3],
 [ T0:4,  T0:5,  T0:6,  T0:7, ..., T15:4, T15:5, T15:6, T15:7],
 ...
 [T16:0, T16:1, T16:2, T16:3, ..., T31:0, T31:1, T31:2, T31:3],
 [T16:4, T16:5, T16:6, T16:7, ..., T31:4, T31:5, T31:6, T31:7]]
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
[[ T0:0|T32:0|T64:0|T96:0,  ..., T15:3|T47:3|T79:3|T111:3],
 [ T0:4|T32:4|T64:4|T96:4,  ..., T15:7|T47:7|T79:7|T111:7],
 ...
 [T16:0|T48:0|T80:0|T112:0, ..., T31:3|T63:3|T95:3|T127:3],
 [T16:4|T48:4|T80:4|T112:4, ..., T31:7|T63:7|T95:7|T127:7]]
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

import torch
import triton
from functools import partial
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

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
    memcpy_1d_kernel[grid](input, output, xnumel, XBLOCK, layout, num_warps=num_warps)


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


def bench_memcpy(impl):
    torch.manual_seed(0)
    xnumel = 4 << 30
    input = torch.randn(xnumel, device="cuda")
    output = torch.empty_like(input)

    fn = lambda: impl(input, output)
    ms = triton.testing.do_bench(fn)
    tbytes = xnumel * input.element_size() >> 40
    print("TB/s", round(tbytes / (ms * 1e-3), 2))


if __name__ == "__main__":
    XBLOCK = 2048
    num_warps = 4
    kernel = partial(memcpy_1d_impl, XBLOCK=XBLOCK, num_warps=num_warps)
    for i in range(0, 3):
        R = 2**i
        layout = gl.BlockedLayout([R], [32], [num_warps], [0])
        impl = partial(kernel, layout=layout)
        bench_memcpy(impl)
