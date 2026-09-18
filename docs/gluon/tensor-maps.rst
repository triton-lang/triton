GPU-published tensor maps
=========================

A producer kernel can choose a tensor on the GPU and publish a TMA descriptor
for subsequent kernels. This is useful when a GPU pointer table selects among
independent allocations, or when preparation chooses between original storage
and a packed, aligned copy. Consumers can reuse the descriptor without repeating
its construction in every program.

The Hopper and Blackwell ``tma`` modules provide two operations:

* ``publish_tensor_descriptor(storage, template, base, shape, strides)`` copies a
  valid tiled descriptor template, replaces its backing pointer, dimensions and
  strides, and publishes it with a GPU-scope tensor-map proxy release.
* ``load_tensor_descriptor(storage, shape, strides, block_shape, dtype, layout)``
  acquires the tensor-map proxy in every consumer CTA and returns a typed
  descriptor usable by the existing TMA load, store, gather and scatter APIs.

The template preserves the tile, element format, swizzle, padding, and other
encoding fields. Create it with the host ``TensorDescriptor`` API, which uses
the CUDA driver to encode a valid representation. Neither operation exposes
raw descriptor fields or requires inline assembly.

Example
-------

The following device helpers select a float16 matrix from a GPU pointer table
and load one tile from the published map::

    import triton
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

    @gluon.jit
    def prepare(maps, template, pointers, rows, cols, row_stride):
        i = gl.program_id(0)
        base = gl.load(pointers + i).to(gl.pointer_type(gl.float16))
        tma.publish_tensor_descriptor(
            maps + 128 * i, template, base,
            [rows, cols], [row_stride, 1])

    @gluon.jit
    def consume(maps, output, rows, cols, row_stride):
        i = gl.program_id(0)
        layout: gl.constexpr = gl.NVMMASharedLayout(128, 16, rank=2)
        desc = tma.load_tensor_descriptor(
            maps + 128 * i, [rows, cols], [row_stride, 1],
            [16, 64], gl.float16, layout)
        tile = gl.allocate_shared_memory(gl.float16, [16, 64], layout)
        bar = mbarrier.allocate_mbarrier()
        mbarrier.init(bar, count=1)
        mbarrier.expect(bar, desc.nbytes_per_cta)
        tma.async_load(desc, [0, 0], bar, tile)
        mbarrier.wait(bar, 0)
        mbarrier.invalidate(bar)
        regs: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
        x = gl.arange(0, 16, gl.SliceLayout(1, regs))
        y = gl.arange(0, 64, gl.SliceLayout(0, regs))
        gl.store(output + i * 1024 + x[:, None] * 64 + y[None, :], tile.load(regs))

The host can launch these kernels as follows::

    import torch

    tensors = [torch.randn((128, 256), device="cuda", dtype=torch.float16)
               for _ in range(20)]
    pointers = torch.tensor([x.data_ptr() for x in tensors],
                            device="cuda", dtype=torch.uint64)
    maps = torch.empty((20, 128), device="cuda", dtype=torch.uint8)
    output = torch.empty((20, 16, 64), device="cuda", dtype=torch.float16)
    layout = gl.NVMMASharedLayout(128, 16, rank=2)
    template = TensorDescriptor.from_tensor(tensors[0], [16, 64], layout)
    prepare[(20,)](maps, template, pointers, 128, 256, 256)
    consume[(20,)](maps, output, 128, 256, 256)

The template's backing tensor supplies a valid initial encoding; the producer
replaces its pointer, shape and strides. Keep ``tensors`` and ``maps`` alive
through consumption. Launch preparation and consumption in the same stream, or
establish an event dependency between their streams.

Ownership and ordering
----------------------

Each map occupies 128 bytes at a 128-byte-aligned address. Exactly one producer
program owns each destination during publication. Within a multi-CTA program,
only CTA zero publishes. Publication does not synchronize different programs.

Keep both descriptor storage and every selected backing allocation alive until
all consumers finish. A raw GPU pointer does not retain the corresponding Python
object. Do not overwrite, free, or repurpose either allocation while consumers
can still access it. Republish only after the previous consumers have finished;
then acquire the new version in each consumer CTA.

The release/acquire pair orders descriptor metadata. It does not replace stream
or event ordering for the producer's payload writes. Conversely, stream ordering
alone does not acquire the tensor-map proxy. ``load_tensor_descriptor`` includes
the proxy acquire and a CTA barrier, including the compiler's existing ptxas
fence workaround. Once acquired, a descriptor can be reused by that CTA without
another acquire while its contents remain unchanged. These APIs use GPU scope
and are intended for producers and consumers on the same device.

Typing and packed formats
-------------------------

The shape and strides supplied to ``load_tensor_descriptor`` are explicit
metadata; the function does not decode them from the opaque map. They must match
the published tensor. Its element type, tile and layout must also match the
encoded descriptor. For a different consumer cluster layout, match the physical
TMA tile per CTA, rather than merely copying the producer's logical block shape.

For ``fp4_padded=True``, use ``uint8``, 128-byte swizzling, and packed-byte shapes,
strides and coordinates. The backing pointer and outer byte strides must be
32-byte aligned. The compiler converts the innermost size to FP4 elements for
``tensormap.replace``. The innermost extent must be a multiple of 64 packed
bytes. Padded FP4 supports global-to-shared TMA loads, not TMA stores.
Other supported element types require 16-byte-aligned
backing pointers and outer byte strides. The last stride is always one.

A ragged descriptor can avoid descriptor mutation when indexing bounded slices
of one backing allocation. It does not directly replace a GPU pointer table
selecting unrelated allocations with the current ``ragged_tma`` API.

Sanitizers
----------

GSan instruments reads and writes of the complete 128-byte descriptor as well
as TMA payload accesses. Its FP4 bounds use packed-byte units. ConSan accounts
for the producer's shared staging storage and the consumer's shared-memory TMA
operations. Sanitizers still require their usual instrumented allocation and
execution setup; an uninstrumented external producer is outside that coverage.

References
----------

* `CUDA: Encoding a Tensor Map on Device <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#encoding-a-tensor-map-on-device>`_
* `CUDA: Usage of a Modified Tensor Map <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#usage-of-a-modified-tensor-map>`_
* `PTX: tensormap.replace <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-tensormap-replace>`_

The CUDA guide explicitly permits zero-initialized descriptor construction on
SM90a. This API uses a valid template so it does not depend on extending that
encoding assumption to other architectures.
