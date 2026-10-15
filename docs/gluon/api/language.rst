Language API
============

.. currentmodule:: triton.experimental.gluon.language

Types
-----

.. autosummary::
    :toctree: generated
    :nosignatures:

    tensor
    shared_memory_descriptor
    distributed_type


Programming Model
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    program_id
    num_programs
    num_warps
    num_ctas
    warp_specialize
    barrier


Creation Ops
------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    allocate_shared_memory
    arange
    cast
    full
    full_like
    zeros
    zeros_like
    to_tensor

Layout Ops
----------

.. autosummary::
    :toctree: generated
    :nosignatures:

    bank_conflicts
    convert_layout
    set_auto_layout
    to_linear_layout

Shape Manipulation Ops
----------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    broadcast
    expand_dims
    join
    map_elementwise
    permute
    ravel
    reshape
    split


Memory Ops
----------

.. autosummary::
    :toctree: generated
    :nosignatures:

    load
    store


Atomic Ops
----------

.. autosummary::
    :toctree: generated
    :nosignatures:

    atomic_add
    atomic_and
    atomic_cas
    atomic_max
    atomic_min
    atomic_or
    atomic_poll
    atomic_xchg
    atomic_xor


Linear Algebra Ops
------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    dot_fma


Indexing Ops
------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    gather
    where


Math Ops
--------

.. autosummary::
    :toctree: generated
    :nosignatures:

    abs
    add
    cdiv
    ceil
    clamp
    cos
    div_rn
    erf
    exp
    exp2
    fdiv
    floor
    fma
    fp4_to_fp
    log
    log2
    maximum
    minimum
    mul
    rsqrt
    sin
    sqrt
    sqrt_rn
    sub
    umulhi


Reduction Ops
-------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    all
    any
    reduce
    reduce_or
    sum
    max
    min
    xor_sum


Boolean warp reductions
~~~~~~~~~~~~~~~~~~~~~~~

The ``all`` and ``any`` operations reduce logical predicates.
The layout determines which physical lanes contribute to each result.

For example, a ``[4, W]`` tensor can assign one row to each of four warps.
Here, ``W`` is the physical warp size: 32 or 64.
Reducing axis 1 produces a ``[4]`` tensor with one Boolean result per row.
The four results can differ.
These tensor results can serve as masks for operations on different rows.

.. code-block:: python

    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl

    @gluon.jit
    def row_votes(Predicates, All, Any, W: gl.constexpr):
        layout: gl.constexpr = gl.BlockedLayout(
            [1, 1], [1, W], [4, 1], [1, 0])
        rows = gl.arange(0, 4, layout=gl.SliceLayout(1, layout))
        columns = gl.arange(0, W, layout=gl.SliceLayout(0, layout))
        predicates = gl.load(Predicates + rows[:, None] * W + columns[None, :])
        gl.store(All + rows, gl.all(predicates, axis=1))
        gl.store(Any + rows, gl.any(predicates, axis=1))

This example requires ``num_warps=4`` and the device's physical warp size.
A row of false predicates produces ``all=False`` and ``any=False``.
A row of true predicates produces ``all=True`` and ``any=True``.
A mixed row produces ``all=False`` and ``any=True``.

The compiler uses a ballot for Boolean AND and OR reductions that span a physical warp.
It also accepts duplicate lane values when their layout represents the same logical predicate.
Other layouts retain the general reduction lowering.
Multiple predicates per lane reduce within each thread before the warp reduction.

Participation and control flow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These reductions use Triton's logical block programming model.
All lanes in each participating warp execute the reduction together.
Tensor masks select logical values; they do not exclude physical lanes from the reduction.
For masked inputs, ``all`` uses ``other=True`` as its reduction identity.
The reduction identity for ``any`` is ``other=False``.

Different warp results must remain tensor elements with valid layouts.
A scalar result represents a reduction of the complete logical input.
The compiler combines warp results when the logical reduction spans multiple warps.
A scalar result can control a branch because it is uniform within the block or warp-specialized region.
Lane-divergent scalar control flow is outside this contract.
The operation provides no memory barrier.

On gfx950, Boolean warp reductions run in ordinary logical block programs.
Upstream does not support ``warp_specialize`` on this target.
Supported ``amd.warp_pipeline_stage`` regions can contain these reductions.
Each participating wave still executes the reduction with all its lanes.

The compiler uses logical ``tt.reduce`` semantics before LLVM lowering.
This operation describes tensor values rather than the hardware's current active-lane mask.
Normal reduction transformations therefore preserve its logical result.
LLVM ballot intrinsics retain their convergence requirements after lowering.


Scan Ops
--------

.. autosummary::
    :toctree: generated
    :nosignatures:

    associative_scan
    cumprod
    cumsum
    histogram


Layout Classes
--------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    AutoLayout
    BlockedLayout
    CoalescedLayout
    DotOperandLayout
    DistributedLinearLayout
    NVMMADistributedLayout
    NVMMASharedLayout
    PaddedSharedLayout
    SharedLinearLayout
    SliceLayout
    SwizzledSharedLayout


Iterators
---------

.. autosummary::
    :toctree: generated
    :nosignatures:

    static_range


Inline Assembly
---------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    inline_asm
    inline_asm_elementwise


Compiler Hints and Debugging
----------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    assume
    max_constancy
    max_contiguous
    multiple_of
    static_assert
    static_print
    device_assert
    device_print
