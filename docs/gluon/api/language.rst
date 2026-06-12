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

    reduce
    reduce_or
    sum
    max
    min
    xor_sum


Scan Ops
--------

.. autosummary::
    :toctree: generated
    :nosignatures:

    associative_scan
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
