triton.language
===============

.. currentmodule:: triton.language


Programming Model
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    tensor
    tensor_descriptor
    program_id
    num_programs


Creation Ops
------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    arange
    cat
    full
    zeros
    zeros_like
    cast


Shape Manipulation Ops
----------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    broadcast
    broadcast_to
    expand_dims
    interleave
    join
    permute
    ravel
    reshape
    split
    trans
    view


Linear Algebra Ops
------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    dot
    dot_scaled


Memory/Pointer Ops
----------

.. autosummary::
    :toctree: generated
    :nosignatures:

    load
    store
    make_tensor_descriptor
    load_tensor_descriptor
    store_tensor_descriptor
    make_block_ptr
    advance


Indexing Ops
------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    flip
    where
    swizzle2d


Math Ops
--------

.. autosummary::
    :toctree: generated
    :nosignatures:

    abs
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
    log
    log2
    maximum
    minimum
    rsqrt
    sigmoid
    sin
    softmax
    sqrt
    sqrt_rn
    umulhi


Reduction Ops
-------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    argmax
    argmin
    max
    min
    reduce
    sum
    xor_sum

Scan/Sort Ops
-------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    associative_scan
    cumprod
    cumsum
    histogram
    sort
    topk
    gather

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

Random Number Generation
------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    randint4x
    randint
    rand
    randn


Iterators
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    range
    static_range


Inline Assembly
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    inline_asm_elementwise


Compiler Hint Ops
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    assume
    debug_barrier
    max_constancy
    max_contiguous
    multiple_of


Debug Ops
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    static_print
    static_assert
    device_print
    device_assert


Valid tl.dtype Data Types
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    is_fp8e4nv
    is_fp8e4b8
    is_fp8e4b15
    is_fp8e5
    is_fp8e5b16
    is_fp16
    is_bf16
    is_fp32
    is_fp64
    is_int1
    is_int8
    is_int16
    is_int32
    is_int64
    is_uint8
    is_uint16
    is_uint32
    is_uint64
    is_void
