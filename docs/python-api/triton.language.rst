triton.language
===============

.. currentmodule:: triton.language


Programming Model
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

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


Shape Manipulation Ops
----------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    broadcast
    broadcast_to
    expand_dims
    ravel
    reshape
    trans
    view


Linear Algebra Ops
------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    dot


Memory Ops
----------

.. autosummary::
    :toctree: generated
    :nosignatures:

    load
    store


Indexing Ops
------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    where


Math Ops
--------

.. autosummary::
    :toctree: generated
    :nosignatures:

    abs
    exp
    log
    fdiv
    cos
    sin
    sqrt
    sigmoid
    softmax
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

Scan Ops
-------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    associative_scan
    cumsum
    cumprod

Atomic Ops
----------

.. autosummary::
    :toctree: generated
    :nosignatures:

    atomic_add
    atomic_cas
    atomic_max
    atomic_min
    atomic_xchg


Comparison ops
--------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    minimum
    maximum

.. _Random Number Generation:

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
