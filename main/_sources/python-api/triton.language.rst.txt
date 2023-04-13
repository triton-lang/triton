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
    zeros


Shape Manipulation Ops
----------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    broadcast_to
    reshape
    ravel



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
    atomic_cas
    atomic_xchg


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
    cos
    sin
    sqrt
    sigmoid
    softmax


Reduction Ops
-------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    max
    min
    sum


Atomic Ops
----------

.. autosummary::
    :toctree: generated
    :nosignatures:

    atomic_cas
    atomic_add
    atomic_max
    atomic_min


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


Compiler Hint Ops
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    multiple_of
