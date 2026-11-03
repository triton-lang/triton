NVIDIA Blackwell
================

TCGen05 and tensor-memory operations require an SM10x or SM11x target. They are
rejected on unsupported targets before instrumentation and lowering.

.. currentmodule:: triton.experimental.gluon.language.nvidia.blackwell

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/gluon-module.rst

    async_copy
    clc
    mbarrier
    tma


.. autosummary::
    :toctree: generated
    :nosignatures:

    add2
    allocate_tensor_memory
    fence_async_shared
    fma2
    max2
    min2
    mma_v2
    mul2
    sub2
    tensor_memory_descriptor
    tensor_memory_descriptor_type
    TensorMemoryLayout
    TensorMemoryScalesLayout
