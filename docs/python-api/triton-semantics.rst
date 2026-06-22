Triton Semantics
================

Triton mostly follows the semantics of NumPy with minor exceptions. In this document, we go over some of the array computing features supported in Triton, and we cover the exceptions where Triton's semantics deviate from that NumPy.

Type Promotion
--------------

**Type Promotion** occurs when tensors of different data types are used in an operation. For binary operations associated to `dunder methods <https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types>`_ and the ternary function ``tl.where`` on its last two arguments, Triton automatically converts the input tensors to a common data type following a hierarchy of kinds (sets of dtypes): ``{bool} < {integral dypes} < {floating point dtypes}``.

The algorithm is as follows:

1. **Kind** If one tensor is of a dtype of a higher kind, the other tensor is promoted to this dtype: ``(int32, bfloat16) -> bfloat16``

2. **Width** If both tensors are of dtypes of the same kind, and one of them is of a higher width, the other one is promoted to this dtype: ``(float32, float16) -> float32``

3. **Prefer float16** If both tensors are of the same width and signedness but different dtypes (``float16`` and ``bfloat16`` or different ``fp8`` types), they are both promoted to ``float16``. ``(float16, bfloat16) -> float16``

4. **Prefer unsigned** Otherwise (same width, different signedness), they are promoted to the unsigned dtype: ``(int32, uint32) -> uint32``

The rules are a bit different when they involve a scalar. By scalar here we mean a numeric literal, a variable marked with `tl.constexpr` or a combination of these. These are represented by NumPy scalars and have types ``bool``, ``int`` and ``float``.

When an operation involves a tensor and a scalar:

1. If the scalar is of a kind lower or equal to the tensor, it will not participate in the promotion: ``(uint8, int) -> uint8``

2. If the scalar is of a higher kind, we choose the lowest dtype in which it fits among ``int32`` < ``uint32`` < ``int64`` < ``uint64`` for ints and ``float32`` < ``float64`` for floats. Then, both the tensor and the scalar are promoted to this dtype: ``(int16, 4.0) -> float32``


Broadcasting
------------

**Broadcasting** allows operations on tensors of different shapes by automatically expanding their shapes to a compatible size without copying the data. This follows the following rules:

1. If one of the tensor shapes is shorter, pad it on the left with ones until both tensors have the same number of dimensions: ``((3, 4), (5, 3, 4)) -> ((1, 3, 4), (5, 3, 4))``

2. Two dimensions are compatible if they are equal, or if one of them is 1. A dimension of 1 will be expanded to match the dimension of the other tensor. ``((1, 3, 4), (5, 3, 4)) -> ((5, 3, 4), (5, 3, 4))``


Differences with NumPy
----------------------

**C rounding in integer division** Operators in Triton follow C semantics rather than Python semantics for efficiency. As such, ``int // int`` implements `rounding towards zero as in C <https://en.wikipedia.org/wiki/Modulo#In_programming_languages>`_ for integers of mixed signs, rather than rounding towards minus infinity as in Python. For the same reason, the modulus operator ``int % int`` (which is defined as ``a % b = a - b * (a // b)``) also follows C semantics rather than Python semantics.

Perhaps confusingly, integer division and modulus follow Python semantics for computations where all the inputs are scalars.

**Out-of-range float-to-integer casts** Casting a floating-point value to an integer type is only defined when the value, rounded towards zero, fits in the target type. If the value is out of range, or is NaN, the result is undefined: it may differ between the compiler and the interpreter (``TRITON_INTERPRET=1``), and across hardware backends and toolkit versions. For example, casting ``inf``, a large value such as ``510.0`` to ``int8``, or ``nan`` to an integer type does not produce a portable result. If you need a defined result, clamp the value into range (for example with ``tl.clamp``) and handle ``NaN`` explicitly before the cast.

**Variable scoping** A variable used after a ``for`` loop or an ``if`` statement must be assigned on *every* path through that block, unlike Python where a variable assigned inside a block remains visible afterwards. Triton does not model variables as being dynamically defined or undefined depending on control flow, so a variable that is only assigned inside the block is not considered defined once the block exits. This holds even when the block is guaranteed to execute, such as a ``range(0, 1)`` loop.

For example, the following raises ``NameError: 'value' is not defined`` because ``value`` is only bound inside the loop body:

.. code-block:: python

    @triton.jit
    def kernel(out_ptr):
        for _ in range(0, 1):
            value = 1.0
        tl.store(out_ptr, value)  # `value` is not defined here

Assign the variable before the block so that it is defined on all paths:

.. code-block:: python

    @triton.jit
    def kernel(out_ptr):
        value = 0.0
        for _ in range(0, 1):
            value = 1.0
        tl.store(out_ptr, value)
