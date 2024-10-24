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
