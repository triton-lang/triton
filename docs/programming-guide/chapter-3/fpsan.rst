=========================================
Floating-Point Sanitizer (FpSan)
=========================================

FpSan is a compiler instrumentation mode that rewrites selected floating-point
Triton IR operations into deterministic "payload algebra" over integer
bit-patterns. Its goal is not to approximate IEEE floating-point arithmetic.
Instead, it preserves selected algebraic structure so that kernels that are
symbolically equivalent under the sanitized semantics continue to agree, while
wrong rewrites, wrong operands, wrong dataflow, or missing synchronization tend
to perturb the result.

This makes FpSan primarily a kernel-checking tool. It is especially useful when
you care more about whether a kernel preserves the intended symbolic
computation than about its exact IEEE result on a particular input.

At a high level, FpSan:

- maps floating-point bit-patterns into an integer payload domain
- replaces supported floating-point ops with integer-domain rewrites chosen to
  preserve selected identities exactly
- maps the resulting payload back into a floating-point bit-pattern so the rest
  of the pipeline can continue

------------
Enabling FpSan
------------

Enable FpSan before the compile or run you want to instrument.

From Python:

.. code-block:: python

   import triton

   triton.knobs.compilation.instrumentation_mode = "fpsan"
   # compile and run kernels here
   triton.knobs.compilation.instrumentation_mode = ""

From the shell:

.. code-block:: bash

   TRITON_INSTRUMENTATION_MODE=fpsan python your_script.py

Notes:

- FpSan is a compiler feature, so it does not apply in interpreter mode.
- On AMD, the backend currently enables FpSan only for ``gfx942``, ``gfx950``,
  and ``gfx1250``.

----------------
How to Use It
----------------

The most effective way to use FpSan is to compare two kernels, or two versions
of one kernel, under the same FpSan mode. Typical uses include:

- comparing an optimized kernel against a simple reference kernel
- comparing a fused kernel against an unfused composition
- comparing two schedule variants that should be mathematically equivalent
- checking that accumulator selection, predication, or TMEM pipelines preserve
  the intended payload flow

FpSan results should only be compared against other FpSan results, not
against ordinary floating-point outputs.

------------------
Payload Model
------------------

For each floating-point width ``w``, FpSan defines a bijection between
floating-point bit-patterns and a w-bit integer payload; arithmetic wraps modulo 2^w.

Conceptually:

- ``embed(x)`` maps a float bit-pattern to an integer payload
- ``unembed(u)`` maps an integer payload back to a float bit-pattern
- sanitized float ops are implemented as ``unembed(F(embed(...)))``

The embedding is deliberately chosen so that a few important constants are
stable:

- ``embed(+0.0) = 0``
- ``embed(+1.0) = 1``
- ``embed(-1.0) = all-ones``

Those fixed points are the reason identities such as ``x + 0 = x`` and
``x * 1 = x`` behave naturally under FpSan.

------------------------------
What FpSan Preserves
------------------------------

FpSan preserves exact identities in the payload algebra selected by each
rewrite. The most important ones are:

- ring identities for add, subtract, unary negation, multiply, FMA, and
  dot-like accumulation
- selected exponential identities for ``exp`` and ``exp2`` (see below for details)
- trigonometric identities for ``sin`` and ``cos``
- payload equality through casts, loads, stores, and copies
- deterministic op-distinguishing tags for unary functions that do not yet have
  a richer algebraic model

This is what makes FpSan valuable for kernel checks: if two kernels should be
the same symbolic computation under the preserved properties, they should produce
the same payloads. This holds assuming a (generally believed) conjecture in
transcendental number theory, Schanuel's conjecture. One of the authors of FpSan
has a [blog post](https://cp4space.hatsya.com/2026/05/03/schanuels-conjecture-and-the-semantics-of-fpsan/)
explaining the theory behind FpSan from a mathematical perspective.

----------------------------------
What FpSan Does Not Preserve
----------------------------------

FpSan is not an IEEE simulator.

In particular, do not rely on it for:

- real floating-point ordering, rounding, NaN propagation, infinities,
  subnormals, or exceptions
- real transcendental semantics for ``log``, ``sqrt``, ``erf``, ``floor``,
  ``ceil``, ``rsqrt``, and similar tagged unary ops
- expected floating-point bit patterns (i.e. for kernels that bitcast
  between floats and integers)

When a property matters for your check, the right question is:
"is this property preserved by the payload rewrite for this specific op
family?"

----------------------
Common Arithmetic Ops
----------------------

Add, Sub, Neg, Mul
==================

Supported operations:

- ``x + y``
- ``x - y``
- ``-x``
- ``x * y``

Rewrite:

- add, subtract, negate, or multiply the embedded payloads, then unembed the
  result

Exact preserved properties:

- ``x + 0 = x``
- ``x - 0 = x``
- ``x - x = 0``
- ``x + (-x) = 0``
- ``-(-x) = x``
- ``x * 1 = x``
- associativity and commutativity of add and mul
- distributivity of mul over add

Important caveat:

- This is ring arithmetic modulo ``2^w``, not IEEE arithmetic.

Min and Max
===========

Supported operations:

- ``tl.minimum(x, y)``
- ``tl.maximum(x, y)``
- ``min(x, y)`` and ``max(x, y)`` in Triton code

Rewrite:

- signed integer ``min`` or ``max`` on payloads

Exact preserved properties:

- idempotence: ``min(x, x) = x`` and ``max(x, x) = x``
- commutativity
- associativity

Important caveats:

- The order is the signed integer order of payloads, not IEEE float order.
- NaN handling, and the exact signed-zero contract, are not modeled.

Division
========

Supported operation:

- ``x / y``

Rewrite:

- ``x / y`` becomes ``embed(x) * inv(embed(y))``, then ``unembed``

Here ``inv`` is:

- the true modular inverse for odd payloads
- a parity-preserving involution for even payloads

Exact preserved properties:

- ``x / 1 = x``
- ``1 / (1 / x) = x``
- for odd payloads, the usual modular inverse laws hold

Important caveats:

- ``x / x = 1`` is not guaranteed for all payloads.
- Division by zero does not produce IEEE infinities or traps.
- This rewrite is chosen for algebraic checking, not numeric realism.

Remainder
=========

Supported operation:

- ``x % y``

Rewrite:

- signed integer remainder on payloads after forcing the denominator odd with
  ``den | 1``

Exact preserved properties:

- same inputs produce the same sanitized remainder payload

Important caveats:

- Real floating-point remainder semantics are not modeled.
- Zero denominators are intentionally mapped to a safe odd payload instead of
  trapping.

FMA
===

Supported operation:

- ``tl.fma(a, b, c)``

Rewrite:

- ``a * b + c`` in payload arithmetic

Exact preserved properties:

- exact agreement with the sanitized expansion ``mul`` followed by ``add``
- ``fma(a, b, c) = a*b + c`` in the payload ring

Important caveat:

- There is no special fused-rounding behavior.

----------------
Unary Math Ops
----------------

``exp2``
========

Supported operation:

- ``tl.exp2(x)``

Rewrite:

- modular exponentiation by a fixed odd generator in payload space

Exact preserved properties:

- ``exp2(x + y) = exp2(x) * exp2(y)``
- ``exp2(0) = 1``
- ``exp2(-x) = 1.0 / exp2(x)``

``exp``
=======

Supported operation:

- ``tl.exp(x)``

Rewrite:

- ``exp(x)`` is implemented as ``exp2(x * rcp_log2)`` in payload space

Exact preserved properties:

- ``exp`` uses the same payload-space construction as ``exp2`` after scaling
  the input by a fixed internal payload constant

``sin`` and ``cos``
===================

Supported operations:

- ``tl.sin(x)``
- ``tl.cos(x)``

Rewrite:

- a deterministic payload-space rewrite chosen to preserve the identities below

Exact preserved properties:

- ``sin(x + y) = sin(x) * cos(y) + cos(x) * sin(y)``
- ``sin(x - y) = sin(x) * cos(y) - cos(x) * sin(y)``
- ``cos(x + y) = cos(x) * cos(y) - sin(x) * sin(y)``
- ``cos(x - y) = cos(x) * cos(y) + sin(x) * sin(y)``
- ``cos(x)^2 + sin(x)^2 = 1``

Important caveat:

- These are not IEEE trig values; they are payload functions chosen to preserve
  the angle identities above.

Tagged Unary Ops
================

Supported operations:

- ``tl.log(x)``
- ``tl.log2(x)``
- ``tl.sqrt(x)``
- ``tl.rsqrt(x)``
- ``tl.erf(x)``
- ``tl.floor(x)``
- ``tl.ceil(x)``
- precise square root variants

Rewrite:

- an invertible payload tag: multiply by an odd constant, xor with an
  op-specific hash, then multiply again

Exact preserved properties:

- payload equality is preserved for the same op: if ``x == y`` in payload
  space, then ``op(x) == op(y)``
- different supported unary ops get different tags

Important caveats:

- These rewrites intentionally do not preserve real mathematical identities
  such as ``sqrt(x)^2 = x`` or ``log(x*y) = log(x) + log(y)``.

--------------------------
Casts and Format Conversions
--------------------------

Float-to-Float Conversions
============================

Supported operations:

- converting a tensor between floating-point types with ``x.to(dtype)``
- implicit float widening and narrowing conversions

Rewrite:

- signed integer extension or truncation in payload space, followed by
  ``unembed``

Exact preserved properties:

- ``0``, ``+1``, and ``-1`` remain stable across the conversion
- sign-extension behavior in the payload domain
- truncation drops high payload bits
- an upcast followed by a downcast is the identity

Important caveat:

- This preserves payload structure, not IEEE conversion semantics.
- Conversions between fp types of the same width do not model any loss of
  precision or range, so for example under fpsan
  ``fn(a.to(tl.float16)).to(tl.bfloat16) == fn(a)`` (for any bfloat16 ``a``).

Packed fp4 conversion
=====================

Rewrite:

- unpack low and high nibbles from the source byte tensor
- reshape and reorder them
- interpret each unpacked nibble directly as a payload in the destination float
  width

Exact preserved properties:

- deterministic unpacking of packed fp4 storage
- exact preservation of the unpacked nibble payloads

Important caveat:

- This is not real fp4 numeric decoding.
- The same raw-payload interpretation is reused by scaled-dot paths for
  ``e2m1``.

----------------------------
Pure Extern Elementwise Ops
----------------------------

Supported operation:

- ``tl.extern_elementwise`` when all of the following hold:

  - the op is ``pure``
  - the result type is float-like
  - there is at least one operand
  - every operand is numeric

Rewrite:

- rotate each operand payload by its argument index
- sum the rotated payloads
- xor the result with a stable hash of the symbol name
- unembed

Exact preserved properties:

- deterministic dependence on all operands and on operand order
- deterministic distinction between different external symbols
- mixed float and integer operands are supported; float operands are embedded,
  integer operands are used directly after signed casting to the result width

Important caveat:

- This is a structural tag, not a numeric model of the external function.

---------------------------
Gluon MMA and Tensor Memory
---------------------------

Supported Gluon operations include:

- ``mma_v2``
- ``warpgroup_mma`` and ``warpgroup_mma_wait``
- ``tcgen05_mma`` and ``tcgen05_mma_scaled``
- ``tcgen05_copy`` and ``tcgen05_commit``
- ``allocate_tensor_memory``
- tensor-memory descriptor methods such as ``load``, ``load_min``,
  ``load_max``, ``store``, ``slice``, ``index``, and ``_reinterpret``
- AMD ``mfma``, ``mfma_scaled``, ``wmma``, ``wmma_scaled``, and
  ``scaled_upcast``

Rewrite:

- perform multiply-add accumulation in payload space
- preserve payload bits across tensor-memory loads, stores, copies, and views
- keep accumulator-selection and predication behavior structurally visible

Exact preserved properties:

- exact matrix-multiply algebra over the payload ring
- exact agreement with sanitized scalar multiply-add expansion
- accumulation with the provided accumulator is preserved as payload addition
- tensor-memory operations preserve payload flow across the pipeline

Important caveats:

- Scaled MMA preserves the sanitizer's payload treatment of low-precision
  operands and scales, not exact hardware-format numeric decoding.
- Tensor-memory operations preserve payload dataflow; they do not make FpSan a
  substitute for race or synchronization checking.
- Currently fpsan is supported on all NVIDIA hardware, as well as AMD ``gfx942``, ``gfx950``,
  and ``gfx1250``.

------------------------------
Practical Guidance for Checks
------------------------------

FpSan is a good fit when you want to check:

- that two kernels implement the same preserved algebra
- that a fused kernel keeps the intended dataflow
- that predication or accumulator-selection logic is wired correctly
- that a tensor-memory or warp-specialized pipeline preserves payload flow

FpSan is a poor fit when you want to check:

- IEEE edge cases
- real transcendental accuracy
- NaN or infinity behavior
- hardware-format decode semantics for low-precision formats

In short, rely on FpSan for structure-preserving kernel validation, and rely on
ordinary numerical tests for IEEE behavior.
