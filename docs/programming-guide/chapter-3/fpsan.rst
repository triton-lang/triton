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

Enable FpSan before the kernel is first compiled.

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

- Triton kernels are compiled lazily, so the instrumentation mode must be set
  before the first compilation of the kernel variant you care about.
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

FpSan results should usually be compared against other FpSan results, not
against ordinary floating-point outputs.

------------------
Payload Model
------------------

For each floating-point width ``w``, FpSan defines a bijection between
floating-point bit-patterns and the ring ``Z / 2^w Z``.

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

- ring identities for add, subtract, multiply, FMA, and dot-like accumulation
- selected exponential identities for ``exp`` and ``exp2``
- selected trigonometric addition identities for ``sin`` and ``cos``
- payload equality through casts, loads, stores, and copies
- deterministic op-distinguishing tags for unary functions that do not yet have
  a richer algebraic model

This is what makes FpSan valuable for kernel checks: if two kernels should be
the same symbolic computation under the preserved laws, they should produce the
same payloads.

----------------------------------
What FpSan Does Not Preserve
----------------------------------

FpSan is not an IEEE simulator.

In particular, do not rely on it for:

- real floating-point ordering, rounding, NaN propagation, infinities,
  subnormals, or exceptions
- the distinction between ``minimumf`` and ``minnumf``, or between
  ``maximumf`` and ``maxnumf``
- real transcendental semantics for ``log``, ``sqrt``, ``erf``, ``floor``,
  ``ceil``, ``rsqrt``, and similar tagged unary ops
- real fp4, fp6, or fp8 numeric decoding in the places where the
  implementation intentionally uses raw payloads instead

When a property matters for your check, the right question is:
"is this property preserved by the payload rewrite for this specific op
family?"

----------------------
Common Arithmetic Ops
----------------------

Add, Sub, Mul
=============

Supported ops:

- ``arith.addf``
- ``arith.subf``
- ``arith.mulf``

Rewrite:

- ``addf -> addi`` on payloads
- ``subf -> subi`` on payloads
- ``mulf -> muli`` on payloads

Exact preserved properties:

- ``x + 0 = x``
- ``x - 0 = x``
- ``x - x = 0``
- ``x * 1 = x``
- associativity and commutativity of add and mul
- distributivity of mul over add

Important caveat:

- This is ring arithmetic modulo ``2^w``, not IEEE arithmetic.

Min and Max
===========

Supported ops:

- ``arith.minimumf``
- ``arith.maximumf``
- ``arith.minnumf``
- ``arith.maxnumf``

Rewrite:

- signed integer ``min`` or ``max`` on payloads

Exact preserved properties:

- idempotence: ``min(x, x) = x`` and ``max(x, x) = x``
- commutativity
- associativity
- absorption with respect to the payload order

Important caveats:

- The order is the signed integer order of payloads, not IEEE float order.
- ``minnum`` and ``minimum``, and ``maxnum`` and ``maximum``, are treated the
  same.
- NaN-specific behavior is not modeled.

Division
========

Supported ops:

- ``arith.divf``
- ``tt.precise_div``

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

Supported ops:

- ``arith.remf``

Rewrite:

- signed integer remainder on payloads after forcing the denominator odd with
  ``den | 1``

Exact preserved properties:

- deterministic remainder semantics in payload space

Important caveats:

- Real floating-point remainder semantics are not modeled.
- Zero denominators are intentionally mapped to a safe odd payload instead of
  trapping.

FMA
===

Supported ops:

- ``math.fma``

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

Supported op:

- ``math.exp2``

Rewrite:

- modular exponentiation by a fixed odd generator in payload space

Exact preserved properties:

- ``exp2(x + y) = exp2(x) * exp2(y)``
- ``exp2(0) = 1``

``exp``
=======

Supported op:

- ``math.exp``

Rewrite:

- ``exp(x)`` is implemented as ``exp2(x * rcp_log2)`` in payload space

Exact preserved properties:

- ``exp(x + y) = exp(x) * exp(y)``
- ``exp(x) = exp2(x * 1/log(2))`` in the sanitized algebra
- ``exp(-x) = 1 / exp(x)``

``sin`` and ``cos``
===================

Supported ops:

- ``math.sin``
- ``math.cos``

Rewrite:

- a payload-space angle-doubling and angle-addition construction based on a
  fixed ``(cos, sin)`` increment

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

Supported ops:

- ``math.log``
- ``math.log2``
- ``math.sqrt``
- ``math.rsqrt``
- ``math.erf``
- ``math.floor``
- ``math.ceil``
- ``tt.precise_sqrt``

Rewrite:

- an invertible payload tag: multiply by an odd constant, xor with an
  op-specific hash, then multiply again

Exact preserved properties:

- deterministic per-op, per-input behavior
- payload equality is preserved for the same op: if ``x == y`` in payload
  space, then ``op(x) == op(y)``
- different supported unary ops get different tags

Important caveats:

- These rewrites intentionally do not preserve real mathematical identities
  such as ``sqrt(x)^2 = x`` or ``log(x*y) = log(x) + log(y)``.
- Think of them as precise operation fingerprints, not numeric models.

--------------------------
Casts and Format Conversions
--------------------------

Float-to-Float Width Changes
============================

Supported ops:

- ``arith.extf``
- ``arith.truncf``
- ``tt.fp_to_fp``

Rewrite:

- signed integer extension or truncation in payload space, followed by
  ``unembed``

Exact preserved properties:

- ``0``, ``+1``, and ``-1`` remain stable across the conversion
- sign-extension behavior in the payload domain
- truncation drops high payload bits exactly

Important caveat:

- This preserves payload structure, not IEEE conversion semantics.

``ttg.fp4_to_fp``
=================

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

Supported op:

- ``tt.extern_elementwise`` when all of the following hold:

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

----------------------
Dot-Like and MMA Ops
----------------------

``tt.dot`` and ``tt.dot_scaled``
================================

Rewrite:

- lower the operation to scratch-memory emulation loops
- load tiles and slices
- perform sanitized multiply-add accumulation in payload space
- store the result tile back through scratch

Exact preserved properties:

- exact matrix-multiply algebra over the payload ring
- exact agreement with sanitized scalar expansion
- accumulation with the provided ``C`` input is preserved as payload addition

Important caveats:

- Current ``tt.dot`` support is limited to rank-2 and rank-3 tensors with
  compatible encodings.
- Current ``tt.dot_scaled`` support is limited to rank-2 tensors with
  encodings and K-packing present; M and N packing are not supported.
- The emulation loop requires the M and N dimensions to tile cleanly with the
  chosen 8x8 emulation tiles.

Scaled Dot Semantics
====================

For ``tt.dot_scaled`` and ``ttng.tc_gen5_mma_scaled``, the implementation is
more specific than plain matmul:

- ``e4m3``, ``e5m2``, ``bf16``, and ``fp16`` operands are mixed according to
  their storage float format and then cast into the compute payload width
- ``e2m1`` currently uses unpacked raw payload bits rather than a real fp4
  numeric decode
- integer scale tensors are interpreted as exponent-only scale bit-patterns and
  then converted into compute payloads
- scales are multiplied into operand payloads before widening to the
  accumulator width and accumulating

This means FpSan preserves the algebra of the implemented scaled-dot lowering,
not the exact real-number semantics of the hardware format.

NVIDIA ``ttng.warp_group_dot``
==============================

Rewrite:

- scratch-based sanitized MMA emulation

Exact preserved properties:

- the same payload matmul laws as ``tt.dot``
- ``useC = false`` ignores the incoming accumulator
- ``useC = true`` adds the incoming accumulator payloads

NVIDIA ``ttng.tc_gen5_mma``
===========================

Rewrite:

- scratch-based sanitized MMA into TMEM-backed scratch state
- optional barrier arrival after the emulated writeback

Exact preserved properties:

- the same payload matmul laws as ``tt.dot``
- ``useD = false`` ignores the incoming accumulator tile
- ``useD = true`` adds the incoming accumulator tile
- ``pred = false`` leaves the accumulator tile unchanged

NVIDIA ``ttng.tc_gen5_mma_scaled``
==================================

Rewrite:

- the same TMEM-backed MMA emulation, but with the scaled-dot operand handling
  described above

Exact preserved properties:

- the scaled payload-matmul algebra implemented by the sanitizer
- the same ``useD`` and ``pred`` behavior as ``ttng.tc_gen5_mma``

--------------------------------
Tensor Memory and Structural Ops
--------------------------------

Supported ops:

- ``ttng.tmem_load``
- ``ttng.tmem_store``
- ``ttng.tmem_copy``
- ``ttng.tc_gen5_commit``

Semantics:

- TMEM allocations are shadowed with global scratch storage, optionally seeded
  from the allocation source value
- TMEM views such as subslices, indexed views, and reinterprets are resolved as
  pointer arithmetic into that shadow scratch buffer
- ``tmem_load``, ``tmem_store``, and ``tmem_copy`` preserve payload bits
  exactly and use scratch memory to model TMEM-backed values across the
  rewritten program
- async dependencies are preserved structurally by returning commit-group
  tokens when the original op carried a dependency result
- ``tc_gen5_commit`` is lowered to a barrier arrive; it does not introduce new
  arithmetic semantics
- scratch pointers are remapped into warp-specialize partition regions, so TMEM
  payload flow is preserved across those region boundaries as well

These ops are important because they let the arithmetic rewrites above keep
working across TMEM-backed pipelines.

-------------------
AMD-Specific Coverage
-------------------

The AMD pass currently adds two decompositions before the common FpSan pass:

- ``amdg.scaled_upcast_fp8``
- ``amdg.scaled_upcast_fp4``

Both are rewritten into already-supported primitives:

- upcast the low-precision operand
- convert the scale into a float value
- multiply

As a result, their effective FpSan semantics are inherited from:

- ``tt.fp_to_fp`` or ``ttg.fp4_to_fp``
- ``arith.mulf``

For integer scale tensors, the AMD helper constructs exponent-like float
bit-patterns before the multiply, matching the intent of the hardware scale
encoding rather than real integer-to-float conversion.

Current backend support is explicitly enabled for AMD ``gfx942``, ``gfx950``,
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
- exact hardware-format decode semantics for low-precision formats that are
  intentionally modeled as raw payloads

In short, rely on FpSan for structure-preserving kernel validation, and rely on
ordinary numerical tests for IEEE behavior.
