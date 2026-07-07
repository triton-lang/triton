"""
Composable Kernels: Host-Built Aggregates as Epilogues
======================================================

In this tutorial you will write a *single* ``matmul_kernel`` and customize its
**epilogue** with a host-constructed ``@triton.aggregate`` passed as one
``tl.constexpr`` argument. Two different launch functions build two different
epilogue objects -- a base ``BiasAdd`` and a derived ``BiasAddScale`` -- and hand
them to the same kernel.

In doing so, you will learn about:

* Writing a ``@triton.aggregate`` that mixes a **runtime data member** (a
  ``tl.tensor`` -- here a bias pointer) with **compile-time data members**
  (``tl.constexpr`` shape / scale).

* **Inheriting** one aggregate from another: ``BiasAddScale(BiasAdd)`` adds a
  ``scale`` field and overrides the ``@triton.jit`` ``apply`` epilogue method.

* Constructing the object on the **host** and passing it as a single
  ``tl.constexpr`` kernel argument. Its runtime members are *lifted to runtime
  kernel parameters*, while its ``tl.constexpr`` members are baked in -- so
  changing the bias tensor reuses the same compiled kernel.

* Using the aggregate's type to drive **compile-time polymorphism**: the kernel
  calls ``epilogue.apply(...)`` and the concrete aggregate type selects which IR
  is emitted, with no runtime branching.

"""

# %%
# Motivation
# ----------
#
# A matmul is rarely the whole story: we usually fuse an *epilogue* onto the
# accumulator -- add a bias, scale the result, apply an activation. We would like
# to write the matmul body **once** and swap epilogues without rewriting the
# kernel or threading an ever-growing list of optional pointers and flags through
# its signature.
#
# A ``@triton.aggregate`` is the natural container for an epilogue: it bundles the
# epilogue's *operands* (a bias tensor) with its *configuration* (output width, a
# scale) and exposes the fused math as a ``@triton.jit`` method. Because the
# object is passed as a single ``tl.constexpr`` argument, its runtime members are
# lifted to kernel parameters automatically and its compile-time members
# specialize the kernel.

import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# %%
# A small validation helper
# -------------------------
#
# Each example checks its Triton result against the equivalent PyTorch expression
# and prints a checkmark with the max absolute difference.


def run_test(expect, fn, label, atol=5e-2):
    print(f"  {label}: ...", end="")
    actual = fn().to(expect.dtype)
    max_diff = torch.max(torch.abs(actual - expect)).item()
    icon = "✅" if torch.allclose(actual, expect, atol=atol) else "❌"
    print(f"\r  {label}: {icon}  (max abs diff {max_diff:.2e})")


# %%
# The base epilogue: ``BiasAdd``
# ------------------------------
#
# ``BiasAdd`` carries one **runtime** data member -- a ``bias`` pointer
# (``tl.tensor``) to a length-``N`` vector -- and one **compile-time** member,
# ``n_cols`` (used to mask the bias load). Its ``@triton.jit`` ``apply`` method is
# the fused epilogue: it adds the per-column bias to the matmul accumulator.
#
# The constructor is a ``@triton.constexpr_function`` so the object can be built
# on the host (we pass a raw ``torch.Tensor`` for the runtime ``bias`` field) and
# referenced by name inside a kernel.


@triton.aggregate
class BiasAdd:
    # Runtime data member: lifted to a runtime kernel parameter at launch.
    bias: tl.tensor
    # Compile-time data member: baked into the kernel.
    n_cols: tl.constexpr

    @triton.constexpr_function
    def __init__(self, bias, n_cols):
        self.bias = bias
        self.n_cols = tl.constexpr(n_cols)

    @triton.jit
    def apply(self, acc, offs_n):
        # The runtime `bias` member drives the load; the compile-time `n_cols`
        # member masks the tail.
        bias = tl.load(self.bias + offs_n, mask=offs_n < self.n_cols, other=0.0)
        return acc + bias[None, :]


# %%
# The derived epilogue: ``BiasAddScale``
# --------------------------------------
#
# ``BiasAddScale`` *inherits* from ``BiasAdd`` -- it reuses the ``bias`` and
# ``n_cols`` fields and adds a **runtime** ``scale``. Like ``bias``, the
# ``scale`` is a ``tl.tensor`` data member, so it is lifted to a runtime kernel
# parameter rather than baked in: changing it reuses the same compiled kernel. It
# overrides ``apply`` to scale the biased accumulator. Aggregate inheritance
# collects the parent fields first, then the child's, so the field order is
# ``(bias, n_cols, scale)``.


@triton.aggregate
class BiasAddScale(BiasAdd):
    # An additional runtime data member on top of the inherited fields: lifted to
    # a runtime kernel parameter, not baked in.
    scale: tl.tensor

    @triton.constexpr_function
    def __init__(self, bias, n_cols, scale):
        self.bias = bias
        self.n_cols = tl.constexpr(n_cols)
        self.scale = scale

    @triton.jit
    def apply(self, acc, offs_n):
        # Reuse the base epilogue (bias add), then apply the runtime scale. The
        # base method is called explicitly with `self`, which is a `BiasAddScale`.
        return BiasAdd.apply(self, acc, offs_n) * self.scale


# %%
# One kernel, any epilogue
# ------------------------
#
# ``matmul_kernel`` computes ``C = A @ B`` and then calls ``epilogue.apply(...)``.
# The ``epilogue`` parameter is a single ``tl.constexpr``: its concrete aggregate
# type (``BiasAdd`` or ``BiasAddScale``) is known at compile time, so the call is
# resolved with **no runtime branching**, and its runtime ``bias`` member was
# lifted to a runtime kernel parameter behind the scenes.


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,  #
    M, N, K,  #
    stride_am, stride_ak,  #
    stride_bk, stride_bn,  #
    stride_cm, stride_cn,  #
    epilogue: tl.constexpr,  # a host-built BiasAdd / BiasAddScale aggregate
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_mask = offs_k[None, :] < K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=k_mask, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # The fused epilogue. `epilogue` is a compile-time constant whose type selects
    # which `apply` (BiasAdd's or BiasAddScale's) is emitted here.
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if not epilogue is None:
        acc = epilogue.apply(acc, offs_cn)
    c = acc.to(c_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# %%
# Two launch functions, two epilogues, one kernel
# -----------------------------------------------
#
# Each launch function builds its epilogue object on the host -- passing the
# ``bias`` tensor as the runtime member -- and hands it to the *same*
# ``matmul_kernel``. ``matmul_bias_add`` builds a ``BiasAdd``;
# ``matmul_bias_add_scale`` builds a ``BiasAddScale``.

BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 64, 32


def _launch(a, b, epilogue):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        epilogue,  #
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,  #
    )
    return c


def matmul(a, b):
    return _launch(a, b, None)

def matmul_bias_add(a, b, bias):
    N = b.shape[1]
    # The base epilogue: built on the host with a runtime `bias` tensor.
    return _launch(a, b, BiasAdd(bias, N))


def matmul_bias_add_scale(a, b, bias, scale):
    N = b.shape[1]
    # The derived epilogue: same runtime `bias`, plus a runtime `scale`.
    return _launch(a, b, BiasAddScale(bias, N, scale))


# %%
# Let's check both against PyTorch.

torch.manual_seed(0)
M, N, K = 512, 256, 128
a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
bias = torch.randn((N, ), device=DEVICE, dtype=torch.float32)
scale = 0.5

ref = (a.float() @ b.float())
run_test(ref, lambda: matmul(a, b), "None epilogue")
run_test(ref + bias, lambda: matmul_bias_add(a, b, bias), "BiasAdd epilogue")
run_test((ref + bias) * scale, lambda: matmul_bias_add_scale(a, b, bias, scale), "BiasAddScale epilogue (derived)")

# %%
# Reusing the compiled kernel across runtime members
# --------------------------------------------------
#
# Because both ``bias`` and ``scale`` are *runtime* members, calling either launch
# function again with a **different bias tensor or scale** -- but the same
# compile-time members (``n_cols``) -- reuses the already-compiled kernel. Only
# changing a ``tl.constexpr`` member (e.g. a block size or ``n_cols``) would
# specialize a new one.

bias2 = torch.randn((N, ), device=DEVICE, dtype=torch.float32)
run_test(ref + bias2, lambda: matmul_bias_add(a, b, bias2), "BiasAdd epilogue (new bias, same kernel)")
run_test((ref + bias2) * 0.25, lambda: matmul_bias_add_scale(a, b, bias2, 0.25),
         "BiasAddScale epilogue (new bias & scale, same kernel)")

# %%
# Takeaways
# ---------
#
# * An epilogue is naturally a ``@triton.aggregate``: it bundles runtime operands
#   (a ``tl.tensor`` bias, a ``tl.tensor`` scale) with compile-time configuration
#   (``n_cols``) and exposes the fused math as a ``@triton.jit`` method.
#
# * Aggregates **inherit**: ``BiasAddScale(BiasAdd)`` reuses the parent's fields,
#   adds its own runtime ``scale`` member, and overrides ``apply`` to extend the
#   epilogue.
#
# * Building the object on the **host** and passing it as one ``tl.constexpr``
#   argument lifts its runtime members to kernel parameters while baking in its
#   constexpr members -- so a single kernel signature stays clean no matter how
#   rich the epilogue is, and changing a runtime member reuses the compiled kernel.
#
# * The aggregate's *type* drives compile-time polymorphism: ``matmul_kernel``
#   calls ``epilogue.apply(...)`` once, and the concrete type selects the emitted
#   IR -- two epilogues, two launch functions, one kernel.
