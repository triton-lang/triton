import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
import re


def test_compile_only_sm100() -> None:

    @triton.jit
    def kernel_add(a, b, c):
        idx = tl.arange(0, 32)
        tl.store(c + idx, tl.load(a + idx) + tl.load(b + idx))

    k = triton.compile(
        triton.compiler.ASTSource(fn=kernel_add, signature={"a": "*fp32", "b": "*fp32", "c": "*fp32"}, constexprs={}),
        target=GPUTarget("cuda", 100, 32))
    ptx = k.asm["ptx"]
    assert ".target sm_100a" in ptx
    assert ".address_size 64" in ptx
    assert k.asm["cubin"] != b""


def test_compile_only_dot() -> None:

    @triton.jit
    def simple_dot(a_base, b_base, out):
        SIZE: tl.constexpr = 64
        a_ptr = a_base + tl.arange(0, SIZE)[:, None] * SIZE + tl.arange(0, SIZE)[None, :]
        b_ptr = b_base + tl.arange(0, SIZE)[:, None] * SIZE + tl.arange(0, SIZE)[None, :]
        a = tl.load(a_ptr)
        b = tl.load(b_ptr)
        c = tl.dot(a, b)
        out_ptr = out + tl.arange(0, SIZE)[:, None] * SIZE + tl.arange(0, SIZE)[None, :]
        tl.store(out_ptr, c)

    k = triton.compile(
        triton.compiler.ASTSource(fn=simple_dot, signature={"a_base": "*fp16", "b_base": "*fp16", "out": "*fp16"},
                                  constexprs={}), target=GPUTarget("cuda", 100, 32))
    ttgir = k.asm["ttgir"]
    pattern = (r"%(?P<A>\d+) = tt\.load"
               r"(.|\n)*?"
               r"%(?P<A_SHMEM>\d+) = ttg\.local_alloc %(?P=A)"
               r"(.|\n)*?"
               r"%(?P<B>\d+) = tt\.load"
               r"(.|\n)*?"
               r"%(?P<B_SHMEM>\d+) = ttg\.local_alloc %(?P=B)"
               r"(.|\n)*?"
               r"%(?P<TMEM_BASE>\d+) = ttng\.tmem_alloc"
               r"(.|\n)*?"
               r"ttng\.tc_gen5_mma %(?P=A_SHMEM), %(?P=B_SHMEM), %(?P=TMEM_BASE)"
               r"(.|\n)*?"
               r"ttng\.tmem_load %(?P=TMEM_BASE)")

    assert re.search(pattern, str(ttgir)), "The TTGIR does not match the expected pattern."

    ptx = k.asm["ptx"]
    pattern = (r"mov\.u32 	%r(?P<G>\d+), global_smem;"
               r"(.|\n)*"
               r"tcgen05\.alloc\.cta_group::1\.sync\.aligned\.shared::cta\.b32 \[%r(?P=G)], 64"
               r"(.|\n)*"
               r"tcgen05\.relinquish_alloc_permit\.cta_group::1\.sync\.aligned"
               r"(.|\n)*"
               r"tcgen05\.st\.sync\.aligned\.16x32bx2.x32.b32"
               r"(.|\n)*"
               r"tcgen05\.mma\.cta_group::1.kind::f16"
               r"(.|\n)*"
               r"tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64"
               r"(.|\n)*"
               r"mbarrier.try_wait.parity.shared.b64"
               r"(.|\n)*"
               r"tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
               r"(.|\n)*"
               r"tcgen05.wait::ld.sync.aligned")
    assert re.search(pattern, str(ptx)), "The PTX does not match the expected pattern."
    assert k.asm["cubin"] != b""


def test_compile_only_k_loop() -> None:

    @triton.jit
    def k_loop(a_base, b_base, out, k_tiles):
        SIZE: tl.constexpr = 128
        offs_k = tl.arange(0, SIZE)
        c = tl.zeros((SIZE, SIZE), dtype=tl.float32)
        for k in range(k_tiles):
            a_ptr = a_base + tl.arange(0, SIZE)[:, None] * SIZE + offs_k[None, :]
            b_ptr = b_base + offs_k[:, None] * SIZE + tl.arange(0, SIZE)[None, :]
            offs_k = offs_k + SIZE
            a = tl.load(a_ptr)
            b = tl.load(b_ptr)
            c += tl.dot(a, b)
        out_ptr = out + tl.arange(0, SIZE)[:, None] * SIZE + tl.arange(0, SIZE)[None, :]
        tl.store(out_ptr, c)

    k = triton.compile(
        triton.compiler.ASTSource(fn=k_loop,
                                  signature={"a_base": "*fp16", "b_base": "*fp16", "out": "*fp16", "k_tiles":
                                             "i32"}, constexprs={}), target=GPUTarget("cuda", 100, 32))
    ttgir = k.asm["ttgir"]

    pattern = (r"%(?P<TMEM_BASE>\w+) = arith.constant dense<0.000000e\+00>"
               r"(.|\n)*?"
               r"%(?P<TMEM>\w+) = ttng\.tmem_alloc %(?P=TMEM_BASE)"
               r"(.|\n)*?"
               r"scf\.for"
               r"(.|\n)*?"
               r"%(?P<A>\w+) = tt\.load"
               r"(.|\n)*?"
               r"%(?P<A_SHMEM>\w+) = ttg\.local_alloc %(?P=A)"
               r"(.|\n)*?"
               r"%(?P<B>\w+) = tt\.load"
               r"(.|\n)*?"
               r"%(?P<B_SHMEM>\w+) = ttg\.local_alloc %(?P=B)"
               r"(.|\n)*?"
               r"ttng\.tc_gen5_mma %(?P=A_SHMEM), %(?P=B_SHMEM), %(?P=TMEM)"
               r"(.|\n)*?"
               r"scf\.yield")

    assert re.search(pattern, str(ttgir)), "The TTGIR does not match the expected pattern."
    assert k.asm["cubin"] != b""


def test_compile_only_dot_mxfp() -> None:

    @triton.jit
    def simple_dot_mxfp(a_base, b_base, a_scale, b_scale, out, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                        BLOCK_K: tl.constexpr):
        PACKED_BLOCK_K_A: tl.constexpr = BLOCK_K
        PACKED_BLOCK_K_B: tl.constexpr = BLOCK_K
        a_ptr = a_base + tl.arange(0, BLOCK_M)[:, None] * PACKED_BLOCK_K_A + tl.arange(0, PACKED_BLOCK_K_A)[None, :]
        b_ptr = b_base + tl.arange(0, PACKED_BLOCK_K_B)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]

        SCALE_BLOCK_K: tl.constexpr = BLOCK_K // 32
        scale_a_ptr = a_scale + tl.arange(0, BLOCK_M)[:, None] * SCALE_BLOCK_K + tl.arange(0, SCALE_BLOCK_K)[None, :]
        scale_b_ptr = b_scale + tl.arange(0, BLOCK_N)[:, None] * SCALE_BLOCK_K + tl.arange(0, SCALE_BLOCK_K)[None, :]

        a = tl.load(a_ptr)
        b = tl.load(b_ptr)
        a_scale = tl.load(scale_a_ptr)
        b_scale = tl.load(scale_b_ptr)
        c = tl.dot_scaled(a, a_scale, "e4m3", b, b_scale, "e4m3")
        out_ptr = out + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        tl.store(out_ptr, c)

    k = triton.compile(
        triton.compiler.ASTSource(
            fn=simple_dot_mxfp, signature={
                "a_base": "*u8", "b_base": "*u8", "a_scale": "*u8", "b_scale": "*u8", "out": "*fp32", "BLOCK_M":
                "constexpr", "BLOCK_N": "constexpr", "BLOCK_K": "constexpr"
            }, constexprs={"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}), target=GPUTarget("cuda", 100, 32))
    ttgir = k.asm["ttgir"]
    pattern = (r"ttng.tc_gen5_mma_scaled (.*) lhs = e4m3 rhs = e4m3")
    assert re.search(pattern, str(ttgir)), "The TTGIR does not match the expected pattern."

    ptx = k.asm["ptx"]
    pattern = (r"tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X")
    assert re.search(pattern, str(ptx)), "The PTX does not match the expected pattern."
    assert k.asm["cubin"] != b""
