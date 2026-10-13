"""Tests for the gfx950 MFMA scheduler (schedule_hint="mfma-schedule").

The fixture is a small Gluon GEMM with local prefetch (the next K tile's LDS
reads are issued before the current tile's MFMAs), which is the loop shape the
scheduler interleaves. Everything is compiled for gfx950 without launching, and
the assertions are on the LLVM IR and the assembly.
"""

import pytest
import triton
from triton.backends.compiler import GPUTarget
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

GFX950 = GPUTarget("hip", "gfx950", 64)


@gluon.jit
def gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, PIN_ACC: gl.constexpr,
                SCHED_BARRIER: gl.constexpr):
    """C[M, N] = A[M, K] @ B[K, N] in fp16 with fp32 accumulation.

    A is row-major and B is a [K, N] view of a row-major [N, K] tensor (K contiguous for both, as the
    async copies vectorize along K). K must be a multiple of 2 * BLOCK_K.

    Two LDS buffers per operand. The K loop keeps tile k in registers, tile k + 1 in the other LDS
    buffer, and refills the freed buffer with tile k + 2, so each iteration's LDS reads do not
    depend on its MFMAs. With PIN_ACC the accumulator is pinned in AGPRs (cd_regclass="a"). With
    SCHED_BARRIER every iteration starts with a sched_barrier, as BlockPingpong's clusters do.
    """
    pid = gl.program_id(0)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    blocked_a: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
    blocked_b: gl.constexpr = gl.BlockedLayout([8, 1], [8, 8], [1, 4], [0, 1])
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                                     warps_per_cta=[2, 2])
    dot_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=8)
    dot_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=8)
    # Unswizzled LDS tiles: a direct-to-LDS load writes lane t's 16 bytes at LDS offset t * 16, and
    # with the blocked layouts above that is exactly where element (row, k) of each tile lives.
    shared_a: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    shared_b: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0, 1])
    smem_a0 = gl.allocate_shared_memory(gl.float16, [BLOCK_M, BLOCK_K], shared_a)
    smem_a1 = gl.allocate_shared_memory(gl.float16, [BLOCK_M, BLOCK_K], shared_a)
    smem_b0 = gl.allocate_shared_memory(gl.float16, [BLOCK_K, BLOCK_N], shared_b)
    smem_b1 = gl.allocate_shared_memory(gl.float16, [BLOCK_K, BLOCK_N], shared_b)

    offs_am = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, blocked_a))
    offs_ak = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, blocked_a))
    offs_bk = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, blocked_b))
    offs_bn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked_b))
    a_offs = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    b_offs = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    a_step = BLOCK_K * stride_ak
    b_step = BLOCK_K * stride_bk

    # Prologue: tile 0 into buffer 0, tile 1 into buffer 1, tile 0 into registers.
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smem_a0, a_ptr, a_offs)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smem_b0, b_ptr, b_offs)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smem_a1, a_ptr + a_step, a_offs)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smem_b1, b_ptr + b_step, b_offs)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.wait_group(1)
    a_regs = smem_a0.load(dot_a)
    b_regs = smem_b0.load(dot_b)

    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfma_layout)
    for k in range(0, gl.cdiv(K, BLOCK_K) - 2, 2):
        if SCHED_BARRIER:
            gl.amd.hint.sched_barrier()
        # Buffer 0 held tile k, now in registers: refill it with tile k + 2, then read tile k + 1.
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smem_a0, a_ptr + (k + 2) * a_step, a_offs)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smem_b0, b_ptr + (k + 2) * b_step, b_offs)
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.wait_group(1)
        a_next = smem_a1.load(dot_a)
        b_next = smem_b1.load(dot_b)
        if PIN_ACC:
            acc = gl.amd.cdna3.mfma(a_regs, b_regs, acc, cd_regclass="a")
        else:
            acc = gl.amd.cdna3.mfma(a_regs, b_regs, acc)
        # Same with the buffers swapped.
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smem_a1, a_ptr + (k + 3) * a_step, a_offs)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smem_b1, b_ptr + (k + 3) * b_step, b_offs)
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.wait_group(1)
        a_regs = smem_a0.load(dot_a)
        b_regs = smem_b0.load(dot_b)
        if PIN_ACC:
            acc = gl.amd.cdna3.mfma(a_next, b_next, acc, cd_regclass="a")
        else:
            acc = gl.amd.cdna3.mfma(a_next, b_next, acc)

    # Epilogue: the last two tiles are in registers and in buffer 1.
    gl.amd.cdna4.async_copy.wait_group(0)
    a_next = smem_a1.load(dot_a)
    b_next = smem_b1.load(dot_b)
    if PIN_ACC:
        acc = gl.amd.cdna3.mfma(a_regs, b_regs, acc, cd_regclass="a")
        acc = gl.amd.cdna3.mfma(a_next, b_next, acc, cd_regclass="a")
    else:
        acc = gl.amd.cdna3.mfma(a_regs, b_regs, acc)
        acc = gl.amd.cdna3.mfma(a_next, b_next, acc)

    offs_cm = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, mfma_layout))
    offs_cn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout))
    gl.store(c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn, acc)


@gluon.jit
def scale_kernel(x_ptr, y_ptr, n, BLOCK: gl.constexpr, WARP_SIZE: gl.constexpr):
    """No MFMA loop: the scheduler must leave this kernel alone."""
    layout: gl.constexpr = gl.BlockedLayout([4], [WARP_SIZE], [4], [0])
    offs = gl.program_id(0) * BLOCK + gl.arange(0, BLOCK, layout=layout)
    x = gl.load(x_ptr + offs, mask=offs < n)
    gl.store(y_ptr + offs, x * 2.0, mask=offs < n)


def _compile_gemm(pin_acc=False, sched_barrier=False, **options):
    # Specialize as the JIT would for aligned tensors: unit K strides become constexpr 1 and the
    # other arguments are 16-byte divisible. The direct-to-LDS loads need both to vectorize.
    signature = {
        "a_ptr": "*fp16", "b_ptr": "*fp16", "c_ptr": "*fp32",  #
        "M": "i32", "N": "i32", "K": "i32",  #
        "stride_am": "i32", "stride_ak": "constexpr",  #
        "stride_bk": "constexpr", "stride_bn": "i32",  #
        "stride_cm": "i32", "stride_cn": "constexpr",  #
        "BLOCK_M": "constexpr", "BLOCK_N": "constexpr", "BLOCK_K": "constexpr", "PIN_ACC": "constexpr", "SCHED_BARRIER":
        "constexpr"
    }
    constexprs = {
        "stride_ak": 1, "stride_bk": 1, "stride_cn": 1,  #
        "BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "PIN_ACC": pin_acc, "SCHED_BARRIER": sched_barrier
    }
    names = list(signature)
    attrs = {(names.index(n), ): [["tt.divisibility", 16]] for n in names if signature[n] != "constexpr"}
    src = gluon._runtime.GluonASTSource(gemm_kernel, signature, constexprs, attrs=attrs)
    return triton.compile(src, target=GFX950, options={"num_warps": 4, **options})


def _compile_scale(target=GFX950, **options):
    src = gluon._runtime.GluonASTSource(
        scale_kernel, {"x_ptr": "*fp32", "y_ptr": "*fp32", "n": "i32", "BLOCK": "constexpr", "WARP_SIZE": "constexpr"},
        {"BLOCK": 1024, "WARP_SIZE": target.warp_size})
    return triton.compile(src, target=target, options={"num_warps": 4, **options})


SCHED_BARRIER = "llvm.amdgcn.sched.barrier"
ACC_PIN = '"=a,0"'


def test_scheduler_off_by_default():
    k = _compile_gemm()
    assert SCHED_BARRIER not in k.asm["llir"]
    assert "v_mfma" in k.asm["amdgcn"]


def test_unknown_hint_is_rejected():
    # A misspelled hint must not pass for an off hint.
    with pytest.raises(ValueError, match="schedule_hint"):
        _compile_gemm(schedule_hint="mfma_schedule")


def test_interleave_schedules_the_hot_loop():
    k = _compile_gemm(schedule_hint="mfma-schedule")
    # The interleave is pinned with sched.barriers in front of the memory anchors.
    assert SCHED_BARRIER in k.asm["llir"]
    amdgcn = k.asm["amdgcn"]
    assert "sched_barrier" in amdgcn
    assert "v_mfma" in amdgcn


def test_interleave_keeps_accumulator_pins():
    # cd_regclass="a" pins every MFMA's C and D with an empty tied inline asm. The scheduler moves
    # the pins with their MFMAs, so all of them survive into the scheduled IR.
    unscheduled = _compile_gemm(pin_acc=True)
    scheduled = _compile_gemm(pin_acc=True, schedule_hint="mfma-schedule")
    assert SCHED_BARRIER in scheduled.asm["llir"]
    pins = unscheduled.asm["llir"].count(ACC_PIN)
    assert pins > 0
    assert scheduled.asm["llir"].count(ACC_PIN) == pins


def test_pre_barriered_loop_is_scheduled_span_by_span():
    # A loop that already carries sched.barriers (BlockPingpong's clusters, or an explicit hint)
    # is cut at them and each span is interleaved on its own; the pre-existing barriers stay.
    before = _compile_gemm(sched_barrier=True)
    pre_existing = before.asm["llir"].count(SCHED_BARRIER)
    assert pre_existing > 0
    k = _compile_gemm(sched_barrier=True, schedule_hint="mfma-schedule")
    assert k.asm["llir"].count(SCHED_BARRIER) > pre_existing


def test_kernel_without_mfma_loop_is_left_alone():
    k = _compile_scale(schedule_hint="mfma-schedule")
    assert SCHED_BARRIER not in k.asm["llir"]


@pytest.mark.parametrize("arch", ["gfx942", "gfx1250"])
def test_other_archs_ignore_the_hint(arch):
    # gfx950-only: on other targets the hint is accepted and does nothing.
    k = _compile_scale(target=GPUTarget("hip", arch, 32 if arch == "gfx1250" else 64), schedule_hint="mfma-schedule")
    assert SCHED_BARRIER not in k.asm["llir"]
