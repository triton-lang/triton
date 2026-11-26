# ruff: noqa: E402
"""Scenarios for test_gluon_cdna4_consan.py. Each one runs in its own process."""
import os
import re
import sys
import tempfile
import warnings

os.environ["TRITON_INSTRUMENTATION_MODE"] = "consan"
import torch
import triton
import triton.language as tl
from triton import knobs
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import warp_pipeline_stage
from triton.experimental.gluon.language.amd.cdna4 import async_copy, buffer
from triton.runtime._allocation import set_profile_allocator

knobs.compilation.instrumentation_mode = "consan"
knobs.refresh_knobs()

PENDING_COPY = "Accessing buffer with pending access. Pending access type: async_copy_global_to_shared"


def alloc_fn(size, alignment, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


set_profile_allocator(alloc_fn)


def _assert_instrumented(compiled, copy_path=None, *, check_copies=True):
    # Saved TTGIR precedes ConSan. Verify the lowered helpers AND the scratch
    # allocation used by the launcher, rather than merely an option or IR name.
    assert compiled.metadata.profile_scratch_size > 0
    assert compiled.metadata.global_scratch_size == 0
    ttgir = compiled.asm["ttgir"]
    assert '"ttg.threads-per-warp" = 64' in ttgir
    llir = compiled.asm["llir"]
    assert "__triton_consan_" in llir
    if check_copies:
        assert "__triton_consan_check_outstanding_commits" in llir
        assert "__triton_consan_stage_access_for_commit" in llir
        assert PENDING_COPY in llir
    if copy_path is not None:
        if copy_path == "buffer":
            assert "amdg.buffer_load_to_local" in ttgir
            assert re.search(r"buffer_load[^\n]*\blds\b", compiled.asm["amdgcn"])
        else:
            assert "ttg.async_copy_global_to_local" in ttgir
            assert "global_load_lds" in compiled.asm["amdgcn"]


@gluon.jit
def _copy(dest, source, offsets, BUFFER: gl.constexpr, VALID: gl.constexpr):
    if BUFFER:
        desc = buffer.make_buffer_descriptor(source, VALID * 4)
        buffer.load_to_shared(dest, desc, offsets, mask=offsets < VALID)
    else:
        async_copy.global_load_to_shared(dest, source + offsets, mask=offsets < VALID, other=0)


@gluon.jit
def _access_kernel(source, output, BUFFER: gl.constexpr, MODE: gl.constexpr, WAIT: gl.constexpr, VEC: gl.constexpr,
                   VALID: gl.constexpr, BLOCK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([VEC], [64], [gl.num_warps()], [0])
    # Reading with a different distribution also exercises ordinary block-level
    # LDS synchronization; there are no lane-varying source-level scalars.
    read_layout: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    offsets = gl.arange(0, BLOCK, layout)
    read_offsets = gl.arange(0, BLOCK, read_layout)
    smem = gl.allocate_shared_memory(gl.float32, [BLOCK], shared)
    if MODE == "ordinary":
        smem.store(offsets.to(gl.float32) + 7)
        before = smem.load(read_layout)
    _copy(smem, source, offsets, BUFFER, VALID)
    async_copy.commit_group()
    if WAIT:
        async_copy.wait_group(0)
    if MODE == "copy_overwrite":
        _copy(smem, source + BLOCK, offsets, BUFFER, VALID)
        async_copy.commit_group()
        async_copy.wait_group(0)
    elif MODE == "store_overwrite":
        smem.store(offsets.to(gl.float32) + 11)
        # Keep the first copy alive across the ordinary store even when it
        # would otherwise have no observable user; safe mode waited above.
        async_copy.wait_group(0)
    values = smem.load(read_layout)
    if MODE == "ordinary":
        values += before
        smem.store(gl.convert_layout(values, layout))
        values = smem.load(read_layout)
    gl.store(output + read_offsets, values)
    async_copy.wait_group(0)


@gluon.jit
def _partial_wait_kernel(source, output, trips, BUFFER: gl.constexpr, WAIT: gl.constexpr, BUFFERS: gl.constexpr,
                         BLOCK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([4], [64], [4], [0])
    # The leading dimension selects buffers; only each payload is laid out in
    # LDS. Keeping the layout rank one makes index(b) a valid rank-one view.
    shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    offsets = gl.arange(0, BLOCK, layout)
    smem = gl.allocate_shared_memory(gl.float32, [BUFFERS, BLOCK], shared)
    # Reuse the same descriptors across a runtime loop. Every group targets a
    # distinct subview, so a partial wait must make precisely the oldest safe.
    for i in range(trips):
        for b in gl.static_range(BUFFERS):
            _copy(smem.index(b), source + (i * BUFFERS + b) * BLOCK, offsets, BUFFER, BLOCK)
            async_copy.commit_group()
        async_copy.wait_group(WAIT)
        first = smem.index(0).load(layout)
        async_copy.wait_group(0)
        gl.store(output + i * BUFFERS * BLOCK + offsets, first)
        for b in gl.static_range(1, BUFFERS):
            values = smem.index(b).load(layout)
            gl.store(output + (i * BUFFERS + b) * BLOCK + offsets, values)


@gluon.jit
def _warp_pipeline_kernel(source, output, trips, STAGED: gl.constexpr, BLOCK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([4], [64], [4], [0])
    shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    offsets = gl.arange(0, BLOCK, layout)
    smem = gl.allocate_shared_memory(gl.float32, [BLOCK], shared)
    for i in range(trips):
        if STAGED:
            with warp_pipeline_stage("copy", priority=1):
                smem.store(gl.load(source + i * BLOCK + offsets))
            with warp_pipeline_stage("consume", priority=0):
                values = smem.load(layout)
                gl.store(output + i * BLOCK + offsets, values + 1)
        else:
            smem.store(gl.load(source + i * BLOCK + offsets))
            values = smem.load(layout)
            gl.store(output + i * BLOCK + offsets, values + 1)


@triton.jit
def _pipelined_matmul(a, b, output, K: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    k = tl.arange(0, 32)
    acc = tl.zeros((BLOCK, BLOCK), tl.float32)
    for i in range(tl.cdiv(K, 32)):
        av = tl.load(a + rows[:, None] * K + (i * 32 + k[None, :]))
        bv = tl.load(b + (i * 32 + k[:, None]) * BLOCK + cols[None, :])
        acc = tl.dot(av, bv, acc)
    tl.store(output + rows[:, None] * BLOCK + cols[None, :], acc)


def _run_access(copy_path, mode, wait, vec, num_warps, valid):
    block = 2048
    source = torch.arange(2 * block, device="cuda", dtype=torch.float32)
    output = torch.empty((block, ), device="cuda", dtype=torch.float32)
    options = {
        "BUFFER": copy_path == "buffer", "MODE": mode, "WAIT": wait, "VEC": vec, "VALID": valid, "BLOCK": block,
        "num_warps": num_warps
    }
    compiled = _access_kernel.warmup(source, output, grid=(1, ), **options)
    _assert_instrumented(compiled, copy_path)
    _access_kernel[(1, )](source, output, **options)
    torch.cuda.synchronize()
    if mode == "copy_overwrite":
        expected = source[block:].clone()
    elif mode == "store_overwrite":
        expected = source[:block] + 11
    else:
        expected = source[:block].clone()
    if valid < block:
        expected[valid:] = 0
    if mode == "ordinary":
        expected += source[:block] + 7
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def _run_partial_wait(copy_path, mode, buffers):
    block, trips = 1024, 9
    source = torch.arange(trips * buffers * block, device="cuda", dtype=torch.float32)
    output = torch.empty_like(source)
    wait = {"full": 0, "partial": buffers - 1, "off_by_one": buffers}[mode]
    options = {"BUFFER": copy_path == "buffer", "WAIT": wait, "BUFFERS": buffers, "BLOCK": block, "num_warps": 4}
    compiled = _partial_wait_kernel.warmup(source, output, trips, grid=(1, ), **options)
    _assert_instrumented(compiled, copy_path)
    assert "scf.for" in compiled.asm["ttgir"]
    _partial_wait_kernel[(1, )](source, output, trips, **options)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, source, rtol=0, atol=0)


def _run_warp_pipeline(staged):
    block, trips = 1024, 5
    source = torch.arange(trips * block, device="cuda", dtype=torch.float32)
    output = torch.empty_like(source)
    options = {"STAGED": staged, "BLOCK": block, "num_warps": 4}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compiled = _warp_pipeline_kernel.warmup(source, output, trips, grid=(1, ), **options)
    messages = [str(w.message) for w in caught]
    for message in messages:
        print(f"Compiler warning: {message}", flush=True)
    if staged:
        assert any("ConSan" in m and "warp_pipeline" in m and "disabled" in m for m in messages), messages
        assert "__triton_consan_" not in compiled.asm["llir"]
        assert compiled.metadata.profile_scratch_size == 0
        assert "s_setprio" in compiled.asm["amdgcn"]
    else:
        assert not messages, messages
        _assert_instrumented(compiled, check_copies=False)
    _warp_pipeline_kernel[(1, )](source, output, trips, **options)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, source + 1, rtol=0, atol=0)


def _run_pingpong(mode):
    # A real matrix loop gives the Triton pipeliner eligible asynchronous LDS
    # copies, unlike an elementwise loop that might silently avoid the feature.
    knobs.amd.use_async_copy = True
    knobs.amd.use_block_pingpong = mode != "disabled"
    stages = 1 if mode == "single_stage" else 3
    # Eight waves and dimensions above 64 select the CDNA4 three-stage async
    # pingpong pattern in BlockPingpong, including its phase-shifted barriers.
    block, k = 128, 256
    # Small integral fp16 inputs make the exact fp32 reference unambiguous.
    a = (torch.arange(block * k, device="cuda") % 5 - 2).to(torch.float16).reshape(block, k)
    b = (torch.arange(k * block, device="cuda") % 7 - 3).to(torch.float16).reshape(k, block)
    output = torch.empty((block, block), device="cuda", dtype=torch.float32)
    options = {"K": k, "BLOCK": block, "num_warps": 8, "num_stages": stages}
    if mode == "pingpong":
        with knobs.compilation.scope():
            knobs.compilation.instrumentation_mode = ""
            baseline = _pipelined_matmul.warmup(a, b, output, grid=(1, ), **options)
        # Prove this particular kernel receives the phase-shifted schedule
        # before testing that ConSan suppresses it. An enabled knob alone does
        # not show that BlockPingpong matched an eligible loop.
        assert "amdg.cond_barrier" in baseline.asm["ttgir"]
        assert "s_barrier" in baseline.asm["amdgcn"]
        assert "__triton_consan_" not in baseline.asm["llir"]
        assert baseline.metadata.profile_scratch_size == 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compiled = _pipelined_matmul.warmup(a, b, output, grid=(1, ), **options)
    messages = [str(w.message) for w in caught]
    for message in messages:
        print(f"Compiler warning: {message}", flush=True)
    if mode == "pingpong":
        assert any("ConSan" in m and "pingpong" in m for m in messages), messages
    else:
        assert not messages, messages
    if stages > 1:
        assert "async_copy_global_to_local" in compiled.asm["ttgir"] or "buffer_load_to_local" in compiled.asm["ttgir"]
        _assert_instrumented(compiled)
    else:
        assert compiled.metadata.profile_scratch_size > 0
        assert "__triton_consan_" in compiled.asm["llir"]
    assert "amdg.cond_barrier" not in compiled.asm["ttgir"]
    _pipelined_matmul[(1, )](a, b, output, **options)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, a.float() @ b.float(), rtol=0, atol=0)


def async_copy_access():
    copy_path, mode, wait, vec, num_warps, valid = sys.argv[2:]
    _run_access(copy_path, mode, wait == "True", int(vec), int(num_warps), int(valid))


def async_copy_partial_wait_loop():
    copy_path, mode, buffers = sys.argv[2:]
    _run_partial_wait(copy_path, mode, int(buffers))


def warp_pipeline_execution():
    _run_warp_pipeline(sys.argv[2] == "True")


def pingpong_execution():
    _run_pingpong(sys.argv[2])


if __name__ == "__main__":
    tests = {
        "async_copy_access": async_copy_access,
        "async_copy_partial_wait_loop": async_copy_partial_wait_loop,
        "warp_pipeline_execution": warp_pipeline_execution,
        "pingpong_execution": pingpong_execution,
    }
    # Warning checks must compile the kernel even when an earlier run filled the normal cache.
    with tempfile.TemporaryDirectory(prefix="triton-cdna4-consan-") as cache_dir, knobs.cache.scope():
        knobs.cache.dir = cache_dir
        tests[sys.argv[1]]()
        torch.cuda.synchronize()
    print("ConSan helper completed", flush=True)
