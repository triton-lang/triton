"""ConSan compile-only checks shared by the AMD targets that support ConSan.

The tests compile for gfx950 and gfx1250, with gfx942 as a target without ConSan
support. They check when the ConSan passes run and how ConSan interacts with
warp pipelines and block ping-pong. They need no GPU.
"""

import warnings

import pytest
import triton
import triton.language as tl
from triton import knobs
from triton._C.libtriton import amd, ir
from triton.backends.amd import compiler
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon._runtime import GluonASTSource
from triton.experimental.gluon.language.amd import warp_pipeline_stage
from triton.runtime.driver import DriverConfig


@pytest.fixture(autouse=True)
def compilation_environment(monkeypatch, tmp_path):

    def driver_access(_):
        pytest.fail("Compile-only tests must not access the GPU driver")

    monkeypatch.setattr(DriverConfig, "active", property(driver_access))
    with knobs.cache.scope(), knobs.runtime.scope(), knobs.compilation.scope(), knobs.amd.scope():
        knobs.cache.dir = str(tmp_path / "cache")
        knobs.runtime.override_arch = None
        knobs.compilation.instrumentation_mode = ""
        knobs.compilation.store_binary_only = False
        knobs.amd.use_async_copy = None
        knobs.amd.use_block_pingpong = None
        yield


@pytest.fixture
def pass_calls(monkeypatch):
    # Observe pass registration while still running the actual compiler passes.
    calls = {"prepare": [], "instrument": [], "pipeline": [], "pingpong": []}

    def observe(namespace, name, key):
        original = getattr(namespace, name)

        def add(pm, *args):
            calls[key].append(args)
            return original(pm, *args)

        monkeypatch.setattr(namespace, name, add)

    observe(compiler.passes.ttgpuir, "add_prepare_consan_captures", "prepare")
    observe(compiler.passes.ttgpuir, "add_concurrency_sanitizer", "instrument")
    observe(compiler.amd.passes.ttgpuir, "add_pipeline", "pipeline")
    observe(compiler.amd.passes.ttgpuir, "add_block_pingpong", "pingpong")
    return calls


@gluon.jit
def _shared_copy(src, dst, count, WARP_SIZE: gl.constexpr, STAGED: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [WARP_SIZE], [4], [0])
    shared_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    shared = gl.allocate_shared_memory(gl.float32, [256], shared_layout)
    offsets = gl.arange(0, 256, layout=layout)
    for i in range(count):
        if STAGED:
            with warp_pipeline_stage("load", priority=1):
                shared.store(gl.load(src + i * 256 + offsets))
            with warp_pipeline_stage("store", priority=0):
                gl.store(dst + i * 256 + offsets, shared.load(layout))
        else:
            shared.store(gl.load(src + i * 256 + offsets))
            gl.store(dst + i * 256 + offsets, shared.load(layout))


def _compile_shared_copy(arch, mode, staged=False):
    warp_size = 32 if arch == "gfx1250" else 64
    source = GluonASTSource(_shared_copy, {"src": "*fp32", "dst": "*fp32", "count": "i32"},
                            constexprs={"WARP_SIZE": warp_size, "STAGED": staged})
    return triton.compile(source, target=GPUTarget("hip", arch, warp_size), options={"instrumentation_mode": mode})


def _assert_instrumentation(compiled, enabled):
    assert ("__triton_consan_" in compiled.asm["llir"]) == enabled
    assert (compiled.metadata.profile_scratch_size > 0) == enabled
    assert compiled.metadata.global_scratch_size == 0
    assert compiled.asm["hsaco"]


@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "gfx1250"])
@pytest.mark.parametrize("mode", ["", "consan"])
def test_consan_target_activation(arch, mode, pass_calls):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compiled = _compile_shared_copy(arch, mode)
    enabled = mode == "consan" and arch in ("gfx950", "gfx1250")
    _assert_instrumentation(compiled, enabled)
    assert bool(pass_calls["prepare"]) == enabled
    assert bool(pass_calls["instrument"]) == enabled
    assert not caught


@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "gfx1250"])
@pytest.mark.parametrize("mode", ["", "consan"])
def test_consan_warp_pipeline_preserves_schedule(arch, mode, pass_calls):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compiled = _compile_shared_copy(arch, mode, staged=True)
    expect_warning = mode == "consan" and arch in ("gfx950", "gfx1250")
    assert len(caught) == int(expect_warning)
    if expect_warning:
        message = str(caught[0].message)
        assert "capture preparation and instrumentation are disabled" in message
        assert "warp_pipeline_stage" in message
        assert "no ConSan race coverage" in message
    assert "triton.warp_pipeline.stage" in compiled.asm["ttgir"]
    assert "s_setprio" in compiled.asm["amdgcn"]
    _assert_instrumentation(compiled, False)
    assert not pass_calls["prepare"]
    assert not pass_calls["instrument"]


@triton.jit
def _matmul(a, b, c, count, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    m = tl.arange(0, BLOCK_M)
    n = tl.arange(0, BLOCK_N)
    k = tl.arange(0, BLOCK_K)
    acc = tl.full((BLOCK_M, BLOCK_N), 0, tl.float32)
    for i in range(count):
        lhs = tl.load(a + m[:, None] * BLOCK_K + k[None, :] + i * BLOCK_M * BLOCK_K)
        rhs = tl.load(b + k[:, None] * BLOCK_N + n[None, :] + i * BLOCK_K * BLOCK_N)
        acc += tl.dot(lhs, rhs)
    tl.store(c + m[:, None] * BLOCK_N + n[None, :], acc)


def _matmul_source(block_m=32, block_n=32, block_k=32):
    return ASTSource(_matmul, {"a": "*fp16", "b": "*fp16", "c": "*fp32", "count": "i32"},
                     constexprs={"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K":
                                 block_k}, attrs={(index, ): [["tt.divisibility", 16]]
                                                  for index in range(3)})


@pytest.mark.parametrize(
    "arch,mode,pingpong,num_stages,expect_warning,expect_pingpong",
    [
        ("gfx950", "consan", None, 3, True, False),
        ("gfx950", "consan", True, 3, True, False),
        ("gfx950", "consan", True, 1, False, False),
        ("gfx950", "consan", False, 3, False, False),
        ("gfx950", "", True, 3, False, True),
        ("gfx942", "consan", True, 3, False, True),
        ("gfx1250", "consan", None, 3, False, False),
        ("gfx1250", "consan", True, 3, True, False),
        ("gfx1250", "consan", True, 1, False, False),
    ],
)
def test_consan_automatic_pingpong(arch, mode, pingpong, num_stages, expect_warning, expect_pingpong, pass_calls):
    knobs.amd.use_block_pingpong = pingpong
    warp_size = 32 if arch == "gfx1250" else 64
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compiled = triton.compile(_matmul_source(), target=GPUTarget("hip", arch, warp_size),
                                  options={"instrumentation_mode": mode, "num_stages": num_stages})
    assert len(caught) == int(expect_warning)
    if expect_warning:
        assert "ConSan disables automatic block pingpong scheduling" in str(caught[0].message)
        assert "phase-shifted warp execution" in str(caught[0].message)
        assert "ConSan checks this kernel without pingpong" in str(caught[0].message)
        assert "phase-shifted schedule is not checked" in str(caught[0].message)
    enabled = mode == "consan" and arch in ("gfx950", "gfx1250")
    _assert_instrumentation(compiled, enabled)
    # The pipeliner's pingpong preparation must agree with the later pass.
    assert len(pass_calls["pipeline"]) == 1
    assert pass_calls["pipeline"][0][1] == expect_pingpong
    assert bool(pass_calls["pingpong"]) == expect_pingpong


@pytest.mark.parametrize(
    "arch,pingpong,num_stages",
    [
        ("gfx950", None, 3),
        ("gfx950", None, 1),
        ("gfx950", True, 2),
        ("gfx950", False, 3),
        ("gfx1250", None, 3),
        ("gfx1250", True, 3),
    ],
)
def test_consan_pingpong_warning_matches_pass_registration(arch, pingpong, num_stages, pass_calls):
    # The warning must fire exactly when the same compilation without ConSan adds the pingpong pass.
    knobs.amd.use_block_pingpong = pingpong
    warp_size = 32 if arch == "gfx1250" else 64
    target = GPUTarget("hip", arch, warp_size)
    triton.compile(_matmul_source(), target=target, options={"num_stages": num_stages})
    adds_pass = bool(pass_calls["pingpong"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        triton.compile(_matmul_source(), target=target,
                       options={"num_stages": num_stages, "instrumentation_mode": "consan"})
    warned = any("ConSan disables automatic block pingpong scheduling" in str(w.message) for w in caught)
    assert warned == adds_pass


def test_consan_removes_phase_shifted_pingpong():
    # Block ping-pong is on by default only on gfx950. This tile and
    # eight-warps configuration qualifies for asymmetric synchronization,
    # unlike smaller tiles or four-warps scheduling.
    knobs.amd.use_block_pingpong = True
    target = GPUTarget("hip", "gfx950", 64)
    source = _matmul_source(128, 128, 64)
    options = {"num_stages": 3, "num_warps": 8}
    baseline = triton.compile(source, target=target, options=options)
    assert "amdg.cond_barrier" in baseline.asm["ttgir"]
    assert "s_barrier" in baseline.asm["amdgcn"]
    with pytest.warns(UserWarning, match="ConSan disables automatic block pingpong scheduling"):
        compiled = triton.compile(source, target=target, options={**options, "instrumentation_mode": "consan"})
    assert "amdg.cond_barrier" not in compiled.asm["ttgir"]
    _assert_instrumentation(compiled, True)


_MARKER_MODULE = """
module {{
  tt.func @kernel(%n: i32) {{
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %c0 to %n step %c1 : i32 {{
      %x = arith.addi %i, %c1 {op_attrs} : i32
      scf.execute_region {{
        scf.yield
      }} {region_attrs}
    }} {loop_attrs}
    tt.return
  }}
}}
"""


@pytest.mark.parametrize(
    "op_attrs,region_attrs,loop_attrs,expected",
    [
        ("", "", "", False),
        # Marker text in an unrelated attribute value is not a marker.
        ('{note = "triton.warp_pipeline.stage"}', "", "", False),
        # A border that the warp pipeliner did not turn into stages is not converted.
        ('{triton.warp_pipeline.border = "load"}', "", "", False),
        # A loop marker without stage regions is not a pipeline.
        ("", "", "{triton.warp_pipeline.pipelined_for}", False),
        # Stage region created by the warp pipeliner.
        ("", '{triton.warp_pipeline.stage = "load"}', "", True),
        ("", '{triton.warp_pipeline.stage = "load"}', "{triton.warp_pipeline.pipelined_for}", True),
    ],
)
def test_has_warp_pipeline_markers(tmp_path, op_attrs, region_attrs, loop_attrs, expected):
    context = ir.context()
    ir.load_dialects(context)
    amd.load_dialects(context)
    path = tmp_path / "markers.ttgir"
    path.write_text(_MARKER_MODULE.format(op_attrs=op_attrs, region_attrs=region_attrs, loop_attrs=loop_attrs))
    module = ir.parse_mlir_module(str(path), context)
    assert amd.has_warp_pipeline(module) == expected
