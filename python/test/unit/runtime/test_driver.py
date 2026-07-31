import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace
import pytest
import torch
import cloudpickle

import triton
import triton.language as tl
from triton._compile_warmup import (
    CompilationTrace,
    _cache_phase_for_item,
    _main,
    _require_complete_warmup,
    _warmup_test_case,
    compile_warmup_only,
    pytest_collection_modifyitems,
    summarize_compile_trace,
)
from triton._compile_warmup_pool import ProcessPoolWarmupDispatcher, _jit_callable_import_path, _jit_dumps
from triton.backends.nvidia.compiler import CUDABackend
from triton.backends.driver import GPUDriver, expand_signature, wrap_handle_tensordesc_impl


@triton.jit
def _compile_warmup_importable_kernel(output):
    tl.store(output, 1)


def test_compile_warmup_only_intercepts_launches():
    launches = []

    class Kernel(triton.KernelInterface):

        def warmup(self, *args, grid, **kwargs):
            launches.append((args, grid, kwargs))
            return "compiled"

    kernel = Kernel()
    previous_getitem = triton.KernelInterface.__getitem__
    previous_assert_close = torch.testing.assert_close

    with compile_warmup_only():
        tensor = torch.empty(16, device="cuda")
        result = kernel[(2, )](tensor, BLOCK_SIZE=16)
        torch.testing.assert_close(tensor, tensor)

    assert result == "compiled"
    assert type(tensor).__name__ == "FakeTensor"
    assert launches == [((tensor, ), (2, ), {"BLOCK_SIZE": 16})]
    assert triton.KernelInterface.__getitem__ is previous_getitem
    assert torch.testing.assert_close is previous_assert_close


def test_compile_warmup_preserves_tensor_view_alignment():
    with compile_warmup_only():
        tensor = torch.empty(16, device="cpu", dtype=torch.float32)
        view = tensor[1:]

        assert tensor.data_ptr() % 256 == 0
        assert view.data_ptr() == tensor.data_ptr() + tensor.element_size()
        assert CUDABackend.get_tensor_specialization(tensor, align=True) == "D"
        assert CUDABackend.get_tensor_specialization(view, align=True) == ""


def test_compile_warmup_process_pool_requires_workers():
    with pytest.raises(ValueError, match="max_workers must be >= 1"):
        ProcessPoolWarmupDispatcher(max_workers=0, trace_directory=None, phase="warmup-test")


def test_compile_warmup_attributes_combined_session_phases(monkeypatch):
    monkeypatch.setenv("TRITON_CI_CACHE_PHASE", "warmup-default")
    config = SimpleNamespace(
        rootpath=Path("/checkout"),
        getoption=lambda _: [
            "python/tutorials/06-fused-attention.py=warmup-attention",
            "python/examples/gluon=warmup-gluon-examples",
        ],
    )

    attention = SimpleNamespace(config=config, path=Path("/checkout/python/tutorials/06-fused-attention.py"))
    example = SimpleNamespace(config=config, path=Path("/checkout/python/examples/gluon/01-attention-forward.py"))
    unit = SimpleNamespace(config=config, path=Path("/checkout/python/test/unit/language/test_core.py"))

    assert _cache_phase_for_item(attention) == "warmup-attention"
    assert _cache_phase_for_item(example) == "warmup-gluon-examples"
    assert _cache_phase_for_item(unit) == "warmup-default"


def test_compile_warmup_serializes_local_jit_function():

    @triton.jit
    def kernel(output, value: tl.constexpr):
        tl.store(output, value)

    restored = cloudpickle.loads(_jit_dumps(kernel))

    assert restored.src == kernel.src
    assert restored.__name__ == kernel.__name__


def test_compile_warmup_serializes_patched_source_globals():

    @triton.jit
    def helper(value):
        return value + 1

    @triton.jit
    def kernel(output, value: tl.constexpr):
        tl.store(output, value)

    global_name = "_compile_warmup_serialization_helper"
    kernel.fn.__globals__[global_name] = helper
    kernel._unsafe_update_src(kernel.src.replace("tl.store(output, value)", f"tl.store(output, {global_name}(value))"))
    try:
        restored = cloudpickle.loads(_jit_dumps(kernel))
        assert restored.cache_key == kernel.cache_key
    finally:
        del kernel.fn.__globals__[global_name]


def test_compile_warmup_serializes_wrapped_module_jit_function(monkeypatch):
    function = _compile_warmup_importable_kernel
    monkeypatch.setattr(sys.modules[__name__], function.__name__, SimpleNamespace(fn=function))

    assert _jit_callable_import_path(function) is None
    restored = cloudpickle.loads(_jit_dumps(function))
    assert restored.cache_key == function.cache_key


def test_compile_warmup_serializes_generated_module_jit_function(monkeypatch):
    function = _compile_warmup_importable_kernel
    monkeypatch.setattr(sys.modules[__name__], "__file__", "/missing/generated_test_driver.py")

    assert _jit_callable_import_path(function) is None
    restored = cloudpickle.loads(_jit_dumps(function))
    assert restored.cache_key == function.cache_key


def test_compilation_trace_matches_warmup_cache_hits(tmp_path):
    times = SimpleNamespace(total=125_000)
    source = SimpleNamespace(name="kernel", fn=SimpleNamespace(_fn_name="package.kernel"), hash=lambda: "source")
    warmup = CompilationTrace(str(tmp_path), "warmup-triton-kernels", "test_matmul.py::test_op[0]")
    unrelated_warmup = CompilationTrace(str(tmp_path), "warmup-gluon", "test_core.py::test_mma[0]")
    runtime = CompilationTrace(str(tmp_path), "triton-kernels", "test_matmul.py::test_op[0]")

    warmup(src=source, metadata={"hash": "warmed"}, metadata_group={}, times=times, cache_hit=False)
    unrelated_warmup(src=source, metadata={"hash": "gluon"}, metadata_group={}, times=times, cache_hit=False)
    runtime(src=source, metadata={"hash": "warmed"}, metadata_group={}, times=times, cache_hit=True)
    runtime(src=source, metadata={"hash": "different"}, metadata_group={}, times=times, cache_hit=False)

    report = summarize_compile_trace(str(tmp_path), "triton-kernels")

    assert report == {
        "phases": {
            "triton-kernels": {
                "events": 2,
                "disk_hits": 1,
                "disk_misses": 1,
                "warmed_hits": 1,
                "warmed_misses": 0,
                "warmed_test_events": 2,
                "warmed_test_hits": 1,
                "warmed_test_misses": 1,
                "incomplete_warmed_test_count": 1,
                "incomplete_warmed_tests": ["test_matmul.py::test_op[0]"],
                "unused_warmup_hashes": 0,
                "compile_seconds": 0.125,
            },
        },
        "warmup_hashes": 2,
        "warmup_tests": 2,
    }
    with pytest.raises(SystemExit, match="not complete cache hits"):
        _require_complete_warmup(report)


def test_compilation_trace_grades_attempted_warmup_tests(tmp_path):
    attempted = tmp_path / "warmup-unit-1.tests"
    attempted.write_text('{"phase": "warmup-unit", "test": "test_core.py::test_missing"}\n')
    times = SimpleNamespace(total=125_000)
    source = SimpleNamespace(name="kernel", fn=SimpleNamespace(_fn_name="package.kernel"), hash=lambda: "source")
    runtime = CompilationTrace(str(tmp_path), "unit", "test_core.py::test_missing")
    runtime(src=source, metadata={"hash": "cold"}, metadata_group={}, times=times, cache_hit=False)

    report = summarize_compile_trace(str(tmp_path), "unit")

    assert report["warmup_tests"] == 1
    assert report["phases"]["unit"]["warmed_test_misses"] == 1
    assert report["phases"]["unit"]["incomplete_warmed_tests"] == ["test_core.py::test_missing"]


def test_compilation_trace_reports_multiple_phases(tmp_path, monkeypatch, capsys):
    times = SimpleNamespace(total=125_000)
    source = SimpleNamespace(name="kernel", fn=SimpleNamespace(_fn_name="package.kernel"), hash=lambda: "source")
    for phase in ("unit", "attention"):
        listener = CompilationTrace(str(tmp_path), phase, f"{phase}::test")
        listener(src=source, metadata={"hash": phase}, metadata_group={}, times=times, cache_hit=True)
    monkeypatch.setattr(sys, "argv",
                        ["warmup", "report", "--directory",
                         str(tmp_path), "--phase", "unit", "--phase", "attention"])

    _main()

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert '"phases": {"unit":' in lines[0]
    assert '"phases": {"attention":' in lines[1]


def test_compile_warmup_skips_unsupported_fake_tensor_specializations():
    items = []
    for module_path, originalname in [
        ("/checkout/python/test/unit/language/test_core.py", "test_atomic_cas"),
        ("/checkout/python/test/unit/language/test_matmul.py", "test_block_scale_fp4"),
        ("/checkout/python/test/unit/language/test_standard.py", "test_maximum_minium"),
        ("/checkout/python/test/unit/language/test_core.py", "test_dot"),
    ]:
        item = SimpleNamespace(
            module=SimpleNamespace(__file__=module_path),
            originalname=originalname,
            markers=[],
        )
        item.add_marker = item.markers.append
        items.append(item)

    pytest_collection_modifyitems(SimpleNamespace(getoption=lambda _: True), items)

    assert [bool(item.markers) for item in items] == [True, False, True, False]


def test_compile_warmup_skips_zero_sized_inner_experts():
    items = []
    for m, inner_expt_opt in [(0, "pad_a"), (4, "pad_a"), (0, None)]:
        item = SimpleNamespace(
            module=SimpleNamespace(__file__="/checkout/python/triton_kernels/tests/test_matmul.py"),
            originalname="test_op",
            callspec=SimpleNamespace(params={"m": m, "n": 8, "k": 16, "inner_expt_opt": inner_expt_opt}),
            markers=[],
        )
        item.add_marker = item.markers.append
        items.append(item)

    pytest_collection_modifyitems(SimpleNamespace(getoption=lambda _: True), items)

    assert [bool(item.markers) for item in items] == [True, False, False]


def test_compile_warmup_replaces_preshuffled_mxfp_conversion():
    from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor

    item = SimpleNamespace(
        module=SimpleNamespace(
            __file__="/checkout/python/test/unit/language/test_matmul.py",
            MXFP4Tensor=MXFP4Tensor,
            MXScaleTensor=MXScaleTensor,
        ),
        originalname="test_preshuffle_scale_mxfp_cdna4",
    )
    previous_conversion = MXFP4Tensor.to

    with compile_warmup_only(), _warmup_test_case(item):
        value = MXFP4Tensor(size=(8, ), device="cuda").random().to(torch.float32)

        assert type(value).__name__ == "FakeTensor"
        assert value.shape == (8, )

    assert MXFP4Tensor.to is previous_conversion


def test_compile_warmup_replaces_warp_specialization_cublas():
    previous_cublas = object()
    module = SimpleNamespace(
        __file__="/checkout/python/test/unit/language/test_warp_specialization.py",
        cublas=previous_cublas,
    )
    item = SimpleNamespace(module=module, originalname="test_warp_specialize_tma_matmul")

    with _warmup_test_case(item):
        assert module.cublas is not previous_cublas
        assert module.cublas.matmul(object(), object(), object()) is None

    assert module.cublas is previous_cublas


def test_compile_warmup_replaces_scaled_dot_finite_check():
    module = SimpleNamespace(__file__="/checkout/python/test/unit/language/test_core.py")
    item = SimpleNamespace(module=module, originalname="test_scaled_dot")
    previous_isfinite = torch.Tensor.isfinite

    with compile_warmup_only(), _warmup_test_case(item):
        assert torch.empty(4, device="cuda").isfinite().all()

    assert torch.Tensor.isfinite is previous_isfinite


@pytest.mark.parametrize("warp_counts", [(2, 4), (2, 2, 2)])
def test_compile_warmup_materializes_gluon_reduce_warp_count(warp_counts):
    layouts = SimpleNamespace(warps_per_cta=lambda layout, shape: warp_counts)
    module = SimpleNamespace(
        __file__="/checkout/python/test/gluon/test_lowerings.py",
        ttgl=SimpleNamespace(_layouts=layouts),
    )
    item = SimpleNamespace(
        module=module,
        originalname="test_reduce_layouts",
        callspec=SimpleNamespace(params={"M": 64, "N": 128, "src_layout": object()}),
    )
    previous_prod = torch.prod

    with compile_warmup_only(), _warmup_test_case(item):
        assert torch.prod(torch.empty(len(warp_counts), device="cpu")) == 8

    assert torch.prod is previous_prod


def test_compile_warmup_replaces_gluon_moe_fake_checks(monkeypatch):
    package = ModuleType("triton_kernels")
    package.__path__ = []
    tensor_details = ModuleType("triton_kernels.tensor_details")
    tensor_details.__path__ = []
    layout_details = ModuleType("triton_kernels.tensor_details.layout_details")
    layout_details.__path__ = []
    blackwell_scale = ModuleType("triton_kernels.tensor_details.layout_details.blackwell_scale")
    blackwell_scale.is_fake = lambda tensor: True
    monkeypatch.setitem(sys.modules, "triton_kernels", package)
    monkeypatch.setitem(sys.modules, "triton_kernels.tensor_details", tensor_details)
    monkeypatch.setitem(sys.modules, "triton_kernels.tensor_details.layout_details", layout_details)
    monkeypatch.setitem(sys.modules, "triton_kernels.tensor_details.layout_details.blackwell_scale", blackwell_scale)

    module_assert_close = object()
    module = SimpleNamespace(
        __file__="/checkout/python/examples/gluon/05-moe-bmm1-fused-gather.py",
        assert_close=module_assert_close,
    )
    item = SimpleNamespace(module=module, originalname="test_op")
    previous_is_fake = blackwell_scale.is_fake

    with _warmup_test_case(item):
        assert module.assert_close is not module_assert_close
        assert not blackwell_scale.is_fake(object())

    assert module.assert_close is module_assert_close
    assert blackwell_scale.is_fake is previous_is_fake


def test_compile_warmup_replaces_triton_kernels_reference(monkeypatch):
    package = ModuleType("triton_kernels")
    package.__path__ = []
    testing = ModuleType("triton_kernels.testing")
    tensor_details = ModuleType("triton_kernels.tensor_details")
    tensor_details.__path__ = []
    layout_details = ModuleType("triton_kernels.tensor_details.layout_details")
    layout_details.__path__ = []
    blackwell_scale = ModuleType("triton_kernels.tensor_details.layout_details.blackwell_scale")
    blackwell_scale.is_fake = lambda tensor: True
    testing.alloc_rand = lambda *args, **kwargs: None
    testing.assert_close = lambda *args, **kwargs: None
    testing.make_slice_sizes = lambda *args, **kwargs: None
    testing._make_slice_sizes_cpu = lambda n_slices, total_size: torch.zeros(n_slices, dtype=torch.int32)
    testing.pad_ragged_tensor = lambda *args, **kwargs: None
    package.testing = testing
    monkeypatch.setitem(sys.modules, "triton_kernels", package)
    monkeypatch.setitem(sys.modules, "triton_kernels.testing", testing)
    monkeypatch.setitem(sys.modules, "triton_kernels.tensor_details", tensor_details)
    monkeypatch.setitem(sys.modules, "triton_kernels.tensor_details.layout_details", layout_details)
    monkeypatch.setitem(sys.modules, "triton_kernels.tensor_details.layout_details.blackwell_scale", blackwell_scale)

    def reference(*args, **kwargs):
        return torch.empty((4, 8), dtype=torch.float32)

    module_assert_close = object()
    module = SimpleNamespace(
        __file__="/checkout/python/triton_kernels/tests/test_matmul.py",
        assert_close=module_assert_close,
        matmul_torch=reference,
    )
    item = SimpleNamespace(
        module=module,
        originalname="test_op",
        callspec=SimpleNamespace(
            params={
                "m": 4,
                "n": 8,
                "n_slices": 1,
                "mode": "plain",
                "inner_expt_opt": None,
                "output_dtype_str": None,
                "act_dtype_str": "float32",
                "swiglu_opts": None,
            }),
    )
    module.DType = lambda _: SimpleNamespace(is_nvfp4=False, has_mx_scale=False, torch_dtype=torch.float32)
    previous_alloc_rand = testing.alloc_rand
    previous_assert_close = testing.assert_close
    previous_make_slice_sizes = testing.make_slice_sizes
    previous_pad_ragged_tensor = testing.pad_ragged_tensor
    previous_is_fake = blackwell_scale.is_fake

    with compile_warmup_only(), _warmup_test_case(item):
        allocation = testing.alloc_rand((4, 8), device="cpu", dtype=torch.float32)
        slice_sizes = testing.make_slice_sizes(3, 12, device="cpu")

        assert allocation.shape == (4, 8)
        assert slice_sizes.shape == (3, )
        reference_output = module.matmul_torch()
        assert reference_output.shape == (4, 8)
        assert reference_output.dtype == torch.float32

    assert module.matmul_torch is reference
    assert module.assert_close is module_assert_close
    assert testing.alloc_rand is previous_alloc_rand
    assert testing.assert_close is previous_assert_close
    assert testing.make_slice_sizes is previous_make_slice_sizes
    assert testing.pad_ragged_tensor is previous_pad_ragged_tensor
    assert blackwell_scale.is_fake is previous_is_fake


def test_is_lazy():
    from importlib import reload
    reload(sys.modules["triton.runtime.driver"])
    reload(sys.modules["triton.runtime"])
    assert triton.runtime.driver._active is None
    assert triton.runtime.driver._default is None
    assert isinstance(triton.runtime.driver.active, getattr(triton.backends.driver, "DriverBase"))
    assert isinstance(triton.runtime.driver.default, getattr(triton.backends.driver, "DriverBase"))
    utils = triton.runtime.driver.active.utils  # noqa: F841


def test_profile_scratch_stream_zero_uses_default_stream(monkeypatch):

    class Scratch:

        def __init__(self):
            self.recorded_streams = []

        def record_stream(self, stream):
            self.recorded_streams.append(stream)

    class DeviceInterface:

        def __init__(self):
            self.default_stream_arg = None
            self.stream_args = []

        def ExternalStream(self, stream, device):
            raise AssertionError("stream 0 must use the default stream")

        def stream(self, stream):
            self.stream_args.append(stream)
            return stream

        def default_stream(self, device):
            self.default_stream_arg = device
            return self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    zeros_calls = []

    def zeros(size, dtype, device):
        zeros_calls.append((size, dtype, device))
        return Scratch()

    device_interface = DeviceInterface()
    driver = SimpleNamespace(get_active_torch_device=lambda: "cuda:0", get_device_interface=lambda: device_interface)
    monkeypatch.setattr(torch, "zeros", zeros)

    scratch = GPUDriver.allocate_default_profile_scratch(driver, 16, 8, 0)

    assert device_interface.default_stream_arg == "cuda:0"
    assert device_interface.stream_args == [device_interface]
    assert zeros_calls == [(16, torch.int8, "cuda:0")]
    assert scratch.recorded_streams == []


def test_kernel_in_thread(device):
    # Test calling in a new thread sets a valid device context
    buf = torch.zeros((38016 * 1024, ), dtype=torch.float32, device=device)

    @triton.jit
    def _kernel(P, BLOCK: tl.constexpr):
        pid = tl.program_id(0).to(tl.int64)
        offset = pid * BLOCK + tl.arange(0, BLOCK)

        p = tl.load(P + offset)
        tl.store(P + offset, p)

    def call_triton():
        N = buf.numel()
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]), )
        _kernel[grid](buf, BLOCK=1024)
        getattr(torch, device).synchronize()

    call_triton()
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(call_triton)
        future.result()


def test_expand_signature_with_aggregate_tensordesc():
    signature = (
        "i32",
        ("tensordesc<fp16[16,32]>", "i64"),
        "tensordesc_im2col<fp32[1,16],input_rank=4,'layout'>",
    )
    expanded = expand_signature(signature, [], "nvTmaDesc")

    assert expanded[0] == "i32"
    assert expanded[1] == ("*fp16", *["i64"] * 4, *["i1"] * 2, *["i32"] * 2, *["i64"] * 3)
    # input_rank=4 drives the number of shape/stride entries for im2col.
    assert expanded[2:] == ["*fp32", *["i64"] * 8, *["i1"] * 2, *["i32"] * 4, *["i64"] * 4]

    expanded = expand_signature(signature, [{}, {}], "nvTmaDesc")
    assert expanded[0] == "i32"
    assert expanded[1] == ("nvTmaDesc", *["i32"] * 2, *["i64"] * 3)
    assert expanded[2:] == ["nvTmaDesc", *["i32"] * 4, *["i64"] * 4]


def test_wrap_tensordesc_handles_aggregate_arguments():
    signature = {0: ("tensordesc<fp16[16,16]>", "i32"), 1: "i64", 2: "tensordesc<fp16[16,16]>"}
    outer_meta = {"tag": "outer"}
    launcher_calls = []

    def launcher(*args):
        launcher_calls.append(args)
        return "ok"

    def make_descriptor(arg, meta, base_args):
        return [("desc", arg, meta, base_args[0]), ("shape", arg)]

    wrapped = wrap_handle_tensordesc_impl(launcher, signature, [None, outer_meta], make_descriptor)
    assert wrapped("meta0", "meta1", (("A", 7), 9, "B")) == "ok"

    assert len(launcher_calls) == 1
    assert launcher_calls[0][0] == "meta0"
    assert launcher_calls[0][1] == "meta1"
    assert launcher_calls[0][2] == [
        (("desc", "A", None, "meta0"), ("shape", "A"), 7),
        9,
        ("desc", "B", outer_meta, "meta0"),
        ("shape", "B"),
    ]


def test_wrap_tensordesc_is_noop_without_tensordesc():

    def launcher(*args):
        return args

    wrapped = wrap_handle_tensordesc_impl(launcher, {0: "i32", 1: ("i64", "constexpr")}, None, lambda *_: [])
    assert wrapped is launcher
