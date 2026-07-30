import sys
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace
import pytest
import torch

import triton
import triton.language as tl
from triton._compile_warmup import (
    CompilationTrace,
    _WarmupComplete,
    _warmup_test_case,
    compile_warmup_only,
    summarize_compile_trace,
)
from triton.backends.nvidia.compiler import CUDABackend
from triton.backends.driver import GPUDriver, expand_signature, wrap_handle_tensordesc_impl


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


def test_compilation_trace_matches_warmup_cache_hits(tmp_path):
    times = SimpleNamespace(total=125_000)
    source = SimpleNamespace(name="kernel", fn=SimpleNamespace(_fn_name="package.kernel"), hash=lambda: "source")
    warmup = CompilationTrace(str(tmp_path), "warmup-triton-kernels")
    runtime = CompilationTrace(str(tmp_path), "triton-kernels")

    warmup(src=source, metadata={"hash": "warmed"}, metadata_group={}, times=times, cache_hit=False)
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
                "compile_seconds": 0.125,
            },
        },
        "warmup_hashes": 1,
    }


def test_compile_warmup_skips_triton_kernels_reference(monkeypatch):
    package = ModuleType("triton_kernels")
    testing = ModuleType("triton_kernels.testing")
    testing.alloc_rand = lambda *args, **kwargs: None
    testing.make_slice_sizes = lambda *args, **kwargs: None
    package.testing = testing
    monkeypatch.setitem(sys.modules, "triton_kernels", package)
    monkeypatch.setitem(sys.modules, "triton_kernels.testing", testing)

    reference = object()
    module = SimpleNamespace(__file__="/checkout/python/triton_kernels/tests/test_matmul.py", matmul_torch=reference)
    item = SimpleNamespace(module=module, originalname="test_op", callspec=SimpleNamespace(params={}))
    previous_alloc_rand = testing.alloc_rand
    previous_make_slice_sizes = testing.make_slice_sizes

    with compile_warmup_only(), _warmup_test_case(item):
        allocation = testing.alloc_rand((4, 8), device="cpu", dtype=torch.float32)
        slice_sizes = testing.make_slice_sizes(3, 12, device="cpu")

        assert allocation.shape == (4, 8)
        assert slice_sizes.shape == (3, )
        with pytest.raises(_WarmupComplete):
            module.matmul_torch()

    assert module.matmul_torch is reference
    assert testing.alloc_rand is previous_alloc_rand
    assert testing.make_slice_sizes is previous_make_slice_sizes


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
