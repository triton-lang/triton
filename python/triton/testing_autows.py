import torch
import pytest
import triton.language as tl
import triton.runtime.driver
from triton.runtime import JITFunction
from triton._C.libtriton import nvidia
import triton.profiler as proton
import triton.profiler.viewer as proton_viewer
from contextlib import contextmanager


def is_sm9x():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


def is_sm10x():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


def common_test_setup(ENABLE_WARP_SPECIALIZATION=None, NUM_WARPS=None):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not is_sm9x() and not is_sm10x():
        pytest.skip("SM9x or SM10x required")
    if (
        is_sm10x()
        and ENABLE_WARP_SPECIALIZATION
        and NUM_WARPS is not None
        and NUM_WARPS != 4
    ):
        # FIXME: See https://jirasw.nvidia.com/browse/OT-90
        pytest.skip("AutoWS only supports 4 warps on Blackwell")
    torch.manual_seed(42)


def get_num_sms():
    return torch.cuda.get_device_properties("cuda").multi_processor_count


def get_shared_memory():
    device = triton.runtime.driver.active.get_active_torch_device()
    properties = triton.runtime.driver.active.utils.get_device_properties(device.index)
    return properties["max_shared_mem"]


def compute_shared_memory(buffers):
    def align(x, last):
        if x % 16 == 0 or last:
            return x
        return x + 16 - x % 16

    return sum(align(x, last=idx == len(buffers) - 1) for idx, x in enumerate(buffers))


def init_check_shared_memory_hook(kernel, expected_shared_memory):
    # adds a hook to triton compilation flow, to check that the actual shared memory used
    # by a compiled kernel matches our computed value

    def hook(*args, **kwargs):
        key = kwargs["compile"]["key"]
        device = kwargs["compile"]["device"]
        kernel_cache = kwargs["fn"].jit_function.device_caches[device][0]
        kernel = kernel_cache[key]
        # print(f"expected {expected_shared_memory}")
        # print(f"actual {kernel.metadata.shared}")
        assert expected_shared_memory == kernel.metadata.shared

    JITFunction.compiled_hook = hook


def clear_check_shared_memory_hook():
    JITFunction.compiled_hook = None


def torch_dtype(DTYPE):
    return torch.float8_e4m3fn if DTYPE == "fp8" else torch.float16


def dtype_size(dtype):
    if dtype == torch.float8_e4m3fn:
        return 1
    if dtype == torch.float16:
        return 2
    raise NotImplementedError


def generate_input(shape, dtype):
    if dtype == torch.float8_e4m3fn:
        return (
            torch.empty(shape, dtype=torch.float16, device="cuda")
            .normal_(mean=0, std=1)
            .to(dtype)
        )
    return torch.randn(shape, dtype=dtype, device="cuda")


def verify_matmul(A, B, C):
    cublas_workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
    C_ref = torch.empty(C.shape, dtype=C.dtype, device="cuda")
    cublas.matmul(A, B, C_ref)
    ERROR_TOLERANCE = 1e-3 if C.dtype == torch.float16 else 1e-2
    torch.testing.assert_close(
        C_ref.to(torch.float16),
        C.to(torch.float16),
        rtol=ERROR_TOLERANCE,
        atol=ERROR_TOLERANCE,
    )


HAS_TMA_DESC = "nv_tma_desc_type" in dir(tl)


class TmaAutoTuneHelper:

    # duck typing wrapper to implement the same interface as TmaDescKernelParam in Triton PR #4498
    class KernelParamWrapper:
        def __init__(self, desc):
            self.desc = desc

        def tma_desc_cpu_ptr(self):
            return self.desc.data_ptr()

    TMA_SIZE = 128

    def __init__(self):
        self.fill_1d_tma_descriptor_inner = (
            triton.runtime.driver.active.utils.fill_1d_tma_descriptor
        )
        self.fill_2d_tma_descriptor_inner = (
            triton.runtime.driver.active.utils.fill_2d_tma_descriptor
        )
        if HAS_TMA_DESC:
            self.descriptors = {}
        else:
            self.cuda_descriptors = {}

    # Call this method outside of the lambda function for grid size
    def init_tma_descriptor(self, name):
        if HAS_TMA_DESC:
            self.descriptors[name] = torch.empty(
                TmaAutoTuneHelper.TMA_SIZE, device="cpu", dtype=torch.int8
            )
        else:
            self.cuda_descriptors[name] = torch.empty(
                TmaAutoTuneHelper.TMA_SIZE, device="cuda", dtype=torch.int8
            )

    # Call this method inside the lambda function for grid size
    def fill_1d_tma_descriptor(self, name, ptr, dim, block_dim, element_size):
        if HAS_TMA_DESC:
            desc_x = self.descriptors[name]
            assert desc_x.data_ptr() % 64 == 0
            self.fill_1d_tma_descriptor_inner(
                ptr, dim, block_dim, element_size, desc_x.data_ptr()
            )
        else:
            desc_x = self.cuda_descriptors[name]
            buf_x = torch.empty_like(desc_x, device="cpu", pin_memory=True)
            self.fill_1d_tma_descriptor_inner(
                ptr, dim, block_dim, element_size, buf_x.data_ptr()
            )
            desc_x.copy_(buf_x, non_blocking=True)

    # Call this method inside the lambda function for grid size
    def fill_2d_tma_descriptor(
        self, name, ptr, dim1, dim0, block_dim1, block_dim0, element_size
    ):
        if HAS_TMA_DESC:
            desc_x = self.descriptors[name]
            assert desc_x.data_ptr() % 64 == 0
            self.fill_2d_tma_descriptor_inner(
                ptr, dim1, dim0, block_dim1, block_dim0, element_size, desc_x.data_ptr()
            )
        else:
            desc_x = self.cuda_descriptors[name]
            buf_x = torch.empty_like(desc_x, device="cpu", pin_memory=True)
            self.fill_2d_tma_descriptor_inner(
                ptr, dim1, dim0, block_dim1, block_dim0, element_size, buf_x.data_ptr()
            )
            desc_x.copy_(buf_x, non_blocking=True)

    def get_tma_descriptor_kernel_param(self, name):
        if HAS_TMA_DESC:
            assert self.descriptors[name] is not None
            return self.KernelParamWrapper(self.descriptors[name])
        else:
            assert self.cuda_descriptors[name] is not None
            return self.cuda_descriptors[name]


@contextmanager
def proton_context():
    proton.activate()
    try:
        yield
    finally:
        proton.deactivate()


def bench_fn(cmd_args, fn, *args):
    for _ in range(cmd_args.warmup):
        fn(*args)
    with proton_context():
        for _ in range(cmd_args.iters):
            fn(*args)


def show_profile(precision, profile_name):
    metric_names = ["time/ms"]
    if precision == "fp8":
        metric_names = ["tflop8/s"] + metric_names
    elif precision == "fp16":
        metric_names = ["tflop16/s"] + metric_names
    file_name = f"{profile_name}.hatchet"
    tree, metrics = proton_viewer.parse(metric_names, file_name)
    proton_viewer.print_tree(tree, metrics)


def run_bench(name, args, run):
    proton.start(name, hook="triton")
    run(True)
    proton.finalize()
    show_profile(args.prec, name)


def common_bench_options(parser):
    parser.add_argument("--auto-ws", action="store_true")
    parser.add_argument("--ttg-ws", action="store_true")
    parser.add_argument("--prec", type=str, choices=["fp8", "fp16"], default="fp16")
    parser.add_argument("--wg-spec", type=str, choices=["tma_load_first", "mma_first", None], default=None)
    parser.add_argument("--NUM_WARPS", type=int, default=4)
    parser.add_argument("--NUM_STAGES", type=int, default=3)

    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--iters", type=int, default=1000)
