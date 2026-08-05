# isort: off
# fmt: off
from dataclasses import dataclass, fields
import itertools
import pytest
import torch
from typing import Union
import triton
from triton._internal_testing import is_compile_warmup
# matmul utilities
import triton_kernels.matmul_details.opt_flags as opt_flags
from triton_kernels.matmul import FlexCtx, PrecisionConfig, FusedActivation, FnSpecs, FnName, Epilogue
from triton_kernels.matmul import apply_precision, matmul_set_idle_sms, matmul, matmul_torch
# numerics utilities
from triton_kernels.numerics import InFlexData, OutFlexData
from triton_kernels.numerics_details.mxfp import upcast_from_mxfp, quantize_mxfp8_fn, quantize_nvfp4_fn, downcast_to_mxfp_torch, upcast_from_mxfp_torch, MXFP_BLOCK_SIZE, NVFP_BLOCK_SIZE
# testing utilities
from triton_kernels.testing import _make_random_block_signs, assert_close, make_random_tensor
# target-specific utilities
from triton_kernels.target_info import is_cuda, is_hip, is_hip_cdna3, is_hip_cdna4, is_hip_gfx1250
from triton_kernels.swiglu import swiglu, swiglu_fn
from triton_kernels.swiglu import PrecisionConfig as SwiGLUPrecisionConfig
from triton_kernels.tensor_details import layout
from triton_kernels.tensor import Tensor, convert_layout, make_ragged_tensor_metadata, wrap_torch_tensor
from triton_kernels.tensor_details.dtype import FP32

# ---------------
# numerics stuff
# ---------------

class DType:

    def __init__(self, dtype_str):
        self.name = dtype_str
        # "Fiber" scales are also known as row scales. The suffix is test-only;
        # plain nvfp4_e2m1 leaves the tensor scale unset.
        self.has_tensor_scale = dtype_str.endswith("_fiber")
        # This tracks the regular fp8 flex scale path. NVFP4 has a tensor scale,
        # but it is handled separately because it also has MX microscale storage.
        self.has_global_scale = dtype_str.startswith("float8")
        self.is_nvfp4 = dtype_str in {"nvfp4_e2m1", "nvfp4_e2m1_fiber"}
        self.has_mx_scale = dtype_str.startswith("mx") or self.is_nvfp4
        self.is_any_float8 = "float8" in dtype_str
        self.uses_fp8e4nv = dtype_str in {"mxfloat8_e4m3fn", "nvfp4_e2m1", "nvfp4_e2m1_fiber"}
        if dtype_str in {"float4_e2m1", "mxfloat4_e2m1", "nvfp4_e2m1", "nvfp4_e2m1_fiber"}:
            self.torch_dtype = torch.uint8
        else:
            self.torch_dtype = getattr(torch, dtype_str.strip("mx"))
        self.is_mxfloat4 = self.has_mx_scale and ("float4" in dtype_str or self.is_nvfp4)
        self.scale_dtype = torch.float8_e4m3fn if self.is_nvfp4 else torch.uint8 if self.has_mx_scale else None
        self.microblock_size = NVFP_BLOCK_SIZE.value if self.is_nvfp4 else MXFP_BLOCK_SIZE.value if self.has_mx_scale else None


@pytest.fixture
def opt_flags_scope():
    opt_flags.reset_opt_flags_constraints()
    yield
    opt_flags.reset_opt_flags_constraints()


def make_constraints(block_m, split_k, is_persistent, epilogue_subtile, hbm_swizzling, weight_dtype_str, num_warps):
    constraints = {
        "block_m": block_m,
        "split_k": split_k,
        "is_persistent": is_persistent,
        "epilogue_subtile": epilogue_subtile,
        "num_warps": num_warps,
    }
    if is_hip() and hbm_swizzling and "float4" in weight_dtype_str:
        # Minimum block size to satisfy scale preshuffling
        if is_hip_gfx1250():
            constraints.update({
                "block_m": 128,
                "block_n": 128,
                "block_k": 128
            })
        else:
            constraints.update({
                "block_m": 32,
                "block_n": 32,
                "block_k": 256
            })
    return constraints

# ---------------
# unit tests
# ---------------


@dataclass
class Case:
    m: int
    n: int
    k: int
    mode: str
    act_dtype_str: str
    weight_dtype_str: str
    output_dtype_str: Union[str, None] = None
    n_slices: int = None
    split_k: int = 1
    a_hbm_swizzling: bool = False
    b_hbm_swizzling: bool = False
    c_hbm_swizzling: bool = False
    shuffle_mxfp4_w_layout: bool = False
    epilogue_subtile: Union[int, None] = None
    a_transpose: bool = False
    b_transpose: bool = False
    c_transpose: bool = False
    colmajor_mxfp_weight: bool = True
    swiglu_opts: tuple[float, float] = None

    def __post_init__(self):
        if self.n_slices is None:
            self.n_slices = 1 if self.mode == "plain" else 10


@pytest.mark.skipif(not is_cuda(), reason="CUDA uses the Philox random-number generator")
@pytest.mark.parametrize("shape", [(33, 47), (3, 65, 96)])
def test_random_block_signs_preserve_cuda_rng(shape, device):
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(17)
        torch.rand(7, device=device)
        generator = torch.cuda.default_generators[torch.cuda.current_device()]
        initial_state = generator.get_state()

        batch_size, rows, cols = (1, *shape) if len(shape) == 2 else shape
        block_size = int(MXFP_BLOCK_SIZE)
        expected = []
        for _, row, col in itertools.product(range(batch_size), range(0, rows, block_size),
                                              range(0, cols, block_size)):
            count = max(min(block_size, rows - row), min(block_size, cols - col))
            values = torch.randint(0, 2, (count,), device=device)
            expected.append(torch.nn.functional.pad(values, (0, block_size - count)))
        final_state = generator.get_state()

        generator.set_state(initial_state)
        actual = _make_random_block_signs(torch.empty(shape, device=device), batch_size, rows, cols,
                                          triton.cdiv(rows, block_size), triton.cdiv(cols, block_size), block_size)
        torch.testing.assert_close(actual.reshape(-1, block_size), torch.stack(expected))
        torch.testing.assert_close(generator.get_state(), final_state)


def _build_test_op_cases():
    test_cases = []
    # zero-sized
    zero_sized_shapes = ((0, 5, 7), (5, 0, 7), (5, 7, 0))
    # split_k=1 preserves existing constrained coverage; None exercises automatic split-K selection.
    for split_k in (1, None):
        test_cases.extend([
            Case(m, n, k, mode, "float16", "float16", split_k=split_k)
            for mode in ("plain", "ragged", "batched")
            for (m, n, k) in zero_sized_shapes
        ])
    test_cases.append(Case(5, 11, 7, "batched", "float16", "float16", n_slices=0, split_k=None))
    empty_output_shapes = ((0, 256, 256), (256, 0, 256))
    test_cases.extend([
        Case(*shape, "plain", "bfloat16", "mxfloat4_e2m1", b_hbm_swizzling=True)
        for shape in empty_output_shapes
    ])
    test_cases.extend([
        Case(*shape, "ragged", "nvfp4_e2m1", "nvfp4_e2m1", "nvfp4_e2m1",
             a_hbm_swizzling=True, b_hbm_swizzling=True, c_hbm_swizzling=True)
        for shape in empty_output_shapes
    ])
    test_cases.append(Case(256, 256, 256, "batched", "nvfp4_e2m1", "nvfp4_e2m1", "nvfp4_e2m1",
                           n_slices=0, a_hbm_swizzling=True, b_hbm_swizzling=True, c_hbm_swizzling=True))
    odd_shape1 = (727, 577, 859)
    odd_shape2 = (720, 576, 768)
    even_shape = (768, 512, 1024)
    # canonical float16
    test_cases.extend([
        Case(*shape, mode, "float16", "float16", split_k=split_k)
      for shape in [odd_shape1, even_shape] for mode in ["ragged", "batched"] for split_k in [1, 5]
    ])
    # native float8
    test_cases.extend([
        Case(*shape, mode, "float8_e5m2", "float8_e5m2", split_k=split_k)
     for shape in [odd_shape1, even_shape] for mode in ["ragged", "batched"] for split_k in [1, 5]
    ])
    test_cases.extend([
        Case(*even_shape, "ragged", "float8_e5m2", "float8_e5m2", epilogue_subtile=val)
        for val in (1, 2, 4)
    ])
    # fp32
    test_cases.extend([
        Case(1024, 1000, 2048, "ragged", "float32", "float32", b_transpose=True)
    ])
    # fp64
    test_cases.extend([
        Case(128, 64, 256, "plain", "float64", "float64", split_k=split_k)
        for split_k in [1, 3]
    ])
    # bfloat16 x mx
    for shape in [odd_shape2, even_shape]:
        test_cases.extend([
            Case(*shape, "plain", "bfloat16", "mxfloat4_e2m1"),
            Case(*shape, "plain", "bfloat16", "mxfloat4_e2m1", b_hbm_swizzling=True),
            Case(*shape, "batched", "bfloat16", "mxfloat4_e2m1"),
            Case(*shape, "batched", "bfloat16", "mxfloat4_e2m1", b_hbm_swizzling=True),
            Case(*shape, "ragged", "bfloat16", "mxfloat4_e2m1"),
            Case(*shape, "ragged", "bfloat16", "mxfloat4_e2m1", b_hbm_swizzling=True),
            Case(*shape, "ragged", "bfloat16", "mxfloat4_e2m1", split_k=9),
            Case(*shape, "ragged", "bfloat16", "mxfloat4_e2m1", split_k=9, b_hbm_swizzling=True),
            Case(*shape, "ragged", "bfloat16", "mxfloat8_e4m3fn"),
            Case(*shape, "ragged", "bfloat16", "mxfloat8_e4m3fn", b_hbm_swizzling=True)
        ])
    test_cases.append(Case(64, 256, 32, "plain", "bfloat16", "mxfloat4_e2m1", b_hbm_swizzling=True))
    test_cases.append(Case(128, 128, 128, "plain", "bfloat16", "nvfp4_e2m1"))
    test_cases.append(Case(128, 128, 128, "plain", "bfloat16", "nvfp4_e2m1_fiber"))
    # float8 x mxfloat
    test_cases.extend([
        Case(16, 256, 256, "ragged", "float8_e5m2", "mxfloat4_e2m1", b_hbm_swizzling=True),
        Case(16, 256, 256, "ragged", "float8_e5m2", "mxfloat4_e2m1", b_hbm_swizzling=True, shuffle_mxfp4_w_layout=True),
        Case(1024, 1024, 1024, "batched", "float8_e5m2", "mxfloat4_e2m1", b_hbm_swizzling=True),
        Case(1024, 1024, 1024, "batched", "float8_e5m2", "mxfloat4_e2m1", b_hbm_swizzling=True, shuffle_mxfp4_w_layout=True),
        Case(1024, 1024, 1024, "batched", "float8_e5m2", "mxfloat4_e2m1"),
        Case(1024, 1024, 1024, "ragged", "float8_e5m2", "mxfloat4_e2m1", split_k=9),
        Case(1024, 1024, 1024, "ragged", "float8_e5m2", "mxfloat4_e2m1", split_k=9, b_hbm_swizzling=True),
        Case(1024, 1024, 1024, "ragged", "float8_e5m2", "mxfloat4_e2m1", split_k=9, b_hbm_swizzling=True, shuffle_mxfp4_w_layout=True),
        Case(300, 400, 416, "ragged", "float8_e5m2", "mxfloat8_e4m3fn"),
        Case(300, 400, 832, "ragged", "float8_e5m2", "mxfloat4_e2m1"),
        Case(300, 400, 832, "ragged", "float8_e5m2", "mxfloat4_e2m1", b_hbm_swizzling=True, shuffle_mxfp4_w_layout=True),
        Case(300, 400, 416, "batched", "float8_e5m2", "mxfloat8_e4m3fn"),
        Case(128, 128, 128, "plain", "float8_e5m2", "nvfp4_e2m1"),
        Case(128, 128, 128, "plain", "float8_e5m2", "nvfp4_e2m1_fiber"),
    ])
    # nvfp4 x dense
    test_cases.append(Case(128, 128, 128, "plain", "nvfp4_e2m1", "bfloat16", "bfloat16"))
    test_cases.append(Case(128, 128, 128, "plain", "nvfp4_e2m1_fiber", "bfloat16", "bfloat16"))
    # mxfloat x mxfloat
    test_cases.extend([
        Case(16, 256, 256, "ragged", "mxfloat8_e4m3fn", "mxfloat4_e2m1"),
        Case(16, 256, 256, "ragged", "mxfloat8_e4m3fn", "mxfloat4_e2m1", b_hbm_swizzling=True),
        Case(1024, 1024, 1024, "ragged", "mxfloat8_e4m3fn", "mxfloat4_e2m1", split_k=9),
        Case(1024, 1024, 1024, "ragged", "mxfloat8_e4m3fn", "mxfloat4_e2m1", split_k=9, b_hbm_swizzling=True),
        Case(1024, 1024, 1024, "ragged", "mxfloat8_e4m3fn", "mxfloat4_e2m1", split_k=9, colmajor_mxfp_weight=False),
        Case(1000, 704, 800, "batched", "mxfloat8_e4m3fn", "mxfloat4_e2m1", b_hbm_swizzling=True, a_hbm_swizzling=True),
        Case(1000, 704, 800, "ragged", "mxfloat8_e4m3fn", "mxfloat4_e2m1", b_hbm_swizzling=True, a_hbm_swizzling=True),
        Case(300, 400, 416, "ragged", "mxfloat8_e4m3fn", "mxfloat4_e2m1", b_hbm_swizzling=True, a_hbm_swizzling=True),
        Case(256, 1024, 512, "ragged", "mxfloat8_e4m3fn", "mxfloat4_e2m1", b_hbm_swizzling=True, a_hbm_swizzling=True),
        Case(300, 400, 416, "ragged", "mxfloat8_e4m3fn", "mxfloat8_e4m3fn"),
        Case(300, 400, 416, "ragged", "mxfloat8_e4m3fn", "mxfloat8_e4m3fn", b_hbm_swizzling=True),
        Case(300, 400, 416, "batched", "mxfloat8_e4m3fn", "mxfloat8_e4m3fn"),
        Case(64, 128, 96, "ragged", "mxfloat8_e4m3fn", "bfloat16", "bfloat16"),
        Case(64, 128, 96, "batched", "mxfloat8_e4m3fn", "bfloat16", "bfloat16"),
        Case(1024, 1024, 1024, "batched", "mxfloat8_e4m3fn", "bfloat16", "bfloat16", split_k=9),
        Case(64, 128, 96, "ragged", "mxfloat8_e4m3fn", "float16", "bfloat16"),
        Case(64, 128, 96, "batched", "mxfloat8_e4m3fn", "float16", "bfloat16"),
        Case(1024, 1024, 1024, "batched", "mxfloat8_e4m3fn", "float16", "bfloat16", split_k=9),
        Case(64, 128, 96, "ragged", "mxfloat8_e4m3fn", "bfloat16", "bfloat16", a_hbm_swizzling=True),
        Case(64, 128, 96, "ragged", "mxfloat8_e4m3fn", "float16", "bfloat16", a_hbm_swizzling=True),
        Case(64, 128, 96, "ragged", "mxfloat4_e2m1", "bfloat16", "bfloat16"),
        Case(64, 128, 96, "batched", "mxfloat4_e2m1", "bfloat16", "bfloat16"),
        Case(1024, 1024, 1024, "batched", "mxfloat4_e2m1", "bfloat16", "bfloat16", split_k=9),
        Case(64, 128, 96, "ragged", "mxfloat4_e2m1", "float16", "bfloat16"),
        Case(64, 128, 96, "batched", "mxfloat4_e2m1", "float16", "bfloat16"),
        Case(1024, 1024, 1024, "batched", "mxfloat4_e2m1", "float16", "bfloat16", split_k=9),
        Case(64, 128, 96, "ragged", "mxfloat4_e2m1", "bfloat16", "bfloat16", a_hbm_swizzling=True),
        Case(64, 128, 96, "ragged", "mxfloat4_e2m1", "float16", "bfloat16", a_hbm_swizzling=True),
        Case(1024, 1024, 1024, "batched", "mxfloat8_e4m3fn", "mxfloat4_e2m1", b_hbm_swizzling=True),
        Case(256, 256, 256, "plain", "mxfloat4_e2m1", "mxfloat4_e2m1", "bfloat16"),
        Case(256, 256, 256, "plain", "mxfloat4_e2m1", "mxfloat4_e2m1", "bfloat16", b_hbm_swizzling=True),
        Case(16, 256, 256, "ragged", "nvfp4_e2m1", "nvfp4_e2m1", "bfloat16", b_hbm_swizzling=True),
        Case(1024, 1024, 1024, "ragged", "nvfp4_e2m1", "nvfp4_e2m1", "bfloat16", split_k=9, b_hbm_swizzling=True),
        Case(1024, 1024, 1024, "ragged", "nvfp4_e2m1", "nvfp4_e2m1", "bfloat16", split_k=9, colmajor_mxfp_weight=False),
        Case(1000, 704, 800, "batched", "nvfp4_e2m1", "nvfp4_e2m1", "bfloat16", b_hbm_swizzling=True, a_hbm_swizzling=True),
        Case(1000, 704, 800, "ragged", "nvfp4_e2m1", "nvfp4_e2m1", "bfloat16", b_hbm_swizzling=True, a_hbm_swizzling=True),
        Case(300, 400, 416, "ragged", "nvfp4_e2m1", "nvfp4_e2m1", "bfloat16", b_hbm_swizzling=True, a_hbm_swizzling=True),
        Case(256, 1024, 512, "ragged", "nvfp4_e2m1", "nvfp4_e2m1", "bfloat16", b_hbm_swizzling=True, a_hbm_swizzling=True),
        Case(128, 256, 256, "ragged", "nvfp4_e2m1", "nvfp4_e2m1", "nvfp4_e2m1"),
        Case(128, 256, 256, "ragged", "nvfp4_e2m1_fiber", "nvfp4_e2m1_fiber", "bfloat16"),
        Case(128, 256, 256, "ragged", "nvfp4_e2m1", "nvfp4_e2m1", "nvfp4_e2m1", c_hbm_swizzling=True, b_hbm_swizzling=True, a_hbm_swizzling=True),
        Case(1024, 1024, 1024, "batched", "nvfp4_e2m1", "nvfp4_e2m1", "bfloat16", b_hbm_swizzling=True),
        Case(1024, 1024, 1024, "batched", "nvfp4_e2m1_fiber", "nvfp4_e2m1_fiber", "bfloat16", b_hbm_swizzling=True),
    ])
    # amd-specific float8
    test_cases.extend([
        Case(300, 400, 400, "ragged", "float8_e4m3fnuz", "float8_e4m3fnuz"),
        Case(1000, 400, 400, "ragged", "float8_e4m3fnuz", "float8_e4m3fnuz"),
        Case(600, 400, 400, "ragged", "float8_e4m3fnuz", "float8_e4m3fnuz", split_k=2),
        Case(300, 400, 400, "ragged", "float8_e4m3fn", "float8_e4m3fn"),
    ])
    # transposes / permutes
    test_cases.extend([
        Case(320, 400, 400, "batched", "float16", "float16",
                a_transpose=a_tr, b_transpose=b_tr, c_transpose=c_tr)
        for a_tr, b_tr, c_tr in itertools.product((False, True), repeat=3)
    ])
    test_cases.extend([
        Case(320, 400, 400, "ragged", "float8_e5m2", "float8_e5m2",
                a_transpose=False, b_transpose=True, c_transpose=False),
        Case(320, 400, 400, "ragged", "float8_e5m2", "float8_e5m2",
                a_transpose=True, b_transpose=True, c_transpose=True),
    ])
    # swiglu
    test_cases.extend([
        Case(*shape, mode, "bfloat16", "bfloat16", split_k=split_k, swiglu_opts=(1.1, 1.4))
     for shape in [odd_shape2, even_shape] for mode in ["ragged", "batched"] for split_k in [1, 5]
    ])
    test_cases.extend([
        Case(*even_shape, "ragged", "bfloat16", "bfloat16", epilogue_subtile=val, swiglu_opts=(1.1, 1.4))
        for val in (1, 2, 4)
    ])
    # swiglu together with mxfp8 downcastepilogue
    test_cases.extend([
        Case(*shape, mode, "mxfloat8_e4m3fn", "mxfloat4_e2m1", a_hbm_swizzling=True, b_hbm_swizzling=True, split_k=split_k, swiglu_opts=(1.1, 7))
     for shape in [odd_shape2, even_shape] for mode in ["ragged", "batched"] for split_k in [1, 5]
    ])
    # swiglu together with nvfp4 downcast epilogue
    test_cases.extend([
        Case(*shape, mode, "bfloat16", "bfloat16", "nvfp4_e2m1", swiglu_opts=(1.1, 7.0))
        for shape in [even_shape]
        for mode in ["ragged", "batched"]
    ])
    test_cases.append(Case(256, 2048, 1024, "plain", "bfloat16", "bfloat16", "nvfp4_e2m1", swiglu_opts=(1.1, 7.0)))

    return test_cases


def _supports_persistent_tma(case, a_dtype, b_dtype, inner_expt_opt):
    batch_size = case.n_slices if case.mode == "batched" or inner_expt_opt is not None else 1
    if batch_size * case.m * case.n == 0:
        return True
    if case.k == 0 or case.c_transpose:
        return False
    if case.a_transpose and case.mode == "ragged" and inner_expt_opt is None:
        return False

    a_bits = 4 if a_dtype.is_mxfloat4 else a_dtype.torch_dtype.itemsize * 8
    a_inner = triton.cdiv(case.k, 2) if a_bits == 4 else case.k
    a_stride = case.m if case.a_transpose else a_inner
    if inner_expt_opt != "pad_a" and a_stride * a_bits % 128:
        return False

    b_bits = 4 if b_dtype.is_mxfloat4 else b_dtype.torch_dtype.itemsize * 8
    b_transpose = case.b_transpose or (b_dtype.is_any_float8 and torch.cuda.get_device_capability()[0] < 10)
    if b_dtype.has_mx_scale and case.colmajor_mxfp_weight:
        b_stride = triton.cdiv(case.k, 2) if b_bits == 4 else case.k
    else:
        b_stride = case.k if b_transpose else case.n
    if not (case.b_hbm_swizzling and b_bits == 4) and b_stride * b_bits % 128:
        return False

    if b_dtype.has_mx_scale and case.colmajor_mxfp_weight:
        scale_stride = triton.cdiv(case.k, b_dtype.microblock_size)
        swizzled_scale = case.b_hbm_swizzling and b_dtype.is_mxfloat4
        if not swizzled_scale and scale_stride != 1:
            return False
    if b_dtype.has_mx_scale and not case.colmajor_mxfp_weight and b_bits == 4:
        return False
    return not (inner_expt_opt is not None and b_transpose and inner_expt_opt != "pad_b")


def _supports_test_op_case(case, block_m, do_gather, do_scatter, inner_expt_opt, do_gamma, is_persistent,
                           capability, cuda, hip, cdna3, cdna4, gfx1250):
    a_dtype = DType(case.act_dtype_str)
    b_dtype = DType(case.weight_dtype_str)
    c_dtype = DType(case.output_dtype_str or case.act_dtype_str)

    if cuda:
        if capability < 10 and (a_dtype.is_nvfp4 or b_dtype.is_nvfp4 or c_dtype.is_nvfp4):
            return False
        if capability < 9 and (a_dtype.uses_fp8e4nv or b_dtype.uses_fp8e4nv or c_dtype.uses_fp8e4nv):
            return False
        if b_dtype.is_any_float8 and capability < 9:
            return False
        if case.act_dtype_str == "float16" and b_dtype.has_mx_scale and capability >= 10:
            return False
        if b_dtype.has_mx_scale and a_dtype.has_global_scale and capability < 10:
            return False
    elif hip:
        if a_dtype.is_nvfp4 or b_dtype.is_nvfp4 or c_dtype.is_nvfp4:
            return False
        if a_dtype.is_any_float8 and b_dtype.has_mx_scale and not (cdna4 or gfx1250):
            return False
        if a_dtype.is_any_float8 and b_dtype.name == "mxfloat8_e4m3fn":
            return False
        if a_dtype.has_mx_scale and b_dtype.has_mx_scale:
            return False
        if a_dtype.name == "mxfloat4_e2m1" and case.weight_dtype_str in {"bfloat16", "float16"}:
            return False
        if is_persistent or (case.split_k is not None and case.split_k > 1):
            return False
        if case.act_dtype_str in ("float32", "float64"):
            return False

    if case.swiglu_opts is not None and do_gamma:
        return False
    if "float8_e4m3fnuz" in (case.weight_dtype_str, case.act_dtype_str) and not cdna3:
        return False

    if case.b_hbm_swizzling:
        if hip and (not (cdna4 or gfx1250) or not b_dtype.has_mx_scale):
            return False
        if capability < 9:
            return False
        if capability < 10 and (b_dtype.name != "mxfloat4_e2m1" or a_dtype.is_mxfloat4):
            return False

    if case.a_hbm_swizzling and (hip or capability < 10 or not a_dtype.has_mx_scale or not is_persistent
                                 or block_m < 128 or do_gather):
        return False
    if case.c_hbm_swizzling and (hip or capability < 10 or do_scatter):
        return False

    if inner_expt_opt is not None:
        if case.mode != "ragged":
            return False
        if a_dtype.has_mx_scale and inner_expt_opt != "pad_a":
            return False
        if b_dtype.has_mx_scale:
            if inner_expt_opt != "pad_b" or (is_persistent and not case.b_hbm_swizzling):
                return False
            if hip and (case.act_dtype_str == "bfloat16" or case.b_hbm_swizzling):
                return False
    if not case.colmajor_mxfp_weight and block_m == 16:
        return False

    if case.shuffle_mxfp4_w_layout and (
        not case.b_hbm_swizzling or hip or capability < 10 or b_dtype.name != "mxfloat4_e2m1"
        or not a_dtype.has_global_scale or not case.colmajor_mxfp_weight or not is_persistent
    ):
        return False

    actual_scatter = do_scatter and case.mode != "batched"
    ragged_mx = case.mode == "ragged" and (a_dtype.has_mx_scale or b_dtype.has_mx_scale)
    can_split_k = not actual_scatter and not ragged_mx and inner_expt_opt is None and not c_dtype.has_mx_scale
    if case.split_k is not None and case.split_k > 1 and not can_split_k:
        return False

    if cuda and capability >= 10:
        requires_persistent = case.a_hbm_swizzling or (
            case.b_hbm_swizzling and case.colmajor_mxfp_weight and b_dtype.is_mxfloat4
        )
        if requires_persistent and not is_persistent:
            return False
        if is_persistent and not _supports_persistent_tma(case, a_dtype, b_dtype, inner_expt_opt):
            return False
    elif cuda and capability < 9 and is_persistent:
        batch_size = case.n_slices if case.mode == "batched" or inner_expt_opt is not None else 1
        if batch_size * case.m * case.n:
            return False
    return True


def _build_test_op_parameters():
    cuda = is_cuda()
    hip = is_hip()
    capability = torch.cuda.get_device_capability()[0]
    cdna3, cdna4, gfx1250 = is_hip_cdna3(), is_hip_cdna4(), is_hip_gfx1250()
    num_warps_options = [4, 8] if cuda and capability == 9 else [None]
    persistent_options = [False, True] if cuda else [False]
    scatter_options = [
        (False, False, None),
        (True, False, None),
        (False, True, None),
        (True, True, None),
        (False, False, "pad_b"),
        (False, False, "pad_a"),
    ]
    cases = _build_test_op_cases()
    parameters = []
    for num_warps, is_persistent, do_gamma, (do_gather, do_scatter, inner_expt_opt), block_m, (index, case) in \
        itertools.product(num_warps_options, persistent_options, (False, True), scatter_options, (16, 128),
                          enumerate(cases)):
        if not _supports_test_op_case(case, block_m, do_gather, do_scatter, inner_expt_opt, do_gamma, is_persistent,
                                      capability, cuda, hip, cdna3, cdna4, gfx1250):
            continue
        values = (num_warps, is_persistent, do_gamma, do_gather, do_scatter, inner_expt_opt, block_m,
                  *(getattr(case, field.name) for field in fields(Case)))
        parameter_id = "-".join(f"swiglu_opts{index}" if isinstance(value, tuple) else str(value) for value in values)
        parameters.append(pytest.param(*values, id=parameter_id))
    return parameters


@pytest.mark.parametrize(
    ", ".join(("num_warps", "is_persistent", "do_gamma", "do_gather", "do_scatter", "inner_expt_opt", "block_m",
               *(field.name for field in fields(Case)))),
    _build_test_op_parameters(),
)
@pytest.mark.enable_warmup(priority=2)
def test_op(m, n, k, split_k, do_gather, do_scatter, inner_expt_opt, do_gamma, is_persistent, num_warps, n_slices,
            mode, act_dtype_str, weight_dtype_str, output_dtype_str, block_m, b_hbm_swizzling, shuffle_mxfp4_w_layout, a_hbm_swizzling, colmajor_mxfp_weight, epilogue_subtile,
            a_transpose, b_transpose, c_transpose,
            swiglu_opts, c_hbm_swizzling, device, opt_flags_scope):
    # We catch and re-invoke pytest.skip(), because otherwise pytest may hold a reference to
    # the frame that called pytest.skip, including all the tensors, leading to OOM.
    skip_message = None
    try:
        _test_op(m, n, k, split_k, do_gather, do_scatter, inner_expt_opt, do_gamma, is_persistent, num_warps, n_slices,
                 mode, act_dtype_str, weight_dtype_str, output_dtype_str, block_m, b_hbm_swizzling, shuffle_mxfp4_w_layout, a_hbm_swizzling, colmajor_mxfp_weight, epilogue_subtile,
                 a_transpose, b_transpose, c_transpose,
                 swiglu_opts, c_hbm_swizzling, device, opt_flags_scope)
    except pytest.skip.Exception as e:
        skip_message = str(e)

    if skip_message is not None:
        pytest.skip(skip_message)

def _test_op(m, n, k, split_k, do_gather, do_scatter, inner_expt_opt, do_gamma, is_persistent, num_warps, n_slices,
            mode, act_dtype_str, weight_dtype_str, output_dtype_str, block_m, b_hbm_swizzling, shuffle_mxfp4_w_layout, a_hbm_swizzling, colmajor_mxfp_weight, epilogue_subtile,
            a_transpose, b_transpose, c_transpose,
            swiglu_opts, c_hbm_swizzling, device, opt_flags_scope):
    if is_compile_warmup() and inner_expt_opt is not None and 0 in (m, n, k):
        pytest.skip("zero-sized inner-expert kernels do not preserve FakeTensor specialization")
    a_dtype = DType(act_dtype_str)
    b_dtype = DType(weight_dtype_str)
    c_dtype = DType(output_dtype_str or act_dtype_str)
    device_capability = torch.cuda.get_device_capability()[0]
    expt_is_inner = (inner_expt_opt is not None)
    # TODO: should construct the test case differently rather than overriding here
    if b_dtype.is_any_float8 and device_capability < 10:
        b_transpose = True

    torch.manual_seed(0)

    # set opt flags constraints
    constraints = make_constraints(block_m, split_k, is_persistent, epilogue_subtile, b_hbm_swizzling, weight_dtype_str, num_warps)
    use_blackwell_shuffled_w_layout = shuffle_mxfp4_w_layout and b_hbm_swizzling
    opt_flags.update_opt_flags_constraints(constraints)

    # --- create conditionals ---
    do_bias = inner_expt_opt is None
    do_gather = do_gather and mode != "batched"
    do_scatter = do_scatter and mode != "batched"
    b_value_hbm_swizzling = None
    if b_hbm_swizzling and colmajor_mxfp_weight and b_dtype.is_mxfloat4:
        b_value_hbm_swizzling = layout.make_default_matmul_mxfp4_w_layout(
            mx_axis=-2,
            allow_blackwell_value_shuffle=use_blackwell_shuffled_w_layout,
        )

    # --- create inputs ---
    a, a_scales, a_ragged_metadata = make_random_tensor(
        shape=(m, k),
        n_slices = n_slices,
        dtype = a_dtype,
        device = device,
        ragged_dim = None if mode != "ragged" else 1 if expt_is_inner else 0,
        mxfp_dim = -1 if a_dtype.has_mx_scale else None,
        transpose = a_transpose,
        ragged_padding = inner_expt_opt is not None and "pad_a" in inner_expt_opt,
        squeeze_batch_dim = mode == "plain",
        scale_hbm_swizzling = layout.make_default_matmul_mx_act_scale_layout if a_hbm_swizzling else None,
    )
    b, b_scale_tri, b_ragged_metadata = make_random_tensor(
        shape=(k, n),
        n_slices = n_slices,
        dtype = b_dtype,
        device = device,
        ragged_dim = None if mode != "ragged" or inner_expt_opt is None else 0,
        mxfp_dim = -2 if b_dtype.has_mx_scale else None,
        transpose = b_transpose,
        ragged_padding = inner_expt_opt is not None and "pad_b" in inner_expt_opt,
        squeeze_batch_dim = mode == "plain",
        is_mx_rowmajor = not colmajor_mxfp_weight,
        value_hbm_swizzling = b_value_hbm_swizzling,
        scale_hbm_swizzling = layout.make_default_matmul_mxfp4_w_scale_layout(mx_axis=-2, num_warps=num_warps) if b_hbm_swizzling and colmajor_mxfp_weight and b_dtype.is_mxfloat4 else None,
    )
    if use_blackwell_shuffled_w_layout:
        assert isinstance(b.storage.layout, layout.BlackwellMX4ValueShuffledLayout)
    gather_indx  = None if not do_gather  else torch.randint(0, max(m, 1), (m, ), dtype=torch.int32, device=device)
    scatter_indx = None if not do_scatter else torch.randperm(m, dtype=torch.int32, device=device)
    bias         = None if not do_bias    else torch.randn(b.shape[:-2] + b.shape[-1:], dtype=torch.float32, device=device)
    gammas       = None if not do_gamma   else 2**torch.randint(-5, 0, (m, ), dtype=torch.float32, device=device)

    # --- create fused activation ---
    fused_activation = None
    if swiglu_opts is not None:
        fused_activation = FusedActivation(FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2), swiglu_opts)

    # --- initialize output ---
    c_shape = (n_slices,) if mode == "batched" or inner_expt_opt is not None else tuple() # batch dim
    c_shape += (scatter_indx.shape[0] if do_scatter else a.shape[-2],) # row dim
    c_shape += (b.shape[-1] // (1 if fused_activation is None else fused_activation.specs.reduction_n) ,) # col dim
    c_storage_shape = c_shape[:-1] + (c_shape[-1] // 2,) if c_dtype.has_mx_scale and c_dtype.is_mxfloat4 else c_shape
    c = torch.empty(c_storage_shape, dtype=c_dtype.torch_dtype, device=device)
    if c_transpose:
        c = c.mT.contiguous().mT

    # --- create precision config ---
    wrap_list = lambda vals: torch.tensor(vals, dtype=torch.float32, device=device)
    flex_a = InFlexData(c_dtype.torch_dtype, wrap_list([1.25])) if c_dtype.has_global_scale else InFlexData()
    flex_b = InFlexData(b_dtype.torch_dtype, wrap_list([1.25])) if b_dtype.has_global_scale else InFlexData()
    if c_dtype.has_global_scale:
        flex_c = OutFlexData(c_dtype.torch_dtype, wrap_list([4.00]), wrap_list([0]), None)
    elif c_dtype.is_nvfp4:
        flex_c = OutFlexData(c_dtype.torch_dtype, wrap_list([0.125]), None, None)
    else:
        flex_c = OutFlexData(c_dtype.torch_dtype, None, None, None)
    precision_opt = PrecisionConfig(
        flex_ctx=FlexCtx(flex_a, flex_b, flex_c),
        acc_scale=2.0 if c_dtype.has_global_scale or b_dtype.has_global_scale else 1.0,
        out_dtype=c_dtype.torch_dtype,
        a_mx_scale=a_scales,
        a_microblock_size=a_dtype.microblock_size,
        b_mx_scale=b_scale_tri,
        b_microblock_size=b_dtype.microblock_size,
    )
    def make_tensor_scale(start, end, shape):
        numel = 1
        for dim in shape:
            numel *= dim
        return torch.linspace(start, end, numel, dtype=torch.float32, device=device).reshape(shape)

    if a_dtype.has_tensor_scale:
        precision_opt.a_mx_tensor_scale = make_tensor_scale(0.5, 1.5, a.shape[:-1])
    if b_dtype.has_tensor_scale:
        precision_opt.b_mx_tensor_scale = make_tensor_scale(1.25, 0.75, b.shape[:-2] + b.shape[-1:])

    # --- create epilogue ---
    epilogue = None
    if c_dtype.has_mx_scale:
        c_scale_shape = c_shape[:-1] + (triton.cdiv(c_shape[-1], c_dtype.microblock_size),)
        c_scale = torch.empty(c_scale_shape, dtype=c_dtype.scale_dtype, device=a.device)
        if c_hbm_swizzling:
            c_scale = wrap_torch_tensor(c_scale)
            c_ragged_metadata = a_ragged_metadata if mode == "ragged" else None
            c_scale = convert_layout(c_scale, layout.BlackwellActMXScaleLayout(c_ragged_metadata))
        precision_opt.c_mx_scale = c_scale
        precision_opt.c_microblock_size = c_dtype.microblock_size
        precision_opt.c_value_pack_factor = 2 if c_dtype.is_mxfloat4 else 1
        epilogue_spec = (
            FnSpecs(FnName.QUANTIZE_NVFP4.name, quantize_nvfp4_fn, (), ())
            if c_dtype.is_nvfp4
            else FnSpecs(FnName.QUANTIZE_MXFP8.name, quantize_mxfp8_fn, (), ())
        )
        epilogue = Epilogue(epilogue_spec, tuple(), tuple(), effective_itemsize=2.0 if c_dtype.is_nvfp4 else 6.0)


    # --- triton implementation ---
    try:
        tri_y = matmul(a, b, bias,
                           a_ragged_metadata, b_ragged_metadata,
                           gather_indx, scatter_indx, precision_opt,
                           gammas=gammas, epilogue=epilogue, c=c,
                           fused_activation=fused_activation)
        if c_dtype.has_global_scale:
            tri_y_scale = precision_opt.flex_ctx.out_data.actual_scale.clone()
    except (opt_flags.InapplicableConstraint, NotImplementedError) as e:
        if is_persistent and c.numel() == 0:
            raise
        pytest.skip(f"inapplicable opt_flags constraint {e}")
    # --- torch implementation ---
    # Fused NVFP4 output quantizes the float32 activation result and applies
    # expected_scale inside downcast_to_mxfp_torch, so keep the reference in
    # float32 until that final downcast instead of letting matmul_torch
    # return bf16 and apply the output scale early.
    reference_a = a.float() if c_dtype.is_nvfp4 and not a_dtype.is_nvfp4 else a
    reference_b = b.float() if c_dtype.is_nvfp4 and not b_dtype.is_nvfp4 else b
    reference_precision = (
        PrecisionConfig(
            a_mx_scale=a_scales,
            a_microblock_size=a_dtype.microblock_size,
            b_mx_scale=b_scale_tri,
            b_microblock_size=b_dtype.microblock_size,
        ) if c_dtype.is_nvfp4 else precision_opt
    )
    if is_compile_warmup():
        apply_precision(reference_a, reference_b, reference_precision)
        reference_dtype = (torch.float32 if c_dtype.is_nvfp4 or inner_expt_opt is not None
                           else torch.bfloat16 if a_dtype.has_mx_scale else a_dtype.torch_dtype)
        ref_y = torch.empty(c_shape[:-1] + (n,), dtype=reference_dtype, device=device)
    else:
        ref_y = matmul_torch(
            reference_a,
            reference_b,
            bias,
            a_ragged_metadata,
            b_ragged_metadata,
            gather_indx,
            scatter_indx,
            reference_precision,
            gammas=gammas,
        )
    if swiglu_opts is not None:
        ref_y = swiglu(ref_y, alpha=swiglu_opts[0], precision_config=SwiGLUPrecisionConfig(swiglu_opts[1]))
    if c_dtype.has_global_scale:
        ref_y_scale = precision_opt.flex_ctx.out_data.actual_scale.clone()

    # --- check results ---
    if c_dtype.has_mx_scale:
        tri_y_scale = precision_opt.c_mx_scale
        if isinstance(tri_y_scale, Tensor):
            tri_y_scale = convert_layout(tri_y_scale, layout.StridedLayout()).storage.data
        tri_y = upcast_from_mxfp(tri_y, tri_y_scale, target_dtype=torch.bfloat16, axis=-1).to(ref_y.dtype)
        if not is_compile_warmup():
            ref_target_dtype = ref_y.dtype
            ref_y, ref_scale = downcast_to_mxfp_torch(
                ref_y,
                c_dtype.torch_dtype,
                axis=-1,
                scale_dtype=c_dtype.scale_dtype,
                microblock_size=c_dtype.microblock_size,
                expected_scale=precision_opt.flex_ctx.out_data.expected_scale,
            )
            ref_y = upcast_from_mxfp_torch(ref_y, ref_scale, target_dtype=ref_target_dtype, axis=-1)
    maxtol, rmstol = None, None
    if c_dtype.is_nvfp4 and a_dtype.is_nvfp4 and b_dtype.is_nvfp4:
        maxtol, rmstol = 6e-1, 4e-2
    elif c_dtype.has_mx_scale:
        maxtol, rmstol = 4e-1, 4e-2
    elif b_dtype.is_mxfloat4:
        maxtol, rmstol = 3e-2, None
    elif c_dtype.torch_dtype == torch.float64:
        maxtol, rmstol = 1e-12, 1e-12
    assert_close(ref_y, tri_y, maxtol=maxtol, rmstol=rmstol)
    if c_dtype.has_global_scale and not is_compile_warmup():
        assert torch.all((ref_y_scale - tri_y_scale).abs() < 1e-10), \
               f"ref_y_scale: {ref_y_scale}, tri_y_scale: {tri_y_scale.item()}"


@pytest.mark.parametrize("is_persistent", [False, True])
@pytest.mark.parametrize("n", [960, 1024, 1536, 1568, 1600, 1632, 1664])
def test_mxfp8_act_scale_store_zeroes_partial_group(n, is_persistent, device):
    if not is_cuda() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("requires Blackwell or newer")

    torch.manual_seed(0)
    m, k = 128, 256
    a = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    b = torch.randn((k, n), device=device, dtype=torch.bfloat16)
    scale_blocks = triton.cdiv(n, MXFP_BLOCK_SIZE.value)
    out_scale = convert_layout(
        wrap_torch_tensor(torch.empty((m, scale_blocks), device=device, dtype=torch.uint8)),
        layout.BlackwellActMXScaleLayout(None),
    )
    out_scale.storage.data.fill_(0xFF)
    epilogue = Epilogue(
        FnSpecs(FnName.QUANTIZE_MXFP8.name, quantize_mxfp8_fn, (), ()),
        tuple(),
        tuple(),
        effective_itemsize=6.0,
    )

    with opt_flags.scoped_opt_flags_constraints(
        {"block_m": 128, "block_k": 128, "is_persistent": is_persistent, "split_k": 1}
    ):
        actual = matmul(
            a,
            b,
            None,
            precision_config=PrecisionConfig(
                c_mx_scale=out_scale,
                c_microblock_size=MXFP_BLOCK_SIZE.value,
                out_dtype=torch.float8_e4m3fn,
            ),
            epilogue=epilogue,
        )

    logical_scale = convert_layout(out_scale, layout.StridedLayout(-1)).storage.data
    actual = upcast_from_mxfp(actual, logical_scale, target_dtype=torch.bfloat16, axis=-1)
    assert_close(torch.matmul(a, b), actual, maxtol=4e-1, rmstol=4e-2)

    group_index, valid_bytes = divmod(scale_blocks, 4)
    if valid_bytes:
        scale_group = out_scale.storage.data.select(-3, group_index).reshape(-1, 4)
        assert torch.count_nonzero(scale_group[:, valid_bytes:]).item() == 0

    first_unused_group = triton.cdiv(scale_blocks, 4)
    n_scale_groups = out_scale.storage.data.shape[-3]
    if first_unused_group < n_scale_groups:
        unused_groups = out_scale.storage.data.narrow(
            -3, first_unused_group, n_scale_groups - first_unused_group
        )
        assert torch.all(unused_groups == 0xFF)


def test_k_ragged_mxfp8_act_scale_swizzling(device):
    if not is_cuda() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("requires Blackwell or newer")

    m, n, k = 64, 128, 96
    a_dtype = DType("mxfloat8_e4m3fn")

    def make_a(scale_layout):
        torch.manual_seed(0)
        return make_random_tensor(
            shape=(m, k),
            n_slices=10,
            ragged_dim=1,
            ragged_padding=True,
            device=device,
            dtype=a_dtype,
            mxfp_dim=-1,
            transpose=False,
            squeeze_batch_dim=False,
            scale_hbm_swizzling=scale_layout,
        )

    # A scale layout is supplied in both cases so K-ragged values get identical padding.
    canonical_a, canonical_scale, canonical_metadata = make_a(layout.StridedLayout(-1))
    swizzled_a, swizzled_scale, swizzled_metadata = make_a(layout.make_default_matmul_mx_act_scale_layout)
    b = torch.randn((k, n), dtype=torch.bfloat16, device=device)
    b_metadata = make_ragged_tensor_metadata(canonical_metadata.slice_sizes, k)

    def run(a, scale, metadata):
        return matmul(
            a,
            b,
            None,
            metadata,
            b_metadata,
            precision_config=PrecisionConfig(
                a_mx_scale=scale,
                a_microblock_size=MXFP_BLOCK_SIZE.value,
                out_dtype=torch.bfloat16,
            ),
        )

    with opt_flags.scoped_opt_flags_constraints({"block_m": 128, "is_persistent": True}):
        swizzled = run(swizzled_a, swizzled_scale, swizzled_metadata)
        canonical = run(canonical_a, canonical_scale, canonical_metadata)
    torch.testing.assert_close(swizzled, canonical)


def test_set_idle_sms():
    if not is_cuda():
        pytest.skip("Only supported on CUDA")
    from triton_kernels.matmul_details.opt_flags import make_opt_flags
    num_idle_sms = 24
    matmul_set_idle_sms(num_idle_sms)
    try:
        flags = make_opt_flags(FP32, FP32, FP32, PrecisionConfig(), \
                               1, 1024, 1024, 1024, None, True, False, 1, False, False, None, torch.float32)
        assert flags.idle_sms == num_idle_sms
        with opt_flags.scoped_opt_flags_constraints({"idle_sms": num_idle_sms + 1}):
            flags = make_opt_flags(FP32, FP32, FP32, PrecisionConfig(), \
                                   1, 1024, 1024, 1024, None, True, False, 1, False, False, None, torch.float32)
            assert flags.idle_sms == num_idle_sms + 1
    finally:
        matmul_set_idle_sms(0)
