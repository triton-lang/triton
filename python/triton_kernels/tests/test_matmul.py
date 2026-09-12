# isort: off
# fmt: off
from dataclasses import dataclass, fields
import itertools
import importlib
from types import SimpleNamespace
import pytest
import torch
from typing import Union
import triton
import triton.language as tl
from triton._internal_testing import is_compile_warmup, is_hopper
# matmul utilities
import triton_kernels.matmul_details.opt_flags as opt_flags
from triton_kernels.matmul import FlexCtx, PrecisionConfig, FusedActivation, FusedComm, FnSpecs, FnName, Epilogue
from triton_kernels.matmul import apply_precision, matmul_set_idle_sms, matmul, matmul_torch
# numerics utilities
from triton_kernels.numerics import InFlexData, OutFlexData
from triton_kernels.meta import Closure
from triton_kernels.numerics_details.mxfp import upcast_from_mxfp, quantize_mxfp8_fn, quantize_nvfp4_fn, downcast_to_mxfp_torch, upcast_from_mxfp_torch, MXFP_BLOCK_SIZE, NVFP_BLOCK_SIZE
# testing utilities
from triton_kernels import fpsan
from triton_kernels.testing import assert_close, make_random_tensor
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
        self.uses_fp8e4nv = dtype_str in {"float8_e4m3fn", "mxfloat8_e4m3fn", "nvfp4_e2m1", "nvfp4_e2m1_fiber"}
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
    # unscaled mixed dtypes
    test_cases.extend([
        Case(*shape, "plain", lhs, rhs, "float32" if other_dtype == "float32" else dtype,
             b_transpose=b_transpose)
        for other_dtype, shape in (
            ("float8_e4m3fn", odd_shape2), ("float32", odd_shape2), ("float32", odd_shape1)
        )
        for dtype in ("float16", "bfloat16")
        for lhs, rhs in ((other_dtype, dtype), (dtype, other_dtype))
        for b_transpose in (False, True)
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
    test_cases.extend([
        Case(256, 256, 128, "ragged", "nvfp4_e2m1", rhs_dtype, "bfloat16", a_hbm_swizzling=True)
        for rhs_dtype in ("bfloat16", "float16")
    ])
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
    # MXFP8 lhs x dense rhs needs TMEM headroom even with fused MX output.
    test_cases.extend([
        Case(32, 256, 128, "plain", "mxfloat8_e4m3fn", rhs_dtype, "mxfloat8_e4m3fn",
             a_hbm_swizzling=True, c_hbm_swizzling=c_hbm_swizzling, swiglu_opts=(1.702, 7.0))
        for rhs_dtype in ["bfloat16", "float16"]
        for c_hbm_swizzling in [False, True]
    ])
    # swiglu together with nvfp4 downcast epilogue
    test_cases.extend([
        Case(*shape, mode, "bfloat16", "bfloat16", "nvfp4_e2m1", swiglu_opts=(1.1, 7.0))
        for shape in [even_shape]
        for mode in ["ragged", "batched"]
    ])
    test_cases.append(Case(256, 2048, 1024, "plain", "bfloat16", "bfloat16", "nvfp4_e2m1", swiglu_opts=(1.1, 7.0)))

    return test_cases

@pytest.mark.parametrize(
    ", ".join(f.name for f in fields(Case)),
    [
        tuple(getattr(case, f.name) for f in fields(Case))
        for case in _build_test_op_cases()
    ],
)
@pytest.mark.parametrize("block_m", [16, 128])
@pytest.mark.parametrize("do_gather, do_scatter, inner_expt_opt", [
    (False, False, None),
    (True, False, None),
    (False, True, None),
    (True, True, None),
    (False, False, "pad_b"),
    (False, False, "pad_a"),
])
@pytest.mark.parametrize("do_gamma", [False,True])
@pytest.mark.parametrize("is_persistent", [False,True])
@pytest.mark.parametrize("num_warps", [4, 8] if is_hopper() else [None])
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
    if is_cuda():
        if device_capability < 10 and (a_dtype.is_nvfp4 or b_dtype.is_nvfp4 or c_dtype.is_nvfp4):
            pytest.skip("NVFP4 matmul only tested on Blackwell or newer")
        if device_capability < 9 and (a_dtype.uses_fp8e4nv or b_dtype.uses_fp8e4nv or c_dtype.uses_fp8e4nv):
            pytest.skip("FP8 E4M3FN tests require Hopper or newer")
        if b_dtype.is_any_float8 and b_dtype.has_mx_scale and device_capability < 9:
            pytest.skip("Scaled FP8 weights require Hopper or newer")
        if act_dtype_str == "float16" and b_dtype.has_mx_scale and device_capability >= 10:
            pytest.skip("float16 x mx not supported with cuda capability >= 10")
        if b_dtype.has_mx_scale and a_dtype.has_global_scale and device_capability < 10:
            pytest.skip("float8 x mx not supported with cuda capability < 10")
        if swiglu_opts is not None and do_gamma:
            pytest.skip("NYI: swiglu and gamma not supported together")

    elif is_hip():
        if a_dtype.is_nvfp4 or b_dtype.is_nvfp4 or c_dtype.is_nvfp4:
            pytest.skip("NVFP4 matmul not tested on AMD GPU")
        if a_dtype.is_any_float8 and b_dtype.has_mx_scale and not (is_hip_cdna4() or is_hip_gfx1250()):
            pytest.skip("float8 x mx only supported on CDNA4 and gfx1250")
        if a_dtype.is_any_float8 and b_dtype.name == "mxfloat8_e4m3fn":
            pytest.skip("NYI: float8 x mxfloat8 not tested on AMD GPU")
        if a_dtype.has_mx_scale and b_dtype.has_mx_scale:
            pytest.skip("NYI: mx x mx not tested on AMD GPU")
        if a_dtype.name == "mxfloat4_e2m1" and weight_dtype_str in {"bfloat16", "float16"}:
            pytest.skip("NYI: MXFP4 x dense FP16/BF16 not tested on AMD GPU")
        if is_persistent:
            pytest.skip("NYI: Persistent kernel not supported on AMD GPU")
        # FIXME: this works on nvidia; looks like some sort of bug on AMD?
        if do_gamma and swiglu_opts is not None:
            pytest.skip("NYI: gamma and swiglu not supported together on AMD GPU")
        if split_k is not None and split_k > 1:
            pytest.skip("splitK hasn't been fully tested on AMD GPU.")
        if act_dtype_str == "float64" or (
            act_dtype_str == "float32" and weight_dtype_str not in ("float16", "bfloat16")
        ):
            pytest.skip("float32/float64 not fully tested on AMD GPU")

    if "float8_e4m3fnuz" in (weight_dtype_str, act_dtype_str) and not is_hip_cdna3():
        pytest.skip("float8_e4m3fnuz only tested on AMD CDNA3 Platform")

    if b_hbm_swizzling:
        if is_hip():
            if not (is_hip_cdna4() or is_hip_gfx1250()):
                pytest.skip("Scale preshuffling on AMD GPU has not been emulated on archs other than CDNA4 and gfx1250 yet.")
            if not b_dtype.has_mx_scale:
                pytest.skip("Non-scale swizzling not supported on CDNA4 yet")
        if device_capability < 9:
            pytest.skip("NYI. Ampere swizzling.")
        if device_capability < 10:
            if b_dtype.name != "mxfloat4_e2m1":
                pytest.skip("NYI. Hopper swizzling just implemented for mxfp4.")
            if a_dtype.is_mxfloat4:
                pytest.skip("Hopper mxfp4 swizzled weights do not support FP4 microscaled lhs.")

    if a_hbm_swizzling:
        # current x scale swizzling requires B200, batched input, microscaled act and persistent case
        if is_hip():
            pytest.skip("NYI. X swizzling not tested on AMD GPU yet.")
        if device_capability < 10:
            pytest.skip("NYI. X swizzling only implemented for B200 for now.")
        if not a_dtype.has_mx_scale:
            pytest.skip(f"NYI. X swizzling only implemented for microscaled activations for now. Got {act_dtype_str}")
        if not is_persistent:
            pytest.skip("NYI. X swizzling only implemented for persistent case for now.")
        if block_m < 128:
            pytest.skip("X swizzling requires block_m >= 128")
        if do_gather:
            pytest.skip("X swizzling does not support gathered activations")

    if c_hbm_swizzling:
        if is_hip() or torch.cuda.get_device_capability()[0] < 10:
            pytest.skip("NYI. Output scale swizzling is only implemented on Blackwell")
        if do_scatter:
            pytest.skip("NYI. Output scale swizzling does not support fused scatter")

    expt_is_inner = (inner_expt_opt is not None)
    if expt_is_inner:
        if mode != "ragged":
            pytest.skip("inner_expt_opt only meaningful with ragged")
        if a_dtype.has_mx_scale and inner_expt_opt != "pad_a":
            pytest.skip("inner_expt_opt and act mx only supported with pad_a")
        if b_dtype.has_mx_scale:
            if inner_expt_opt != "pad_b":
                pytest.skip("inner_expt_opt and weight mx only supported with pad_b")
            if is_persistent and not b_hbm_swizzling:
                pytest.skip("FIXME: Fatal Python error: Aborted")
            if is_hip():
                if act_dtype_str == "bfloat16":
                    pytest.skip("FIXME: failed to translate module to LLVM IR")
                if b_hbm_swizzling:
                    pytest.skip("NYI: nner_expt_opt and HBM swizzling")
    if not colmajor_mxfp_weight:
        if block_m == 16:
            pytest.skip("PassManager::run failed from Triton compiler")
    # TODO: construct MX cases with the required layout rather than overriding here.
    if b_dtype.is_any_float8 and (a_dtype.has_mx_scale or b_dtype.has_mx_scale) and device_capability < 10:
        b_transpose = True

    torch.manual_seed(0)

    # set opt flags constraints
    constraints = make_constraints(block_m, split_k, is_persistent, epilogue_subtile, b_hbm_swizzling, weight_dtype_str, num_warps)
    use_blackwell_shuffled_w_layout = shuffle_mxfp4_w_layout and b_hbm_swizzling
    if shuffle_mxfp4_w_layout:
        if not b_hbm_swizzling:
            pytest.skip("Shuffled MXFP4 weight layout only applies with b_hbm_swizzling")
        if is_hip() or device_capability < 10:
            pytest.skip("Shuffled MXFP4 weight layout requires Blackwell or newer")
        if b_dtype.name != "mxfloat4_e2m1":
            pytest.skip("Shuffled MXFP4 weight layout only supports mxfloat4_e2m1 weights")
        if not a_dtype.has_global_scale:
            pytest.skip("Shuffled MXFP4 weight layout is only tested with FP8 activations")
        if not colmajor_mxfp_weight:
            pytest.skip("Shuffled MXFP4 weight layout requires column-major MXFP weights")
        if not is_persistent:
            pytest.skip("Shuffled MXFP4 weight layout requires the persistent TMA kernel")
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
    flex_a = InFlexData(a_dtype.torch_dtype, wrap_list([1.25])) if a_dtype.has_global_scale else InFlexData()
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
        effective_itemsize = 2.0 if c_dtype.is_nvfp4 else 6.0
        if c_dtype.is_nvfp4 and fused_activation is not None and fused_activation.specs.reduction_n > 1:
            effective_itemsize = 4 * fused_activation.specs.reduction_n
        epilogue = Epilogue(epilogue_spec, tuple(), tuple(), effective_itemsize=effective_itemsize)


    # --- triton implementation ---
    is_mixed = (a_dtype.torch_dtype != b_dtype.torch_dtype
                and not a_dtype.has_mx_scale and not b_dtype.has_mx_scale)
    promoted_reference_succeeded = False
    try:
        if is_mixed:
            promoted_dtype = next(common for lhs, rhs, common in _mixed_dtype_cases()
                                  if {lhs, rhs} == {a_dtype.torch_dtype, b_dtype.torch_dtype})
            promoted_y = matmul(a.to(promoted_dtype), b.to(promoted_dtype), bias,
                            a_ragged_metadata, b_ragged_metadata,
                            gather_indx, scatter_indx, precision_opt,
                            gammas=gammas, epilogue=epilogue, fused_activation=fused_activation)
            promoted_reference_succeeded = True
        tri_y = matmul(a, b, bias,
                           a_ragged_metadata, b_ragged_metadata,
                           gather_indx, scatter_indx, precision_opt,
                           gammas=gammas, epilogue=epilogue, c=c,
                           fused_activation=fused_activation)
        if c_dtype.has_global_scale:
            tri_y_scale = precision_opt.flex_ctx.out_data.actual_scale.clone()
    except (opt_flags.InapplicableConstraint, NotImplementedError) as e:
        if promoted_reference_succeeded or (is_persistent and c.numel() == 0):
            raise
        pytest.skip(f"inapplicable opt_flags constraint {e}")
    if is_mixed and not is_compile_warmup():
        torch.testing.assert_close(tri_y.contiguous().view(torch.uint8), promoted_y.contiguous().view(torch.uint8),
                                   rtol=0, atol=0)
    # --- torch implementation ---
    # Fused NVFP4 output quantizes the float32 activation result and applies
    # expected_scale inside downcast_to_mxfp_torch, so keep the reference in
    # float32 until that final downcast instead of letting matmul_torch
    # return bf16 and apply the output scale early.
    # FP32 outputs must also avoid rounding to the activation dtype.
    reference_a = (
        a.float() if c_dtype.torch_dtype == torch.float32 or (c_dtype.is_nvfp4 and not a_dtype.is_nvfp4) else a
    )
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
        reference_dtype = (torch.float32 if c_dtype.is_nvfp4 or c_dtype.torch_dtype == torch.float32 or inner_expt_opt is not None
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


@pytest.mark.parametrize("shape, fp8_lhs, constraints", [
    ((273, 544, 576), True, dict(block_m=64, block_n=256, block_k=128, split_k=1, is_persistent=True)),
    ((273, 544, 576), False, dict(block_m=128, block_n=256, block_k=128, split_k=1, is_persistent=True, swap_xw=False)),
    ((128, 256, 1024), True, {}),
    ((273, 544, 576), False, {}),
])
def test_matmul_mixed_fp8_resource_limits(shape, fp8_lhs, constraints, device, opt_flags_scope):
    if is_cuda() and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("requires Hopper or newer")
    if is_hip() and constraints.get("is_persistent"):
        pytest.skip("Persistent kernel not supported on AMD GPU")

    torch.manual_seed(0)
    m, n, k = shape
    a_dtype, b_dtype = (torch.float8_e4m3fn, torch.bfloat16) if fp8_lhs else (torch.bfloat16, torch.float8_e4m3fn)
    a = torch.randn((m, k), device=device).to(a_dtype)
    b = torch.randn((n, k), device=device).to(b_dtype).mT

    opt_flags.update_opt_flags_constraints(constraints)
    actual = matmul(a, b, None, precision_config=PrecisionConfig(out_dtype=torch.bfloat16))

    expected = torch.matmul(a.float(), b.float()).to(torch.bfloat16)
    assert_close(expected, actual)


@pytest.mark.parametrize("dtype, other_dtype, allow_tf32", [
    (torch.float16, torch.float8_e4m3fn, True),
    *[(torch.float32, dtype, allow_tf32)
      for dtype in (torch.float16, torch.bfloat16)
      for allow_tf32 in (False, True)],
])
@pytest.mark.parametrize("high_precision_lhs", [False, True])
@pytest.mark.parametrize("is_persistent", [False, True])
def test_matmul_mixed_preserves_precision(dtype, other_dtype, allow_tf32, high_precision_lhs, is_persistent,
                                        device, opt_flags_scope):
    if other_dtype == torch.float8_e4m3fn and is_cuda() and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("requires Hopper or newer")
    if is_persistent and (is_hip() or torch.cuda.get_device_capability()[0] < 9):
        pytest.skip("persistent matmul requires Hopper or newer")

    # FP32 values expose FP16 overflow; the residual exposes BF16 rounding
    # and TF32 use when disabled. The FP8 case preserves FP16's extra mantissa bits.
    scale = 2**16 if dtype == torch.float32 else 1
    residual = scale * 2**(-10 if allow_tf32 else -20)
    a = torch.zeros((128, 128), dtype=dtype, device=device)
    b = torch.zeros_like(a)
    a[:, 0] = 1
    a[:, 1] = -1
    b[0, :] = scale + residual
    b[1, :] = scale
    a = a.to(other_dtype)
    if high_precision_lhs:
        a, b = b.mT.contiguous(), a.mT.contiguous()

    opt_flags.update_opt_flags_constraints(dict(is_persistent=is_persistent, split_k=1))
    actual = matmul(a, b, None, precision_config=PrecisionConfig(out_dtype=dtype, allow_tf32=allow_tf32))

    expected = torch.full((128, 128), residual, dtype=dtype, device=device)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fp32_lhs", [False, True])
@pytest.mark.parametrize("allow_tf32", [False, True])
@pytest.mark.parametrize("shape, step, constraints, b_transpose, out_dtype", [
    (shape, step, constraints, b_transpose, None)
    for shape, step, constraints in [
        ((16, 512, 256), 1, {}),
        ((65536, 128, 128), 1, {}),
        ((65536, 128, 132), 1, {}),
        ((727, 577, 859), 1, {}),
        ((512, 256, 128), 2, {}),
        *[(shape, 1, dict(is_persistent=is_persistent, split_k=1))
          for shape in ((16, 512, 256), (65536, 128, 128))
          for is_persistent in (False, True)],
    ]
    for b_transpose in (False, True)
] + [
    ((128, 256, 128), 1, dict(is_persistent=True, split_k=1), False, None),
    ((8192, 2048, 256), 1, {}, True, torch.bfloat16),
])
@pytest.mark.enable_warmup(priority=2)
def test_matmul_mixed_fp32_matches_cast(dtype, fp32_lhs, b_transpose, allow_tf32, shape, step,
                                       constraints, out_dtype, device, opt_flags_scope):
    if constraints.get("is_persistent") and (is_hip() or torch.cuda.get_device_capability()[0] < 9):
        pytest.skip("persistent matmul requires Hopper or newer")

    torch.manual_seed(0)
    m, n, k = shape
    a_dtype, b_dtype = (torch.float32, dtype) if fp32_lhs else (dtype, torch.float32)
    a_step, b_step = (1, step) if fp32_lhs else (step, 1)
    a = torch.randn((m, k * a_step), dtype=a_dtype, device=device)[:, ::a_step]
    if b_transpose:
        b = torch.randn((n, k * b_step), dtype=b_dtype, device=device)[:, ::b_step].mT
    else:
        b = torch.randn((k, n * b_step), dtype=b_dtype, device=device)[:, ::b_step]

    opt_flags.update_opt_flags_constraints(constraints)
    expected = matmul(a.float(), b.float(), None,
                      precision_config=PrecisionConfig(out_dtype=out_dtype, allow_tf32=allow_tf32))
    actual = matmul(a, b, None, precision_config=PrecisionConfig(out_dtype=out_dtype, allow_tf32=allow_tf32))
    if not is_compile_warmup():
        assert actual.dtype == expected.dtype == (out_dtype or torch.float32)
        torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)


def _supported_float_dtypes():
    target = triton.runtime.driver.active.get_current_target()
    supported_fp8 = triton.compiler.make_backend(target).parse_options({}).supported_fp8_dtypes
    float8 = (
        (torch.float8_e4m3fn, "fp8e4nv"),
        (torch.float8_e5m2, "fp8e5"),
        (torch.float8_e4m3fnuz, "fp8e4b8"),
        (torch.float8_e5m2fnuz, "fp8e5b16"),
    )
    return tuple(dtype for dtype, name in float8 if name in supported_fp8) + (
        torch.float16, torch.bfloat16, torch.float32, torch.float64,
    )


def _mixed_dtype_cases():
    float8 = [dtype for dtype in _supported_float_dtypes() if torch.finfo(dtype).bits == 8]
    # This table is independent of the implementation's promotion helper.
    return [
        *[(lhs, rhs, torch.float16) for lhs, rhs in itertools.combinations(float8, 2)],
        *[(lhs, rhs, rhs) for lhs, rhs in itertools.product(
            float8, (torch.float16, torch.bfloat16, torch.float32, torch.float64))],
        (torch.float16, torch.bfloat16, torch.float32),
        (torch.float16, torch.float32, torch.float32),
        (torch.float16, torch.float64, torch.float64),
        (torch.bfloat16, torch.float32, torch.float32),
        (torch.bfloat16, torch.float64, torch.float64),
        (torch.float32, torch.float64, torch.float64),
    ]


_MIXED_MATMUL_CASES = [
    ((64, 128, 1024), {}),
    ((67, 80, 272), dict(is_persistent=False, split_k=3)),
    ((67, 80, 272), dict(is_persistent=True, split_k=1)),
]

_MIXED_MATMUL_OUTPUT_CASES = [
    (out_dtype, *case)
    for out_dtype, case in itertools.product([None, *_supported_float_dtypes()], _MIXED_MATMUL_CASES)
] + [(torch.float64, (128, 256, 128), dict(is_persistent=True, split_k=1))]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("b_transpose", [False, True])
@pytest.mark.parametrize("allow_tf32", [False, True])
@pytest.mark.parametrize("a_dtype, b_dtype, promoted_dtype, out_dtype, shape, constraints, repeats", [
    (*dtypes, *case, 1)
    for dtypes, case in itertools.product(_mixed_dtype_cases(), _MIXED_MATMUL_OUTPUT_CASES)
] + [
    # Exercise successive output tiles and repeated launches of the persistent kernel.
    (torch.float32, dtype, torch.float32, torch.float32, (8192, 2048, 128), {}, 2)
    for dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    if dtype in _supported_float_dtypes()
] + [
    # Cover automatic split-K and the smaller regular tile for FP8 weights.
    (torch.bfloat16, dtype, torch.bfloat16, out_dtype, shape, {}, 1)
    for dtype, (out_dtype, shape) in itertools.product(
        (torch.float8_e4m3fn, torch.float8_e5m2), [
            *itertools.product(
                (torch.float32, torch.float64),
                ((1024, 1024, 1024), (4096, 1024, 272), (4097, 1024, 272), (2049, 4097, 272))),
            *itertools.product(
                (torch.float8_e4m3fn, torch.float8_e5m2, torch.float16, torch.bfloat16),
                ((4097, 1031, 272), (4096, 4096, 264))),
        ])
    if dtype in _supported_float_dtypes() and out_dtype in _supported_float_dtypes()
] + [
    # FP64 output must fit the automatic persistent tile and preserve split-K.
    (dtype, torch.float32, torch.float32, torch.float64, (4096, 4096, 128), {}, 1)
    for dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.float16, torch.bfloat16)
    if dtype in _supported_float_dtypes()
] + [
    (torch.float16, torch.bfloat16, torch.float32, torch.float64, (4096, 4096, 128), {}, 1),
    (torch.float16, torch.float32, torch.float32, torch.float64,
     (1024, 1024, 1024), dict(is_persistent=True), 1),
])
@pytest.mark.enable_warmup(priority=2)
def test_matmul_mixed_dtypes(a_dtype, b_dtype, promoted_dtype, reverse, b_transpose, allow_tf32,
                            out_dtype, shape, constraints, repeats, device, opt_flags_scope):
    if constraints.get("is_persistent") and (is_hip() or torch.cuda.get_device_capability()[0] < 9):
        pytest.skip("persistent matmul requires Hopper or newer")
    if reverse:
        a_dtype, b_dtype = b_dtype, a_dtype

    torch.manual_seed(0)
    m, n, k = shape
    a = torch.randn((m, k), dtype=torch.float64, device=device).to(a_dtype)
    b = torch.randn((n, k) if b_transpose else (k, n), dtype=torch.float64, device=device).to(b_dtype)
    if b_transpose:
        b = b.mT
    if a.element_size() == 1 or b.element_size() == 1:
        fp8 = a if a.element_size() == 1 else b.mT
        bits = {
            torch.float8_e4m3fn: [0x00, 0x80, 0x01, 0x81, 0x7f, 0xff],
            torch.float8_e5m2: [0x00, 0x80, 0x01, 0x81, 0x7c, 0xfc, 0x7d, 0x7e, 0x7f, 0xfd, 0xfe, 0xff],
            torch.float8_e4m3fnuz: [0x00, 0x01, 0x81, 0x80],
            torch.float8_e5m2fnuz: [0x00, 0x01, 0x81, 0x80],
        }[fp8.dtype]
        # Isolate special values to a few output rows or columns; keep the rest random.
        fp8[:len(bits)].zero_()
        fp8[:len(bits), 0] = torch.tensor(bits, dtype=torch.uint8, device=device).view(fp8.dtype)
    opt_flags.update_opt_flags_constraints(constraints)
    expected = matmul(a.to(promoted_dtype), b.to(promoted_dtype), None,
                      precision_config=PrecisionConfig(out_dtype=out_dtype, allow_tf32=allow_tf32))
    check_accuracy = repeats > 1 and not allow_tf32 and expected.dtype == torch.float32 and not is_compile_warmup()
    if check_accuracy:
        a_fp64, b_fp64 = a.double(), b.double()
        reference = a_fp64 @ b_fp64
        finite = torch.isfinite(reference)
        # A length-K FP32 dot has error bounded by gamma_(2K) * sum(abs(a*b)).
        roundoff = k * torch.finfo(torch.float32).eps
        atol = roundoff / (1 - roundoff) * (a_fp64.abs() @ b_fp64.abs())[finite].max().item()
        torch.testing.assert_close(expected.double(), reference, rtol=0, atol=atol, equal_nan=True,
                                   msg=lambda msg: f"Promoted-input matmul versus FP64 reference:\n{msg}")
    for _ in range(repeats):
        actual = matmul(a, b, None, precision_config=PrecisionConfig(out_dtype=out_dtype, allow_tf32=allow_tf32))
        if not is_compile_warmup():
            assert actual.dtype == expected.dtype == (promoted_dtype if out_dtype is None else out_dtype)
            if check_accuracy:
                torch.testing.assert_close(actual.double(), reference, rtol=0, atol=atol, equal_nan=True,
                                           msg=lambda msg: f"Mixed-input matmul versus FP64 reference:\n{msg}")
                torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
            torch.testing.assert_close(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8),
                                       rtol=0, atol=0)
            if actual.dtype in (torch.float32, torch.float64):
                int_dtype, nan_bits = {
                    torch.float32: (torch.int32, 0x7fc00000),
                    torch.float64: (torch.int64, 0x7ff8000000000000),
                }[actual.dtype]
                assert torch.all(actual.view(int_dtype)[torch.isnan(actual)] == nan_bits)


@pytest.mark.parametrize("out_dtype, midpoint", [
    (torch.float8_e4m3fn, 1.0625),
    (torch.float8_e5m2, 1.125),
])
@pytest.mark.parametrize("a_dtype, b_dtype", [
    (torch.float64, torch.float16),
    (torch.float16, torch.float64),
    (torch.float64, torch.float64),
])
@pytest.mark.parametrize("shape, constraints", _MIXED_MATMUL_CASES)
@pytest.mark.parametrize("accumulate", [False, True])
def test_matmul_fp64_fp8_output_rounding(a_dtype, b_dtype, out_dtype, midpoint, shape, constraints, accumulate,
                                        device, opt_flags_scope):
    if out_dtype not in _supported_float_dtypes():
        pytest.skip("output format is not supported by this backend")
    if constraints.get("is_persistent") and (is_hip() or torch.cuda.get_device_capability()[0] < 9):
        pytest.skip("persistent matmul requires Hopper or newer")
    m, n, k = shape
    values = torch.tensor([midpoint - 2**-30, midpoint, midpoint + 2**-30,
                           -midpoint - 2**-30, -midpoint, -midpoint + 2**-30], dtype=torch.float64, device=device)
    a = torch.zeros((m, k), dtype=a_dtype, device=device)
    b = torch.zeros((k, n), dtype=b_dtype, device=device)
    if a_dtype == torch.float64:
        a[:, 0] = values.repeat(triton.cdiv(m, values.numel()))[:m]
        b[0, :] = 1
        expected = a[:, :1].expand(m, n)
    else:
        a[:, 0] = 1
        b[0, :] = values.repeat(triton.cdiv(n, values.numel()))[:n]
        expected = b[:1, :].expand(m, n)
    c = torch.full((m, n), 0.5, dtype=torch.float64, device=device).to(out_dtype) if accumulate else None
    if accumulate:
        expected = expected + 0.5
    opt_flags.update_opt_flags_constraints(dict(constraints, split_k=1) if accumulate else constraints)
    actual = matmul(a, b, None, c=c, c_acc_in=c,
                    precision_config=PrecisionConfig(out_dtype=out_dtype, allow_tf32=False))
    expected = expected.float().to(out_dtype)
    torch.testing.assert_close(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8),
                               rtol=0, atol=0)


@pytest.mark.parametrize("a_dtype, b_dtype, promoted_dtype", [
    (torch.float16, torch.bfloat16, torch.float32),
    (torch.float32, torch.float64, torch.float64),
])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("allow_tf32", [False, True])
@pytest.mark.parametrize("is_persistent", [False, True])
def test_matmul_mixed_range_and_precision(a_dtype, b_dtype, promoted_dtype, reverse, allow_tf32,
                                         is_persistent, device, opt_flags_scope):
    if is_persistent and (is_hip() or torch.cuda.get_device_capability()[0] < 9):
        pytest.skip("persistent matmul requires Hopper or newer")
    a = torch.zeros((128, 128), dtype=a_dtype, device=device)
    b = torch.zeros((128, 128), dtype=b_dtype, device=device)
    a[:, 1] = -1
    if b_dtype == torch.bfloat16:
        a[:, 0] = 1 + 2**-10
        b[:2, :] = 2**16
        residual = 2**6
    else:
        a[:, 0] = 1 + 2**-20
        b[0, :] = 1
        b[1, :] = 1 - 2**-40
        residual = 2**-20 + 2**-40
    if reverse:
        a, b = b.mT.contiguous(), a.mT.contiguous()

    opt_flags.update_opt_flags_constraints(dict(is_persistent=is_persistent, split_k=1))
    actual = matmul(a, b, None, precision_config=PrecisionConfig(allow_tf32=allow_tf32))
    expected = torch.full((128, 128), residual, dtype=promoted_dtype, device=device)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("a_dtype, b_dtype, promoted_dtype", _mixed_dtype_cases())
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("b_transpose", [False, True])
@pytest.mark.parametrize("allow_tf32", [False, True])
@pytest.mark.parametrize("out_dtype", _supported_float_dtypes())
def test_matmul_mixed_accumulation(a_dtype, b_dtype, promoted_dtype, reverse, b_transpose, allow_tf32, out_dtype,
                                 device, opt_flags_scope):
    if reverse:
        a_dtype, b_dtype = b_dtype, a_dtype
    torch.manual_seed(0)
    m, n, k = 4096, 4096, 128
    a = torch.randn((m, k), dtype=torch.float64, device=device).to(a_dtype)
    b = torch.randn((n, k) if b_transpose else (k, n), dtype=torch.float64, device=device).to(b_dtype)
    if b_transpose:
        b = b.mT
    actual = torch.randn((m, n), dtype=torch.float64, device=device).to(out_dtype)
    expected = actual.clone()
    config = PrecisionConfig(out_dtype=out_dtype, allow_tf32=allow_tf32)
    expected = matmul(a.to(promoted_dtype), b.to(promoted_dtype), None, c=expected, c_acc_in=expected,
                      precision_config=config)
    actual = matmul(a, b, None, c=actual, c_acc_in=actual, precision_config=config)
    torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)


@pytest.mark.parametrize("a_dtype, b_dtype, promoted_dtype, out_dtype, shape, constraints, b_transpose", [
    (torch.float8_e4m3fn, dtype, dtype, out_dtype, shape, constraints, b_transpose)
    for dtype in (torch.float16, torch.bfloat16)
    for out_dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float8_e4m3fn, torch.float8_e5m2)
    for shape, constraints in _MIXED_MATMUL_CASES
    for b_transpose in (False, True)
] + [
    (a_dtype, b_dtype, promoted_dtype, out_dtype, (16, 16, 32), dict(is_persistent=False, split_k=1), False)
    for a_dtype, b_dtype, promoted_dtype in [
        (torch.float8_e4m3fn, torch.float8_e5m2, torch.float16),
        (torch.float8_e5m2, torch.bfloat16, torch.bfloat16),
        (torch.float16, torch.bfloat16, torch.float32),
        (torch.float32, torch.float64, torch.float64),
    ]
    for out_dtype in (None, torch.float16, torch.float8_e5m2)
])
@pytest.mark.parametrize("reverse", [False, True])
def test_matmul_mixed_dtypes_fpsan(a_dtype, b_dtype, promoted_dtype, reverse, out_dtype, shape,
                                  constraints, b_transpose, device, opt_flags_scope, fresh_knobs):
    if a_dtype not in _supported_float_dtypes() or b_dtype not in _supported_float_dtypes():
        pytest.skip("input format is not supported by this backend")
    if constraints.get("is_persistent") and (is_hip() or torch.cuda.get_device_capability()[0] < 9):
        pytest.skip("persistent matmul requires Hopper or newer")
    if is_hip() and not (is_hip_cdna3() or is_hip_cdna4() or is_hip_gfx1250()):
        pytest.skip("FPSan requires gfx942, gfx950, or gfx1250")
    if reverse:
        a_dtype, b_dtype = b_dtype, a_dtype
    torch.manual_seed(0)
    m, n, k = shape
    a = torch.randn((m, k), dtype=torch.float64, device=device).to(a_dtype)
    b = torch.randn((n, k) if b_transpose else (k, n), dtype=torch.float64, device=device).to(b_dtype)
    if b_transpose:
        b = b.mT
    payload_dtype = {
        torch.float16: torch.int16, torch.bfloat16: torch.int16,
        torch.float32: torch.int32, torch.float64: torch.int64,
    }[promoted_dtype]
    fresh_knobs.compilation.instrumentation_mode = ""
    # FPSan widening sign-extends payloads; a Torch float cast would change them.
    reference_a, reference_b = [
        x if x.dtype == promoted_dtype else fpsan.unembed(fpsan.embed(x).to(payload_dtype), promoted_dtype)
        for x in (a, b)
    ]
    fresh_knobs.compilation.instrumentation_mode = "fpsan"
    fresh_knobs.compilation.fpsan_homomorphic_casts = False
    opt_flags.update_opt_flags_constraints(constraints)
    expected = matmul(reference_a, reference_b, None,
                      precision_config=PrecisionConfig(out_dtype=out_dtype, allow_tf32=False))
    actual = matmul(a, b, None, precision_config=PrecisionConfig(out_dtype=out_dtype, allow_tf32=False))
    assert actual.dtype == expected.dtype == (promoted_dtype if out_dtype is None else out_dtype)
    torch.testing.assert_close(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8),
                               rtol=0, atol=0)


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


@pytest.fixture
def matmul_launch_stub(monkeypatch):
    module = importlib.import_module("triton_kernels.matmul")
    flags = opt_flags.OptFlags(
        block_m=32, block_n=64, block_k=32, num_warps=4, num_stages=2,
        group_m=1, xcd_swizzle=1, w_cache_modifier="", split_k=1,
        is_persistent=False, idle_sms=0, epilogue_subtile=1, arch=80,
        occupancy_target=1, target_kernel_kwargs={},
    )
    launches = []

    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: launches.append((args, kwargs))

    kernels = SimpleNamespace(_matmul=Kernel(), _p_matmul=Kernel())
    monkeypatch.setattr(module, "make_opt_flags", lambda *args, **kwargs: flags)
    monkeypatch.setattr(module, "get_swap_xw", lambda *args: False)
    monkeypatch.setattr(module.specializations, "get", lambda **kwargs: kernels)
    return flags, launches


@pytest.mark.parametrize("fused, output_kind", [(True, "none"), (True, "meta"), (True, "real"),
                                               (False, "none"), (False, "real")])
@pytest.mark.parametrize("m, n", [(32, 64), (0, 64), (32, 0)])
def test_fused_comm_output_metadata(matmul_launch_stub, monkeypatch, fused, output_kind, m, n):
    _, launches = matmul_launch_stub
    a = torch.empty((m, 32), dtype=torch.float16)
    b = torch.empty((32, n), dtype=torch.float16)
    c = None if output_kind == "none" else torch.empty(
        (m, n), dtype=torch.float16, device="meta" if output_kind == "meta" else "cpu")
    comm = FusedComm(torch.empty(1, dtype=torch.uint64), None, None) if fused else None
    omit_local_output = fused and output_kind != "real"
    allocations = []
    original_empty = torch.empty

    def record_empty(*args, **kwargs):
        tensor = original_empty(*args, **kwargs)
        allocations.append(tensor)
        return tensor

    monkeypatch.setattr(torch, "empty", record_empty)
    result = matmul(a, b, None, c=c, fused_comm=comm)
    assert result.shape == (m, n)
    assert result.dtype == a.dtype
    assert result.is_meta == omit_local_output
    if output_kind == "real":
        assert result.data_ptr() == c.data_ptr()
    if omit_local_output:
        assert all(t.is_meta or t.untyped_storage().nbytes() == 0 for t in allocations)
    assert len(launches) == int(m * n != 0)
    if launches:
        args, kwargs = launches[0]
        assert not args[0].is_meta and not args[1].is_meta
        assert args[0].dtype == result.dtype and args[1].dtype == result.dtype
        assert kwargs["Y_TMA_MODE"] is None
        if omit_local_output:
            assert args[0].numel() == args[1].numel() == 0


@pytest.mark.parametrize("output_kind", ["none", "meta"])
@pytest.mark.parametrize("unsupported", ["split_k", "output_tma", "accumulation", "mx_scale"])
def test_fused_comm_no_local_output_rejects_unsupported(matmul_launch_stub, output_kind, unsupported):
    flags, launches = matmul_launch_stub
    a = torch.empty((32, 32), dtype=torch.float16)
    b = torch.empty((32, 64), dtype=torch.float16)
    c = torch.empty((32, 64), dtype=torch.float16, device="meta") if output_kind == "meta" else None
    comm = FusedComm(torch.empty(1, dtype=torch.uint64), None, None)
    config = PrecisionConfig()
    c_acc_in = None
    if unsupported == "split_k":
        flags.split_k = 2
    elif unsupported == "output_tma":
        flags.use_output_tma = True
    elif unsupported == "accumulation":
        c_acc_in = torch.empty((32, 64), dtype=torch.float16)
    else:
        config.c_mx_scale = torch.empty((32, 2), dtype=torch.uint8)
    error = NotImplementedError if unsupported == "mx_scale" else ValueError
    with pytest.raises(error, match="[Ff]used comm"):
        matmul(a, b, None, c=c, fused_comm=comm, precision_config=config, c_acc_in=c_acc_in)
    assert not launches


@triton.jit
def _fused_comm_map_dst(base_m, rows, base_n, cols, rows_per_peer: tl.constexpr):
    return (rows // rows_per_peer)[:, None], rows % rows_per_peer, cols


@triton.jit
def _fused_comm_writes_issued():
    pass


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n_peers", [1, 2])
@pytest.mark.parametrize("ragged", [False, True])
@pytest.mark.parametrize("is_persistent", [False, True])
def test_fused_comm_no_local_output(dtype, n_peers, ragged, is_persistent):
    if not is_cuda() or torch.cuda.device_count() < n_peers:
        pytest.skip("Requires CUDA devices for all communication peers")
    if is_persistent and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Persistent matmul requires compute capability >= 9")
    device = torch.cuda.current_device()
    peer_devices = [(device + peer) % torch.cuda.device_count() for peer in range(n_peers)]
    if n_peers > 1 and not torch.cuda.can_device_access_peer(*peer_devices):
        pytest.skip("Requires CUDA peer access")
    a = torch.randn((1024, 128), device=device, dtype=dtype)
    b = torch.randn((2, 128, 256) if ragged else (128, 256), device=device, dtype=dtype)
    metadata = make_ragged_tensor_metadata(
        torch.tensor([512, 512], device=device, dtype=torch.int32), 1024) if ragged else None
    scatter = torch.arange(1023, -1, -1, device=device, dtype=torch.int32) if ragged else None
    destinations = [torch.empty((1024 // n_peers, 256), device=peer, dtype=dtype) for peer in peer_devices]
    if n_peers > 1:
        destinations[1].copy_(destinations[0])
        torch.cuda.synchronize(peer_devices[1])
    handles = torch.tensor([d.data_ptr() for d in destinations], device=device, dtype=torch.uint64)
    comm = FusedComm(handles, Closure(_fused_comm_map_dst, (1024 // n_peers,)),
                     Closure(_fused_comm_writes_issued, ()))
    config = PrecisionConfig(out_dtype=dtype)

    def run(c=None):
        return matmul(a, b, None, a_ragged_metadata=metadata, scatter_indx=scatter,
                      precision_config=config, fused_comm=comm, c=c)

    def collected():
        torch.cuda.synchronize(device)
        return torch.cat([d.to(device) for d in destinations])

    expected = torch.cat([a[:512] @ b[0], a[512:] @ b[1]]).flip(0) if ragged else a @ b
    with opt_flags.scoped_opt_flags_constraints({"split_k": 1, "is_persistent": is_persistent}):
        local_output = torch.empty((1024, 256), device=device, dtype=dtype)
        assert run(local_output).data_ptr() == local_output.data_ptr()
        reference = collected()
        torch.testing.assert_close(reference, expected, rtol=0.02, atol=0.1)
        for c in (None, torch.empty((1024, 256), device="meta", dtype=dtype)):
            for destination in destinations:
                destination.zero_()
                torch.cuda.synchronize(destination.device)
            result = run(c)
            assert result.is_meta and result.shape == local_output.shape and result.dtype == dtype
            torch.testing.assert_close(collected(), reference, rtol=0, atol=0)
        torch.cuda.synchronize(device)
        before = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        run()
        torch.cuda.synchronize(device)
        assert torch.cuda.memory_allocated(device) == before
        assert torch.cuda.max_memory_allocated(device) == before
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for _ in range(3):
            for destination in destinations:
                destination.zero_()
                torch.cuda.synchronize(destination.device)
            graph.replay()
            torch.testing.assert_close(collected(), reference, rtol=0, atol=0)
