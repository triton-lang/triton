# isort: off
# fmt: off
from dataclasses import dataclass, fields
import itertools
import pytest
import torch
from typing import Union
import triton
# matmul utilities
import triton_kernels.matmul_details.opt_flags as opt_flags
from triton_kernels.matmul import FlexCtx, PrecisionConfig, FusedActivation, FnSpecs, FnName, Epilogue
from triton_kernels.matmul import matmul_set_idle_sms, matmul, matmul_torch
# numerics utilities
from triton_kernels.numerics import InFlexData, OutFlexData
from triton_kernels.numerics_details.mxfp import upcast_from_mxfp, quantize_mxfp8_fn, downcast_to_mxfp_torch, upcast_from_mxfp_torch, MXFP_BLOCK_SIZE
# testing utilities
from triton_kernels.testing import assert_close, make_random_tensor
# target-specific utilities
from triton_kernels.target_info import is_hip, is_hip_cdna3, is_cuda, is_hip_cdna4
from triton_kernels.swiglu import swiglu, swiglu_fn
from triton_kernels.swiglu import PrecisionConfig as SwiGLUPrecisionConfig
from triton_kernels.tensor_details import layout

# ---------------
# numerics stuff
# ---------------

class DType:

    def __init__(self, dtype_str):
        self.has_global_scale = dtype_str.startswith("float8")
        self.has_mx_scale = dtype_str.startswith("mx")
        to_torch_dtype = lambda name: torch.uint8 if name == "float4_e2m1" else getattr(torch, name)
        self.torch_dtype = to_torch_dtype(dtype_str.strip("mx"))
        self.is_mxfloat4 = self.has_mx_scale and "float4" in dtype_str



# Scope to ensure that the opt_flags_constraints are reset after the test
@pytest.fixture
def opt_flags_scope(request):
    yield
    opt_flags.reset_opt_flags_constraints()


def make_constraints(block_m, split_k, is_persistent, epilogue_subtile, hbm_swizzling, weight_dtype_str):
    constraints = {
        "block_m": block_m,
        "split_k": split_k,
        "is_persistent": is_persistent,
        "epilogue_subtile": epilogue_subtile,
    }
    if is_hip() and hbm_swizzling and "float4" in weight_dtype_str:
        # Minimum block size to satisfy scale preshuffling
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
    n_slices: int = None
    split_k: int = 1
    a_hbm_swizzling: bool = False
    b_hbm_swizzling: bool = False
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
    test_cases.extend([
        Case(m, n, k, mode, "float16", "float16")
        for mode in ("ragged", "batched")
        for (m, n, k) in ((0, 5, 7), (5, 0, 7), (5, 7, 0))
    ])
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
    # float8 x mxfloat
    test_cases.extend([
        Case(16, 256, 256, "ragged", "float8_e5m2", "mxfloat4_e2m1", b_hbm_swizzling=True),
        Case(1024, 1024, 1024, "batched", "float8_e5m2", "mxfloat4_e2m1", b_hbm_swizzling=True),
        Case(1024, 1024, 1024, "batched", "float8_e5m2", "mxfloat4_e2m1"),
        Case(1024, 1024, 1024, "ragged", "float8_e5m2", "mxfloat4_e2m1", split_k=9),
        Case(1024, 1024, 1024, "ragged", "float8_e5m2", "mxfloat4_e2m1", split_k=9, b_hbm_swizzling=True),
        Case(300, 400, 400, "ragged", "float8_e5m2", "mxfloat8_e4m3fn"),
        Case(300, 400, 832, "ragged", "float8_e5m2", "mxfloat4_e2m1"),
        Case(300, 400, 400, "batched", "float8_e5m2", "mxfloat8_e4m3fn"),
    ])
    # mxfloat x mxfloat
    test_cases.extend([
        Case(16, 256, 256, "ragged", "mxfloat8_e4m3fn", "mxfloat4_e2m1", b_hbm_swizzling=True),
        Case(1024, 1024, 1024, "ragged", "mxfloat8_e4m3fn", "mxfloat4_e2m1", split_k=9, b_hbm_swizzling=True),
        Case(1024, 1024, 1024, "ragged", "mxfloat8_e4m3fn", "mxfloat4_e2m1", split_k=9, colmajor_mxfp_weight=False),
        Case(1000, 704, 800, "batched", "mxfloat8_e4m3fn", "mxfloat4_e2m1", b_hbm_swizzling=True, a_hbm_swizzling=True),
        Case(300, 400, 400, "ragged", "mxfloat8_e4m3fn", "mxfloat8_e4m3fn"),
        Case(300, 400, 400, "ragged", "mxfloat8_e4m3fn", "mxfloat8_e4m3fn", b_hbm_swizzling=True),
        Case(300, 400, 400, "batched", "mxfloat8_e4m3fn", "mxfloat8_e4m3fn"),
        Case(1024, 1024, 1024, "batched", "mxfloat8_e4m3fn", "mxfloat4_e2m1", b_hbm_swizzling=True),
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
        Case(*shape, mode, "mxfloat8_e4m3fn", "mxfloat4_e2m1", b_hbm_swizzling=True, split_k=split_k, swiglu_opts=(1.1, 7))
     for shape in [odd_shape2, even_shape] for mode in ["ragged", "batched"] for split_k in [1, 5]
    ])

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
def test_op(m, n, k, split_k, do_gather, do_scatter, inner_expt_opt, do_gamma, is_persistent, n_slices,
            mode, act_dtype_str, weight_dtype_str, block_m, b_hbm_swizzling, a_hbm_swizzling, colmajor_mxfp_weight, epilogue_subtile,
            a_transpose, b_transpose, c_transpose,
            swiglu_opts, device, opt_flags_scope):
    # We catch and re-invoke pytest.skip(), because otherwise pytest may hold a reference to
    # the frame that called pytest.skip, including all the tensors, leading to OOM.
    skip_message = None
    try:
        _test_op(m, n, k, split_k, do_gather, do_scatter, inner_expt_opt, do_gamma, is_persistent, n_slices,
                 mode, act_dtype_str, weight_dtype_str, block_m, b_hbm_swizzling, a_hbm_swizzling, colmajor_mxfp_weight, epilogue_subtile,
                 a_transpose, b_transpose, c_transpose,
                 swiglu_opts, device, opt_flags_scope)
    except pytest.skip.Exception as e:
        skip_message = str(e)

    if skip_message is not None:
        pytest.skip(skip_message)

def _test_op(m, n, k, split_k, do_gather, do_scatter, inner_expt_opt, do_gamma, is_persistent, n_slices,
            mode, act_dtype_str, weight_dtype_str, block_m, b_hbm_swizzling, a_hbm_swizzling, colmajor_mxfp_weight, epilogue_subtile,
            a_transpose, b_transpose, c_transpose,
            swiglu_opts, device, opt_flags_scope):
    # TODO: remove when Triton FP8 supports proper RTNE
    if is_cuda():
        if "float8" in weight_dtype_str and torch.cuda.get_device_capability()[0] < 9:
            pytest.skip("Float8 not tested on A100")
        if act_dtype_str == "float16" and "mx" in weight_dtype_str and torch.cuda.get_device_capability()[0] >= 10:
            pytest.skip("float16 x mx not supported with cuda capability >= 10")
        if weight_dtype_str.startswith("mx"):
            if "float8" in act_dtype_str and torch.cuda.get_device_capability()[0] < 10:
                pytest.skip("float8 x mx not supported with cuda capability < 10")
        if swiglu_opts is not None and do_gamma:
            pytest.skip("NYI: swiglu and gamma not supported together")

    elif is_hip():
        if "float8" in act_dtype_str and "mx" in weight_dtype_str and not is_hip_cdna4():
            pytest.skip("float8 x mx only supported on CDNA4")
        if "float8" in act_dtype_str and "mxfloat8" in weight_dtype_str:
            pytest.skip("NYI: float8 x mxfloat8 not tested on AMD GPU")
        if act_dtype_str.startswith("mx") and weight_dtype_str.startswith("mx"):
            pytest.skip("NYI: mx x mx not tested on AMD GPU")
        if is_persistent:
            pytest.skip("NYI: Persistent kernel not supported on AMD GPU")
        # FIXME: this works on nvidia; looks like some sort of bug on AMD?
        if do_gamma and swiglu_opts is not None:
            pytest.skip("NYI: gamma and swiglu not supported together on AMD GPU")
        if split_k is not None and split_k > 1:
            pytest.skip("splitK hasn't been fully tested on AMD GPU.")

    if "float8_e4m3fnuz" in (weight_dtype_str, act_dtype_str) and not is_hip_cdna3():
        pytest.skip("float8_e4m3fnuz only tested on AMD CDNA3 Platform")

    if b_hbm_swizzling:
        if is_hip():
            if not is_hip_cdna4():
                pytest.skip("Scale preshuffling on AMD GPU has not been emulated on non-CDNA4 arch yet.")
            if "mx" not in weight_dtype_str:
                pytest.skip("Non-scale swizzling not supported on CDNA4 yet")
        if torch.cuda.get_device_capability()[0] < 9:
            pytest.skip("NYI. Ampere swizzling.")
        if torch.cuda.get_device_capability()[0] < 10:
            if "mxfloat4" not in weight_dtype_str:
                pytest.skip("NYI. Hopper swizzling just implemented for mxfp4.")

    if a_hbm_swizzling:
        # current x scale swizzling requires B200, batched input, mxfloat8 act and is persistent case
        if is_hip():
            pytest.skip("NYI. X swizzling not tested on AMD GPU yet.")
        if torch.cuda.get_device_capability()[0] < 10:
            pytest.skip("NYI. X swizzling only implemented for B200 for now.")
        if mode != "batched":
            pytest.skip("NYI. X swizzling only implemented for batched input for now.")
        if not act_dtype_str.startswith("mxfloat8"):
            pytest.skip(f"NYI. X swizzling only implemented for mxfloat8 act for now. Got {act_dtype_str}")
        if not is_persistent:
            pytest.skip("NYI. X swizzling only implemented for persistent case for now.")
        if block_m < 128:
            pytest.skip("NYI. X swizzling only implemented for block_m = 128 for now.")

    expt_is_inner = (inner_expt_opt is not None)
    if expt_is_inner:
        if mode != "ragged":
            pytest.skip("inner_expt_opt only meaningful with ragged")
        if "mx" in act_dtype_str and inner_expt_opt != "pad_a":
            pytest.skip("inner_expt_opt and act mx only supported with pad_a")
        if "mx" in weight_dtype_str:
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
        if torch.cuda.get_device_capability()[0] < 10:
            pytest.skip("transposed mxfp weight not supported with cuda capability < 10")
        if block_m == 16:
            pytest.skip("PassManager::run failed from Triton compiler")
    # TODO: should construct the test case differently rather than overriding here
    if "float8" in weight_dtype_str and torch.cuda.get_device_capability()[0] < 10:
        b_transpose = True

    torch.manual_seed(0)

    # set opt flags constraints
    constraints = make_constraints(block_m, split_k, is_persistent, epilogue_subtile, b_hbm_swizzling, weight_dtype_str)
    opt_flags.update_opt_flags_constraints(constraints)

    a_dtype = DType(act_dtype_str)
    b_dtype = DType(weight_dtype_str)
    c_dtype = DType(act_dtype_str)

    # --- create conditionals ---
    do_bias = inner_expt_opt is None
    do_gather = do_gather and mode != "batched"
    do_scatter = do_scatter and mode != "batched"

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
        scale_hbm_swizzling = layout.make_default_matmul_mxfp8_act_scale_layout if a_hbm_swizzling else None,
        scale_hbm_swizzling_args = {},
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
        value_hbm_swizzling = layout.make_default_matmul_mxfp4_w_layout if b_hbm_swizzling and colmajor_mxfp_weight and b_dtype.is_mxfloat4 else None,
        value_hbm_swizzling_args = {"mx_axis":-2},
        scale_hbm_swizzling = layout.make_default_matmul_mxfp4_w_scale_layout if b_hbm_swizzling and colmajor_mxfp_weight and b_dtype.is_mxfloat4 else None,
        scale_hbm_swizzling_args = {"mx_axis":-2, "num_warps":8},
    )
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
    c = torch.empty(c_shape, dtype=c_dtype.torch_dtype, device=device)
    if c_transpose:
        c = c.mT.contiguous().mT

    # --- create precision config ---
    wrap_list = lambda vals: torch.tensor(vals, dtype=torch.float32, device=device)
    flex_a = InFlexData(c_dtype.torch_dtype, wrap_list([1.25])) if c_dtype.has_global_scale else InFlexData()
    flex_b = InFlexData(b_dtype.torch_dtype, wrap_list([1.25])) if b_dtype.has_global_scale else InFlexData()
    flex_c = OutFlexData(c_dtype.torch_dtype, wrap_list([4.00]), wrap_list([0]), None) if c_dtype.has_global_scale else OutFlexData()
    precision_opt = PrecisionConfig(
        flex_ctx=FlexCtx(flex_a, flex_b, flex_c),
        acc_scale=2.0 if c_dtype.has_global_scale or b_dtype.has_global_scale else 1.0,
        out_dtype=c_dtype.torch_dtype,
        a_mx_scale=a_scales,
        b_mx_scale=b_scale_tri,
    )

    # --- create epilogue ---
    epilogue = None
    if c_dtype.has_mx_scale:
        c_scale_shape = c_shape[:-1] + (triton.cdiv(c_shape[-1], MXFP_BLOCK_SIZE),)
        c_scale = torch.empty(c_scale_shape, dtype=torch.uint8, device=a.device)
        precision_opt.c_mx_scale = c_scale
        epilogue_spec = FnSpecs(FnName.QUANTIZE_MXFP8.name, quantize_mxfp8_fn, (), ())
        epilogue = Epilogue(epilogue_spec, tuple(), tuple(), effective_itemsize=6.0)


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
        pytest.skip(f"inapplicable opt_flags constraint {e}")
    # --- torch implementation ---
    ref_y = matmul_torch(a, b, bias,  #
                        a_ragged_metadata, b_ragged_metadata,
                        gather_indx, scatter_indx, precision_opt,
                        gammas=gammas)
    if swiglu_opts is not None:
        ref_y = swiglu(ref_y, alpha=swiglu_opts[0], precision_config=SwiGLUPrecisionConfig(swiglu_opts[1]))
    if c_dtype.has_global_scale:
        ref_y_scale = precision_opt.flex_ctx.out_data.actual_scale.clone()

    # --- check results ---
    if c_dtype.has_mx_scale:
        tri_y = upcast_from_mxfp(tri_y, precision_opt.c_mx_scale, target_dtype=torch.bfloat16, axis=-1).to(ref_y.dtype)
        ref_y = upcast_from_mxfp_torch(*downcast_to_mxfp_torch(ref_y, c_dtype.torch_dtype, axis=-1), target_dtype=ref_y.dtype, axis=-1)
    maxtol, rmstol = None, None
    if c_dtype.has_mx_scale:
        maxtol, rmstol = 4e-1, 4e-2
    elif b_dtype.is_mxfloat4:
        maxtol, rmstol = 3e-2, None
    assert_close(ref_y, tri_y, maxtol=maxtol, rmstol=rmstol)
    if c_dtype.has_global_scale:
        assert torch.all((ref_y_scale - tri_y_scale).abs() < 1e-10), \
               f"ref_y_scale: {ref_y_scale}, tri_y_scale: {tri_y_scale.item()}"


def test_set_idle_sms():
    if not is_cuda():
        pytest.skip("Only supported on CUDA")
    from triton_kernels.matmul_details.opt_flags import make_opt_flags
    num_idle_sms = 24
    matmul_set_idle_sms(num_idle_sms)
    flags = make_opt_flags(torch.float32, torch.float32, torch.float32, PrecisionConfig(), \
                           1, 1024, 1024, 1024, None, True, False, 1, False, False, None)
    assert flags.idle_sms == num_idle_sms
