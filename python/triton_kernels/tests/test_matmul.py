# isort: off
# fmt: off
from dataclasses import dataclass, fields, replace
import itertools
import pytest
import torch
from typing import Union
import triton
# matmul utilities
import triton_kernels.matmul_details.opt_flags as opt_flags
from triton_kernels.matmul import FlexCtx, PrecisionConfig, FusedActivation, FnSpecs, FnName, Epilogue
from triton_kernels.matmul import matmul_set_idle_sms, matmul, matmul_torch
from triton_kernels.swiglu import swiglu, swiglu_fn, PrecisionConfig as SwiGLUPrecisionConfig
from triton_kernels.tensor import convert_layout, wrap_torch_tensor, FP4, make_ragged_tensor_metadata
from triton_kernels.tensor_details import layout
# numerics utilities
from triton_kernels.numerics import InFlexData, OutFlexData
from triton_kernels.numerics_details.mxfp import downcast_to_mxfp, upcast_from_mxfp, quantize_mxfp8_fn, downcast_to_mxfp_torch, upcast_from_mxfp_torch, MXFP_BLOCK_SIZE
# testing utilities
from triton_kernels.testing import assert_close, compute_actual_scale
# target-specific utilities
from triton_kernels.target_info import is_hip, is_hip_cdna3, is_cuda, is_hip_cdna4

# ---------------
# initialize data
# ---------------


def alloc_rand(shape, device, dtype, requires_grad=False):
    if dtype.itemsize == 1:
        tmp = 2**-(torch.randint(4, 8, shape, device=device, dtype=torch.float16))
        return tmp.to(dtype).requires_grad_(requires_grad)

    return torch.randn(shape, device=device, dtype=dtype, requires_grad=requires_grad)


def alloc_rand_like(x):
    return alloc_rand(x.shape, x.device, x.dtype, x.requires_grad)

def make_slice_sizes(n_slices, total_size, device="cuda"):
    dtype = torch.int32
    if total_size < 0:
        raise ValueError("total_size must be non-negative")
    if n_slices <= 0:
        return torch.zeros((0,), dtype=dtype, device=device)
    if total_size == 0:
        return torch.zeros((n_slices,), dtype=dtype, device=device)
    # always set one slice size to zero
    probs = torch.ones(n_slices, device=device) / n_slices
    if n_slices > 1:
        probs[2] += probs[1]
        probs[1] = 0.
    assignments = torch.multinomial(probs, total_size, replacement=True)
    counts = torch.bincount(assignments, minlength=n_slices).to(dtype)
    assert counts.sum().item() == total_size
    assert len(counts) == n_slices
    return counts

def pad_rows_to_multiples(A, indices, multiple=128, pad_value=float('nan')):
    """
    Insert padding so that each row A[i] (for i in indices)
    appears at an output row index that is a multiple of `multiple`.
    """
    D = A.size(1)
    out = []
    for i_cur, i_next in zip(indices[:-1], indices[1:]):
        size = (i_next - i_cur)
        size_padded = ((size + multiple - 1) // multiple) * multiple
        cur = torch.full((size_padded, D), pad_value, dtype=A.dtype, device=A.device)
        cur[:size, :] = A[i_cur:i_next, :]
        out.append(cur)
    return torch.vstack(out)


def init_compute_data(m, n, k, n_slices, mode, act_dtype, weight_dtype,
                      has_y_gammas, device="cuda",
                      inner_expt_opt=None):
    torch.manual_seed(0)
    assert mode in {'batched', "plain", 'ragged'}
    shape_a = (n_slices, m, k) if mode == 'batched' else (m, k)
    batch_size = tuple() if (mode == "plain" or inner_expt_opt is not None) else (n_slices, )
    a = alloc_rand(shape_a, device=device, dtype=act_dtype)
    b = alloc_rand(batch_size + (k, n), device=device, dtype=weight_dtype)
    bias = alloc_rand(batch_size + (n, ), device=device, dtype=torch.float32)
    gammas = 2**torch.randint(-5, 0, (m, ), device=device, dtype=torch.float32) if has_y_gammas else None
    return a, b, bias, gammas


# ---------------
# numerics stuff
# ---------------

def dtype_str_to_torch(dtype_str: str) -> torch.dtype:
    dtype_str = dtype_str.strip("mx")
    return torch.uint8 if dtype_str == "float4_e2m1" else getattr(torch, dtype_str)

class DType:

    def __init__(self, dtype_str):
        self.has_global_scale = dtype_str.startswith("float8")
        self.has_mx_scale = dtype_str.startswith("mx")
        self.torch_dtype = dtype_str_to_torch(dtype_str)


def init_precision(out_dtype, act_use_flexpoint, weight_dtype, weight_mxfp, n_slices=1, expt_is_inner=False, device="cuda"):
    weight_use_flexpoint = weight_dtype.itemsize == 1 and not weight_mxfp
    # flexpoint
    make_tensor = lambda val0, val1: torch.tensor([val0, val1] * (n_slices // 2) + ([val0] if n_slices % 2 else []), #
                                                  dtype=torch.float32, device=device)
    make_scalar_scale = lambda val: torch.tensor([val], dtype=torch.float32, device=device)
    make_tensor_scale = lambda val0, val1, is_tensor: make_tensor(val0, val1) if is_tensor else make_scalar_scale(val0)
    flex_ctx = FlexCtx(
        lhs_data= InFlexData(
            dtype=out_dtype,
            scale=make_scalar_scale(1.25)
        ) if act_use_flexpoint else InFlexData(),
        rhs_data=InFlexData(
            dtype=weight_dtype,
            scale=make_tensor_scale(1.50, 1.25, not expt_is_inner),
        ) if weight_use_flexpoint else InFlexData(),
        out_data=OutFlexData(
            dtype=out_dtype,
            expected_scale=make_scalar_scale(4.00),
            actual_scale=make_scalar_scale(0),
            checksum_scale=None,
        ) if act_use_flexpoint else OutFlexData(),
    )
    return PrecisionConfig(flex_ctx=flex_ctx, acc_scale=2.0 if act_use_flexpoint or weight_use_flexpoint else 1.0,
                           out_dtype=out_dtype)


def apply_precision(x_tri, w_tri, x_ref, w_ref, bias_tri, gammas_tri, precision_config):
    flex_ctx = precision_config.flex_ctx

    def apply(x, scale):
        if scale is None:
            x = x.clone()
        elif scale.numel() == 1:
            x = x.float() * scale
        else:
            assert x.ndim == 3
            assert scale.numel() == x.shape[0]
            x = x.float() * scale[:, None, None]
        return x

    return (
        x_ref if x_ref is not None else apply(x_tri, flex_ctx.lhs_data.scale),
        w_ref if w_ref is not None else apply(w_tri, flex_ctx.rhs_data.scale),
        None if bias_tri is None else apply(bias_tri, None),
        None if gammas_tri is None else apply(gammas_tri, None),
    )




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

def init_ragged_metadata(n_slices, slice_dim, device):
    slice_sizes = make_slice_sizes(n_slices, slice_dim, device=device)
    return make_ragged_tensor_metadata(slice_sizes, slice_dim)

def convert_to_mxfp(x, x_dtype, block_m, hbm_swizzling, is_mxfp4, is_colmajor):
    mx_axis = x.ndim - 2
    # compute layouts
    x_layout, x_layout_opts = layout.StridedLayout, dict()
    x_scale_layout, x_scale_layout_opts = layout.StridedLayout, dict()
    if hbm_swizzling and is_mxfp4:
        x_layout, x_layout_opts = layout.make_default_matmul_mxfp4_w_layout(mx_axis=mx_axis)
        x_scale_layout, x_scale_layout_opts = layout.make_default_matmul_mxfp4_w_scale_layout(
            mx_axis=mx_axis, num_warps=8)
    # downcast to mxfp
    if is_colmajor:
        # create storage
        x, x_scale_tri = downcast_to_mxfp(x, x_dtype, axis=mx_axis)
        x_tri_dtype = FP4 if is_mxfp4 else x_dtype
        x_ref = upcast_from_mxfp(x, x_scale_tri, torch.bfloat16, axis=mx_axis)
        # create tensor object
        x = convert_layout(wrap_torch_tensor(x, x_tri_dtype), x_layout, **x_layout_opts)
        x_scale_tri = convert_layout(wrap_torch_tensor(x_scale_tri), x_scale_layout, **x_scale_layout_opts)
    else:
        if torch.cuda.get_device_capability()[0] < 10:
            pytest.skip("transposed mxfp weight not supported with cuda capability < 10")
        if block_m == 16:
            pytest.skip("PassManager::run failed from Triton compiler")
        # TODO: swizzling for rowmajor

        # A typical use case is we already quantized col-major weight,
        # and we want matmul with its transposed row-major weight w/o
        # requantization.

        # put abs_max of each 32x32 block to diagonal so scales of transposed agree
        x_ndim = x.ndim
        if x_ndim == 2:
            x = x.unsqueeze(0)
        BLOCK_SIZE = int(MXFP_BLOCK_SIZE)
        for e, i, j in itertools.product(range(x.shape[0]), range(0, x.shape[1], BLOCK_SIZE), range(0, x.shape[2], BLOCK_SIZE)):
            i_end = min(i+BLOCK_SIZE, x.shape[1])
            j_end = min(j+BLOCK_SIZE, x.shape[2])
            block = x[e, i:i_end, j:j_end]
            m_abs = block.abs().max()
            i_len = i_end - i
            j_len = j_end - j
            min_len = min(i_len, j_len)
            signs = torch.randint(0, 2, (max(i_len, j_len),), device=x.device) * 2 - 1
            block.diagonal(dim1=-2, dim2=-1)[:] = signs[:min_len] * m_abs
            if j_len > i_len:
                block[i_len - 1, i_len:] = signs[min_len:] * m_abs
            elif i_len > j_len:
                block[j_len:, j_len - 1] = signs[min_len:] * m_abs
        if x_ndim == 2:
            x = x.squeeze(0)

        # matmul with rowmajor weight expects scale is separately
        # constructed (not much additional memory needed).
        _, x_scale_tri = downcast_to_mxfp(x, x_dtype, axis=mx_axis)
        # reuse quantized value from colmajor
        wT_tri, wT_scale_tri = downcast_to_mxfp(x.mT.contiguous(), x_dtype, axis=mx_axis)
        x_ref = upcast_from_mxfp(wT_tri, wT_scale_tri, torch.bfloat16, axis=mx_axis).mT.contiguous()
        x_tri_dtype = FP4 if is_mxfp4 else x_dtype
        x = wrap_torch_tensor(wT_tri.data.mT, x_tri_dtype)

    return x, x_scale_tri, x_ref

def compute_y_shape(mode, gather_indx, scatter_indx, expt_is_inner, n_slices, m, n, x_tri):
    if mode == "batched":
        return (x_tri.shape[0], x_tri.shape[-2], n)
    elif scatter_indx is not None:
        return (scatter_indx.shape[0], n)
    elif expt_is_inner:
        return (n_slices, m, n)
    n_rows = x_tri.shape[-2] if gather_indx is None else gather_indx.shape[0]
    return (n_rows, n)

def pad_ragged_tensor(x, x_ragged_metadata, transpose):
    multiple = 128
    if transpose:
        y = pad_rows_to_multiples(x.T, x_ragged_metadata.slice_offs, multiple=multiple, pad_value=0).T.contiguous()
    else:
        y = pad_rows_to_multiples(x, x_ragged_metadata.slice_offs, multiple=multiple, pad_value=0).contiguous()
    y_ragged_metadata = replace(x_ragged_metadata,
                                slice_offs=x_ragged_metadata.block_offs(multiple) * multiple,
                                slice_sizes_divisibility=multiple)
    return y, y_ragged_metadata

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
    hbm_swizzling: bool = False
    epilogue_subtile: Union[int, None] = None
    a_transpose: bool = False
    b_transpose: bool = False
    c_transpose: bool = False
    colmajor_mxfp_weight: bool = True

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
            Case(*shape, "plain", "bfloat16", "mxfloat4_e2m1", hbm_swizzling=True, epilogue_subtile=4),
            Case(*shape, "batched", "bfloat16", "mxfloat4_e2m1"),
            Case(*shape, "batched", "bfloat16", "mxfloat4_e2m1", hbm_swizzling=True),
            Case(*shape, "ragged", "bfloat16", "mxfloat4_e2m1"),
            Case(*shape, "ragged", "bfloat16", "mxfloat4_e2m1", hbm_swizzling=True),
            Case(*shape, "ragged", "bfloat16", "mxfloat4_e2m1", split_k=9),
            Case(*shape, "ragged", "bfloat16", "mxfloat4_e2m1", split_k=9, hbm_swizzling=True),
            Case(*shape, "ragged", "bfloat16", "mxfloat8_e4m3fn"),
            Case(*shape, "ragged", "bfloat16", "mxfloat8_e4m3fn", hbm_swizzling=True)
        ])
    # float8 x mxfloat
    test_cases.extend([
        Case(16, 256, 256, "ragged", "float8_e5m2", "mxfloat4_e2m1", hbm_swizzling=True),
        Case(1024, 1024, 1024, "batched", "float8_e5m2", "mxfloat4_e2m1", hbm_swizzling=True),
        Case(1024, 1024, 1024, "batched", "float8_e5m2", "mxfloat4_e2m1"),
        Case(1024, 1024, 1024, "ragged", "float8_e5m2", "mxfloat4_e2m1", split_k=9),
        Case(1024, 1024, 1024, "ragged", "float8_e5m2", "mxfloat4_e2m1", split_k=9, hbm_swizzling=True),
        Case(300, 400, 400, "ragged", "float8_e5m2", "mxfloat8_e4m3fn"),
        Case(300, 400, 832, "ragged", "float8_e5m2", "mxfloat4_e2m1"),
        Case(300, 400, 400, "batched", "float8_e5m2", "mxfloat8_e4m3fn"),
    ])
    # mxfloat x mxfloat
    test_cases.extend([
        Case(16, 256, 256, "ragged", "mxfloat8_e4m3fn", "mxfloat4_e2m1", hbm_swizzling=True),
        Case(1024, 1024, 1024, "ragged", "mxfloat8_e4m3fn", "mxfloat4_e2m1", split_k=9, hbm_swizzling=True),
        Case(1024, 1024, 1024, "ragged", "mxfloat8_e4m3fn", "mxfloat4_e2m1", split_k=9, colmajor_mxfp_weight=False),
        Case(300, 400, 400, "ragged", "mxfloat8_e4m3fn", "mxfloat8_e4m3fn"),
        Case(300, 400, 400, "ragged", "mxfloat8_e4m3fn", "mxfloat8_e4m3fn", hbm_swizzling=True),
        Case(300, 400, 400, "batched", "mxfloat8_e4m3fn", "mxfloat8_e4m3fn"),
        Case(1024, 1024, 1024, "batched", "mxfloat8_e4m3fn", "mxfloat4_e2m1", hbm_swizzling=True),
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
@pytest.mark.parametrize("has_y_gammas", [False, True])
@pytest.mark.parametrize("is_persistent", [False, True])
def test_op(m, n, k, split_k, do_gather, do_scatter, inner_expt_opt, has_y_gammas, is_persistent, n_slices,
            mode, act_dtype_str, weight_dtype_str, block_m, hbm_swizzling, colmajor_mxfp_weight, epilogue_subtile,
            a_transpose, b_transpose, c_transpose,
            device, opt_flags_scope):
    # We catch and re-invoke pytest.skip(), because otherwise pytest may hold a reference to
    # the frame that called pytest.skip, including all the tensors, leading to OOM.
    skip_message = None
    try:
        _test_op(m, n, k, split_k, do_gather, do_scatter, inner_expt_opt, has_y_gammas, is_persistent, n_slices,
                 mode, act_dtype_str, weight_dtype_str, block_m, hbm_swizzling, colmajor_mxfp_weight, epilogue_subtile,
                 a_transpose, b_transpose, c_transpose,
                 device, opt_flags_scope)
    except pytest.skip.Exception as e:
        skip_message = str(e)

    if skip_message is not None:
        pytest.skip(skip_message)

def _test_op(m, n, k, split_k, do_gather, do_scatter, inner_expt_opt, has_y_gammas, is_persistent, n_slices,
            mode, act_dtype_str, weight_dtype_str, block_m, hbm_swizzling, colmajor_mxfp_weight, epilogue_subtile,
            a_transpose, b_transpose, c_transpose,
            device, opt_flags_scope):
    # TODO: remove when Triton FP8 supports proper RTNE
    if is_cuda():
        if "float8" in weight_dtype_str and torch.cuda.get_device_capability()[0] < 9:
            pytest.skip("Float8 not tested on A100")
        if act_dtype_str == "float16" and "mx" in weight_dtype_str and torch.cuda.get_device_capability()[0] >= 10:
            pytest.skip("float16 x mx not supported with cuda capability >= 10")
        if weight_dtype_str.startswith("mx"):
            if "float8" in act_dtype_str and torch.cuda.get_device_capability()[0] < 10:
                pytest.skip("float8 x mx not supported with cuda capability < 10")

    elif is_hip():
        if "float8" in act_dtype_str and "mx" in weight_dtype_str and not is_hip_cdna4():
            pytest.skip("float8 x mx only supported on CDNA4")
        if "float8" in act_dtype_str and "mxfloat8" in weight_dtype_str:
            pytest.skip("NYI: float8 x mxfloat8 not tested on AMD GPU")
        if act_dtype_str.startswith("mx") and weight_dtype_str.startswith("mx"):
            pytest.skip("NYI: mx x mx not tested on AMD GPU")
        if is_persistent:
            pytest.skip("NYI: Persistent kernel not supported on AMD GPU")
        if split_k is not None and split_k > 1:
            pytest.skip("splitK hasn't been fully tested on AMD GPU.")

    if "float8_e4m3fnuz" in (weight_dtype_str, act_dtype_str) and not is_hip_cdna3():
        pytest.skip("float8_e4m3fnuz only tested on AMD CDNA3 Platform")

    if hbm_swizzling:
        if is_hip():
            if not is_hip_cdna4():
                pytest.skip("Scale preshuffling on AMD GPU has not been emulated on non-CDNA4 arch yet.")
            if "mx" not in weight_dtype_str:
                pytest.skip("Non-scale swizzling not supported on CDNA4 yet")
            if n % 32 != 0 or k % (32 * 8) != 0:
                pytest.skip(f"Shape {m}x{n}x{k} is not supported for scale swizzling on AMD GPU")
        if torch.cuda.get_device_capability()[0] < 9:
            pytest.skip("NYI. Ampere swizzling.")
        if torch.cuda.get_device_capability()[0] < 10:
            if "mxfloat4" not in weight_dtype_str:
                pytest.skip("NYI. Hopper swizzling just implemented for mxfp4.")
            if k % 64 != 0 or n % 64 != 0:
                # Automatic padding not implemented for Hopper swizzle
                pytest.skip("Hopper swizzling acts on a 64x64 tile (4x1 mma tiles).")

    expt_is_inner = (inner_expt_opt is not None)
    if expt_is_inner:
        if mode != "ragged":
            pytest.skip("inner_expt_opt only meaningful with ragged")
        if "mx" in act_dtype_str and inner_expt_opt != "pad_a":
            pytest.skip("inner_expt_opt and act mx only supported with pad_a")
        if "mx" in weight_dtype_str:
            if inner_expt_opt != "pad_b":
                pytest.skip("inner_expt_opt and weight mx only supported with pad_b")
            if is_persistent and not hbm_swizzling:
                pytest.skip("FIXME: Fatal Python error: Aborted")
            if is_hip():
                if act_dtype_str == "bfloat16":
                    pytest.skip("FIXME: failed to translate module to LLVM IR")
                if hbm_swizzling:
                    pytest.skip("NYI: nner_expt_opt and HBM swizzling")
    # TODO: should construct the test case differently rather than overriding here
    if "float8" in weight_dtype_str and torch.cuda.get_device_capability()[0] < 10:
        b_transpose = True

    torch.manual_seed(0)
    # set opt flags constraints
    constraints = make_constraints(block_m, split_k, is_persistent, epilogue_subtile, hbm_swizzling, weight_dtype_str)
    opt_flags.update_opt_flags_constraints(constraints)

    weight_mxfp = weight_dtype_str.startswith("mx")
    weight_mxfp4 = weight_mxfp and "float4" in weight_dtype_str
    act_mxfp8 = act_dtype_str.startswith("mx")
    act_is_float8 = act_dtype_str.startswith("float8")
    a_mxfp8 = act_mxfp8
    c_mxfp8 = act_mxfp8
    b_mxfp = weight_mxfp

    weight_dtype = dtype_str_to_torch(weight_dtype_str)
    act_dtype = dtype_str_to_torch(act_dtype_str)
    precision_opt = init_precision(act_dtype, act_is_float8, weight_dtype, weight_mxfp,
                                   n_slices, expt_is_inner, device=device)

    gather_indx = None if not do_gather else torch.randint(0, max(m, 1), (m, ), device=device, dtype=torch.int32)
    scatter_indx = None if not do_scatter else torch.randperm(m, device=device, dtype=torch.int32)
    a_ragged_metadata = None if mode != "ragged" else init_ragged_metadata(n_slices, m if not expt_is_inner else k, device)
    b_ragged_metadata = None if not expt_is_inner else replace(a_ragged_metadata)


    a_tri, b_tri, bias_tri, gammas_tri = init_compute_data(m, n, k, n_slices,
                                                           mode, torch.bfloat16 if act_mxfp8 else act_dtype,  #
                                                           torch.bfloat16 if weight_mxfp else weight_dtype,
                                                           has_y_gammas, device=device,
                                                           inner_expt_opt=inner_expt_opt)
    if inner_expt_opt is not None:
        bias_tri = None

    a_ref = None
    if inner_expt_opt is not None and "pad_a" in inner_expt_opt:
        a_tri, a_ragged_metadata = pad_ragged_tensor(a_tri, a_ragged_metadata, transpose=True)
    if a_transpose:
        a_tri = a_tri.mT.contiguous().mT
    if a_mxfp8:
        a_tri, a_mx_scales_tri = downcast_to_mxfp(a_tri, act_dtype, axis=-1)
        a_ref = upcast_from_mxfp(a_tri, a_mx_scales_tri, torch.bfloat16, axis=-1)
        precision_opt.a_mx_scale = a_mx_scales_tri


    b_ref = None
    if inner_expt_opt is not None and "pad_b" in inner_expt_opt:
        b_tri, b_ragged_metadata = pad_ragged_tensor(b_tri, b_ragged_metadata, transpose=False)
    if b_transpose:
        b_tri = b_tri.mT.contiguous().mT
    if b_mxfp:
        b_tri, b_scale_tri, b_ref = convert_to_mxfp(b_tri, weight_dtype, block_m, hbm_swizzling, weight_mxfp4, colmajor_mxfp_weight)
        precision_opt.b_mx_scale = b_scale_tri


    c_shape = compute_y_shape(mode, gather_indx, scatter_indx, expt_is_inner, n_slices, m, n, a_tri)
    c_tri = torch.empty(c_shape, dtype=act_dtype, device=device)
    if c_transpose:
        c_tri = c_tri.mT.contiguous().mT


    a_ref, b_ref, bias_ref, gammas_ref = apply_precision(a_tri, b_tri, a_ref, b_ref, bias_tri, gammas_tri, precision_opt)


    epilogue = None
    if c_mxfp8:
        c_scale_shape = c_shape[:-1] + (triton.cdiv(c_shape[-1], MXFP_BLOCK_SIZE),)
        c_scale = torch.empty(c_scale_shape, dtype=torch.uint8, device=a_tri.device)
        precision_opt.c_mx_scale = c_scale
        epilogue_spec = FnSpecs(FnName.QUANTIZE_MXFP8.name, quantize_mxfp8_fn, (), ())
        epilogue = Epilogue(epilogue_spec, tuple(), tuple(), effective_itemsize=6.0)

    if mode == "batched":
        a_ragged_metadata, gather_indx, scatter_indx = None, None, None

    # triton
    try:
        tri_y = matmul(a_tri, b_tri, bias_tri,
                           a_ragged_metadata, b_ragged_metadata,
                           gather_indx, scatter_indx, precision_opt,
                           gammas=gammas_ref, epilogue=epilogue, c=c_tri)
    except (opt_flags.InapplicableConstraint, NotImplementedError) as e:
        pytest.skip(f"inapplicable opt_flags constraint {e}")
    ref_y = matmul_torch(a_ref, b_ref, bias_ref,  #
                             a_ragged_metadata=a_ragged_metadata,
                             b_ragged_metadata=b_ragged_metadata,
                             gather_indx=gather_indx,
                             scatter_indx=scatter_indx,
                             gammas=gammas_ref)

    def scale(val, scal):
        if scal is None:
            return val
        elif scal.numel() == 1:
            return val / scal
        else:
            assert val.ndim == 3
            return val / scal[:, None, None]

    # check that y_tri_in was used if provided
    if c_tri is not None:
        assert tri_y.data_ptr() == c_tri.data_ptr()
        assert tri_y.shape == c_tri.shape
        assert tri_y.stride() == c_tri.stride()
    if act_mxfp8:
        tri_y = upcast_from_mxfp(tri_y, precision_opt.c_mx_scale, target_dtype=torch.bfloat16, axis=-1).to(ref_y.dtype)
        ref_y_quant, ref_y_scale = downcast_to_mxfp_torch(ref_y, act_dtype, axis=-1)
        ref_y = upcast_from_mxfp_torch(ref_y_quant, ref_y_scale, target_dtype=ref_y.dtype, axis=-1)
        maxtol = 4e-1
        rmstol = 4e-2
    elif weight_mxfp4:
        if act_is_float8:
            maxtol = 8e-2
        else:
            maxtol = 3e-2
        rmstol = None
    else:
        maxtol = None
        rmstol = None

    flex = precision_opt.flex_ctx
    assert_close(scale(ref_y, flex.out_data.expected_scale), tri_y, maxtol=maxtol, rmstol=rmstol)

    if act_is_float8:
        tri_y_scale = flex.out_data.actual_scale.clone()
        ref_y_scale = compute_actual_scale(ref_y, tri_y.dtype, tri_y_scale.numel() > 1)
        assert torch.all((ref_y_scale - tri_y_scale).abs() < 1e-10), \
               f"ref_y_scale: {ref_y_scale}, tri_y_scale: {tri_y_scale.item()}"


# Test that we don't use unsupported block sizes.
@pytest.mark.parametrize("m", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("n", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("k", [8, 16, 32, 64, 128])
def test_small_batch_matmul(m, n, k):
    if is_hip():
        pytest.skip("Not fully tested on AMD")

    if m * n * k > 16384:
        pytest.skip()

    BATCH_SIZE = 10000

    def _make_tensor(shape, dtype, trans):
        if trans:
            shape = (shape[0], shape[2], shape[1])
        t = alloc_rand(shape, "cuda", dtype)
        return t.transpose(1, 2) if trans else t

    for a_transpose, b_transpose, bias, dtype in itertools.product(
        (False, True),
        (False, True),
        (False, True),
        (torch.float16, torch.bfloat16, torch.float8_e5m2),
    ):
        if (
            torch.cuda.get_device_capability()[0] < 10
            and dtype is torch.float8_e5m2
            and (not b_transpose)
        ):
            continue  # Not supported

        x = _make_tensor((BATCH_SIZE, m, k), dtype, a_transpose)
        w = _make_tensor((BATCH_SIZE, k, n), dtype, b_transpose)
        bias = _make_tensor((BATCH_SIZE, n), torch.float32, False) if bias else None
        tri_y = matmul(x, w, bias)

        # ref_y = matmul_torch(x.float(), w.float(), bias)

        # This is faster than matmul_torch.
        ref_y = torch.bmm(x.float(), w.float())
        if bias is not None:
            ref_y += bias[:, None, :]

        assert_close(
            ref_y,
            tri_y,
            maxtol=4e-1 if dtype is torch.float8_e5m2 else None,
            rmstol=4e-2 if dtype is torch.float8_e5m2 else None,
        )


def test_set_idle_sms():
    if not is_cuda():
        pytest.skip("Only supported on CUDA")
    from triton_kernels.matmul_details.opt_flags import make_opt_flags
    num_idle_sms = 24
    matmul_set_idle_sms(num_idle_sms)
    flags = make_opt_flags(torch.float32, torch.float32, torch.float32, PrecisionConfig(), \
                           1, 1024, 1024, 1024, None, True, False, 1, False, False, None)
    assert flags.idle_sms == num_idle_sms


@pytest.mark.parametrize("m, n, k, mode", [
    (1200, 704, 608, "ragged"),
    (800, 800, 400, "batched"),
])
@pytest.mark.parametrize("split_k", [1, 2])
@pytest.mark.parametrize("do_gather, do_scatter", [
    (False, False),
    (True, False),
    (False, True),
    (True, True),
])
@pytest.mark.parametrize("is_persistent, epilogue_subtile", [
    (False, None),
    (True, 1),
    (True, 4),
])
@pytest.mark.parametrize("swiglu_alpha, swiglu_limit", [
    (1.1, 1.4),
    (1.0, 1.2),
    (0.7, 1.0),
])
def test_fused_act(m, n, k, mode, split_k, do_gather, do_scatter, is_persistent, epilogue_subtile,
                   swiglu_alpha, swiglu_limit, device, opt_flags_scope):
    torch.manual_seed(0)
    constraints = {
        "is_persistent": is_persistent,
        "epilogue_subtile": epilogue_subtile,
        "split_k": split_k,
    }
    n_slices = 1
    opt_flags.update_opt_flags_constraints(constraints)

    weight_dtype, act_dtype = torch.float16, torch.float16
    if mode == "ragged" and do_gather:
        gindx = torch.randint(0, m, (m, ), device=device).to(torch.int32) if do_gather and m > 0 else None
    if mode == "ragged" and do_scatter:
        sindx = torch.randperm(m, device=device).to(torch.int32) if do_scatter else None
    if mode == "ragged":
        slice_dim = m
        slice_sizes = make_slice_sizes(n_slices, slice_dim, device=device)
        x_ragged_metadata = make_ragged_tensor_metadata(slice_sizes, slice_dim)

    precision_opt = init_precision(act_dtype, str(act_dtype).startswith("torch.float8"), weight_dtype, False, mode, n_slices, device=device)
    x, w, bias, _, _ = init_compute_data(m, n, k, x_ragged_metadata, gindx, sindx, n_slices, mode,
                                         act_dtype, weight_dtype, False, device=device)

    if mode == "batched":
        x_ragged_metadata, gindx, sindx = None, None, None

    try:
        a = swiglu(matmul(x, w, bias, x_ragged_metadata, gindx, sindx, precision_opt), swiglu_alpha,
                   precision_config=SwiGLUPrecisionConfig(swiglu_limit))
        b = matmul(
            x, w, bias, x_ragged_metadata, gindx, sindx, precision_opt,
            fused_activation=FusedActivation(FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2),
                                             (swiglu_alpha, swiglu_limit)))
    except opt_flags.InapplicableConstraint:
        pytest.skip("inapplicable constraint")

    assert_close(a, b)


@pytest.mark.parametrize("m, n, k", [
    (320, 2**19, 0),
    (4096, 4096, 0),
])
@pytest.mark.parametrize("view_x_as_zero_cols", [False, True])
def test_zero_reduction_dim(m, n, k, view_x_as_zero_cols):
    torch.manual_seed(0)

    if view_x_as_zero_cols:
        x = torch.randn(m, m, device="cuda", dtype=torch.bfloat16)
        x = x[:0, :].transpose(-1, -2)
    else:
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.float32)

    try:
        tri_y = matmul(x, w, bias)
    except opt_flags.InapplicableConstraint:
        pytest.skip("inapplicable constraint")
    ref_y = matmul_torch(x, w, bias, round_x=lambda x, idx: x, round_y=lambda y: y)

    assert_close(ref_y, tri_y)
