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
from triton_kernels.matmul import FlexCtx, PrecisionConfig, FnSpecs, FnName, Epilogue
from triton_kernels.matmul import matmul_set_idle_sms, matmul, matmul_torch
from triton_kernels.tensor import convert_layout, wrap_torch_tensor, FP4, make_ragged_tensor_metadata
from triton_kernels.tensor_details import layout
# numerics utilities
from triton_kernels.numerics import InFlexData, OutFlexData
from triton_kernels.numerics_details.mxfp import downcast_to_mxfp, upcast_from_mxfp, quantize_mxfp8_fn, downcast_to_mxfp_torch, upcast_from_mxfp_torch, MXFP_BLOCK_SIZE
# testing utilities
from triton_kernels.testing import assert_close
# target-specific utilities
from triton_kernels.target_info import is_hip, is_hip_cdna3, is_cuda, is_hip_cdna4

# ---------------
# initialize data
# ---------------


def normalize_blocks(x, BLOCK_SIZE = None):
    if BLOCK_SIZE is None:
        BLOCK_SIZE = int(MXFP_BLOCK_SIZE)
    x_ndim = x.ndim
    if x_ndim == 2:
        x = x.unsqueeze(0)
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
    return x

def alloc_rand(shape, device, dtype, requires_grad=False):
    if dtype.itemsize == 1:
        tmp = 2**-(torch.randint(4, 8, shape, device=device, dtype=torch.float16))
        return tmp.to(dtype).requires_grad_(requires_grad)
    ret = torch.randn(shape, device=device, dtype=dtype, requires_grad=requires_grad)
    ret = normalize_blocks(ret)
    return ret


def alloc_rand_like(x):
    return alloc_rand(x.shape, x.device, x.dtype, x.requires_grad)

def make_slice_sizes(n_slices, total_size, device="cuda"):
    torch.manual_seed(0)
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
        self.torch_dtype = dtype_str_to_torch(dtype_str.strip("mx"))
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

def make_random_tensor(shape, n_slices, ragged_dim, ragged_padding, device, dtype, mxfp_dim, transpose, squeeze_batch_dim, hbm_swizzling=False, is_mx_rowmajor=False):
    # allocate buffer
    buffer_shape = ((n_slices,) if ragged_dim is None else tuple()) + shape
    buffer_dtype = torch.bfloat16 if dtype.has_mx_scale else dtype.torch_dtype
    buffer = alloc_rand(buffer_shape, device=device, dtype=buffer_dtype)
    if squeeze_batch_dim:
        buffer = buffer.squeeze(0)
    # handle raggedness
    ragged_metadata = None
    if ragged_dim is not None:
        slice_sizes = make_slice_sizes(n_slices, shape[ragged_dim], device=device)
        ragged_metadata = make_ragged_tensor_metadata(slice_sizes, shape[ragged_dim])
    if ragged_padding:
        buffer, ragged_metadata = pad_ragged_tensor(buffer, ragged_metadata, ragged_dim==1)
    # handle transpose
    if transpose:
        buffer = buffer.mT.contiguous().mT
    # handle mxfp
    scales = None
    if mxfp_dim is not None:
        assert dtype.has_mx_scale
        buffer_dtype = dtype.torch_dtype
        if is_mx_rowmajor:
            scales = downcast_to_mxfp(buffer, buffer_dtype, axis=mxfp_dim)[1]
            buffer = downcast_to_mxfp(buffer.mT.contiguous(), buffer_dtype, axis=mxfp_dim)[0].mT
        else:
            buffer, scales = downcast_to_mxfp(buffer, buffer_dtype, axis=mxfp_dim)
        buffer = wrap_torch_tensor(buffer, FP4 if dtype.is_mxfloat4 else buffer_dtype)
        scales = wrap_torch_tensor(scales)
        if dtype.is_mxfloat4 and hbm_swizzling and not is_mx_rowmajor:
            # convert buffer to swizzled hbm layout
            buffer_layout, buffer_layout_opts = layout.make_default_matmul_mxfp4_w_layout(mx_axis=mxfp_dim)
            buffer = convert_layout(buffer, buffer_layout, **buffer_layout_opts)
            # convert scales to swizzled hbm layout
            scale_layout, scale_layout_opts = layout.make_default_matmul_mxfp4_w_scale_layout(mx_axis=mxfp_dim, num_warps=8)
            scales = convert_layout(scales, scale_layout, **scale_layout_opts)
    return buffer, scales, ragged_metadata


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
@pytest.mark.parametrize("do_gamma", [False, True])
@pytest.mark.parametrize("is_persistent", [False, True])
def test_op(m, n, k, split_k, do_gather, do_scatter, inner_expt_opt, do_gamma, is_persistent, n_slices,
            mode, act_dtype_str, weight_dtype_str, block_m, hbm_swizzling, colmajor_mxfp_weight, epilogue_subtile,
            a_transpose, b_transpose, c_transpose,
            device, opt_flags_scope):
    # We catch and re-invoke pytest.skip(), because otherwise pytest may hold a reference to
    # the frame that called pytest.skip, including all the tensors, leading to OOM.
    skip_message = None
    try:
        _test_op(m, n, k, split_k, do_gather, do_scatter, inner_expt_opt, do_gamma, is_persistent, n_slices,
                 mode, act_dtype_str, weight_dtype_str, block_m, hbm_swizzling, colmajor_mxfp_weight, epilogue_subtile,
                 a_transpose, b_transpose, c_transpose,
                 device, opt_flags_scope)
    except pytest.skip.Exception as e:
        skip_message = str(e)

    if skip_message is not None:
        pytest.skip(skip_message)

def _test_op(m, n, k, split_k, do_gather, do_scatter, inner_expt_opt, do_gamma, is_persistent, n_slices,
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
    constraints = make_constraints(block_m, split_k, is_persistent, epilogue_subtile, hbm_swizzling, weight_dtype_str)
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
        hbm_swizzling = hbm_swizzling,
        is_mx_rowmajor = not colmajor_mxfp_weight,
    )
    gather_indx  = None if not do_gather  else torch.randint(0, max(m, 1), (m, ), dtype=torch.int32, device=device)
    scatter_indx = None if not do_scatter else torch.randperm(m, dtype=torch.int32, device=device)
    bias         = None if not do_bias    else torch.randn(b.shape[:-2] + b.shape[-1:], dtype=torch.float32, device=device)
    gammas       = None if not do_gamma   else 2**torch.randint(-5, 0, (m, ), dtype=torch.float32, device=device)

    # --- initialize output ---
    c_shape = (n_slices,) if mode == "batched" or inner_expt_opt is not None else tuple() # batch dim
    c_shape += (scatter_indx.shape[0] if do_scatter else a.shape[-2],) # row dim
    c_shape += (b.shape[-1],) # col dim
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

    # triton
    try:
        tri_y = matmul(a, b, bias,
                           a_ragged_metadata, b_ragged_metadata,
                           gather_indx, scatter_indx, precision_opt,
                           gammas=gammas, epilogue=epilogue, c=c)
        if c_dtype.has_global_scale:
            tri_y_scale = precision_opt.flex_ctx.out_data.actual_scale.clone()
    except (opt_flags.InapplicableConstraint, NotImplementedError) as e:
        pytest.skip(f"inapplicable opt_flags constraint {e}")
    ref_y = matmul_torch(a, b, bias,  #
                        a_ragged_metadata, b_ragged_metadata,
                        gather_indx, scatter_indx, precision_opt,
                        gammas=gammas)
    if c_dtype.has_global_scale:
        ref_y_scale = precision_opt.flex_ctx.out_data.actual_scale.clone()

    # check that y_tri_in was used if provided
    if c is not None:
        assert tri_y.data_ptr() == c.data_ptr()
        assert tri_y.shape == c.shape
        assert tri_y.stride() == c.stride()
    if c_dtype.has_mx_scale:
        tri_y = upcast_from_mxfp(tri_y, precision_opt.c_mx_scale, target_dtype=torch.bfloat16, axis=-1).to(ref_y.dtype)
        ref_y_quant, ref_y_scale = downcast_to_mxfp_torch(ref_y, c_dtype.torch_dtype, axis=-1)
        ref_y = upcast_from_mxfp_torch(ref_y_quant, ref_y_scale, target_dtype=ref_y.dtype, axis=-1)
        maxtol = 4e-1
        rmstol = 4e-2
    elif b_dtype.is_mxfloat4:
        maxtol = 3e-2
        rmstol = None
    else:
        maxtol = None
        rmstol = None

    assert_close(ref_y, tri_y, maxtol=maxtol, rmstol=rmstol)
    if c_dtype.has_global_scale:
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


# @pytest.mark.parametrize("m, n, k, mode", [
#     (1200, 704, 608, "ragged"),
#     (800, 800, 400, "batched"),
# ])
# @pytest.mark.parametrize("split_k", [1, 2])
# @pytest.mark.parametrize("do_gather, do_scatter", [
#     (False, False),
#     (True, False),
#     (False, True),
#     (True, True),
# ])
# @pytest.mark.parametrize("is_persistent, epilogue_subtile", [
#     (False, None),
#     (True, 1),
#     (True, 4),
# ])
# @pytest.mark.parametrize("swiglu_alpha, swiglu_limit", [
#     (1.1, 1.4),
#     (1.0, 1.2),
#     (0.7, 1.0),
# ])
# def test_fused_act(m, n, k, mode, split_k, do_gather, do_scatter, is_persistent, epilogue_subtile,
#                    swiglu_alpha, swiglu_limit, device, opt_flags_scope):
#     torch.manual_seed(0)
#     constraints = {
#         "is_persistent": is_persistent,
#         "epilogue_subtile": epilogue_subtile,
#         "split_k": split_k,
#     }
#     n_slices = 1
#     opt_flags.update_opt_flags_constraints(constraints)

#     weight_dtype, act_dtype = torch.float16, torch.float16
#     if mode == "ragged" and do_gather:
#         gindx = torch.randint(0, m, (m, ), device=device).to(torch.int32) if do_gather and m > 0 else None
#     if mode == "ragged" and do_scatter:
#         sindx = torch.randperm(m, device=device).to(torch.int32) if do_scatter else None
#     if mode == "ragged":
#         slice_dim = m
#         slice_sizes = make_slice_sizes(n_slices, slice_dim, device=device)
#         x_ragged_metadata = make_ragged_tensor_metadata(slice_sizes, slice_dim)

#     precision_opt = init_precision(act_dtype, str(act_dtype).startswith("torch.float8"), weight_dtype, False, mode, n_slices, device=device)
#     x, w, bias, _, _ = init_compute_data(m, n, k, x_ragged_metadata, gindx, sindx, n_slices, mode,
#                                          act_dtype, weight_dtype, False, device=device)

#     if mode == "batched":
#         x_ragged_metadata, gindx, sindx = None, None, None

#     try:
#         a = swiglu(matmul(x, w, bias, x_ragged_metadata, gindx, sindx, precision_opt), swiglu_alpha,
#                    precision_config=SwiGLUPrecisionConfig(swiglu_limit))
#         b = matmul(
#             x, w, bias, x_ragged_metadata, gindx, sindx, precision_opt,
#             fused_activation=FusedActivation(FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2),
#                                              (swiglu_alpha, swiglu_limit)))
#     except opt_flags.InapplicableConstraint:
#         pytest.skip("inapplicable constraint")

#     assert_close(a, b)


# @pytest.mark.parametrize("m, n, k", [
#     (320, 2**19, 0),
#     (4096, 4096, 0),
# ])
# @pytest.mark.parametrize("view_x_as_zero_cols", [False, True])
# def test_zero_reduction_dim(m, n, k, view_x_as_zero_cols):
#     torch.manual_seed(0)

#     if view_x_as_zero_cols:
#         x = torch.randn(m, m, device="cuda", dtype=torch.bfloat16)
#         x = x[:0, :].transpose(-1, -2)
#     else:
#         x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
#     w = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
#     bias = torch.randn(n, device="cuda", dtype=torch.float32)

#     try:
#         tri_y = matmul(x, w, bias)
#     except opt_flags.InapplicableConstraint:
#         pytest.skip("inapplicable constraint")
#     ref_y = matmul_torch(x, w, bias, round_x=lambda x, idx: x, round_y=lambda y: y)

#     assert_close(ref_y, tri_y)
