from dataclasses import dataclass
import torch
import triton
import triton.language as tl
from triton_kernels.numerics_details.mxfp import quantize_mxfp8_fn
from triton_kernels.numerics_details.flexpoint import float_to_flex, load_scale
from triton_kernels.numerics import InFlexData, OutFlexData, MAX_FINITE_FLOAT8E4B8, MAX_FINITE_FLOAT8E4NV, MAX_FINITE_FLOAT8E5
from typing import Optional
from .specialize import SpecializationModule, ClosureArg, FnSpecs


@dataclass(frozen=True)
class PostprocessFn:
    specs: FnSpecs = FnSpecs.default()
    fn_args: tuple[object] = tuple()


# Return strides in this order: (reduction dim, non-reduction dim #0, non-reduction dim #1).
def _get_strides(t, dim, strides=None):
    if t is None:
        return 0, 0, 0

    assert t.ndim == 3
    assert dim in (0, 1, 2)
    nonred = tuple(d for d in (0, 1, 2) if d != dim)
    if strides is None:
        strides = t.stride()
    return strides[dim], strides[nonred[0]], strides[nonred[1]]


def reduce_launch_metadata(grid, kernel, args):
    from .proton_opts import launch_metadata_allow_sync
    ret = dict()
    X, Y, Mask, dim = args["X"], args["Y"], args["Mask"], args["DIM"]
    nbits = X.dtype.itemsize * 8
    ret["name"] = f"{kernel.name} {tuple(X.shape)}->{tuple(Y.shape)}"

    # TODO: Currently not counting scale or mx.
    if Mask is None:
        ret[f"flops{nbits}"] = X.numel() - Y.numel()
        ret["bytes"] = X.numel() * X.element_size() + Y.numel() * Y.element_size()
    else:
        m = (Mask != 0)
        total_loads = m.sum()
        total_adds = (m.sum(dim=dim) - 1).clamp(min=0).sum()
        if launch_metadata_allow_sync():
            total_loads = total_loads.item()
            total_adds = total_adds.item()
        ret[f"flops{nbits}"] = total_adds
        ret["bytes"] = total_loads * X.element_size() + Y.numel() * Y.element_size()
    return ret


@triton.jit(launch_metadata=reduce_launch_metadata)
def _reduce_forward(X, stride_xr: tl.int64, stride_x0: tl.int64, stride_x1,  # x tensor (input)
                    XMx, stride_xmxr, stride_xmx0, stride_xmx1,  # x mx scale
                    Y, stride_y0: tl.int64, stride_y1,  # y tensor (output)
                    YMx, stride_ymx0, stride_ymx1,  # y mx scale
                    Mask, stride_mr, stride_m0, stride_m1,  # mask tensor
                    Scale, stride_sr, stride_s0, stride_s1,  # scale tensor
                    UnpaddedBatchSize,  # optional scalar tensor
                    # shape (K = reduction dim; S0, IN_S1 = input dims, OUT_S1 = output dims)
                    K: tl.constexpr, S0, X_S1, Y_S1,  #
                    POSTPROCESS_FN1: tl.constexpr, postprocess_fn1_args,  #
                    POSTPROCESS_FN2: tl.constexpr, postprocess_fn2_args,  #
                    XFlex,  # x flex (global) scale
                    YFlexExpected, YFlexActual, YFlexChecksum,
                    Y_FLEX_SATURATE_INF: tl.constexpr,  # y flex (global) scale
                    IS_MASK_NONE: tl.constexpr,  #
                    BROADCAST_R: tl.constexpr,  #
                    BROADCAST_S0: tl.constexpr,  #
                    BROADCAST_S1: tl.constexpr,  #
                    IS_SCALE_NONE: tl.constexpr,  #
                    SCALE_BROADCAST_R: tl.constexpr,  #
                    SCALE_BROADCAST_S0: tl.constexpr,  #
                    SCALE_BROADCAST_S1: tl.constexpr,  #
                    BLOCK_S0: tl.constexpr,  #
                    BLOCK_X_S1: tl.constexpr,  #
                    BLOCK_Y_S1: tl.constexpr,  #
                    DIM,  # only used for launch_metadata
                    ):
    pid_s0 = tl.program_id(0)
    pid_s1 = tl.program_id(1)
    tl.static_assert(BLOCK_X_S1 % 32 == 0)
    BLOCK_X_SMX1: tl.constexpr = BLOCK_X_S1 // 32
    BLOCK_Y_SMX1: tl.constexpr = BLOCK_Y_S1 // 32
    offs_s0 = pid_s0 * BLOCK_S0 + tl.arange(0, BLOCK_S0)
    offs_x_s1 = pid_s1 * BLOCK_X_S1 + tl.arange(0, BLOCK_X_S1)
    offs_x_smx1 = pid_s1 * BLOCK_X_SMX1 + tl.arange(0, BLOCK_X_SMX1)
    if UnpaddedBatchSize is not None:
        unpadded = tl.load(UnpaddedBatchSize).to(tl.int32)
        if pid_s0 * BLOCK_S0 >= unpadded:
            return
        valid_s0 = offs_s0 < unpadded
    else:
        valid_s0 = offs_s0 < S0
    valid_x_s1 = offs_x_s1 < X_S1
    valid_in_smx1 = offs_x_smx1 < tl.cdiv(X_S1, 32)
    y = tl.zeros((BLOCK_S0, BLOCK_X_S1), dtype=tl.float32)
    x_flex_scale = load_scale(XFlex)
    for k in (tl.static_range if K <= 8 else tl.range)(0, K):
        x_ptrs = X + k * stride_xr + offs_s0[:, None] * stride_x0 + offs_x_s1[None, :] * stride_x1
        mask = valid_s0[:, None] & valid_x_s1[None, :]
        if not IS_MASK_NONE:
            k_term = 0 if BROADCAST_R else (k * stride_mr)
            s0_term = 0 if BROADCAST_S0 else (offs_s0[:, None] * stride_m0)
            s1_term = 0 if BROADCAST_S1 else (offs_x_s1[None, :] * stride_m1)
            m_ptrs = Mask + k_term + s0_term + s1_term
            m = tl.load(m_ptrs, mask=mask, other=1).to(tl.int1)
            mask &= m
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        if XMx is not None:
            xmx_ptrs = XMx + k * stride_xmxr + offs_s0[:, None] * stride_xmx0 + offs_x_smx1[None, :] * stride_xmx1
            xmx = tl.load(xmx_ptrs, mask=valid_s0[:, None] & valid_in_smx1[None, :], other=0.0)
            xmx = (xmx.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
            x = (xmx[:, :, None] * x.reshape([BLOCK_S0, BLOCK_X_S1 // 32, 32])).reshape([BLOCK_S0, BLOCK_X_S1])
        x = x * x_flex_scale
        if not IS_SCALE_NONE:
            k_term_s = 0 if SCALE_BROADCAST_R else (k * stride_sr)
            s0_term_s = 0 if SCALE_BROADCAST_S0 else (offs_s0[:, None] * stride_s0)
            s1_term_s = 0 if SCALE_BROADCAST_S1 else (offs_x_s1[None, :] * stride_s1)
            s_ptrs = Scale + k_term_s + s0_term_s + s1_term_s
            s = tl.load(s_ptrs, mask=mask, other=1)
            x = tl.fma(x, s, 0.0)
        y += x
    if POSTPROCESS_FN1 is not None:
        y = POSTPROCESS_FN1(y, *postprocess_fn1_args)
    offs_y_s1 = pid_s1 * BLOCK_Y_S1 + tl.arange(0, BLOCK_Y_S1)
    offs_y_smx1 = pid_s1 * BLOCK_Y_SMX1 + tl.arange(0, BLOCK_Y_SMX1)
    valid_y_s1 = offs_y_s1 < Y_S1
    valid_y_smx1 = offs_y_smx1 < tl.cdiv(Y_S1, 32)
    y = float_to_flex(y, YFlexExpected, YFlexActual, YFlexChecksum, None, Y, Y_FLEX_SATURATE_INF)
    # TODO (phil): keeping for backward compatibility, but will remove !
    if YMx is None and POSTPROCESS_FN2 is not None:
        y = POSTPROCESS_FN2(y, *postprocess_fn2_args, target_dtype=Y.dtype.element_ty)
    y_ptrs = Y + offs_s0[:, None] * stride_y0 + offs_y_s1[None, :] * stride_y1
    if YMx is not None:
        y, y_scale = quantize_mxfp8_fn(y, valid_y_s1[None, :])
        y_mx_ptrs = YMx + offs_s0[:, None] * stride_ymx0 + offs_y_smx1[None, :] * stride_ymx1
        tl.store(y_mx_ptrs, y_scale, mask=valid_s0[:, None] & valid_y_smx1[None, :])
    tl.store(y_ptrs, y, mask=valid_s0[:, None] & valid_y_s1[None, :])


forward_specializations = SpecializationModule(
    "reduce_forward",
    kernels=[("_reduce_forward", _reduce_forward)],
    closure_args={
        "postprocess_fn1": ClosureArg("POSTPROCESS_FN1", "postprocess_fn1_args"),
        "postprocess_fn2": ClosureArg("POSTPROCESS_FN2", "postprocess_fn2_args"),
    },
)


def reduce_forward(
    x: torch.Tensor,
    dim: int,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    x_mxscale: Optional[torch.Tensor] = None,
    x_flex: Optional[InFlexData] = InFlexData(),
    y_dtype: Optional[torch.dtype] = None,
    y_flex: Optional[OutFlexData] = OutFlexData(),
    y_flex_saturate_inf: bool = False,
    y_has_mx: Optional[bool] = None,
    y: Optional[torch.Tensor] = None,
    postprocess_fn1: Optional[PostprocessFn] = None,
    # TODO: keeping for backward compatibility, but will remove !
    postprocess_fn2: Optional[PostprocessFn] = None,
    unpadded_batch_size: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Performs a reduction over the specified dimension of the input tensor,
    optionally multiplied by `scale` and ignoring masked elements.

    Arguments:
        - x: Tensor
          input tensor to reduce.
        - dim: int
          dimension along which `x` should be reduce.
        - mask: Optional[torch.Tensor]
          integer mask of the same shape as `x` (or broadcastable to it).
          entries that are `0` are ignored in the reduction.
          if `mask is None`, all elements are included.
        - scale: Optional[torch.Tensor]
          scale factors of the same shape as `x` (or broadcastable to it).
          the reduction is performed over `x * scale`. If `scale is None`,
          a value of 1 is used everywhere.
        - unpadded_batch_size: Optional[torch.Tensor]
          Optional single-element tensor specifying the number of entries to reduce along the first dimension.

    Returns:
        - output: torch.Tensor
          The reduced tensor with `dim` removed.
        - output_mxscale: Optional[torch.Tensor]
          The output mx scale if input is micro-scaled, else None.
    """
    if x.ndim != 3:
        raise NotImplementedError("reduce only supports 3D inputs in this implementation")
    if dim < 0:
        dim += x.ndim
    if dim not in (0, 1, 2):
        raise ValueError("dim must be in {0,1,2}")
    if x_mxscale is not None:
        if dim == 2:
            raise ValueError("reduction over the micro-scaled dimension not supported")
        assert x.shape[:-1] == x_mxscale.shape[:-1]
        assert triton.cdiv(x.shape[-1], 32) * 32 == x_mxscale.shape[-1] * 32
    # assert not y_flex.is_per_batch
    if postprocess_fn1 is None:
        postprocess_fn1 = PostprocessFn()
    if postprocess_fn2 is None:
        postprocess_fn2 = PostprocessFn()
    if y_dtype is None:
        y_dtype = x.dtype
    if y_flex is None:
        y_flex = OutFlexData()
    if x_flex is None:
        x_flex = InFlexData()
    if y_has_mx is None:
        y_has_mx = x_mxscale is not None
    # input shapes
    dims = (0, 1, 2)
    nonred = tuple(d for d in dims if d != dim)
    S0, X_S1 = x.shape[nonred[0]], x.shape[nonred[1]]
    Y_S1 = X_S1 // postprocess_fn1.specs.reduction_n
    if y is None:
        y = torch.empty((S0, Y_S1), device=x.device, dtype=y_dtype)
    assert y.shape == (S0, Y_S1), f"y.shape: {y.shape} != ({S0}, {Y_S1})"
    y_mxscale = None
    if y_has_mx:
        y_mxscale = torch.empty((S0, triton.cdiv(Y_S1, 32)), device=x.device, dtype=torch.uint8)
    # Strides for X along reduced and non-reduced dims
    stride_xr = x.stride(dim)
    stride_x0 = x.stride(nonred[0])
    stride_x1 = x.stride(nonred[1])
    # Strides for X mx scales
    stride_xmxr = None if x_mxscale is None else x_mxscale.stride(dim)
    stride_xmx0 = None if x_mxscale is None else x_mxscale.stride(nonred[0])
    stride_xmx1 = None if x_mxscale is None else x_mxscale.stride(nonred[1])
    # Strides for Y mx scales
    stride_ymx0 = None if y_mxscale is None else y_mxscale.stride(0)
    stride_ymx1 = None if y_mxscale is None else y_mxscale.stride(1)
    # Mask strides (broadcast allowed via stride 0)
    stride_mr, stride_m0, stride_m1 = _get_strides(mask, dim)
    # Scale strides (broadcast allowed via stride 0)
    stride_sr, stride_s0, stride_s1 = _get_strides(scale, dim)
    K = x.shape[dim]
    # Always use the 2D tiled kernel with constexpr metaprogramming for mask broadcasting
    BLOCK_S0 = 32
    BLOCK_X_S1 = 128
    BLOCK_Y_S1 = 128 // postprocess_fn1.specs.reduction_n
    grid = (triton.cdiv(S0, BLOCK_S0), triton.cdiv(Y_S1, BLOCK_Y_S1))
    reduce_kernel = forward_specializations.get(postprocess_fn1=postprocess_fn1.specs,
                                                postprocess_fn2=postprocess_fn2.specs)._reduce_forward
    reduce_kernel[grid](
        x_flex.reinterpret(x), stride_xr, stride_x0, stride_x1,  #
        x_mxscale, stride_xmxr, stride_xmx0, stride_xmx1,  #
        y_flex.reinterpret(y), y.stride(0), y.stride(1),  #
        y_mxscale, stride_ymx0, stride_ymx1,  #
        mask, stride_mr, stride_m0, stride_m1,  #
        scale, stride_sr, stride_s0, stride_s1,  #
        unpadded_batch_size,  #
        K, S0, X_S1, Y_S1,  #
        *postprocess_fn1.fn_args, *postprocess_fn2.fn_args,  #
        x_flex.scale, y_flex.expected_scale, y_flex.actual_scale, y_flex.checksum_scale,  #
        y_flex_saturate_inf,  #
        IS_MASK_NONE=(mask is None),  #
        BROADCAST_R=(stride_mr == 0),  #
        BROADCAST_S0=(stride_m0 == 0),  #
        BROADCAST_S1=(stride_m1 == 0),  #
        IS_SCALE_NONE=(scale is None),  #
        SCALE_BROADCAST_R=(stride_sr == 0),  #
        SCALE_BROADCAST_S0=(stride_s0 == 0),  #
        SCALE_BROADCAST_S1=(stride_s1 == 0),  #
        BLOCK_S0=BLOCK_S0,  #
        BLOCK_X_S1=BLOCK_X_S1,  #
        BLOCK_Y_S1=BLOCK_Y_S1,  #
        DIM=dim,  #
        num_warps=4  #
    )
    return y, y_mxscale


# ------------------------------------------------------------


@triton.jit
def _reduce_backward(
    dY,
    stride_y0: tl.int64,
    stride_y1,  # upstream grad (S0, Y_S1)
    dX,
    stride_xr: tl.int64,
    stride_x0: tl.int64,
    stride_x1,  # grad wrt X (K, S0, X_S1) in the chosen layout
    XMx,
    stride_xmxr,
    stride_xmx0,
    stride_xmx1,  # input micro-scales (optional)
    Mask,
    stride_mr,
    stride_m0,
    stride_m1,  # mask (optional)
    Scale,
    stride_sr,
    stride_s0,
    stride_s1,  # scale (optional)
    UnpaddedBatchSize,  # optional scalar tensor
    K,
    S0,
    X_S1,
    Y_S1,  # shapes
    XFlex,  # global input flex scale (scalar device buffer)
    IS_MASK_NONE: tl.constexpr,
    BROADCAST_R: tl.constexpr,
    BROADCAST_S0: tl.constexpr,
    BROADCAST_S1: tl.constexpr,
    IS_SCALE_NONE: tl.constexpr,
    SCALE_BROADCAST_R: tl.constexpr,
    SCALE_BROADCAST_S0: tl.constexpr,
    SCALE_BROADCAST_S1: tl.constexpr,
    REDUCTION_N: tl.constexpr,  # maps X_S1 -> Y_S1 (grouped sum in fwd)
    BLOCK_S0: tl.constexpr,
    BLOCK_X_S1: tl.constexpr,
):
    # Tile over (S0, X_S1). We loop over the reduction K dimension.
    pid_s0 = tl.program_id(0)
    pid_s1 = tl.program_id(1)

    tl.static_assert(BLOCK_X_S1 % 32 == 0)
    BLOCK_X_SMX1: tl.constexpr = BLOCK_X_S1 // 32

    offs_s0 = pid_s0 * BLOCK_S0 + tl.arange(0, BLOCK_S0)
    offs_x_s1 = pid_s1 * BLOCK_X_S1 + tl.arange(0, BLOCK_X_S1)
    offs_x_smx1 = pid_s1 * BLOCK_X_SMX1 + tl.arange(0, BLOCK_X_SMX1)

    if UnpaddedBatchSize is not None:
        unpadded = tl.load(UnpaddedBatchSize).to(tl.int32)
        if pid_s0 * BLOCK_S0 >= unpadded:
            return
        valid_s0 = offs_s0 < unpadded
    else:
        valid_s0 = offs_s0 < S0
    valid_x_s1 = offs_x_s1 < X_S1
    valid_in_smx1 = offs_x_smx1 < tl.cdiv(X_S1, 32)

    # Map X_S1 positions to their Y_S1 group index (grouped-sum fwd)
    offs_y_from_x = offs_x_s1 // REDUCTION_N
    valid_y_from_x = offs_y_from_x < Y_S1

    # Load upstream grad; broadcasting over the REDUCTION_N group happens via indexing.
    dy_ptrs = dY + offs_s0[:, None] * stride_y0 + offs_y_from_x[None, :] * stride_y1
    dy = tl.load(dy_ptrs, mask=valid_s0[:, None] & valid_y_from_x[None, :], other=0.0).to(tl.float32)

    # Global flex scale (scalar)
    x_flex_scale = load_scale(XFlex)

    # Loop over the reduced dimension
    for k in tl.range(0, K, num_stages=2):
        g = dy
        # Multiply by input micro-scale per group of 32 lanes if present
        if XMx is not None:
            xmx_ptrs = XMx + k * stride_xmxr + offs_s0[:, None] * stride_xmx0 + offs_x_smx1[None, :] * stride_xmx1
            xmx = tl.load(xmx_ptrs, mask=valid_s0[:, None] & valid_in_smx1[None, :], other=0)
            xmx = (xmx.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
            g = (g.reshape([BLOCK_S0, BLOCK_X_S1 // 32, 32]) * xmx[:, :, None]).reshape([BLOCK_S0, BLOCK_X_S1])
        # Multiply by global input flex scale
        g = g * x_flex_scale
        # Multiply by per-element Scale if provided
        if not IS_SCALE_NONE:
            k_term_s = 0 if SCALE_BROADCAST_R else (k * stride_sr)
            s0_term_s = 0 if SCALE_BROADCAST_S0 else (offs_s0[:, None] * stride_s0)
            s1_term_s = 0 if SCALE_BROADCAST_S1 else (offs_x_s1[None, :] * stride_s1)
            s_ptrs = Scale + k_term_s + s0_term_s + s1_term_s
            s = tl.load(s_ptrs, mask=valid_s0[:, None] & valid_x_s1[None, :], other=1)
            g = g * s
        # Apply mask if provided
        if not IS_MASK_NONE:
            k_term = 0 if BROADCAST_R else (k * stride_mr)
            s0_term = 0 if BROADCAST_S0 else (offs_s0[:, None] * stride_m0)
            s1_term = 0 if BROADCAST_S1 else (offs_x_s1[None, :] * stride_m1)
            m_ptrs = Mask + k_term + s0_term + s1_term
            m = tl.load(m_ptrs, mask=valid_s0[:, None] & valid_x_s1[None, :], other=1)
            g = tl.where(m != 0, g, 0.0)
        #
        dx_ptrs = dX + k * stride_xr + offs_s0[:, None] * stride_x0 + offs_x_s1[None, :] * stride_x1
        tl.store(dx_ptrs, g, mask=valid_s0[:, None] & valid_x_s1[None, :])


def reduce_backward(
    dy: torch.Tensor,
    x_shape: tuple[int, int, int],
    dim: int,
    *,
    mask: Optional[torch.Tensor],
    scale: Optional[torch.Tensor],
    x_mxscale: Optional[torch.Tensor],
    x_flex: Optional[InFlexData],
    postprocess_fn1: Optional[PostprocessFn],
    x_strides: tuple[int, int, int],
    x_mx_strides: Optional[tuple[int, int, int]],
    mask_strides: Optional[tuple[int, int, int]],
    scale_strides: Optional[tuple[int, int, int]],
    dx: torch.Tensor,
    unpadded_batch_size: Optional[torch.Tensor] = None,
):
    # Shapes/axes handling mirrors `reduce(...)`
    if dim < 0:
        dim += 3
    dims = (0, 1, 2)
    nonred = tuple(d for d in dims if d != dim)

    S0, X_S1 = x_shape[nonred[0]], x_shape[nonred[1]]
    K = x_shape[dim]

    # Postprocess grouping (grouped sum). Default is identity (1).
    reduction_n = (postprocess_fn1.specs.reduction_n if postprocess_fn1 is not None else FnSpecs.default().reduction_n)
    Y_S1 = X_S1 // reduction_n
    assert dy.shape == (S0, Y_S1), f"dY shape {dy.shape} mismatch with (S0={S0}, Y_S1={Y_S1})"

    # Strides for dX must match the element size of the tensor passed to the kernel.
    # If we reinterpret the dtype (e.g., flex/float8), use the reinterpreted view's strides.
    dx_view = x_flex.reinterpret(dx)
    stride_xr, stride_x0, stride_x1 = _get_strides(dx_view, dim)
    stride_xmxr = stride_xmx0 = stride_xmx1 = 0
    if x_mxscale is not None:
        stride_xmxr, stride_xmx0, stride_xmx1 = x_mx_strides

    stride_mr, stride_m0, stride_m1 = _get_strides(mask, dim, mask_strides)
    stride_sr, stride_s0, stride_s1 = _get_strides(scale, dim, scale_strides)

    # Launch configuration mirrors forward (but we tile over X_S1, not Y_S1)
    BLOCK_S0 = 64
    BLOCK_X_S1 = 128
    grid = (triton.cdiv(S0, BLOCK_S0), triton.cdiv(X_S1, BLOCK_X_S1))

    _reduce_backward[grid](
        dy,
        dy.stride(0),
        dy.stride(1),
        dx_view,
        stride_xr,
        stride_x0,
        stride_x1,
        x_mxscale,
        stride_xmxr,
        stride_xmx0,
        stride_xmx1,
        mask,
        stride_mr,
        stride_m0,
        stride_m1,
        scale,
        stride_sr,
        stride_s0,
        stride_s1,
        unpadded_batch_size,
        K,
        S0,
        X_S1,
        Y_S1,
        x_flex.scale,
        IS_MASK_NONE=(mask is None),
        BROADCAST_R=(stride_mr == 0),
        BROADCAST_S0=(stride_m0 == 0),
        BROADCAST_S1=(stride_m1 == 0),
        IS_SCALE_NONE=(scale is None),
        SCALE_BROADCAST_R=(stride_sr == 0),
        SCALE_BROADCAST_S0=(stride_s0 == 0),
        SCALE_BROADCAST_S1=(stride_s1 == 0),
        REDUCTION_N=reduction_n,
        BLOCK_S0=BLOCK_S0,
        BLOCK_X_S1=BLOCK_X_S1,
        num_warps=4,
    )


# ------------------------------------------------------------

backward_specializations = SpecializationModule(
    "reduce_backward",
    kernels=[("_reduce_backward", _reduce_backward)],
    closure_args={
        "postprocess_fn1": ClosureArg("POSTPROCESS_FN1", "postprocess_fn1_args"),
        "postprocess_fn2": ClosureArg("POSTPROCESS_FN2", "postprocess_fn2_args"),
    },
)


class _ReduceAutograd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, dim: int, mask: Optional[torch.Tensor], scale: Optional[torch.Tensor],
                x_mxscale: Optional[torch.Tensor], x_flex: Optional[InFlexData], y_dtype: Optional[torch.dtype],
                y_flex: Optional[OutFlexData], y_flex_saturate_inf: bool, y_has_mx: Optional[bool],
                y: Optional[torch.Tensor], postprocess_fn1: Optional[PostprocessFn],
                postprocess_fn2: Optional[PostprocessFn], unpadded_batch_size: Optional[torch.Tensor]):
        # Run your existing Triton forward
        y, y_mx = reduce_forward(
            x=x,
            dim=dim,
            mask=mask,
            scale=scale,
            x_mxscale=x_mxscale,
            x_flex=x_flex,
            y_dtype=y_dtype,
            y_flex=y_flex,
            y_flex_saturate_inf=y_flex_saturate_inf,
            y_has_mx=y_has_mx,
            y=y,
            postprocess_fn1=postprocess_fn1,
            postprocess_fn2=postprocess_fn2,
            unpadded_batch_size=unpadded_batch_size,
        )

        # Save everything needed for backward (no tensors are modified)
        ctx.dim = dim
        ctx.x_shape = tuple(x.shape)
        ctx.x_dtype = x.dtype
        ctx.device = x.device
        ctx.mask = mask
        ctx.scale = scale
        ctx.x_mxscale = x_mxscale
        ctx.x_flex = x_flex if x_flex is not None else InFlexData()
        ctx.postprocess_fn1 = postprocess_fn1 if postprocess_fn1 is not None else PostprocessFn()
        ctx.x_strides = tuple(x.stride())
        ctx.x_mx_strides = tuple(x_mxscale.stride()) if x_mxscale is not None else None
        ctx.mask_strides = tuple(mask.stride()) if mask is not None else None
        ctx.scale_strides = tuple(scale.stride()) if scale is not None else None
        ctx.y_has_mx = bool(y_mx is not None)
        ctx.unpadded_batch_size = unpadded_batch_size

        return y, y_mx

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor, grad_y_mxscale: Optional[torch.Tensor] = None):
        # We do not support grads through MX-quantized outputs (no torch compute in bwd)
        if ctx.y_has_mx:
            raise NotImplementedError("Backward with y_mxscale (MX-quantized outputs) is not supported.")

        # Allocate grad for x; (no torch compute)
        dx = torch.empty(ctx.x_shape, dtype=ctx.x_dtype, device=grad_y.device)

        reduce_backward(
            dy=grad_y,
            x_shape=ctx.x_shape,
            dim=ctx.dim,
            mask=ctx.mask,
            scale=ctx.scale,
            x_mxscale=ctx.x_mxscale,
            x_flex=ctx.x_flex,
            postprocess_fn1=ctx.postprocess_fn1,
            x_strides=ctx.x_strides,
            x_mx_strides=ctx.x_mx_strides,
            mask_strides=ctx.mask_strides,
            scale_strides=ctx.scale_strides,
            dx=dx,
            unpadded_batch_size=ctx.unpadded_batch_size,
        )
        return dx, None, None, None, None, None, None, None, None, None, None, None, None, None


def reduce(
    x: torch.Tensor,
    dim: int,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    x_mxscale: Optional[torch.Tensor] = None,
    x_flex: Optional[InFlexData] = InFlexData(),
    y: Optional[torch.Tensor] = None,
    y_dtype: Optional[torch.dtype] = None,
    y_flex: Optional[OutFlexData] = OutFlexData(),
    y_flex_saturate_inf: bool = False,
    y_has_mx: Optional[bool] = None,
    postprocess_fn1: Optional[PostprocessFn] = None,
    postprocess_fn2: Optional[PostprocessFn] = None,
    unpadded_batch_size: Optional[torch.Tensor] = None,
):
    return _ReduceAutograd.apply(x, dim, mask, scale, x_mxscale, x_flex, y_dtype, y_flex,  #
                                 y_flex_saturate_inf, y_has_mx, y, postprocess_fn1, postprocess_fn2,
                                 unpadded_batch_size)


# ------------------------------------------------------------


def compute_actual_scale(x, dtype, per_batch_scale=False):
    max_finite = {
        torch.float8_e5m2: MAX_FINITE_FLOAT8E5,
        torch.float8_e4m3fn: MAX_FINITE_FLOAT8E4NV,
        torch.float8_e4m3fnuz: MAX_FINITE_FLOAT8E4B8,
    }[dtype]
    maxvals = x.abs().amax(dim=tuple(range(1, x.ndim))) if per_batch_scale else x.abs().max()
    return maxvals / max_finite


def reduce_torch(x: torch.Tensor, dim: int, mask: Optional[torch.Tensor] = None,  #
                 scale: Optional[torch.Tensor] = None,  #
                 x_mxscale: Optional[torch.Tensor] = None,  #
                 x_flex: Optional[InFlexData] = InFlexData(), y_flex: Optional[OutFlexData] = OutFlexData(),
                 y_flex_saturate_inf: bool = False, postprocess_fn1: Optional[callable] = None,
                 unpadded_batch_size: Optional[torch.Tensor] = None):
    from triton_kernels.numerics_details.mxfp import downcast_to_mxfp_torch, upcast_from_mxfp_torch
    x_dtype = x.dtype
    # upcast input
    if x_mxscale is not None:
        x = upcast_from_mxfp_torch(x, x_mxscale, torch.float32, axis=-1)
    x = x.to(torch.float32)
    if x_flex is not None:
        x *= x_flex.scale
    # upcast scale
    if scale is None:
        scale = torch.ones(1, dtype=torch.float32, device=x.device)
    scale = scale.to(torch.float32)
    # initialize mask
    if mask is None:
        mask = torch.ones(1, dtype=torch.bool, device=x.device)
    mask = mask.to(torch.bool)
    ret = torch.where(mask, x * scale, 0).sum(dim=dim)
    if postprocess_fn1 is not None:
        ret = postprocess_fn1(ret)
    if y_flex is not None:
        y_flex.actual_scale.copy_(compute_actual_scale(ret, x_dtype, y_flex.is_per_batch))
        ret = (ret / y_flex.expected_scale).to(x_dtype)
    # downcast output
    ret_mxscale = None
    if x_mxscale is not None:
        assert y_flex is None
        ret, ret_mxscale = downcast_to_mxfp_torch(ret, torch.float8_e4m3fn, axis=-1)
    return ret.to(x_dtype), ret_mxscale
