import triton
import triton.language as tl


@triton.jit
def _gather_indx(X, shape_xm, shape_xn, stride_xm, stride_xn, Y, shape_ym, shape_yn, stride_ym, stride_yn, SrcIndx,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # 2D program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # row and column offsets for this block
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # masks for valid output rows (in Y) and valid columns (in X/Y)
    mask_rows = off_m < shape_ym
    mask_cols = off_n < shape_xn
    # Load source indices for the block of rows (one per output row)
    # SrcIndx is assumed 1D with length == shape_ym
    srcs = tl.load(SrcIndx + off_m, mask=mask_rows, other=-1)
    # Validate src indices (they must be within [0, shape_xm))
    srcs_valid = mask_rows & (srcs >= 0) & (srcs < shape_xm)
    x_offs = srcs[:, None] * stride_xm + off_n[None, :] * stride_xn
    y_offs = off_m[:, None] * stride_ym + off_n[None, :] * stride_yn
    # load
    load_mask = srcs_valid[:, None] & mask_cols[None, :]
    vals = tl.load(X + x_offs, mask=load_mask, other=0.0)
    # store
    tl.store(Y + y_offs, vals, mask=load_mask)
