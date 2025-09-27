from typing import Optional, Tuple
import torch
import triton
import triton.language as tl


def batchnorm_forward(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-5,
    training: bool = True,
    running_mean: Optional[torch.Tensor] = None,
    running_var: Optional[torch.Tensor] = None,
    momentum: float = 0.1,
    layout: str = "NCHW",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BatchNorm forward (NCHW only) using Triton kernels.
    Returns (y, saved_mean, saved_var). In eval mode, saved_* echo inputs.
    """
    if not x.is_cuda:
        raise ValueError("CUDA tensor required")
    if x.ndim not in (2, 4):
        raise ValueError(f"Unsupported ndim={x.ndim}; expected 2 or 4")
    if layout not in ("NCHW",):
        raise ValueError("Only NCHW layout supported in v1")
    C = x.shape[1]
    if gamma is None or beta is None:
        raise ValueError("gamma and beta must be provided")
    if gamma.numel() != C or beta.numel() != C:
        raise ValueError("gamma/beta must have size equal to channel dimension")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if not training:
        if running_mean is None or running_var is None:
            raise ValueError("running_mean/var required for eval mode")
        if running_mean.numel() != C or running_var.numel() != C:
            raise ValueError("running stats must match channel dimension")

    # Prepare contiguous channel-first view: (C, R)
    if x.ndim == 2:
        N = x.shape[0]
        R = N
        x_cf = x.transpose(0, 1).contiguous()  # (C, N)
        inv_perm_to_orig = (1, 0)
        out_shape = (N, C)
    else:
        N, C_, H, W = x.shape
        assert C_ == C
        R = N * H * W
        x_cf = x.contiguous().permute(1, 0, 2, 3).contiguous().view(C, R)  # (C, R)
        inv_perm_to_orig = (1, 0, 2, 3)
        out_shape = (N, C, H, W)

    device = x.device
    x_dtype = x.dtype

    # Stats (mean, var) in fp32
    mean = torch.empty((C,), device=device, dtype=torch.float32)
    var = torch.empty((C,), device=device, dtype=torch.float32)

    if training:
        # Launch stats kernel: per-channel Welford across R
        BLOCK_R = 1024
        grid = (C,)
        _bn_stats_kernel[grid](
            x_cf, mean, var,
            C, R,
            BLOCK_R=BLOCK_R,
        )
        saved_mean, saved_var = mean, var
    else:
        # Use provided running stats
        saved_mean = running_mean.to(torch.float32, copy=True)
        saved_var = running_var.to(torch.float32, copy=True)

    # Precompute inv_std
    inv_std = (saved_var + eps).rsqrt()

    # Normalize
    y_cf = torch.empty_like(x_cf, dtype=x_dtype)
    BLOCK_R = 1024
    grid = (C,)
    _bn_norm_kernel[grid](
        x_cf, y_cf,
        saved_mean, inv_std,
        gamma.to(torch.float32), beta.to(torch.float32),
        C, R,
        BLOCK_R=BLOCK_R,
    )

    # Restore original layout
    if x.ndim == 2:
        y = y_cf.transpose(0, 1).contiguous().view(out_shape)
    else:
        y = y_cf.view(C, N, H, W).permute(1, 0, 2, 3).contiguous()

    return y, saved_mean, saved_var


@triton.jit
def _bn_stats_kernel(x_cf, mean_out, var_out, C, R, BLOCK_R: tl.constexpr):
    c = tl.program_id(0)
    if c >= C:
        return
    # Pointers
    x_ptr = x_cf + c * R
    # Welford accumulators in fp32
    count = tl.float32(0)
    mean = tl.float32(0)
    M2 = tl.float32(0)

    offs = tl.arange(0, BLOCK_R)
    for r0 in range(0, R, BLOCK_R):
        idx = r0 + offs
        mask = idx < R
        vals = tl.load(x_ptr + idx, mask=mask, other=0).to(tl.float32)
        # Process active elements
        k_active = tl.sum(mask, axis=0)
        # Iterate within the block (unrolled reduction)
        # Convert to vector form: update per element
        for i in range(0, BLOCK_R):
            m = mask[i]
            if m:
                x = vals[i]
                count_new = count + 1.0
                delta = x - mean
                mean = mean + delta / count_new
                delta2 = x - mean
                M2 = M2 + delta * delta2
                count = count_new

    # Finalize
    var = tl.where(count > 1.0, M2 / count, 0.0)
    tl.store(mean_out + c, mean)
    tl.store(var_out + c, var)


@triton.jit
def _bn_norm_kernel(x_cf, y_cf, mean, inv_std, gamma, beta, C, R, BLOCK_R: tl.constexpr):
    c = tl.program_id(0)
    if c >= C:
        return
    x_ptr = x_cf + c * R
    y_ptr = y_cf + c * R
    m = tl.load(mean + c)
    istd = tl.load(inv_std + c)
    g = tl.load(gamma + c)
    b = tl.load(beta + c)

    offs = tl.arange(0, BLOCK_R)
    for r0 in range(0, R, BLOCK_R):
        idx = r0 + offs
        mask = idx < R
        x = tl.load(x_ptr + idx, mask=mask, other=0)
        x_f = x.to(tl.float32)
        y = (x_f - m) * istd
        y = y * g + b
        y = y.to(tl.type_of(x))
        tl.store(y_ptr + idx, y, mask=mask)



