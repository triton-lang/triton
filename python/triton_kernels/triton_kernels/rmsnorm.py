import torch
import triton
import triton.language as tl

def rmsnorm_launch_metadata(grid, kernel, args):
    """Metadata for the profiler and auto-tuner."""
    return {
        "name": f"{kernel.name} [M={args['M']}, N={args['N']}]",
        "bytes_read": args['M'] * args['N'] * 2 * 2, # Assuming FP16 inputs/weights
        "bytes_written": args['M'] * args['N'] * 2,
    }

@triton.jit(launch_metadata=rmsnorm_launch_metadata)
def _rmsnorm_kernel(
    X,           # Input tensor ptr
    Weight,      # Gamma weight ptr
    Out,         # Output tensor ptr
    stride_x,    # Stride of input row
    stride_out,  # Stride of output row
    M,           # Number of rows (Batch * SeqLen)
    N,           # Number of columns (Hidden Dim)
    eps: tl.constexpr,         # Epsilon for stability
    BLOCK_N: tl.constexpr      # Block size for reduction (Next power of 2 of N)
):
    # 1. Map program to a specific row
    row_idx = tl.program_id(0)
    
    if row_idx >= M:
        return

    # 2. Setup pointers for the current row
    x_row_ptr = X + (row_idx * stride_x)
    out_row_ptr = Out + (row_idx * stride_out)
    
    # 3. Create offsets and mask
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N
    
    # 4. Load Input Row and Weights
    x = tl.load(x_row_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(Weight + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # 5. RMSNorm Math (in FP32 for precision)
    # x^2
    x_sq = x * x
    
    # sum(x^2)
    sum_sq = tl.sum(x_sq, axis=0)
    
    # Mean of squares: sum(x^2) / N
    mean_sq = sum_sq / N
    
    # Reciprocal Square Root: 1 / sqrt(mean + eps)
    rsqrt = tl.rsqrt(mean_sq + eps)
    
    # Normalize and Scale
    out = (x * rsqrt) * w
    
    # 6. Store Output (Convert back to original dtype implicitly)
    tl.store(out_row_ptr + offsets, out, mask=mask)

def rms_norm(x, weight, eps=1e-6):
    """
    Applies Root Mean Square Normalization to the input tensor.
    Fused implementation for memory bandwidth efficiency.
    """
    assert x.is_contiguous(), "Input must be contiguous"
    assert x.shape[-1] == weight.shape[0], "Weight dimension must match input hidden dimension"
    
    # Reshape input to 2D for generic processing (M x N)
    original_shape = x.shape
    x_2d = x.view(-1, original_shape[-1])
    M, N = x_2d.shape
    
    out = torch.empty_like(x_2d)
    
    # Hardware config
    BLOCK_N = triton.next_power_of_2(N)
    num_warps = 4 if BLOCK_N <= 2048 else 8
    
    # Grid size: 1D grid mapped to rows
    grid = (M, )
    
    _rmsnorm_kernel[grid](
        x_2d, weight, out,
        x_2d.stride(0), out.stride(0),
        M, N,
        eps,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps
    )
    
    return out.view(original_shape)