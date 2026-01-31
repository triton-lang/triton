"""
Fused Rotary Positional Embedding (RoPE)
========================================

In this tutorial, you will write a high-performance kernel for Rotary Positional Embeddings (RoPE).
RoPE is a standard component in modern Large Language Models (LLMs) like Llama, Mistral, and DeepSeek.

In doing so, you will learn about:

* Handling vector splitting and rotation in Triton.
* Memory coalescing strategies for complex-number-like operations.
* Benchmarking against PyTorch native implementations.

"""

# %%
# Motivations
# -----------
#
# RoPE encodes positional information by rotating the query and key vectors in the embedding space.
# Given a vector :math:`x` and rotation factors :math:`\cos\theta, \sin\theta`, the rotation is defined as:
#
# .. math::
#    y_1 = x_1 \cdot \cos\theta - x_2 \cdot \sin\theta
#
#    y_2 = x_1 \cdot \sin\theta + x_2 \cdot \cos\theta
#
# where :math:`x_1` and :math:`x_2` represent the first and second halves of the head dimension.
# Doing this naively in PyTorch involves slicing, multiplication, and concatenation, which creates high memory overhead.
# A fused kernel can perform this in-place or with a single memory pass.

import torch

import triton
import triton.language as tl


# %%
# Kernel Implementation
# ---------------------
# The kernel loads the first half and second half of the vector separately,
# applies the rotation, and stores the result.

@triton.jit
def _rope_kernel(
    x_ptr,       # Pointer to input
    y_ptr,       # Pointer to output
    c_ptr,       # Pointer to cosine table
    s_ptr,       # Pointer to sine table
    stride_row,  # Stride to move to next token
    half_dim,    # Half of Head Dimension
    BLOCK_SIZE: tl.constexpr,
):
    # 1. Map program ID to row (token) index
    row_idx = tl.program_id(0)
    
    # 2. Calculate offsets for this row
    row_start_x = x_ptr + row_idx * stride_row
    row_start_y = y_ptr + row_idx * stride_row
    
    # We process the first half [0...d/2] and second half [d/2...d] simultaneously
    # Offsets for the first half
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < half_dim

    # 3. Load Data
    # Load First Half (x1)
    ptr_x1 = row_start_x + offsets
    x1 = tl.load(ptr_x1, mask=mask, other=0.0)

    # Load Second Half (x2)
    ptr_x2 = row_start_x + offsets + half_dim
    x2 = tl.load(ptr_x2, mask=mask, other=0.0)

    # Load Cosine and Sine (Assumed shape [half_dim])
    # In practice, these might broadcast along batch/seq, handled here for simplicity
    cos = tl.load(c_ptr + offsets, mask=mask, other=1.0)
    sin = tl.load(s_ptr + offsets, mask=mask, other=0.0)

    # 4. Apply Rotation (The Math)
    # y1 = x1 * cos - x2 * sin
    # y2 = x1 * sin + x2 * cos
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos

    # 5. Store Result
    tl.store(row_start_y + offsets, y1, mask=mask)
    tl.store(row_start_y + offsets + half_dim, y2, mask=mask)


# %%
# Python Wrapper
# --------------
# We create a helper function to enqueue the kernel.

def rope(x, cos, sin):
    """
    Apply RoPE to input tensor x.
    x: [Batch, Seq, Heads, Dim] or [Total_Tokens, Dim]
    cos, sin: [Dim/2]
    """
    n_rows = x.numel() // x.shape[-1]
    n_cols = x.shape[-1]
    half_dim = n_cols // 2
    
    # Create output tensor
    y = torch.empty_like(x)
    
    # Block size should cover the half_dim
    # We use next power of 2 for efficiency
    BLOCK_SIZE = triton.next_power_of_2(half_dim)
    
    # Launch Kernel
    # Grid: One program per row (token)
    grid = (n_rows,)
    
    _rope_kernel[grid](
        x, y, cos, sin,
        x.stride(0) if x.ndim == 2 else x.stride(-2), # Simplified stride logic for tutorial
        half_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return y


# %%
# Unit Test
# ---------
# We ensure the implementation matches the standard PyTorch logic.

def test_rope():
    torch.manual_seed(0)
    # Dimensions
    BATCH, SEQ, DIM = 2, 128, 64
    N_ROWS = BATCH * SEQ
    
    # Random Data
    x = torch.randn(N_ROWS, DIM, device='cuda', dtype=torch.float32)
    # Cos/Sin for half dimension
    cos = torch.randn(DIM // 2, device='cuda', dtype=torch.float32)
    sin = torch.randn(DIM // 2, device='cuda', dtype=torch.float32)

    # PyTorch Baseline
    def torch_rope_ref(x, cos, sin):
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    # Run Both
    y_torch = torch_rope_ref(x, cos, sin)
    y_triton = rope(x, cos, sin)

    # Compare
    if torch.allclose(y_triton, y_torch, atol=1e-5):
        print("Correctness Check Passed!")
    else:
        print("Correctness Check Failed!")
        print("Max diff:", (y_triton - y_torch).abs().max())


# %%
# Benchmark
# ---------
# We benchmark against PyTorch to demonstrate the speedup.

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # X-axis: Number of tokens
        x_vals=[1024 * i for i in range(2, 64, 4)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=["Triton", "PyTorch"],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='rope-performance',
        args={'DIM': 128},
    )
)
def benchmark(N, DIM, provider):
    x = torch.randn(N, DIM, device='cuda', dtype=torch.float32)
    cos = torch.randn(DIM // 2, device='cuda', dtype=torch.float32)
    sin = torch.randn(DIM // 2, device='cuda', dtype=torch.float32)
    
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'torch':
        def torch_op():
            half = DIM // 2
            return torch.cat([x[:, :half] * cos - x[:, half:] * sin, 
                              x[:, :half] * sin + x[:, half:] * cos], dim=-1)
        ms, min_ms, max_ms = triton.testing.do_bench(torch_op, quantiles=quantiles)
        
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: rope(x, cos, sin), quantiles=quantiles)
        
    # Calculate GB/s (Read X, Cos, Sin + Write Y)
    gbps = lambda ms: (3 * x.numel() * x.element_size()) * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    test_rope()
    benchmark.run(show_plots=True, print_data=True)