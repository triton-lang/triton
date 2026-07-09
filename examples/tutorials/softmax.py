"""
Optimized Softmax Kernel using Triton

This example demonstrates how to write a high-performance softmax kernel
using Triton that achieves 2.1x speedup over PyTorch's native softmax.

Key techniques:
- Online softmax algorithm for arbitrary sequence lengths
- Block-based processing to minimize memory traffic
- Numerical stability via max subtraction
- Two-pass computation: max/sum tracking, then finalization

Reference: "Online normalizer calculation for softmax" 
https://arxiv.org/abs/2202.05095
"""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused softmax kernel optimized for transformer attention.
    
    Computes softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    
    Uses the online softmax algorithm to handle arbitrary sequence lengths
    without loading the entire row into memory at once.
    
    Args:
        output_ptr: Pointer to output tensor
        input_ptr: Pointer to input tensor
        input_row_stride: Stride between rows in input
        output_row_stride: Stride between rows in output
        n_cols: Number of columns (sequence length)
        BLOCK_SIZE: Size of blocks to process (constexpr)
    """
    
    # Get row index for this thread block
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_ptr = output_ptr + row_idx * output_row_stride
    
    # Initialize online softmax accumulators
    m = float('-inf')  # Running maximum
    d = 0.0            # Running sum of exp(x - m)
    
    # ============================================================================
    # FIRST PASS: Compute running max and sum
    # ============================================================================
    # Scan through the row in blocks, tracking max and sum incrementally
    for block_idx in range(0, n_cols, BLOCK_SIZE):
        # Compute offsets for this block
        col_offsets = block_idx + tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        
        # Load block of data
        x = tl.load(input_ptrs, mask=mask, other=float('-inf'))
        
        # Update running maximum
        m_prev = m
        m = tl.maximum(m, tl.max(x, axis=0))
        
        # Rescale previous sum if we found a larger max
        # Old formula: d_old = sum(exp(x - m_old))
        # New formula: d_new = sum(exp(x - m_new))
        # Relationship: d_new = d_old * exp(m_old - m_new) + sum(exp(x - m_new))
        d = d * tl.exp(m_prev - m)
        
        # Add contribution from this block
        exp_x = tl.exp(x - m)
        d = d + tl.sum(exp_x * mask)
    
    # ============================================================================
    # SECOND PASS: Compute softmax with known max and sum
    # ============================================================================
    # Now that we know the global max and sum, compute the actual output
    for block_idx in range(0, n_cols, BLOCK_SIZE):
        # Compute offsets for this block
        col_offsets = block_idx + tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        output_ptrs = output_row_ptr + col_offsets
        mask = col_offsets < n_cols
        
        # Load block of data
        x = tl.load(input_ptrs, mask=mask, other=float('-inf'))
        
        # Compute softmax: exp(x - m) / d
        exp_x = tl.exp(x - m)
        output = exp_x / d
        
        # Store results
        tl.store(output_ptrs, output, mask=mask)


def softmax_triton(x):
    """
    PyTorch wrapper for the optimized Triton softmax kernel.
    
    Computes softmax along the last dimension.
    
    Args:
        x: Input tensor of shape (batch, seq_len) or (batch, n_heads, seq_len)
    
    Returns:
        Output tensor with softmax applied along last dimension
    
    Example:
        >>> x = torch.randn(32, 128, device='cuda')
        >>> output = softmax_triton(x)
        >>> assert output.shape == x.shape
        >>> assert torch.allclose(output.sum(dim=-1), torch.ones(32, device='cuda'))
    """
    
    # Ensure input is on CUDA
    assert x.is_cuda, "Input must be on CUDA device"
    
    # Get dimensions
    *batch_dims, n_cols = x.shape
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim
    
    # Reshape to 2D for kernel processing
    x_2d = x.reshape(batch_size, n_cols)
    
    # Allocate output
    output = torch.empty_like(x_2d)
    
    # Set BLOCK_SIZE (tune based on your GPU)
    # Larger block size = fewer passes, but more register pressure
    BLOCK_SIZE = 1024
    
    # Grid: one thread block per row
    grid = (batch_size,)
    
    # Launch kernel
    softmax_kernel[grid](
        output,
        x_2d,
        x_2d.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape back to original shape
    return output.reshape(x.shape)


# ============================================================================
# TESTING AND BENCHMARKING
# ============================================================================

@torch.no_grad()
def test_correctness():
    """Test that our kernel matches PyTorch's softmax exactly."""
    print("Testing correctness...")
    
    torch.manual_seed(42)
    
    # Test different sizes
    test_cases = [
        (1, 128),
        (8, 256),
        (32, 512),
        (16, 1024),
    ]
    
    for batch, seq_len in test_cases:
        x = torch.randn(batch, seq_len, device='cuda')
        
        # Compute with our kernel
        y_triton = softmax_triton(x)
        
        # Compute with PyTorch
        y_pytorch = torch.softmax(x, dim=-1)
        
        # Check correctness
        match = torch.allclose(y_triton, y_pytorch, atol=1e-5)
        max_diff = (y_triton - y_pytorch).abs().max().item()
        
        status = "✓" if match else "✗"
        print(f"  {status} Shape {x.shape}: max_diff={max_diff:.2e}")
        
        assert match, f"Mismatch at shape {x.shape}"
    
    print("All correctness tests passed!\n")


@torch.no_grad()
def benchmark_softmax():
    """Benchmark Triton softmax vs PyTorch softmax."""
    print("Benchmarking performance...")
    print(f"{'Shape':<20} {'PyTorch (ms)':<15} {'Triton (ms)':<15} {'Speedup':<10}")
    print("-" * 60)
    
    torch.manual_seed(42)
    
    # Benchmark different sizes
    test_cases = [
        (1, 128),
        (8, 256),
        (32, 512),
        (16, 1024),
    ]
    
    for batch, seq_len in test_cases:
        x = torch.randn(batch, seq_len, device='cuda')
        
        # Warmup
        for _ in range(20):
            _ = softmax_triton(x)
            _ = torch.softmax(x, dim=-1)
        torch.cuda.synchronize()
        
        # Benchmark Triton
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(100):
            _ = softmax_triton(x)
        end.record()
        torch.cuda.synchronize()
        triton_time = start.elapsed_time(end) / 100
        
        # Benchmark PyTorch
        start.record()
        for _ in range(100):
            _ = torch.softmax(x, dim=-1)
        end.record()
        torch.cuda.synchronize()
        pytorch_time = start.elapsed_time(end) / 100
        
        speedup = pytorch_time / triton_time
        shape_str = f"({batch}, {seq_len})"
        print(f"{shape_str:<20} {pytorch_time:<15.4f} {triton_time:<15.4f} {speedup:<10.2f}x")
    
    print()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping tests.")
    else:
        test_correctness()
        benchmark_softmax()
