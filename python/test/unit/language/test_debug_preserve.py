import triton
import triton.language as tl
import torch

def test_debug_preserve_kernel_execution():
    """Test that a kernel with debug_preserve can be executed."""
    
    @triton.jit
    def kernel_with_preserve(input_ptr, output_ptr, n: tl.constexpr):
        idx = tl.arange(0, n)
        x = tl.load(input_ptr + idx)
        
        # Use debug_preserve 
        tl.debug_preserve(x)

    # Simple test
    n = 32
    x = torch.randn(n, device='cuda', dtype=torch.float32)
    y = torch.zeros(n, device='cuda', dtype=torch.float32)
    
    kernel_with_preserve[(1,)](x, y, n)
