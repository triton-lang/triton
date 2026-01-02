import pytest
import torch
from triton._internal_testing import is_cuda, is_hip, is_hip_cdna3, is_hip_cdna4
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor


def supports_block_scaling():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


@pytest.mark.parametrize("m, n, k", [(16, 16, 16), (32, 16, 16), (16, 32, 16), (16, 16, 32)])
@pytest.mark.parametrize("dtype_str", ["float8_e4m3fn", "float8_e4m3fnuz", "float16"])
def test_blaslt(m, n, k, dtype_str, device):
    dtype = getattr(torch, dtype_str)

    if is_cuda():
        from triton._C.libtriton import nvidia as vendor
        if dtype_str == "float8_e4m3fnuz":
            pytest.skip("float8_e4m3fnuz is not supported on CUDA")
        if dtype == torch.float8_e4m3fn and torch.cuda.get_device_capability()[0] < 9:
            pytest.skip("fp8 is only supported on CUDA with cc >= 90")
        c_dtype = dtype
        make_handle = lambda workspace: vendor.cublas.CublasLt(workspace)
    elif is_hip():
        from triton._C.libtriton import amd as vendor
        if dtype_str == "float8_e4m3fnuz" and not is_hip_cdna3():
            pytest.skip("float8_e4m3fnuz is only supported on HIP CDNA3")
        if dtype_str == "float8_e4m3fn" and not is_hip_cdna4():
            pytest.skip("float8_e4m3fn is only supported on HIP CDNA4")
        c_dtype = torch.float16 if dtype_str in ("float8_e4m3fnuz", "float8_e4m3fn") else dtype
        make_handle = lambda workspace: vendor.hipblas.HipblasLt(workspace)
    else:
        pytest.skip("test_blaslt is only supported on CUDA or HIP")

    torch.manual_seed(123)
    workspace_size = 32 * 1024 * 1024

    def limited_rand(elements, shape):
        total_elems = torch.prod(torch.tensor(shape)).item()
        indices = torch.randint(0, len(elements), (total_elems, ), device=device)
        return elements[indices].view(shape)

    elements = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32, device=device)
    a = limited_rand(elements, (m, k)).to(dtype)
    b = limited_rand(elements, (k, n)).to(dtype)

    c = torch.zeros((m, n), dtype=c_dtype, device=device)

    b = b.T.contiguous()

    workspace = torch.empty(workspace_size, dtype=torch.int8, device=device)
    handle = make_handle(workspace)

    handle.matmul(a, b, c)

    ref = torch.matmul(a.to(torch.float16), b.to(torch.float16).T)

    assert torch.allclose(c.to(torch.float16), ref, atol=2.0)


@pytest.mark.parametrize("m, n, k", [(256, 256, 512), (512, 512, 512), (1024, 1024, 1024)])
def test_block_scaled_matmul_mxfp8(m, n, k, device):
    """Test block-scaled matmul with MXFP8 format (FP8 E4M3 inputs, E8M0 scales)."""
    if not is_cuda():
        pytest.skip("block_scaled_matmul is only supported on CUDA")
    if not supports_block_scaling():
        pytest.skip("block_scaled_matmul requires compute capability 10.0 (Blackwell)")

    from triton._C.libtriton import nvidia

    torch.manual_seed(42)

    # Constants for MXFP8
    VEC_SIZE = 32  # 32-element groups for E8M0 scales

    # Create workspace and cuBLAS handle
    workspace_size = 32 * 1024 * 1024
    workspace = torch.empty(workspace_size, dtype=torch.uint8, device=device)
    handle = nvidia.cublas.CublasLt(workspace)

    # Generate random FP8 inputs
    a_fp32 = torch.randn(m, k, device=device, dtype=torch.float32)
    b_fp32 = torch.randn(n, k, device=device, dtype=torch.float32)

    # Convert to FP8 E4M3
    a = a_fp32.to(torch.float8_e4m3fn)
    b = b_fp32.to(torch.float8_e4m3fn)

    # Generate scales in the expected 4D layout, then reshape to 5D and flatten
    # Scale shape: [M // 128, K // VEC_SIZE // 4, 32, 16]
    a_scale_shape = [m // 128, k // VEC_SIZE // 4, 32, 16]
    b_scale_shape = [n // 128, k // VEC_SIZE // 4, 32, 16]

    epsilon = 1e-8
    a_scale_raw = torch.rand(a_scale_shape, device=device) + epsilon
    b_scale_raw = torch.rand(b_scale_shape, device=device) + epsilon

    # Convert to MXScaleTensor (E8M0 format)
    a_scale_mx = MXScaleTensor(a_scale_raw)
    b_scale_mx = MXScaleTensor(b_scale_raw)
    a_scale = a_scale_mx.data
    b_scale = b_scale_mx.data

    # Reshape to 5D for TMA and flatten for cuBLAS
    a_scale_5d = a_scale.reshape(1, a_scale_shape[0], a_scale.shape[1], 2, 256)
    b_scale_5d = b_scale.reshape(1, b_scale_shape[0], b_scale.shape[1], 2, 256)
    a_scale_cublas = a_scale_5d.contiguous().flatten()
    b_scale_cublas = b_scale_5d.contiguous().flatten()

    # Prepare output tensor
    output = torch.empty((m, n), dtype=torch.float16, device=device)

    # Call cuBLAS block-scaled matmul
    handle.block_scaled_matmul_mxfp8(a, b, output, a_scale_cublas, b_scale_cublas)

    # Compute reference using PyTorch
    def unpack_scale(packed):
        packed = packed.reshape(*packed.shape[:-2], 32, 4, 4)
        num_chunk_m, num_chunk_k, _, _, _ = packed.shape
        return packed.permute(0, 3, 2, 1, 4).reshape(num_chunk_m * 128, num_chunk_k * 4).contiguous()

    a_scale_ref = a_scale_mx.to(torch.float32)
    b_scale_ref = b_scale_mx.to(torch.float32)
    a_scale_ref = unpack_scale(a_scale_ref).repeat_interleave(VEC_SIZE, dim=1)[:m, :k]
    b_scale_ref = unpack_scale(b_scale_ref).repeat_interleave(VEC_SIZE, dim=1).T.contiguous()[:k, :n]

    ref = torch.matmul(a.to(torch.float32) * a_scale_ref, b.to(torch.float32).T * b_scale_ref)

    torch.testing.assert_close(output.to(torch.float32), ref, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("m, n, k", [(256, 256, 512), (512, 512, 512), (1024, 1024, 1024)])
def test_block_scaled_matmul_nvfp4(m, n, k, device):
    """Test block-scaled matmul with NVFP4 format (packed FP4 inputs, FP8 E4M3 scales)."""
    if not is_cuda():
        pytest.skip("block_scaled_matmul is only supported on CUDA")
    if not supports_block_scaling():
        pytest.skip("block_scaled_matmul requires compute capability 10.0 (Blackwell)")

    from triton._C.libtriton import nvidia

    torch.manual_seed(42)

    # Constants for NVFP4
    VEC_SIZE = 16  # 16-element groups for FP8 E4M3 scales

    # Create workspace and cuBLAS handle
    workspace_size = 32 * 1024 * 1024
    workspace = torch.empty(workspace_size, dtype=torch.uint8, device=device)
    handle = nvidia.cublas.CublasLt(workspace)

    # Generate random MXFP4 tensors
    a_ref = MXFP4Tensor(size=(m, k), device=device).random()
    b_ref = MXFP4Tensor(size=(n, k), device=device).random()

    # Pack two FP4 elements per byte along K dimension
    a = a_ref.to_packed_tensor(dim=1)  # (M, K//2) in uint8
    b = b_ref.to_packed_tensor(dim=1)  # (N, K//2) in uint8

    # Generate scales in the expected 4D layout
    # Scale shape: [M // 128, K // VEC_SIZE // 4, 32, 16]
    a_scale_shape = [m // 128, k // VEC_SIZE // 4, 32, 16]
    b_scale_shape = [n // 128, k // VEC_SIZE // 4, 32, 16]

    epsilon = 1e-8
    a_scale_raw = torch.rand(a_scale_shape, device=device) + epsilon
    b_scale_raw = torch.rand(b_scale_shape, device=device) + epsilon

    # For NVFP4, scales are FP8 E4M3
    a_scale = a_scale_raw.to(torch.float8_e4m3fn)
    b_scale = b_scale_raw.to(torch.float8_e4m3fn)

    # Flatten for cuBLAS (use original 4D layout, not 5D reshaped)
    a_scale_cublas = a_scale.contiguous().flatten()
    b_scale_cublas = b_scale.contiguous().flatten()

    # Prepare output tensor
    output = torch.empty((m, n), dtype=torch.float16, device=device)

    # Call cuBLAS block-scaled matmul
    handle.block_scaled_matmul_nvfp4(a, b, output, a_scale_cublas, b_scale_cublas)

    # Compute reference using PyTorch
    def unpack_scale(packed):
        packed = packed.reshape(*packed.shape[:-2], 32, 4, 4)
        num_chunk_m, num_chunk_k, _, _, _ = packed.shape
        return packed.permute(0, 3, 2, 1, 4).reshape(num_chunk_m * 128, num_chunk_k * 4).contiguous()

    a_scale_ref = a_scale.to(torch.float32)
    b_scale_ref = b_scale.to(torch.float32)
    a_scale_ref = unpack_scale(a_scale_ref).repeat_interleave(VEC_SIZE, dim=1)[:m, :k]
    b_scale_ref = unpack_scale(b_scale_ref).repeat_interleave(VEC_SIZE, dim=1).T.contiguous()[:k, :n]

    ref = torch.matmul(a_ref.to(torch.float32) * a_scale_ref, b_ref.to(torch.float32).T * b_scale_ref)

    torch.testing.assert_close(output.to(torch.float32), ref, atol=1e-1, rtol=1e-1)
