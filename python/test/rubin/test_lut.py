import torch
import triton

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    TensorMemoryLUTLayout,
    allocate_tensor_memory,
    tma,
    mbarrier,
    tcgen05_mma,
    tcgen05_commit,
    tcgen05_copy,
    fence_async_shared,
)


def extract_3bit_elements(indices):
    n, k_packed = indices.shape
    k = (k_packed * 8) // 3

    device = indices.device

    # Calculate bit positions for all elements
    bit_positions = torch.arange(k, device=device) * 3
    byte_positions = bit_positions // 8
    bit_offsets = bit_positions % 8

    # Extend indices with zero column for boundary handling
    extended_indices = torch.cat([indices, torch.zeros(n, 1, dtype=torch.uint8, device=device)], dim=1)

    # Gather byte pairs for all positions (n, k)
    byte0 = extended_indices[:, byte_positions].to(torch.int32)
    byte1 = extended_indices[:, byte_positions + 1].to(torch.int32)

    # Combine into 16-bit values
    byte_pairs = byte0 | (byte1 << 8)

    # Extract 3-bit values by shifting and masking
    values = (byte_pairs >> bit_offsets) & 0b111

    return values.to(torch.uint8)


def decompress(indices, LUT):
    n = indices.shape[0]
    k = LUT.shape[1] * 64
    device = indices.device

    # Extract all 3-bit indices at once (n, k)
    extracted_indices = extract_3bit_elements(indices)

    # Calculate LUT indices for each position
    # row_lut_idx: which LUT row to use (i // 8)
    # col_lut_idx: which LUT column block to use (j // 64)
    row_lut_idx = torch.arange(n, device=device) // 8  # (n,)
    col_lut_idx = torch.arange(k, device=device) // 64  # (k,)

    # Use advanced indexing to gather from LUT
    # LUT[row_lut_idx[:, None], col_lut_idx[None, :]] gives (n, k, 8)
    # Then we index into the last dimension using extracted_indices
    luts = LUT[row_lut_idx[:, None], col_lut_idx[None, :]]  # (n, k, 8)

    # Gather values using extracted_indices as the final index
    B = torch.gather(luts, 2, extracted_indices.unsqueeze(-1).long()).squeeze(-1)

    return B.to(LUT.dtype)


def get_random_lut_and_indices(n, k):
    k_packed = k // 8 * 3

    indices = torch.randint(low=0, high=256, size=(n, k_packed), dtype=torch.uint8, device="cuda")
    LUT = (
        torch.randn((n // 8, k // 64, 8), dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn).view(torch.uint8)
    )

    return (
        LUT,
        indices,
    )


def get_shared_swizzling_zero(N, K):
    import math

    bitwidth = 8
    bases = []

    def log2_int(x):
        return x.bit_length() - 1

    for i in range(log2_int(128 // bitwidth)):
        bases.append([0, 1 << i])
    for i in range(log2_int(N)):
        bases.append([1 << i, 0])
    for i in range(log2_int(K // (128 // bitwidth))):
        offset = int(math.log2(128 // bitwidth)) + i
        bases.append([0, 1 << offset])
    return gl.SharedLinearLayout(bases)


def transform_b_to_core_matrices_layout(B, BLOCK_N, BLOCK_K):
    N, K = B.shape
    assert N % BLOCK_N == 0, f"N={N} must be divisible by BLOCK_N={BLOCK_N}"
    assert K % BLOCK_K == 0, f"K={K} must be divisible by BLOCK_K={BLOCK_K}"

    # Core matrix dimensions
    CM_ROWS = 8
    CM_COLS = 16
    assert BLOCK_N % CM_ROWS == 0, f"BLOCK_N={BLOCK_N} must be divisible by {CM_ROWS}"
    assert BLOCK_K % CM_COLS == 0, f"BLOCK_K={BLOCK_K} must be divisible by {CM_COLS}"

    num_blocks_n = N // BLOCK_N
    num_blocks_k = K // BLOCK_K
    num_cm_n = BLOCK_N // CM_ROWS
    num_cm_k = BLOCK_K // CM_COLS

    # Reshape to expose full structure: blocks and core matrices within blocks
    # [num_blocks_n, num_cm_n, CM_ROWS, num_blocks_k, num_cm_k, CM_COLS]
    B_reshaped = B.reshape(num_blocks_n, num_cm_n, CM_ROWS, num_blocks_k, num_cm_k, CM_COLS)

    # Transform to core-matrices format: group by core matrix (cm_n, cm_k) first
    # Within each block, permute from [num_cm_n, CM_ROWS, num_cm_k, CM_COLS]
    # to [num_cm_n, num_cm_k, CM_ROWS, CM_COLS]
    # In 6D tensor, apply permute(0, 2, 1, 3) to dims [1, 2, 4, 5]:
    # (0, 1, 2, 3, 4, 5) -> (0, 1, 4, 3, 2, 5)
    B_transformed = B_reshaped.permute(0, 1, 4, 3, 2, 5)
    # Now: [num_blocks_n, num_cm_n, num_cm_k, num_blocks_k, CM_ROWS, CM_COLS]

    # Rearrange blocks to be contiguous
    B_transformed = B_transformed.permute(0, 3, 1, 2, 4, 5)
    # Now: [num_blocks_n, num_blocks_k, num_cm_n, num_cm_k, CM_ROWS, CM_COLS]

    # Make the contiguous dim 128B for efficient TMA and reduce the rank to 5
    B_transformed = B_transformed.reshape(num_blocks_n, num_blocks_k, num_cm_n, num_cm_k, CM_ROWS * CM_COLS)

    return B_transformed.contiguous()


@gluon.jit
def core_matrices_to_k_major(b_smem_flat, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr):
    # Core matrix dimensions: 8 rows × 16 bytes
    CM_ROWS: gl.constexpr = 8
    CM_COLS: gl.constexpr = 16

    num_cm_n: gl.constexpr = BLOCK_N // CM_ROWS
    num_cm_k: gl.constexpr = BLOCK_K // CM_COLS

    # Reshape to expose core-matrices format: [num_cm_n, num_cm_k, CM_ROWS, CM_COLS]
    # This is how the data is logically organized after host transform + TMA
    b_smem = b_smem_flat.reshape((num_cm_n, num_cm_k, CM_ROWS, CM_COLS))

    # Permute to get the layout MMA expects: [num_cm_n, CM_ROWS, num_cm_k, CM_COLS]
    # This is the inverse of the host transform
    b_smem = b_smem.permute((0, 2, 1, 3))

    # Reshape back to [BLOCK_N, BLOCK_K]
    b_smem = b_smem.reshape((BLOCK_N, BLOCK_K))

    return b_smem


@gluon.jit
def mma_lut_kernel(
    a_desc,
    b_ptr,
    b_stride0,
    b_stride1,
    b_desc_tma,
    lut_desc,
    c_desc,
    b_layout: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    num_warps: gl.constexpr,
    use_tma_for_b: gl.constexpr,
):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    a_smem = gl.allocate_shared_memory(dtype, a_desc.block_type.shape, a_desc.layout)

    if not use_tma_for_b:
        b_smem = gl.allocate_shared_memory(gl.uint8, [BLOCK_N, BLOCK_K], b_layout)
    else:
        b_smem = gl.allocate_shared_memory(b_desc_tma.dtype, b_desc_tma.block_type.shape, b_desc_tma.layout)

    lut_smem = gl.allocate_shared_memory(gl.int8, lut_desc.block_type.shape, lut_desc.layout)

    tma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(tma_bar, count=1)
    mma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(mma_bar, count=1)
    phase = 0

    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)

    lut_tmem = allocate_tensor_memory(gl.int8, (32, 16), TensorMemoryLUTLayout())

    load_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, num_warps], [1, 0])
    offs_n = gl.arange(0, BLOCK_N, gl.SliceLayout(1, load_layout))
    offs_k = gl.arange(0, BLOCK_K, gl.SliceLayout(0, load_layout))

    block_n = pid_n

    use_acc = False

    for k_iter in range(0, K // BLOCK_K):
        k = k_iter * BLOCK_K

        if use_tma_for_b:
            block_k = k_iter
            # TMA for A, B, and LUT
            mbarrier.expect(
                tma_bar, a_desc.block_type.nbytes + b_desc_tma.block_type.nbytes + lut_desc.block_type.nbytes
            )
            tma.async_copy_global_to_shared(a_desc, [off_m, k], tma_bar, a_smem)
            tma.async_copy_global_to_shared(b_desc_tma, [block_n, block_k, 0, 0, 0], tma_bar, b_smem)
            tma.async_copy_global_to_shared(lut_desc, [off_n, k_iter, 0], tma_bar, lut_smem)
            # Apply inverse transformation: reshape and permute to get unswizzled core-matrices layout
            b_smem_transformed = core_matrices_to_k_major(b_smem, BLOCK_N, BLOCK_K)
        else:
            # TMA for A and LUT, gl.load for B
            mbarrier.expect(tma_bar, a_desc.block_type.nbytes + lut_desc.block_type.nbytes)
            tma.async_copy_global_to_shared(a_desc, [off_m, k], tma_bar, a_smem)
            tma.async_copy_global_to_shared(lut_desc, [off_n, k_iter, 0], tma_bar, lut_smem)

            b_load_ptr = b_ptr + (off_n + offs_n)[:, None] * b_stride0 + (k + offs_k)[None, :] * b_stride1
            b_data = gl.load(b_load_ptr)
            b_smem.store(b_data)
            b_smem_transformed = b_smem

        fence_async_shared()

        mbarrier.wait(tma_bar, phase=phase)

        # unswizzle lut into (32, 16) "row major": (num_lut_n, num_lut_k * 8)
        lut_smem_2d = (
            lut_smem.reshape((lut_smem.shape[0], lut_smem.shape[1], 8, 2, 8))
            .permute((0, 2, 1, 3, 4))
            .reshape((lut_smem.shape[0] * 8, lut_smem.shape[1] * 2 * 8))
        )

        tcgen05_copy(lut_smem_2d, lut_tmem)

        b = b_smem_transformed.permute((1, 0))

        tcgen05_mma(a_smem, b, acc_tmem, use_acc=use_acc, lut=lut_tmem)
        tcgen05_commit(mma_bar)
        mbarrier.wait(mma_bar, phase=phase)
        use_acc = True

        phase ^= 1

    mbarrier.invalidate(tma_bar)
    mbarrier.invalidate(mma_bar)

    acc_reg_layout: gl.constexpr = acc_tmem.get_reg_layout(num_warps=num_warps)
    acc = acc_tmem.load(acc_reg_layout)

    c_smem = gl.allocate_shared_memory(gl.float32, c_desc.block_type.shape, c_desc.layout)
    c_smem.store(acc)
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)


def matmul_lut(A, B, lut, C, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, use_tma_for_b=False):
    M, N = C.shape

    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float8e4nv)
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)

    b_layout = get_shared_swizzling_zero(BLOCK_N, BLOCK_K)

    if use_tma_for_b:
        # Transform B to core-matrices format
        # [num_blocks_n, num_blocks_k, num_cm_n, num_cm_k, CM_ROWS * CM_COLS]
        B_transformed = transform_b_to_core_matrices_layout(B, BLOCK_N, BLOCK_K)
        num_cm_n = BLOCK_N // 8
        num_cm_k = BLOCK_K // 16

        b_layout_tma = gl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=8, rank=5)
        b_desc_tma = TensorDescriptor.from_tensor(B_transformed, [1, 1, num_cm_n, num_cm_k, 128], b_layout_tma)
    else:
        b_desc_tma = None

    # Copy 4 * 8 luts along M and 2 luts along K (corresponding to BLOCK_M = 256 and BLOCK_K = 128)
    lut_tile_shape = [4, 1, 128]
    lut_layout = gl.NVMMASharedLayout.get_default_for(lut_tile_shape, gl.int8)
    lut_desc = TensorDescriptor.from_tensor(lut, lut_tile_shape, lut_layout)

    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float32)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    out = mma_lut_kernel[grid](
        a_desc,
        B,
        B.stride(0),
        B.stride(1),
        b_desc_tma,
        lut_desc,
        c_desc,
        b_layout,
        BLOCK_N,
        BLOCK_K,
        num_warps=num_warps,
        use_tma_for_b=use_tma_for_b,
    )

    # print(out.asm["ttgir"])


def test_lut(use_tma_for_b=False):
    K = 128
    N = 256
    M = 128

    A = torch.randn(M, K, device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
    C = torch.empty(M, N, device="cuda", dtype=torch.float32)

    LUT_orig, B_indices = get_random_lut_and_indices(N, K)

    # One chunk: 8x2 LUTs, k major, 128B
    num_chunk_n = LUT_orig.shape[0] // 8
    num_chunk_k = LUT_orig.shape[1] // 2

    # 128B chunks are contiguously laid out so that they can be efficiently copied by TMA
    # Depending on BLOCK_N, multiple chunks along N are copied at once (e.g. 4 for BLOCK_N = 256)
    LUT = (
        LUT_orig.reshape(num_chunk_n, 8, num_chunk_k, 2, 8)
        .permute(0, 2, 1, 3, 4)
        .reshape(num_chunk_n, num_chunk_k, 128)
        .contiguous()
    )

    B_indices_padded = torch.nn.functional.pad(B_indices, (0, 128 - B_indices.shape[1]), "constant", 0)
    matmul_lut(A, B_indices_padded, LUT, C, 128, 256, 128, 4, use_tma_for_b=use_tma_for_b)

    B_uint8 = decompress(B_indices, LUT_orig)
    B = B_uint8.view(torch.float8_e4m3fn)

    C_ref = A.to(torch.float32) @ B.T.to(torch.float32)

    torch.testing.assert_close(C_ref, C, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    print("Testing with gl.load for B...")
    test_lut(use_tma_for_b=False)

    print("\nTesting with TMA for B...")
    test_lut(use_tma_for_b=True)
