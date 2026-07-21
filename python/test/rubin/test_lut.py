import pytest
import torch
import triton

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    tma,
    mbarrier,
    tcgen05_commit,
    tcgen05_copy,
    fence_async_shared,
)
from triton.experimental.gluon.language.nvidia.rubin import TensorMemoryLUTLayout, tcgen05_mma
from triton._internal_testing import is_rubin


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
    LUT = (torch.randn((n // 8, k // 64, 8), dtype=torch.float32,
                       device="cuda").to(torch.float8_e4m3fn).view(torch.uint8))

    return (
        LUT,
        indices,
    )


def pack_lut_for_tma(lut):
    assert lut.dim() == 3, f"expected logical LUT tensor of rank 3, got {lut.shape}"
    assert lut.shape[2] == 8, f"expected 8 LUT entries, got {lut.shape[2]}"
    assert lut.shape[0] % 32 == 0, f"lut rows={lut.shape[0]} must be divisible by 32"
    assert lut.shape[1] % 2 == 0, f"lut K-tiles={lut.shape[1]} must be divisible by 2"
    num_chunk_n = lut.shape[0] // 32
    num_chunk_k = lut.shape[1] // 2
    # One top-level chunk corresponds to one 32x2 collection of LUT tables,
    # i.e. four adjacent 128B rows that one 512B tmem_copy operation copies.
    return (lut.reshape(num_chunk_n, 32, num_chunk_k, 2,
                        8).reshape(num_chunk_n, 4, 8, num_chunk_k, 2,
                                   8).permute(0, 3, 1, 2, 4, 5).reshape(num_chunk_n, num_chunk_k, 4, 128).contiguous())


def make_packed_b_tma_descriptor(B, BLOCK_N, BLOCK_K_PACKED, is_n_major=True):
    B_transformed = transform_b_to_core_matrices_layout(B, BLOCK_N, BLOCK_K_PACKED, is_n_major=is_n_major)
    num_cm_n = BLOCK_N // 8
    num_cm_k = BLOCK_K_PACKED // 16
    b_layout_tma = gl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=8, rank=5)
    if is_n_major:
        b_block_shape = [1, 1, num_cm_k, num_cm_n, 128]
    else:
        b_block_shape = [1, 1, num_cm_n, num_cm_k, 128]
    return TensorDescriptor.from_tensor(B_transformed, b_block_shape, b_layout_tma)


def make_lut_tma_descriptor(lut, BLOCK_N, BLOCK_K):
    assert BLOCK_N % 256 == 0, f"BLOCK_N={BLOCK_N} must be divisible by 256"
    assert BLOCK_K % 128 == 0, f"BLOCK_K={BLOCK_K} must be divisible by 128"
    lut_tile_shape = [BLOCK_N // 256, BLOCK_K // 128, 4, 128]
    # After lut_to_tmem_layout, an unswizzled 4D tile linearizes to the
    # canonical 32x16B-per-copy LUT layout, including for wider K.
    lut_layout = gl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=8, rank=4)
    return TensorDescriptor.from_tensor(lut, lut_tile_shape, lut_layout)


def transform_b_to_core_matrices_layout(B, BLOCK_N, BLOCK_K, is_n_major=True):
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

    # Transform to core-matrices format and expose (cm_n, cm_k) axes.
    # Within each block, permute from [num_cm_n, CM_ROWS, num_cm_k, CM_COLS]
    # to [num_cm_n, num_cm_k, CM_ROWS, CM_COLS]
    # In 6D tensor, apply permute(0, 2, 1, 3) to dims [1, 2, 4, 5]:
    # (0, 1, 2, 3, 4, 5) -> (0, 1, 4, 3, 2, 5)
    B_transformed = B_reshaped.permute(0, 1, 4, 3, 2, 5)
    # Now: [num_blocks_n, num_cm_n, num_cm_k, num_blocks_k, CM_ROWS, CM_COLS]

    # For n-major tiling, make the packed K core-matrix count the outer
    # logical-count dimension. This is the layout accepted by the limited
    # non-pow2 shared-layout invariant.
    if is_n_major:
        B_transformed = B_transformed.permute(0, 3, 2, 1, 4, 5)
    else:
        B_transformed = B_transformed.permute(0, 3, 1, 2, 4, 5)

    # Make the contiguous dim 128B for efficient TMA and reduce to rank 5.
    B_transformed = B_transformed.reshape(num_blocks_n, num_blocks_k, num_cm_k, num_cm_n, CM_ROWS * CM_COLS)

    return B_transformed.contiguous()


@gluon.jit
def core_matrices_to_operand_layout(b_smem_flat, BLOCK_N: gl.constexpr, BLOCK_K_PACKED: gl.constexpr,
                                    is_n_major: gl.constexpr):
    # Core matrix dimensions: 8 rows × 16 bytes
    CM_ROWS: gl.constexpr = 8
    CM_COLS: gl.constexpr = 16

    num_cm_n: gl.constexpr = BLOCK_N // CM_ROWS
    num_cm_k: gl.constexpr = BLOCK_K_PACKED // CM_COLS

    if is_n_major:
        b_smem = b_smem_flat.reshape((num_cm_k, num_cm_n, CM_ROWS, CM_COLS))
        b_smem = b_smem.permute((1, 2, 0, 3))
    else:
        b_smem = b_smem_flat.reshape((num_cm_n, num_cm_k, CM_ROWS, CM_COLS))
        b_smem = b_smem.permute((0, 2, 1, 3))

    # Reshape back to [BLOCK_N, BLOCK_K]
    b_smem = b_smem.reshape((BLOCK_N, BLOCK_K_PACKED))

    return b_smem


@gluon.jit
def lut_to_tmem_layout(lut_smem):
    # Expose each 512B tmem_copy tile as 32 rows by 16 columns, then concatenate
    # successive BLOCK_K/128 LUT groups along the TMEM column dimension.
    # This is the inverse of the host-side transform pack_lut_for_tma
    return (lut_smem.reshape((lut_smem.shape[0], lut_smem.shape[1], 32, 16)).permute((0, 2, 1, 3)).reshape(
        (lut_smem.shape[0] * 32, lut_smem.shape[1] * 16)))


@gluon.jit
def mma_lut_kernel(
    a_desc,
    b_desc_tma,
    lut_desc,
    c_desc,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    BLOCK_K_PACKED: gl.constexpr,
    b_is_n_major: gl.constexpr,
    num_warps: gl.constexpr,
):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    a_smem = gl.allocate_shared_memory(dtype, a_desc.block_type.shape, a_desc.layout)
    b_smem = gl.allocate_shared_memory(b_desc_tma.dtype, b_desc_tma.block_type.shape, b_desc_tma.layout)
    lut_smem = gl.allocate_shared_memory(gl.int8, lut_desc.block_type.shape, lut_desc.layout)

    tma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(tma_bar, count=1)
    mma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(mma_bar, count=1)
    phase = 0

    # The TMEM encoding describes the 128-row MMA instruction tile. The
    # allocation may contain multiple such tiles when BLOCK_M is 256.
    tmem_layout: gl.constexpr = TensorMemoryLayout([128, BLOCK_N], col_stride=1)
    acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)

    num_lut_rows: gl.constexpr = BLOCK_N // 8
    num_lut_cols: gl.constexpr = BLOCK_K // 128 * 16
    lut_tmem = allocate_tensor_memory(gl.int8, (num_lut_rows, num_lut_cols), TensorMemoryLUTLayout())

    block_n = pid_n
    lut_block_n = pid_n * (BLOCK_N // 256)

    use_acc = False

    for k_iter in range(0, K // BLOCK_K):
        k = k_iter * BLOCK_K
        lut_block_k = k_iter * (BLOCK_K // 128)

        block_k = k_iter
        mbarrier.expect(tma_bar, a_desc.block_type.nbytes + b_desc_tma.block_type.nbytes +
                        lut_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(a_desc, [off_m, k], tma_bar, a_smem)
        tma.async_copy_global_to_shared(b_desc_tma, [block_n, block_k, 0, 0, 0], tma_bar, b_smem)
        tma.async_copy_global_to_shared(lut_desc, [lut_block_n, lut_block_k, 0, 0], tma_bar, lut_smem)
        # Apply inverse transformation to get the operand's core-matrix layout.
        b_smem_transformed = core_matrices_to_operand_layout(b_smem, BLOCK_N, BLOCK_K_PACKED, b_is_n_major)

        fence_async_shared()

        mbarrier.wait(tma_bar, phase=phase)

        tcgen05_copy(lut_to_tmem_layout(lut_smem), lut_tmem)

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


def matmul_lut(
    A,
    B,
    lut,
    C,
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    BLOCK_K_PACKED,
    num_warps,
    b_is_n_major=True,
):
    M, N = C.shape

    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float8e4nv)
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)

    b_desc_tma = make_packed_b_tma_descriptor(B, BLOCK_N, BLOCK_K_PACKED, is_n_major=b_is_n_major)

    lut_desc = make_lut_tma_descriptor(lut, BLOCK_N, BLOCK_K)

    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float32)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    mma_lut_kernel[grid](
        a_desc,
        b_desc_tma,
        lut_desc,
        c_desc,
        BLOCK_N,
        BLOCK_K,
        BLOCK_K_PACKED,
        b_is_n_major,
        num_warps=num_warps,
    )


@pytest.mark.skipif(not is_rubin(), reason="Requires Rubin")
@pytest.mark.parametrize("BLOCK_M", [128, 256])
@pytest.mark.parametrize("BLOCK_K", [128, 256])
@pytest.mark.parametrize("b_is_n_major", [True, False], ids=["n_major", "k_major"])
def test_lut_packed_b_tma(BLOCK_M, BLOCK_K, b_is_n_major):
    K = 4096
    N = 4096
    M = 4096
    BLOCK_N = 256
    BLOCK_K_PACKED = BLOCK_K * 3 // 8

    A = torch.randn(M, K, device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
    C = torch.empty(M, N, device="cuda", dtype=torch.float32)

    LUT_orig, B_indices_packed = get_random_lut_and_indices(N, K)
    LUT = pack_lut_for_tma(LUT_orig)

    if not b_is_n_major:
        pytest.xfail("k_major core-matrix tiling does not satisfy the limited non-pow2 shared-layout invariant")

    matmul_lut(
        A,
        B_indices_packed,
        LUT,
        C,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        BLOCK_K_PACKED,
        4,
        b_is_n_major=b_is_n_major,
    )

    B_uint8 = decompress(B_indices_packed, LUT_orig)
    B = B_uint8.view(torch.float8_e4m3fn)
    C_ref = A.to(torch.float32) @ B.T.to(torch.float32)
    torch.testing.assert_close(C_ref, C, rtol=1e-3, atol=1e-3)
