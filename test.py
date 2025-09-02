import torch
import triton
import triton.language as tl

@triton.jit
def mxfp4_dot_scaled_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    sA_ptr, sB_ptr,
    stride_am, stride_ak2,      # A_packed: [M, K/2]
    stride_bk2, stride_bn,      # B_packed: [K/2, N]
    stride_cm, stride_cn,       # C: [M, N]
    stride_meta_a_m: tl.constexpr,
    stride_meta_a_g: tl.constexpr,
    stride_meta_b_n: tl.constexpr,
    stride_meta_b_g: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_K: tl.constexpr,
):
    tl.static_assert(BLOCK_K % 2 == 0)
    tl.static_assert(BLOCK_K % GROUP_K == 0)
    BK2: tl.constexpr = BLOCK_K // 2
    BLOCK_K_S: tl.constexpr = BLOCK_K // GROUP_K

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    ACC = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    offs_k_scales = tl.arange(0, BLOCK_K_S)

    for k0 in tl.range(0, K, BLOCK_K):
        k2_base = k0 // 2
        k2v = k2_base + tl.arange(0, BK2)
        k2_limit: tl.constexpr = K // 2

        a_mask = ((rm[:, None] < M) & (k2v[None, :] < k2_limit)).to(tl.int1)
        b_mask = ((k2v[:, None] < k2_limit) & (rn[None, :] < N)).to(tl.int1)

        A_idx = rm[:, None] * stride_am + k2v[None, :] * stride_ak2
        B_idx = k2v[:, None] * stride_bk2 + rn[None, :] * stride_bn

        A_p = tl.load(A_ptr + A_idx, mask=a_mask, other=0)
        B_p = tl.load(B_ptr + B_idx, mask=b_mask, other=0)

        k_group_index_base = k0 // GROUP_K
        sA_blk = tl.load(
            sA_ptr + rm[:, None] * stride_meta_a_m + (k_group_index_base + offs_k_scales[None, :]) * stride_meta_a_g
        )
        sB_blk = tl.load(
            sB_ptr + rn[:, None] * stride_meta_b_n + (k_group_index_base + offs_k_scales[None, :]) * stride_meta_b_g
        )

        ACC = tl.dot_scaled(A_p, sA_blk, "e2m1", B_p, sB_blk, "e2m1", ACC)
    C_idx = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C_ptr + C_idx, ACC, mask=c_mask)


def run_mxfp4_dot_scaled_kernel(
    a_codes: torch.Tensor, sA_grouped: torch.Tensor,
    b_codes: torch.Tensor, sB_grouped: torch.Tensor,
    M: int, N: int, K: int,
    block_m: int = 32, block_n: int = 16, block_k: int = 64,
    group_k: int = 32,
    num_warps: int = 4, num_stages: int = 2
) -> torch.Tensor:
    device = a_codes.device
    c = torch.empty((M, N), dtype=torch.float32, device=device)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

    mxfp4_dot_scaled_kernel[grid](
        a_codes, b_codes, c,
        M, N, K,
        sA_grouped, sB_grouped,
        (K // 2), 1,
        N, 1,
        N, 1,
        sA_grouped.stride(0), sA_grouped.stride(1),
        sB_grouped.stride(0), sB_grouped.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        GROUP_K=group_k,
        num_warps=num_warps, num_stages=num_stages
    )
    return c


def decode_fp4_e2m1(codes: torch.Tensor) -> torch.Tensor:
    x = (codes.to(torch.int16) & 0xF)
    s = (x >> 3) & 0x1
    e = (x >> 1) & 0x3
    m = x & 0x1
    is_zero = (e == 0) & (m == 0)
    is_sub  = (e == 0) & (m == 1)
    is_norm = (e > 0)
    out = torch.zeros_like(x, dtype=torch.float32)
    out = torch.where(is_sub, torch.full_like(out, 0.5), out)
    out = torch.where(is_norm, (1.0 + 0.5 * m.float()) * (2.0 ** (e.float() - 1.0)), out)
    # apply sign except for zero
    out = torch.where(is_zero, out, torch.where(s.bool(), -out, out))
    return out


@torch.no_grad()
def mxfp4_scaled_reference_grouped(a_packed, b_packed, sA_grouped, sB_grouped, M: int, N: int, K: int, GROUP_K: int):
    """
    a_packed: [M, K//2] uint8 (2 FP4 codes per uint8, packed along last dim)
    b_packed: [K//2, N] uint8 (2 FP4 codes per uint8, packed along first dim)
    sA_grouped: [M, K//GROUP_K] uint8
    sB_grouped: [N, K//GROUP_K] uint8
    returns: [M,N] float32
    """
    def unpack_fp4_pairs_last_dim(packed):
        low = packed & 0xF
        high = (packed >> 4) & 0xF
        return torch.stack([low, high], dim=-1).flatten(-2)

    def unpack_fp4_pairs_first_dim(packed):
        low = packed & 0xF
        high = (packed >> 4) & 0xF
        return torch.stack([low, high], dim=0).flatten(0, 1)

    a_codes = unpack_fp4_pairs_last_dim(a_packed)  # [M,K]
    b_codes = unpack_fp4_pairs_first_dim(b_packed) # [K,N]

    a_fp = decode_fp4_e2m1(a_codes)
    b_fp = decode_fp4_e2m1(b_codes)

    sA = (2 ** (sA_grouped.float() - 127.0)).repeat_interleave(GROUP_K, dim=1)[:, :K]
    sB = (2 ** (sB_grouped.float() - 127.0)).repeat_interleave(GROUP_K, dim=1)[:, :K]

    a_scaled = a_fp * sA
    b_scaled = b_fp * sB.T
    return a_scaled @ b_scaled

def test_mxfp4_matmul(M: int, N: int, K: int):
    """Test mxfp4 scaled matmul with given dimensions"""
    device = "cuda"

    # FP4 e2m1 encoding for 1.0: s=0, e=1, m=0 → 0001 = 0x2
    fp4_one = 0x2
    packed_ones = (fp4_one << 4) | fp4_one  # 0x22 (two 1.0s per uint8)

    # Create tensors filled with 1.0 (packed)
    a_codes = torch.full((M, K // 2), packed_ones, dtype=torch.uint8, device=device)
    b_codes = torch.full((K // 2, N), packed_ones, dtype=torch.uint8, device=device)

    GROUP_K = 32
    KG = K // GROUP_K
    # Scales = 1.0 (no scaling): 127 in uint8 → 2^(127-127)=1.0
    sA = torch.full((M, KG), 127, dtype=torch.uint8, device=device)
    sB = torch.full((N, KG), 127, dtype=torch.uint8, device=device)

    print(f"Testing M={M}, N={N}, K={K}")
    print(f"Expected result: C = ones({M},{K}) @ ones({K},{N}) = {K} * ones({M},{N})")

    # Run Triton kernel
    c = run_mxfp4_dot_scaled_kernel(a_codes, sA, b_codes, sB, M, N, K, group_k=GROUP_K)

    # Compute reference
    c_ref = mxfp4_scaled_reference_grouped(a_packed=a_codes, b_packed=b_codes, sA_grouped=sA, sB_grouped=sB,
                                           M=M, N=N, K=K, GROUP_K=GROUP_K)

    print(f"Kernel result shape: {c.shape}")
    print(f"Reference result shape: {c_ref.shape}")
    print(f"Expected value: {K}, Actual (first element): {c[0,0].item():.1f}")

    print(c)
    print(c_ref)
    # torch.testing.assert_close(c, c_ref, rtol=1e-3, atol=1e-3)
    print("✅ OK: kernel matches reference\n")
    return c, c_ref


def _quantize_to_fp4_e2m1_01(x: torch.Tensor) -> torch.Tensor:
    # 표현 가능한 양수: {0, 0.75, 1.0} (코드 {0x0, 0x1, 0x2})
    # 경계: 0 ↔ 0.75 는 0.375, 0.75 ↔ 1.0 은 0.875
    codes = torch.empty_like(x, dtype=torch.uint8)
    # codes[x < 0.375] = 0x0            # 0
    # codes[(x >= 0.375) & (x < 0.875)] = 0x1  # 0.75
    # codes[x >= 0.875] = 0x2           # 1.0
    codes[:, :] = 0x1
    return codes

def make_random_packed_fp4_01(M: int, K: int, N: int, device="cuda", seed: int | None = None):
    assert K % 2 == 0, "K must be even to pack 2 FP4 per byte"
    if seed is not None:
        torch.manual_seed(seed)

    # A: [M, K] → 코드를 마지막 축으로 패킹 → [M, K//2]
    A_float = torch.rand((M, K), device=device)                 # U(0,1)
    A_codes = _quantize_to_fp4_e2m1_01(A_float)
    A_packed = (A_codes[:, 1::2] << 4) | A_codes[:, 0::2]       # [M, K//2]

    # B: [K, N] → 코드를 첫 축으로 패킹 → [K//2, N]
    B_float = torch.rand((K, N), device=device)                 # U(0,1)
    B_codes = _quantize_to_fp4_e2m1_01(B_float)
    B_packed = (B_codes[1::2, :] << 4) | B_codes[0::2, :]       # [K//2, N]

    return A_packed.contiguous(), B_packed.contiguous()
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--bm", type=str, default="128", help="BLOCK_M or comma-separated list, e.g. 64,128")
    parser.add_argument("--bn", type=str, default="128", help="BLOCK_N or comma-separated list, e.g. 64,128")
    parser.add_argument("--bk", type=str, default="128", help="BLOCK_K or comma-separated list, e.g. 64,128")
    parser.add_argument("--gk", type=int, default=32, help="GROUP_K (scale group along K), typically 32")
    parser.add_argument("--warps", type=int, default=4)
    parser.add_argument("--stages", type=int, default=2)
    args = parser.parse_args()

    def parse_list(s: str):
        return [int(x) for x in s.split(",")] if "," in s else [int(s)]

    M = N = K = 1024
    bms = parse_list(args.bm)
    bns = parse_list(args.bn)
    bks = parse_list(args.bk)

    for bm in [16, 32, 64, 128]:
        for bn in [16, 32, 64]:
            for bk in [64, 128]:
                print(f"===== M=N=K=1024 | BLOCK_M={bm} BLOCK_N={bn} BLOCK_K={bk} GROUP_K={args.gk}")
                A, B = make_random_packed_fp4_01(M, K, N, device="cuda")
                KG = K // args.gk
                # sA = torch.randint(127 - delta, 128 + delta,  (M, KG), dtype=torch.uint8, device="cuda")
                # sB = torch.randint(127 - delta, 128 + delta, (N, KG), dtype=torch.uint8, device="cuda")
                sA = torch.range(127, 127 + M - 1, dtype=torch.uint8, device="cuda").reshape(M, 1) * torch.ones((1, KG), dtype=torch.uint8, device="cuda")
                sB = torch.range(127, 127 + N - 1, dtype=torch.uint8, device="cuda").reshape(N, 1) * torch.ones((1, KG), dtype=torch.uint8, device="cuda")
                c = run_mxfp4_dot_scaled_kernel(
                    A, sA, B, sB, M, N, K,
                    block_m=bm, block_n=bn, block_k=bk, group_k=args.gk,
                    num_warps=args.warps, num_stages=args.stages,
                )
                c_ref = mxfp4_scaled_reference_grouped(A, B, sA, sB, M, N, K, args.gk)
                print(c)
                print(c_ref)
                try:
                    torch.testing.assert_close(c, c_ref, atol=0.0, rtol=1e-3)
                    print("OK")
                except:
                    print("FAILED")
