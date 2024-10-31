import triton
import triton.language as tl


@triton.jit()
def streamk_gemm(
    A,
    B,
    C,
    bias_ptr,
    P,
    locks,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bias,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = (pid % NUM_XCDS) * (NUM_SMS // NUM_XCDS) + (pid // NUM_XCDS)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)
    total_tiles = num_pid_m * num_pid_n
    if NUM_SMS > 0 and total_tiles > NUM_SMS:
        total_streamk_tiles = total_tiles % NUM_SMS
        # total_streamk_tiles = total_streamk_tiles + NUM_SMS
        total_full_tiles = total_tiles - total_streamk_tiles
        total_streamk_iters = total_streamk_tiles * iters_per_tile
        streamk_iters_pcu = total_streamk_iters // NUM_SMS
        streamk_remainder_iters = total_streamk_iters % NUM_SMS
    else:
        total_full_tiles = total_tiles
        total_streamk_tiles = 0
        streamk_iters_pcu = 0
        streamk_remainder_iters = 0
        total_streamk_iters = 0

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32

    for tile_id in range(pid, total_full_tiles, NUM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rk = tl.arange(0, BLOCK_SIZE_K)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            A_BASE = tl.multiple_of(A_BASE, (1, 16))
            B_BASE = tl.multiple_of(B_BASE, (16, 1))
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)
            acc += tl.dot(a, b)

        c = acc.to(C.type.element_ty)
        if BIAS:
            c += bias[:, None]

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.store(C_, c, mask=mask)

    start_iter = total_full_tiles * iters_per_tile + pid * streamk_iters_pcu + tl.minimum(pid, streamk_remainder_iters)
    last_iter = total_full_tiles * iters_per_tile + (pid + 1) * streamk_iters_pcu + tl.minimum(
        pid + 1, streamk_remainder_iters)
    while start_iter < last_iter:
        remainder = start_iter % iters_per_tile
        end_iter = tl.minimum(start_iter + (iters_per_tile - remainder), last_iter)
        tile_id = start_iter // iters_per_tile
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rk = tl.arange(0, BLOCK_SIZE_K)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak + BLOCK_SIZE_K * stride_ak * remainder
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn + BLOCK_SIZE_K * stride_bk * remainder
        A_BASE = tl.multiple_of(A_BASE, (1, 16))
        B_BASE = tl.multiple_of(B_BASE, (16, 1))

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for current_iter in range(start_iter, end_iter):
            if EVEN_K:
                a = tl.load(A_BASE)
                b = tl.load(B_BASE)
            else:
                global_k_offset = (current_iter % iters_per_tile) * BLOCK_SIZE_K
                k_mask = global_k_offset + rk < K
                a = tl.load(A_BASE, mask=k_mask[None, :], other=0.0)
                b = tl.load(B_BASE, mask=k_mask[:, None], other=0.0)
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        tile_iter = tile_id * iters_per_tile

        if start_iter != tile_iter:
            rm1 = tl.arange(0, BLOCK_SIZE_M)
            rn1 = tl.arange(0, BLOCK_SIZE_N)
            rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
            P_ = P + pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
            tl.store(P_, acc, cache_modifier=".wt")
            tl.store(locks + pid, 1, cache_modifier=".wt")
        else:
            next_pid = pid + 1
            tile_iter_end = tile_iter + iters_per_tile
            end = end_iter
            while (end < tile_iter_end and next_pid < NUM_SMS):
                while tl.load(locks + next_pid, cache_modifier=".cv", volatile=True) != 1:
                    pass
                rm1 = tl.arange(0, BLOCK_SIZE_M)
                rn1 = tl.arange(0, BLOCK_SIZE_N)
                rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
                P_ = P + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
                acc += tl.load(tl.multiple_of(P_, (1, 16)), cache_modifier=".cv")
                end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)
                next_pid += 1

            c = acc.to(C.type.element_ty)
            if BIAS:
                c += bias[:, None]

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            tl.store(C_, c, mask=mask)

        start_iter = end_iter
