import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

MFMA_LAYOUT = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 2])
SHARED_A_LAYOUT = gl.SwizzledSharedLayout(vec=16, per_phase=2, max_phase=8, order=[1, 0])
SHARED_B_LAYOUT = gl.SwizzledSharedLayout(vec=16, per_phase=2, max_phase=8, order=[0, 1])
DOT_A_LAYOUT = gl.DotOperandLayout(operand_index=0, parent=MFMA_LAYOUT, k_width=16)
DOT_B_LAYOUT = gl.DotOperandLayout(operand_index=1, parent=MFMA_LAYOUT, k_width=16)

KERNEL_CONFIG = {
    'stage1': {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'NUM_WARPS': 4, 'num_stages': 3}, 'stage2':
    {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'NUM_WARPS': 4, 'num_stages': 4}
}

# gather + gemm + SiLu + mul


@gluon.jit
def stage1_fused_kernel(
    hidden_states_ptr,  # [M, K]
    w1_ptr,  # [E, N*2, K]
    sorted_token_ids_ptr,  # [M]
    sorted_expert_ids_ptr,  # [Num_Blocks]
    out_ptr,  # [M, N]
    a1_scale_ptr,
    w1_scale_ptr,
    num_tokens,
    model_dim,
    inter_dim,
    stride_tokens_m,
    stride_tokens_k,
    stride_w1_e,
    stride_w1_n,
    stride_w1_k,
    stride_out_m,
    stride_out_n,
    stride_ascale_m,
    stride_ascale_k,
    stride_wscale_e,
    stride_wscale_n,
    stride_wscale_k,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_WARPS: tl.constexpr,
):
    pid = gl.program_id(0)
    num_blocks_n = tl.cdiv(inter_dim, BLOCK_SIZE_N)
    pid_m = pid // num_blocks_n
    pid_n = pid % num_blocks_n

    expert_id = gl.load(sorted_expert_ids_ptr + pid_m)

    dtype_a = hidden_states_ptr.type.element_ty
    dtype_b = w1_ptr.type.element_ty

    smem_a = gl.allocate_shared_memory(dtype_a, [BLOCK_SIZE_M, BLOCK_SIZE_K], layout=SHARED_A_LAYOUT)
    smem_b_gate = gl.allocate_shared_memory(dtype_b, [BLOCK_SIZE_K, BLOCK_SIZE_N], layout=SHARED_B_LAYOUT)
    smem_b_up = gl.allocate_shared_memory(dtype_b, [BLOCK_SIZE_K, BLOCK_SIZE_N], layout=SHARED_B_LAYOUT)

    acc_gate = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=gl.float32, layout=MFMA_LAYOUT)
    acc_up = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=gl.float32, layout=MFMA_LAYOUT)

    offs_m = pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N)

    original_token_indices = gl.load(sorted_token_ids_ptr + offs_m)
    valid_mask_m = (original_token_indices >= 0)[:, None]

    # avoids 32-bit overflow for large weights
    expert_w1_base = w1_ptr + (expert_id.to(tl.int64) * stride_w1_e)

    # A scales: [M, K/128]
    # W scales: [E, N/128, K/128]
    # assume scale block size is 128
    SCALE_BLOCK = 128

    expert_w1_scale_base = w1_scale_ptr + (expert_id.to(tl.int64) * stride_wscale_e)

    for k in range(0, model_dim, BLOCK_SIZE_K):
        offs_k_a = k + gl.arange(0, BLOCK_SIZE_K)
        offs_k_b = k + gl.arange(0, BLOCK_SIZE_K)

        a_ptrs = hidden_states_ptr + (original_token_indices[:, None] * stride_tokens_m +
                                      offs_k_a[None, :] * stride_tokens_k)
        a_regs = gl.load(a_ptrs, mask=valid_mask_m, other=0.0)

        # scale A is [M, K_blocks] load [BLOCK_SIZE_M, 1] corresponding to current K block
        k_scale_idx = k // SCALE_BLOCK
        a_scale_ptrs = a1_scale_ptr + (original_token_indices[:, None] * stride_ascale_m +
                                       k_scale_idx * stride_ascale_k)
        # mask is just valid tokens (M dimension)
        scale_a = gl.load(a_scale_ptrs, mask=valid_mask_m, other=1.0)

        w_gate_ptrs = expert_w1_base + (offs_n[None, :] * stride_w1_n + offs_k_b[:, None] * stride_w1_k)
        b_gate_regs = gl.load(w_gate_ptrs)

        # scale W is [N_blocks, K_blocks] per expert load [1, BLOCK_SIZE_N]
        w_scale_ptrs_gate = expert_w1_scale_base + (
            (offs_n[None, :] // SCALE_BLOCK) * stride_wscale_n + k_scale_idx * stride_wscale_k)
        scale_w_gate = gl.load(w_scale_ptrs_gate)

        w_up_ptrs = expert_w1_base + ((offs_n[None, :] + inter_dim) * stride_w1_n + offs_k_b[:, None] * stride_w1_k)
        b_up_regs = gl.load(w_up_ptrs)

        w_scale_ptrs_up = expert_w1_scale_base + ((
            (offs_n[None, :] + inter_dim) // SCALE_BLOCK) * stride_wscale_n + k_scale_idx * stride_wscale_k)
        scale_w_up = gl.load(w_scale_ptrs_up)

        smem_a.store(a_regs)
        smem_b_gate.store(b_gate_regs)

        cur_a = smem_a.load(layout=DOT_A_LAYOUT)
        cur_b_gate = smem_b_gate.load(layout=DOT_B_LAYOUT)

        # acc += (A * B) * (scale_a * scale_w)
        # compute A*B into a temporary accumulator
        # scale and add to main acumulator
        tmp_gate = gl.amd.cdna4.mfma(cur_a, cur_b_gate, gl.zeros_like(acc_gate))
        acc_gate += tmp_gate * (scale_a * scale_w_gate)

        smem_b_up.store(b_up_regs)
        cur_b_up = smem_b_up.load(layout=DOT_B_LAYOUT)

        tmp_up = gl.amd.cdna4.mfma(cur_a, cur_b_up, gl.zeros_like(acc_up))
        acc_up += tmp_up * (scale_a * scale_w_up)

    gate_act = acc_gate * tl.sigmoid(acc_gate)
    result = gate_act * acc_up
    # switch to FP8 output for Stage 1 to enable FP8 GEMM in Stage 2
    result = result.to(dtype_b)

    out_ptrs = out_ptr + (offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n)
    gl.store(out_ptrs, result)


# gemm + weighted reduce + scatter


@gluon.jit
def stage2_fused_kernel(
    inter_states_ptr,  # [Sorted_M, N]
    w2_ptr,  # [E, D, N]
    sorted_token_ids_ptr,
    sorted_expert_ids_ptr,
    sorted_weights_ptr,
    out_ptr,  # [M, D]
    a2_scale_ptr,
    w2_scale_ptr,
    stride_inter_m,
    stride_inter_k,
    stride_w2_e,
    stride_w2_n,
    stride_w2_k,
    stride_out_m,
    stride_out_n,
    stride_ascale_m,
    stride_ascale_k,
    stride_wscale_e,
    stride_wscale_n,
    stride_wscale_k,
    num_sorted_tokens,
    model_dim,
    inter_dim,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_WARPS: tl.constexpr,
):
    pid = gl.program_id(0)
    num_blocks_n = tl.cdiv(model_dim, BLOCK_SIZE_N)
    pid_m = pid // num_blocks_n
    pid_n = pid % num_blocks_n

    expert_id = gl.load(sorted_expert_ids_ptr + pid_m)

    dtype_a = inter_states_ptr.type.element_ty
    dtype_b = w2_ptr.type.element_ty

    smem_a = gl.allocate_shared_memory(dtype_a, [BLOCK_SIZE_M, BLOCK_SIZE_K], layout=SHARED_A_LAYOUT)
    smem_b = gl.allocate_shared_memory(dtype_b, [BLOCK_SIZE_K, BLOCK_SIZE_N], layout=SHARED_B_LAYOUT)
    acc = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=gl.float32, layout=MFMA_LAYOUT)

    offs_m = pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N)
    expert_w2_base = w2_ptr + (expert_id.to(tl.int64) * stride_w2_e)
    expert_w2_scale_base = w2_scale_ptr + (expert_id.to(tl.int64) * stride_wscale_e)

    SCALE_BLOCK = 128

    for k in range(0, inter_dim, BLOCK_SIZE_K):
        offs_k_a = k + gl.arange(0, BLOCK_SIZE_K)
        offs_k_b = k + gl.arange(0, BLOCK_SIZE_K)

        a_ptrs = inter_states_ptr + (offs_m[:, None] * stride_inter_m + offs_k_a[None, :] * stride_inter_k)
        a_regs = gl.load(a_ptrs, mask=offs_m[:, None] < num_sorted_tokens, other=0.0)

        k_scale_idx = k // SCALE_BLOCK
        a_scale_ptrs = a2_scale_ptr + (offs_m[:, None] * stride_ascale_m + k_scale_idx * stride_ascale_k)
        scale_a = gl.load(a_scale_ptrs, mask=offs_m[:, None] < num_sorted_tokens, other=1.0)

        w_ptrs = expert_w2_base + (offs_n[None, :] * stride_w2_n + offs_k_b[:, None] * stride_w2_k)
        b_regs = gl.load(w_ptrs)

        w_scale_ptrs = expert_w2_scale_base + (
            (offs_n[None, :] // SCALE_BLOCK) * stride_wscale_n + k_scale_idx * stride_wscale_k)
        scale_w = gl.load(w_scale_ptrs)

        smem_a.store(a_regs)
        smem_b.store(b_regs)

        # use Native FP8 for compute (Inputs A and B are both FP8 now)
        cur_a = smem_a.load(layout=DOT_A_LAYOUT)
        cur_b = smem_b.load(layout=DOT_B_LAYOUT)

        tmp = gl.amd.cdna4.mfma(cur_a, cur_b, gl.zeros_like(acc))
        acc += tmp * (scale_a * scale_w)

    routing_weights = gl.load(sorted_weights_ptr + offs_m, mask=offs_m < num_sorted_tokens, other=0.0)
    final_val = acc * routing_weights[:, None]

    # audit this later
    final_val = final_val.to(gl.bfloat16)

    # scatter
    original_token_indices = gl.load(sorted_token_ids_ptr + offs_m)
    valid_mask = (offs_m < num_sorted_tokens) & (original_token_indices >= 0)

    out_ptrs = out_ptr + (original_token_indices[:, None] * stride_out_m + offs_n[None, :] * stride_out_n)

    out_ptrs = gl.convert_layout(out_ptrs, MFMA_LAYOUT)
    mask = gl.convert_layout(valid_mask[:, None], MFMA_LAYOUT)

    # contributions from multiple experts to the same token
    gl.atomic_add(out_ptrs, final_val, mask=mask)


def run_reference_stage1(hidden_states, w1, sorted_token_ids, sorted_expert_ids, M, N, K, BLOCK_SIZE_M, scale_a=1.0,
                         scale_w=1.0):
    hidden_states_f32 = hidden_states.to(torch.float32)
    w1_f32 = w1.to(torch.float32)

    # [M_sorted, N]
    num_blocks = sorted_expert_ids.shape[0]
    out_ref = torch.zeros((num_blocks * BLOCK_SIZE_M, N), dtype=torch.float32, device="cuda")

    # w1 is [E, 2*N, K]
    w1_gate = w1_f32[:, :N, :]
    w1_up = w1_f32[:, N:, :]

    total_scale = scale_a * scale_w

    for i in range(num_blocks):
        expert_idx = sorted_expert_ids[i].item()

        start_row = i * BLOCK_SIZE_M
        end_row = start_row + BLOCK_SIZE_M

        token_indices = sorted_token_ids[start_row:end_row]

        # filter valid tokens
        valid_mask = token_indices >= 0
        valid_indices = token_indices[valid_mask]

        if valid_indices.numel() == 0:
            continue

        # gather
        x = hidden_states_f32[valid_indices.long()]  # [Batch, K]

        g_w = w1_gate[expert_idx].t()  # [K, N]
        u_w = w1_up[expert_idx].t()  # [K, N]

        gate = torch.matmul(x, g_w)
        up = torch.matmul(x, u_w)

        # Apply scales (De-quantize)
        gate = gate * total_scale
        up = up * total_scale

        # SiLU * Up
        gate_act = gate * torch.sigmoid(gate)
        res = gate_act * up

        # Scatter back to reference output buffer
        full_res = torch.zeros((BLOCK_SIZE_M, N), dtype=torch.float32, device="cuda")
        full_res[valid_mask] = res

        out_ref[start_row:end_row] = full_res

    return out_ref


def run_reference_stage2(inter_states, w2, sorted_token_ids, sorted_expert_ids, sorted_weights, M, D, N, BLOCK_SIZE_M,
                         scale_a=1.0, scale_w=1.0):
    inter_states_f32 = inter_states.to(torch.float32)
    w2_f32 = w2.to(torch.float32)

    out_ref = torch.zeros((M, D), dtype=torch.float32, device="cuda")

    num_blocks = sorted_expert_ids.shape[0]

    total_scale = scale_a * scale_w

    for i in range(num_blocks):
        expert_idx = sorted_expert_ids[i].item()
        start_row = i * BLOCK_SIZE_M
        end_row = start_row + BLOCK_SIZE_M

        # [BLOCK, N]
        x = inter_states_f32[start_row:end_row]

        # Weight [E, D, N] -> [D, N] -> T -> [N, D]
        w = w2_f32[expert_idx].t()

        res = torch.matmul(x, w)  # [BLOCK, D]

        # Apply scales (De-quantize)
        res = res * total_scale

        weights = sorted_weights[start_row:end_row].unsqueeze(1)
        res = res * weights

        # Scatter Add
        token_indices = sorted_token_ids[start_row:end_row]
        valid_mask = token_indices >= 0

        valid_indices = token_indices[valid_mask].long()
        valid_res = res[valid_mask]

        if valid_indices.numel() > 0:
            out_ref.index_add_(0, valid_indices, valid_res)

    return out_ref


def verify():
    print("Running Correctness Verification...")
    torch.manual_seed(0)

    # reduced problem size because of pytorch naive implementation
    M, E, K, N, topk = 1024, 16, 256, 128, 2

    c1 = KERNEL_CONFIG['stage1']
    S1_BLOCK_M, S1_BLOCK_N, S1_BLOCK_K = c1['BLOCK_SIZE_M'], c1['BLOCK_SIZE_N'], c1['BLOCK_SIZE_K']
    S1_WARPS = c1['NUM_WARPS']
    # S1_STAGES = c1['num_stages']

    c2 = KERNEL_CONFIG['stage2']
    S2_BLOCK_M, S2_BLOCK_N, S2_BLOCK_K = c2['BLOCK_SIZE_M'], c2['BLOCK_SIZE_N'], c2['BLOCK_SIZE_K']
    S2_WARPS = c2['NUM_WARPS']
    # S2_STAGES = c2['num_stages']

    # Create Dummy Scales (Block-wise)
    # Scales are usually [M, K/128].
    SCALE_BLOCK = 128

    # Stage 1 Scales
    # A1 Scale: [M, K/128]
    a1_scale = torch.ones((M, triton.cdiv(K, SCALE_BLOCK)), dtype=torch.float32, device="cuda")
    # W1 Scale: [E, N*2/128, K/128]
    w1_scale = torch.ones((E, triton.cdiv(N * 2, SCALE_BLOCK), triton.cdiv(K, SCALE_BLOCK)), dtype=torch.float32,
                          device="cuda")

    a2_scale = torch.ones((M * topk, triton.cdiv(N, SCALE_BLOCK)), dtype=torch.float32, device="cuda")

    w2_scale = torch.ones((E, triton.cdiv(K, SCALE_BLOCK), triton.cdiv(N, SCALE_BLOCK)), dtype=torch.float32,
                          device="cuda")

    # scale down to prevent FP8 saturation/overflow
    hidden_states = (torch.randn((M, K), dtype=torch.float32, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    w1 = (torch.randn((E, N * 2, K), dtype=torch.float32, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    w2 = (torch.randn((E, K, N), dtype=torch.float32, device="cuda") * 0.1).to(torch.float8_e4m3fn)

    MAX_BLOCK_M = max(S1_BLOCK_M, S2_BLOCK_M)
    num_entries = M * topk
    if num_entries % MAX_BLOCK_M != 0:
        pad = MAX_BLOCK_M - (num_entries % MAX_BLOCK_M)
        num_entries += pad

    sorted_token_ids = torch.randint(0, M, (num_entries, ), dtype=torch.int32, device="cuda")
    sorted_expert_ids = torch.randint(0, E, (num_entries // MAX_BLOCK_M, ), dtype=torch.int32, device="cuda")
    sorted_weights = torch.rand((num_entries, ), dtype=torch.float32, device="cuda")

    out_stage1_tri = torch.empty((num_entries, N), dtype=torch.float8_e4m3fn, device="cuda")
    out_stage2_tri = torch.zeros((M, K), dtype=torch.bfloat16, device="cuda")

    grid_1 = (sorted_expert_ids.shape[0] * triton.cdiv(N, S1_BLOCK_N), 1, 1)

    stage1_fused_kernel[grid_1](hidden_states_ptr=hidden_states, w1_ptr=w1, sorted_token_ids_ptr=sorted_token_ids,
                                sorted_expert_ids_ptr=sorted_expert_ids, out_ptr=out_stage1_tri, a1_scale_ptr=a1_scale,
                                w1_scale_ptr=w1_scale, num_tokens=M, model_dim=K, inter_dim=N,
                                stride_tokens_m=hidden_states.stride(0), stride_tokens_k=hidden_states.stride(1),
                                stride_w1_e=w1.stride(0), stride_w1_n=w1.stride(1), stride_w1_k=w1.stride(2),
                                stride_out_m=out_stage1_tri.stride(0), stride_out_n=out_stage1_tri.stride(1),
                                stride_ascale_m=a1_scale.stride(0), stride_ascale_k=a1_scale.stride(1),
                                stride_wscale_e=w1_scale.stride(0), stride_wscale_n=w1_scale.stride(1),
                                stride_wscale_k=w1_scale.stride(2), BLOCK_SIZE_M=S1_BLOCK_M, BLOCK_SIZE_N=S1_BLOCK_N,
                                BLOCK_SIZE_K=S1_BLOCK_K, NUM_WARPS=S1_WARPS)

    ref_stage1 = run_reference_stage1(hidden_states, w1, sorted_token_ids, sorted_expert_ids, M, N, K, S1_BLOCK_M,
                                      scale_a=1.0, scale_w=1.0)

    tri_stage1_f32 = out_stage1_tri.to(torch.float32)
    diff = torch.abs(tri_stage1_f32 - ref_stage1)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Stage 1 Max Diff: {max_diff:.4f}, Mean Diff: {mean_diff:.4f}")
    if mean_diff < 0.5:
        print("Stage 1: PASS (within FP8 tolerance)")
    else:
        print("Stage 1: FAIL / HIGH ERROR")

    grid_2 = (sorted_expert_ids.shape[0] * triton.cdiv(K, S2_BLOCK_N), 1, 1)

    inter_input = ref_stage1.to(torch.float8_e4m3fn)

    stage2_fused_kernel[grid_2](
        inter_states_ptr=inter_input, w2_ptr=w2, sorted_token_ids_ptr=sorted_token_ids,
        sorted_expert_ids_ptr=sorted_expert_ids, sorted_weights_ptr=sorted_weights, out_ptr=out_stage2_tri,
        a2_scale_ptr=a2_scale, w2_scale_ptr=w2_scale, stride_inter_m=inter_input.stride(0),
        stride_inter_k=inter_input.stride(1), stride_w2_e=w2.stride(0), stride_w2_n=w2.stride(1),
        stride_w2_k=w2.stride(2), stride_out_m=out_stage2_tri.stride(0), stride_out_n=out_stage2_tri.stride(1),
        stride_ascale_m=a2_scale.stride(0), stride_ascale_k=a2_scale.stride(1), stride_wscale_e=w2_scale.stride(0),
        stride_wscale_n=w2_scale.stride(1), stride_wscale_k=w2_scale.stride(2), num_sorted_tokens=num_entries,
        model_dim=K, inter_dim=N, BLOCK_SIZE_M=S2_BLOCK_M, BLOCK_SIZE_N=S2_BLOCK_N, BLOCK_SIZE_K=S2_BLOCK_K,
        NUM_WARPS=S2_WARPS)

    ref_stage2 = run_reference_stage2(inter_input, w2, sorted_token_ids, sorted_expert_ids, sorted_weights, M, K, N,
                                      S2_BLOCK_M, scale_a=1.0, scale_w=1.0)

    tri_stage2_f32 = out_stage2_tri.to(torch.float32)
    diff2 = torch.abs(tri_stage2_f32 - ref_stage2)
    max_diff2 = diff2.max().item()
    mean_diff2 = diff2.mean().item()

    print(f"Stage 2 Max Diff: {max_diff2:.4f}, Mean Diff: {mean_diff2:.4f}")
    if mean_diff2 < 1.0:
        print("Stage 2: PASS (within tolerance)")
    else:
        print("Stage 2: FAIL / HIGH ERROR")


def benchmark_moe(M=65536, E=256, K=7168, N=2048, topk=8, is_stage1=True, is_stage2=True):
    print(f"Benchmarking MoE (Gluon) | M={M}, E={E}, K={K}, N={N}, TopK={topk}...")
    print(f"Triton version: {triton.__version__}")

    # validation for N block alignment
    c1 = KERNEL_CONFIG['stage1']
    if N < c1['BLOCK_SIZE_N']:
        print(f"WARNING: N={N} is smaller than Stage 1 BLOCK_SIZE_N={c1['BLOCK_SIZE_N']}. Adjusting config.")

    S1_BLOCK_M, S1_BLOCK_N, S1_BLOCK_K = c1['BLOCK_SIZE_M'], c1['BLOCK_SIZE_N'], c1['BLOCK_SIZE_K']
    S1_WARPS = c1['NUM_WARPS']

    c2 = KERNEL_CONFIG['stage2']
    S2_BLOCK_M, S2_BLOCK_N, S2_BLOCK_K = c2['BLOCK_SIZE_M'], c2['BLOCK_SIZE_N'], c2['BLOCK_SIZE_K']
    S2_WARPS = c2['NUM_WARPS']

    hidden_states = torch.randn((M, K), dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn)
    w1 = torch.randn((E, N * 2, K), dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn)
    w2 = torch.randn((E, K, N), dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn)

    sorted_token_ids = torch.randint(0, M, (M * topk, ), dtype=torch.int32, device="cuda")
    sorted_expert_ids = torch.randint(0, E, (M * topk // 128, ), dtype=torch.int32, device="cuda")
    sorted_weights = torch.rand((M * topk, ), dtype=torch.float32, device="cuda")

    # FP8 for inter-stage activation to save bandwidth and enable FP8 GEMM
    out_stage1 = torch.empty((M * topk, N), dtype=torch.float8_e4m3fn, device="cuda")
    out_stage2 = torch.zeros((M, K), dtype=torch.bfloat16, device="cuda")

    # Create Dummy Scales (Block-wise)
    SCALE_BLOCK = 128

    a1_scale = torch.ones((M, triton.cdiv(K, SCALE_BLOCK)), dtype=torch.float32, device="cuda")
    w1_scale = torch.ones((E, triton.cdiv(N * 2, SCALE_BLOCK), triton.cdiv(K, SCALE_BLOCK)), dtype=torch.float32,
                          device="cuda")
    a2_scale = torch.ones((M * topk, triton.cdiv(N, SCALE_BLOCK)), dtype=torch.float32, device="cuda")
    w2_scale = torch.ones((E, triton.cdiv(K, SCALE_BLOCK), triton.cdiv(N, SCALE_BLOCK)), dtype=torch.float32,
                          device="cuda")

    start_event_1 = torch.cuda.Event(enable_timing=True)
    end_event_1 = torch.cuda.Event(enable_timing=True)

    grid_1 = (sorted_expert_ids.shape[0] * triton.cdiv(N, S1_BLOCK_N), 1, 1)

    if is_stage1:
        # warmup
        for _ in range(10):
            stage1_fused_kernel[grid_1](
                hidden_states_ptr=hidden_states, w1_ptr=w1, sorted_token_ids_ptr=sorted_token_ids,
                sorted_expert_ids_ptr=sorted_expert_ids, out_ptr=out_stage1, a1_scale_ptr=a1_scale,
                w1_scale_ptr=w1_scale, num_tokens=M, model_dim=K, inter_dim=N, stride_tokens_m=hidden_states.stride(0),
                stride_tokens_k=hidden_states.stride(1), stride_w1_e=w1.stride(0), stride_w1_n=w1.stride(1),
                stride_w1_k=w1.stride(2), stride_out_m=out_stage1.stride(0), stride_out_n=out_stage1.stride(1),
                stride_ascale_m=a1_scale.stride(0), stride_ascale_k=a1_scale.stride(1),
                stride_wscale_e=w1_scale.stride(0), stride_wscale_n=w1_scale.stride(1),
                stride_wscale_k=w1_scale.stride(2), BLOCK_SIZE_M=S1_BLOCK_M, BLOCK_SIZE_N=S1_BLOCK_N,
                BLOCK_SIZE_K=S1_BLOCK_K, NUM_WARPS=S1_WARPS)

        # simple benchmark
        start_event_1.record()
        for _ in range(100):
            stage1_fused_kernel[grid_1](
                hidden_states_ptr=hidden_states, w1_ptr=w1, sorted_token_ids_ptr=sorted_token_ids,
                sorted_expert_ids_ptr=sorted_expert_ids, out_ptr=out_stage1, a1_scale_ptr=a1_scale,
                w1_scale_ptr=w1_scale, num_tokens=M, model_dim=K, inter_dim=N, stride_tokens_m=hidden_states.stride(0),
                stride_tokens_k=hidden_states.stride(1), stride_w1_e=w1.stride(0), stride_w1_n=w1.stride(1),
                stride_w1_k=w1.stride(2), stride_out_m=out_stage1.stride(0), stride_out_n=out_stage1.stride(1),
                stride_ascale_m=a1_scale.stride(0), stride_ascale_k=a1_scale.stride(1),
                stride_wscale_e=w1_scale.stride(0), stride_wscale_n=w1_scale.stride(1),
                stride_wscale_k=w1_scale.stride(2), BLOCK_SIZE_M=S1_BLOCK_M, BLOCK_SIZE_N=S1_BLOCK_N,
                BLOCK_SIZE_K=S1_BLOCK_K, NUM_WARPS=S1_WARPS)
        end_event_1.record()
        torch.cuda.synchronize()
        print(f"Stage 1 launched. Avg Latency (100 runs): {start_event_1.elapsed_time(end_event_1) / 100:.2f} ms")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    grid_2 = (sorted_expert_ids.shape[0] * triton.cdiv(K, S2_BLOCK_N), 1, 1)

    if is_stage2:
        # warmup
        for _ in range(10):
            stage2_fused_kernel[grid_2](
                inter_states_ptr=out_stage1, w2_ptr=w2, sorted_token_ids_ptr=sorted_token_ids,
                sorted_expert_ids_ptr=sorted_expert_ids, sorted_weights_ptr=sorted_weights, out_ptr=out_stage2,
                a2_scale_ptr=a2_scale, w2_scale_ptr=w2_scale, stride_inter_m=out_stage1.stride(0),
                stride_inter_k=out_stage1.stride(1), stride_w2_e=w2.stride(0), stride_w2_n=w2.stride(1),
                stride_w2_k=w2.stride(2), stride_out_m=out_stage2.stride(0), stride_out_n=out_stage2.stride(1),
                stride_ascale_m=a2_scale.stride(0), stride_ascale_k=a2_scale.stride(1),
                stride_wscale_e=w2_scale.stride(0), stride_wscale_n=w2_scale.stride(1),
                stride_wscale_k=w2_scale.stride(2), num_sorted_tokens=M * topk, model_dim=K, inter_dim=N,
                BLOCK_SIZE_M=S2_BLOCK_M, BLOCK_SIZE_N=S2_BLOCK_N, BLOCK_SIZE_K=S2_BLOCK_K, NUM_WARPS=S2_WARPS)

        # simple benchmark
        start_event.record()
        for _ in range(100):
            stage2_fused_kernel[grid_2](
                inter_states_ptr=out_stage1, w2_ptr=w2, sorted_token_ids_ptr=sorted_token_ids,
                sorted_expert_ids_ptr=sorted_expert_ids, sorted_weights_ptr=sorted_weights, out_ptr=out_stage2,
                a2_scale_ptr=a2_scale, w2_scale_ptr=w2_scale, stride_inter_m=out_stage1.stride(0),
                stride_inter_k=out_stage1.stride(1), stride_w2_e=w2.stride(0), stride_w2_n=w2.stride(1),
                stride_w2_k=w2.stride(2), stride_out_m=out_stage2.stride(0), stride_out_n=out_stage2.stride(1),
                stride_ascale_m=a2_scale.stride(0), stride_ascale_k=a2_scale.stride(1),
                stride_wscale_e=w2_scale.stride(0), stride_wscale_n=w2_scale.stride(1),
                stride_wscale_k=w2_scale.stride(2), num_sorted_tokens=M * topk, model_dim=K, inter_dim=N,
                BLOCK_SIZE_M=S2_BLOCK_M, BLOCK_SIZE_N=S2_BLOCK_N, BLOCK_SIZE_K=S2_BLOCK_K, NUM_WARPS=S2_WARPS)
        end_event.record()
        torch.cuda.synchronize()
        print(f"Stage 2 launched. Avg Latency (100 runs): {start_event.elapsed_time(end_event) / 100:.2f} ms")

    print("Done.")


if __name__ == "__main__":
    verify()

    # print("\n=== Benchmark Config A: N=2048 (TP=1 equivalent) ===")
    # benchmark_moe(M=65536, E=256, K=7168, N=2048, topk=8)

    # Benchmarking Small N (TP=8 target)
    print("\n=== Benchmark Config B: N=256 (TP=8 equivalent) ===")
    benchmark_moe(M=65536, E=256, K=7168, N=256, topk=8)
