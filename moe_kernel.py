import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    sorted_token_ids_ptr,
    num_valid_tokens,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    pid_z = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    if SPLIT_K == 1:
        offs_k = tl.arange(0, BLOCK_SIZE_K)
    else:
        offs_k = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))

    a_ptrs = a_ptr + offs_token[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    acc_dtype = tl.float32 if a_ptr.type.element_ty != tl.int8 else tl.int32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk

    c = accumulator.to(c_ptr.type.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)


def invoke_moe(a, b, c, bias, block_m, block_n, block_k, group_m, split_k,
           num_warps, num_stages, waves_per_eu, mfmaInstrSize, kpack,
           use_bias, sorted_token_ids_ptr, num_valid_tokens):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    #assert a.is_contiguous(), "Matrix A must be contiguous"
    #assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # 1D launch kernel where each block gets its own program.

    grid = triton.cdiv(M, block_m) * triton.cdiv(N, block_n), split_k
    stride_bias = bias.stride(0) if use_bias else 0
    EVEN_K = K % block_k == 0
    matmul_kernel[grid](a,
                        b,
                        c,
                        M,
                        N,
                        K,
                        a.stride(0),
                        a.stride(1),
                        b.stride(0),
                        b.stride(1),
                        c.stride(0),
                        c.stride(1),
                        sorted_token_ids_ptr,
                        num_valid_tokens,
                        BLOCK_SIZE_M=block_m,
                        BLOCK_SIZE_N=block_n,
                        BLOCK_SIZE_K=block_k,
                        GROUP_SIZE_M=group_m,
                        SPLIT_K=split_k,
                        num_warps=num_warps,
                        num_stages=num_stages,
                        waves_per_eu=waves_per_eu,
                        matrix_instr_nonkdim=mfmaInstrSize,
                        kpack=kpack,
                        EVEN_K=EVEN_K)
    return c

def read_config(config):
    block_m = config["BLOCK_SIZE_M"]
    block_n = config["BLOCK_SIZE_N"]
    block_k = config["BLOCK_SIZE_K"]
    group_m = config["GROUP_SIZE_M"]
    split_k = config["SPLIT_K"]
    num_warps = config["num_warps"]
    num_stages = config["num_stages"]
    waves_per_eu = config["waves_per_eu"]
    kpack = config["kpack"]
    mfmaInstrSize = config["matrix_instr_nonkdim"]
    return block_m, block_n, block_k, group_m, split_k, num_warps, num_stages, waves_per_eu, mfma_instr_size, kpack



# test_correctness(M, N, K, col_a, col_b, dtype_a, dtype_b, dtype_c,
#                  init_type, config, bias_vector, verbose):
if __name__ == "__main__":
    config = {'M': 16, 'N': 4096, 'K': 1024, 'rowMajorA': 'T', 'rowMajorB': 'N', 'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1, 'num_warps': 8, 'num_stages': 0, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16, 'kpack': 2}
    # Processing Items
    M = config["M"]
    N = config["N"]
    K = config["K"]
    col_a = config["rowMajorA"] is "N"
    col_b = config["rowMajorB"] is "N"
    dtype_a = "fp16"
    dtype_b = "fp16"
    dtype_c = "fp16"
    init_type = "randn"
    bias_vector = False

    # Load Configs
    block_m, block_n, block_k, group_m, split_k, num_warps, num_stages, waves_per_eu, mfmaInstrSize, kpack = read_config(
    config)


    use_bias = bias_vector
    torch.manual_seed(0)
    #a = torch.randn((M, K), device='cuda', dtype=datatype)
    #b = torch.randn((K, N), device='cuda', dtype=datatype)
    a, a_fp16 = gen_input(M, K, dtype_a, col_a, 1, init_type, device='cuda')
    b, b_fp16 = gen_input(K, N, dtype_b, col_b, 2, init_type, device='cuda')
    bias = None
    if use_bias:
        bias, bias_fp16 = gen_input(M,
                                    1,
                                    dtype_b,
                                    col_b,
                                    2,
                                    init_type,
                                    device='cuda')
        bias = bias.squeeze()
        bias_fp16 = bias.squeeze()
    # Allocates output.
    c = torch.zeros((M, N),
                    device=a.device,
                    dtype=tl_to_torch_types[name_to_tl_types[dtype_c]])

    ## Allocate a fake expert_ids_ptr as 0,1,...,M-1
    sorted_token_ids_ptr = torch.arange(0, M, dtype=torch.int32, device=a.device)
    num_valid_tokens = 8
    triton_output = matmul(a, b, c, bias, block_m, block_n, block_k, group_m,
                           split_k, num_warps, num_stages, waves_per_eu,
                           mfmaInstrSize, kpack, use_bias, sorted_token_ids_ptr, num_valid_tokens)
    a[8:16, :] = 0
    #torch_output = torch.matmul(a, b)
    #torch.save(torch_output, 'tensor_cache_16x40961024_num-valid-tokens=8.pt')
    torch_output = torch.load('tensor_cache_16x40961024_num-valid-tokens=8.pt', weights_only=True)
    if use_bias:
        torch_output += bias_fp16[:, None]
    rtol = 0 if torch.version.hip is None else 1e-2
    atol = 1e-3 if split_k == 1 else 4e-2
    row_a_str = 'N' if col_a else 'T'
    row_b_str = 'N' if col_b else 'T'
    size_str = ''
    torch.set_printoptions(precision=2)
    torch.set_printoptions(linewidth=400)
    torch.set_printoptions(threshold=2048)
    torch.set_printoptions(sci_mode=False)
    if verbose:
        size_str = f'SIZE M: {M}, N: {N}, K: {K}, trans: {row_a_str}{row_b_str}'
    if torch.allclose(triton_output.to(torch.float16),
                      torch_output,
                      atol=atol,
                      rtol=rtol):
        print(f'{size_str} Correct✅')
    else:
        print(f"triton_output={triton_output}")
        print(f"torch_output={torch_output}")
        print(f"diff={torch_output - triton_output}")
        print(f'{size_str} Incorrect❌')
