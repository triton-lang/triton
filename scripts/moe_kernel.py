import re
import torch
import triton
import triton.language as tl
import argparse

from triton.tools.mxfp import MXScaleTensor, MXFP4Tensor


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K: tl.constexpr, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
                  stride_cn, sorted_token_ids_ptr, num_valid_tokens, BLOCK_SIZE_M: tl.constexpr,
                  BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, SPLIT_K: tl.constexpr,
                  GROUP_SIZE_M: tl.constexpr, EVEN_K: tl.constexpr):
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

    a_ptrs = a_ptr + offs_token[:, None] * \
        stride_am + offs_k[None, :] * stride_ak
    # a_ptrs = a_ptr + offs_token_id[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    acc_dtype = tl.float32 if a_ptr.type.element_ty != tl.int8 else tl.int32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        # a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk

    c = accumulator.to(c_ptr.type.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * \
        offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)


# @triton.autotune(
#     configs=[
#         triton.Config(
#             {
#                 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2,
#                 'kpack': 2, 'matrix_instr_nonkdim': 16
#             }, num_warps=4, num_stages=2),
#         triton.Config(
#             {
#                 'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2,
#                 'kpack': 2, 'matrix_instr_nonkdim': 0
#             }, num_warps=8, num_stages=2),
#         triton.Config(
#             {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4, 'waves_per_eu': 0},
#             num_warps=8, num_stages=2),
#         triton.Config(
#             {
#                 'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2,
#                 'kpack': 1, 'matrix_instr_nonkdim': 0
#             }, num_warps=8, num_stages=2),
#         triton.Config(
#             {
#                 'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'waves_per_eu': 0,
#                 'kpack': 1
#             }, num_warps=8, num_stages=2),
#         triton.Config(
#             {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'waves_per_eu': 0},
#             num_warps=8, num_stages=2),
#         triton.Config(
#             {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
#             num_warps=8, num_stages=2),
#     ],
#     key=['M', 'N', 'K'],
#     use_cuda_graph=True,
# )
@triton.jit
def scaled_dot_kernel(a_ptr, b_ptr, c_ptr, a_scale, b_scale, M, N, K: tl.constexpr, stride_scale,  #
                      stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, sorted_token_ids_ptr,
                      num_valid_tokens, DTYPE_A: tl.constexpr,  #
                      DTYPE_B: tl.constexpr,  #
                      BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                      SPLIT_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr, EVEN_K: tl.constexpr):
    DIV_FACTOR_A: tl.constexpr = 2 if DTYPE_A == "e2m1" else 1
    DIV_FACTOR_B: tl.constexpr = 2 if DTYPE_B == "e2m1" else 1
    pid = tl.program_id(axis=0)
    pid_z = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
        # pid_m = pid % num_pid_m
        # pid_n = pid // num_pid_m
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    if SPLIT_K == 1:
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_ak = tl.arange(0, BLOCK_SIZE_K // DIV_FACTOR_A)
        offs_bk = tl.arange(0, BLOCK_SIZE_K // DIV_FACTOR_B)
    else:
        offs_k = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_bn_scale = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    a_ptrs = a_ptr + offs_token[:, None] * \
        stride_am + offs_ak[None, :] * stride_ak
    b_ptrs = b_ptr + offs_bk[:, None] * \
        stride_bk + offs_bn[None, :] * stride_bn
    offs_scale_k = tl.arange(0, BLOCK_SIZE_K // 32)

    a_scale_ptr = a_scale + offs_token[:, None] * \
        stride_scale + offs_scale_k[None, :]
    b_scale_ptr = b_scale + \
        offs_bn_scale[:, None] * stride_scale + offs_scale_k[None, :]

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        # a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs)
        scale_a = tl.load(a_scale_ptr)
        scale_b = tl.load(b_scale_ptr)
        accumulator = tl.dot_scaled(a, scale_a, DTYPE_A, b, scale_b, DTYPE_B, accumulator)
        a_ptrs += (BLOCK_SIZE_K // DIV_FACTOR_A) * SPLIT_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // DIV_FACTOR_B) * SPLIT_K * stride_bk
        a_scale_ptr += (BLOCK_SIZE_K // 32) * SPLIT_K
        b_scale_ptr += (BLOCK_SIZE_K // 32) * SPLIT_K

    c = accumulator.to(c_ptr.type.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * \
        offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)


def gen_input(M, N, ty_name, needTrans, seed, init_type, device='cuda'):
    d_type = name_to_tl_types[ty_name]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    @triton.jit
    def copy_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        input = tl.load(input_ptr + offsets, mask=mask)
        output = input
        tl.store(output_ptr + offsets, output, mask=mask)

    def init_by_size_and_type(size, dtype, init_type):
        if init_type == 'hpl':
            return torch.empty(size, device='cuda', dtype=dtype).uniform_(-0.5, 0.5)
        # This init type has element[i] in row[j] equal to sin(i+j*N)
        elif init_type == 'trig_float':
            M, N = size
            return torch.reshape(torch.arange(0, M * N), (M, N)).sin().to(dtype=dtype, device='cuda')
        elif init_type == 'zeros':
            return torch.zeros(size, dtype=dtype, device='cuda')
        elif init_type == "randn":
            return torch.randn(size, dtype=dtype, device='cuda')
        elif init_type == 'ones':
            return torch.ones(size, dtype=dtype, device='cuda')
        elif init_type == 'const_layer':
            ret = []
            dim0, dim1 = size
            for i in range(dim0):
                ret.append(torch.full((dim1, ), float(i + 1) / dim0, dtype=dtype, device='cuda'))
            return torch.stack(ret).contiguous()
        else:
            raise ValueError("Bad matrix initialization type.")

    raw_data = init_by_size_and_type((N, M) if needTrans else (M, N), torch.float32, init_type)
    if needTrans:
        raw_data = raw_data.T
    if (d_type == tl.float8e4b8 and TORCH_HAS_FP8E4B8) or \
            (d_type == tl.float8e5b16 and TORCH_HAS_FP8E5B16) or not d_type.is_fp8():
        input = raw_data.to(tl_to_torch_types[d_type])
        input_f16 = input.to(torch.float16)
    else:
        f8_tensor = raw_data.to(torch.int8)
        # keep only two bits of exponent to avoid overflow
        f8_tensor = f8_tensor & 0b00111111
        input = triton.reinterpret(f8_tensor, d_type)
        input_f16 = torch.empty_like(f8_tensor, dtype=torch.float16)

        def grid(meta):
            return (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

        n_elements = raw_data.numel()
        copy_kernel[grid](input, input_f16, n_elements, BLOCK_SIZE=1024)

    return input, input_f16


def invoke_moe(a, b, c, bias, block_m, block_n, block_k, group_m, split_k, num_warps, num_stages, waves_per_eu,
               mfmaInstrSize, kpack, use_bias, sorted_token_ids_ptr, num_valid_tokens):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    # assert a.is_contiguous(), "Matrix A must be contiguous"
    # assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # 1D launch kernel where each block gets its own program.

    grid = triton.cdiv(M, block_m) * triton.cdiv(N, block_n), split_k
    stride_bias = bias.stride(0) if use_bias else 0
    EVEN_K = K % block_k == 0
    print(f'stride_am: {a.stride(0)}, stride_ak: {a.stride(1)}')
    matmul_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                        sorted_token_ids_ptr, num_valid_tokens, BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n,
                        BLOCK_SIZE_K=block_k, GROUP_SIZE_M=group_m, SPLIT_K=split_k, num_warps=num_warps,
                        num_stages=num_stages, waves_per_eu=waves_per_eu, matrix_instr_nonkdim=mfmaInstrSize,
                        kpack=kpack, EVEN_K=EVEN_K)
    return c


def invoke_scaled_moe(a, b, c, a_scale, b_scale, bias, block_m, block_n, block_k, dtype_a, dtype_b, group_m, split_k,
                      num_warps, num_stages, waves_per_eu, mfmaInstrSize, kpack, use_bias, sorted_token_ids_ptr,
                      num_valid_tokens):
    # Check constraints.
    div_factor_a = 2 if dtype_a == 'float4' else 1
    div_factor_b = 2 if dtype_b == 'float4' else 1

    # print(f'{a.shape=}, {b.shape=}, {div_factor_a=}, {div_factor_b=}')

    assert (a.shape[1] * div_factor_a) == (b.shape[0] * div_factor_b), "Incompatible dimensions"
    # assert a.is_contiguous(), "Matrix A must be contiguous"
    # assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    K *= div_factor_b
    assert a_scale.stride(0) == b_scale.stride(0)
    # print(f'stride_scale: {a_scale.stride(0)}')
    # 1D launch kernel where each block gets its own program.

    dtype_converter = {'float8e5': 'e5m2', 'float8e4nv': 'e4m3', 'float4': 'e2m1'}

    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n), split_k)
    # stride_bias = bias.stride(0) if use_bias else 0
    EVEN_K = K % block_k == 0
    # print(f'{M=}, {N=}, {K=}, {block_m=}, {block_n=}, {block_k=}')
    scaled_dot_kernel[grid](a, b, c, a_scale, b_scale, M, N, K,
                            b_scale.stride(0), a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0),
                            c.stride(1), sorted_token_ids_ptr, num_valid_tokens, DTYPE_A=dtype_converter[dtype_a],
                            DTYPE_B=dtype_converter[dtype_b], BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n,
                            BLOCK_SIZE_K=block_k, GROUP_SIZE_M=group_m, SPLIT_K=split_k, num_warps=num_warps,
                            num_stages=num_stages, waves_per_eu=waves_per_eu, matrix_instr_nonkdim=mfmaInstrSize,
                            kpack=kpack, EVEN_K=EVEN_K)
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
    mfma_instr_size = config["matrix_instr_nonkdim"]
    return block_m, block_n, block_k, group_m, split_k, num_warps, num_stages, waves_per_eu, mfma_instr_size, kpack


name_to_tl_types = {
    'int8': tl.int8,
    'int32': tl.int32,
    'fp16': tl.float16,
    'fp32': tl.float32,
    'bf16': tl.bfloat16,
    'fp8': tl.float8e4b8,
    'bf8': tl.float8e5b16,
}

tl_to_torch_types = {
    tl.float16: torch.float16,
    tl.bfloat16: torch.bfloat16,
    tl.float32: torch.float32,
    tl.int8: torch.int8,
    tl.int32: torch.int32,
}


def f8_to_f16(x, dtype):

    @triton.jit
    def kernel(Y, X, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(X + offs, mask=mask)
        tl.store(Y + offs, x, mask=mask)

    ret = torch.empty(x.shape, dtype=torch.float16, device=x.device)

    def grid(META):
        return (triton.cdiv(x.numel(), META['BLOCK_SIZE']), )

    dtype = getattr(tl, dtype)
    kernel[grid](ret, triton.reinterpret(x, dtype), ret.numel(), BLOCK_SIZE=1024)
    return ret


def create_operand(dtype: str, size0: int, size1: int, k_dim: int, transpose: bool = True, pack_along_k: bool = True):
    if dtype == "float8e5":
        if transpose:
            v = torch.randint(20, 40, (size0, size1), dtype=torch.uint8).view(torch.float8_e5m2).to('cuda')
            v_ref = f8_to_f16(v.view(torch.float8_e5m2), dtype).to(torch.float32)
        else:
            v = torch.randint(20, 40, (size1, size0), dtype=torch.uint8).view(torch.float8_e5m2).to('cuda').T
            v_ref = f8_to_f16(v.view(torch.float8_e5m2).T, dtype).to(torch.float32).T
    elif dtype == "float8e4nv":
        if transpose:
            v = torch.randint(20, 40, (size0, size1), dtype=torch.uint8).view(torch.float8_e4m3fn).to('cuda')
            v_ref = f8_to_f16(v.view(torch.float8_e4m3fn), dtype).to(torch.float32)
        else:
            v = torch.randint(20, 40, (size1, size0), dtype=torch.uint8).view(torch.float8_e4m3fn).to('cuda').T
            v_ref = f8_to_f16(v.view(torch.float8_e4m3fn).T, dtype).to(torch.float32).T
    else:
        # float4
        if pack_along_k:
            pack_dim = k_dim
        else:
            pack_dim = (k_dim + 1) % 2
        if transpose:
            v_mxfp4 = MXFP4Tensor(size=(size0, size1), device='cuda').random()
            v = v_mxfp4.to_packed_tensor(dim=pack_dim)
            v_ref = v_mxfp4.to(torch.float32)
        else:
            v_mxfp4 = MXFP4Tensor(size=(size1, size0), device='cuda').random()
            v = v_mxfp4.to_packed_tensor(dim=(pack_dim + 1) % 2).T
            v_ref = v_mxfp4.to(torch.float32).T
    return v, v_ref


def get_type(provider):
    res = re.findall(r'\(.*?\)', provider)
    return res[0][1:-1].split('/', 1)


x_vals = [(1024 * v, 1024 * v, 1024 * v) for v in range(1, 9)]
x_vals += [(4864, 4096, 8192), (9728, 8192, 65536), (4864, 8192, 4096)]
x_vals += [(16, 4096, 2048), (16, 4096, 4096)]
# x_vals += [(16, 16, 1024 * v) for v in range(1, 9)]
# x_vals += [(32, 32, 1024 * v) for v in range(1, 9)]

line_vals = []
line_names = []
for dtype_a, dtype_b in (('float8e4nv', 'float8e4nv'), ('float8e4nv', 'float4'), ('float4', 'float4')):
    for backend in ('triton', 'torch'):
        line_vals.append(f'{backend}({dtype_a}/{dtype_b})')
        line_names.append(f'{backend}({dtype_a}/{dtype_b})')


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=x_vals,
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_names,
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="scaled-moe-performance",
        args={},
    ))
def benchmark(M, N, K, provider):
    global block_m, block_n, block_k, dtype_c, num_stages, num_warps, waves_per_eu, mfmaInstrSize

    group_m = 1
    split_k = 1
    kpack = 1

    bias = None
    use_bias = False

    dtype_a, dtype_b = get_type(provider)
    a, a_fp16 = create_operand(dtype_a, M, K, 1)
    b, b_fp16 = create_operand(dtype_b, K, N, 0, False)  # TN
    a_scale_mxfp4 = MXScaleTensor(size=(M, (K + 32 - 1) // 32), device='cuda').random(high=32.0)
    b_scale_mxfp4 = MXScaleTensor(size=(N, (K + 32 - 1) // 32), device='cuda').random(high=32.0)
    a_scale = a_scale_mxfp4.data
    b_scale = b_scale_mxfp4.data
    a_scale_ref = a_scale_mxfp4.to(torch.float32).repeat_interleave(32, dim=1)[:M, :K]
    b_scale_ref = b_scale_mxfp4.to(torch.float32).repeat_interleave(32, dim=1).T.contiguous()[:K, :N]

    quantiles = [0.5, 0.2, 0.8]
    if provider.startswith('triton'):
        c = torch.zeros((M, N), device=a.device, dtype=tl_to_torch_types[name_to_tl_types[dtype_c]])

        sorted_token_ids_ptr = torch.arange(0, M, dtype=torch.int32, device=a.device)
        num_valid_tokens = M // 2

        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: invoke_scaled_moe(a, b, c, a_scale, b_scale, bias, block_m, block_n, block_k, dtype_a, dtype_b,
                                      group_m, split_k, num_warps, num_stages, waves_per_eu, mfmaInstrSize, kpack,
                                      use_bias, sorted_token_ids_ptr, num_valid_tokens), quantiles=quantiles)
    else:
        a_fp16[(M // 2):, :] = 0
        a_scale_ref[(M // 2):, :] = 0
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a_fp16 * a_scale_ref, b_fp16 * b_scale_ref),
                                                     quantiles=quantiles)

    def perf(ms):
        return (2 * M * N * K + 2 * M * N * K / 32) * 1e-12 / (ms * 1e-3)

    return perf(ms), perf(max_ms), perf(min_ms)


def run_test(M, N, K, block_m, block_n, block_k, dtype_a, dtype_b, dtype_c='fp32', col_a=False, col_b=True, group_m=1,
             split_k=1, num_warps=4, num_stages=1, waves_per_eu=1, mfmaInstrSize=16, kpack=1, use_bias=False,
             scaled=True, init_type="randn"):
    torch.manual_seed(42)

    if scaled:
        a, a_fp16 = create_operand(dtype_a, M, K, 1)
        b, b_fp16 = create_operand(dtype_b, K, N, 0, False)  # TN
        a_scale_mxfp4 = MXScaleTensor(size=(M, (K + 32 - 1) // 32), device='cuda').random(high=32.0)
        b_scale_mxfp4 = MXScaleTensor(size=(N, (K + 32 - 1) // 32), device='cuda').random(high=32.0)
        a_scale = a_scale_mxfp4.data
        b_scale = b_scale_mxfp4.data
        a_scale_ref = a_scale_mxfp4.to(torch.float32).repeat_interleave(32, dim=1)[:M, :K]
        b_scale_ref = b_scale_mxfp4.to(torch.float32).repeat_interleave(32, dim=1).T.contiguous()[:K, :N]
    else:
        a, a_fp16 = gen_input(M, K, dtype_a, col_a, 1, init_type, device='cuda')
        b, b_fp16 = gen_input(K, N, dtype_b, col_b, 2, init_type, device='cuda')

    bias = None
    if use_bias:
        bias, bias_fp16 = gen_input(M, 1, dtype_b, col_b, 2, init_type, device='cuda')
        bias = bias.squeeze()
        bias_fp16 = bias.squeeze()
    # Allocates output.
    c = torch.zeros((M, N), device=a.device, dtype=tl_to_torch_types[name_to_tl_types[dtype_c]])

    # Allocate a fake expert_ids_ptr as 0,1,...,M-1
    sorted_token_ids_ptr = torch.arange(0, M, dtype=torch.int32, device=a.device)
    num_valid_tokens = M // 2
    if scaled:
        triton_output = invoke_scaled_moe(a, b, c, a_scale, b_scale, bias, block_m, block_n, block_k, dtype_a, dtype_b,
                                          group_m, split_k, num_warps, num_stages, waves_per_eu, mfmaInstrSize, kpack,
                                          use_bias, sorted_token_ids_ptr, num_valid_tokens)
        a_fp16[(M // 2):, :] = 0
        a_scale_ref[(M // 2):, :] = 0
        torch_output = torch.matmul(a_fp16 * a_scale_ref, b_fp16 * b_scale_ref)
    else:
        triton_output = invoke_moe(a, b, c, bias, block_m, block_n, block_k, group_m, split_k, num_warps, num_stages,
                                   waves_per_eu, mfmaInstrSize, kpack, use_bias, sorted_token_ids_ptr, num_valid_tokens)
        a[8:16, :] = 0
        torch_output = torch.matmul(a, b)
    if use_bias:
        torch_output += bias_fp16[:, None]
    if dtype_a == 'float4' and dtype_b == 'float4':
        rtol = 1e-1
        atol = 1e-1
    else:
        rtol = 0 if torch.version.hip is None else 1e-2
        atol = 1e-2 if split_k == 1 else 4e-2
    row_a_str = 'N' if col_a else 'T'
    row_b_str = 'N' if col_b else 'T'
    size_str = ''
    torch.set_printoptions(linewidth=400)
    torch.set_printoptions(threshold=2048)
    torch.set_printoptions(sci_mode=False)
    size_str = f'SIZE M: {M}, N: {N}, K: {K}, trans: {row_a_str}{row_b_str}'
    if not torch.allclose(triton_output, torch_output, atol=atol, rtol=rtol):
        print(f"triton_output={triton_output}")
        print(f"torch_output={torch_output}")
        print(f"diff={torch_output - triton_output}")
        print(f"div={torch_output / triton_output}")
        # mismatch = torch_output[:8, :] != triton_output[:8, :]
        # print(f'Num mismatch: {torch.sum(mismatch, dim=1)}')
        print(f'{size_str} Incorrect❌')
        torch.testing.assert_close(triton_output, torch_output, atol=atol, rtol=rtol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()

    # block_m = 256
    # block_n = 256
    # block_k = 128

    block_m = 16
    block_n = 256
    block_k = 256

    num_stages = 1
    num_warps = 4
    waves_per_eu = 1
    mfmaInstrSize = 16
    dtype_c = "fp32"

    if args.benchmark:
        benchmark.run(print_data=True)
    else:
        for M, N, K in x_vals:
            for dtype_a, dtype_b in (("float8e4nv", "float8e4nv"), ("float8e4nv", "float4"), ("float4", "float4")):
                try:
                    run_test(M, N, K, block_m, block_n, block_k, dtype_a, dtype_b, dtype_c, num_stages=num_stages,
                             num_warps=num_warps, waves_per_eu=waves_per_eu, mfmaInstrSize=mfmaInstrSize)
                except Exception as e:
                    print(f'Failed test: {M=}, {N=}, {K=}, '
                          f'{block_m=}, {block_n=}, {block_k=}, '
                          f'{dtype_a=}, {dtype_b=}, '
                          f'{num_stages=}, {num_warps=}, '
                          f'{waves_per_eu=}, {mfmaInstrSize=}')
                    raise e
        print(f'All tests pass✅')
