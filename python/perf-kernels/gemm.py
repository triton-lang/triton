import torch
import triton
import triton.language as tl
import sys
import argparse
import pytest
import re

from utils.benchmark_utils import get_available_models, get_model_configs


@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4, 'waves_per_eu': 0},
            num_warps=8, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'waves_per_eu': 2,
                'kpack': 2, 'matrix_instr_nonkdim': 16
            }, num_warps=8, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'waves_per_eu': 0,
                'kpack': 1
            }, num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'waves_per_eu': 0},
            num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 32, 'waves_per_eu': 2},
            num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
    use_cuda_graph=True,
)
@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % args['BLOCK_SIZE_K'] == 0,
})
@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    a_scale_ptr,
    b_scale_ptr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    APPLY_SCALE: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # TODO(vgokhale): Add XCD remapping.
    pid = tl.program_id(axis=0)
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

    tl.assume(pid_m > 0)
    tl.assume(pid_n > 0)

    # Create pointers for first block of A and B input matrices
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    if APPLY_SCALE:
        a_scale = tl.load(a_scale_ptr)
        b_scale = tl.load(b_scale_ptr)

    acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # Apply scale to recover dynamic range reduced due to lower precision inputs.
    if APPLY_SCALE:
        accumulator = accumulator * a_scale * b_scale
    # Apply activation function, if specified.
    # TODO(vgokhale): Add different types of activations.
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(c_ptr.type.element_ty)

    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# Activation function.
@triton.jit
def leaky_relu(x):
    x = x + 1
    return tl.where(x >= 0, x, 0.01 * x)


# Wrapper for gemm kernel.
def matmul(a, b, c, a_scale, b_scale, scale_a8_b8=False, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions!!!"
    assert a.dtype == b.dtype, "Mixed dtype GEMMs are not supported!!!"
    M, K = a.shape
    K, N = b.shape
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a,
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
        a_scale,
        b_scale,
        APPLY_SCALE=scale_a8_b8,
        ACTIVATION=activation,
    )


name_to_torch_types = {
    'int8': torch.int8,
    'int32': torch.int32,
    'fp16': torch.float16,
    'fp32': torch.float32,
    'bf16': torch.bfloat16,
    'fp8e5': torch.float8_e5m2fnuz,
    'fp8e4': torch.float8_e4m3fnuz,
}

dtype_max = {
    dtype: (torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)).max
    for dtype in [
        torch.float8_e5m2fnuz,
        torch.float8_e4m3fnuz,
        torch.int8,
    ]
}


def dtype_is_8_bit(dtype):
    return (dtype is torch.float8_e5m2fnuz) or \
           (dtype is torch.float8_e4m3fnuz) or \
           (dtype is torch.int8)


def gen_input(M, N, dtype, needTrans, seed, device='cuda'):
    torch.manual_seed(seed)

    if needTrans:
        raw_data = torch.randn((N, M), dtype=torch.float32, device='cuda').T
    else:
        raw_data = torch.randn((M, N), dtype=torch.float32, device='cuda')
    scale = None
    if dtype_is_8_bit(dtype):
        max_val = torch.max(torch.abs(raw_data))
        scale = max_val / dtype_max[dtype]
        raw_data = raw_data / scale

    input = raw_data.to(dtype)
    input_f32 = input.to(torch.float32)

    return input, input_f32, scale


def get_x_vals():
    x_vals = [(1024 * v, 1024 * v, 1024 * v) for v in range(1, 9)]

    x_vals += [(4864, 4096, 8192), (9728, 8192, 65536), (4864, 8192, 4160)]

    return x_vals


# Unit tests
#TODO(vgokhale): Test activation.
@pytest.mark.parametrize(
    "M, N, K, in_dtype, out_dtype, col_a, col_b",
    [(*shape, in_dtype, out_dtype, col_a, col_b)
     for shape in get_x_vals()
     for in_dtype, out_dtype in [('fp16', 'fp16'), ('bf16', 'bf16'), ('fp32', 'fp32'), (
         'fp8e4', 'fp16'), ('fp8e5', 'fp16'), ('int8', 'int8'), ('int8', 'int32')]
     # Defines if a matrix is row or column major.
     for col_a in [True, False]
     for col_b in [True, False]])
def test_correctness(M, N, K, col_a, col_b, in_dtype, out_dtype):
    torch_in_dtype = name_to_torch_types[in_dtype]
    a, a_fp32, a_scale = gen_input(M, K, torch_in_dtype, col_a, 1, device='cuda')
    b, b_fp32, b_scale = gen_input(K, N, torch_in_dtype, col_b, 2, device='cuda')
    torch_out_dtype = name_to_torch_types[out_dtype]
    c = torch.empty((M, N), device=a.device, dtype=torch_out_dtype)
    # For 8-bit, we have scaled to the dynamic range of the data type.
    # This requires us to compute in fp32 because for e5m2, the range is same as fp16 (e5m10).
    # If we use fp16 it is possible to return infs from the torch.matmul call.
    if dtype_is_8_bit(torch_in_dtype):
        matmul(a, b, c, a_scale, b_scale, scale_a8_b8=True, activation="")
        torch_output = torch.matmul(a_fp32, b_fp32)
        torch_output = torch_output * a_scale * b_scale
    # For other dtypes, use the same torch matmul as the dtype.
    else:
        matmul(a, b, c, a_scale=None, b_scale=None, scale_a8_b8=False, activation="")
        torch_output = torch.matmul(a.to(torch_in_dtype), b.to(torch_in_dtype))
    if out_dtype == 'int8':
        torch.testing.assert_close(c.to(torch.float32),
                                   torch_output.to(torch.int8).to(torch.float32), atol=1e-3, rtol=1e-2)
    else:
        torch.testing.assert_close(c, torch_output.to(torch_out_dtype), atol=5e-3, rtol=1e-2)


def get_type(provider):
    res = re.findall(r'\(.*?\)', provider)
    return res[0][1:-1]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],
        x_vals=get_x_vals(),
        line_arg='provider',
        line_vals=[
            'rocblas(fp16)', 'rocblas(bf16)', 'triton(fp16)', 'triton(bf16)', 'triton(int8)', 'triton(fp8e4)',
            'triton(fp8e5)'
        ],
        line_names=[
            "rocBLAS.Fp16", "rocBLAS.Bf16", "Triton.Fp16", "Triton.Bf16", "Triton.Int8", "Triton.Fp8E4", "Triton.Fp8E5"
        ],
        ylabel="TFLOPS",
        plot_name="matmul-performance",
        args={},
    ))
def benchmark(M, N, K, provider, model=None):
    in_dtype = name_to_torch_types[get_type(provider)]
    out_dtype = in_dtype

    quantiles = [0.5, 0.2, 0.8]
    if 'rocblas' in provider:
        a = torch.randn((M, K), dtype=in_dtype, device='cuda')
        b = torch.randn((K, N), dtype=in_dtype, device='cuda')

        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    else:  # triton, different data types
        assert "triton" in provider
        a, _, a_scale = gen_input(M, K, in_dtype, False, 1, device='cuda')
        b, _, b_scale = gen_input(K, N, in_dtype, True, 2, device='cuda')
        # Allocates output.
        c = torch.empty((M, N), device=a.device, dtype=out_dtype)

        if dtype_is_8_bit(in_dtype):
            a_scale = a_scale.item()
            b_scale = b_scale.item()
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b, c, a_scale, b_scale, activation=""),
                                                     quantiles=quantiles)
        global verbose
        if verbose:
            print(f'SIZE: {M},{N},{K}   Best tuning config: ({matmul_kernel.best_config()})')
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="GEMM tutorial example",
        allow_abbrev=False,
    )

    parser.add_argument('-model_configs', type=str, default="model_configs.json", help="Model config json file.")

    available_models = get_available_models(model_families=["llama3"])  # Dynamically load model names
    model_help = (
        "Model name to benchmark. Select from: [" + ", ".join(available_models) +
        "]. Use 'all' to benchmark all models. Not providing runs the default benchmark script with custom configs.")
    parser.add_argument('-model', type=str, default=None, help=model_help)
    parser.add_argument('-b', type=int, default=0, help="Batch size used together with model.")
    parser.add_argument('-sq', type=int, default=0, help="Sequence length used together with model.")

    parser.add_argument("-v", action='store_true', default=False, help="Print out the best tuning config")
    parser.add_argument("-M", type=int, default=0)
    parser.add_argument("-N", type=int, default=0)
    parser.add_argument("-K", type=int, default=0)

    args = parser.parse_args()

    return args


def main():
    # assign to a global verbose var to indicate whether print
    # best tuning config
    global verbose
    args = parse_args()
    verbose = args.v

    if args.model:
        config_file = args.model_configs
        configs = get_model_configs(config_path=config_file, model_families=["llama3"], model=args.model)
        mnk_list = []
        batch_size = args.b if args.b else 1

        for model_name, config in configs.items():
            seq_len = args.sq if args.sq else 4096
            M, N, K = batch_size * seq_len, config["hidden_size"], config["intermediate_size"]
            mnk_list.append((model_name, M, N, K))

        benchmark.benchmarks.x_names = ['model', 'M', 'N', 'K']
        benchmark.benchmarks.x_vals = mnk_list

    if args.M or args.N or args.K:
        assert args.model is None, "Providing both -model and -M/N/K is not compatible! -model already fixes -M/N/K."

    if args.M and args.N and args.K:
        x_vals = [(args.M, args.N, args.K)]
        benchmark.benchmarks.x_vals = x_vals

    benchmark.run(show_plots=True, print_data=True)


if __name__ == '__main__':
    sys.exit(main())
