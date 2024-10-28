#!/usr/bin/env python3

import argparse
import sys
import yaml
import os
from pathlib import Path
import glob

import torch
import triton
import triton.language as tl

from matmul_kernel import matmul_kernel

from datetime import datetime
import multiprocessing
import pandas as pd

from utils.file_generator import (
    gen_configStr,
    generate_compile_driver,
    generate_matmul_kernels,
    generate_profile_tasks,
    read_config,
)
from utils.utils import (
    get_default_tuning_result_filename,
    get_filename_compile_driver,
    get_filename_myKernels,
    get_filename_profile_driver,
    get_output_dir,
    name_to_tl_types,
    patch_triton_compiler,
    run_bash_command,
    run_bash_command_wrapper,
    tl_to_torch_types,
    TORCH_HAS_FP8E4B8,
    TORCH_HAS_FP8E5B16,
)


def is_hip_available():
    try:
        __import__("hip")
    except ImportError:
        return False
    else:
        return True


def get_full_tuning_space():
    configs = []

    block_mn_range = [16, 32, 64, 128, 256]
    block_k_range = [16, 32, 64, 128, 256]
    split_k_range = [1, 2, 4, 5, 6, 8, 10, 12, 16, 18, 24]
    num_warps_range = [1, 2, 4, 8]
    group_m_range = [1, 2, 4, 8, 16, 32]
    # For now we see better perf with num_stages=2 for all gemm configs we care
    # But keep this explicit so that we do not forget we may need to set it to
    # other values in the future
    num_stage_range = [2]
    waves_per_eu_range = [0]
    matrix_instr_nonkdim_range = [16, 32]
    kpack_range = [1, 2]
    sched_variants = ["\"default\"", "\"iglp0\"", "\"ck_v3\""]

    for block_m in block_mn_range:
        for block_n in block_mn_range:
            for block_k in block_k_range:
                for num_warps in num_warps_range:
                    for group_m in group_m_range:
                        for split_k in split_k_range:
                            for num_stages in num_stage_range:
                                for waves_per_eu in waves_per_eu_range:
                                    for matrix_instr_nonkdim in matrix_instr_nonkdim_range:
                                        for sched_variant in sched_variants:
                                            for kpack in kpack_range:
                                                configs.append({
                                                    'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K':
                                                    block_k, 'GROUP_SIZE_M': group_m, 'SPLIT_K': split_k, 'num_warps':
                                                    num_warps, 'num_stages': num_stages, 'waves_per_eu': waves_per_eu,
                                                    'matrix_instr_nonkdim': matrix_instr_nonkdim, 'kpack': kpack,
                                                    'instruction_sched_variant': sched_variant
                                                })

    return configs


def get_default_config():
    full_configs = get_full_tuning_space()
    return full_configs[0]


def prune_configs(M, N, K, configs, elemBytes_a, elemBytes_b):
    pruned_configs = []

    if M < 32 or N < 32:
        mfma = 16
    else:
        mfma = 32

    # TODO (zhanglx): figure out the boundary between large and small gemms
    large_gemm = False
    if M >= 2048 and N >= 2048:
        large_gemm = True

    for config in configs:
        BLOCK_SIZE_M = config.get("BLOCK_SIZE_M")
        BLOCK_SIZE_N = config.get("BLOCK_SIZE_N")
        BLOCK_SIZE_K = config.get("BLOCK_SIZE_K")
        num_warps = config.get("num_warps")
        num_stages = config.get("num_stages")
        matrix_instr_nonkdim = config.get("matrix_instr_nonkdim")
        if matrix_instr_nonkdim > mfma:
            continue
        if mfma == 4 and BLOCK_SIZE_K < 64:
            continue
        # some layouts could not work properly in case
        # number elemens per thread is less 1
        if BLOCK_SIZE_M * BLOCK_SIZE_N < 64:
            continue
        SPLIT_K = config.get("SPLIT_K")
        GROUP_M = config.get("GROUP_SIZE_M")
        if BLOCK_SIZE_M < matrix_instr_nonkdim or BLOCK_SIZE_N < matrix_instr_nonkdim:
            continue
        if M <= matrix_instr_nonkdim and BLOCK_SIZE_M != matrix_instr_nonkdim:
            continue
        if N <= matrix_instr_nonkdim and BLOCK_SIZE_N != matrix_instr_nonkdim:
            continue
        # Skip BLOCK_SIZE that is too large compare to M/N
        # unless BLOCK_SIZE is already small enough
        if BLOCK_SIZE_M > M * 2 and BLOCK_SIZE_M != 16:
            continue
        if BLOCK_SIZE_N > N * 2 and BLOCK_SIZE_N != 16:
            continue
        # skip large split_k when not necessary
        if SPLIT_K != 1 and not need_split_k(M, N, K):
            continue
        # skip split_k that leads to EVEN_K = false
        leap = SPLIT_K * BLOCK_SIZE_K
        modv = K % leap
        if modv != 0 and SPLIT_K != 1:
            continue
        # skip large GROUP_M
        if GROUP_M * BLOCK_SIZE_M > M and GROUP_M != 1:
            continue
        # out of shared memory resource
        # TODO (zhanglx): This does not consider the LDS usage in the epilogue
        LDSA = BLOCK_SIZE_K * BLOCK_SIZE_M * elemBytes_a
        LDSB = BLOCK_SIZE_K * BLOCK_SIZE_N * elemBytes_b
        if num_stages <= 1:
            # No pipeline, buffer A and buffer B can re-use each other
            LDS = max(LDSA, LDSB)
        else:
            # Pipeline, we need (num_stages - 1) buffers for both A and B at the same time
            LDS = (LDSA + LDSB) * (num_stages - 1)
        if LDS > 65536:
            continue
        # Skip small block sizes and num_warps for large gemm
        # For fp16 and f8, we want to only use BLOCK_SIZE >= 64
        if large_gemm:
            if BLOCK_SIZE_M < 64 or BLOCK_SIZE_N < 64:
                continue
            if BLOCK_SIZE_K < 64:
                continue
            if num_warps < 4:
                continue
            # check if tiling is integer multiple of GEMM size because we have no boundary check
            if M % BLOCK_SIZE_M != 0 or N % BLOCK_SIZE_N != 0:
                continue

        pruned_configs.append(config)

    return pruned_configs


def need_split_k(SIZE_M, SIZE_N, SIZE_K):
    return (SIZE_M < 64 or SIZE_N < 64) and SIZE_K > 1024


def extract_kernel_time(M, N, K, config, df):
    configStr = gen_configStr(config)
    df = df[df['KernelName'].str.contains(configStr)]

    first_value = df['DurationNs'].iloc[0]
    filtered_data = df['DurationNs'][df['DurationNs'] <= first_value]
    new_meanTime = filtered_data.tail(100).mean()

    return config, new_meanTime


def profile_batch_kernels(M, N, K, gpuid, gpus, jobs, verbose):
    ngpus = len(gpus)
    gpuIdx = gpus.index(gpuid)
    if gpuIdx + 1 > jobs:
        return
    os.environ['ROCR_VISIBLE_DEVICES'] = str(gpuid)
    jobId = gpuIdx
    while jobId < jobs:
        kernel_name = get_filename_profile_driver(M, N, K, jobId)
        if verbose:
            print(f"profiling {kernel_name} on GPU {gpuid}")
        here = Path(__file__).parent
        run_bash_command_wrapper(
            f"PYTHONPATH={here} rocprof --stats -o {get_output_dir()}/results_{jobId}.csv python {get_filename_profile_driver(M, N, K, jobId)}",
            capture=(verbose < 2))
        jobId += ngpus


def tune_gemm_config(M, N, K, col_a, col_b, dtype_a, dtype_b, dtype_c, init_type, configs, run_bench, jobs, iters,
                     skipWarmup, verbose=0, num_threads=32, gpus=[0], rotating_buffer_size=256, bias_size=0,
                     icache_flush=False):

    # precompile the kernels in parallel
    start_time = datetime.now()
    if not skipWarmup:
        # Generate kernel out of all configs
        fname = generate_compile_driver(M, N, K, col_a, col_b, dtype_a, dtype_b, dtype_c, init_type, configs,
                                        rotating_buffer_size, bias_size)
        here = Path(__file__).parent
        run_bash_command(f"PYTHONPATH={here} python {fname} -n {num_threads}", capture=(verbose < 2))
    compile_end = datetime.now()
    compile_time = compile_end - start_time
    if verbose:
        print(f"compile time: {compile_time}", flush=True)

    # Generate kernels out of all configs
    generate_profile_tasks(M, N, K, col_a, col_b, dtype_a, dtype_b, dtype_c, init_type, configs, jobs, iters, run_bench,
                           rotating_buffer_size, bias_size, icache_flush)

    # profile generated kernels
    running = [
        multiprocessing.Process(target=profile_batch_kernels, args=(M, N, K, gpu_id, gpus, jobs, verbose))
        for gpu_id in gpus
    ]
    for p in running:
        p.start()
    for p in running:
        p.join()

    profile_end = datetime.now()
    profile_time = profile_end - compile_end
    if verbose:
        print(f"profile time: {profile_time}", flush=True)

    # post process results.csv to get the best config and minTime
    # TODO: process the file in parallel
    minTime = 1024 * 1024 * 1024
    thread_pool = multiprocessing.Pool(processes=num_threads)
    tasks = []
    idx = 0
    df_prof = [pd.read_csv(f"{get_output_dir()}/results_{i}.csv") for i in range(jobs)]
    for config in configs:
        file_idx = idx % jobs
        tasks += [thread_pool.apply_async(extract_kernel_time, args=(M, N, K, config, df_prof[file_idx]))]
        idx += 1
    thread_pool.close()
    thread_pool.join()

    for task in tasks:
        config, myTime = task.get()
        if myTime:
            min_us = myTime / 1000
            if min_us < minTime:
                minTime = min_us
                bestConfig = config
        else:
            min_us = -1
            print(f"invalid config(post processing): SIZE {M} {N} {K}: {config}", flush=True)
    post_end = datetime.now()
    post_time = post_end - profile_end
    if verbose:
        print(f"post procesing time: {post_time}", flush=True)
    return minTime, bestConfig, compile_time, profile_time, post_time


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
            temp = torch.randn(size, dtype=dtype, device='cuda')
            return temp
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
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
        n_elements = raw_data.numel()
        copy_kernel[grid](input, input_f16, n_elements, BLOCK_SIZE=1024)

    return input, input_f16


# generate inputs/outputs according to rotating tensor size
def gen_rotating_tensors(M, N, K, dtype_a, need_Trans_a, dtype_b, need_Trans_b, dtype_c, seed, init_type,
                         rotating_buffer_size, bias_size, device='cuda'):
    a_size = M * K * type_name_to_bytes(dtype_a)
    b_size = K * N * type_name_to_bytes(dtype_b)
    c_size = M * N * type_name_to_bytes(dtype_c)
    bias_size = bias_size * type_name_to_bytes(dtype_c)

    total_size = a_size + b_size + c_size + bias_size
    block_count = rotating_buffer_size * 1024 * 1024 // total_size
    block_count = max(1, block_count)

    # generate input and outputs
    a = []
    b = []
    c = []
    bias = []
    for i in range(block_count):
        in_a, in_a_fp16 = gen_input(M, K, dtype_a, need_Trans_a, 1, init_type, device='cuda')
        a.append(in_a)
        in_b, in_b_fp16 = gen_input(K, N, dtype_b, need_Trans_b, 2, init_type, device='cuda')
        b.append(in_b)
        out_c = torch.zeros((M, N), dtype=tl_to_torch_types[name_to_tl_types[dtype_c]], device='cuda')
        c.append(out_c)
        if bias_size > 0:
            bs, bs_fp16 = gen_input(M, 1, dtype_b, need_Trans_b, 2, init_type, device='cuda')
            bias.append(bs.squeeze(dim=1))

    in_outs = {"rotating_num": block_count, "input_a": a, "input_b": b, "output_c": c, "bias": bias}

    return in_outs


def matmul(a, b, c, bias, block_m, block_n, block_k, group_m, split_k, num_warps, num_stages, waves_per_eu,
           mfmaInstrSize, kpack, use_bias, sched_variant):
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
    num_xcds = 1 if split_k > 1 else 8
    matmul_kernel[grid](a, b, c, bias, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0),
                        c.stride(1), stride_bias=stride_bias, BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n,
                        BLOCK_SIZE_K=block_k, GROUP_SIZE_M=group_m, SPLIT_K=split_k, num_warps=num_warps,
                        num_stages=num_stages, waves_per_eu=waves_per_eu, matrix_instr_nonkdim=mfmaInstrSize,
                        kpack=kpack, BIAS=use_bias, EVEN_K=EVEN_K, GRID_MN=grid[0], NUM_XCDS=num_xcds,
                        instruction_sched_variant=sched_variant)
    return c


def test_correctness(M, N, K, col_a, col_b, dtype_a, dtype_b, dtype_c, init_type, config, bias_vector, verbose):
    block_m, block_n, block_k, group_m, split_k, num_warps, num_stages, waves_per_eu, mfmaInstrSize, kpack, sched_variant = read_config(
        config)
    use_bias = bias_vector
    torch.manual_seed(0)
    #a = torch.randn((M, K), device='cuda', dtype=datatype)
    #b = torch.randn((K, N), device='cuda', dtype=datatype)
    a, a_fp16 = gen_input(M, K, dtype_a, col_a, 1, init_type, device='cuda')
    b, b_fp16 = gen_input(K, N, dtype_b, col_b, 2, init_type, device='cuda')
    bias = None
    if use_bias:
        bias, bias_fp16 = gen_input(M, 1, dtype_b, col_b, 2, init_type, device='cuda')
        bias = bias.squeeze(dim=1)
        bias_fp16 = bias.squeeze(dim=1)
    # Allocates output.
    c = torch.zeros((M, N), device=a.device, dtype=tl_to_torch_types[name_to_tl_types[dtype_c]])
    triton_output = matmul(a, b, c, bias, block_m, block_n, block_k, group_m, split_k, num_warps, num_stages,
                           waves_per_eu, mfmaInstrSize, kpack, use_bias, sched_variant)
    torch_output = torch.matmul(a_fp16, b_fp16)
    if use_bias:
        torch_output += bias_fp16[:, None]
    rtol = 0 if torch.version.hip is None else 1e-2
    atol = 1e-3 if split_k == 1 else 4e-2
    row_a_str = 'N' if col_a else 'T'
    row_b_str = 'N' if col_b else 'T'
    size_str = ''
    if verbose:
        size_str = f'SIZE M: {M}, N: {N}, K: {K}, trans: {row_a_str}{row_b_str}'
    if torch.allclose(triton_output.to(torch.float16), torch_output, atol=atol, rtol=rtol):
        print(f'{size_str} Correct✅')
    else:
        print(f"triton_output={triton_output}")
        print(f"torch_output={torch_output}")
        print(f'{size_str} Incorrect❌')


def parse_args():
    parser = argparse.ArgumentParser(
        prog="tune a specific gemm size",
        allow_abbrev=False,
    )

    parser.add_argument("-m", type=int, default=0)
    parser.add_argument("-n", type=int, default=0)
    parser.add_argument("-k", type=int, default=0)
    parser.add_argument("-col_a", action='store_true', default=False, help='whether matrix a is column major')
    parser.add_argument("-col_b", action='store_true', default=False, help='whether matrix b is column major')
    parser.add_argument("-dtype_a", type=str, default='fp16', help="matrix a element data type")
    parser.add_argument("-dtype_b", type=str, default='fp16', help="matrix b element data type")
    parser.add_argument("-dtype_c", type=str, default='fp16', help="output element data type")
    parser.add_argument("--ngpus", type=int, default=0, help='number of GPUs used in the profiling step')
    parser.add_argument("--gpu_ids", type=lambda s: [int(id) for id in s.split(',')], default=[],
                        help='list of gpu ids to use for tuning')
    parser.add_argument("--gemm_size_file", type=str, default="", help='yaml file to indicate matrix size')
    parser.add_argument("--o", type=str, default='', help='yaml file to store tuning results')
    parser.add_argument("--keep", action='store_true', default=False, help='keep generated files')
    parser.add_argument("--compare", action='store_true', default=False, help="Whether check result correctness")
    parser.add_argument("--compare_wo_tuning", action='store_true', default=False,
                        help="Whether check result correctness without tuning.")
    parser.add_argument("--benchmark", action='store_true', default=False, help="Benchmark the given config")
    parser.add_argument("--time_breakdown", action='store_true', default=False,
                        help="Show detailed time breakdown of each step during the tuning")
    parser.add_argument("--verbose", action='store_true', default=False,
                        help="enables time_breakdown and additional logging messages")
    parser.add_argument("--num_threads", type=int, default=32,
                        help="number of threads to use for kernel compilation and post processing")
    parser.add_argument("--jobs", type=int, default=1, help="number of tasks during the profiling process")
    parser.add_argument("--iters", type=int, default=200, help="number of iterations used in --benchmark mode")
    parser.add_argument("--init_type", type=str, default='randn', choices=['randn', 'hpl', 'trig_float', 'zeros'],
                        help="Input tensor initialization (default normal distribution)")
    parser.add_argument(
        "--rotating_tensor", type=int, default=0, help="total size (MB) of all tensors (a, b, c, bias)."
        " The default value is 0 (no rotating tensor)."
        " When set, it needs to be larger than the L1, L2, MALL size)")
    parser.add_argument("--bias_vector", action='store_true', default=False, help="apply bias vector")
    parser.add_argument("--icache_flush", action='store_true', default=False,
                        help="apply icache flush in tuning performance")
    parser.add_argument("--no_warmup", action='store_true', default=False,
                        help="Whether we want to skip the compilation stage")
    parser.add_argument("--hack_triton_compiler", action='store_true', default=False,
                        help="Modify the triton source to avoid backend query")
    args = parser.parse_args()
    if not args.o:
        if args.benchmark:
            args.o = f"{get_output_dir()}/benchmarking_results.csv"
        else:
            args.o = get_default_tuning_result_filename()

    return args


def process_item(item):
    M = item['M']
    N = item['N']
    K = item['K']
    col_a = False if item['rowMajorA'] == 'T' else True
    col_b = False if item['rowMajorB'] == 'T' else True
    del item['M']
    del item['N']
    del item['K']
    del item['rowMajorA']
    del item['rowMajorB']
    return M, N, K, col_a, col_b, item


def type_name_to_bytes(ty_name):
    if '32' in ty_name:
        return 4
    if '16' in ty_name:
        return 2
    if '8' in ty_name:
        return 1
    else:
        print(f"Unrecognized input type name {ty_name}")
        sys.exit(1)


def format_output(unformatted):
    if unformatted < 0.0001:
        formatted = "{:.3e}".format(unformatted)
    elif unformatted > 1000:
        formatted = "{:.1f}".format(unformatted)
    else:
        formatted = "{:.2f}".format(unformatted)
    return formatted


def get_rocm_version():
    torch_hip_version = torch.version.hip
    vers = torch_hip_version.split('.')
    ret_ver = '$rocm_version'
    if len(vers) >= 2:
        ret_ver = vers[0] + '.' + vers[1]
    return ret_ver


def main():
    args = parse_args()
    matrix_size_file = args.gemm_size_file
    output_file = args.o
    keepTmp = args.keep
    run_bench = args.benchmark
    jobs = args.jobs
    iters = args.iters
    skipWarmup = args.no_warmup
    hack_triton = args.hack_triton_compiler

    # Get GPU ids
    ngpus = args.ngpus
    gpu_ids = args.gpu_ids
    if ngpus != 0 and gpu_ids:
        print("--ngpus and --gpu_ids are mutually exclusive options")
        return os.EX_USAGE
    if ngpus == 0 and not gpu_ids:
        ngpus = 1
    if ngpus != 0:
        gpus = range(ngpus)
    if gpu_ids:
        gpus = gpu_ids

    if run_bench:
        gpus = [gpus[0]]
        jobs = 1

    # Get element type
    dtype_a = args.dtype_a
    dtype_b = args.dtype_b
    dtype_c = args.dtype_c
    if dtype_a not in name_to_tl_types or dtype_b not in name_to_tl_types or dtype_c not in name_to_tl_types:
        print(f"Unsupported dtype_a {args.dtype_a} or dtype_b {args.dtype_b} or dtype_c {args.dtype_c}")
        print("Supported types: ", list(name_to_tl_types.keys()))
        sys.exit(1)
    rotating_buffer_size = args.rotating_tensor
    bias_vector = args.bias_vector
    icache_flush = args.icache_flush
    if icache_flush:
        if not is_hip_available():
            print("************************************************************************************************")
            print("  `icache-flush` is disabled for this run.")
            print("  `icache-flush` needs python-hip module, which is unavailable.")
            print("  python-hip module can be installed as:")
            print(f"      `python3 -m pip install -i https://test.pypi.org/simple hip-python~={get_rocm_version()}`")
            print("************************************************************************************************")
            icache_flush = False

    mnks = []
    # TODO: make it more robust to get user input
    init_type = args.init_type
    if matrix_size_file == "" or not os.path.isfile(matrix_size_file):
        M = args.m
        N = args.n
        K = args.k
        col_a = args.col_a
        col_b = args.col_b
        mnks = [(M, N, K, col_a, col_b, None)]
    else:
        with open(matrix_size_file) as file:
            matrix_sizes = yaml.safe_load(file)
        for item in matrix_sizes:
            M, N, K, col_a, col_b, item = process_item(item)
            mnks.append((M, N, K, col_a, col_b, item))

    # Check correctness from given configs
    if args.compare_wo_tuning:
        for (M, N, K, col_a, col_b, myConfig) in mnks:
            if myConfig is None:
                raise Exception("kernel config is None, need to provide a tuning config")
            test_correctness(M, N, K, col_a, col_b, dtype_a, dtype_b, dtype_c, init_type, myConfig, bias_vector, True)
        return

    configs_full = get_full_tuning_space()

    start_time = datetime.now()
    # Append to the output file so that we can save all results into one file
    f_results = open(output_file, 'a')
    if run_bench:
        print(f"Benchmarking gemm with {dtype_a} inputs")
        print("trans     M      N      K    TFLOPS   us")
        f_results.write("trans,M,N,K,TFLOPS,us\n")
    else:
        print(f"Tuning {len(mnks)} gemm sizes starts at: {start_time}", flush=True)

    f_results.close()

    ## Before tuning starts, clear cache and previously generated kernel files
    run_bash_command("rm -rf ~/.triton/cache")
    run_bash_command(f"rm -rf {get_filename_myKernels()}")

    ## Modify triton compiler
    ## Hacky !!!
    if hack_triton:
        patch_triton_compiler()

    configs = []

    ## Big for loop of tuning
    ## Each iteration performs tuning for one gemm size
    for (M, N, K, col_a, col_b, myConfig) in mnks:

        f_results = open(output_file, 'a')

        start_local_time = datetime.now()
        # Obtain a pruned tuning space according to gemm size
        # If running benchmark, use the provided config
        pruned_configs = [myConfig] if run_bench else prune_configs(M, N, K, configs_full, type_name_to_bytes(dtype_a),
                                                                    type_name_to_bytes(dtype_b))

        ## Only append new configs from the current gemm size
        delta_configs = [config for config in pruned_configs if config not in configs]
        configs += delta_configs

        ## Append new configs into the tuning space
        generate_matmul_kernels(delta_configs)

        row_a_str = 'N' if col_a else 'T'
        row_b_str = 'N' if col_b else 'T'
        size_str = f'SIZE: {M} {N} {K} {row_a_str}{row_b_str}'
        if not run_bench:
            print(f"{size_str} nConfigs: {len(pruned_configs)}", end=" ", flush=True)
        else:
            print(f"{row_a_str}{row_b_str}    {M:5d}  {N:5d}  {K:5d}    ", end="")
            f_results.write(f"{row_a_str}{row_b_str},{M},{N},{K},")

        # The main tuning funtion for one gemm size
        verbose_level = 0
        if args.time_breakdown:
            verbose_level = 1
        if args.verbose:
            verbose_level = 2
        # we consider bias size as M for now.
        bias_size = M if bias_vector else 0
        minTime, bestConfig, compile_time, profile_time, post_time = tune_gemm_config(
            M, N, K, col_a, col_b, dtype_a, dtype_b, dtype_c, init_type, pruned_configs, run_bench, jobs, iters,
            skipWarmup, num_threads=args.num_threads, gpus=gpus, verbose=verbose_level,
            rotating_buffer_size=rotating_buffer_size, bias_size=bias_size, icache_flush=icache_flush)

        # post processing the numbers
        perf_tflops = lambda us: 2 * M * N * K * 1e-12 / (us * 1e-6)
        tri_tflops = perf_tflops(minTime)
        formatted_tflops = format_output(tri_tflops)
        minTime = format_output(minTime)
        if not run_bench:
            print(f'\nTFLOPS: {formatted_tflops}; time(us): {minTime}', end=" ", flush=True)

        bestConfig_compact_str = gen_configStr(bestConfig)
        if not run_bench:
            print(f'\nbest_config: {bestConfig_compact_str}', end=" ", flush=True)

        # write best config to tuning_results.yaml
        if run_bench:
            print(f"{formatted_tflops}     {minTime}")
            f_results.write(f"{formatted_tflops},{minTime}\n")

        sizeDict = {'M': M, 'N': N, 'K': K, 'rowMajorA': row_a_str, 'rowMajorB': row_b_str}
        sizeDict.update(bestConfig)
        if not run_bench:
            f_results.write("- " + str(sizeDict) + " ")
            f_results.write(f'# TFLOPS: {formatted_tflops} time(us): {minTime}\n')

        # remove generated files if asked to
        if not keepTmp:
            if not skipWarmup:
                os.remove(get_filename_compile_driver())
                try:
                    os.remove(get_filename_compile_driver() + ".failed_configs")
                except OSError:
                    pass
            for i in range(jobs):
                generated_script = get_filename_profile_driver(M, N, K, i)
                os.remove(generated_script)
                for f in glob.glob(f"results_{i}.*"):
                    os.remove(f)

        # Check correctness if asked to
        if args.compare:
            print("correctness: ", end=" ", flush=True)
            test_correctness(M, N, K, col_a, col_b, dtype_a, dtype_b, dtype_c, init_type, bestConfig, bias_vector,
                             False)
        elif not run_bench:
            print("", flush=True)

        end_local_time = datetime.now()
        if not run_bench:
            print(
                f">>> Elapsed time: {end_local_time - start_local_time} = {compile_time} (compile) + {profile_time} (profile) + {post_time} (post processing)",
                flush=True)

        f_results.close()
        ## End big loop for tuning

    end_time = datetime.now()
    tuning_time = end_time - start_time
    if not run_bench:
        print(f"Tuning ends at: {end_time}")
        print(f"Total tuning time (h:m:s): {tuning_time}")

    if hack_triton:
        print("Triton compiler is hacked, don't forget to git restore the changes :)")


if __name__ == '__main__':
    sys.exit(main())
