# fp8
import argparse
import sys
import yaml
import os
import glob
import subprocess

import torch
import triton
import triton.language as tl

from streamk_kernel import streamk_gemm

from datetime import datetime
import multiprocessing
import pandas as pd

device_oi = 650. / 3.0


def get_full_tuning_space():
    configs = []

    block_mn_range = [16, 32, 64, 128, 256]
    block_k_range = [16, 32, 64, 128, 256]
    num_warps_range = [1, 2, 4, 8]
    group_m_range = [1, 4, 8, 16, 32]
    # For now we see better perf with num_stages=0 for all gemm configs we care
    # But keep this explicit so that we do not forget we may need to set it to
    # other values in the future
    num_stage_range = [0]
    waves_per_eu_range = [0]
    matrix_instr_nonkdim_range = [16, 32]
    kpack_range = [1, 2]

    for block_m in block_mn_range:
        for block_n in block_mn_range:
            for block_k in block_k_range:
                for num_warps in num_warps_range:
                    for group_m in group_m_range:
                        for num_stages in num_stage_range:
                            for waves_per_eu in waves_per_eu_range:
                                for matrix_instr_nonkdim in matrix_instr_nonkdim_range:
                                    for kpack in kpack_range:
                                        configs.append({
                                            'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': block_k,
                                            'GROUP_SIZE_M': group_m, 'num_warps': num_warps, 'num_stages': num_stages,
                                            'waves_per_eu': waves_per_eu, 'matrix_instr_nonkdim': matrix_instr_nonkdim,
                                            'kpack': kpack
                                        })

    return configs


def get_gemm_oi(M, N, K):
    FLOPs = 2 * M * N * K
    # 4 for fp32
    # to do check dtype for bytesmoved
    bytesmoved = (M * K + K * N + 2 * M * N) * 4
    return FLOPs / bytesmoved


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
        matrix_instr_nonkdim = config.get("matrix_instr_nonkdim")
        kpack = config.get("kpack")
        if matrix_instr_nonkdim > mfma:
            continue
        if mfma == 4 and BLOCK_SIZE_K < 64:
            continue
        # some layouts could not work properly in case
        # number elemens per thread is less 1
        if BLOCK_SIZE_M * BLOCK_SIZE_N < 64:
            continue
        GROUP_M = config.get("GROUP_SIZE_M")
        if BLOCK_SIZE_M < matrix_instr_nonkdim or BLOCK_SIZE_N < matrix_instr_nonkdim:
            continue
        if BLOCK_SIZE_K == 16 and matrix_instr_nonkdim == 16 and kpack == 2:
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
        # skip large GROUP_M
        if GROUP_M * BLOCK_SIZE_M > M and GROUP_M != 1:
            continue
        # out of shared memory resource
        # TODO (zhanglx): This does not consider the LDS usage in the epilogue
        LDS = BLOCK_SIZE_K * BLOCK_SIZE_M * elemBytes_a + BLOCK_SIZE_K * BLOCK_SIZE_N * elemBytes_b
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

        pruned_configs.append(config)

    return pruned_configs


def run_bash_command_wrapper(commandstring, capture=True):
    try:
        run_bash_command(commandstring, capture)
    except subprocess.CalledProcessError:
        if not capture:
            print(f"running {commandstring} one more time")
        run_bash_command(commandstring, capture)


def run_bash_command(commandstring, capture=True):
    if capture:
        proc = subprocess.run(commandstring, shell=True, check=True, executable='/bin/bash', stdout=subprocess.PIPE)
        return proc.stdout.splitlines()
    proc = subprocess.run(commandstring, shell=True, check=True, executable='/bin/bash')
    return None


def read_config(config):
    block_m = config.get('BLOCK_SIZE_M')
    block_n = config.get('BLOCK_SIZE_N')
    block_k = config.get('BLOCK_SIZE_K')
    group_m = config.get('GROUP_SIZE_M')
    num_warps = config.get('num_warps')
    num_stages = config.get('num_stages')
    waves_per_eu = config.get('waves_per_eu')
    mfma_instr_size = config.get('matrix_instr_nonkdim')
    kpack = config.get('kpack')
    return block_m, block_n, block_k, group_m, num_warps, num_stages, waves_per_eu, mfma_instr_size, kpack


def gen_kernel_and_configStr_from_config(M, N, K, num_cus, EVEN_K, config, dtype_a, dtype_b, dtype_c, dtype_p,
                                         dtype_lock):
    block_m, block_n, block_k, group_m, num_warps, num_stages, waves_per_eu, mfmaInstrSize, kpack = read_config(config)
    torch_dtype_a = 'fp16'
    torch_dtype_b = 'fp16'
    torch_dtype_c = 'fp16'
    torch_dtype_p = 'fp32'
    torch_dtype_lock = 'int32'
    if dtype_a:
        torch_dtype_a = tl_to_torch_types[name_to_tl_types[dtype_a]]
    if dtype_b:
        torch_dtype_b = tl_to_torch_types[name_to_tl_types[dtype_b]]
    if dtype_c:
        torch_dtype_c = tl_to_torch_types[name_to_tl_types[dtype_c]]
    if dtype_p:
        torch_dtype_p = tl_to_torch_types[name_to_tl_types[dtype_p]]
    if dtype_lock:
        torch_dtype_lock = tl_to_torch_types[name_to_tl_types[dtype_lock]]
    configStr = f"M{M}_N{N}_K{K}_BM{block_m}_BN{block_n}_BK{block_k}_GM{group_m}_nW{num_warps}_nS{num_stages}_EU{waves_per_eu}_kP{kpack}_mfma{mfmaInstrSize}"

    matmul_def_str = f"""
def matmul_{configStr}(a, b, c, P, locks, M, N, K, num_cus, am, ak, bk, bn, cm, cn, warmup=False):
    grid = num_cus
    #print(f'config: streamk_gemm_{configStr}', flush=True)
    if warmup:
        streamk_gemm_{configStr}.warmup(
            {torch_dtype_a}, {torch_dtype_b}, {torch_dtype_c}, {torch_dtype_p}, {torch_dtype_lock},
            M, N, K, num_cus,
            am, ak, bk, bn, cm, cn,
            BLOCK_SIZE_M = {block_m},
            BLOCK_SIZE_N = {block_n},
            BLOCK_SIZE_K = {block_k},
            GROUP_SIZE_M = {group_m},
            num_warps = {num_warps},
            num_stages = {num_stages},
            waves_per_eu = {waves_per_eu},
            matrix_instr_nonkdim = {mfmaInstrSize},
            kpack = {kpack},
            EVEN_K = {EVEN_K},
            grid=(1,)
        )
        return None
    else:
        streamk_gemm_{configStr}[grid,](
            a, b, c, P, locks,
            M, N, K, num_cus,
            am, ak, bk, bn, cm, cn,
            BLOCK_SIZE_M = {block_m},
            BLOCK_SIZE_N = {block_n},
            BLOCK_SIZE_K = {block_k},
            GROUP_SIZE_M = {group_m},
            num_warps = {num_warps},
            num_stages = {num_stages},
            waves_per_eu = {waves_per_eu},
            matrix_instr_nonkdim = {mfmaInstrSize},
            kpack = {kpack},
            EVEN_K = {EVEN_K}
        )
        return c

def try_config_{configStr}(M, N, K, num_cus, am, ak, bk, bn, cm, cn):
    try:
        matmul_{configStr}(None, None, None, None, None, M, N, K, num_cus, am, ak, bk, bn, cm, cn, True)
        return True
    except Exception as e:
        print(f'invalid config(compilation): {configStr}: ', e, flush=True)
        return False
"""
    return configStr, matmul_def_str


def generated_kernel_name(M, N, K, gpu_id):
    return f"generated_kernel{M}-{N}-{K}-{gpu_id}.py"


# Open {len(gpus)} files
# generated_kernelM-N-K-{gpus[0]}.py, generated_kernelM-N-K-{gpus[1]}.py, ..., generated_kernelM-N-K-{gpus[-1]}.py
# and generate
# 1. matmul kernels of all configs
# 2. wrapper function matmul to invoke all the generated kernels
# 3. Another wraper function try_config to invoke matmul function
# 4. test_gemm to invoke
# 4.1 run try_config in parallel
# 4.2 matmul in a loop of 10 iterations
def generate_kernel(M, N, K, num_cus, col_a, col_b, dtype_a, dtype_b, dtype_c, dtype_p, dtype_lock, init_type, configs,
                    jobs, iters, run_bench):
    filenames = []
    for i in range(jobs):
        filenames.append(generated_kernel_name(M, N, K, i))
    f_kernel = [open(path, 'w') for path in filenames]

    # write imports
    import_str = """import torch
import triton
import triton.language as tl
import argparse
import sys
import multiprocessing
from tune_streamk import gen_input
"""
    for fi in range(jobs):
        f_kernel[fi].write(import_str + "\n")

    # write definitions of streamk_gemm_xxx
    # and matmul_xxx and try_config
    with open("streamk_kernel.py") as file:
        streamk_gemm_code = file.read()
    idx = 0
    for config in configs:
        file_idx = idx % jobs
        EVEN_K = True if K % config.get('BLOCK_SIZE_K') == 0 else False
        configStr, matmul_def_str = gen_kernel_and_configStr_from_config(M, N, K, num_cus, EVEN_K, config, dtype_a,
                                                                         dtype_b, dtype_c, dtype_p, dtype_lock)
        # Copy the streamk_gemm with name replaced
        streamk_gemm_config = streamk_gemm_code.replace("streamk_gemm", f"streamk_gemm_{configStr}")
        streamk_gemm_config = streamk_gemm_config.replace("import triton.language as tl", "")
        streamk_gemm_config = streamk_gemm_config.replace("import triton", "")
        f_kernel[file_idx].write(streamk_gemm_config + "\n\n")
        f_kernel[file_idx].write(matmul_def_str + "\n")
        idx += 1

    # write test_gemm
    # pre string
    block_m = config.get('BLOCK_SIZE_M')
    block_n = config.get('BLOCK_SIZE_N')
    test_gemm_pre_str = f"""def test_gemm(M, N, K, num_cus, num_threads):
    thread_pool = multiprocessing.Pool(processes=num_threads)
    a, a_fp16 = gen_input(M, K, '{dtype_a}', {col_a}, 1, '{init_type}', device='cuda')
    b, b_fp16 = gen_input(K, N, '{dtype_b}', {col_b}, 2, '{init_type}', device='cuda')
    c = torch.zeros((M, N), device=a.device, dtype={tl_to_torch_types[name_to_tl_types[dtype_c]]})
    task_args = (M, N, K, num_cus,
                 a.stride(0), a.stride(1),
                 b.stride(0), b.stride(1),
                 c.stride(0), c.stride(1))

    if num_threads > 1:
        results = []
        config_names = []
"""
    for fi in range(jobs):
        f_kernel[fi].write(test_gemm_pre_str + "\n")

    # warm up call of all matmul functions in parallel
    idx = 0
    for config in configs:
        EVEN_K = True if K % config.get('BLOCK_SIZE_K') == 0 else False
        configStr, _ = gen_kernel_and_configStr_from_config(M, N, K, num_cus, EVEN_K, config, None, None, None, None,
                                                            None)
        task_str = f"        results += [thread_pool.apply_async(try_config_{configStr}, args=task_args)]\n" + \
                   f"        config_names += ['{configStr}']\n"
        f_kernel[idx % jobs].write(task_str)
        idx += 1

    for fi in range(jobs):
        threadpool_str = """
        failed_configs = []
        for i in range(len(results)):
            results[i].wait()
            res = results[i].get()
            if not res:
                failed_configs += [config_names[i]]
        thread_pool.close()
        thread_pool.join()
        with open("{filename}.failed_configs", "w") as f:
            for cfg in failed_configs:
                f.write(cfg + "\\n")
    else:
        try:
            with open("{filename}.failed_configs", "r") as f:
                failed_configs = [cfg.strip() for cfg in f.readlines()]
        except Exception:
            failed_configs = []
        """.format(filename=filenames[fi])
        f_kernel[fi].write(threadpool_str)
    # call all matmul_xxx functions
    idx = 0
    runs = iters if run_bench else 200
    for config in configs:
        EVEN_K = True if K % config.get('BLOCK_SIZE_K') == 0 else False
        configStr, _ = gen_kernel_and_configStr_from_config(M, N, K, num_cus, EVEN_K, config, None, None, None, None,
                                                            None)
        block_m = config.get('BLOCK_SIZE_M')
        block_n = config.get('BLOCK_SIZE_N')
        matmul_call_str = f"""
        if '{configStr}' not in failed_configs:
            print(f"{configStr}")
            for i in range({runs}):
                locks = torch.zeros((num_cus,), device = "cuda", dtype = torch.int32)
                P = torch.zeros((num_cus,  {block_m}*{block_n}), device="cuda", dtype=torch.float32)
                d = matmul_{configStr}(a, b, c, P, locks, M, N, K, num_cus, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))"""
        f_kernel[idx % jobs].write(matmul_call_str + "\n")
        idx += 1
    # post string
    for fi in range(jobs):
        f_kernel[fi].write("            return d\n")

    # def main and call test_gemm
    def_main_str = """
def main():
    parser = argparse.ArgumentParser(
        prog="tune a specific gemm size",
        allow_abbrev=False,)
    parser.add_argument("-n", type=int, default=1, help='number of threads')
    args = parser.parse_args()
    numThreads = args.n
    num_cus = 304
    """
    test_gemm_call_str = f'test_gemm({M}, {N}, {K}, num_cus, numThreads)'
    for fi in range(jobs):
        f_kernel[fi].write(def_main_str)
        f_kernel[fi].write(test_gemm_call_str + "\n\n")
        f_kernel[fi].write("""if __name__ == '__main__':
   sys.exit(main())""")
        f_kernel[fi].close()


def extract_kernel_time(M, N, K, num_cus, EVEN_K, config, df):
    # Correct the header by removing 'sig' and 'obj' to reduce number from 21 to 19
    # once the bug is fixed, we should not need below two lines
    cols = [
        'Index', 'KernelName', 'gpu-id', 'queue-id', 'queue-index', 'pid', 'tid', 'grd', 'wgr', 'lds', 'scr',
        'arch_vgpr', 'accum_vgpr', 'sgpr', 'wave_size', 'DispatchNs', 'BeginNs', 'EndNs', 'CompleteNs'
    ]
    df.columns = cols

    configStr, _ = gen_kernel_and_configStr_from_config(M, N, K, num_cus, EVEN_K, config, None, None, None, None, None)

    filtered_df = df[df['KernelName'].str.contains(configStr, na=False)].copy()
    filtered_df['DurationNs'] = filtered_df['EndNs'] - filtered_df['BeginNs']
    meanTime = filtered_df['DurationNs'].tail(100).mean()
    return config, meanTime


def profile_batch_kernels(M, N, K, num_cus, gpuid, gpus, jobs, verbose):
    ngpus = len(gpus)
    gpuIdx = gpus.index(gpuid)
    if gpuIdx + 1 > jobs:
        return
    os.environ['ROCR_VISIBLE_DEVICES'] = str(gpuid)
    jobId = gpuIdx
    while jobId < jobs:
        if verbose:
            print(f"profiling {generated_kernel_name(M, N, K, jobId)} on GPU {gpuid}")
        run_bash_command_wrapper(
            f"rocprofv2 --plugin file --plugin-version 1 --kernel-trace -o {jobId} python {generated_kernel_name(M, N, K, jobId)}",
            capture=(verbose < 2))
        jobId += ngpus


def tune_gemm_config(M, N, K, num_cus, col_a, col_b, dtype_a, dtype_b, dtype_c, dtype_p, dtype_lock, init_type, configs,
                     run_bench, jobs, iters, skipWarmup, verbose=0, num_threads=16, gpus=[0]):
    # Generate kernel out of all configs
    generate_kernel(M, N, K, num_cus, col_a, col_b, dtype_a, dtype_b, dtype_c, dtype_p, dtype_lock, init_type, configs,
                    jobs, iters, run_bench)

    # remove any compiled kernel in the cache
    run_bash_command("rm -rf ~/.triton/cache")

    # precompile the kernels in parallel
    start_time = datetime.now()
    if not skipWarmup:
        for i in range(jobs):
            run_bash_command(f"python {generated_kernel_name(M, N, K, i)} -n {num_threads}", capture=(verbose < 2))
    compile_end = datetime.now()
    compile_time = compile_end - start_time
    if verbose:
        print(f"compile time: {compile_time}", flush=True)

    # profile generated kernels
    running = [
        multiprocessing.Process(target=profile_batch_kernels, args=(M, N, K, num_cus, gpu_id, gpus, jobs, verbose))
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
    df_prof = [
        pd.read_csv(f"results_{i}.csv", skiprows=1, header=None, delimiter=',', quotechar='"', escapechar='\\')
        for i in range(jobs)
    ]
    for config in configs:
        EVEN_K = True if K % config.get('BLOCK_SIZE_K') == 0 else False
        file_idx = idx % jobs
        tasks += [
            thread_pool.apply_async(extract_kernel_time, args=(M, N, K, num_cus, EVEN_K, config, df_prof[file_idx]))
        ]
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


def matmul(a, b, c, P, locks, num_cus, block_m, block_n, block_k, group_m, num_warps, num_stages, waves_per_eu,
           mfmaInstrSize, kpack, EVEN_K):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    #assert a.is_contiguous(), "Matrix A must be contiguous"
    #assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # 1D launch kernel where each block gets its own program.

    grid = num_cus

    streamk_gemm[
        grid,
    ](a, b, c, P, locks, M, N, K, num_cus, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
      BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=block_k, GROUP_SIZE_M=group_m, num_warps=num_warps,
      num_stages=num_stages, waves_per_eu=waves_per_eu, matrix_instr_nonkdim=mfmaInstrSize, kpack=kpack, EVEN_K=EVEN_K)
    return c


def test_correctness(M, N, K, num_cus, col_a, col_b, dtype_a, dtype_b, dtype_c, init_type, config, verbose):
    block_m, block_n, block_k, group_m, num_warps, num_stages, waves_per_eu, mfmaInstrSize, kpack = read_config(config)
    torch.manual_seed(0)
    #a = torch.randn((M, K), device='cuda', dtype=datatype)
    #b = torch.randn((K, N), device='cuda', dtype=datatype)
    a, a_fp16 = gen_input(M, K, dtype_a, col_a, 1, init_type, device='cuda')
    b, b_fp16 = gen_input(K, N, dtype_b, col_b, 2, init_type, device='cuda')
    # Allocates output.
    print(f"{block_k}")
    EVEN_K = K % block_k == 0
    c = torch.zeros((M, N), device=a.device, dtype=tl_to_torch_types[name_to_tl_types[dtype_c]])
    locks = torch.zeros((num_cus, ), device="cuda", dtype=torch.int32)
    P = torch.zeros((num_cus, block_m * block_n), device="cuda", dtype=torch.float32)
    triton_output = matmul(a, b, c, P, locks, num_cus, block_m, block_n, block_k, group_m, num_warps, num_stages,
                           waves_per_eu, mfmaInstrSize, kpack, EVEN_K)
    torch_output = torch.matmul(a_fp16, b_fp16)
    # print(f"triton_output={triton_output}")
    # print(f"torch_output={torch_output}")
    rtol = 0 if torch.version.hip is None else 1e-2
    atol = 1e-3
    row_a_str = 'N' if col_a else 'T'
    row_b_str = 'N' if col_b else 'T'
    size_str = ''
    if verbose:
        size_str = f'SIZE M: {M}, N: {N}, K: {K}, trans: {row_a_str}{row_b_str}'
    if torch.allclose(triton_output.to(torch.float16), torch_output, atol=atol, rtol=rtol):
        print(f'{size_str} Correct✅')
    else:
        print(f'{size_str} Incorrect❌')


def get_default_tuning_result_filename():
    git_branch_name = run_bash_command("git rev-parse --abbrev-ref HEAD")
    git_branch_name = git_branch_name[0].decode()
    git_commit_hash = run_bash_command("git rev-parse --short HEAD")
    git_commit_hash = git_commit_hash[0].decode()

    dt_string = datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
    defaultName = f"tuning_results_{git_branch_name}@{git_commit_hash}_{dt_string}.yaml"
    return defaultName


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
    parser.add_argument("--o", type=str, default=get_default_tuning_result_filename(),
                        help='yaml file to store tuning results')
    parser.add_argument("--keep", action='store_true', default=False, help='keep generated files')
    parser.add_argument("--compare", action='store_true', default=False, help="Whether check result correctness")
    parser.add_argument("--compare_wo_tuning", action='store_true', default=False,
                        help="Whether check result correctness")
    parser.add_argument("--benchmark", action='store_true', default=False, help="Benchmark the given config")
    parser.add_argument("--time_breakdown", action='store_true', default=False,
                        help="Show detailed time breakdown of each step during the tuning")
    parser.add_argument("--verbose", action='store_true', default=False,
                        help="enables time_breakdown and additional logging messages")
    parser.add_argument("--num_threads", type=int, default=16,
                        help="number of threads to use for kernel compilation and post processing")
    parser.add_argument("--jobs", type=int, default=1, help="number of generated files")
    parser.add_argument("--iters", type=int, default=1000, help="number of generated files")
    parser.add_argument("--init_type", type=str, default='randn',
                        help="Initialization type for input matrices (default uniform rand [0, 1.0)])")
    parser.add_argument("--no_warmup", action='store_true', default=False, help="Do not call the warmup kernel")
    args = parser.parse_args()

    return args


TORCH_HAS_FP8E5B16 = hasattr(torch, 'float8_e5m2fnuz')
TORCH_HAS_FP8E4B8 = hasattr(torch, 'float8_e4m3fnuz')
tl_to_torch_types = {
    tl.float16: torch.float16,
    tl.bfloat16: torch.bfloat16,
    tl.float32: torch.float32,
    tl.int8: torch.int8,
    tl.int32: torch.int32,
}
if TORCH_HAS_FP8E5B16:
    tl_to_torch_types[tl.float8e5b16] = torch.float8_e5m2fnuz
if TORCH_HAS_FP8E4B8:
    tl_to_torch_types[tl.float8e4b8] = torch.float8_e4m3fnuz

name_to_tl_types = {
    'int8': tl.int8,
    'int32': tl.int32,
    'fp16': tl.float16,
    'fp32': tl.float32,
    'bf16': tl.bfloat16,
    'fp8': tl.float8e4b8,
    'bf8': tl.float8e5b16,
}


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


def main():
    args = parse_args()
    matrix_size_file = args.gemm_size_file
    tuning_output_file = args.o
    keepTmp = args.keep
    run_bench = args.benchmark
    jobs = args.jobs
    iters = args.iters
    skipWarmup = args.no_warmup
    num_cus = 304

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
    dtype_p = 'fp32'
    dtype_lock = 'int32'
    if dtype_a not in name_to_tl_types or dtype_b not in name_to_tl_types or dtype_c not in name_to_tl_types:
        print(f"Unsupported dtype_a {args.dtype_a} or dtype_b {args.dtype_b} or dtype_c {args.dtype_c}")
        print("Supported types: ", list(name_to_tl_types.keys()))
        sys.exit(1)

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
            test_correctness(M, N, K, num_cus, col_a, col_b, dtype_a, dtype_b, dtype_c, init_type, myConfig, True)
        return

    configs_full = get_full_tuning_space()

    start_time = datetime.now()
    if run_bench:
        print(f"Benchmarking gemm with {dtype_a} inputs")
        print("trans     M      N      K    TFLOPS   us")
    else:
        print(f"Tuning {len(mnks)} gemm sizes starts at: {start_time}", flush=True)
        f_results = open(tuning_output_file, 'w')

    for (M, N, K, col_a, col_b, myConfig) in mnks:
        start_local_time = datetime.now()
        # Obtain a pruned tuning space according to gemm size
        # If running benchmark, use the provided config
        pruned_configs = [myConfig] if run_bench else prune_configs(M, N, K, configs_full, type_name_to_bytes(dtype_a),
                                                                    type_name_to_bytes(dtype_b))

        row_a_str = 'N' if col_a else 'T'
        row_b_str = 'N' if col_b else 'T'
        size_str = f'SIZE: {M} {N} {K} {row_a_str}{row_b_str}'
        if not run_bench:
            print(f"{size_str} nConfigs: {len(pruned_configs)}", end=" ", flush=True)
        else:
            print(f"{row_a_str}{row_b_str}    {M:5d}  {N:5d}  {K:5d}    ", end="")

        # The main tuning funtion for one gemm size
        verbose_level = 0
        if args.time_breakdown:
            verbose_level = 1
        if args.verbose:
            verbose_level = 2
        minTime, bestConfig, compile_time, profile_time, post_time = tune_gemm_config(
            M, N, K, num_cus, col_a, col_b, dtype_a, dtype_b, dtype_c, dtype_p, dtype_lock, init_type, pruned_configs,
            run_bench, jobs, iters, skipWarmup, num_threads=args.num_threads, gpus=gpus, verbose=verbose_level)

        EVEN_K = True if K % bestConfig.get('BLOCK_SIZE_K') == 0 else False
        # post processing the numbers
        perf_tflops = lambda us: 2 * M * N * K * 1e-12 / (us * 1e-6)
        tri_tflops = perf_tflops(minTime)
        formatted_tflops = format_output(tri_tflops)
        minTime = format_output(minTime)
        if not run_bench:
            print(f'TFLOPS: {formatted_tflops} time(us): {minTime}', end=" ", flush=True)

        bestConfig_compact_str, _ = gen_kernel_and_configStr_from_config(M, N, K, num_cus, EVEN_K, bestConfig, None,
                                                                         None, None, None, None)
        if not run_bench:
            print(f'best_config: {bestConfig_compact_str}', end=" ", flush=True)

        # write best config to tuning_results.yaml
        if run_bench:
            print(f"{formatted_tflops}     {minTime}")

        sizeDict = {'M': M, 'N': N, 'K': K, 'rowMajorA': row_a_str, 'rowMajorB': row_b_str}
        sizeDict.update(bestConfig)
        if not run_bench:
            f_results.write("- " + str(sizeDict) + " ")
            f_results.write(f'# TFLOPS: {formatted_tflops} time(us): {minTime}\n')

        # remove generated files if asked to
        if not keepTmp:
            for i in range(jobs):
                generated_script = generated_kernel_name(M, N, K, i)
                os.remove(generated_script)
                if not skipWarmup:
                    os.remove(generated_script + ".failed_configs")
                for f in glob.glob(f"results_{i}.*"):
                    os.remove(f)

        # Check correctness if asked to
        if args.compare:
            print("correctness: ", end=" ", flush=True)
            test_correctness(M, N, K, num_cus, col_a, col_b, dtype_a, dtype_b, dtype_c, init_type, bestConfig, False)
        elif not run_bench:
            print("", flush=True)

        end_local_time = datetime.now()
        if not run_bench:
            print(
                f">>> Elapsed time: {end_local_time - start_local_time} = {compile_time} (compile) + {profile_time} (profile) + {post_time} (post processing)",
                flush=True)

    if not run_bench:
        f_results.close()

    end_time = datetime.now()
    tuning_time = end_time - start_time
    if not run_bench:
        print(f"Tuning ends at: {end_time}")
        print(f"Total tuning time (h:m:s): {tuning_time}")


if __name__ == '__main__':
    sys.exit(main())
