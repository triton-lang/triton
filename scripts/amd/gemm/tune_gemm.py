import argparse
import sys
import yaml
import os
import glob
import subprocess

import torch
import triton
import triton.language as tl

from matmul_kernel import matmul_kernel

from datetime import datetime


def get_full_tuning_space():
    configs = []

    block_mn_range = [16, 32, 64, 128, 256]
    block_k_range = [16, 32, 64, 128, 256]
    split_k_range = [1, 2, 4, 5, 6, 8, 10, 12, 16, 18, 24]
    num_warps_range = [1, 2, 4, 8]
    group_m_range = [1, 4, 8]
    # For now we see better perf with num_stages=0 for all gemm configs we care
    # But keep this explicit so that we do not forget we may need to set it to
    # other values in the future
    num_stage_range = [1, 0]

    for block_m in block_mn_range:
        for block_n in block_mn_range:
            for block_k in block_k_range:
                for num_warps in num_warps_range:
                    for group_m in group_m_range:
                        for split_k in split_k_range:
                            for num_stages in num_stage_range:
                                configs.append({'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': block_k, 'GROUP_SIZE_M': group_m, 'SPLIT_K': split_k, 'num_warps': num_warps, 'num_stages': num_stages})

    return configs


def prune_configs(M, N, K, configs):
    pruned_configs = []

    ## TODO: improve how we deal with mfma16 vs mfma32
    ## after it becomes a tuning parameter
    mfma_type = os.getenv('MFMA_TYPE')
    if mfma_type == '16':
        mfma = 16
    else:
        mfma = 32

    for config in configs:
        BLOCK_SIZE_M = config.get("BLOCK_SIZE_M")
        BLOCK_SIZE_N = config.get("BLOCK_SIZE_N")
        BLOCK_SIZE_K = config.get("BLOCK_SIZE_K")
        SPLIT_K = config.get("SPLIT_K")
        GROUP_M = config.get("GROUP_SIZE_M")
        if BLOCK_SIZE_M < mfma or BLOCK_SIZE_N < mfma:
            continue
        if M <= mfma and BLOCK_SIZE_M != mfma:
            continue
        if N <= mfma and BLOCK_SIZE_N != mfma:
            continue
        # skip large split_k when not necessary
        if SPLIT_K != 1 and not need_split_k(M, N, K):
            continue
        # skip split_k that leads to EVEN_K = false
        leap = SPLIT_K * BLOCK_SIZE_K
        modv = K % leap
        if modv != 0:
            continue
        # skip large GROUP_M
        if GROUP_M * BLOCK_SIZE_M > M and GROUP_M != 1:
            continue
        ## out of shared memory resource
        LDS = BLOCK_SIZE_K * BLOCK_SIZE_M + BLOCK_SIZE_K * BLOCK_SIZE_N
        if LDS * 2 > 65536:
            continue

        pruned_configs.append(config)

    return pruned_configs


def need_split_k(SIZE_M, SIZE_N, SIZE_K):
    return (SIZE_M < 64 or SIZE_N < 64) and SIZE_K > 1024


def run_bash_command(commandstring):
    proc = subprocess.run(commandstring, shell=True, check=True, executable='/bin/bash', stdout = subprocess.PIPE)
    return proc.stdout.splitlines()


def read_config(config):
    block_m = config.get('BLOCK_SIZE_M')
    block_n = config.get('BLOCK_SIZE_N')
    block_k = config.get('BLOCK_SIZE_K')
    group_m = config.get('GROUP_SIZE_M')
    split_k = config.get('SPLIT_K')
    num_warps = config.get('num_warps')
    num_stages = config.get('num_stages')
    return block_m, block_n, block_k, group_m, split_k, num_warps, num_stages


def gen_kernel_and_configStr_from_config(M, N, K, config):
    block_m, block_n, block_k, group_m, split_k, num_warps, num_stages = read_config(config)
    configStr = f"M{M}_N{N}_K{K}_BM{block_m}_BN{block_n}_BK{block_k}_GM{group_m}_SK{split_k}_nW{num_warps}_nS{num_stages}"

    matmul_def_str = f"""
def matmul_{configStr}(a, b, c):
    M, K = a.shape
    K, N = b.shape
    grid = triton.cdiv(M, {block_m}) * triton.cdiv(N, {block_n}), {split_k}
    print(f'config: matmul_kernel_{configStr}')
    matmul_kernel_{configStr}[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M = {block_m},
        BLOCK_SIZE_N = {block_n},
        BLOCK_SIZE_K = {block_k},
        GROUP_SIZE_M = {group_m},
        SPLIT_K = {split_k},
        num_warps = {num_warps},
        num_stages = {num_stages}
    )
    return c

def try_config_{configStr}(M, N, K, dtype):
    a = torch.randn((M, K), device='cuda', dtype=dtype)
    b = torch.randn((K, N), device='cuda', dtype=dtype)
    c = torch.zeros((M, N), device=a.device, dtype=a.dtype)
    try:
        matmul_{configStr}(a, b, c)
    except Exception:
        print(f'invalid config {configStr}')
"""
    return configStr, matmul_def_str

## Open a file generated_kernelMNK.py and generate
## 1. matmul kernels of all configs
## 2. wrapper function matmul to invoke all the generated kernels
## 3. Another wraper function try_config to invoke matmul function
## 4. test_gemm to invoke
##    4.1 run try_config in parallel
##    4.2 matmul in a loop of 10 iterations
def generate_kernel(M, N, K, configs):
    f_kernel = open(f'generated_kernel{M}{N}{K}.py', 'w')

    ### write imports
    import_str = """import torch
import triton
import triton.language as tl
import argparse
import sys
import multiprocessing
"""
    f_kernel.write(import_str + "\n")

    ### write definitions of matmul_kernel_xxx
    ### and matmul_xxx and try_config
    with open("matmul_kernel.py") as file:
        matmul_kernel_code = file.read();
    for config in configs:
        configStr, matmul_def_str = gen_kernel_and_configStr_from_config(M, N, K, config)
        ## Copy the matmul_kernel with name replaced
        matmul_kernel_config = matmul_kernel_code.replace("matmul_kernel", f"matmul_kernel_{configStr}")
        matmul_kernel_config = matmul_kernel_config.replace("import triton.language as tl", "")
        matmul_kernel_config = matmul_kernel_config.replace("import triton", "")
        f_kernel.write(matmul_kernel_config + "\n\n")
        f_kernel.write(matmul_def_str + "\n")

    ### write test_gemm
    # pre string
    test_gemm_pre_str = """def test_gemm(M, N, K, dtype, num_threads):
    thread_pool = multiprocessing.Pool(processes=num_threads)
    a = torch.randn((M, K), device='cuda', dtype=dtype)
    b = torch.randn((K, N), device='cuda', dtype=dtype)
    c = torch.zeros((M, N), device=a.device, dtype=a.dtype)
    task_args = (M, N, K, dtype)
"""
    f_kernel.write(test_gemm_pre_str + "\n")

    # warm up call of all matmul functions in parallel
    for config in configs:
        configStr, _ = gen_kernel_and_configStr_from_config(M, N, K, config)
        task_str = f"    thread_pool.apply_async(try_config_{configStr}, args=task_args)\n"
        f_kernel.write(task_str)

    # call all matmul_xxx functions
    for config in configs:
        configStr, _ = gen_kernel_and_configStr_from_config(M, N, K, config)
        matmul_call_str = f"""
    for i in range(10):
        d = matmul_{configStr}(a, b, c)"""
        f_kernel.write(matmul_call_str + "\n")
    # post string
    f_kernel.write("    return d\n")

    ### def main and call test_gemm
    def_main_str = """
def main():
    parser = argparse.ArgumentParser(
        prog="tune a specific gemm size",
        allow_abbrev=False,)
    parser.add_argument("-n", type=int, default=1, help='number of threads')
    args = parser.parse_args()
    numThreads = args.n
    """
    test_gemm_call_str = f'test_gemm({M}, {N}, {K}, torch.float16, numThreads)'
    f_kernel.write(def_main_str)
    f_kernel.write(test_gemm_call_str + "\n\n")
    f_kernel.write("""if __name__ == '__main__':
    sys.exit(main())""")
    f_kernel.close()


def tune_gemm_config(M, N, K, configs):
    ## Generate kernel out of all configs
    generate_kernel(M, N, K, configs)

    ## remove any compiled kernel in the cache
    run_bash_command("rm -rf ~/.triton/cache")

    ## precompile the kernels in parallel
    ## TODO: parameterize numThreads at this level
    run_bash_command(f"python generated_kernel{M}{N}{K}.py -n 16")

    ## profile generated kernels
    run_bash_command(f"rocprof --stats python generated_kernel{M}{N}{K}.py")

    ## post process results.csv to get the best config and minTime
    ## TODO: process the file in parallel
    minTime = 1024 * 1024 * 1024
    for config in configs:
        configStr, _ = gen_kernel_and_configStr_from_config(M, N, K, config)
        parse_result_cmd = f'sed -n \'/matmul_kernel_{configStr}/p\' results.csv | awk -F \',\' \'{{print $NF}}\' | tail -n1'
        parsed_outputs = run_bash_command(parse_result_cmd)
        if parsed_outputs:
            min_us = int(parsed_outputs[0]) / 1000
            if min_us < minTime:
                minTime = min_us
                bestConfig = config
        else:
            min_us = -1
            print(f"invalid config: SIZE {M} {N} {K}: {config}")
    return minTime, bestConfig


def matmul(a, b, c, block_m, block_n, block_k, group_m, split_k, num_warps, num_stages):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # 1D launch kernel where each block gets its own program.

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        META['SPLIT_K']
    )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M = block_m,
        BLOCK_SIZE_N = block_n,
        BLOCK_SIZE_K = block_k,
        GROUP_SIZE_M = group_m,
        SPLIT_K = split_k,
        num_warps = num_warps,
        num_stages = num_stages,
    )
    return c


def test_correctness(M, N, K, config, verbose, datatype = torch.float16):
    block_m, block_n, block_k, group_m, split_k, num_warps, num_stages = read_config(config)

    torch.manual_seed(0)
    a = torch.randn((M, K), device='cuda', dtype=datatype)
    b = torch.randn((K, N), device='cuda', dtype=datatype)
    # Allocates output.
    c = torch.zeros((M, N), device=a.device, dtype=a.dtype)
    triton_output = matmul(a, b, c, block_m, block_n, block_k, group_m, split_k, num_warps, num_stages)
    torch_output = torch.matmul(a, b)
    #print(f"triton_output={triton_output}")
    #print(f"torch_output={torch_output}")
    rtol = 0 if torch.version.hip is None else 1e-2
    size_str = ''
    if verbose:
        size_str = f'SIZE M: {M}, N: {N}, K: {K} '
    if torch.allclose(triton_output, torch_output, atol=1e-1, rtol=rtol):
        print(f'{size_str}✅')
    else:
        print(f'{size_str}❌')


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
    parser.add_argument("--gemm_size_file", type=str, default="", help='yaml file to indicate matrix size')
    parser.add_argument("--tuning_results_file", type=str, default=get_default_tuning_result_filename(), help='yaml file to store tuning results')
    parser.add_argument("--keep", action='store_true', default=False, help='keep generated files')
    parser.add_argument("--compare", action='store_true', default=False, help="Whether check result correctness")
    parser.add_argument("--compare_wo_tuning", action='store_true', default=False, help="Whether check result correctness")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    matrix_size_file = args.gemm_size_file
    tuning_output_file = args.tuning_results_file
    keepTmp = args.keep

    mnks = []
    ## TODO: make it more robust to get user input
    if matrix_size_file == "" or not os.path.isfile(matrix_size_file):
        M = args.m
        N = args.n
        K = args.k
        mnks = [(M, N, K)]
    else:
        with open(matrix_size_file) as file:
            matrix_sizes = yaml.safe_load(file)
        for sizes in matrix_sizes:
            M = sizes['M']
            N = sizes['N']
            K = sizes['K']
            mnks.append((M, N, K))

    ## Check correctness from given configs
    if args.compare_wo_tuning:
        for item in matrix_sizes:
            M = item['M']
            N = item['N']
            K = item['K']
            del item['M']
            del item['N']
            del item['K']
            test_correctness(M, N, K, item, True)
        return

    configs_full = get_full_tuning_space()

    start_time = datetime.now()

    f_results = open(tuning_output_file, 'w')
    for (M, N, K) in mnks:
        ## Obtain a pruned tuning space according to gemm size
        pruned_configs = prune_configs(M, N, K, configs_full)

        size_str = f'SIZE: {M} {N} {K}'
        print(f"{size_str} nConfigs: {len(pruned_configs)}", end=" ", flush=True)

        ## The main tuning funtion for one gemm size
        minTime, bestConfig = tune_gemm_config(M, N, K, pruned_configs)

        ## post processing the numbers
        perf_tflops = lambda us: 2 * M * N * K * 1e-12 / (us * 1e-6)
        tri_tflops = perf_tflops(minTime)
        if tri_tflops < 0.0001:
            formatted_tflops = "{:.3e}".format(tri_tflops)
        else:
            formatted_tflops = "{:.2f}".format(tri_tflops)
        print(f'TFLOPS: {formatted_tflops} time(us): {minTime}', end=" ")

        bestConfig_compact_str, _ = gen_kernel_and_configStr_from_config(M, N, K, bestConfig)
        print(f'best_config: {bestConfig_compact_str}', end=" ")

        ## write best config to tuning_results.yaml
        sizeDict = {'M': M, 'N': N, 'K': K}
        sizeDict.update(bestConfig)
        f_results.write("- " + str(sizeDict) + " ")
        f_results.write(f'# TFLOPS: {formatted_tflops} time(us): {minTime:.2f}\n')

        ## remove generated files if asked to
        if not keepTmp:
            os.remove(f"generated_kernel{M}{N}{K}.py")
            for f in glob.glob("results.*"):
                os.remove(f)

        ## Check correctness if asked to
        if args.compare:
            print("correctness: ", end=" ")
            test_correctness(M, N, K, bestConfig, False)
        else:
            print("")

    f_results.close()

    end_time = datetime.now()
    tuning_time = end_time - start_time
    print(f"Tuning time (h:m:s): {tuning_time}")


if __name__ == '__main__':
    sys.exit(main())
