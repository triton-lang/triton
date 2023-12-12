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
import multiprocessing


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
    waves_per_eu_range = [0,1,2,3,4]

    for block_m in block_mn_range:
        for block_n in block_mn_range:
            for block_k in block_k_range:
                for num_warps in num_warps_range:
                    for group_m in group_m_range:
                        for split_k in split_k_range:
                            for num_stages in num_stage_range:
                                for waves_per_eu in waves_per_eu_range:
                                    configs.append({'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': block_k, 'GROUP_SIZE_M': group_m, 'SPLIT_K': split_k, 'num_warps': num_warps, 'num_stages': num_stages, 'waves_per_eu': waves_per_eu})

    return configs


def prune_configs(M, N, K, configs):
    pruned_configs = []

    if M < 32 or N < 32:
        mfma = 16
    else:
        mfma = 32

    for config in configs:
        BLOCK_SIZE_M = config.get("BLOCK_SIZE_M")
        BLOCK_SIZE_N = config.get("BLOCK_SIZE_N")
        BLOCK_SIZE_K = config.get("BLOCK_SIZE_K")
        if mfma == 4 and BLOCK_SIZE_K < 64:
            continue
        # some layouts could not work properly in case
        # number elemens per thread is less 1
        if BLOCK_SIZE_M * BLOCK_SIZE_N < 64:
            continue
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


def run_bash_command(commandstring, capture=True):
    if capture:
        proc = subprocess.run(commandstring, shell=True, check=True, executable='/bin/bash', stdout = subprocess.PIPE)
        return proc.stdout.splitlines()
    proc = subprocess.run(commandstring, shell=True, check=True, executable='/bin/bash')
    return None


def read_config(config):
    block_m = config.get('BLOCK_SIZE_M')
    block_n = config.get('BLOCK_SIZE_N')
    block_k = config.get('BLOCK_SIZE_K')
    group_m = config.get('GROUP_SIZE_M')
    split_k = config.get('SPLIT_K')
    num_warps = config.get('num_warps')
    num_stages = config.get('num_stages')
    waves_per_eu = config.get('waves_per_eu')
    return block_m, block_n, block_k, group_m, split_k, num_warps, num_stages, waves_per_eu


def gen_kernel_and_configStr_from_config(M, N, K, config):
    block_m, block_n, block_k, group_m, split_k, num_warps, num_stages, waves_per_eu = read_config(config)
    configStr = f"M{M}_N{N}_K{K}_BM{block_m}_BN{block_n}_BK{block_k}_GM{group_m}_SK{split_k}_nW{num_warps}_nS{num_stages}_EU{waves_per_eu}"

    matmul_def_str = f"""
def matmul_{configStr}(a, b, c, M, N, K, am, ak, bk, bn, cm, cn, warmup=False):
    #M, K = a.shape
    #K, N = b.shape
    grid = triton.cdiv(M, {block_m}) * triton.cdiv(N, {block_n}), {split_k}
    print(f'config: matmul_kernel_{configStr}', flush=True)
    if warmup:
        matmul_kernel_{configStr}.warmup(
            torch.float16, torch.float16, torch.float16,
            M, N, K,
            am, ak, bk, bn, cm, cn,
            BLOCK_SIZE_M = {block_m},
            BLOCK_SIZE_N = {block_n},
            BLOCK_SIZE_K = {block_k},
            GROUP_SIZE_M = {group_m},
            SPLIT_K = {split_k},
            num_warps = {num_warps},
            num_stages = {num_stages},
            waves_per_eu = {waves_per_eu},
            grid=(1,)
        )
        return None
    else:
        matmul_kernel_{configStr}[grid](
            a, b, c,
            M, N, K,
            am, ak, bk, bn, cm, cn,
            BLOCK_SIZE_M = {block_m},
            BLOCK_SIZE_N = {block_n},
            BLOCK_SIZE_K = {block_k},
            GROUP_SIZE_M = {group_m},
            SPLIT_K = {split_k},
            num_warps = {num_warps},
            num_stages = {num_stages},
            waves_per_eu = {waves_per_eu}
        )
        return c

def try_config_{configStr}(M, N, K, am, ak, bk, bn, cm, cn, dtype):
    #a = torch.randn((M, K), device='cuda', dtype=dtype)
    #b = torch.randn((K, N), device='cuda', dtype=dtype)
    #c = torch.zeros((M, N), device=a.device, dtype=a.dtype)
    try:
        matmul_{configStr}(None, None, None, M, N, K, am, ak, bk, bn, cm, cn, True)
        return True
    except Exception as e:
        print(f'invalid config(compilation): {configStr}: ', e, flush=True)
        return False
"""
    return configStr, matmul_def_str


def generated_kernel_name(M, N, K, gpu_id):
    return f"generated_kernel{M}-{N}-{K}-{gpu_id}.py"


## Open {len(gpus)} files
## generated_kernelM-N-K-{gpus[0]}.py, generated_kernelM-N-K-{gpus[1]}.py, ..., generated_kernelM-N-K-{gpus[-1]}.py
## and generate
## 1. matmul kernels of all configs
## 2. wrapper function matmul to invoke all the generated kernels
## 3. Another wraper function try_config to invoke matmul function
## 4. test_gemm to invoke
##    4.1 run try_config in parallel
##    4.2 matmul in a loop of 10 iterations
def generate_kernel(M, N, K, configs, gpus):
    filenames = []
    ngpus = len(gpus)
    for gpu_id in gpus:
        filenames.append(generated_kernel_name(M, N, K, gpu_id))
    f_kernel = [open(path, 'w') for path in filenames]

    ### write imports
    import_str = """import torch
import triton
import triton.language as tl
import argparse
import sys
import multiprocessing
"""
    for fi in range(ngpus):
        f_kernel[fi].write(import_str + "\n")

    ### write definitions of matmul_kernel_xxx
    ### and matmul_xxx and try_config
    with open("matmul_kernel.py") as file:
        matmul_kernel_code = file.read()
    idx = 0
    for config in configs:
        file_idx = idx % ngpus
        configStr, matmul_def_str = gen_kernel_and_configStr_from_config(M, N, K, config)
        ## Copy the matmul_kernel with name replaced
        matmul_kernel_config = matmul_kernel_code.replace("matmul_kernel", f"matmul_kernel_{configStr}")
        matmul_kernel_config = matmul_kernel_config.replace("import triton.language as tl", "")
        matmul_kernel_config = matmul_kernel_config.replace("import triton", "")
        f_kernel[file_idx].write(matmul_kernel_config + "\n\n")
        f_kernel[file_idx].write(matmul_def_str + "\n")
        idx += 1

    ### write test_gemm
    # pre string
    test_gemm_pre_str = """def test_gemm(M, N, K, dtype, num_threads):
    thread_pool = multiprocessing.Pool(processes=num_threads)
    a = torch.randn((M, K), device='cuda', dtype=dtype)
    b = torch.randn((K, N), device='cuda', dtype=dtype)
    c = torch.zeros((M, N), device=a.device, dtype=a.dtype)
    task_args = (M, N, K,
                 a.stride(0), a.stride(1),
                 b.stride(0), b.stride(1),
                 c.stride(0), c.stride(1), dtype)

    if num_threads > 1:
        results = []
        config_names = []
"""
    for fi in range(ngpus):
        f_kernel[fi].write(test_gemm_pre_str + "\n")

    # warm up call of all matmul functions in parallel
    idx = 0
    for config in configs:
        configStr, _ = gen_kernel_and_configStr_from_config(M, N, K, config)
        task_str = f"        results += [thread_pool.apply_async(try_config_{configStr}, args=task_args)]\n" + \
                   f"        config_names += ['{configStr}']\n"
        f_kernel[idx % ngpus].write(task_str)
        idx += 1

    for fi in range(ngpus):
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
        """.format(filename = filenames[fi])
        f_kernel[fi].write(threadpool_str)
    # call all matmul_xxx functions
    idx = 0
    for config in configs:
        configStr, _ = gen_kernel_and_configStr_from_config(M, N, K, config)
        matmul_call_str = f"""
        if '{configStr}' not in failed_configs:
            for i in range(10):
                d = matmul_{configStr}(a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))"""
        f_kernel[idx % ngpus].write(matmul_call_str + "\n")
        idx += 1
    # post string
    for fi in range(ngpus):
        f_kernel[fi].write("        return d\n")

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
    for fi in range(ngpus):
       f_kernel[fi].write(def_main_str)
       f_kernel[fi].write(test_gemm_call_str + "\n\n")
       f_kernel[fi].write("""if __name__ == '__main__':
   sys.exit(main())""")
       f_kernel[fi].close()

def extract_kernel_time(M, N, K, config, gpuid):
    configStr, _ = gen_kernel_and_configStr_from_config(M, N, K, config)
    parse_result_cmd = f'sed -n \'/matmul_kernel_{configStr}/p\' results-{gpuid}.csv | awk -F \',\' \'{{print $NF}}\' | tail -n1'
    parsed_outputs = run_bash_command(parse_result_cmd)
    return config, parsed_outputs


def profile_batch_kernels(M, N, K, gpuid, verbose):
    os.environ['ROCR_VISIBLE_DEVICES'] = str(gpuid)
    run_bash_command(f"rocprof --stats -o results-{gpuid}.csv python {generated_kernel_name(M, N, K, gpuid)}", capture=(verbose < 2))


def tune_gemm_config(M, N, K, configs, verbose=0, num_threads=16, gpus = [0]):
    ## Generate kernel out of all configs
    generate_kernel(M, N, K, configs, gpus)

    ## remove any compiled kernel in the cache
    run_bash_command("rm -rf ~/.triton/cache")

    ## precompile the kernels in parallel
    start_time = datetime.now()
    for gpu_id in gpus:
        run_bash_command(f"python {generated_kernel_name(M, N, K, gpu_id)} -n {num_threads}", capture=(verbose < 2))
    compile_end = datetime.now()
    compile_time = compile_end - start_time
    if verbose:
        print(f"compile time: {compile_time}", flush=True)

    ## profile generated kernels
    running = [multiprocessing.Process(target=profile_batch_kernels, args=(M,N,K,gpu_id,verbose)) for gpu_id in gpus]
    for p in running:
        p.start()
    for p in running:
        p.join()

    profile_end = datetime.now()
    profile_time = profile_end - compile_end
    if verbose:
        print(f"profile time: {profile_time}", flush=True)

    ## post process results.csv to get the best config and minTime
    ## TODO: process the file in parallel
    minTime = 1024 * 1024 * 1024
    thread_pool = multiprocessing.Pool(processes=num_threads)
    tasks = []
    idx = 0
    for config in configs:
        file_idx = idx % len(gpus)
        gpu_id = gpus[file_idx]
        tasks += [thread_pool.apply_async(extract_kernel_time, args=(M, N, K, config, gpu_id))]
        idx += 1
    thread_pool.close()
    thread_pool.join()

    for task in tasks:
        config, parsed_outputs = task.get()
        if parsed_outputs:
            min_us = int(parsed_outputs[0]) / 1000
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


def matmul(a, b, c, block_m, block_n, block_k, group_m, split_k, num_warps, num_stages, waves_per_eu):
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
        waves_per_eu = waves_per_eu,
    )
    return c


def test_correctness(M, N, K, config, verbose, datatype = torch.float16):
    block_m, block_n, block_k, group_m, split_k, num_warps, num_stages, waves_per_eu = read_config(config)

    torch.manual_seed(0)
    a = torch.randn((M, K), device='cuda', dtype=datatype)
    b = torch.randn((K, N), device='cuda', dtype=datatype)
    # Allocates output.
    c = torch.zeros((M, N), device=a.device, dtype=a.dtype)
    triton_output = matmul(a, b, c, block_m, block_n, block_k, group_m, split_k, num_warps, num_stages, waves_per_eu)
    torch_output = torch.matmul(a, b)
    #print(f"triton_output={triton_output}")
    #print(f"torch_output={torch_output}")
    rtol = 0 if torch.version.hip is None else 1e-2
    size_str = ''
    if verbose:
        size_str = f'SIZE M: {M}, N: {N}, K: {K} '
    if torch.allclose(triton_output, torch_output, atol=1e-1, rtol=rtol):
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
    parser.add_argument("--ngpus", type=int, default=0, help='number of GPUs used in the profiling step')
    parser.add_argument("--gpu_ids", type=lambda s: [int(id) for id in s.split(',')], default=[], help='list of gpu ids to use for tuning')
    parser.add_argument("--gemm_size_file", type=str, default="", help='yaml file to indicate matrix size')
    parser.add_argument("--tuning_results_file", type=str, default=get_default_tuning_result_filename(), help='yaml file to store tuning results')
    parser.add_argument("--keep", action='store_true', default=False, help='keep generated files')
    parser.add_argument("--compare", action='store_true', default=False, help="Whether check result correctness")
    parser.add_argument("--compare_wo_tuning", action='store_true', default=False, help="Whether check result correctness")
    parser.add_argument("--time_breakdown", action='store_true', default=False, help="Show detailed time breakdown of each step during the tuning")
    parser.add_argument("--verbose", action='store_true', default=False, help="enables time_breakdown and additional logging messages")
    parser.add_argument("--num_threads", type=int, default=16, help="number of threads to use for kernel compilation and post processing")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    matrix_size_file = args.gemm_size_file
    tuning_output_file = args.tuning_results_file
    keepTmp = args.keep
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
    print(f"Tuning starts at: {start_time}", flush=True)

    f_results = open(tuning_output_file, 'w')
    for (M, N, K) in mnks:
        start_local_time = datetime.now()
        ## Obtain a pruned tuning space according to gemm size
        pruned_configs = prune_configs(M, N, K, configs_full)

        size_str = f'SIZE: {M} {N} {K}'
        print(f"{size_str} nConfigs: {len(pruned_configs)}", end=" ", flush=True)

        ## The main tuning funtion for one gemm size
        verbose_level = 0
        if args.time_breakdown:
            verbose_level = 1
        if args.verbose:
            verbose_level = 2
        minTime, bestConfig, compile_time, profile_time, post_time = tune_gemm_config(M, N, K, pruned_configs, num_threads=args.num_threads, gpus = gpus, verbose=verbose_level)

        ## post processing the numbers
        perf_tflops = lambda us: 2 * M * N * K * 1e-12 / (us * 1e-6)
        tri_tflops = perf_tflops(minTime)
        if tri_tflops < 0.0001:
            formatted_tflops = "{:.3e}".format(tri_tflops)
        else:
            formatted_tflops = "{:.2f}".format(tri_tflops)
        print(f'TFLOPS: {formatted_tflops} time(us): {minTime}', end=" ", flush=True)

        bestConfig_compact_str, _ = gen_kernel_and_configStr_from_config(M, N, K, bestConfig)
        print(f'best_config: {bestConfig_compact_str}', end=" ", flush=True)

        ## write best config to tuning_results.yaml
        sizeDict = {'M': M, 'N': N, 'K': K}
        sizeDict.update(bestConfig)
        f_results.write("- " + str(sizeDict) + " ")
        f_results.write(f'# TFLOPS: {formatted_tflops} time(us): {minTime:.2f}\n')

        ## remove generated files if asked to
        if not keepTmp:
            for gpu_id in gpus:
                generated_script = generated_kernel_name(M, N, K, gpu_id)
                os.remove(generated_script)
                os.remove(generated_script + ".failed_configs")
                for f in glob.glob(f"results-{gpu_id}.*"):
                    os.remove(f)

        ## Check correctness if asked to
        if args.compare:
            print("correctness: ", end=" ", flush=True)
            test_correctness(M, N, K, bestConfig, False)
        else:
            print("", flush=True)

        end_local_time = datetime.now()
        print(f">>> Elapsed time: {end_local_time - start_local_time} = {compile_time} (compile) + {profile_time} (profile) + {post_time} (post processing)", flush=True)

    f_results.close()

    end_time = datetime.now()
    tuning_time = end_time - start_time
    print(f"Tuning ends at: {end_time}")
    print(f"Total tuning time (h:m:s): {tuning_time}")


if __name__ == '__main__':
    sys.exit(main())
