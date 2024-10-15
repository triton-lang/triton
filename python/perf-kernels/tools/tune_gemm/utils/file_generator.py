import os

from .utils import (
    get_filename_compile_driver,
    get_filename_myKernels,
    get_filename_profile_driver,
    get_filename_without_extension,
    name_to_tl_types,
    tl_to_torch_types,
)


def read_config(config):
    block_m = config.get('BLOCK_SIZE_M')
    block_n = config.get('BLOCK_SIZE_N')
    block_k = config.get('BLOCK_SIZE_K')
    group_m = config.get('GROUP_SIZE_M')
    split_k = config.get('SPLIT_K')
    num_warps = config.get('num_warps')
    num_stages = config.get('num_stages')
    waves_per_eu = config.get('waves_per_eu')
    mfma_instr_size = config.get('matrix_instr_nonkdim')
    kpack = config.get('kpack')
    sched_variant = config.get('instruction_sched_variant')
    return block_m, block_n, block_k, group_m, split_k, num_warps, num_stages, waves_per_eu, mfma_instr_size, kpack, sched_variant


def gen_configStr(config):
    block_m, block_n, block_k, group_m, split_k, num_warps, num_stages, waves_per_eu, mfmaInstrSize, kpack, sched_variant = read_config(
        config)

    ## {M}_{N}_{K} is removed since the same kernel can be used for differen gemm sizes
    configStr = f"BM{block_m}_BN{block_n}_BK{block_k}_GM{group_m}_SK{split_k}_nW{num_warps}_nS{num_stages}_EU{waves_per_eu}_kP{kpack}_mfma{mfmaInstrSize}_sched{sched_variant[1:-1].upper()}"

    return configStr


def generate_matmul_kernels(configs):
    """
    Generate kernels based on configs and append them to get_filename_myKernels()

    Use the matmul_kernel template (../matmul_kernel.py) and append config to the
    kernel name. E.g. matmul_kernel_BM256_BN256_BK64_GM1_SK1_nW1_nS0_EU0_kP2_mfma16()
    """

    if len(configs) == 0:
        return

    f_kernel = open(get_filename_myKernels(), 'a')

    # write imports
    import_str = """import triton
import triton.language as tl"""
    f_kernel.write(import_str)

    with open(os.path.dirname(os.path.abspath(__file__)) + "/../matmul_kernel.py") as file:
        matmul_kernel_code = file.read()

    for config in configs:
        configStr = gen_configStr(config)
        # Copy the matmul_kernel with name replaced
        matmul_kernel_config = matmul_kernel_code.replace("matmul_kernel", f"matmul_kernel_{configStr}")
        matmul_kernel_config = matmul_kernel_config.replace("import triton.language as tl", "")
        matmul_kernel_config = matmul_kernel_config.replace("import triton", "")
        f_kernel.write(matmul_kernel_config)

    f_kernel.close()


## construct the configStr and generate the wrapper function matmul_{configStr}()
## If `warmup` is set, the generated kernel will be **compiled**
def gen_kernel_and_configStr_from_config(config, EVEN_K, dtype_a, dtype_b, dtype_c, bias_size, warmup):
    block_m, block_n, block_k, group_m, split_k, num_warps, num_stages, waves_per_eu, mfmaInstrSize, kpack, sched_variant = read_config(
        config)

    configStr = gen_configStr(config)

    use_bias = bias_size > 0

    ## Let's enable xcd-based pid remapping only when split-K is NOT used
    ## Also #xcd is fixed to 8. If we are tuning for MI308, please change it to 4
    num_xcds = 1 if split_k > 1 else 8

    if warmup:
        torch_dtype_a = 'fp16'
        torch_dtype_b = 'fp16'
        torch_dtype_c = 'fp16'
        if dtype_a:
            torch_dtype_a = tl_to_torch_types[name_to_tl_types[dtype_a]]
        if dtype_b:
            torch_dtype_b = tl_to_torch_types[name_to_tl_types[dtype_b]]
        if dtype_c:
            torch_dtype_c = tl_to_torch_types[name_to_tl_types[dtype_c]]

        matmul_def_str = f"""
def matmul_{configStr}(M, N, K, am, ak, bk, bn, cm, cn, biasn):
    grid_mn = triton.cdiv(M, {block_m}) * triton.cdiv(N, {block_n})
    matmul_kernel_{configStr}.warmup(
        {torch_dtype_a}, {torch_dtype_b}, {torch_dtype_c}, {torch_dtype_c},
        M, N, K,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = {block_m},
        BLOCK_SIZE_N = {block_n},
        BLOCK_SIZE_K = {block_k},
        GROUP_SIZE_M = {group_m},
        SPLIT_K = {split_k},
        num_warps = {num_warps},
        num_stages = {num_stages},
        waves_per_eu = {waves_per_eu},
        matrix_instr_nonkdim = {mfmaInstrSize},
        kpack = {kpack},
        BIAS = {use_bias},
        EVEN_K = {EVEN_K},
        GRID_MN = grid_mn,
        NUM_XCDS = {num_xcds},
        instruction_sched_variant = {sched_variant},
        grid=(1,),
    )
    return None

def try_compile_config_{configStr}(M, N, K, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_{configStr}(M, N, K, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): {configStr}: ', e, flush=True)
        return False
"""
    else:
        matmul_def_str = f"""
def matmul_{configStr}(a, b, c, bias, M, N, K, am, ak, bk, bn, cm, cn, biasn):
    grid = triton.cdiv(M, {block_m}) * triton.cdiv(N, {block_n}), {split_k}
    matmul_kernel_{configStr}[grid](
        a, b, c, bias,
        M, N, K,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = {block_m},
        BLOCK_SIZE_N = {block_n},
        BLOCK_SIZE_K = {block_k},
        GROUP_SIZE_M = {group_m},
        SPLIT_K = {split_k},
        num_warps = {num_warps},
        num_stages = {num_stages},
        waves_per_eu = {waves_per_eu},
        matrix_instr_nonkdim = {mfmaInstrSize},
        kpack = {kpack},
        BIAS = {use_bias},
        EVEN_K = {EVEN_K},
        GRID_MN = grid[0],
        NUM_XCDS = {num_xcds},
        instruction_sched_variant = {sched_variant},
    )
    return c
"""
    return configStr, matmul_def_str


def generate_compile_driver(M, N, K, col_a, col_b, dtype_a, dtype_b, dtype_c, init_type, configs, rotating_buffer_size,
                            bias_size):
    """
    Generate a single file that contains all kernels in the tuning space.
    This file is used to **compile** the kernels in parallel
    """

    filename = get_filename_compile_driver()
    f_kernel = open(filename, 'w')

    # write imports
    import_str = f"""import torch
import triton
import triton.language as tl
import argparse
import sys
import multiprocessing
from tune_gemm import gen_rotating_tensors
from {get_filename_without_extension(get_filename_myKernels())} import *
"""

    f_kernel.write(import_str + "\n")

    for config in configs:
        EVEN_K = True if K % config.get('BLOCK_SIZE_K') == 0 else False
        configStr, matmul_def_str = gen_kernel_and_configStr_from_config(config, EVEN_K, dtype_a, dtype_b, dtype_c,
                                                                         bias_size, True)
        # Copy the matmul_kernel with name replaced
        f_kernel.write(matmul_def_str + "\n")

    # write compile_kernels
    # pre string
    stride_a_str = "1, M" if col_a else "M, 1"
    stride_b_str = "1, N" if col_b else "N, 1"
    stride_c_str = "N, 1"
    compile_kernels_pre_str = f"""def compile_kernels(M, N, K, rotating_buffer_size, bias_size, num_threads):
    thread_pool = multiprocessing.Pool(processes=num_threads)

    assert bias_size == M or bias_size == 0

    stride_bias = 1 if bias_size > 0 else 0
    stride_am, stride_ak = {stride_a_str}
    stride_bk, stride_bn = {stride_b_str}
    stride_cm, stride_cn = {stride_c_str}
    task_args = (M, N, K,
                 stride_am, stride_ak,
                 stride_bk, stride_bn,
                 stride_cm, stride_cn, stride_bias)

    results = []
    config_names = []
"""
    f_kernel.write(compile_kernels_pre_str + "\n")

    # warm up call of all matmul functions in parallel
    for config in configs:
        configStr = gen_configStr(config)
        task_str = f"    results += [thread_pool.apply_async(try_compile_config_{configStr}, args=task_args)]\n" + \
                   f"    config_names += ['{configStr}']\n"
        f_kernel.write(task_str)

    threadpool_str = """
    failed_configs = []
    for i in range(len(results)):
        results[i].wait()
        res = results[i].get()
        if not res:
            failed_configs += [config_names[i]]
    thread_pool.close()
    thread_pool.join()
    if failed_configs:
        with open("{filename}.failed_configs", "w") as f:
            for cfg in failed_configs:
                f.write(cfg + "\\n")
""".format(filename=filename)
    f_kernel.write(threadpool_str)

    # def main and call compile_kernels
    def_main_str = f"""
def main():
    parser = argparse.ArgumentParser(
        prog="tune a specific gemm size",
        allow_abbrev=False,)
    parser.add_argument("-n", type=int, default=32, help='number of threads')
    parser.add_argument("-rotating_tensor", type=int, default={rotating_buffer_size}, help='size of rotating buffer (MB), default: {rotating_buffer_size}')
    args = parser.parse_args()
    numThreads = args.n
    rotating_buffer_size = args.rotating_tensor
    """
    compile_kernels_call_str = f'compile_kernels({M}, {N}, {K}, rotating_buffer_size, {bias_size}, numThreads)'

    f_kernel.write(def_main_str)
    f_kernel.write(compile_kernels_call_str + "\n\n")
    f_kernel.write("""if __name__ == '__main__':
   sys.exit(main())""")
    f_kernel.close()

    return filename


def generate_profile_tasks(M, N, K, col_a, col_b, dtype_a, dtype_b, dtype_c, init_type, configs, jobs, iters, run_bench,
                           rotating_buffer_size, bias_size, icache_flush):
    """
    Open {len(jobs)} files
    generated_kernelM-N-K-0.py, generated_kernelM-N-K-1.py, ..., generated_kernelM-N-K-{njobs-1}.py
    and generate
    1. matmul kernels of all configs
    2. wrapper function matmul to invoke all the generated kernels
    3. test_gemm to invoke matmul in a loop of {iters} iterations
    """

    filenames = []
    for i in range(jobs):
        filenames.append(get_filename_profile_driver(M, N, K, i))
    f_kernel = [open(path, 'w') for path in filenames]

    # write imports
    import_str = f"""import torch
import triton
import triton.language as tl
import argparse
import sys
import multiprocessing
from tune_gemm import gen_rotating_tensors
from {get_filename_without_extension(get_filename_myKernels())} import *
"""
    if icache_flush:
        import_str += """
from icache_flush import icache_flush
"""
    for fi in range(jobs):
        f_kernel[fi].write(import_str + "\n")

    idx = 0
    for config in configs:
        file_idx = idx % jobs
        EVEN_K = True if K % config.get('BLOCK_SIZE_K') == 0 else False
        configStr, matmul_def_str = gen_kernel_and_configStr_from_config(config, EVEN_K, dtype_a, dtype_b, dtype_c,
                                                                         bias_size, False)
        # Copy the matmul_kernel with name replaced
        f_kernel[file_idx].write(matmul_def_str + "\n")
        idx += 1

    # write test_gemm
    # pre string
    test_gemm_pre_str = f"""def test_gemm(M, N, K, rotating_buffer_size, bias_size):
    tensors = gen_rotating_tensors(M, N, K, '{dtype_a}', {col_a}, '{dtype_b}', {col_b}, '{dtype_c}',
                                   1, '{init_type}', rotating_buffer_size, bias_size, device='cuda')

    a = tensors['input_a'][0]
    b = tensors['input_b'][0]
    c = tensors['output_c'][0]
    assert bias_size == M or bias_size == 0

    stride_bias = tensors['bias'][0].stride(0) if bias_size > 0 else 0

    try:
        with open("{get_filename_compile_driver()}.failed_configs", "r") as f:
            failed_configs = [cfg.strip() for cfg in f.readlines()]
    except Exception:
        failed_configs = []
"""
    for fi in range(jobs):
        f_kernel[fi].write(test_gemm_pre_str + "\n")

    # call all matmul_xxx functions
    idx = 0
    runs = iters if run_bench else 120
    for config in configs:
        configStr = gen_configStr(config)
        matmul_call_str = f"""
    if '{configStr}' not in failed_configs:
        rotating_num = tensors['rotating_num']
        for i in range({runs}):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0"""
        if icache_flush:
            matmul_call_str += """
            icache_flush()"""
        matmul_call_str += f"""
            d = matmul_{configStr}(a, b, c, bias, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)"""
        f_kernel[idx % jobs].write(matmul_call_str + "\n")
        idx += 1
    # post string
    for fi in range(jobs):
        f_kernel[fi].write("    return d\n")

    # def main and call test_gemm
    def_main_str = f"""
def main():
    parser = argparse.ArgumentParser(
        prog="tune a specific gemm size",
        allow_abbrev=False,)
    parser.add_argument("-n", type=int, default=1, help='number of threads')
    parser.add_argument("-rotating_tensor", type=int, default={rotating_buffer_size}, help='size of rotating buffer (MB), default: {rotating_buffer_size}')
    args = parser.parse_args()
    numThreads = args.n
    rotating_buffer_size = args.rotating_tensor
    """
    test_gemm_call_str = f'test_gemm({M}, {N}, {K}, rotating_buffer_size, {bias_size})'
    for fi in range(jobs):
        f_kernel[fi].write(def_main_str)
        f_kernel[fi].write(test_gemm_call_str + "\n\n")
        f_kernel[fi].write("""if __name__ == '__main__':
   sys.exit(main())""")
        f_kernel[fi].close()
