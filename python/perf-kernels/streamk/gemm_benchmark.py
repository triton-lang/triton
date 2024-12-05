import os
import json
import torch
import triton
import numpy as np

from utils.solution_selection import tunedtree, tunedarr, solution_params
from utils.gemm_wrapper import matmul

if torch.cuda.is_available():
    current_device_index = torch.cuda.current_device()
    current_device = torch.cuda.get_device_properties(current_device_index)
    sm = current_device.multi_processor_count
else:
    sm = 304


def selection_test():
    #    max_m, max_n, max_k = 8193, 8193, 8193
    #    A_max = torch.randn(max_m, max_k, device="cuda", dtype=torch.float16)
    #    B_max = torch.randn(max_n, max_k, device="cuda", dtype=torch.float16)
    #    C_max = torch.zeros(max_m, max_n, device="cuda", dtype=torch.float16)
    #    locks_max = torch.zeros((sm,), device="cuda", dtype=torch.int32)
    #    P_max = torch.zeros((sm, 256*256), device="cuda", dtype=torch.float32)
    #    bias_max = torch.zeros((max_m,), device="cuda", dtype=torch.float16)

    # Temporary tensors with the maximum size
    #    A = torch.empty(max_m, max_k, device="cuda", dtype=torch.float16)
    #    B = torch.empty(max_n, max_k, device="cuda", dtype=torch.float16)
    #    output = torch.empty(max_m, max_n, device="cuda", dtype=torch.float16)
    #    locks = torch.empty((sm,), device="cuda", dtype=torch.int32)
    #    P = torch.empty((sm, 256*256), device="cuda", dtype=torch.float32)
    #    bias = torch.empty((max_m,), device="cuda", dtype=torch.float16)

    # Remove existing benchmark file if it exists
    if os.path.exists('benchmark.json'):
        os.remove('benchmark.json')

    with open("benchmark.json", "a") as f:
        for m in range(128, 8193, 250):
            for n in range(128, 8193, 250):
                for k in range(128, 8193, 250):
                    print(m, n, k)
                    # To do: need find a way to reduce allocation for A and B, the commented out code
                    # has segfault issue atm.
                    # Point A and B to the appropriate slices of A_max and B_max
                    # A.set_(A_max.storage(), 0, (m, k))
                    # B.set_(B_max.storage(), 0, (n, k)).T
                    A = torch.randn(m, k, device="cuda", dtype=torch.float16)
                    B = torch.randn(n, k, device="cuda", dtype=torch.float16).T

                    expected = A @ B
                    pytorch_ms = triton.testing.do_bench(lambda: A @ B)

                    dist, treeidx = tunedtree.query(np.array([m, n, k]).reshape(1, -1))
                    print(f"{dist}")
                    mt = solution_params[tunedarr[treeidx[0][0]]]
                    print(f"{mt}")
                    BLK_M = mt['BLOCK_SIZE_M']
                    BLK_N = mt['BLOCK_SIZE_N']
                    BLK_K = mt['BLOCK_SIZE_K']
                    gsize_m = mt['GROUP_SIZE_M']
                    two_tiles = 'True'
                    num_stages = mt['num_stages']
                    num_warps = mt['num_warps']
                    waves_per_eu = mt['waves_per_eu']
                    mfmaInstrSize = mt['matrix_instr_nonkdim']
                    kpack = mt['kpack']
                    total_blocks_M = triton.cdiv(m, BLK_M)
                    total_blocks_N = triton.cdiv(n, BLK_N)
                    total_tiles = total_blocks_M * total_blocks_N

                    #   output.set_(C_max.storage(), 0, (m, n))
                    #   locks.set_(locks_max.storage(), 0, (sm,))
                    #   P.set_(P_max.storage(), 0, (sm, BLK_M*BLK_N))
                    #   bias.set_(bias_max.storage(), 0, (m,))
                    output = torch.zeros(m, n, device="cuda", dtype=torch.float16)
                    locks = torch.zeros((sm, ), device="cuda", dtype=torch.int32)
                    P = torch.zeros((sm, BLK_M * BLK_N), device="cuda", dtype=torch.float32)
                    bias = torch.zeros((m, ), device="cuda", dtype=torch.float16)
                    triton_ms = triton.testing.do_bench(
                        lambda: matmul.apply(A, B, output, bias, P, locks, sm, BLK_M, BLK_N, BLK_K, gsize_m, two_tiles,
                                             num_stages, num_warps, waves_per_eu, mfmaInstrSize, kpack))

                    max_disc = 0.0
                    # large tolerance to accommodate for large K (rounding due to half precision)
                    assert max_disc <= 5., (
                        f"pb size: {m}x{n}x{k} - max discrepancy: {max_disc} - sm: {sm}, 2 tiles: {two_tiles}\n{output}\n{expected}"
                    )
                    info = {
                        "m": m,
                        "n": n,
                        "k": k,
                        "MT0": BLK_M,
                        "MT1": BLK_N,
                        "DepU": BLK_K,
                        "sm": sm,
                        "GROUP_SIZE_M": gsize_m,
                        "total_tiles": total_tiles,
                        "num_warps": num_warps,
                        "mfmaInstrSize": mfmaInstrSize,
                        "kpack": kpack,
                        "disc": max_disc,
                        "triton_ms": triton_ms,
                        "pytorch_ms": pytorch_ms,
                    }
                    json.dump(info, f)
                    f.write('\n')


selection_test()
