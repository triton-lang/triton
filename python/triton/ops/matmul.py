import torch
import triton.language as tl
import triton
from .matmul_perf_model import *

def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()

@triton.heuristics({
    'EVEN_K': lambda nargs: nargs['K'] % (nargs['BLOCK_K'] * nargs['SPLIT_K']) == 0,
})
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64 , 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64 , 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64 , 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32 , 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64 , 'BLOCK_N': 32 , 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 64 , 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=5, num_warps=2),

        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=6, num_warps=2),

        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=6, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 2}, num_stages=4, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=7, num_warps=4),

        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=5, num_warps=4),

        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 2}, num_stages=5, num_warps=2, pre_hook=init_to_zero('C')),

        # SPILT_K, num_stages = 2
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 64 , 'BLOCK_K': 64, 'SPLIT_K': 2}, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 64 , 'BLOCK_K': 64, 'SPLIT_K': 4}, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 64 , 'BLOCK_K': 64, 'SPLIT_K': 8}, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 64 , 'BLOCK_K': 64, 'SPLIT_K': 16}, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 64 , 'BLOCK_K': 64, 'SPLIT_K': 16}, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 64 , 'BLOCK_K': 64, 'SPLIT_K': 32}, num_warps=2, pre_hook=init_to_zero('C')),

        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 32 , 'BLOCK_K': 64, 'SPLIT_K': 2}, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 32 , 'BLOCK_K': 64, 'SPLIT_K': 4}, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 32 , 'BLOCK_K': 64, 'SPLIT_K': 8}, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 32 , 'BLOCK_K': 64, 'SPLIT_K': 16}, num_warps=2, pre_hook=init_to_zero('C')),

        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 2}, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 8}, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 16}, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 2}, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 8}, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 16}, num_warps=4, pre_hook=init_to_zero('C')),

        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 64 , 'BLOCK_K': 32, 'SPLIT_K': 2}, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 64 , 'BLOCK_K': 32, 'SPLIT_K': 4}, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 64 , 'BLOCK_K': 32, 'SPLIT_K': 8}, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 64 , 'BLOCK_K': 32, 'SPLIT_K': 16}, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 32 , 'BLOCK_K': 32, 'SPLIT_K': 2}, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 32 , 'BLOCK_K': 32, 'SPLIT_K': 4}, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 32 , 'BLOCK_K': 32, 'SPLIT_K': 8}, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 32 , 'BLOCK_K': 32, 'SPLIT_K': 16}, num_warps=2, pre_hook=init_to_zero('C')),

        # # SPLIT_K num_stages=3
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 64 , 'BLOCK_K': 64, 'SPLIT_K': 2}, num_stages=3, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 64 , 'BLOCK_K': 64, 'SPLIT_K': 4}, num_stages=3, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 64 , 'BLOCK_K': 64, 'SPLIT_K': 8}, num_stages=3, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 64 , 'BLOCK_K': 64, 'SPLIT_K': 16}, num_stages=3, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 64 , 'BLOCK_K': 64, 'SPLIT_K': 16}, num_stages=3, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 64 , 'BLOCK_K': 64, 'SPLIT_K': 32}, num_stages=3, num_warps=2, pre_hook=init_to_zero('C')),

        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 32 , 'BLOCK_K': 64, 'SPLIT_K': 2}, num_stages=3, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 32 , 'BLOCK_K': 64, 'SPLIT_K': 4}, num_stages=3, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 32 , 'BLOCK_K': 64, 'SPLIT_K': 8}, num_stages=3, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 32 , 'BLOCK_K': 64, 'SPLIT_K': 16}, num_stages=3, num_warps=2, pre_hook=init_to_zero('C')),

        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 2}, num_stages=3, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_stages=3, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 8}, num_stages=3, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 16}, num_stages=3, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 2}, num_stages=3, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_stages=3, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 8}, num_stages=3, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 16}, num_stages=3, num_warps=4, pre_hook=init_to_zero('C')),

        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 64 , 'BLOCK_K': 32, 'SPLIT_K': 2}, num_stages=3, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 64 , 'BLOCK_K': 32, 'SPLIT_K': 4}, num_stages=3, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 64 , 'BLOCK_K': 32, 'SPLIT_K': 8}, num_stages=3, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 64 , 'BLOCK_K': 32, 'SPLIT_K': 16}, num_stages=3, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 32 , 'BLOCK_K': 32, 'SPLIT_K': 2}, num_stages=3, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 32 , 'BLOCK_K': 32, 'SPLIT_K': 4}, num_stages=3, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 32 , 'BLOCK_K': 32, 'SPLIT_K': 8}, num_stages=3, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 32 , 'BLOCK_K': 32, 'SPLIT_K': 16}, num_stages=3, num_warps=2, pre_hook=init_to_zero('C')),

        # # SPLIT_K num_stages=4
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 64 , 'BLOCK_K': 64, 'SPLIT_K': 2}, num_stages=4, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 64 , 'BLOCK_K': 64, 'SPLIT_K': 4}, num_stages=4, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 64 , 'BLOCK_K': 64, 'SPLIT_K': 8}, num_stages=4, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 64 , 'BLOCK_K': 64, 'SPLIT_K': 16}, num_stages=4, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 64 , 'BLOCK_K': 64, 'SPLIT_K': 16}, num_stages=4, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 64 , 'BLOCK_K': 64, 'SPLIT_K': 32}, num_stages=4, num_warps=2, pre_hook=init_to_zero('C')),

        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 32 , 'BLOCK_K': 64, 'SPLIT_K': 2}, num_stages=4, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 32 , 'BLOCK_K': 64, 'SPLIT_K': 4}, num_stages=4, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 32 , 'BLOCK_K': 64, 'SPLIT_K': 8}, num_stages=4, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 32 , 'BLOCK_K': 64, 'SPLIT_K': 16}, num_stages=4, num_warps=2, pre_hook=init_to_zero('C')),

        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 2}, num_stages=4, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_stages=4, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 8}, num_stages=4, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 16}, num_stages=4, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 2}, num_stages=4, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_stages=4, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 8}, num_stages=4, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 16}, num_stages=4, num_warps=4, pre_hook=init_to_zero('C')),

        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 64 , 'BLOCK_K': 32, 'SPLIT_K': 2}, num_stages=4, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 64 , 'BLOCK_K': 32, 'SPLIT_K': 4}, num_stages=4, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 64 , 'BLOCK_K': 32, 'SPLIT_K': 8}, num_stages=4, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 64 , 'BLOCK_K': 32, 'SPLIT_K': 16}, num_stages=4, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 32 , 'BLOCK_K': 32, 'SPLIT_K': 2}, num_stages=4, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 32 , 'BLOCK_K': 32, 'SPLIT_K': 4}, num_stages=4, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 32 , 'BLOCK_K': 32, 'SPLIT_K': 8}, num_stages=4, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 32 , 'BLOCK_K': 32, 'SPLIT_K': 16}, num_stages=4, num_warps=2, pre_hook=init_to_zero('C')),


        # BLOCK_M = 32
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=6, num_warps=2),

        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=6, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 2}, num_stages=4, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=7, num_warps=4),

        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=5, num_warps=4),

        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 2}, num_stages=5, num_warps=2, pre_hook=init_to_zero('C')),

        # SPILT_K, num_stages = 2
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 64 , 'BLOCK_K': 64, 'SPLIT_K': 2}, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 64 , 'BLOCK_K': 64, 'SPLIT_K': 4}, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 64 , 'BLOCK_K': 64, 'SPLIT_K': 8}, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 64 , 'BLOCK_K': 64, 'SPLIT_K': 16}, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 64 , 'BLOCK_K': 64, 'SPLIT_K': 16}, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 64 , 'BLOCK_K': 64, 'SPLIT_K': 32}, num_warps=2, pre_hook=init_to_zero('C')),

        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 32 , 'BLOCK_K': 64, 'SPLIT_K': 2}, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 32 , 'BLOCK_K': 64, 'SPLIT_K': 4}, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 32 , 'BLOCK_K': 64, 'SPLIT_K': 8}, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 32 , 'BLOCK_K': 64, 'SPLIT_K': 16}, num_warps=2, pre_hook=init_to_zero('C')),

        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 2}, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 8}, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 16}, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 2}, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 8}, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 16}, num_warps=4, pre_hook=init_to_zero('C')),

        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 64 , 'BLOCK_K': 32, 'SPLIT_K': 2}, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 64 , 'BLOCK_K': 32, 'SPLIT_K': 4}, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 64 , 'BLOCK_K': 32, 'SPLIT_K': 8}, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 64 , 'BLOCK_K': 32, 'SPLIT_K': 16}, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 32 , 'BLOCK_K': 32, 'SPLIT_K': 2}, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 32 , 'BLOCK_K': 32, 'SPLIT_K': 4}, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 32 , 'BLOCK_K': 32, 'SPLIT_K': 8}, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 32 , 'BLOCK_K': 32, 'SPLIT_K': 16}, num_warps=2, pre_hook=init_to_zero('C')),

        # # SPLIT_K num_stages=3
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 64 , 'BLOCK_K': 64, 'SPLIT_K': 2}, num_stages=3, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 64 , 'BLOCK_K': 64, 'SPLIT_K': 4}, num_stages=3, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 64 , 'BLOCK_K': 64, 'SPLIT_K': 8}, num_stages=3, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 64 , 'BLOCK_K': 64, 'SPLIT_K': 16}, num_stages=3, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 64 , 'BLOCK_K': 64, 'SPLIT_K': 16}, num_stages=3, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 64 , 'BLOCK_K': 64, 'SPLIT_K': 32}, num_stages=3, num_warps=2, pre_hook=init_to_zero('C')),

        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 32 , 'BLOCK_K': 64, 'SPLIT_K': 2}, num_stages=3, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 32 , 'BLOCK_K': 64, 'SPLIT_K': 4}, num_stages=3, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 32 , 'BLOCK_K': 64, 'SPLIT_K': 8}, num_stages=3, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 32 , 'BLOCK_K': 64, 'SPLIT_K': 16}, num_stages=3, num_warps=2, pre_hook=init_to_zero('C')),

        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 2}, num_stages=3, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_stages=3, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 8}, num_stages=3, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 16}, num_stages=3, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 2}, num_stages=3, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_stages=3, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 8}, num_stages=3, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 16}, num_stages=3, num_warps=4, pre_hook=init_to_zero('C')),

        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 64 , 'BLOCK_K': 32, 'SPLIT_K': 2}, num_stages=3, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 64 , 'BLOCK_K': 32, 'SPLIT_K': 4}, num_stages=3, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 64 , 'BLOCK_K': 32, 'SPLIT_K': 8}, num_stages=3, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 64 , 'BLOCK_K': 32, 'SPLIT_K': 16}, num_stages=3, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 32 , 'BLOCK_K': 32, 'SPLIT_K': 2}, num_stages=3, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 32 , 'BLOCK_K': 32, 'SPLIT_K': 4}, num_stages=3, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 32 , 'BLOCK_K': 32, 'SPLIT_K': 8}, num_stages=3, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 32 , 'BLOCK_K': 32, 'SPLIT_K': 16}, num_stages=3, num_warps=2, pre_hook=init_to_zero('C')),

        # # SPLIT_K num_stages=4
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 64 , 'BLOCK_K': 64, 'SPLIT_K': 2}, num_stages=4, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 64 , 'BLOCK_K': 64, 'SPLIT_K': 4}, num_stages=4, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 64 , 'BLOCK_K': 64, 'SPLIT_K': 8}, num_stages=4, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 64 , 'BLOCK_K': 64, 'SPLIT_K': 16}, num_stages=4, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 64 , 'BLOCK_K': 64, 'SPLIT_K': 16}, num_stages=4, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 64 , 'BLOCK_K': 64, 'SPLIT_K': 32}, num_stages=4, num_warps=2, pre_hook=init_to_zero('C')),

        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 32 , 'BLOCK_K': 64, 'SPLIT_K': 2}, num_stages=4, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 32 , 'BLOCK_K': 64, 'SPLIT_K': 4}, num_stages=4, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 32 , 'BLOCK_K': 64, 'SPLIT_K': 8}, num_stages=4, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 32 , 'BLOCK_K': 64, 'SPLIT_K': 16}, num_stages=4, num_warps=2, pre_hook=init_to_zero('C')),

        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 2}, num_stages=4, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_stages=4, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 8}, num_stages=4, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 16}, num_stages=4, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 2}, num_stages=4, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_stages=4, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 8}, num_stages=4, num_warps=4, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 16}, num_stages=4, num_warps=4, pre_hook=init_to_zero('C')),

        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 64 , 'BLOCK_K': 32, 'SPLIT_K': 2}, num_stages=4, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 64 , 'BLOCK_K': 32, 'SPLIT_K': 4}, num_stages=4, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 64 , 'BLOCK_K': 32, 'SPLIT_K': 8}, num_stages=4, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 64 , 'BLOCK_K': 32, 'SPLIT_K': 16}, num_stages=4, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 32 , 'BLOCK_K': 32, 'SPLIT_K': 2}, num_stages=4, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 32 , 'BLOCK_K': 32, 'SPLIT_K': 4}, num_stages=4, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 32 , 'BLOCK_K': 32, 'SPLIT_K': 8}, num_stages=4, num_warps=2, pre_hook=init_to_zero('C')),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 32 , 'BLOCK_K': 32, 'SPLIT_K': 16}, num_stages=4, num_warps=2, pre_hook=init_to_zero('C')),

    ],
    key=['M', 'N', 'K'],
    prune_configs_by={
      'prune_num_stages_by' : prune_num_stages,
      'perf_model': estimate_matmul_time,
      'top_k': 10
    },
)
@triton.jit
def _kernel(A, B, C, M, N, K, 
            stride_am, stride_ak, 
            stride_bk, stride_bn, 
            stride_cm, stride_cn, 
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr):
    # matrix multiplication
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z*BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(K, 0, -BLOCK_K*SPLIT_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    acc = acc.to(tl.float16)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


class _matmul(torch.autograd.Function):
    kernel = _kernel

    _locks = dict()

    @staticmethod
    def _call(a, b):
        device = a.device
        # handle non-contiguous inputs if necessary
        if a.stride(0) > 1 and a.stride(1) > 1:
            a = a.contiguous()
        if b.stride(0) > 1 and b.stride(1) > 1:
            b = b.contiguous()
        # checks constraints
        assert a.shape[1] == b.shape[0], "incompatible dimensions"
        M, K = a.shape
        _, N = b.shape
        # allocates output
        c = torch.empty((M, N), device=device, dtype=a.dtype)
        # launch kernel
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), META['SPLIT_K'])
        _kernel[grid](a, b, c, M, N, K, 
                      a.stride(0), a.stride(1), 
                      b.stride(0), b.stride(1), 
                      c.stride(0), c.stride(1), 
                      GROUP_M=8)
        return c

    @staticmethod
    def forward(ctx, a, b):
        return _matmul._call(a, b)


matmul = _matmul.apply
