import torch
import triton
import time
import triton.language as tl
from perf_model import *

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
        # triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 256, 'BLOCK_K': 64, 'SPLIT_K': 2}, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 256, 'BLOCK_K': 64, 'SPLIT_K': 4}, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 256, 'BLOCK_K': 64, 'SPLIT_K': 8}, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 256, 'BLOCK_K': 64, 'SPLIT_K': 16}, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 2}, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 4}, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 8}, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 16}, num_warps=4, pre_hook=init_to_zero('C')),

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
        # triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 256, 'BLOCK_K': 64, 'SPLIT_K': 2}, num_stages=3, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 256, 'BLOCK_K': 64, 'SPLIT_K': 4}, num_stages=3, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 256, 'BLOCK_K': 64, 'SPLIT_K': 8}, num_stages=3, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 256, 'BLOCK_K': 64, 'SPLIT_K': 16}, num_stages=3, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 2}, num_stages=3, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 4}, num_stages=3, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 8}, num_stages=3, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 16}, num_stages=3, num_warps=4, pre_hook=init_to_zero('C')),

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
        # triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 256, 'BLOCK_K': 64, 'SPLIT_K': 2}, num_stages=4, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 256, 'BLOCK_K': 64, 'SPLIT_K': 4}, num_stages=4, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 256, 'BLOCK_K': 64, 'SPLIT_K': 8}, num_stages=4, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 256, 'BLOCK_K': 64, 'SPLIT_K': 16}, num_stages=4, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 2}, num_stages=4, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 4}, num_stages=4, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 8}, num_stages=4, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 16 , 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 16}, num_stages=4, num_warps=4, pre_hook=init_to_zero('C')),

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
        # triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 256, 'BLOCK_K': 64, 'SPLIT_K': 2}, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 256, 'BLOCK_K': 64, 'SPLIT_K': 4}, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 256, 'BLOCK_K': 64, 'SPLIT_K': 8}, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 256, 'BLOCK_K': 64, 'SPLIT_K': 16}, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 2}, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 4}, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 8}, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 16}, num_warps=4, pre_hook=init_to_zero('C')),

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
        # triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 256, 'BLOCK_K': 64, 'SPLIT_K': 2}, num_stages=3, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 256, 'BLOCK_K': 64, 'SPLIT_K': 4}, num_stages=3, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 256, 'BLOCK_K': 64, 'SPLIT_K': 8}, num_stages=3, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 256, 'BLOCK_K': 64, 'SPLIT_K': 16}, num_stages=3, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 2}, num_stages=3, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 4}, num_stages=3, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 8}, num_stages=3, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 16}, num_stages=3, num_warps=4, pre_hook=init_to_zero('C')),

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
        # triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 256, 'BLOCK_K': 64, 'SPLIT_K': 2}, num_stages=4, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 256, 'BLOCK_K': 64, 'SPLIT_K': 4}, num_stages=4, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 256, 'BLOCK_K': 64, 'SPLIT_K': 8}, num_stages=4, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 256, 'BLOCK_K': 64, 'SPLIT_K': 16}, num_stages=4, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 2}, num_stages=4, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 4}, num_stages=4, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 8}, num_stages=4, num_warps=4, pre_hook=init_to_zero('C')),
        # triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 16}, num_stages=4, num_warps=4, pre_hook=init_to_zero('C')),

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


def matmul(a, b):
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
    launcher = _kernel[grid]
    pgm = launcher(a, b, c, M, N, K, 
                  a.stride(0), a.stride(1), 
                  b.stride(0), b.stride(1), 
                  c.stride(0), c.stride(1), 
                  GROUP_M=8)
    # print(launcher.kernel.config_timing)
    at = launcher.kernel
    return c, pgm, at


torch.manual_seed(0)
# current
mm_shapes = [
  # [16, 1024, 1024], # Triton: 6.88; Torch: 8.07
  # [16, 1024, 2048], # Triton: 8.93; Torch: 9.40
  # [16, 1024, 3072], # Triton: 11.32; Torch: 10.80
  # [16, 1024, 4096], # Triton: 12.81; Torch: 12.36
  # [16, 1024, 6144], # Triton: 15.78; Torch: 15.23
  # [16, 1024, 8192], # Triton: 18.91; Torch: 18.07
  # [16, 3072, 1024], # Triton: 10.08; Torch: 8.70
  # [16, 3072, 2048], # Triton: 16.98; Torch: 13.03
  # [16, 3072, 3072], # Triton: 20.58; Torch: 17.73
  # [16, 3072, 4096], # Triton: 25.43; Torch: 26.39
  # [16, 3072, 6144], # Triton: 35.27; Torch: 35.70
  # [16, 3072, 8192], # Triton: 43.70; Torch: 44.87
  # [16, 2048, 1024], # Triton: 9.23; Torch: 7.80
  # [16, 4096, 1024], # Triton: 11.12; Torch: 10.46
  # [16, 6144, 1024], # Triton: 13.98; Torch: 13.32
  # [16, 8192, 1024], # Triton: 17.45; Torch: 16.93
  # [16, 2048, 3072], # Triton: 15.29; Torch: 15.29
  # [16, 4096, 3072], # Triton: 25.49; Torch: 23.57
  # [16, 6144, 3072], # Triton: 34.20; Torch: 33.10
  # [16, 8192, 3072], # Triton: 43.02; Torch: 42.60

  
  [1, 1024, 4096 ], # Triton: 10.85; Torch: 11.96
                    # >> OpenAI: 0.0197; Triton-atomic: 0.0204; Torch: 0.0209
  [1, 1024, 8192 ], # Triton: 17.70; Torch: 18.59
                   # >> OpenAI: 0.0284; Triton-atomic: 0.0299; Torch: 0.0301
  [1, 1024, 12288], # Triton: 23.51; Torch: 25.28
                    # >> OpenAI: 0.0383; Triton-atomic: 0.0401; Torch: 0.0397
  [1, 4096, 1024 ], # Triton: 10.45; Torch: 10.49
                    # >> OpenAI: 0.0172; Triton-atomic: 0.0211; Torch: 0.0185
  [1, 8192, 1024 ], # Triton: 15.88; Torch: 17.00
                    # >> OpenAI: 0.0255; Triton-atomic: 0.0316; Torch: 0.0275
  [1, 12288, 1024], # Triton: 22.73; Torch: 23.57
                    # >> OpenAI: 0.0352; Triton-atomic: 0.0414; Torch: 0.0366

  [16, 1024, 4096 ],
  [16, 1024, 8192 ],
  [16, 1024, 12288],
  [16, 4096, 1024 ],
  [16, 8192, 1024 ],
  [16, 12288, 1024],

  [32, 1024, 4096 ],
  [32, 1024, 8192 ],
  [32, 1024, 12288],
  [32, 4096, 1024 ],
  [32, 8192, 1024 ],
  [32, 12288, 1024],

  # [8192, 8192, 8192],
]

REPEATS=50
LAYERS=100

triton_res = []
torch_res = []
speedups  = []
for M, N, K in mm_shapes:
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    triton_output, pgm, at = matmul(a, b)
    torch_output = torch.matmul(a, b)
    assert triton.testing.allclose(triton_output, torch_output)

    # print(pgm.get_sass())
    # if hasattr(at, 'configs_timings'):
    #   for config, timing in at.configs_timings.items():
    #     print(config)
    #     print(f'{timing[0]*1000:.1f}us')
    # print('best:')
    # print(at.best_config)

    do_bench = False
    if do_bench:
      triton_ms, _, _ = triton.testing.do_bench(lambda: matmul(a, b))
      torch_ms, _, _ = triton.testing.do_bench(lambda: torch.matmul(a, b))
      print(f'(do_bench) {M}, {N}, {K} >> Triton: {triton_ms*1000:.2f}; Torch: {torch_ms*1000:.2f}')



    if M*N >= 4096*4096:
      LAYERS=10
    as_ = []
    bs_ = []
    cs_graph = [None for _ in range(LAYERS)]
    cs_stream = [None for _ in range(LAYERS)]
    for i in range(LAYERS):
      as_.append(torch.randn((M, K), device='cuda', dtype=torch.float16)*0.1)
      bs_.append(torch.randn((K, N), device='cuda', dtype=torch.float16)*0.1)
    torch_g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(torch_g):
      for i in range(LAYERS):
        torch.matmul(as_[i], bs_[i])
    torch.cuda.synchronize()
    torch_g.replay()

    torch.cuda.synchronize()
    start = time.time_ns()
    for _ in range(REPEATS):
      torch_g.replay()
    torch.cuda.synchronize()
    end = time.time_ns()
    torch_us = (end-start)/REPEATS/LAYERS/1000

    triton_g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(triton_g):
      for i in range(LAYERS):
        cs_graph[i], _, _ = matmul(as_[i], bs_[i])
        # cs_graph[0], _, _ = matmul(as_[0], bs_[0])
    triton_g.replay()

    # to verify results
    cs_graph[i].zero_()

    torch.cuda.synchronize()
    start = time.time_ns()
    for _ in range(REPEATS):
      triton_g.replay()
    torch.cuda.synchronize()
    end = time.time_ns()
    triton_us = (end-start)/REPEATS/LAYERS/1000
    print(f'(cu_graph) {M}, {N}, {K} >> Triton: {triton_us:.2f}; Torch: {torch_us:.2f}')
    # print(f'           Memory:  {2*(M*K+M*N+K*N) / triton_us * 1e-6:.3f} TB/s')
    # print(f'           Compute: {2*M*N*K/triton_us*1e-6:.2f} TFLOPS')

    triton_res.append(triton_us)
    torch_res.append(torch_us)
    speedups.append(torch_us/triton_us)

    #
    # Using stream
    # 
    use_stream = False
    if use_stream:
      torch.cuda.synchronize()
      start = time.time_ns()
      for _ in range(REPEATS):
        for i in range(LAYERS):
          cs_stream[i], _, _ = matmul(as_[i], bs_[i])
      torch.cuda.synchronize()
      end = time.time_ns()
      triton_stream_us = (end-start)/REPEATS/LAYERS/1000

      torch.cuda.synchronize()
      start = time.time_ns()
      for _ in range(REPEATS):
        for i in range(LAYERS):
          torch.matmul(as_[i], bs_[i])
      torch.cuda.synchronize()
      end = time.time_ns()
      torch_stream_us = (end-start)/REPEATS/LAYERS/1000

      for i in range(LAYERS):
        triton.testing.assert_almost_equal(cs_graph[i], cs_stream[i])
        assert cs_graph[i].data_ptr() != cs_stream[i].data_ptr()
      print(f'(cu_stream) {M}, {N}, {K} >> Triton: {triton_stream_us:.2f}; Torch: {torch_stream_us:.2f}')

# # geo_mean = 1.0
# # for speedup in speedups:
# #   geo_mean *= speedup
# # geo_mean**(1/len(speedups))
# print(f'Average speedup: {sum(speedups)/len(speedups):.3f}, '
#       f'Max speedup: {max(speedups):.3f}, '
#       f'Min speedup: {min(speedups):.3f}')