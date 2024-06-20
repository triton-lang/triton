import torch

import triton
import triton.language as tl


@triton.jit
def flush_TMA_cache(desc_ptr):
    tl.inline_asm_elementwise("fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg", "=r, l",
                              [desc_ptr], dtype=tl.int32, is_pure=False, pack=1)


def create_1d_tma_descriptor(ptr, dim, block_dim, element_size):
    TMA_SIZE = 128
    desc = torch.empty(TMA_SIZE, dtype=torch.int8)
    triton.runtime.driver.active.utils.fill_1d_tma_descriptor(ptr, dim, block_dim, element_size, desc.data_ptr())
    gpu_desc = desc.cuda()
    # TMA cache is not being flushed in between dispacthes, therefore we should
    # manually flush the cache every time we create a new TMA descriptor to make
    # sure the following dispatch don't use stale cache when accessing TMA.
    flush_TMA_cache[(1, )](gpu_desc, num_warps=1)
    return gpu_desc


def create_2d_tma_descriptor(ptr, dim1, dim0, block_dim1, block_dim0, element_size):
    TMA_SIZE = 128
    desc = torch.empty(TMA_SIZE, dtype=torch.int8)
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(ptr, dim1, dim0, block_dim1, block_dim0, element_size,
                                                              desc.data_ptr())
    gpu_desc = desc.cuda()
    # TMA cache is not being flushed in between dispacthes, therefore we should
    # manually flush the cache every time we create a new TMA descriptor to make
    # sure the following dispatch don't use stale cache when accessing TMA.
    flush_TMA_cache[(1, )](gpu_desc, num_warps=1)
    return gpu_desc
