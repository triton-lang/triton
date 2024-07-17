import torch

import triton


# Constructs a 1D TMA descriptor in mutable GPU memory.
#
# Note: on the first use of a new descriptor, each SM must invalidate the descriptor's
# address in TMA cache via fence.proxy.tensormap::generic.acquire.gpu.
def create_1d_tma_descriptor(ptr, dim, block_dim, element_size):
    TMA_SIZE = 128
    desc = torch.empty(TMA_SIZE, dtype=torch.int8)
    triton.runtime.driver.active.utils.fill_1d_tma_descriptor(ptr, dim, block_dim, element_size, desc.data_ptr())
    gpu_desc = desc.cuda()
    return gpu_desc


# Constructs a 2D TMA descriptor in mutable GPU memory.
#
# Note: on the first use of a new descriptor, each SM must invalidate the descriptor's
# address in TMA cache via fence.proxy.tensormap::generic.acquire.gpu.
def create_2d_tma_descriptor(ptr, dim1, dim0, block_dim1, block_dim0, element_size):
    TMA_SIZE = 128
    desc = torch.empty(TMA_SIZE, dtype=torch.int8)
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(ptr, dim1, dim0, block_dim1, block_dim0, element_size,
                                                              desc.data_ptr())
    gpu_desc = desc.cuda()
    return gpu_desc
