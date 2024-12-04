import torch

import triton


class TmaDescKernelParam:
    TMA_DESC_SIZE = 128

    def __init__(self, ptr, dims, block_dims, element_size):
        self.desc = torch.empty(self.TMA_DESC_SIZE, dtype=torch.uint8, device="cpu")
        assert len(dims) == len(block_dims)
        assert 1 <= len(dims) <= 5
        assert self.desc.data_ptr() % 64 == 0

        if len(dims) == 1:
            fill_desc_func = triton.runtime.driver.active.utils.fill_1d_tma_descriptor
        elif len(dims) == 2:
            fill_desc_func = triton.runtime.driver.active.utils.fill_2d_tma_descriptor
        elif len(dims) == 3:
            fill_desc_func = triton.runtime.driver.active.utils.fill_3d_tma_descriptor
        elif len(dims) == 4:
            fill_desc_func = triton.runtime.driver.active.utils.fill_4d_tma_descriptor
        else:
            fill_desc_func = triton.runtime.driver.active.utils.fill_5d_tma_descriptor

        fill_desc_func(ptr, *dims, *block_dims, element_size, self.desc.data_ptr())

    # Return a CUtensorMap* pointer in host memory
    def tma_desc_cpu_ptr(self):
        return self.desc.data_ptr()


def create_1d_tma_descriptor(ptr, dim, block_dim, element_size):
    return TmaDescKernelParam(ptr, [dim], [block_dim], element_size)


def create_2d_tma_descriptor(ptr, dim1, dim0, block_dim1, block_dim0, element_size):
    return TmaDescKernelParam(ptr, [dim1, dim0], [block_dim1, block_dim0], element_size)
