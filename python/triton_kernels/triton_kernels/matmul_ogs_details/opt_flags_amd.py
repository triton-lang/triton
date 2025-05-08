import torch
import triton
from triton_kernels.target_info import get_cdna_version


def compute_block_nk(n, block_m, grid_m, num_xcds, lhs_dtype, rhs_dtype, microscaling_ctx):
    lhs_width = lhs_dtype.itemsize
    rhs_width = rhs_dtype.itemsize if microscaling_ctx.weight_scale is None else 0.5

    # block_n:
    n_cu = torch.cuda.get_device_properties(0).multi_processor_count
    if n is not None:
        if n <= 128 and (n & (n - 1)) == 0:
            block_n = n
        else:
            block_n = max(32, min(256, triton.next_power_of_2(grid_m * n * num_xcds // n_cu)))
    elif block_m > 64:
        block_n = 256
    else:
        block_n = 128

    if get_cdna_version() == 4 and block_m == 128:
        block_n = 512

    # block_k needs to match the cacheline size (128B)
    block_k = int(128 // min(lhs_width, rhs_width))

    # TODO: block_k = 128 seems to work better for now.
    #       perhaps due to increased number of k loops to pipeline
    if microscaling_ctx.weight_scale is not None:
        block_k = 128
    return block_n, block_k
