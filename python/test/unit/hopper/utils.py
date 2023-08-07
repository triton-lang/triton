import torch


def get_variant_golden(a, b):
    SIZE_M = a.shape[0]
    SIZE_K = a.shape[1]
    SIZE_N = b.shape[1]
    assert a.shape[1] == b.shape[0]
    zero_M_K = torch.zeros((SIZE_M, SIZE_K), dtype=a.dtype).cuda()
    zero_3M_K = torch.zeros((3 * SIZE_M, SIZE_K), dtype=a.dtype).cuda()
    zero_K_N = torch.zeros((SIZE_K, SIZE_N), dtype=b.dtype).cuda()
    zero_3K_N = torch.zeros((3 * SIZE_K, SIZE_N), dtype=b.dtype).cuda()
    a_padded = torch.cat((a, zero_M_K, zero_M_K), 0)
    a_padded = torch.cat((a_padded, zero_3M_K, zero_3M_K), 1)
    b_padded = torch.cat((b, zero_K_N, zero_K_N), 0)
    b_padded = torch.cat((b_padded, zero_3K_N, zero_3K_N), 1)
    c_padded = torch.matmul(a_padded, b_padded)
    return c_padded[:SIZE_M, :SIZE_N]

# It's not easy to get a proper error threshold in different size
# Here the gemm calculation is padded to a different size in order to get
# a variant version of the golden result. And the error between golden and
# golden_variant provide reference on selecting the proper rtol / atol.


def get_proper_err(golden, golden_variant):
    golden_diff = golden - golden_variant
    golden_abs_err = torch.max(torch.abs(golden_diff)).item()
    # avoid problems when golden_rel_err is 'inf'
    abs_golden = torch.abs(golden) + torch.full_like(golden, torch.finfo(golden.dtype).smallest_normal)
    golden_rel_err = torch.max(torch.abs(golden_diff) / abs_golden).item()
    return (golden_abs_err, golden_rel_err)
