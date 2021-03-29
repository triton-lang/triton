import torch
import triton
import os


class _matmul(torch.autograd.Function):
    src = triton.read(os.path.join(os.path.dirname(__file__), "matmul.c"))

    _DEFAULT_CONFIGS = [
        triton.config(defines={"TM": "128", "TN": "128", "TK": "32", "SPLITK": "1"}, num_warps=4),
        triton.config(defines={'TM': '64', 'TN': '128', 'TK': '32', 'SPLITK': '1'}, num_warps=4),
        triton.config(defines={'TM': '128', 'TN': '64', 'TK': '32', 'SPLITK': '1'}, num_warps=4),
        triton.config(defines={'TM': '64', 'TN': '64', 'TK': '64', 'SPLITK': '1'}, num_warps=4),
        triton.config(defines={'TM': '32', 'TN': '128', 'TK': '64', 'SPLITK': '1'}, num_warps=4),
        triton.config(defines={'TM': '128', 'TN': '32', 'TK': '64', 'SPLITK': '1'}, num_warps=4),
        triton.config(defines={'TM': '64', 'TN': '32', 'TK': '64', 'SPLITK': '1'}, num_warps=2),
        triton.config(defines={'TM': '32', 'TN': '64', 'TK': '64', 'SPLITK': '1'}, num_warps=2),
        triton.config(defines={'TM': '32', 'TN': '128', 'TK': '32', 'SPLITK': '2'}, num_warps=4),
        triton.config(defines={'TM': '32', 'TN': '128', 'TK': '32', 'SPLITK': '2'}, num_warps=4),
        triton.config(defines={'TM': '128', 'TN': '32', 'TK': '32', 'SPLITK': '4'}, num_warps=4),
        triton.config(defines={'TM': '128', 'TN': '32', 'TK': '32', 'SPLITK': '4'}, num_warps=4),
    ]
    _CONFIGS = _DEFAULT_CONFIGS

    @staticmethod
    def largest_pow2_divisor(N):
        if N % 8 == 0:
            return 8
        if N % 4 == 0:
            return 4
        if N % 2 == 0:
            return 2
        return 1

    _locks = dict()
    _kernels = dict()

    @staticmethod
    def _call(a, b):
        dtype = a.dtype
        device = a.device
        # allocate output
        M, K = a.shape
        K, N = b.shape
        c = torch.empty((M, N), dtype=dtype, device=device)
        # handle non-contiguous inputs if necessary
        if a.stride(0) > 1 and a.stride(1) > 1:
            a = a.contiguous()
        if b.stride(0) > 1 and b.stride(1) > 1:
            b = b.contiguous()
        # kernel hash
        is_a_row = a.stride(1) == 1
        is_b_row = b.stride(1) == 1
        lda = a.stride(0) if is_a_row else a.stride(1)
        ldb = b.stride(0) if is_b_row else b.stride(1)
        ldc = c.stride(0)
        lda_pow2_div = _matmul.largest_pow2_divisor(lda)
        ldb_pow2_div = _matmul.largest_pow2_divisor(ldb)
        ldc_pow2_div = _matmul.largest_pow2_divisor(ldc)
        is_tk_div_k = K % 64 == 0
        key = (
            device,
            dtype,
            is_a_row,
            is_b_row,
            lda_pow2_div,
            ldb_pow2_div,
            ldc_pow2_div,
            is_tk_div_k,
        )
        if key not in _matmul._kernels:
            defines = {
                "TYPE": dtype,
                "STRIDE_AM": "lda" if is_a_row else "1",
                "STRIDE_AK": "1" if is_a_row else "lda",
                "STRIDE_BK": "ldb" if is_b_row else "1",
                "STRIDE_BN": "1" if is_b_row else "ldb",
                "LDA_POW2_DIV": lda_pow2_div,
                "LDB_POW2_DIV": ldb_pow2_div,
                "LDC_POW2_DIV": ldc_pow2_div,
                "IS_TK_DIV_K": int(is_tk_div_k),
            }
            _matmul._kernels[key] = triton.kernel(
                _matmul.src,
                device,
                defines=defines,
                autotune_configs=_matmul._CONFIGS,
                autotune_key=["M", "N", "K"],
            )
        kernel = _matmul._kernels[key]
        # # locks for split-k
        if device not in _matmul._locks:
            _matmul._locks[device] = torch.zeros(1024 * 1024, dtype=torch.int32, device=device)
        locks = _matmul._locks[device]
        # enqueue
        alpha = 1.0
        args = [a.data_ptr(), b.data_ptr(), c.data_ptr(), alpha, M, N, K, lda, ldb, ldc, locks.data_ptr()]
        grid = lambda opt: [triton.cdiv(M, opt.TM) * triton.cdiv(N, opt.TN), 1, opt.SPLITK]
        kernel(*args, grid=grid)
        return c

    @staticmethod
    def forward(ctx, a, b):
        c = _matmul._call(a, b)
        return c


matmul = _matmul.apply
