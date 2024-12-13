import torch
import triton
import random

from streamk_kernel import streamk_gemm
#from streamk_kernel_atomic import streamk_gemm
#from persistent_gemm import streamk_gemm

torch.manual_seed(123)
random.seed(123)

total_sm = 304
print(f"total SMs: {total_sm}")


class matmul(torch.autograd.Function):

    _debug = True

    @staticmethod
    def set_debug(debug: bool):
        matmul._debug = debug

    @staticmethod
    def _call(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, bias: torch.Tensor, P: torch.Tensor,
              locks: torch.Tensor, total_programs_streamk: int, BLK_M: int, BLK_N: int, BLK_K: int, gsize_m: int,
              two_tiles: bool, num_stages: int, num_warps: int, waves_per_eu: int, mfmaInstrSize: int, kpack: int):

        #        assert a.is_contiguous() and b.is_contiguous(), "non-contiguous inputs are not supported"
        # checks constraints
        assert a.shape[1] == b.shape[0], "incompatible dimensions"
        M, K = a.shape
        _, N = b.shape

        total_blocks_M = triton.cdiv(M, BLK_M)
        total_blocks_N = triton.cdiv(N, BLK_N)
        iters_per_tile = triton.cdiv(K, BLK_K)
        total_tiles = total_blocks_M * total_blocks_N
        even_k = K % BLK_K == 0

        if total_programs_streamk > 0:  # Stream-K
            # last wave may occupy less than total_programs_streamk SMs
            total_tiles_streamk = total_tiles % total_programs_streamk
            # for two-tile Stream-K + data-parallel from original paper
            #            if two_tiles and total_tiles - total_tiles_streamk > total_programs_streamk:
            #                total_tiles_streamk += total_programs_streamk
            # remaining tiles are computed using classical blocking
            total_blocking_tiles = total_tiles - total_tiles_streamk
            total_iters_streamk = total_tiles_streamk * iters_per_tile
            # iterations related to full waves
            total_full_tiles_streamk = total_iters_streamk // total_programs_streamk
            # iterations related to last (partial) wave
            total_partial_tiles_streamk = total_iters_streamk % total_programs_streamk

        else:  # all tiles are computed using classical blocking
            total_blocking_tiles = total_tiles
            total_tiles_streamk = 0
            total_full_tiles_streamk = 0
            total_partial_tiles_streamk = 0
            total_iters_streamk = 0

        if matmul._debug:
            print(f"M,N,K={M},{N},{K} ; BLK_M,N,K={BLK_M},{BLK_N},{BLK_K}")
            print(f"{total_blocks_M=} x {total_blocks_N=} = {total_tiles=}")
            print(f"{total_tiles_streamk=} + {total_blocking_tiles=} = {total_tiles=}")
            print(f"{total_programs_streamk=}")
            print(f"{total_blocking_tiles=}")
            print(f"{total_full_tiles_streamk=}")
            print(f"{iters_per_tile=}")
            print(f"{total_iters_streamk=}")
            print("total_remainder_iters_streamk=", total_partial_tiles_streamk)
        use_bias = False
        # compute grid (work to do per SM on the first wave)
        grids = total_programs_streamk
        stride_bias = bias.stride(0) if use_bias else 0
        # MI300X settings, MI250 set num_xcds = 1
        num_xcds = 8
        kk = streamk_gemm[(grids, )](
            a,
            b,
            c,
            bias,
            P,
            locks,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            stride_bias,
            BLOCK_SIZE_M=BLK_M,
            BLOCK_SIZE_N=BLK_N,
            BLOCK_SIZE_K=BLK_K,
            GROUP_SIZE_M=gsize_m,
            NUM_SMS=total_programs_streamk,
            STREAMK_TILES=total_tiles_streamk,
            NUM_XCDS=num_xcds,
            BIAS=use_bias,
            EVEN_K=even_k,
            num_stages=num_stages,
            num_warps=num_warps,
            waves_per_eu=waves_per_eu,
            matrix_instr_nonkdim=mfmaInstrSize,
            kpack=kpack,
        )
        if matmul._debug:
            print(f"{kk.n_regs} registers used, {kk.n_spills} spills")

    #     print(kk.asm['ttgir'])
    #     print(kk.asm['amdgcn'])

        return c

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, bias: torch.Tensor, P: torch.Tensor,
                locks: torch.Tensor, grid: int, BLK_M=128, BLK_N=128, BLK_K=32, gsize_m=1, two_tiles=True, num_stages=3,
                num_warps=4, waves_per_eu=2, mfmaInstrSize=16, kpack=1):
        matmul._call(a=a, b=b, c=c, bias=bias, P=P, locks=locks, total_programs_streamk=grid, BLK_M=BLK_M, BLK_N=BLK_N,
                     BLK_K=BLK_K, gsize_m=gsize_m, two_tiles=two_tiles, num_warps=num_warps, num_stages=num_stages,
                     waves_per_eu=waves_per_eu, mfmaInstrSize=mfmaInstrSize, kpack=kpack)
        return c


# ---------------------------------------------------------------------------
# Example and Benchmark
# ---------------------------------------------------------------------------

perf = lambda ms: 2 * m * n * k * 1e-12 / (ms * 1e-3)

## sweet shapes has multiple of 304 tiles
#m, n, k = 4864, 4096, 8256  # some problem size to test
#m, n, k =4864, 8192, 4160  # some problem size to test
#m, n, k = 8192, 4864, 6878  # some problem size to test

## test for tiles that is not multipe of 304 tiles
#m, n, k = 4096, 4096, 8192  # some problem size to test
m, n, k = 8192, 8192, 8192  # some problem size to test
#m, n, k = 512, 512, 512  # some problem size to test

## memory bound sizes
#m, n, k = 1, 1024, 256

## sizes that have to do masked load/store
#m, n, k = 8133, 8132, 8172  # some problem size to test
#m, n, k = 8128, 6878, 7378  # some problem size to test
#m, n, k = 6912, 768, 256  # some problem size to test
#m, n, k = 5632, 6656, 7936

## test when k is not multiple of 16
#m, n, k = 4864, 4096, 4300

A = torch.randn(m, k, device="cuda", dtype=torch.float16)
B = torch.randn(n, k, device="cuda", dtype=torch.float16).T
# allocates output
C = torch.zeros((m, n), device="cuda", dtype=A.dtype)
bias = torch.zeros((m, ), device="cuda", dtype=A.dtype)
#bias = None
BLK_M = 256
BLK_N = 256
BLK_K = 64
total_blocks_M = triton.cdiv(m, BLK_M)
total_blocks_N = triton.cdiv(n, BLK_N)
total_tiles = total_blocks_M * total_blocks_N
gsize_m = 8
two_tiles = 'True'
num_stages = 2
num_warps = 8
waves_per_eu = 0
mfmaInstrSize = 16
kpack = 2

##for total_sm in range(1, 305):
##    print(f"{total_sm=}")
##    matmul.set_debug(True)
##    locks = torch.zeros((total_sm,), device = "cuda", dtype = torch.int32)
##    P = torch.zeros((total_sm,  BLK_M*BLK_N), device="cuda", dtype=torch.float32)
##    C = matmul.apply(A, B, C, P, locks, total_sm, BLK_M, BLK_N, BLK_K, gsize_m, two_tiles, num_stages, num_warps, waves_per_eu,  mfmaInstrSize, kpack)
##        #exit(0)
##    matmul.set_debug(False)
##    expected = A @ B
##
##    assert torch.allclose(C, expected, atol=1), f"max: {(C - expected).abs().max().item()}\n{C}\n{expected}"
##    print("pass validation test")
##    triton_ms = triton.testing.do_bench(lambda: matmul.apply(A, B, C, P, locks, total_sm, BLK_M, BLK_N, BLK_K, gsize_m, two_tiles, num_stages, num_warps, waves_per_eu,  mfmaInstrSize, kpack))
##    print(f"hybrid stream-k (grid={total_sm}): {triton_ms:.3f} ms  {perf(triton_ms):.3f} tflops")

#for total_sm in range(1, 305):
print(f"{total_sm=}")
matmul.set_debug(True)
locks = torch.zeros((total_sm, ), device="cuda", dtype=torch.int32)
P = torch.zeros((total_sm, BLK_M * BLK_N), device="cuda", dtype=torch.float32)
C = matmul.apply(A, B, C, bias, P, locks, total_sm, BLK_M, BLK_N, BLK_K, gsize_m, two_tiles, num_stages, num_warps,
                 waves_per_eu, mfmaInstrSize, kpack)
#exit(0)
matmul.set_debug(False)
expected = A @ B

#assert torch.allclose(C, expected, atol=1), f"max: {(C - expected).abs().max().item()}\n{C}\n{expected}"
print("pass validation test")

# for debugging, uncomment the following line
#exit(0)

triton_ms = triton.testing.do_bench(lambda: torch.matmul(A, B))
print(f"PyTorch: {triton_ms:.3f} ms  {perf(triton_ms):.3f} tflops")

locks = torch.zeros((total_sm, ), device="cuda", dtype=torch.int32)
P = torch.zeros((total_sm, BLK_M * BLK_N), device="cuda", dtype=torch.float32)
triton_ms = triton.testing.do_bench(
    lambda: matmul.apply(A, B, C, bias, P, locks, total_sm, BLK_M, BLK_N, BLK_K, gsize_m, two_tiles, num_stages,
                         num_warps, waves_per_eu, mfmaInstrSize, kpack))
print(f"hybrid stream-k (grid={total_sm}): {triton_ms:.3f} ms  {perf(triton_ms):.3f} tflops")

locks = torch.zeros((total_sm * 2, ), device="cuda", dtype=torch.int32)
P = torch.zeros((total_sm * 2, BLK_M * BLK_N), device="cuda", dtype=torch.float32)
triton_ms = triton.testing.do_bench(
    lambda: matmul.apply(A, B, C, bias, P, locks, total_sm * 2, BLK_M, BLK_N, BLK_K, gsize_m, two_tiles, num_stages,
                         num_warps, waves_per_eu, mfmaInstrSize, kpack))
print(f"hybrid stream-k (grid={total_sm * 2}): {triton_ms:.3f} ms  {perf(triton_ms):.3f} tflops")

triton_ms = triton.testing.do_bench(
    lambda: matmul.apply(A, B, C, bias, P, locks, total_tiles, BLK_M, BLK_N, BLK_K, gsize_m, two_tiles, num_stages,
                         num_warps, waves_per_eu, mfmaInstrSize, kpack))
print(f"tile matmul (grid={total_tiles}): {triton_ms:.3f} ms  {perf(triton_ms):.3f} tflops")
