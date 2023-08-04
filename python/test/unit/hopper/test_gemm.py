# Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import itertools
import os
import re

import pytest
import torch
from torch.testing import assert_close

import triton
import triton.language as tl


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


@triton.jit
def matmul_no_scf_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    FLOAT16_OUTPUT: tl.constexpr, USE_TMA_EPILOGUE: tl.constexpr
):
    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                    offsets=(0, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                                    offsets=(0, 0), block_shape=(BLOCK_K, BLOCK_N), order=(0, 1))
    a = tl.load(a_block_ptr)
    b = tl.load(b_block_ptr)

    c = tl.dot(a, b)

    if FLOAT16_OUTPUT:
        c = c.to(tl.float16)

    if USE_TMA_EPILOGUE:
        c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                        offsets=(0, 0), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
        tl.store(c_block_ptr, c)
    else:
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, c)


@pytest.mark.parametrize('M,N,K,NUM_CTAS,NUM_WARPS,TRANS_A,TRANS_B,OUTPUT_TYPE,USE_TMA_EPILOGUE',
                         itertools.chain(
                             *[
                                 [
                                     # numCTAs = 1, no TMA multicast:
                                     [64, 16, 16, 1, 4, False, True, "float16", USE_TMA_EPILOGUE],
                                     [64, 32, 16, 1, 4, False, True, "float16", USE_TMA_EPILOGUE],
                                     [64, 64, 16, 1, 4, False, True, "float16", USE_TMA_EPILOGUE],
                                     [64, 64, 16, 1, 4, False, True, "float32", USE_TMA_EPILOGUE],
                                     [64, 64, 32, 1, 4, False, True, "float32", USE_TMA_EPILOGUE],
                                     [64, 64, 64, 1, 4, False, True, "float32", USE_TMA_EPILOGUE],
                                     [128, 128, 16, 1, 4, False, True, "float16", USE_TMA_EPILOGUE],
                                     [128, 128, 16, 1, 4, False, True, "float32", USE_TMA_EPILOGUE],
                                     # static mask, cluster 4x1
                                     [256, 64, 16, 4, 4, False, True, "float16", USE_TMA_EPILOGUE],
                                     [256, 64, 16, 4, 4, False, True, "float32", USE_TMA_EPILOGUE],
                                     # dynamic mask, cluster 2x2
                                     [128, 128, 16, 4, 4, False, True, "float16", USE_TMA_EPILOGUE],
                                     [128, 128, 16, 4, 4, False, True, "float32", USE_TMA_EPILOGUE],
                                     # small M, N
                                     [16, 16, 16, 1, 4, False, True, "float32", USE_TMA_EPILOGUE],
                                     [16, 32, 16, 1, 4, False, True, "float32", USE_TMA_EPILOGUE],
                                     [32, 16, 16, 1, 4, False, True, "float32", USE_TMA_EPILOGUE],
                                     [32, 32, 16, 1, 4, False, True, "float32", USE_TMA_EPILOGUE],
                                 ] for USE_TMA_EPILOGUE in [True, False]
                             ]))
@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 9, reason="Requires compute capability >= 9")
def test_gemm_no_scf(M, N, K, NUM_CTAS, NUM_WARPS, TRANS_A, TRANS_B, OUTPUT_TYPE, USE_TMA_EPILOGUE):
    if (TRANS_A):
        a = torch.randn((K, M), device='cuda', dtype=torch.float16).T
    else:
        a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    if (TRANS_B):
        b = torch.randn((N, K), device='cuda', dtype=torch.float16).T
    else:
        b = torch.randn((K, N), device='cuda', dtype=torch.float16)

    if OUTPUT_TYPE == "float16":
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    else:
        c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    matmul_no_scf_kernel[(1, 1)](a_ptr=a, b_ptr=b, c_ptr=c,
                                 M=M, N=N, K=K,
                                 stride_am=a.stride(0), stride_ak=a.stride(1),
                                 stride_bk=b.stride(0), stride_bn=b.stride(1),
                                 stride_cm=c.stride(0), stride_cn=c.stride(1),
                                 BLOCK_M=M, BLOCK_N=N, BLOCK_K=K,
                                 num_warps=NUM_WARPS,
                                 num_ctas=NUM_CTAS,
                                 FLOAT16_OUTPUT=(OUTPUT_TYPE == "float16"),
                                 USE_TMA_EPILOGUE=USE_TMA_EPILOGUE)
    a_f32 = a.to(torch.float32)
    b_f32 = b.to(torch.float32)
    golden = torch.matmul(a_f32, b_f32)
    golden_variant = get_variant_golden(a_f32, b_f32)
    golden_abs_err, golden_rel_err = get_proper_err(golden, golden_variant)
    torch.set_printoptions(profile="full")
    assert_close(
        c,
        golden,
        rtol=max(1e-2, 1.1 * golden_rel_err),
        atol=max(1e-3, 1.1 * golden_abs_err),
        check_dtype=False)


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, w_ptr, bias_ptr, z_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_wm, stride_wn,
    stride_zm, stride_zn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    out_dtype: tl.constexpr, USE_TMA_STORE: tl.constexpr,
    ADD_MATRIX: tl.constexpr, ADD_ROWS: tl.constexpr, ADD_COLS: tl.constexpr,
    DO_SOFTMAX: tl.constexpr, CHAIN_DOT: tl.constexpr,
    A_ORDER_0: tl.constexpr, A_ORDER_1: tl.constexpr,
    B_ORDER_0: tl.constexpr, B_ORDER_1: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    block_offset_m = pid_m * BLOCK_M
    block_offset_n = pid_n * BLOCK_N

    a_tile_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                   offsets=(block_offset_m, 0), block_shape=(BLOCK_M, BLOCK_K), order=(A_ORDER_0, A_ORDER_1))
    b_tile_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                                   offsets=(0, block_offset_n), block_shape=(BLOCK_K, BLOCK_N), order=(B_ORDER_0, B_ORDER_1))
    w_tile_ptr = tl.make_block_ptr(base=w_ptr, shape=(N, N), strides=(stride_wm, stride_wn),
                                   offsets=(0, block_offset_n), block_shape=(BLOCK_N, BLOCK_N), order=(0, 1))
    z = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_m = block_offset_m + tl.arange(0, BLOCK_M)
    offs_n = block_offset_n + tl.arange(0, BLOCK_N)
    z_ptrs = z_ptr + offs_m[:, None] * stride_zm + offs_n[None, :] * stride_zn
    bias_ptrs = bias_ptr + offs_m[:, None] * stride_zm + offs_n[None, :] * stride_zn
    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_tile_ptr, boundary_check=(0, 1))
        b = tl.load(b_tile_ptr, boundary_check=(0, 1))
        z += tl.dot(a, b)
        a_tile_ptr = tl.advance(a_tile_ptr, [0, BLOCK_K])
        b_tile_ptr = tl.advance(b_tile_ptr, [BLOCK_K, 0])

    z = z.to(out_dtype)

    if ADD_MATRIX:
        z += tl.load(bias_ptrs, mask=mask)
    if ADD_ROWS:
        ZRs = bias_ptr + offs_m * stride_zm
        z += tl.load(ZRs)[:, None]
    if ADD_COLS:
        ZCs = bias_ptr + offs_n * stride_zn
        z += tl.load(ZCs)[None, :]
    if DO_SOFTMAX:
        max = tl.max(z, 1)
        z = z - max[:, None]
        num = tl.exp(z.to(tl.float32)).to(max.dtype)
        den = tl.sum(num, 1)
        z = num / den[:, None]
    if CHAIN_DOT:
        w = tl.load(w_tile_ptr)
        z = tl.dot(z.to(w.dtype), w)
        z = z.to(out_dtype)

    if USE_TMA_STORE:
        z_block_ptr = tl.make_block_ptr(base=z_ptr, shape=(M, N), strides=(stride_zm, stride_zn),
                                        offsets=(block_offset_m, block_offset_n), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
        tl.store(z_block_ptr, z, boundary_check=(0, 1))
    else:
        tl.store(z_ptrs, z, mask=mask)


@pytest.mark.parametrize('BLOCK_M,BLOCK_N,BLOCK_K,NUM_WARPS,NUM_CTAS,M,N,K,TRANS_A,TRANS_B,epilogue,out_dtype,USE_TMA_STORE,NUM_STAGES',
                         [(128, 128, 64, 4, 1, *shape_w_c, 'none', out_dtype, use_tma_store, 3)
                          for shape_w_c in [
                             # bad from cublas-important-layers
                             [4096, 1, 1024, False, False],
                             [2048, 204, 1000, True, False],
                         ]
                             for out_dtype in ['float16', 'float32']
                             for use_tma_store in [False, True]
                         ] + [(*shape_w_c, trans_a, trans_b, epilogue, out_dtype, use_tma_store, num_stages)
                              # softmax works for one CTA
                              for shape_w_c in [
                             [64, 64, 16, 4, 1, 64, 64, 64],
                             [128, 128, 64, 4, 1, None, None, None],
                             [16, 16, 64, 4, 1, 16, 16, 64],
                             [64, 64, 32, 8, 1, 64, 64, 64],
                             [128, 128, 64, 4, 1, 128, 128, 128],
                         ]
                             for epilogue in ['softmax']
                             for out_dtype in ['float16', 'float32']
                             for use_tma_store in [False, True]
                             for trans_a in [False, True]
                             for trans_b in [False, True]
                             for num_stages in [3]
                         ] + [(*shape_w_c, trans_a, trans_b, epilogue, out_dtype, use_tma_store, num_stages)
                              for shape_w_c in [
                             [64, 64, 16, 4, 1, 128, 128, 64],
                             *[[256, 64, 16, num_warps, num_ctas, 256, 256, 64] for num_warps in [4, 8] for num_ctas in [1, 2, 4]],
                             # for chain-dot
                             [128, 128, 64, 4, 1, None, None, None],
                             [64, 64, 16, 4, 1, None, None, None],
                             # small BLOCK_M and BLOCK_K
                             [16, 16, 64, 4, 1, 128, 128, 64],
                             *[[16, 32, 64, num_warps, num_ctas, 256, 256, 256] for num_warps in [4, 8] for num_ctas in [1, 2]],
                             # repeat
                             [64, 64, 32, 8, 1, 128, 256, 64],
                             [64, 64, 16, 8, 2, 128, 128, 64],
                             # irregular shape
                             [128, 128, 64, 4, 1, 500, 200, 128],
                             [128, 128, 64, 4, 2, 513, 193, 192],
                         ]
                             for epilogue in ['none', 'add-matrix', 'add-rows', 'add-cols', 'chain-dot']
                             for out_dtype in ['float16', 'float32']
                             for use_tma_store in [False, True]
                             for trans_a in [False, True]
                             for trans_b in [False, True]
                             for num_stages in [3]
                             if not (epilogue == 'chain-dot' and (shape_w_c[5] is not None or shape_w_c[0] != shape_w_c[1]))
                         ] + [(*shape_w_c, trans_a, trans_b, 'none', out_dtype, use_tma_store, num_stages)
                              for shape_w_c in [
                             [64, 64, 32, 4, 1, 128, 256, 64],
                             [128, 128, 16, 4, 4, 512, 256, 64],
                             [128, 256, 32, 4, 8, 256, 256, 192],
                             [512, 256, 32, 4, 8, 1024, 256, 192],
                             # BLOCK_K >= 128
                             [64, 128, 128, 4, 1, 512, 256, 256],
                             [128, 128, 128, 4, 1, 256, 256, 192],
                             [128, 128, 128, 4, 2, 256, 256, 192],
                             # small BLOCK_M and BLOCK_K
                             [16, 32, 32, 4, 1, 128, 256, 64],
                             [32, 32, 16, 4, 1, 256, 256, 192],
                             [16, 32, 64, 4, 4, 512, 256, 64],
                         ]
                             for out_dtype in ['float16', 'float32']
                             for use_tma_store in [False, True]
                             for trans_a in [False, True]
                             for trans_b in [False, True]
                             for num_stages in [3]
                         ] + [(64, n, 16, 4, 1, 512, 256, 256, False, True, 'none', out_dtype, use_tma_store, num_stages)
                              # loop over instr shapes
                              for n in [16, 32, 64, 128, 256]
                              for out_dtype in ['float16', 'float32']
                              for use_tma_store in [False, True]
                              for num_stages in [2, 4, 5, 7]
                              ] + [(*shape_w_c, *shape, False, True, 'none', out_dtype, use_tma_store, num_stages)
                                   # irregular shapes
                                   for shape_w_c in [
                                       [128, 128, 64, 4, 1],
                                       [256, 128, 64, 4, 2],
                                       [128, 128, 128, 4, 2],
                              ]
                             for shape in list(itertools.product([*range(512, 4096, 360)], [*range(512, 4096, 360)], [512, 1024]))
                             for out_dtype in ['float16', 'float32']
                             for use_tma_store in [False, True]
                             for num_stages in [2, 3, 4]
                         ])
@pytest.mark.skipif(torch.cuda.get_device_capability()
                    [0] < 9, reason="Requires compute capability >= 9")
def test_gemm(BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS, NUM_CTAS, M, N, K, TRANS_A, TRANS_B, epilogue, out_dtype, USE_TMA_STORE, NUM_STAGES):
    # with ENABLE_TMA=0 and ENABLE_MMA_V3=0
    if '-'.join(map(str, [BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS, NUM_CTAS, M, N, K])) in [
        '16-32-64-4-4-512-256-64',
    ]:
        pytest.skip('shapePerCTA[1] < 16 not supported')

    # with ENABLE_TMA=0 and ENABLE_MMA_V3=0
    if '-'.join(map(str, [BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS, NUM_CTAS, M, N, K, TRANS_B])) in [
        '16-32-64-4-1-256-256-256-False',
        '16-32-64-4-2-256-256-256-False',
        '16-32-64-4-2-256-256-256-True',
        '16-32-64-8-2-256-256-256-False',
        '16-32-64-8-2-256-256-256-True',
    ]:
        pytest.skip('illegal memory access.')

    # with ENABLE_TMA=1 and ENABLE_MMA_V3=1
    if '-'.join(map(str, [BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS, NUM_CTAS, M, N, K])) in [
        '64-64-32-8-1-128-256-64',
    ]:
        pytest.skip('Tensor-likes are not close!')

    # with ENABLE_TMA=1 and ENABLE_MMA_V3=1
    if NUM_CTAS > 1:
        pytest.skip('Segmentation fault')

    M = BLOCK_M if M is None else M
    N = BLOCK_N if N is None else N
    K = BLOCK_K if K is None else K

    if (TRANS_A):
        a = torch.randn((K, M), device='cuda', dtype=torch.float16).T
        a_order = [0, 1]
    else:
        a = torch.randn((M, K), device='cuda', dtype=torch.float16)
        a_order = [1, 0]

    if (TRANS_B):
        b = torch.randn((N, K), device='cuda', dtype=torch.float16).T
        b_order = [0, 1]
    else:
        b = torch.randn((K, N), device='cuda', dtype=torch.float16)
        b_order = [1, 0]

    if out_dtype == 'float16' and epilogue != 'softmax':
        # TODO: for out_dtype == 'float16' and epilogue == 'softmax', it will
        # fail with the following error: 'llvm.fmul' op requires the same type
        # for all operands and results
        out_dtype = tl.float16
        torch_out_dtype = torch.float16
    else:
        out_dtype = tl.float32
        torch_out_dtype = torch.float32

    # avoid out of memory
    if epilogue in ['add-matrix', 'add-rows', 'add-cols']:
        bias = torch.randn((M, N), device='cuda', dtype=torch_out_dtype)
    else:
        bias = torch.randn((1, 1), device='cuda', dtype=torch_out_dtype)

    if epilogue == 'chain-dot':
        w = torch.randn((N, N), device='cuda', dtype=torch.float16).T
    else:
        w = torch.randn((1, 1), device='cuda', dtype=torch.float16).T

    z = torch.full((M, N), 1., device='cuda', dtype=torch_out_dtype)

    # torch result
    a_f32 = a.to(torch.float32)
    b_f32 = b.to(torch.float32)
    dot = torch.matmul(a_f32, b_f32)

    def process_epilogue(d, bias, w, epilogue):
        if epilogue == 'add-matrix':
            ref = d + bias
        elif epilogue == 'add-rows':
            ref = d + bias[:, 0][:, None]
        elif epilogue == 'add-cols':
            ref = d + bias[0, :][None, :]
        elif epilogue == 'softmax':
            num = torch.exp(d - torch.max(d, dim=-1, keepdims=True)[0])
            denom = torch.sum(num, dim=-1, keepdims=True)
            ref = num / denom
            # ref = torch.softmax(d, 1)
        elif epilogue == 'chain-dot':
            ref = torch.matmul(d, w.to(torch.float32))
        else:
            ref = d
        return ref
    golden = process_epilogue(dot, bias, w, epilogue)

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    pgm = matmul_kernel[grid](a_ptr=a, b_ptr=b, w_ptr=w, bias_ptr=bias, z_ptr=z,
                              M=M, N=N, K=K,
                              stride_am=a.stride(0), stride_ak=a.stride(1),
                              stride_bk=b.stride(0), stride_bn=b.stride(1),
                              stride_wm=w.stride(0), stride_wn=w.stride(1),
                              stride_zm=z.stride(0), stride_zn=z.stride(1),
                              BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_SIZE_M=8,
                              out_dtype=out_dtype,
                              USE_TMA_STORE=USE_TMA_STORE,
                              ADD_MATRIX=epilogue == 'add-matrix',
                              ADD_ROWS=epilogue == 'add-rows',
                              ADD_COLS=epilogue == 'add-cols',
                              DO_SOFTMAX=epilogue == 'softmax',
                              CHAIN_DOT=epilogue == 'chain-dot',
                              A_ORDER_0=a_order[0], A_ORDER_1=a_order[1],
                              B_ORDER_0=b_order[0], B_ORDER_1=b_order[1],
                              num_warps=NUM_WARPS, num_ctas=NUM_CTAS, num_stages=NUM_STAGES)

    torch.set_printoptions(profile="full")
    # print("abs_err: {}, rel_err: {}".format(golden_abs_err, golden_rel_err))
    # print("golden: ")
    # print(golden)
    # print("result: ")
    # print(z)
    # print("max_gap: {}".format(torch.max(torch.abs(z - golden))))
    golden = torch.nn.functional.normalize(golden)
    z = torch.nn.functional.normalize(z)
    assert_close(z, golden,
                 rtol=1e-2,
                 atol=1e-3,
                 check_dtype=False)

    enable_mmav3 = os.environ.get('ENABLE_MMA_V3', 'not found').lower()
    if enable_mmav3 in ["on", "true", "1"] and BLOCK_M >= 64 and NUM_CTAS == 1 and BLOCK_N <= 256:
        ptx = pgm.asm['ptx']
        assert re.search(r'wgmma.mma_async.sync.aligned.m\d+n{}k16(?:.row.col)?.f32.f16.f16'.format(BLOCK_N), ptx)
