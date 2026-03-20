# References:
# - triton
#   - python/test/unit/language/test_matmul.py
# - aiter (commit: aab726524b952a23cb71577bd48c91a2db21c983)
#   - aiter/ops/triton/mha.py
#   - aiter/test_mha_common.py

# Example of the driver file
'''
#!/bin/bash

ROCPLAY_PATH=$(realpath ../../rocplay/rocplaycap-src-4.*)
ROCCAP_BIN=${ROCPLAY_PATH}/bin/roccap
ROCCAP_OPTIONS="capture --loglevel trace"

CURR_DIR=$PWD
TOP_OUTPUT_DIR="${CURR_DIR}/cap-mxfa"
rm -rf ${TOP_OUTPUT_DIR}
mkdir -p ${TOP_OUTPUT_DIR}

for CASE in 0 1 2 3; do
  for Q_TYPE in "e4m3"; do
    for KV_TYPE in "e4m3" "e2m1"; do
      ${ROCCAP_BIN} ${ROCCAP_OPTIONS} python3 ./test_mxfp_fa.py --case ${CASE} --q-type ${Q_TYPE} --kv-type ${KV_TYPE}
      dir_name=${TOP_OUTPUT_DIR}/${CASE}/${Q_TYPE}-${KV_TYPE}
      mkdir -p ${dir_name}
      mv ./roc_capture* ${dir_name}
    done
  done
done
'''

import os
# ruff: noqa: E402
import pytest
import torch
import triton
import triton.language as tl
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor
import math
from einops import repeat
from triton._internal_testing import is_hip_gfx1250, is_hip_cdna4

# For FA, the P=softmax(S) is performed very accurately in fp32 and
# P is in range [0, 1]; these then get down-cast to e4m3 which only has
# 3 bits of mantissa to represent the results known to be in the range [0, 1].
# For e4m3, the step between representable values can be used to determine
# the expected errors. The step size between e4m3 representable values
# grows relatively large in the range 0, 1 as we get closer to 1.
# For values above 0.5, the step size is 1/16.
# P range    ; Step; Error (assuming perfect round-to-nearest)
# [.5,  1.0 ]; 1/16; 1/32
# [.25,  .5 ]; 1/32; 1/64
# [.125, .25]; 1/64; 1/128
# Therefore half of the elements of P have an error of 1/16/2=0.03
# assuming perfect round-to-nearest.
# Other half of elements will have smaller errors, but also a smaller significance.
# So, the input to the 2nd dot are fp8 A operands in range [0, 1] with above error,
# other operands are "exactly" represented in their respective precisions.
# As we sum up many products, we expect the error to shrink;
# expected error shrinks as the sqrt(num_elements); e.g.
# tolerance = 1/16/2/sqrt(256) = 0.00195
RTOL = 0.00195
# ATOL < RTOL because results are far from zero.
ATOL = RTOL / 10
# While we hope that the large softmax downcast errors will cancel out,
# they won't some fraction of the time.
# Therefore we only expect most elements to be within tolerance.
# PTOL is the percent of elements which must meet the above criteria.
# Therefore the overall strategy for FA accuracy is to make sure most
# elements are highly accurate, rather than checking that all elements
# are barely accurate.
PTOL = .95

# Tolerances which 100% of elements must meet.
RTOL_100 = 0.15
ATOL_100 = RTOL_100 / 10


@triton.jit
def unshuffle_scale(x, non_k_dim, k_dim, preshuffle_factor: tl.constexpr):
    """ Unshuffle scales inside the kernel to restore the original shape. """
    block_non_k: tl.constexpr = non_k_dim // preshuffle_factor
    kwidth: tl.constexpr = 4 if k_dim >= 4 else k_dim
    block_k: tl.constexpr = k_dim // kwidth
    x = tl.reshape(x, (block_non_k, block_k, preshuffle_factor // 4, 4, kwidth))
    x = tl.permute(x, (0, 3, 2, 1, 4))
    x = tl.reshape(x, (non_k_dim, k_dim))
    return x


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    k_desc_or_ptrs,
    v_desc_or_ptrs,
    stride_kn,
    stride_vk,
    q_scale,
    k_scale_ptrs,
    v_scale_ptrs,
    stride_k_scale_n,
    stride_v_scale_n,
    block_min,
    block_max,
    q_type: tl.constexpr,
    kv_type: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    SM_SCALE: tl.constexpr,
    BLOCK_SCALE_FACTOR: tl.constexpr,
    KV_PACK_DIV: tl.constexpr,
    PRESHUFFLE_K_FACTOR: tl.constexpr,
    PRESHUFFLE_V_FACTOR: tl.constexpr,
    DISABLE_MASKING: tl.constexpr,
    USE_TDM: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634

    for i in range(block_min, block_max, BLOCK_N):
        if USE_TDM:
            k = k_desc_or_ptrs.load([i, 0]).T
        else:
            k = tl.load(k_desc_or_ptrs)

        if USE_TDM:
            k_scale = k_scale_ptrs.load([i // BLOCK_N, 0])
            k_scale = unshuffle_scale(k_scale, BLOCK_N, BLOCK_DMODEL // BLOCK_SCALE_FACTOR, PRESHUFFLE_K_FACTOR)
        else:
            k_scale = tl.load(k_scale_ptrs)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot_scaled(q, q_scale, q_type, k, k_scale, kv_type)

        # get max scores so far
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        m_ij_scaled = m_ij * SM_SCALE * RCP_LN2

        # scale and subtract max
        q_shifted = qk * SM_SCALE * RCP_LN2 - m_ij_scaled[:, None]

        # Compute scaled QK and softmax probabilities
        p = tl.math.exp2(q_shifted)

        # CAVEAT: Must update l_ij before applying dropout
        l_ij = tl.sum(p, 1)

        # update output accumulator
        # alpha is an adjustment factor for acc and li as we loop and find new maxes
        # store the diff in maxes to adjust acc and li as we discover new maxes
        m_diff_scaled = m_i * SM_SCALE * RCP_LN2 - m_ij_scaled
        alpha = tl.math.exp2(m_diff_scaled)
        acc = acc * alpha[:, None]

        if USE_TDM:
            v = v_desc_or_ptrs.load([i // KV_PACK_DIV, 0])
        else:
            v = tl.load(v_desc_or_ptrs)

        if USE_TDM:
            v_scale = v_scale_ptrs.load([0, i // BLOCK_N * stride_v_scale_n])
            v_scale = unshuffle_scale(v_scale, BLOCK_DMODEL, BLOCK_N // BLOCK_SCALE_FACTOR, PRESHUFFLE_V_FACTOR)
        else:
            v_scale = tl.load(v_scale_ptrs)

        # update m_i and l_i
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        acc += tl.dot_scaled(p.to(tl.float8e4nv), None, 'e4m3', v, v_scale, kv_type)

        if not USE_TDM:
            k_desc_or_ptrs += BLOCK_N * stride_kn
            k_scale_ptrs += BLOCK_N * stride_k_scale_n

        if not USE_TDM:
            v_desc_or_ptrs += (BLOCK_N // KV_PACK_DIV) * stride_vk
            v_scale_ptrs += (BLOCK_N // BLOCK_SCALE_FACTOR) * stride_v_scale_n

    return acc, l_i, m_i


@triton.jit
def _attn_fwd(
    q_ptr: torch.Tensor,
    k_ptr: torch.Tensor,
    v_ptr: torch.Tensor,
    q_scale_ptr: torch.Tensor,
    k_scale_ptr: torch.Tensor,
    v_scale_ptr: torch.Tensor,
    out_ptr: torch.Tensor,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_q_scale_z,
    stride_q_scale_h,
    stride_q_scale_m,
    stride_q_scale_k,
    stride_k_scale_z,
    stride_k_scale_h,
    stride_k_scale_n,
    stride_k_scale_k,
    stride_v_scale_z,
    stride_v_scale_h,
    stride_v_scale_n,
    stride_v_scale_k,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    sm_scale,
    q_type: tl.constexpr,
    kv_type: tl.constexpr,
    SEQLEN_Q: tl.constexpr,
    SEQLEN_K: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BATCH,
    BLOCK_SCALE_FACTOR: tl.constexpr,
    KV_PACK_DIV: tl.constexpr,
    PRESHUFFLE_K_FACTOR: tl.constexpr,
    PRESHUFFLE_V_FACTOR: tl.constexpr,
    DISABLE_MASKING: tl.constexpr,
    USE_TDM: tl.constexpr,
):
    NUM_BLOCKS = (SEQLEN_Q + BLOCK_M - 1) // BLOCK_M
    seqlen_q = SEQLEN_Q
    seqlen_k = SEQLEN_K

    # workgroup id ranging: 0,1,2,...., (BATCH * NUM_Q_HEADS * NUM_BLOCKS - 1)
    wid = tl.program_id(0)
    n_blocks = (seqlen_k + BLOCK_N - 1) // BLOCK_N

    # offsets
    off_q_head = wid % NUM_Q_HEADS
    start_m = (wid // NUM_Q_HEADS) % NUM_BLOCKS
    off_z = (wid // (NUM_BLOCKS * NUM_Q_HEADS)) % BATCH
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_n_packed = tl.arange(0, BLOCK_N // KV_PACK_DIV)
    offs_n_scale = tl.arange(0, BLOCK_N // BLOCK_SCALE_FACTOR)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_d_packed = tl.arange(0, BLOCK_DMODEL // KV_PACK_DIV)
    offs_d_scale = tl.arange(0, BLOCK_DMODEL // BLOCK_SCALE_FACTOR)
    off_k_head = off_q_head

    # q       [BLOCK_M, BLOCK_DMODEL]
    # q_scale [BLOCK_M, BLOCK_DMODEL / 32]
    if USE_TDM:
        q_desc_or_ptrs = tl.make_tensor_descriptor(
            base=q_ptr + off_z * stride_qz + off_q_head * stride_qh + start_m * BLOCK_M * stride_qm,
            shape=(BATCH * seqlen_q * NUM_Q_HEADS, BLOCK_DMODEL), strides=(stride_qm, stride_qk),
            block_shape=(BLOCK_M, BLOCK_DMODEL))
    else:
        q_offs = (off_z * stride_qz + off_q_head * stride_qh + offs_m[:, None] * stride_qm +
                  offs_d[None, :] * stride_qk)
        q_desc_or_ptrs = q_ptr + q_offs

    q_scale_offs = (off_z * stride_q_scale_z + off_q_head * stride_q_scale_h + offs_m[:, None] * stride_q_scale_m +
                    offs_d_scale[None, :] * stride_q_scale_k)
    q_scale_ptrs = q_scale_ptr + q_scale_offs

    # k       [BLOCK_DMODEL / KV_PACK_DIV, BLOCK_N]
    # k_scale [BLOCK_N, BLOCK_DMODEL / 32]
    if USE_TDM:
        # TDM requires last_stride=1, so we need to load K in a different way then transpose it.
        k_desc_or_ptrs = tl.make_tensor_descriptor(base=k_ptr + off_z * stride_kz + off_k_head * stride_kh,
                                                   shape=(BATCH * seqlen_k * NUM_K_HEADS // KV_PACK_DIV, BLOCK_DMODEL),
                                                   strides=(stride_kn, stride_kk),
                                                   block_shape=(BLOCK_N, BLOCK_DMODEL // KV_PACK_DIV))
    else:
        k_offs = (off_z * stride_kz + off_k_head * stride_kh + offs_d_packed[:, None] * stride_kk +
                  offs_n[None, :] * stride_kn)
        k_desc_or_ptrs = k_ptr + k_offs

    if USE_TDM:
        kscale_shape = (BATCH * seqlen_k * NUM_K_HEADS, BLOCK_DMODEL // BLOCK_SCALE_FACTOR)
        k_scale_ptrs = tl.make_tensor_descriptor(
            base=k_scale_ptr + off_z * stride_k_scale_z + off_k_head * stride_k_scale_h,
            shape=(kscale_shape[0] // PRESHUFFLE_K_FACTOR, kscale_shape[1] * PRESHUFFLE_K_FACTOR),
            strides=(stride_k_scale_n, stride_k_scale_k),
            block_shape=(BLOCK_N // PRESHUFFLE_K_FACTOR, BLOCK_DMODEL // BLOCK_SCALE_FACTOR * PRESHUFFLE_K_FACTOR))
    else:
        k_scale_offs = (off_z * stride_k_scale_z + off_k_head * stride_k_scale_h + offs_n[:, None] * stride_k_scale_n +
                        offs_d_scale[None, :] * stride_k_scale_k)
        k_scale_ptrs = k_scale_ptr + k_scale_offs

    # v       [BLOCK_N / KV_PACK_DIV, BLOCK_DMODEL]
    # v_scale [BLOCK_DMODEL, BLOCK_N / 32]
    if USE_TDM:
        v_desc_or_ptrs = tl.make_tensor_descriptor(
            base=v_ptr + off_z * stride_vz + off_k_head * stride_vh,
            shape=(BATCH * (seqlen_k // KV_PACK_DIV) * NUM_K_HEADS, BLOCK_DMODEL), strides=(stride_vn, stride_vk),
            block_shape=(BLOCK_N // KV_PACK_DIV, BLOCK_DMODEL))
    else:
        v_offs = (off_z * stride_vz + off_k_head * stride_vh + offs_n_packed[:, None] * stride_vn +
                  offs_d[None, :] * stride_vk)
        v_desc_or_ptrs = v_ptr + v_offs

    if USE_TDM:
        vscale_shape = (BATCH * BLOCK_DMODEL * NUM_K_HEADS, seqlen_k // BLOCK_SCALE_FACTOR)
        v_scale_ptrs = tl.make_tensor_descriptor(
            base=v_scale_ptr + off_z * stride_v_scale_z + off_k_head * stride_v_scale_h,
            shape=(vscale_shape[0] // PRESHUFFLE_V_FACTOR, vscale_shape[1] * PRESHUFFLE_V_FACTOR),
            strides=(stride_v_scale_n, 1),
            block_shape=(BLOCK_DMODEL // PRESHUFFLE_V_FACTOR, BLOCK_N // BLOCK_SCALE_FACTOR * PRESHUFFLE_V_FACTOR))
        stride_v_scale_n = BLOCK_N // BLOCK_SCALE_FACTOR * PRESHUFFLE_V_FACTOR
    else:
        v_scale_offs = (off_z * stride_v_scale_z + off_k_head * stride_v_scale_h + offs_d[:, None] * stride_v_scale_k +
                        offs_n_scale[None, :] * stride_v_scale_n)
        v_scale_ptrs = v_scale_ptr + v_scale_offs

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    q_mask = True if DISABLE_MASKING else offs_m[:, None] < seqlen_q
    if USE_TDM:
        q = q_desc_or_ptrs.load([0, 0])
    else:
        q = tl.load(q_desc_or_ptrs, mask=q_mask, other=0.0)
    q_scale = tl.load(q_scale_ptrs, mask=q_mask, other=0x7F)

    block_min = 0
    block_max = n_blocks * BLOCK_N
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, k_desc_or_ptrs, v_desc_or_ptrs, stride_kn, stride_vn, q_scale,
                                    k_scale_ptrs, v_scale_ptrs, stride_k_scale_n, stride_v_scale_n, block_min,
                                    block_max, q_type, kv_type, BLOCK_M, BLOCK_N, BLOCK_DMODEL, sm_scale,
                                    BLOCK_SCALE_FACTOR, KV_PACK_DIV, PRESHUFFLE_K_FACTOR, PRESHUFFLE_V_FACTOR,
                                    DISABLE_MASKING, USE_TDM)

    # epilogue
    # This helps the compiler do Newton Raphson on l_i vs on acc which is much larger.
    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip

    # If seqlen_q > seqlen_k but the delta is not a multiple of BLOCK_M,
    # then we have one block with a row of all NaNs which come from computing
    # softmax over a row of all -infs (-inf - inf = NaN). We check for that here
    # and store 0s where there are NaNs as these rows should've been zeroed out.
    end_m_idx = (start_m + 1) * BLOCK_M

    # write back O
    overflow_size = end_m_idx - seqlen_q

    if USE_TDM:
        o_desc_or_ptrs = tl.make_tensor_descriptor(
            base=out_ptr + off_z * stride_oz + off_q_head * stride_oh + start_m * BLOCK_M * stride_om,
            shape=(BATCH * seqlen_q * NUM_Q_HEADS, BLOCK_DMODEL), strides=(stride_om, stride_on),
            block_shape=(BLOCK_M, BLOCK_DMODEL))

    else:
        offs_out = (off_z * stride_oz + off_q_head * stride_oh + offs_m[:, None] * stride_om +
                    offs_d[None, :] * stride_on)
    out_mask = tl.full([BLOCK_M, BLOCK_DMODEL], 1, dtype=tl.int1)
    if overflow_size > 0:
        out_mask = out_mask & (offs_m[:, None] < seqlen_q)

    out_mask = True if DISABLE_MASKING else out_mask
    op = acc.to(out_ptr.dtype.element_ty)
    if USE_TDM:
        o_desc_or_ptrs.store([0, 0], op)
    else:
        tl.store(out_ptr + offs_out, op, mask=out_mask)


def preshuffle_scale(x: torch.Tensor, preshuffle_factor: int = 128):
    """ Preshuffle scales for scaled wmma instruction.
    In scaled wmma instruction, scales takes following shapes in global memory:
    - a_scale: [M, K // 32]
    - b_scale: [N, K // 32]

    To have vectorized memory access, it's better to store scales in a packed block scale layout. In this
    layout, scales are stored contiguously in the shape of:
    - a_scale: [M // 32 // 4, K // 32 // 4, 32, 4, 4]
    - b_scale: [N // 32 // 4, K // 32 // 4, 32, 4, 4]

    The output shape will be
    - a_scale: [M // preshuffle_factor, K * preshuffle_factor]
    - b_scale: [N // preshuffle_factor, K * preshuffle_factor]

    In this way, we can load scales from global memory in a more vectorized way. Then inside the kernel, we
    permute and reshape scales to canonical shapes required by scaled wmma.
    """
    *prefix, non_k, k = x.shape
    scale_kwidth = 4 if k >= 4 else k
    num_chunk_m = non_k // preshuffle_factor
    num_chunk_k = k // scale_kwidth

    batch = math.prod(prefix)
    x = x.reshape(batch, non_k, k)
    x = x.view(batch, num_chunk_m, 4, preshuffle_factor // 4, num_chunk_k, scale_kwidth)
    x = x.permute(0, 1, 4, 3, 2, 5).contiguous()
    x = x.view(batch, num_chunk_m, k * preshuffle_factor)

    return x.view(*prefix, non_k // preshuffle_factor, k * preshuffle_factor)


def attn_fwd(q, k, v, q_scale, k_scale, v_scale, config, args, block_scale_factor):
    softmax_scale = q.shape[-1]**(-0.5)

    o = torch.zeros_like(q, dtype=torch.float32)

    batch, seqlen_q, num_q_heads, head_sz = q.shape
    num_k_heads = k.shape[2]
    q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
    k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
    v_strides = (v.stride(0), v.stride(2), v.stride(1), v.stride(3))
    q_scale_strides = (q_scale.stride(0), q_scale.stride(2), q_scale.stride(1), q_scale.stride(3))
    k_scale_strides = (k_scale.stride(0), k_scale.stride(2), k_scale.stride(1), k_scale.stride(3))
    v_scale_strides = (v_scale.stride(0), v_scale.stride(2), v_scale.stride(1), v_scale.stride(3))
    o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))

    block_n = config["BLOCK_N"]
    preshuffle_k_factor = 128 if block_n >= 128 else block_n
    preshuffle_v_factor = 128 if head_sz >= 128 else head_sz
    # Preshuffle scales
    if args.tdm:
        # k_scale: [BATCH, NUM_K_HEADS, SEQLEN_K, HEAD_SZ / 32]
        k_scale = preshuffle_scale(k_scale.permute(0, 2, 1, 3), preshuffle_k_factor)
        k_scale_strides = (k_scale.stride(0), k_scale.stride(1), k_scale.stride(2), k_scale.stride(3))
        # v_scale: [BATCH, NUM_K_HEADS, HEAD_SZ, SEQLEN_K / 32]
        v_scale = preshuffle_scale(v_scale.permute(0, 2, 3, 1), preshuffle_v_factor)
        v_scale_strides = (v_scale.stride(0), v_scale.stride(1), v_scale.stride(2), v_scale.stride(3))

    q = q.cuda()
    k = k.cuda()
    v = v.cuda()
    q_scale = q_scale.cuda()
    k_scale = k_scale.cuda()
    v_scale = v_scale.cuda()
    o = o.cuda()

    q_type = args.q_type
    kv_type = args.kv_type
    kv_pack_div = 2 if kv_type == 'e2m1' else 1

    grid = lambda META: (batch * num_q_heads * triton.cdiv(seqlen_q, META["BLOCK_M"]), )

    handle = _attn_fwd[grid](q, k, v, q_scale, k_scale, v_scale, o, *q_strides, *k_strides, *v_strides,
                             *q_scale_strides, *k_scale_strides, *v_scale_strides, *o_strides, softmax_scale, q_type,
                             kv_type, SEQLEN_Q=q.shape[1], SEQLEN_K=k.shape[1], NUM_Q_HEADS=num_q_heads,
                             NUM_K_HEADS=num_k_heads, BLOCK_DMODEL=head_sz, BATCH=batch, BLOCK_M=config["BLOCK_M"],
                             BLOCK_N=config["BLOCK_N"], BLOCK_SCALE_FACTOR=block_scale_factor, KV_PACK_DIV=kv_pack_div,
                             PRESHUFFLE_K_FACTOR=preshuffle_k_factor, PRESHUFFLE_V_FACTOR=preshuffle_v_factor,
                             DISABLE_MASKING=args.disable_masking, num_warps=config["NUM_WARPS"],
                             num_stages=config["NUM_STAGES"], USE_TDM=args.tdm)

    if args.dump_ir != 'none':
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        filename = f'attn_kernel.{args.dump_ir}'
        with open(os.path.join(curr_dir, filename), "w") as file:
            file.write(handle.asm[args.dump_ir])

    return o.cpu()


def attn_ref(q, k, v, q_scale, k_scale, v_scale):
    dtype_og = q.dtype

    q = q * q_scale
    k = k * k_scale
    v = v * v_scale

    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]

    scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    output = torch.einsum("bhts,bshd->bthd", attention, v)

    return output.to(dtype=dtype_og)


def get_percent_close(tensor, reference, atol, rtol) -> float:
    elements_close = torch.isclose(tensor, reference, atol=atol, rtol=rtol)
    num_close = sum(torch.flatten(elements_close))
    percent_close = num_close / torch.numel(reference)
    return percent_close


def run_mha(config, args):
    # MXFP block scale factor: one scale value per 32 elements
    BLOCK_SCALE_FACTOR = 32

    BATCH = config['BATCH']
    SEQLEN_Q = config['SEQLEN_Q']
    SEQLEN_K = config['SEQLEN_K']
    NUM_Q_HEADS = config['NUM_Q_HEADS']
    NUM_K_HEADS = config['NUM_K_HEADS']
    HEAD_SZ = config['HEAD_SZ']

    # Intialize the data to exactly presentable values near 1. Having all values near the
    # same order of magnitude makes the accumulation higher percision. Data is initialized as
    # mxfp8 data=[20, 40] and scales=[1/16, 2] -> data*scale=[1.25, 80];
    # mxfp4 data=[1, 4] and scales=[1/4, 16] -> data*scale=[0.25, 64].
    def create_operand(dtype: str, b: int, s: int, h: int, d: int, pack_dim: int = -1):
        if dtype == 'e4m3':
            v = torch.randint(20, 40, (b, s, h, d), dtype=torch.uint8).to(torch.float8_e4m3fn)
            v_ref = v.to(torch.float32)
        elif dtype == 'e5m2':
            v = torch.randint(20, 40, (b, s, h, d), dtype=torch.uint8).to(torch.float8_e5m2)
            v_ref = v.to(torch.float32)
        else:
            assert dtype == 'e2m1'
            assert pack_dim >= 0
            data = torch.randint(1, 5, (b, s, h, d))
            v_mxfp4 = MXFP4Tensor(data=data)
            v = v_mxfp4.to_packed_tensor(pack_dim)
            v_ref = v_mxfp4.to(torch.float32)
        return v, v_ref

    def create_scale(dtype: str, b: int, s: int, h: int, d: int, scale_dim: int):
        size = [b, s, h, d]
        size[scale_dim] //= BLOCK_SCALE_FACTOR
        low = 1.0 / 16
        high = 2
        if dtype == 'e2m1':
            # Scales should offset the magnitude of the data so that net data
            # is near 1, this keeps the sum from exploding toward inaccuracy.
            low = 1.0 / 4
            high = 16
        scale = MXScaleTensor(size=tuple(size)).random(low=low, high=high)
        scale_ref = scale.to(torch.float32).repeat_interleave(BLOCK_SCALE_FACTOR, dim=scale_dim)
        return scale.data, scale_ref

    torch.random.manual_seed(0)
    q, q_ref = create_operand(args.q_type, BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ)
    k, k_ref = create_operand(args.kv_type, BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ, pack_dim=3)
    v, v_ref = create_operand(args.kv_type, BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ, pack_dim=1)
    q_scale, q_scale_ref = create_scale(args.q_type, BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ, scale_dim=3)
    k_scale, k_scale_ref = create_scale(args.kv_type, BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ, scale_dim=3)
    v_scale, v_scale_ref = create_scale(args.kv_type, BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ, scale_dim=1)

    triton_out = attn_fwd(q, k, v, q_scale, k_scale, v_scale, config, args, BLOCK_SCALE_FACTOR)
    torch_out = attn_ref(q_ref, k_ref, v_ref, q_scale_ref, k_scale_ref, v_scale_ref)

    try:
        torch.testing.assert_close(triton_out, torch_out, atol=ATOL_100, rtol=RTOL_100)
    except Exception as err:
        print("❌ Triton and Torch differ")
        print(err)
        if args.verbose:
            print(f"{triton_out=}")
            print(f"{torch_out=}")
        pytest.fail()
        return

    # Check high tolerances for most elements.
    percent_close = get_percent_close(triton_out, torch_out, ATOL, RTOL)
    if percent_close >= PTOL:
        print("✅ Triton within tolerances.")
        return
    else:
        print("❌ Triton %f%% < %f%% elements within rtol=%f, atol=%f." % (percent_close * 100, PTOL * 100, RTOL, ATOL))
        pytest.fail()


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("num_heads", [1, 16])
@pytest.mark.parametrize("seqlen", [256, 512, 1024])
@pytest.mark.parametrize("head_sz", [128, 64])
@pytest.mark.parametrize("block_m", [128, 64])
@pytest.mark.parametrize("q_type", ["e4m3"])
@pytest.mark.parametrize("kv_type", ["e4m3", "e2m1"])
@pytest.mark.parametrize("num_stages", [1, 3])
@pytest.mark.parametrize("USE_TDM", [True, False])
def test_mha(batch, num_heads, seqlen, head_sz, block_m, q_type, kv_type, num_stages, USE_TDM):
    if not (is_hip_gfx1250() or is_hip_cdna4()):
        pytest.skip("MXFP FA kernels are only tested on AMD GFX1250 or CDNA4.")
    if kv_type == "e2m1" and USE_TDM:
        pytest.skip("Numerical failures need investigation.")
    block_n = 128
    config = {
        "BATCH": batch,  #
        "NUM_Q_HEADS": num_heads,  #
        "NUM_K_HEADS": num_heads,  #
        "SEQLEN_Q": seqlen,  #
        "SEQLEN_K": seqlen,  #
        "HEAD_SZ": head_sz,  #
        "BLOCK_M": block_m,  #
        "BLOCK_N": block_n,  #
        "WAVES_PER_EU": 1,  #
        "NUM_WARPS": 4,  #
        "NUM_CTAS": 1,  #
        "NUM_STAGES": num_stages
    }

    class Args:

        def __init__(self, q_type, kv_type):
            self.q_type = q_type
            self.kv_type = kv_type
            self.verbose = False
            self.disable_masking = False
            self.dump_ir = 'none'
            self.tdm = USE_TDM

    args = Args(q_type, kv_type)

    run_mha(config, args)


def generate_configs(args):
    MAX_BATCH = 64
    num_stages = args.num_stages if args.num_stages != -1 else 3
    BLOCK_M_FOR_HEAD_SIZE_128 = 128 if args.kv_type == 'e4m3' or args.kv_type == "e5m2" else 256
    base_configs = [
        # HEAD_SZ == 128
        {
            "BATCH": 1, "NUM_Q_HEADS": 16, "NUM_K_HEADS": 16, "SEQLEN_Q": 8192, "SEQLEN_K": 8192, "HEAD_SZ": 128,
            "BLOCK_M": BLOCK_M_FOR_HEAD_SIZE_128, "BLOCK_N": 128, "WAVES_PER_EU": 1, "NUM_WARPS": 4, "NUM_CTAS": 1,
            "NUM_STAGES": num_stages
        },
        {
            "BATCH": MAX_BATCH, "NUM_Q_HEADS": 16, "NUM_K_HEADS": 16, "SEQLEN_Q": 1, "SEQLEN_K": 8192, "HEAD_SZ": 128,
            "BLOCK_M": BLOCK_M_FOR_HEAD_SIZE_128, "BLOCK_N": 128, "WAVES_PER_EU": 1, "NUM_WARPS": 4, "NUM_CTAS": 1,
            "NUM_STAGES": num_stages
        },
        # HEAD_SZ == 64
        {
            "BATCH": 1, "NUM_Q_HEADS": 16, "NUM_K_HEADS": 16, "SEQLEN_Q": 8192, "SEQLEN_K": 8192, "HEAD_SZ": 64,
            "BLOCK_M": 512, "BLOCK_N": 64, "WAVES_PER_EU": 1, "NUM_WARPS": 4, "NUM_CTAS": 1, "NUM_STAGES": num_stages
        },
        {
            "BATCH": MAX_BATCH, "NUM_Q_HEADS": 16, "NUM_K_HEADS": 16, "SEQLEN_Q": 1, "SEQLEN_K": 8192, "HEAD_SZ": 64,
            "BLOCK_M": 512, "BLOCK_N": 64, "WAVES_PER_EU": 1, "NUM_WARPS": 4, "NUM_CTAS": 1, "NUM_STAGES": num_stages
        },
    ]
    return base_configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-type", choices=["e5m2", "e4m3"], default="e4m3",
                        help="data type for K and V (default e4m3)")
    parser.add_argument("--kv-type", choices=["e5m2", "e4m3", "e2m1"], default="e4m3",
                        help="data type for K and V (default e2m1)")
    parser.add_argument("--dump-ir", choices=['none', 'ttir', 'ttgir', 'llir', 'amdgcn'], default="none",
                        help="dump IR format")
    parser.add_argument("-c", "--case", type=int, required=True, help='case id')
    parser.add_argument("-t", "--tdm", action='store_true', default=False, help='enable TDM')
    parser.add_argument("--num-stages", type=int, default=-1, required=False, help='num stages')
    parser.add_argument("-m", "--disable-masking", action='store_true', help='use masked loads')
    parser.add_argument("-v", "--verbose", action='store_true', help='verbose output')
    args = parser.parse_args()

    print(f'{args.q_type=}; {args.kv_type=}; {args.disable_masking=}')
    print(f'Testing with {RTOL=}; {ATOL=}; {PTOL=}')

    configs = generate_configs(args)
    config = configs[args.case]

    print(f'{config=}')
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    filename = 'mxfa-curr-config.txt'
    with open(os.path.join(curr_dir, filename), "w") as file:
        file.write(f'{config=}\n')
        file.write(f'{args.q_type=}\n')
        file.write(f'{args.kv_type=}\n')
        file.write(f'{args.disable_masking=}\n')

    run_mha(config, args)
