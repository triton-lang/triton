import triton
import triton.language as tl

from triton.language.core import builtin
from triton.language.core import _aggregate as aggregate

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.tools.tensor_descriptor import TensorDescriptor


@builtin
def get_1d_blocked(num_warps, num_threads, _builder=None):
    return gl.BlockedLayout([1], [num_threads], [num_warps], [1])


@gluon.jit
def _attn_fwd(sm_scale, M, Z, H, N_CTX,  #
              desc_q, desc_k, desc_v, desc_o,  #
              HEAD_DIM: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,  #
              STAGE: gl.constexpr, dtype: gl.constexpr,  #
              num_warps: gl.constexpr, threads_per_warp: gl.constexpr):
    tl.static_assert(BLOCK_N <= HEAD_DIM, "BLOCK_N must be less than or equal to HEAD_DIM")

    start_m = gl.program_id(0)
    off_hz = gl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX
    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M

    layout_1d: gl.constexpr = get_1d_blocked(num_warps, threads_per_warp)
    offs_m = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout_1d)


def attention_forward(q, k, v, causal, sm_scale):
    HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
    HEAD_DIM_V = v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V

    stage = 3 if causal else 1
    o = torch.empty_like(q)
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

    y_dim = q.shape[0] * q.shape[1] * q.shape[2]

    BLOCK_M = 256
    BLOCK_N = min(HEAD_DIM_K, 128)

    desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM, 1], block_shape=[BLOCK_M, HEAD_DIM_K])
    desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM, 1], block_shape=[BLOCK_N, HEAD_DIM_K])
    desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM, 1], block_shape=[BLOCK_N, HEAD_DIM_K])
    desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM, 1], block_shape=[BLOCK_M, HEAD_DIM_K])

    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)

    kernel = _attn_fwd.warmup(  #
        sm_scale, M, q.shape[0], q.shape[1], q.shape[2],  #
        desc_q, desc_k, desc_v, desc_o,  #
        HEAD_DIM_K, BLOCK_M, BLOCK_N,  #
        stage, q.dtype,  #
        num_warps=4, threads_per_warp=32,  #
        grid=grid,  #
    )
    print(kernel.asm["ttgir"])


if __name__ == "__main__":
    import torch

    BATCH = 4
    H = 32
    N_CTX = 16 * 1024
    HEAD_DIM = 128
    causal = True

    dtype = torch.float16
    device = "cuda"

    sm_scale = 1.3
    q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device)
    k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device)
    v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device)

    attention_forward(q, k, v, causal, sm_scale)
