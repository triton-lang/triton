import torch
import triton
import triton.language as tl

from triton_kernels.matmul_details.opt_flags_details.opt_flags_nvidia import compute_split_k
from triton_kernels.tensor import FP4, Tensor, dtype_to_torch_dtype
from triton_kernels.tensor_details.layout import StridedLayout


@triton.jit
def _load_tile(ptr, row, col, valid_rows, width, row_stride, col_stride, other=0.0):
    offsets = row[:, None].to(tl.int64) * row_stride + col[None, :].to(tl.int64) * col_stride
    mask = valid_rows[:, None] & (col[None, :] < width)
    return tl.load(ptr + offsets, mask, other)


@triton.jit
def _pad_selector_word(word, word_idx, last_word, groups: tl.constexpr):
    # Unused nibbles in the last word must also be legal selectors.
    valid_bits: tl.constexpr = (1 << (4 * groups)) - 1
    padded = (word & valid_bits) | (0x44444444 & ~valid_bits)
    return tl.where(word_idx[None, :] == last_word, padded, word)


@triton.jit
def _sparse_matmul(X, W, E, SX, SW, Bias, Out, Gather, Scatter, Schedule, SliceOffsets, SliceSizes, M, N: tl.constexpr,
                   K: tl.constexpr, M_TILES, PARTIAL_GROUPS: tl.constexpr, X_STRIDE: tl.constexpr,
                   W_STRIDE: tl.constexpr, E_STRIDE: tl.constexpr, SX_STRIDE: tl.constexpr, SW_STRIDE: tl.constexpr,
                   OUT_STRIDE: tl.constexpr, BIAS_STRIDE: tl.constexpr, XScale, WScale, Alpha, BK: tl.constexpr,
                   NVFP4: tl.constexpr, SCALE_K: tl.constexpr, BLOCK_M: tl.constexpr, SPLIT_K: tl.constexpr):
    # The instruction compresses its left operand. Compute W.T @ X.T.
    pid = tl.program_id(0)
    n_tiles = tl.cdiv(N, 128)
    n_tile = pid % n_tiles
    batch = pid // n_tiles // M_TILES
    expert = batch
    m_tile = pid // n_tiles % M_TILES
    start = 0
    if Schedule is not None:
        entry = tl.load(Schedule + m_tile)
        if entry == -1:
            return
        expert = entry & 65535
        m_tile = (entry.to(tl.uint32) >> 16).to(tl.int32)
        start = tl.load(SliceOffsets + expert)
        M = start + tl.load(SliceSizes + expert)
    n = n_tile * 128 + tl.arange(0, 128)
    m = start + m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    xm = m
    ym = m
    if Gather is not None:
        xm = tl.load(Gather + m, m < M, -1)
    if Scatter is not None:
        ym = tl.load(Scatter + m, m < M, -1)
    xmask = (m < M) & (xm >= 0)
    X += batch.to(tl.int64) * X_STRIDE[0]
    W += expert.to(tl.int64) * W_STRIDE[0]
    E += expert.to(tl.int64) * E_STRIDE[0]
    split = tl.program_id(1)
    Out += (batch.to(tl.int64) * SPLIT_K + split) * OUT_STRIDE[0]
    if NVFP4:
        SX += batch.to(tl.int64) * SX_STRIDE[0]
        SW += expert.to(tl.int64) * SW_STRIDE[0]
    pack: tl.constexpr = 2 if NVFP4 else 1
    words: tl.constexpr = 2
    acc = tl.full((128, BLOCK_M), 0, tl.float32)
    for tile in range(split, tl.cdiv(K, BK), SPLIT_K):
        xk = tile * (BK // pack) + tl.arange(0, BK // pack)
        wk = tile * (BK // (2 * pack)) + tl.arange(0, BK // (2 * pack))
        ek = tile * words + tl.arange(0, words)
        x = _load_tile(X, xm, xk, xmask, K // pack, X_STRIDE[1], X_STRIDE[2])
        w = _load_tile(W, n, wk, n < N, K // (2 * pack), W_STRIDE[2], W_STRIDE[1])
        e = _load_tile(E, n, ek, n < N, tl.cdiv(K, BK // words), E_STRIDE[1], E_STRIDE[2], 0x44444444)
        if PARTIAL_GROUPS:
            e = _pad_selector_word(e, ek, K // (BK // words), PARTIAL_GROUPS)
        if NVFP4:
            sk = tile * (BK // SCALE_K) + tl.arange(0, BK // SCALE_K)
            xs = _load_tile(SX, xm, sk, xmask, K // SCALE_K, SX_STRIDE[1], SX_STRIDE[2])
            ws = _load_tile(SW, n, sk, n < N, K // SCALE_K, SW_STRIDE[2], SW_STRIDE[1])
        else:
            ws, xs = None, None
        acc = tl.sparse_dot(w, tl.trans(x), e, acc, ws, xs)
    y = tl.trans(acc)
    if SPLIT_K == 1:
        y = _epilogue(y, n[None, :], N, expert, Bias, BIAS_STRIDE, XScale, WScale, Alpha)
    out_offsets = ym[:, None].to(tl.int64) * OUT_STRIDE[1] + n[None, :].to(tl.int64) * OUT_STRIDE[2]
    tl.store(Out + out_offsets, y, ((ym >= 0) & (m < M))[:, None] & (n[None, :] < N))


@triton.jit
def _epilogue(y, n, N, expert, Bias, BIAS_STRIDE: tl.constexpr, XScale, WScale, Alpha):
    x_scale = tl.load(XScale) if XScale is not None else 1.0
    w_scale = tl.load(WScale) if WScale is not None else 1.0
    y *= x_scale * w_scale
    if Bias is not None:
        y += tl.load(Bias + expert.to(tl.int64) * BIAS_STRIDE[0] + n.to(tl.int64) * BIAS_STRIDE[1], n < N, 0)
    return y * Alpha


@triton.jit
def _reduce_partials(Partials, Out, Bias, XScale, WScale, Alpha, M: tl.constexpr, N: tl.constexpr,
                     OUT_STRIDE: tl.constexpr, BIAS_STRIDE: tl.constexpr, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    tiles = tl.cdiv(M * N, BLOCK)
    pid = tl.program_id(0)
    batch = pid // tiles
    offset = (pid % tiles).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    split = tl.arange(0, triton.next_power_of_2(SPLIT_K))
    base = (batch.to(tl.int64) * SPLIT_K + split) * M * N
    partial = tl.load(Partials + base[:, None] + offset[None, :],
                      (split[:, None] < SPLIT_K) & (offset[None, :] < M * N), 0)
    n = offset % N
    y = _epilogue(tl.sum(partial, 0), n, N, batch, Bias, BIAS_STRIDE, XScale, WScale, Alpha)
    out_offset = batch.to(tl.int64) * OUT_STRIDE[0] + (offset // N) * OUT_STRIDE[1] + n * OUT_STRIDE[2]
    tl.store(Out + out_offset, y, offset < M * N)


def _storage(tensor, packed_axis=None):
    if not isinstance(tensor, Tensor):
        return tensor
    if not isinstance(tensor.storage.layout, StridedLayout):
        raise ValueError("sparse matmul requires strided storage")
    expected = list(tensor.data.shape)
    if tensor.dtype == FP4:
        if packed_axis is None or tensor.storage.layout.order(tensor.ndim)[0] != packed_axis % tensor.ndim:
            raise ValueError("NVFP4 values must be packed along K")
        expected[packed_axis] *= 2
    if tensor.shape != expected:
        raise ValueError("logical sparse operand shape does not match its storage")
    storage_dtype = dtype_to_torch_dtype(tensor.dtype)
    if storage_dtype.itemsize != tensor.data.element_size():
        raise ValueError("sparse operand dtype must match its storage width")
    return tensor.data.view(storage_dtype)


def _strides(tensor, rank):
    if tensor is None:
        return (0, ) * rank
    return (0, ) * (rank - tensor.ndim) + tensor.stride()


def sparse_matmul(a, b, sparsity_metadata, bias, config, c, alpha, ragged, gather, scatter):
    a, b = _storage(a, -1), _storage(b, -2)
    sparsity_metadata = _storage(sparsity_metadata)
    if not all(isinstance(t, torch.Tensor) and t.ndim in (2, 3) for t in (a, b, sparsity_metadata)):
        raise ValueError("sparse matmul expects strided matrices or batches of matrices")
    if ragged is None and a.shape[:-2] != b.shape[:-2]:
        raise ValueError("input and weight batch dimensions must match")
    if ragged is not None and (a.ndim != 2 or b.ndim != 3 or b.shape[0] != ragged.n_slices):
        raise ValueError("ragged-M matmul requires rank-2 activations and one weight matrix per expert")
    for index in (gather, scatter):
        if index is not None and (a.ndim != 2 or index.ndim != 1 or index.dtype not in (torch.int32, torch.int64)
                                  or index.stride(0) != 1):
            raise ValueError("routing indices must be contiguous integer vectors for rank-2 activations")
    cc = torch.cuda.get_device_capability(a.device) if a.is_cuda and torch.version.cuda else (0, 0)
    if not ((8, 9) <= cc < (10, 0) or cc in ((10, 0), (10, 3))):
        raise ValueError("sparse matmul requires Ada, Hopper or Blackwell SM100/SM103")
    if a.dtype != b.dtype or a.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2, torch.uint8):
        raise ValueError("expected matching FP8 or byte-packed NVFP4 operands")
    nvfp4 = a.dtype == torch.uint8
    if nvfp4 and cc < (10, 0):
        raise ValueError("native paired NVFP4 sparse matmul requires SM100 or SM103")
    m, stored_k = a.shape[-2:]
    k = stored_k * (2 if nvfp4 else 1)
    n = b.shape[-1]
    group = 8 if nvfp4 else 4
    if k % group or (nvfp4 and k % 16) or b.shape[-2] * 2 != stored_k:
        raise ValueError("compressed weights must store half the logical K values")
    if sparsity_metadata.dtype != torch.int32 or sparsity_metadata.shape != (*b.shape[:-2], n, triton.cdiv(
            k, group * 8)):
        raise ValueError("expected one int32 sparsity metadata word per eight selector groups per output column")
    if sparsity_metadata.stride(-2) != 1:
        raise ValueError("b_sparsity_metadata must be column-major (stride(-2) == 1)")
    xs, ws = _storage(config.a_mx_scale), _storage(config.b_mx_scale)
    scale_k = config.a_microblock_size or 16
    if nvfp4:
        if (config.b_microblock_size or 16) != scale_k or k % scale_k:
            raise ValueError("NVFP4 requires equal scale block sizes that divide K")
        for scale, shape in ((xs, (*a.shape[:-2], m, k // scale_k)), (ws, (*b.shape[:-2], k // scale_k, n))):
            if not isinstance(scale, torch.Tensor) or scale.dtype != torch.float8_e4m3fn or scale.shape != shape:
                raise ValueError(f"NVFP4 requires E4M3 scales with shape {shape}")
    elif xs is not None or ws is not None:
        raise ValueError("microscaling requires NVFP4")
    if bias is not None and bias.shape not in ((n, ), (*b.shape[:-2], n)):
        raise ValueError("sparse matmul expects a bias vector of length N")
    for scale in (config.a_mx_tensor_scale, config.b_mx_tensor_scale):
        if scale is not None and scale.numel() != 1:
            raise ValueError("tensor scale must be scalar")
    dtype = config.out_dtype or (c.dtype if c is not None else torch.bfloat16 if nvfp4 else a.dtype)
    rows = m if gather is None else gather.numel()
    if scatter is not None and scatter.numel() != rows:
        raise ValueError("scatter must have one entry per routed row")
    output_shape = (*a.shape[:-2], rows, n)
    if c is None:
        # Remapping can leave output rows with no scheduled expert.
        allocate = torch.zeros if ragged is not None or scatter is not None else torch.empty
        c = allocate(output_shape, device=a.device, dtype=dtype)
    if c.shape != output_shape or c.dtype != dtype:
        raise ValueError("output shape or dtype does not match sparse matmul")
    for t in (b, sparsity_metadata, xs, ws, config.a_mx_tensor_scale, config.b_mx_tensor_scale, bias, c, gather,
              scatter):
        if t is not None and t.device != a.device:
            raise ValueError("sparse matmul operands must share a device")
    batch = a.shape[0] if a.ndim == 3 else 1
    schedule = ragged.block_schedule(128) if ragged is not None else None
    offsets = ragged.slice_offs if ragged is not None else None
    sizes = ragged.slice_sizes if ragged is not None else None
    if offsets is not None and any(t.device != a.device for t in (offsets, schedule, sizes)):
        raise ValueError("routing metadata must be on the same device as the activations")
    block_m = 128 if ragged is not None else min(128, max(16, triton.next_power_of_2(max(rows, 1))))
    m_tiles = triton.cdiv(rows, block_m) if ragged is None else min(schedule.numel(),
                                                                    ragged.n_blocks(ragged.n_slices, rows, 128))
    if batch and m_tiles and n:
        bk = 128 if nvfp4 else 64
        tiles = triton.cdiv(n, 128) * m_tiles * batch
        split_k = 1
        if ragged is None and gather is None and scatter is None:
            split_k = compute_split_k(bk, k, tiles, device=a.device)
            # FP8 needs a substantially underfilled grid to amortize the FP32 scratch traffic.
            if not nvfp4 and split_k < 4:
                split_k = 1
        # Each CTA writes an FP32 partial. Apply bias, scales and the output cast once after reduction.
        dest = c if split_k == 1 else torch.empty((batch * split_k, rows, n), device=a.device, dtype=torch.float32)
        with torch.cuda.device(a.device):
            _sparse_matmul[(tiles, split_k)](a, b, sparsity_metadata, xs, ws, bias, dest, gather, scatter, schedule,
                                             offsets, sizes, rows, n, k, m_tiles, k // group % 8, _strides(a, 3),
                                             _strides(b, 3), _strides(sparsity_metadata, 3), _strides(xs, 3),
                                             _strides(ws, 3), _strides(dest, 3), _strides(bias, 2),
                                             config.a_mx_tensor_scale, config.b_mx_tensor_scale, alpha, bk, nvfp4,
                                             scale_k, block_m, split_k, num_warps=4)
            if split_k > 1:
                _reduce_partials[(batch * triton.cdiv(rows * n, 64), )](dest, c, bias, config.a_mx_tensor_scale,
                                                                        config.b_mx_tensor_scale, alpha, rows, n,
                                                                        _strides(c, 3), _strides(bias, 2), split_k, 64,
                                                                        num_warps=4)
    return c
