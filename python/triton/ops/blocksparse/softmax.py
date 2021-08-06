from functools import lru_cache
from typing import Tuple

import pytest
import torch
import triton
import triton.language as tl


def next_power_of_2(n):
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    return n


def num_warps(n):
    if n < 512:
        return 4
    if n < 2048:
        return 8
    return 16


@triton.heuristics(
    {"num_warps": lambda *args, **meta: num_warps(args[5] * meta["SPARSE_BLOCK_SIZE"])}
)
@triton.heuristics(
    {
        "MAX_N_COLS": lambda *args, **meta: next_power_of_2(
            args[5] * meta["SPARSE_BLOCK_SIZE"]
        )
    }
)
@triton.jit
def softmax_forward_kernel(
    # Pointer to a tensor that is (batch, n_heads, M, N) and we want to compute the
    # softmax across the M dimension
    input_ptr,
    # Scale to apply to the logits before softmax. This can be helpful because the softmax
    # is computed in fp32 which prevents overflow or underflow when scaling
    scale,
    index_mapping_ptr,  # Pointer to precomputed mapping from block ids to indices
    key_padding_mask,
    attention_mask,
    max_size,  # Used by triton heuristics decorator to set block size and num_warps
    stride_batch_x,
    stride_batch_key_padding_mask,
    stride_batch_attn_mask,
    **meta,
):
    MAX_N_COLS = meta["MAX_N_COLS"]
    SPARSE_BLOCK_SIZE = meta["SPARSE_BLOCK_SIZE"]
    # TODO: Understand and rename everything from here to rbmn
    pid_row = tl.program_id(0)
    pid_batch = tl.program_id(1)
    # create index ranges
    row_in_sparse_block = pid_row % SPARSE_BLOCK_SIZE
    sparse_block_id = pid_row // SPARSE_BLOCK_SIZE
    range_x_cols = tl.arange(0, MAX_N_COLS) % SPARSE_BLOCK_SIZE
    range_block_cols = tl.arange(0, MAX_N_COLS) // SPARSE_BLOCK_SIZE
    # We have our block id, but we now read the header to convert that to indices
    # in the original data.
    # The header is laid out as (size_0, offset_0, size_1, offset_1, ...), this give us
    # the offsets of ther id in the index mapping and the length of each row in blocks
    header = index_mapping_ptr + sparse_block_id * 2
    row_len = tl.load(header + 0)
    offset = tl.load(header + 1)
    check = range_block_cols < row_len
    rbmn = tl.where(check, range_block_cols, row_len - 1)

    block_id = tl.load(index_mapping_ptr + offset + rbmn * 4 + 0)
    column_id = tl.load(index_mapping_ptr + offset + rbmn * 4 + 1)
    row_id = tl.load(index_mapping_ptr + offset + rbmn * 4 + 2)
    input_ptrs = (
        input_ptr
        # First advance to the relevant batch
        + pid_batch * stride_batch_x
        # Then advance to the relevant block
        + block_id * SPARSE_BLOCK_SIZE * SPARSE_BLOCK_SIZE
        # Then advance to the row we are processing
        + row_in_sparse_block * SPARSE_BLOCK_SIZE
        # Then get a range starting from there and going for the max number of columns
        # We will mask out the extra pointers if it doesnt have the max columns
        + range_x_cols
    )
    inputs = tl.load(input_ptrs, mask=check, other=-float("inf"))
    # This was a suggestion from Philippe. We are unsure if it improves performance
    # or not. Seems inconsistent. It may need to go in another place.
    inputs = tl.multiple_of(inputs, 8)  # compiler hint
    inputs = inputs.to(tl.float32)
    # TODO: See if this needs to be hidden behind a flag
    if meta["APPLY_SCALE"]:
        inputs = inputs * scale
    if meta["APPLY_KP_MASK"]:
        key_padding_mask_ptr = (
            key_padding_mask
            + pid_batch * stride_batch_key_padding_mask
            + column_id * SPARSE_BLOCK_SIZE
            + range_x_cols
        )
        key_padding_mask = tl.load(
            key_padding_mask_ptr, mask=check, other=-float("inf")
        )
        if meta["KP_MASK_MUL"]:
            key_padding_mask = tl.where(key_padding_mask == 0, -float("inf"), 0.0)
        inputs = inputs + key_padding_mask
    # apply attention mask
    if meta["APPLY_ATTN_MASK"]:
        attn_mask_ptr = (
            attention_mask
            + column_id * SPARSE_BLOCK_SIZE
            + row_id * SPARSE_BLOCK_SIZE * stride_batch_attn_mask
            + row_in_sparse_block * stride_batch_attn_mask
            + range_x_cols
        )
        attn_m = tl.load(attn_mask_ptr, mask=check, other=-float("inf"))
        # TODO: See if this (and kp_mask) are ever off. Would be simpler to just require
        # that the mask is a bool mask.
        if meta["ATTN_MASK_MUL"]:
            attn_m = tl.where(attn_m == 0, -float("inf"), 0.0)
        inputs = inputs + attn_m
    # computation
    probs = tl.softmax(inputs)
    # Op is in-place, we write back to the same memory location
    tl.store(input_ptrs, probs, mask=check)


@triton.heuristics(
    {"num_warps": lambda *args, **meta: num_warps(args[4] * meta["BLOCK"])}
)
@triton.heuristics(
    {"TN": lambda *args, **meta: next_power_of_2(args[4]) * meta["BLOCK"]}
)
@triton.jit
def softmax_backward_kernel(X, scale, DX, LUT, sizemax, stride_zx, stride_zdx, **meta):
    pidhm = tl.program_id(0)
    pidz = tl.program_id(1)
    TN = meta["TN"]
    BLOCK = meta["BLOCK"]
    # create index ranges
    rxm = pidhm % BLOCK
    rbm = pidhm // BLOCK
    rxn = tl.arange(0, TN) % BLOCK
    rbn = tl.arange(0, TN) // BLOCK
    # extract information from look-up table
    header = LUT + rbm * 2
    size = tl.load(header + 0)
    offset = tl.load(header + 1)
    # bounds checking on lut
    check = rbn < size
    rbmn = tl.where(check, rbn, size - 1)
    # initialize pointers to block-sparse input
    blockid = tl.load(LUT + offset + rbmn * 4)
    X = X + pidz * stride_zx + blockid * BLOCK * BLOCK + rxm * BLOCK + rxn
    DX = DX + pidz * stride_zdx + blockid * BLOCK * BLOCK + rxm * BLOCK + rxn
    # compute fused softmax backward
    x = tl.load(X, mask=check, other=0)
    dx = tl.load(DX, mask=check, other=0)
    x = x.to(tl.float32)
    dx = dx.to(tl.float32)
    y = x * (dx - tl.sum(x * dx, 0)) * scale
    tl.store(DX, y, mask=check)


class BlocksparseSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        scale,
        key_padding_mask,
        attn_mask,
        kp_mask_mode,
        attn_mask_mode,
        sparse_block_size,
        lut,
        maxlut,
        n_heads,
        n_rows,
    ):

        # handle None key_padding_mask
        if key_padding_mask is None:
            stride_zkpm = 0
            key_padding_mask = torch.empty(0, dtype=x.dtype, device=x.device)
        else:
            stride_zkpm = key_padding_mask.stride(0)

        # handle None attention_mask
        if attn_mask is None:
            stride_zattnm = 0
            attn_mask = torch.empty(0, dtype=x.dtype, device=x.device)
        else:
            stride_zattnm = attn_mask.stride(0)

        # run kernel
        n_batch, n_blocks, block_size_1, block_size_2 = x.shape
        assert sparse_block_size == block_size_1 == block_size_2
        meta = {
            "SPARSE_BLOCK_SIZE": sparse_block_size,
            "APPLY_SCALE": scale != 1.0,
            "APPLY_KP_MASK": key_padding_mask is not None,
            "APPLY_ATTN_MASK": attn_mask is not None,
            "KP_MASK_MUL": kp_mask_mode == "mul",
            "ATTN_MASK_MUL": attn_mask_mode == "mul",
        }
        grid = lambda opt: [n_heads * n_rows, n_batch]
        softmax_forward_kernel[grid](
            x,
            scale,
            lut,
            key_padding_mask,
            attn_mask,
            maxlut,
            x.stride(0),
            stride_zkpm,
            stride_zattnm,
            force_nc_cache=True,
            **meta,
        )

        # save to context
        ctx.mark_dirty(x)
        ctx.save_for_backward(x, lut)
        ctx.sparse_block_size = sparse_block_size
        ctx.n_heads = n_heads
        ctx.n_rows = n_rows
        ctx.maxlut = maxlut
        ctx.scale = scale
        ctx.kp_mask_mode = kp_mask_mode
        ctx.attn_mask_mode = attn_mask_mode
        return x

    @staticmethod
    def backward(ctx, dx):
        # retrieve from context
        x, lut = ctx.saved_tensors
        # run kernel
        M = x.shape[0]
        grid = lambda opt: [ctx.n_heads * ctx.n_rows, M]
        softmax_backward_kernel[grid](
            x,
            ctx.scale,
            dx,
            lut,
            ctx.maxlut,
            x.stride(0),
            dx.stride(0),
            force_nc_cache=True,
            BLOCK=ctx.sparse_block_size,
        )
        return (
            dx,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class BlocksparseSoftmax:
    @lru_cache
    def make_index_mapping(self):
        return make_index_mapping(self.block_mask)

    def __init__(self, block_mask: torch.Tensor, block_size: int):
        self.block_mask = block_mask
        self.block_size = block_size

    def __call__(
        self,
        x,
        scale=1.0,
        key_padding_mask=None,
        attn_mask=None,
        key_padding_mask_mode="add",
        attn_mask_mode="add",
    ):
        if attn_mask is not None and attn_mask.dtype != x.dtype:
            raise ValueError("Attention mask must be %s" % x.dtype)
        if key_padding_mask is not None and key_padding_mask.dtype != x.dtype:
            raise ValueError("Key padding mask must be %s" % x.dtype)
        lut, max_lut = self.make_index_mapping()
        n_heads = self.block_mask.shape[0]
        n_rows = self.block_mask.shape[1] * self.block_size
        x = BlocksparseSoftmaxFunction.apply(
            x,
            scale,
            key_padding_mask,
            attn_mask,
            key_padding_mask_mode,
            attn_mask_mode,
            self.block_size,
            lut,
            max_lut,
            n_heads,
            n_rows,
        )
        return x


def make_index_mapping(block_mask: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Cache some information for lookup in the kernel. This is cached into a flat
    tensor so it can easily be read in Triton.

    Specifically, we cache a few pieces of information:
    1. block_idx: which block we are processing. This is just an arange
    2. indices for the nonzero blocks. These are flattened out to being
        all the column indices then all the row indices then all the head indices.
    3. TODO: Figure out what the header is

    :param block_mask: The block mask. Has shape (n_heads, n_blocks, n_blocks)
    :returns: Tuple of a tensor containing

    """
    # block indices
    block_idxs = torch.arange(block_mask.sum())
    head, rows, columns = block_mask.nonzero().split(1, dim=1)
    # Interleave indices as [0, col_0, row_0, head_0, 1, col_1, row_1, head_1, ...]
    core = torch.stack(
        (block_idxs, columns[:, 0], rows[:, 0], head[:, 0]), dim=1
    ).flatten()

    n_blocks_per_col = block_mask.sum(dim=-1).flatten()
    block_offsets = torch.zeros_like(n_blocks_per_col)
    block_offsets[1:] = torch.cumsum(n_blocks_per_col[:-1], dim=0)

    # We are computing offsets in this tensor, so we need to account for its header
    header_offset = 2 * n_blocks_per_col.numel()
    # Each block has a block_id, col_id, row_id, and head_id stacked together so
    # we  multiply by 4
    block_offsets = block_offsets * 4 + header_offset
    header = torch.stack((n_blocks_per_col, block_offsets), dim=1).flatten()
    tensor = torch.cat((header, core)).type(torch.int32).cuda()
    return tensor, int(n_blocks_per_col.max())
