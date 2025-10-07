from dataclasses import dataclass
import triton
import torch
import triton.language as tl
from ..target_info import is_hip

# ---------------------------------------------------------------------------- #
# metadata
# ---------------------------------------------------------------------------- #


@dataclass
class RaggedTensorMetadata:
    """
    Example:
    `batch_sizes`= [15 17 0 127]
    `batch_offs`= [0 15 32 32 332]
    `block_offs_data` = {
        16: [0 1 3 3 11]
        32: [0 1 2 2 6]
        64: [0 1 2 2 4]
        128: [0 1 2 2 3]
    }
    `block_schedule_data` = {
        16:  [(0, 0) (0, 1) (0, 3) (1, 3) (2, 3) ... (7, 3) -1 ... -1]
        32:  [(0, 0) (0, 1) (0, 3) (1, 3) (2, 3) (3, 3) -1 ...     -1]
        64:  [(0, 0) (0, 1) (0, 3) (1, 3) (2, 3) -1 ...            -1]
        128: [(0, 0) (0, 1) (0, 3) (1, 3) -1 ...                   -1]
    }
    """
    # batch_sizes[i] is the number of tokens in batch i
    batch_sizes: torch.Tensor
    # batch_offs = [0] + cumsum(batch_sizes)
    # i.e., batch_offs[i] is the offset of the first token for
    # batch `i` in a `batch_sizes`-shaped ragged tensor
    batch_offs: torch.Tensor
    # block_offs_data[k] = [0] + cumsum(ceil_div(batch_sizes, 16 * k))
    # i.e., `block_offs_data[k][i]` is the offset of the first block of
    # `16*k`` token for batch `i` in a `bath_sizes`-shaped ragged tensor
    block_offs_data: torch.Tensor
    # let `num_blocks[k] = block_offs_data[k, 1:] - block_offs_data[k, :-1]
    # block_schedule_data[k] = cat(*[[(batch, blk) for blk in range(blks)] for batch, blks in enumerate(num_blocks)])
    # i.e., if the schedule of batch `i` is [(i, 0), (i, 1), ..., (i, num_blocks[k][i] - 1)]
    # then `block_schedule_data[k]` is the concatenation of the schedules for all batches
    # NOTE 1: `block_schedule_data[k][j]` is a packed 32-bit integer
    # NOTE 2: because the size of `block_schedule_data[k]` is data-dependent, we pad it with -1s
    # up to an user-provided upper bound
    block_schedule_data: torch.Tensor

    def __post_init__(self):
        assert self.block_offs_data.shape[0] == len(RaggedTensorMetadata.block_sizes())
        assert self.block_schedule_data.shape[0] == len(RaggedTensorMetadata.block_sizes())
        assert self.block_offs_data.dtype == torch.int32
        assert self.block_schedule_data.dtype == torch.int32
        if self.batch_sizes is not None:
            assert self.batch_sizes.dtype == torch.int32
        if self.batch_offs is not None:
            assert self.batch_offs.dtype == torch.int32

    def block_offs(self, block_size):
        return self.block_offs_data[RaggedTensorMetadata.block_sizes().index(block_size)]

    def block_schedule(self, block_size):
        return self.block_schedule_data[RaggedTensorMetadata.block_sizes().index(block_size)]

    @staticmethod
    def block_sizes_log2():
        return range(4, 9) if is_hip() else range(4, 8)

    @staticmethod
    def block_sizes():
        return [2**x for x in RaggedTensorMetadata.block_sizes_log2()]


# utilities
# --------------------------------------------------------- #


def exact_div(x, y):
    assert x % y == 0
    return x // y


def empty_aligned(shape, dtype, device, pad_size):
    cdiv = lambda x, y: (x + y - 1) // y
    pad = lambda x: cdiv(x, pad_size) * pad_size
    ret = torch.empty((*shape[:-1], pad(shape[-1])), dtype=dtype, device=device)
    ret_slices = (*[slice(None)] * (len(shape) - 1), slice(0, shape[-1]))
    return ret[ret_slices], ret.numel()


def max_n_tiles(n_batches, n_gates):
    if n_gates <= n_batches:
        return n_gates
    return n_batches - 1 - ((n_batches - n_gates - 1) // RaggedTensorMetadata.block_sizes()[0])


# triton implementation
# --------------------------------------------------------- #


@triton.jit
def _cdiv_pow2(n, log2_k):
    return (n + ((1 << log2_k) - 1)) >> log2_k


@triton.jit
def _ragged_tensor_metadata_memset(Hist, n_batches, MDStarts, tile_starts_stridem, MDTileInfo, first_tile_dim_log2,
                                   SIZES: tl.constexpr, BLOCK: tl.constexpr):

    pid = tl.program_id(0)

    if pid <= SIZES:

        MDStarts += pid * tile_starts_stridem
        x_tile = tl.zeros([BLOCK], dtype=MDStarts.dtype.element_ty)
        Tile_ptrs = MDStarts + tl.arange(0, BLOCK)
        tile_dim_log2 = tl.where(pid == 0, 0, pid + first_tile_dim_log2 - 1)

        for i in range(0, n_batches + 1, BLOCK):

            offs_n = tl.arange(0, BLOCK) + i
            mask_n0 = offs_n < n_batches
            hist_tok = tl.load(Hist + offs_n, mask=mask_n0, other=0)
            hist_tile = _cdiv_pow2(hist_tok, tile_dim_log2)

            tile_starts = tl.cumsum(hist_tile, 0) + x_tile
            x_tile += tl.sum(hist_tile, 0).to(MDStarts.dtype.element_ty)
            tl.store(Tile_ptrs, tile_starts - hist_tile)
            Tile_ptrs += BLOCK

    else:

        pid -= (SIZES + 1)
        TileInfoOut = MDTileInfo + pid * BLOCK + tl.arange(0, BLOCK)
        tl.store(TileInfoOut, 0xffffffff)


@triton.jit
def _ragged_tensor_metadata_compute(Hist, MDTileStarts, tile_starts_stridem, MDTileInfo, tile_info_stridem,
                                    first_tile_dim_log2, SIZES: tl.constexpr, BLOCK: tl.constexpr):

    pid = tl.program_id(0)

    expt_id = pid // SIZES
    buff_id = pid % SIZES

    MDTileStarts += buff_id * tile_starts_stridem
    MDTileInfo += buff_id * tile_info_stridem

    n_tokens = tl.load(Hist + expt_id)
    tile_dim_log2 = first_tile_dim_log2 + buff_id
    n_blocks = _cdiv_pow2(n_tokens, tile_dim_log2)

    tile_off = tl.load(MDTileStarts + expt_id)
    MDTileInfo += tile_off

    for block_off in range(0, n_blocks, BLOCK):
        block_offs = block_off + tl.arange(0, BLOCK)
        data = (block_offs << 16) + expt_id
        tl.store(MDTileInfo + block_offs, data, mask=block_offs < n_blocks)


def make_ragged_tensor_metadata(batch_sizes, n_gates):
    assert batch_sizes.ndim == 1
    n_batches = batch_sizes.shape[0]
    block_sizes_log2 = RaggedTensorMetadata.block_sizes_log2()
    block_size_num = len(block_sizes_log2)
    MEMSET_BLOCK = 512
    dtype = torch.int32
    device = batch_sizes.device
    batch_offs_combined, _ = empty_aligned((block_size_num + 1, n_batches + 1), dtype, device, MEMSET_BLOCK)
    block_schedule_data, n_memset_elts = empty_aligned((block_size_num, max_n_tiles(n_batches, n_gates)), dtype, device,
                                                       MEMSET_BLOCK)
    batch_offs, block_offs_data = batch_offs_combined[0], batch_offs_combined[1:]
    n_memset_blocks = exact_div(n_memset_elts, MEMSET_BLOCK)

    _ragged_tensor_metadata_memset[(batch_offs_combined.shape[0] + n_memset_blocks, )](
        batch_sizes, n_batches,  #
        batch_offs_combined, batch_offs_combined.stride(0),  #
        block_schedule_data,  #
        block_sizes_log2[0], SIZES=len(block_sizes_log2), BLOCK=MEMSET_BLOCK,  # optimization parameters
        num_warps=4)

    _ragged_tensor_metadata_compute[(block_size_num * n_batches, )](
        batch_sizes, block_offs_data, block_offs_data.stride(0), block_schedule_data,
        block_schedule_data.stride(0),  # outputs
        block_sizes_log2[0], SIZES=len(block_sizes_log2), BLOCK=512,  # optimization parameters
        num_warps=4)

    return RaggedTensorMetadata(batch_sizes, batch_offs, block_offs_data, block_schedule_data)


# reference implementation
# --------------------------------------------------------- #


def make_ragged_tensor_metadata_torch(batch_sizes, n_gates):
    assert batch_sizes.ndim == 1
    n_batches = batch_sizes.shape[0]
    # offset for each experts
    device = batch_sizes.device
    batch_offs = torch.cumsum(batch_sizes, dim=0)
    batch_offs = torch.cat((torch.zeros(1, device=device), batch_offs))
    batch_offs = batch_offs.int()
    # maximum number of tiles for all values of `block_size` considered
    if n_gates <= n_batches:
        max_n_tiles = n_gates
    else:
        # ceil_div(n_gates - n_experts + 1, d_tile) + n_experts - 1
        # ceil_div(x, y): -(-x // y)
        max_n_tiles = n_batches - 1 - ((n_batches - n_gates - 1) // min(RaggedTensorMetadata.block_sizes()))
    # fill up tile offset/infos for each block
    col = torch.arange(max_n_tiles, device=device)
    batch_vals = torch.arange(n_batches, device=device)[:, None]

    def _build_schedule(block_off, n_tiles):
        total_tiles = int(block_off[-1].item())
        out = -torch.ones(max_n_tiles, dtype=torch.int32, device=device)
        if total_tiles == 0:
            return out
        tmp = -torch.ones(total_tiles, dtype=torch.int32, device=device)
        map_idxs = block_off[:-1, None] + col[None, :]
        mask = col[None, :] < n_tiles[:, None]
        tmp.index_put_((map_idxs[mask], ), (batch_vals + (col << 16)[None, :]).int()[mask])
        take = min(max_n_tiles, total_tiles)
        out[:take] = tmp[:take]
        return out

    block_offs = dict()
    block_pid_map = dict()
    for block_size in RaggedTensorMetadata.block_sizes():
        n_tiles = (batch_sizes + block_size - 1) // block_size
        block = torch.cumsum(n_tiles, dim=0)
        block = torch.cat((torch.zeros(1, device=device), block)).int()
        block_offs[block_size] = block
        block_pid_map[block_size] = _build_schedule(block, n_tiles)
    block_offs = torch.stack(list(block_offs.values()))
    block_pid_map = torch.stack(list(block_pid_map.values()))
    return RaggedTensorMetadata(batch_sizes, batch_offs, block_offs, block_pid_map)
