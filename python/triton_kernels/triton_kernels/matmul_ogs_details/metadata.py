from dataclasses import dataclass
import torch
import triton
import triton.language as tl


@dataclass
class ExptData:
    hist: torch.Tensor
    offs: torch.Tensor
    offs_sum: torch.Tensor
    blocks: torch.Tensor
    buffer: torch.Tensor


@triton.jit
def _matmul_metadata_memset(Hist, n_expts_tot, MDTokStarts, MDTileStarts, MDTileInfo, BLOCK: tl.constexpr,
                            TILE_DIM: tl.constexpr, extra_block: tl.constexpr):
    pid = tl.program_id(0)

    TileInfoOut = MDTileInfo + (pid - 1) * BLOCK + tl.arange(0, BLOCK)

    # if pid == 0 - initialize cumsums
    if pid == 0:
        x_tok = tl.zeros([BLOCK], dtype=MDTokStarts.dtype.element_ty)
        x_tile = tl.zeros([BLOCK], dtype=MDTileStarts.dtype.element_ty)

        Tok_ptrs = MDTokStarts + tl.arange(0, BLOCK)
        Tile_ptrs = MDTileStarts + tl.arange(0, BLOCK)

        for i in range(0, n_expts_tot, BLOCK):
            offs_n = tl.arange(0, BLOCK) + i
            if extra_block:
                # we need an extra block at the end just to contain the final
                # sum; this only happens if our total number of experts is an
                # exact multiple of BLOCK, obviating the need for any masking
                hist_tok = tl.load(Hist + offs_n)
            else:
                mask = offs_n < n_expts_tot
                hist_tok = tl.load(Hist + offs_n, mask=mask, other=0)
            hist_tile = tl.cdiv(hist_tok, TILE_DIM)
            tok_starts = tl.cumsum(hist_tok, 0) + x_tok
            x_tok += tl.sum(hist_tok, 0).to(MDTokStarts.dtype.element_ty)
            tile_starts = tl.cumsum(hist_tile, 0) + x_tile
            x_tile += tl.sum(hist_tile, 0).to(MDTileStarts.dtype.element_ty)

            tl.store(Tok_ptrs, tok_starts - hist_tok)
            tl.store(Tile_ptrs, tile_starts - hist_tile)

            Tok_ptrs += BLOCK
            Tile_ptrs += BLOCK

        if extra_block:
            tl.store(Tok_ptrs, x_tok)
            tl.store(Tile_ptrs, x_tile)

    else:

        tl.store(TileInfoOut, 0xffffffff)


@triton.jit
def _matmul_metadata_compute(Hist, MDTileStarts, MDTileInfo, BLOCK: tl.constexpr, TILE_DIM: tl.constexpr):

    expt_id = tl.program_id(0)
    n_tokens = tl.load(Hist + expt_id)
    n_blocks = tl.cdiv(n_tokens, TILE_DIM)

    tile_off = tl.load(MDTileStarts + expt_id)
    MDTileInfo += tile_off
    # MDTileInfo += tl.load(MDTilesStart + expt_id)
    for block_off in range(0, n_blocks, BLOCK):
        block_offs = block_off + tl.arange(0, BLOCK)
        data = (block_offs << 16) + expt_id
        tl.store(MDTileInfo + block_offs, data, mask=block_offs < n_blocks)


def compute_metadata(routing_data, n_rows, block_m):
    if routing_data.expt_hist is None:
        return ExptData(None, None, None, None, None)
    MEMSET_BLOCK = 128
    HIST2_BLOCK_M = 512
    device = routing_data.expt_hist.device
    n_expts_tot = routing_data.n_expts_tot
    cdiv = triton.cdiv
    if n_rows <= n_expts_tot:
        grid_m = n_rows
    else:
        grid_m = n_expts_tot - 1 - ((n_expts_tot - n_rows - 1) // block_m)

    n_expts_pad = cdiv(n_expts_tot, MEMSET_BLOCK) * MEMSET_BLOCK
    pad2 = cdiv(n_expts_tot + 1, MEMSET_BLOCK) * MEMSET_BLOCK
    extra_block = (n_expts_pad != pad2)
    pids = cdiv(grid_m, MEMSET_BLOCK) + 1

    metadata_size = n_expts_pad + 2 * pad2 + MEMSET_BLOCK * (pids - 1)

    metadata = torch.empty(metadata_size, dtype=torch.int32, device=device)

    md_hist = routing_data.expt_hist[:n_expts_tot]
    md_offs = metadata[:n_expts_tot + 1]
    md_tile_starts = metadata[pad2:][:n_expts_tot + 1]
    md_offs_sum = md_tile_starts[-1]
    md_tile_infos = metadata[2 * pad2:][:grid_m]
    _matmul_metadata_memset[(pids, )](
        routing_data.expt_hist, n_expts_tot, md_offs, md_tile_starts, md_tile_infos,
        BLOCK=MEMSET_BLOCK,  # optimization parameters
        TILE_DIM=block_m,  # constants
        extra_block=extra_block, num_warps=1)
    _matmul_metadata_compute[(n_expts_tot, )](
        routing_data.expt_hist, md_tile_starts, md_tile_infos,  # outputs
        BLOCK=HIST2_BLOCK_M,  # optimization parameters
        TILE_DIM=block_m,  # constants
        num_warps=4)
    return ExptData(md_hist, md_offs, md_offs_sum, md_tile_infos, metadata)
