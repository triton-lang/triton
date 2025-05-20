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
def _matmul_metadata_memset(Hist, n_expts_tot, MDHist, MDTokStarts, MDTileStarts, MDTileSum,
                            BLOCK: tl.constexpr, TILE_DIM: tl.constexpr):
    pid = tl.program_id(0)

    TileInfoOut = MDTileSum + pid * BLOCK + tl.arange(0, BLOCK)

    # if pid == 0 - initialize cumsums
    if pid == 0:
        x_tok = tl.zeros([BLOCK], dtype=MDTokStarts.dtype.element_ty)
        x_tile = tl.zeros([BLOCK], dtype=MDTileStarts.dtype.element_ty)
        tl.store(MDTokStarts, 0)
        tl.store(MDTileStarts, 0)
        for i in range(0, n_expts_tot, BLOCK):
            offs_n = tl.arange(0, BLOCK) + i
            mask = offs_n < n_expts_tot
            hist_tok = tl.load(Hist + offs_n, mask=mask, other=0)
            hist_tile = tl.cdiv(hist_tok, TILE_DIM)
            tok_starts = tl.cumsum(hist_tok, 0) + x_tok
            x_tok += tl.sum(hist_tok, 0).to(MDTokStarts.dtype.element_ty)
            tile_starts = tl.cumsum(hist_tile, 0) + x_tile
            x_tile += tl.sum(hist_tile, 0).to(MDTileStarts.dtype.element_ty)

            tl.store(MDHist + offs_n, hist_tok)
            tl.store(MDTokStarts + offs_n, tok_starts - hist_tok)
            tl.store(MDTileStarts + offs_n, tile_starts - hist_tile)

        tl.store(TileInfoOut, x_tile)

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
    pids = cdiv(grid_m, MEMSET_BLOCK) + 1

    metadata_size = 3 * n_expts_pad + MEMSET_BLOCK * pids

    metadata = torch.empty(metadata_size, dtype=torch.int32, device=device)

    md_hist = metadata[:n_expts_tot]
    md_offs = metadata[n_expts_pad:][:n_expts_tot]
    md_offs_sum = metadata[n_expts_pad*3]
    md_tile_starts = metadata[n_expts_pad*2:][:n_expts_tot]
    md_tile_infos = metadata[n_expts_pad*3 + MEMSET_BLOCK:][:grid_m]
    _matmul_metadata_memset[(pids, )](
        routing_data.expt_hist, n_expts_tot, md_hist, md_offs, md_tile_starts, md_offs_sum,
        BLOCK=MEMSET_BLOCK,  # optimization parameters
        TILE_DIM=block_m,  # constants
        num_warps=1
    )
    _matmul_metadata_compute[(n_expts_tot, )](
        routing_data.expt_hist, md_tile_starts, md_tile_infos,  # outputs
        BLOCK=HIST2_BLOCK_M,  # optimization parameters
        TILE_DIM=block_m,  # constants
    )
    return ExptData(md_hist, md_offs, md_offs_sum, md_tile_infos, metadata)
