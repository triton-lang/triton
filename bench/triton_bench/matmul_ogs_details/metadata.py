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
def _memset_metadata(Metadata, metadata_size, Lock, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(Lock, 0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(Metadata + offs, 0xffffffff, mask=offs < metadata_size)


@triton.jit
def _compute_metadata(Hist, n_expts_tot, Lock, MDHist, MDTokensStart, MDTilesStart, MDTileInfo,
                      N_EXPTS_PAD: tl.constexpr, BLOCK: tl.constexpr, TILE_DIM: tl.constexpr):
    expt_id = tl.program_id(0)
    n_tokens = tl.load(Hist + expt_id)
    n_blocks = tl.cdiv(n_tokens, TILE_DIM)
    # first pid to reach this initializes histograms and cumsums
    if tl.atomic_cas(Lock, 0, 1) == 0:
        offs_n = tl.arange(0, N_EXPTS_PAD)
        mask = offs_n < n_expts_tot
        hist = tl.load(Hist + offs_n, mask=mask)
        tl.store(MDHist + offs_n, hist, mask=mask)
        tokens_start = tl.cumsum(hist, 0)
        tl.store(MDTokensStart, 0)
        tl.store(MDTokensStart + 1 + offs_n, tokens_start, mask=mask)
        tiles_start = tl.cumsum(tl.cdiv(hist, TILE_DIM), 0)
        tl.store(MDTilesStart, 0)
        tl.store(MDTilesStart + 1 + offs_n, tiles_start, mask=mask)
        tl.debug_barrier()
        tl.atomic_xchg(Lock, 0)
        tl.debug_barrier()
    # spin until content of `MDTilesStart` is initialized
    while tl.atomic_add(Lock, 0) == 1:
        pass
    MDTileInfo += tl.load(MDTilesStart + expt_id)
    for block_off in range(0, n_blocks, BLOCK):
        block_offs = block_off + tl.arange(0, BLOCK)
        data = (block_offs << 16) + expt_id
        tl.store(MDTileInfo + block_offs, data, mask=block_offs < n_blocks)


def compute_metadata(routing_data, n_rows, block_m):
    if routing_data.expt_hist is None:
        return ExptData(None, None, None, None, None)
    MEMSET_BLOCK = 512
    HIST2_BLOCK_M = 512
    device = routing_data.expt_hist.device
    n_expts_tot = routing_data.n_expts_tot
    cdiv = triton.cdiv
    if n_rows <= n_expts_tot:
        grid_m = n_rows
    else:
        grid_m = n_expts_tot - 1 - ((n_expts_tot - n_rows - 1) // block_m)
    n_expts_pad = cdiv(n_expts_tot, 128) * 128
    metadata_size = 3 * n_expts_tot + 2 + grid_m
    metadata = torch.empty(metadata_size, dtype=torch.int32, device=device)
    md_hist = metadata[:n_expts_tot]
    md_tok_starts = metadata[n_expts_tot:n_expts_tot * 2 + 1]
    md_tile_starts = metadata[n_expts_tot * 2 + 1:n_expts_tot * 3 + 2]
    md_tile_infos = metadata[n_expts_tot * 3 + 2:]
    lock = torch.empty((1, ), dtype=torch.int32, device=device)
    _memset_metadata[(cdiv(metadata_size, MEMSET_BLOCK), )](metadata, metadata_size, lock, BLOCK=MEMSET_BLOCK)
    _compute_metadata[(n_expts_tot, )](routing_data.expt_hist, n_expts_tot, lock, md_hist, md_tok_starts,
                                       md_tile_starts, md_tile_infos, N_EXPTS_PAD=n_expts_pad, BLOCK=HIST2_BLOCK_M,
                                       TILE_DIM=block_m)

    hist = metadata[:n_expts_tot]
    offs = metadata[n_expts_tot:2 * n_expts_tot + 1]
    offs_sum = metadata[3 * n_expts_tot + 2 - 1]
    blocks = metadata[n_expts_tot + 2 * (n_expts_tot + 1):]
    return ExptData(hist, offs, offs_sum, blocks, metadata)
