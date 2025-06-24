import triton
import triton.language as tl


@triton.jit
def _cdiv_pow2(n, log2_k):
    return (n + ((1 << log2_k) - 1)) >> log2_k


@triton.jit
def _expt_data_memset(Hist, n_expts_tot, MDTokStarts, MDTileStarts, tile_starts_stridem, MDTileInfo, tile_infos_stridem,
                      first_tile_dim_log2, BLOCK: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    tile_dim_log2 = first_tile_dim_log2 + pid_m
    # if pid == 0 - initialize cumsums
    if pid_n == 0:
        MDTileStarts += pid_m * tile_starts_stridem

        x_tok = tl.zeros([BLOCK], dtype=MDTokStarts.dtype.element_ty)
        x_tile = tl.zeros([BLOCK], dtype=MDTileStarts.dtype.element_ty)

        Tok_ptrs = MDTokStarts + tl.arange(0, BLOCK)
        Tile_ptrs = MDTileStarts + tl.arange(0, BLOCK)

        for i in range(0, n_expts_tot + 1, BLOCK):
            offs_n = tl.arange(0, BLOCK) + i
            mask_n0 = offs_n < n_expts_tot
            mask_n1 = offs_n < n_expts_tot + 1
            hist_tok = tl.load(Hist + offs_n, mask=mask_n0, other=0)
            hist_tile = _cdiv_pow2(hist_tok, tile_dim_log2)
            tok_starts = tl.cumsum(hist_tok, 0) + x_tok
            x_tok += tl.sum(hist_tok, 0).to(MDTokStarts.dtype.element_ty)
            tile_starts = tl.cumsum(hist_tile, 0) + x_tile
            x_tile += tl.sum(hist_tile, 0).to(MDTileStarts.dtype.element_ty)

            tl.store(Tok_ptrs, tok_starts - hist_tok, mask=mask_n1)
            tl.store(Tile_ptrs, tile_starts - hist_tile, mask=mask_n1)

            Tok_ptrs += BLOCK
            Tile_ptrs += BLOCK

    else:
        MDTileInfo += pid_m * tile_infos_stridem
        TileInfoOut = MDTileInfo + (pid_n - 1) * BLOCK + tl.arange(0, BLOCK)
        tl.store(TileInfoOut, 0xffffffff)


@triton.jit
def _expt_data_compute(Hist, MDTileStarts, tile_starts_stridem, MDTileInfo, tile_info_stridem, first_tile_dim_log2,
                       BLOCK: tl.constexpr):
    expt_id = tl.program_id(0)
    buff_id = tl.program_id(1)

    MDTileStarts += buff_id * tile_starts_stridem
    MDTileInfo += buff_id * tile_info_stridem

    n_tokens = tl.load(Hist + expt_id)
    tile_dim_log2 = first_tile_dim_log2 + buff_id
    n_blocks = _cdiv_pow2(n_tokens, tile_dim_log2)

    tile_off = tl.load(MDTileStarts + expt_id)
    MDTileInfo += tile_off
    # MDTileInfo += tl.load(MDTilesStart + expt_id)
    for block_off in range(0, n_blocks, BLOCK):
        block_offs = block_off + tl.arange(0, BLOCK)
        data = (block_offs << 16) + expt_id
        tl.store(MDTileInfo + block_offs, data, mask=block_offs < n_blocks)
