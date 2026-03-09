from triton.language import core


@core.extern
def memrealtime(_semantic=None):
    """
    Returns a 64-bit real time-counter value
    """
    target_arch = _semantic.builder.options.arch
    asm_str = """s_memrealtime $0
                 s_waitcnt vmcnt(0)"""
    if 'gfx11' in target_arch:
        asm_str = """s_sendmsg_rtn_b64 $0, sendmsg(MSG_RTN_GET_REALTIME)
                     s_waitcnt lgkmcnt(0)"""
    elif 'gfx12' in target_arch:
        asm_str = """s_sendmsg_rtn_b64 $0, sendmsg(MSG_RTN_GET_REALTIME)
                     s_wait_kmcnt 0"""
    return core.inline_asm_elementwise(
        asm_str,
        "=r",
        [],
        dtype=core.int64,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def smid(_semantic=None):
    """
    Returns the compute unit / workgroup processor ID for the current wave.

    GCN/CDNA (gfx9xx): reads CU_ID, SH_ID, and SE_ID fields from HW_REG_HW_ID
    (register 4) as a packed value.  On multi-XCC parts (gfx942/gfx950) the
    XCC_ID is NOT included; values are unique only within a single XCC.

    RDNA (gfx10xx/gfx11xx/gfx12xx): reads WGP_ID from HW_REG_HW_ID1
    (register 23).  Values are unique only within a shader array.
    """
    target_arch = _semantic.builder.options.arch
    if 'gfx9' in target_arch:
        # HW_REG_HW_ID (reg 4), bits [15:8]:
        #   [11:8]  CU_ID  (4 bits)
        #   [12]    SH_ID  (1 bit)
        #   [15:13] SE_ID  (2-3 bits depending on chip)
        asm_str = "s_getreg_b32 $0, hwreg(4, 8, 8)"
    elif 'gfx10' in target_arch or 'gfx11' in target_arch or 'gfx12' in target_arch:
        # HW_REG_HW_ID1 (reg 23), bits [13:10]: WGP_ID (4 bits)
        asm_str = "s_getreg_b32 $0, hwreg(23, 10, 4)"
    else:
        raise ValueError(f"smid is not supported on {target_arch}")
    return core.inline_asm_elementwise(
        asm_str,
        "=r",
        [],
        dtype=core.int32,
        is_pure=True,
        pack=1,
        _semantic=_semantic,
    )


@core.builtin
def num_threads(_semantic=None):
    opts = _semantic.builder.options
    return core.constexpr(opts.num_warps * opts.warp_size)


@core.builtin
def num_warps(_semantic=None):
    return core.constexpr(_semantic.builder.options.num_warps)
