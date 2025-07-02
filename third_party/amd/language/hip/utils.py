from triton.language import core


@core.extern
def memrealtime(_semantic=None):
    """
    Returns a 64-bit real time-counter value
    """
    target_arch = _semantic.builder.options.arch
    if 'gfx11' in target_arch or 'gfx12' in target_arch:
        return core.inline_asm_elementwise(
            """
            s_sendmsg_rtn_b64 $0, sendmsg(MSG_RTN_GET_REALTIME)
            s_waitcnt lgkmcnt(0)
            """,
            "=r",
            [],
            dtype=core.int64,
            is_pure=False,
            pack=1,
            _semantic=_semantic,
        )
    else:
        return core.inline_asm_elementwise(
            """
            s_memrealtime $0
            s_waitcnt vmcnt(0)
            """,
            "=r",
            [],
            dtype=core.int64,
            is_pure=False,
            pack=1,
            _semantic=_semantic,
        )
