from triton.language import core


@core.extern
def memrealtime(_semantic=None):
    """
    Returns a 64-bit real time-counter value
    """
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
