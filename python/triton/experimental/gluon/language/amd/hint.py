from .._core import builtin, _unwrap_if_constexpr

__all__ = ["disable_xdl_arb_stall", "sched_barrier"]


@builtin
def disable_xdl_arb_stall(_semantic=None):
    """Allow a gfx1250 wave to issue independent WMMAs back-to-back."""
    return _semantic.inline_asm(
        "s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1",
        "",
        (),
        (),
        False,
    )


@builtin
def sched_barrier(allow=None, _semantic=None):
    """Insert a barrier that constrains LLVM instruction scheduling.

    The barrier prevents the LLVM compiler's instruction scheduler from moving
    instructions across this point. It is a LLVM compiler hint only.

    Args:
        allow (str or tuple[str, ...], optional): Instruction classes allowed to
            cross the barrier. None means that no instruction classes may cross.
            Currently only None is allowed.
    """
    allow = _unwrap_if_constexpr(allow)
    if allow is None or isinstance(allow, str):
        allow = (allow, )

    # map instruction classes to their corresponding masks
    instruction_masks = {None: 0}
    mask = 0
    for instruction in allow:
        instruction = _unwrap_if_constexpr(instruction)
        if instruction is not None and not isinstance(instruction, str):
            raise TypeError("instruction classes must be strings")
        if instruction not in instruction_masks:
            raise ValueError(f"Unsupported instruction class: {instruction!r}")
        mask |= instruction_masks[instruction]

    _semantic.builder.create_amd_sched_barrier(mask)
