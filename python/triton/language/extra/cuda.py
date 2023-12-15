from .. import core


@core.extern
def globaltimer(_builder=None):
    return core.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=core.int64, is_pure=False, pack=1,
                                       _builder=_builder)


@core.extern
def smid(_builder=None):
    return core.inline_asm_elementwise("mov.u32 $0, %smid;", "=r", [], dtype=core.int32, is_pure=True, pack=1,
                                       _builder=_builder)


@core.builtin
def num_threads(_builder=None):
    return core.constexpr(_builder.target.num_warps * 32)
