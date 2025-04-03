from triton.language import core


@core.extern
def gdc_wait(_builder=None):
    return core.inline_asm_elementwise("griddepcontrol.wait; // dummy $0", "=r", [], dtype=core.int32, is_pure=False,
                                       pack=1, _builder=_builder)


@core.extern
def gdc_launch_dependents(_builder=None):
    return core.inline_asm_elementwise("griddepcontrol.launch_dependents; // dummy $0", "=r", [], dtype=core.int32,
                                       is_pure=True, pack=1, _builder=_builder)
