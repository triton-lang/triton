from . import core


# TODO pawel: handle non f32 types for all of the new functions
@core.extern
def exp2(arg0, _builder=None):
    arg0 = core._to_tensor(arg0, _builder)
    return core.tensor(_builder.create_exp2(arg0.handle), arg0.type)


@core.extern
def div_rn(arg0, arg1, _builder=None):
    arg0 = core._to_tensor(arg0, _builder)
    arg1 = core._to_tensor(arg1, _builder)
    return core.tensor(_builder.create_precise_divf(arg0.handle, arg1.handle), arg0.type)


@core.extern
def sqrt_rn(arg0, _builder=None):
    arg0 = core._to_tensor(arg0, _builder)
    return core.tensor(_builder.create_precise_sqrt(arg0.handle), arg0.type)


@core.extern
def log2(arg0, _builder=None):
    arg0 = core._to_tensor(arg0, _builder)
    return core.tensor(_builder.create_log2(arg0.handle), arg0.type)


@core.extern
def erf(arg0, _builder=None):
    arg0 = core._to_tensor(arg0, _builder)
    return core.tensor(_builder.create_erf(arg0.handle), arg0.type)
