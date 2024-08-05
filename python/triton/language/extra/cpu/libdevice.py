from triton.language import core


@core.extern
def acos(arg0, _builder=None):
    return core.tensor(_builder.create_acos(arg0.handle), arg0.type)


@core.extern
def acosh(arg0, _builder=None):
    return core.tensor(_builder.create_acosh(arg0.handle), arg0.type)


@core.extern
def asin(arg0, _builder=None):
    return core.tensor(_builder.create_asin(arg0.handle), arg0.type)


@core.extern
def asinh(arg0, _builder=None):
    return core.tensor(_builder.create_asinh(arg0.handle), arg0.type)


@core.extern
def atan(arg0, _builder=None):
    return core.tensor(_builder.create_atan(arg0.handle), arg0.type)


@core.extern
def atanh(arg0, _builder=None):
    return core.tensor(_builder.create_atanh(arg0.handle), arg0.type)


@core.extern
def cbrt(arg0, _builder=None):
    return core.tensor(_builder.create_cbrt(arg0.handle), arg0.type)


@core.extern
def cos(arg0, _builder=None):
    return core.tensor(_builder.create_cos(arg0.handle), arg0.type)


@core.extern
def cosh(arg0, _builder=None):
    return core.tensor(_builder.create_cosh(arg0.handle), arg0.type)


@core.extern
def erf(arg0, _builder=None):
    return core.tensor(_builder.create_erf(arg0.handle), arg0.type)


@core.extern
def exp(arg0, _builder=None):
    return core.tensor(_builder.create_exp(arg0.handle), arg0.type)


@core.extern
def exp2(arg0, _builder=None):
    return core.tensor(_builder.create_exp2(arg0.handle), arg0.type)


@core.extern
def log(arg0, _builder=None):
    return core.tensor(_builder.create_log(arg0.handle), arg0.type)


@core.extern
def log2(arg0, _builder=None):
    return core.tensor(_builder.create_log2(arg0.handle), arg0.type)


@core.extern
def log10(arg0, _builder=None):
    return core.tensor(_builder.create_log10(arg0.handle), arg0.type)


@core.extern
def sin(arg0, _builder=None):
    return core.tensor(_builder.create_sin(arg0.handle), arg0.type)


@core.extern
def sinh(arg0, _builder=None):
    return core.tensor(_builder.create_sinh(arg0.handle), arg0.type)


@core.extern
def tan(arg0, _builder=None):
    return core.tensor(_builder.create_tan(arg0.handle), arg0.type)


@core.extern
def tanh(arg0, _builder=None):
    return core.tensor(_builder.create_tanh(arg0.handle), arg0.type)
