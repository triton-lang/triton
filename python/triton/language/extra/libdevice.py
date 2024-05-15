from .cuda import libdevice as cuda_libdevice
from .hip import libdevice as hip_libdevice
from triton.language import core
from functools import wraps
from typing import TypeVar

T = TypeVar('T')


def dispatch(fn: T) -> T:
    """Dispatch a function to a correct implementation."""
    assert callable(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        _backend = kwargs["_builder"].options.backend_name
        if _backend == 'cuda':
            _curr_libdevice_module = cuda_libdevice
        elif _backend == 'hip':
            _curr_libdevice_module = hip_libdevice
        else:
            raise RuntimeError('unknown backend')

        try:
            _impl = getattr(_curr_libdevice_module, fn.__name__)
        except AttributeError:
            raise RuntimeError(f'`{_backend}` does not provide support for `{fn.__name__}` extra function')

        return _impl(*args, **kwargs)

    return wrapper


@core.extern
@dispatch
def clz(arg0, _builder=None):
    ...


@core.extern
@dispatch
def popc(arg0, _builder=None):
    ...


@core.extern
@dispatch
def byte_perm(arg0, arg1, arg2, _builder=None):
    ...


@core.extern
@dispatch
def mulhi(arg0, arg1, _builder=None):
    ...


@core.extern
@dispatch
def mul24(arg0, arg1, _builder=None):
    ...


@core.extern
@dispatch
def brev(arg0, _builder=None):
    ...


@core.extern
@dispatch
def sad(arg0, arg1, arg2, _builder=None):
    ...


@core.extern
@dispatch
def abs(arg0, _builder=None):
    ...


@core.extern
@dispatch
def floor(arg0, _builder=None):
    ...


@core.extern
@dispatch
def rcp64h(arg0, _builder=None):
    ...


@core.extern
@dispatch
def rsqrt(arg0, _builder=None):
    ...


@core.extern
@dispatch
def ceil(arg0, _builder=None):
    ...


@core.extern
@dispatch
def trunc(arg0, _builder=None):
    ...


@core.extern
@dispatch
def exp2(arg0, _builder=None):
    ...


@core.extern
@dispatch
def saturatef(arg0, _builder=None):
    ...


@core.extern
@dispatch
def fma_rn(arg0, arg1, arg2, _builder=None):
    ...


@core.extern
@dispatch
def fma_rz(arg0, arg1, arg2, _builder=None):
    ...


@core.extern
@dispatch
def fma_rd(arg0, arg1, arg2, _builder=None):
    ...


@core.extern
@dispatch
def fma_ru(arg0, arg1, arg2, _builder=None):
    ...


@core.extern
@dispatch
def fast_dividef(arg0, arg1, _builder=None):
    ...


@core.extern
@dispatch
def div_rn(arg0, arg1, _builder=None):
    ...


@core.extern
@dispatch
def div_rz(arg0, arg1, _builder=None):
    ...


@core.extern
@dispatch
def div_rd(arg0, arg1, _builder=None):
    ...


@core.extern
@dispatch
def div_ru(arg0, arg1, _builder=None):
    ...


@core.extern
@dispatch
def rcp_rn(arg0, _builder=None):
    ...


@core.extern
@dispatch
def rcp_rz(arg0, _builder=None):
    ...


@core.extern
@dispatch
def rcp_rd(arg0, _builder=None):
    ...


@core.extern
@dispatch
def rcp_ru(arg0, _builder=None):
    ...


@core.extern
@dispatch
def sqrt_rn(arg0, _builder=None):
    ...


@core.extern
@dispatch
def sqrt_rz(arg0, _builder=None):
    ...


@core.extern
@dispatch
def sqrt_rd(arg0, _builder=None):
    ...


@core.extern
@dispatch
def sqrt_ru(arg0, _builder=None):
    ...


@core.extern
@dispatch
def sqrt(arg0, _builder=None):
    ...


@core.extern
@dispatch
def add_rn(arg0, arg1, _builder=None):
    ...


@core.extern
@dispatch
def add_rz(arg0, arg1, _builder=None):
    ...


@core.extern
@dispatch
def add_rd(arg0, arg1, _builder=None):
    ...


@core.extern
@dispatch
def add_ru(arg0, arg1, _builder=None):
    ...


@core.extern
@dispatch
def mul_rn(arg0, arg1, _builder=None):
    ...


@core.extern
@dispatch
def mul_rz(arg0, arg1, _builder=None):
    ...


@core.extern
@dispatch
def mul_rd(arg0, arg1, _builder=None):
    ...


@core.extern
@dispatch
def mul_ru(arg0, arg1, _builder=None):
    ...


@core.extern
@dispatch
def double2float_rn(arg0, _builder=None):
    ...


@core.extern
@dispatch
def double2float_rz(arg0, _builder=None):
    ...


@core.extern
@dispatch
def double2float_rd(arg0, _builder=None):
    ...


@core.extern
@dispatch
def double2float_ru(arg0, _builder=None):
    ...


@core.extern
@dispatch
def double2int_rn(arg0, _builder=None):
    ...


@core.extern
@dispatch
def double2int_rz(arg0, _builder=None):
    ...


@core.extern
@dispatch
def double2int_rd(arg0, _builder=None):
    ...


@core.extern
@dispatch
def double2int_ru(arg0, _builder=None):
    ...


@core.extern
@dispatch
def double2uint_rn(arg0, _builder=None):
    ...


@core.extern
@dispatch
def double2uint_rz(arg0, _builder=None):
    ...


@core.extern
@dispatch
def double2uint_rd(arg0, _builder=None):
    ...


@core.extern
@dispatch
def double2uint_ru(arg0, _builder=None):
    ...


@core.extern
@dispatch
def int2double_rn(arg0, _builder=None):
    ...


@core.extern
@dispatch
def uint2double_rn(arg0, _builder=None):
    ...


@core.extern
@dispatch
def float2int_rn(arg0, _builder=None):
    ...


@core.extern
@dispatch
def float2int_rz(arg0, _builder=None):
    ...


@core.extern
@dispatch
def float2int_rd(arg0, _builder=None):
    ...


@core.extern
@dispatch
def float2int_ru(arg0, _builder=None):
    ...


@core.extern
@dispatch
def float2uint_rn(arg0, _builder=None):
    ...


@core.extern
@dispatch
def float2uint_rz(arg0, _builder=None):
    ...


@core.extern
@dispatch
def float2uint_rd(arg0, _builder=None):
    ...


@core.extern
@dispatch
def float2uint_ru(arg0, _builder=None):
    ...


@core.extern
@dispatch
def int2float_rn(arg0, _builder=None):
    ...


@core.extern
@dispatch
def int2float_rz(arg0, _builder=None):
    ...


@core.extern
@dispatch
def int2float_rd(arg0, _builder=None):
    ...


@core.extern
@dispatch
def int2float_ru(arg0, _builder=None):
    ...


@core.extern
@dispatch
def uint2float_rn(arg0, _builder=None):
    ...


@core.extern
@dispatch
def uint2float_rz(arg0, _builder=None):
    ...


@core.extern
@dispatch
def uint2float_rd(arg0, _builder=None):
    ...


@core.extern
@dispatch
def uint2float_ru(arg0, _builder=None):
    ...


@core.extern
@dispatch
def hiloint2double(arg0, arg1, _builder=None):
    ...


@core.extern
@dispatch
def double2loint(arg0, _builder=None):
    ...


@core.extern
@dispatch
def double2hiint(arg0, _builder=None):
    ...


@core.extern
@dispatch
def float2ll_rn(arg0, _builder=None):
    ...


@core.extern
@dispatch
def float2ll_rz(arg0, _builder=None):
    ...


@core.extern
@dispatch
def float2ll_rd(arg0, _builder=None):
    ...


@core.extern
@dispatch
def float2ll_ru(arg0, _builder=None):
    ...


@core.extern
@dispatch
def float2ull_rn(arg0, _builder=None):
    ...


@core.extern
@dispatch
def float2ull_rz(arg0, _builder=None):
    ...


@core.extern
@dispatch
def float2ull_rd(arg0, _builder=None):
    ...


@core.extern
@dispatch
def float2ull_ru(arg0, _builder=None):
    ...


@core.extern
@dispatch
def double2ll_rn(arg0, _builder=None):
    ...


@core.extern
@dispatch
def double2ll_rz(arg0, _builder=None):
    ...


@core.extern
@dispatch
def double2ll_rd(arg0, _builder=None):
    ...


@core.extern
@dispatch
def double2ll_ru(arg0, _builder=None):
    ...


@core.extern
@dispatch
def double2ull_rn(arg0, _builder=None):
    ...


@core.extern
@dispatch
def double2ull_rz(arg0, _builder=None):
    ...


@core.extern
@dispatch
def double2ull_rd(arg0, _builder=None):
    ...


@core.extern
@dispatch
def double2ull_ru(arg0, _builder=None):
    ...


@core.extern
@dispatch
def ll2float_rn(arg0, _builder=None):
    ...


@core.extern
@dispatch
def ll2float_rz(arg0, _builder=None):
    ...


@core.extern
@dispatch
def ll2float_rd(arg0, _builder=None):
    ...


@core.extern
@dispatch
def ll2float_ru(arg0, _builder=None):
    ...


@core.extern
@dispatch
def ull2float_rn(arg0, _builder=None):
    ...


@core.extern
@dispatch
def ull2float_rz(arg0, _builder=None):
    ...


@core.extern
@dispatch
def ull2float_rd(arg0, _builder=None):
    ...


@core.extern
@dispatch
def ull2float_ru(arg0, _builder=None):
    ...


@core.extern
@dispatch
def ll2double_rn(arg0, _builder=None):
    ...


@core.extern
@dispatch
def ll2double_rz(arg0, _builder=None):
    ...


@core.extern
@dispatch
def ll2double_rd(arg0, _builder=None):
    ...


@core.extern
@dispatch
def ll2double_ru(arg0, _builder=None):
    ...


@core.extern
@dispatch
def ull2double_rn(arg0, _builder=None):
    ...


@core.extern
@dispatch
def ull2double_rz(arg0, _builder=None):
    ...


@core.extern
@dispatch
def ull2double_rd(arg0, _builder=None):
    ...


@core.extern
@dispatch
def ull2double_ru(arg0, _builder=None):
    ...


@core.extern
@dispatch
def int_as_float(arg0, _builder=None):
    ...


@core.extern
@dispatch
def float_as_int(arg0, _builder=None):
    ...


@core.extern
@dispatch
def uint_as_float(arg0, _builder=None):
    ...


@core.extern
@dispatch
def float_as_uint(arg0, _builder=None):
    ...


@core.extern
@dispatch
def longlong_as_double(arg0, _builder=None):
    ...


@core.extern
@dispatch
def double_as_longlong(arg0, _builder=None):
    ...


@core.extern
@dispatch
def fast_sinf(arg0, _builder=None):
    ...


@core.extern
@dispatch
def fast_cosf(arg0, _builder=None):
    ...


@core.extern
@dispatch
def fast_log2f(arg0, _builder=None):
    ...


@core.extern
@dispatch
def fast_logf(arg0, _builder=None):
    ...


@core.extern
@dispatch
def fast_expf(arg0, _builder=None):
    ...


@core.extern
@dispatch
def fast_tanf(arg0, _builder=None):
    ...


@core.extern
@dispatch
def fast_exp10f(arg0, _builder=None):
    ...


@core.extern
@dispatch
def fast_log10f(arg0, _builder=None):
    ...


@core.extern
@dispatch
def fast_powf(arg0, arg1, _builder=None):
    ...


@core.extern
@dispatch
def hadd(arg0, arg1, _builder=None):
    ...


@core.extern
@dispatch
def rhadd(arg0, arg1, _builder=None):
    ...


@core.extern
@dispatch
def sub_rn(arg0, arg1, _builder=None):
    ...


@core.extern
@dispatch
def sub_rz(arg0, arg1, _builder=None):
    ...


@core.extern
@dispatch
def sub_rd(arg0, arg1, _builder=None):
    ...


@core.extern
@dispatch
def sub_ru(arg0, arg1, _builder=None):
    ...


@core.extern
@dispatch
def rsqrt_rn(arg0, _builder=None):
    ...


@core.extern
@dispatch
def ffs(arg0, _builder=None):
    ...


@core.extern
@dispatch
def rint(arg0, _builder=None):
    ...


@core.extern
@dispatch
def llrint(arg0, _builder=None):
    ...


@core.extern
@dispatch
def nearbyint(arg0, _builder=None):
    ...


@core.extern
@dispatch
def isnan(arg0, _builder=None):
    ...


@core.extern
@dispatch
def signbit(arg0, _builder=None):
    ...


@core.extern
@dispatch
def copysign(arg0, arg1, _builder=None):
    ...


@core.extern
@dispatch
def finitef(arg0, _builder=None):
    ...


@core.extern
@dispatch
def isinf(arg0, _builder=None):
    ...


@core.extern
@dispatch
def nextafter(arg0, arg1, _builder=None):
    ...


@core.extern
@dispatch
def sin(arg0, _builder=None):
    ...


@core.extern
@dispatch
def cos(arg0, _builder=None):
    ...


@core.extern
@dispatch
def sinpi(arg0, _builder=None):
    ...


@core.extern
@dispatch
def cospi(arg0, _builder=None):
    ...


@core.extern
@dispatch
def tan(arg0, _builder=None):
    ...


@core.extern
@dispatch
def log2(arg0, _builder=None):
    ...


@core.extern
@dispatch
def exp(arg0, _builder=None):
    ...


@core.extern
@dispatch
def exp10(arg0, _builder=None):
    ...


@core.extern
@dispatch
def cosh(arg0, _builder=None):
    ...


@core.extern
@dispatch
def sinh(arg0, _builder=None):
    ...


@core.extern
@dispatch
def tanh(arg0, _builder=None):
    ...


@core.extern
@dispatch
def atan2(arg0, arg1, _builder=None):
    ...


@core.extern
@dispatch
def atan(arg0, _builder=None):
    ...


@core.extern
@dispatch
def asin(arg0, _builder=None):
    ...


@core.extern
@dispatch
def acos(arg0, _builder=None):
    ...


@core.extern
@dispatch
def log(arg0, _builder=None):
    ...


@core.extern
@dispatch
def log10(arg0, _builder=None):
    ...


@core.extern
@dispatch
def log1p(arg0, _builder=None):
    ...


@core.extern
@dispatch
def acosh(arg0, _builder=None):
    ...


@core.extern
@dispatch
def asinh(arg0, _builder=None):
    ...


@core.extern
@dispatch
def atanh(arg0, _builder=None):
    ...


@core.extern
@dispatch
def expm1(arg0, _builder=None):
    ...


@core.extern
@dispatch
def hypot(arg0, arg1, _builder=None):
    ...


@core.extern
@dispatch
def rhypot(arg0, arg1, _builder=None):
    ...


@core.extern
@dispatch
def norm3d(arg0, arg1, arg2, _builder=None):
    ...


@core.extern
@dispatch
def rnorm3d(arg0, arg1, arg2, _builder=None):
    ...


@core.extern
@dispatch
def norm4d(arg0, arg1, arg2, arg3, _builder=None):
    ...


@core.extern
@dispatch
def rnorm4d(arg0, arg1, arg2, arg3, _builder=None):
    ...


@core.extern
@dispatch
def cbrt(arg0, _builder=None):
    ...


@core.extern
@dispatch
def rcbrt(arg0, _builder=None):
    ...


@core.extern
@dispatch
def j0(arg0, _builder=None):
    ...


@core.extern
@dispatch
def j1(arg0, _builder=None):
    ...


@core.extern
@dispatch
def y0(arg0, _builder=None):
    ...


@core.extern
@dispatch
def y1(arg0, _builder=None):
    ...


@core.extern
@dispatch
def yn(arg0, arg1, _builder=None):
    ...


@core.extern
@dispatch
def jn(arg0, arg1, _builder=None):
    ...


@core.extern
@dispatch
def cyl_bessel_i0(arg0, _builder=None):
    ...


@core.extern
@dispatch
def cyl_bessel_i1(arg0, _builder=None):
    ...


@core.extern
@dispatch
def erf(arg0, _builder=None):
    ...


@core.extern
@dispatch
def erfinv(arg0, _builder=None):
    ...


@core.extern
@dispatch
def erfc(arg0, _builder=None):
    ...


@core.extern
@dispatch
def erfcx(arg0, _builder=None):
    ...


@core.extern
@dispatch
def erfcinv(arg0, _builder=None):
    ...


@core.extern
@dispatch
def normcdfinv(arg0, _builder=None):
    ...


@core.extern
@dispatch
def normcdf(arg0, _builder=None):
    ...


@core.extern
@dispatch
def lgamma(arg0, _builder=None):
    ...


@core.extern
@dispatch
def ldexp(arg0, arg1, _builder=None):
    ...


@core.extern
@dispatch
def scalbn(arg0, arg1, _builder=None):
    ...


@core.extern
@dispatch
def fmod(arg0, arg1, _builder=None):
    ...


@core.extern
@dispatch
def remainder(arg0, arg1, _builder=None):
    ...


@core.extern
@dispatch
def fma(arg0, arg1, arg2, _builder=None):
    ...


@core.extern
@dispatch
def pow(arg0, arg1, _builder=None):
    ...


@core.extern
@dispatch
def tgamma(arg0, _builder=None):
    ...


@core.extern
@dispatch
def round(arg0, _builder=None):
    ...


@core.extern
@dispatch
def llround(arg0, _builder=None):
    ...


@core.extern
@dispatch
def fdim(arg0, arg1, _builder=None):
    ...


@core.extern
@dispatch
def ilogb(arg0, _builder=None):
    ...


@core.extern
@dispatch
def logb(arg0, _builder=None):
    ...


@core.extern
@dispatch
def isfinited(arg0, _builder=None):
    ...
