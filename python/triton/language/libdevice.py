import os

from . import core, extern

LIBDEVICE_PATH = os.path.dirname(
    os.path.abspath(__file__)) + "/libdevice.10.bc"


@extern.extern
def clz(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.int32,): ("__nv_clz", core.int32),
                               (core.int64,): ("__nv_clzll", core.int32),
                               }, _builder)


@extern.extern
def popc(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.int32,): ("__nv_popc", core.int32),
                               (core.int64,): ("__nv_popcll", core.int32),
                               }, _builder)


@extern.extern
def byte_perm(arg0, arg1, arg2, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, arg2, ],
                              {(core.int32, core.int32, core.int32,): ("__nv_byte_perm", core.int32),
                               }, _builder)


@extern.extern
def min(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.int32, core.int32,): ("__nv_min", core.int32),
                               (core.uint32, core.uint32,): ("__nv_umin", core.uint32),
                               (core.int64, core.int64,): ("__nv_llmin", core.int64),
                               (core.uint64, core.uint64,): ("__nv_ullmin", core.uint64),
                               (core.float32, core.float32,): ("__nv_fminf", core.float32),
                               (core.float64, core.float64,): ("__nv_fmin", core.float64),
                               }, _builder)


@extern.extern
def max(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.int32, core.int32,): ("__nv_max", core.int32),
                               (core.uint32, core.uint32,): ("__nv_umax", core.uint32),
                               (core.int64, core.int64,): ("__nv_llmax", core.int64),
                               (core.uint64, core.uint64,): ("__nv_ullmax", core.uint64),
                               (core.float32, core.float32,): ("__nv_fmaxf", core.float32),
                               (core.float64, core.float64,): ("__nv_fmax", core.float64),
                               }, _builder)


@extern.extern
def mulhi(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.int32, core.int32,): ("__nv_mulhi", core.int32),
                               (core.uint32, core.uint32,): ("__nv_umulhi", core.uint32),
                               (core.int64, core.int64,): ("__nv_mul64hi", core.int64),
                               (core.uint64, core.uint64,): ("__nv_umul64hi", core.uint64),
                               }, _builder)


@extern.extern
def mul24(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.int32, core.int32,): ("__nv_mul24", core.int32),
                               (core.uint32, core.uint32,): ("__nv_umul24", core.uint32),
                               }, _builder)


@extern.extern
def brev(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.int32,): ("__nv_brev", core.int32),
                               (core.int64,): ("__nv_brevll", core.int64),
                               }, _builder)


@extern.extern
def sad(arg0, arg1, arg2, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, arg2, ],
                              {(core.int32, core.int32, core.uint32,): ("__nv_sad", core.int32),
                               (core.uint32, core.uint32, core.uint32,): ("__nv_usad", core.uint32),
                               }, _builder)


@extern.extern
def abs(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.int32,): ("__nv_abs", core.int32),
                               (core.int64,): ("__nv_llabs", core.int64),
                               (core.float32,): ("__nv_fabsf", core.float32),
                               (core.float64,): ("__nv_fabs", core.float64),
                               }, _builder)


@extern.extern
def floor(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_floorf", core.float32),
                               (core.float64,): ("__nv_floor", core.float64),
                               }, _builder)


@extern.extern
def rcp64h(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float64,): ("__nv_rcp64h", core.float64),
                               }, _builder)


@extern.extern
def rsqrt(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_rsqrtf", core.float32),
                               (core.float64,): ("__nv_rsqrt", core.float64),
                               }, _builder)


@extern.extern
def ceil(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float64,): ("__nv_ceil", core.float64),
                               (core.float32,): ("__nv_ceilf", core.float32),
                               }, _builder)


@extern.extern
def trunc(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float64,): ("__nv_trunc", core.float64),
                               (core.float32,): ("__nv_truncf", core.float32),
                               }, _builder)


@extern.extern
def exp2(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_exp2f", core.float32),
                               (core.float64,): ("__nv_exp2", core.float64),
                               }, _builder)


@extern.extern
def saturatef(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_saturatef", core.float32),
                               }, _builder)


@extern.extern
def fma_rn(arg0, arg1, arg2, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, arg2, ],
                              {(core.float32, core.float32, core.float32,): ("__nv_fmaf_rn", core.float32),
                               (core.float64, core.float64, core.float64,): ("__nv_fma_rn", core.float64),
                               }, _builder)


@extern.extern
def fma_rz(arg0, arg1, arg2, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, arg2, ],
                              {(core.float32, core.float32, core.float32,): ("__nv_fmaf_rz", core.float32),
                               (core.float64, core.float64, core.float64,): ("__nv_fma_rz", core.float64),
                               }, _builder)


@extern.extern
def fma_rd(arg0, arg1, arg2, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, arg2, ],
                              {(core.float32, core.float32, core.float32,): ("__nv_fmaf_rd", core.float32),
                               (core.float64, core.float64, core.float64,): ("__nv_fma_rd", core.float64),
                               }, _builder)


@extern.extern
def fma_ru(arg0, arg1, arg2, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, arg2, ],
                              {(core.float32, core.float32, core.float32,): ("__nv_fmaf_ru", core.float32),
                               (core.float64, core.float64, core.float64,): ("__nv_fma_ru", core.float64),
                               }, _builder)


@extern.extern
def fast_dividef(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.float32, core.float32,): ("__nv_fast_fdividef", core.float32),
                               }, _builder)


@extern.extern
def div_rn(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.float32, core.float32,): ("__nv_fdiv_rn", core.float32),
                               (core.float64, core.float64,): ("__nv_ddiv_rn", core.float64),
                               }, _builder)


@extern.extern
def div_rz(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.float32, core.float32,): ("__nv_fdiv_rz", core.float32),
                               (core.float64, core.float64,): ("__nv_ddiv_rz", core.float64),
                               }, _builder)


@extern.extern
def div_rd(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.float32, core.float32,): ("__nv_fdiv_rd", core.float32),
                               (core.float64, core.float64,): ("__nv_ddiv_rd", core.float64),
                               }, _builder)


@extern.extern
def div_ru(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.float32, core.float32,): ("__nv_fdiv_ru", core.float32),
                               (core.float64, core.float64,): ("__nv_ddiv_ru", core.float64),
                               }, _builder)


@extern.extern
def rcp_rn(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_frcp_rn", core.float32),
                               (core.float64,): ("__nv_drcp_rn", core.float64),
                               }, _builder)


@extern.extern
def rcp_rz(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_frcp_rz", core.float32),
                               (core.float64,): ("__nv_drcp_rz", core.float64),
                               }, _builder)


@extern.extern
def rcp_rd(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_frcp_rd", core.float32),
                               (core.float64,): ("__nv_drcp_rd", core.float64),
                               }, _builder)


@extern.extern
def rcp_ru(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_frcp_ru", core.float32),
                               (core.float64,): ("__nv_drcp_ru", core.float64),
                               }, _builder)


@extern.extern
def sqrt_rn(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_fsqrt_rn", core.float32),
                               (core.float64,): ("__nv_dsqrt_rn", core.float64),
                               }, _builder)


@extern.extern
def sqrt_rz(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_fsqrt_rz", core.float32),
                               (core.float64,): ("__nv_dsqrt_rz", core.float64),
                               }, _builder)


@extern.extern
def sqrt_rd(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_fsqrt_rd", core.float32),
                               (core.float64,): ("__nv_dsqrt_rd", core.float64),
                               }, _builder)


@extern.extern
def sqrt_ru(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_fsqrt_ru", core.float32),
                               (core.float64,): ("__nv_dsqrt_ru", core.float64),
                               }, _builder)


@extern.extern
def sqrt(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_sqrtf", core.float32),
                               (core.float64,): ("__nv_sqrt", core.float64),
                               }, _builder)


@extern.extern
def add_rn(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.float64, core.float64,): ("__nv_dadd_rn", core.float64),
                               (core.float32, core.float32,): ("__nv_fadd_rn", core.float32),
                               }, _builder)


@extern.extern
def add_rz(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.float64, core.float64,): ("__nv_dadd_rz", core.float64),
                               (core.float32, core.float32,): ("__nv_fadd_rz", core.float32),
                               }, _builder)


@extern.extern
def add_rd(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.float64, core.float64,): ("__nv_dadd_rd", core.float64),
                               (core.float32, core.float32,): ("__nv_fadd_rd", core.float32),
                               }, _builder)


@extern.extern
def add_ru(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.float64, core.float64,): ("__nv_dadd_ru", core.float64),
                               (core.float32, core.float32,): ("__nv_fadd_ru", core.float32),
                               }, _builder)


@extern.extern
def mul_rn(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.float64, core.float64,): ("__nv_dmul_rn", core.float64),
                               (core.float32, core.float32,): ("__nv_fmul_rn", core.float32),
                               }, _builder)


@extern.extern
def mul_rz(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.float64, core.float64,): ("__nv_dmul_rz", core.float64),
                               (core.float32, core.float32,): ("__nv_fmul_rz", core.float32),
                               }, _builder)


@extern.extern
def mul_rd(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.float64, core.float64,): ("__nv_dmul_rd", core.float64),
                               (core.float32, core.float32,): ("__nv_fmul_rd", core.float32),
                               }, _builder)


@extern.extern
def mul_ru(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.float64, core.float64,): ("__nv_dmul_ru", core.float64),
                               (core.float32, core.float32,): ("__nv_fmul_ru", core.float32),
                               }, _builder)


@extern.extern
def double2float_rn(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float64,): ("__nv_double2float_rn", core.float32),
                               }, _builder)


@extern.extern
def double2float_rz(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float64,): ("__nv_double2float_rz", core.float32),
                               }, _builder)


@extern.extern
def double2float_rd(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float64,): ("__nv_double2float_rd", core.float32),
                               }, _builder)


@extern.extern
def double2float_ru(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float64,): ("__nv_double2float_ru", core.float32),
                               }, _builder)


@extern.extern
def double2int_rn(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float64,): ("__nv_double2int_rn", core.int32),
                               }, _builder)


@extern.extern
def double2int_rz(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float64,): ("__nv_double2int_rz", core.int32),
                               }, _builder)


@extern.extern
def double2int_rd(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float64,): ("__nv_double2int_rd", core.int32),
                               }, _builder)


@extern.extern
def double2int_ru(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float64,): ("__nv_double2int_ru", core.int32),
                               }, _builder)


@extern.extern
def double2uint_rn(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float64,): ("__nv_double2uint_rn", core.int32),
                               }, _builder)


@extern.extern
def double2uint_rz(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float64,): ("__nv_double2uint_rz", core.int32),
                               }, _builder)


@extern.extern
def double2uint_rd(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float64,): ("__nv_double2uint_rd", core.int32),
                               }, _builder)


@extern.extern
def double2uint_ru(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float64,): ("__nv_double2uint_ru", core.int32),
                               }, _builder)


@extern.extern
def int2double_rn(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.int32,): ("__nv_int2double_rn", core.float64),
                               }, _builder)


@extern.extern
def uint2double_rn(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.uint32,): ("__nv_uint2double_rn", core.float64),
                               }, _builder)


@extern.extern
def float2int_rn(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_float2int_rn", core.int32),
                               }, _builder)


@extern.extern
def float2int_rz(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_float2int_rz", core.int32),
                               }, _builder)


@extern.extern
def float2int_rd(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_float2int_rd", core.int32),
                               }, _builder)


@extern.extern
def float2int_ru(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_float2int_ru", core.int32),
                               }, _builder)


@extern.extern
def float2uint_rn(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_float2uint_rn", core.int32),
                               }, _builder)


@extern.extern
def float2uint_rz(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_float2uint_rz", core.int32),
                               }, _builder)


@extern.extern
def float2uint_rd(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_float2uint_rd", core.int32),
                               }, _builder)


@extern.extern
def float2uint_ru(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_float2uint_ru", core.int32),
                               }, _builder)


@extern.extern
def int2float_rn(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.int32,): ("__nv_int2float_rn", core.float32),
                               }, _builder)


@extern.extern
def int2float_rz(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.int32,): ("__nv_int2float_rz", core.float32),
                               }, _builder)


@extern.extern
def int2float_rd(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.int32,): ("__nv_int2float_rd", core.float32),
                               }, _builder)


@extern.extern
def int2float_ru(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.int32,): ("__nv_int2float_ru", core.float32),
                               }, _builder)


@extern.extern
def uint2float_rn(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.uint32,): ("__nv_uint2float_rn", core.float32),
                               }, _builder)


@extern.extern
def uint2float_rz(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.uint32,): ("__nv_uint2float_rz", core.float32),
                               }, _builder)


@extern.extern
def uint2float_rd(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.uint32,): ("__nv_uint2float_rd", core.float32),
                               }, _builder)


@extern.extern
def uint2float_ru(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.uint32,): ("__nv_uint2float_ru", core.float32),
                               }, _builder)


@extern.extern
def hiloint2double(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.int32, core.int32,): ("__nv_hiloint2double", core.float64),
                               }, _builder)


@extern.extern
def double2loint(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float64,): ("__nv_double2loint", core.int32),
                               }, _builder)


@extern.extern
def double2hiint(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float64,): ("__nv_double2hiint", core.int32),
                               }, _builder)


@extern.extern
def float2ll_rn(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_float2ll_rn", core.int64),
                               }, _builder)


@extern.extern
def float2ll_rz(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_float2ll_rz", core.int64),
                               }, _builder)


@extern.extern
def float2ll_rd(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_float2ll_rd", core.int64),
                               }, _builder)


@extern.extern
def float2ll_ru(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_float2ll_ru", core.int64),
                               }, _builder)


@extern.extern
def float2ull_rn(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_float2ull_rn", core.int64),
                               }, _builder)


@extern.extern
def float2ull_rz(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_float2ull_rz", core.int64),
                               }, _builder)


@extern.extern
def float2ull_rd(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_float2ull_rd", core.int64),
                               }, _builder)


@extern.extern
def float2ull_ru(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_float2ull_ru", core.int64),
                               }, _builder)


@extern.extern
def double2ll_rn(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float64,): ("__nv_double2ll_rn", core.int64),
                               }, _builder)


@extern.extern
def double2ll_rz(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float64,): ("__nv_double2ll_rz", core.int64),
                               }, _builder)


@extern.extern
def double2ll_rd(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float64,): ("__nv_double2ll_rd", core.int64),
                               }, _builder)


@extern.extern
def double2ll_ru(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float64,): ("__nv_double2ll_ru", core.int64),
                               }, _builder)


@extern.extern
def double2ull_rn(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float64,): ("__nv_double2ull_rn", core.int64),
                               }, _builder)


@extern.extern
def double2ull_rz(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float64,): ("__nv_double2ull_rz", core.int64),
                               }, _builder)


@extern.extern
def double2ull_rd(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float64,): ("__nv_double2ull_rd", core.int64),
                               }, _builder)


@extern.extern
def double2ull_ru(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float64,): ("__nv_double2ull_ru", core.int64),
                               }, _builder)


@extern.extern
def ll2float_rn(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.int64,): ("__nv_ll2float_rn", core.float32),
                               }, _builder)


@extern.extern
def ll2float_rz(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.int64,): ("__nv_ll2float_rz", core.float32),
                               }, _builder)


@extern.extern
def ll2float_rd(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.int64,): ("__nv_ll2float_rd", core.float32),
                               }, _builder)


@extern.extern
def ll2float_ru(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.int64,): ("__nv_ll2float_ru", core.float32),
                               }, _builder)


@extern.extern
def ull2float_rn(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.uint64,): ("__nv_ull2float_rn", core.float32),
                               }, _builder)


@extern.extern
def ull2float_rz(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.uint64,): ("__nv_ull2float_rz", core.float32),
                               }, _builder)


@extern.extern
def ull2float_rd(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.uint64,): ("__nv_ull2float_rd", core.float32),
                               }, _builder)


@extern.extern
def ull2float_ru(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.uint64,): ("__nv_ull2float_ru", core.float32),
                               }, _builder)


@extern.extern
def ll2double_rn(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.int64,): ("__nv_ll2double_rn", core.float64),
                               }, _builder)


@extern.extern
def ll2double_rz(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.int64,): ("__nv_ll2double_rz", core.float64),
                               }, _builder)


@extern.extern
def ll2double_rd(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.int64,): ("__nv_ll2double_rd", core.float64),
                               }, _builder)


@extern.extern
def ll2double_ru(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.int64,): ("__nv_ll2double_ru", core.float64),
                               }, _builder)


@extern.extern
def ull2double_rn(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.uint64,): ("__nv_ull2double_rn", core.float64),
                               }, _builder)


@extern.extern
def ull2double_rz(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.uint64,): ("__nv_ull2double_rz", core.float64),
                               }, _builder)


@extern.extern
def ull2double_rd(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.uint64,): ("__nv_ull2double_rd", core.float64),
                               }, _builder)


@extern.extern
def ull2double_ru(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.uint64,): ("__nv_ull2double_ru", core.float64),
                               }, _builder)


@extern.extern
def int_as_float(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.int32,): ("__nv_int_as_float", core.float32),
                               }, _builder)


@extern.extern
def float_as_int(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_float_as_int", core.int32),
                               }, _builder)


@extern.extern
def uint_as_float(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.uint32,): ("__nv_uint_as_float", core.float32),
                               }, _builder)


@extern.extern
def float_as_uint(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_float_as_uint", core.int32),
                               }, _builder)


@extern.extern
def longlong_as_double(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.int64,): ("__nv_longlong_as_double", core.float64),
                               }, _builder)


@extern.extern
def double_as_longlong(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float64,): ("__nv_double_as_longlong", core.int64),
                               }, _builder)


@extern.extern
def fast_sinf(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_fast_sinf", core.float32),
                               }, _builder)


@extern.extern
def fast_cosf(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_fast_cosf", core.float32),
                               }, _builder)


@extern.extern
def fast_log2f(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_fast_log2f", core.float32),
                               }, _builder)


@extern.extern
def fast_logf(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_fast_logf", core.float32),
                               }, _builder)


@extern.extern
def fast_expf(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_fast_expf", core.float32),
                               }, _builder)


@extern.extern
def fast_tanf(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_fast_tanf", core.float32),
                               }, _builder)


@extern.extern
def fast_exp10f(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_fast_exp10f", core.float32),
                               }, _builder)


@extern.extern
def fast_log10f(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_fast_log10f", core.float32),
                               }, _builder)


@extern.extern
def fast_powf(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.float32, core.float32,): ("__nv_fast_powf", core.float32),
                               }, _builder)


@extern.extern
def hadd(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.int32, core.int32,): ("__nv_hadd", core.int32),
                               (core.uint32, core.uint32,): ("__nv_uhadd", core.uint32),
                               }, _builder)


@extern.extern
def rhadd(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.int32, core.int32,): ("__nv_rhadd", core.int32),
                               (core.uint32, core.uint32,): ("__nv_urhadd", core.uint32),
                               }, _builder)


@extern.extern
def sub_rn(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.float32, core.float32,): ("__nv_fsub_rn", core.float32),
                               (core.float64, core.float64,): ("__nv_dsub_rn", core.float64),
                               }, _builder)


@extern.extern
def sub_rz(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.float32, core.float32,): ("__nv_fsub_rz", core.float32),
                               (core.float64, core.float64,): ("__nv_dsub_rz", core.float64),
                               }, _builder)


@extern.extern
def sub_rd(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.float32, core.float32,): ("__nv_fsub_rd", core.float32),
                               (core.float64, core.float64,): ("__nv_dsub_rd", core.float64),
                               }, _builder)


@extern.extern
def sub_ru(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.float32, core.float32,): ("__nv_fsub_ru", core.float32),
                               (core.float64, core.float64,): ("__nv_dsub_ru", core.float64),
                               }, _builder)


@extern.extern
def rsqrt_rn(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_frsqrt_rn", core.float32),
                               }, _builder)


@extern.extern
def ffs(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.int32,): ("__nv_ffs", core.int32),
                               (core.int64,): ("__nv_ffsll", core.int32),
                               }, _builder)


@extern.extern
def rint(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_rintf", core.float32),
                               (core.float64,): ("__nv_rint", core.float64),
                               }, _builder)


@extern.extern
def llrint(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_llrintf", core.int64),
                               (core.float64,): ("__nv_llrint", core.int64),
                               }, _builder)


@extern.extern
def nearbyint(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_nearbyintf", core.float32),
                               (core.float64,): ("__nv_nearbyint", core.float64),
                               }, _builder)


@extern.extern
def isnan(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_isnanf", core.int32),
                               (core.float64,): ("__nv_isnand", core.int32),
                               }, _builder)


@extern.extern
def signbit(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_signbitf", core.int32),
                               (core.float64,): ("__nv_signbitd", core.int32),
                               }, _builder)


@extern.extern
def copysign(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.float32, core.float32,): ("__nv_copysignf", core.float32),
                               (core.float64, core.float64,): ("__nv_copysign", core.float64),
                               }, _builder)


@extern.extern
def finitef(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_finitef", core.int32),
                               }, _builder)


@extern.extern
def isinf(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_isinff", core.int32),
                               (core.float64,): ("__nv_isinfd", core.int32),
                               }, _builder)


@extern.extern
def nextafter(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.float32, core.float32,): ("__nv_nextafterf", core.float32),
                               (core.float64, core.float64,): ("__nv_nextafter", core.float64),
                               }, _builder)


@extern.extern
def sin(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_sinf", core.float32),
                               (core.float64,): ("__nv_sin", core.float64),
                               }, _builder)


@extern.extern
def cos(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_cosf", core.float32),
                               (core.float64,): ("__nv_cos", core.float64),
                               }, _builder)


@extern.extern
def sinpi(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_sinpif", core.float32),
                               (core.float64,): ("__nv_sinpi", core.float64),
                               }, _builder)


@extern.extern
def cospi(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_cospif", core.float32),
                               (core.float64,): ("__nv_cospi", core.float64),
                               }, _builder)


@extern.extern
def tan(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_tanf", core.float32),
                               (core.float64,): ("__nv_tan", core.float64),
                               }, _builder)


@extern.extern
def log2(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_log2f", core.float32),
                               (core.float64,): ("__nv_log2", core.float64),
                               }, _builder)


@extern.extern
def exp(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_expf", core.float32),
                               (core.float64,): ("__nv_exp", core.float64),
                               }, _builder)


@extern.extern
def exp10(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_exp10f", core.float32),
                               (core.float64,): ("__nv_exp10", core.float64),
                               }, _builder)


@extern.extern
def cosh(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_coshf", core.float32),
                               (core.float64,): ("__nv_cosh", core.float64),
                               }, _builder)


@extern.extern
def sinh(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_sinhf", core.float32),
                               (core.float64,): ("__nv_sinh", core.float64),
                               }, _builder)


@extern.extern
def tanh(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_tanhf", core.float32),
                               (core.float64,): ("__nv_tanh", core.float64),
                               }, _builder)


@extern.extern
def atan2(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.float32, core.float32,): ("__nv_atan2f", core.float32),
                               (core.float64, core.float64,): ("__nv_atan2", core.float64),
                               }, _builder)


@extern.extern
def atan(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_atanf", core.float32),
                               (core.float64,): ("__nv_atan", core.float64),
                               }, _builder)


@extern.extern
def asin(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_asinf", core.float32),
                               (core.float64,): ("__nv_asin", core.float64),
                               }, _builder)


@extern.extern
def acos(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_acosf", core.float32),
                               (core.float64,): ("__nv_acos", core.float64),
                               }, _builder)


@extern.extern
def log(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_logf", core.float32),
                               (core.float64,): ("__nv_log", core.float64),
                               }, _builder)


@extern.extern
def log10(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_log10f", core.float32),
                               (core.float64,): ("__nv_log10", core.float64),
                               }, _builder)


@extern.extern
def log1p(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_log1pf", core.float32),
                               (core.float64,): ("__nv_log1p", core.float64),
                               }, _builder)


@extern.extern
def acosh(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_acoshf", core.float32),
                               (core.float64,): ("__nv_acosh", core.float64),
                               }, _builder)


@extern.extern
def asinh(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_asinhf", core.float32),
                               (core.float64,): ("__nv_asinh", core.float64),
                               }, _builder)


@extern.extern
def atanh(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_atanhf", core.float32),
                               (core.float64,): ("__nv_atanh", core.float64),
                               }, _builder)


@extern.extern
def expm1(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_expm1f", core.float32),
                               (core.float64,): ("__nv_expm1", core.float64),
                               }, _builder)


@extern.extern
def hypot(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.float32, core.float32,): ("__nv_hypotf", core.float32),
                               (core.float64, core.float64,): ("__nv_hypot", core.float64),
                               }, _builder)


@extern.extern
def rhypot(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.float32, core.float32,): ("__nv_rhypotf", core.float32),
                               (core.float64, core.float64,): ("__nv_rhypot", core.float64),
                               }, _builder)


@extern.extern
def norm3d(arg0, arg1, arg2, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, arg2, ],
                              {(core.float32, core.float32, core.float32,): ("__nv_norm3df", core.float32),
                               (core.float64, core.float64, core.float64,): ("__nv_norm3d", core.float64),
                               }, _builder)


@extern.extern
def rnorm3d(arg0, arg1, arg2, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, arg2, ],
                              {(core.float32, core.float32, core.float32,): ("__nv_rnorm3df", core.float32),
                               (core.float64, core.float64, core.float64,): ("__nv_rnorm3d", core.float64),
                               }, _builder)


@extern.extern
def norm4d(arg0, arg1, arg2, arg3, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, arg2, arg3, ],
                              {(core.float32, core.float32, core.float32, core.float32,): ("__nv_norm4df", core.float32),
                               (core.float64, core.float64, core.float64, core.float64,): ("__nv_norm4d", core.float64),
                               }, _builder)


@extern.extern
def rnorm4d(arg0, arg1, arg2, arg3, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, arg2, arg3, ],
                              {(core.float32, core.float32, core.float32, core.float32,): ("__nv_rnorm4df", core.float32),
                               (core.float64, core.float64, core.float64, core.float64,): ("__nv_rnorm4d", core.float64),
                               }, _builder)


@extern.extern
def cbrt(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_cbrtf", core.float32),
                               (core.float64,): ("__nv_cbrt", core.float64),
                               }, _builder)


@extern.extern
def rcbrt(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_rcbrtf", core.float32),
                               (core.float64,): ("__nv_rcbrt", core.float64),
                               }, _builder)


@extern.extern
def j0(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_j0f", core.float32),
                               (core.float64,): ("__nv_j0", core.float64),
                               }, _builder)


@extern.extern
def j1(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_j1f", core.float32),
                               (core.float64,): ("__nv_j1", core.float64),
                               }, _builder)


@extern.extern
def y0(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_y0f", core.float32),
                               (core.float64,): ("__nv_y0", core.float64),
                               }, _builder)


@extern.extern
def y1(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_y1f", core.float32),
                               (core.float64,): ("__nv_y1", core.float64),
                               }, _builder)


@extern.extern
def yn(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.int32, core.float32,): ("__nv_ynf", core.float32),
                               (core.int32, core.float64,): ("__nv_yn", core.float64),
                               }, _builder)


@extern.extern
def jn(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.int32, core.float32,): ("__nv_jnf", core.float32),
                               (core.int32, core.float64,): ("__nv_jn", core.float64),
                               }, _builder)


@extern.extern
def cyl_bessel_i0(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_cyl_bessel_i0f", core.float32),
                               (core.float64,): ("__nv_cyl_bessel_i0", core.float64),
                               }, _builder)


@extern.extern
def cyl_bessel_i1(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_cyl_bessel_i1f", core.float32),
                               (core.float64,): ("__nv_cyl_bessel_i1", core.float64),
                               }, _builder)


@extern.extern
def erf(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_erff", core.float32),
                               (core.float64,): ("__nv_erf", core.float64),
                               }, _builder)


@extern.extern
def erfinv(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_erfinvf", core.float32),
                               (core.float64,): ("__nv_erfinv", core.float64),
                               }, _builder)


@extern.extern
def erfc(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_erfcf", core.float32),
                               (core.float64,): ("__nv_erfc", core.float64),
                               }, _builder)


@extern.extern
def erfcx(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_erfcxf", core.float32),
                               (core.float64,): ("__nv_erfcx", core.float64),
                               }, _builder)


@extern.extern
def erfcinv(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_erfcinvf", core.float32),
                               (core.float64,): ("__nv_erfcinv", core.float64),
                               }, _builder)


@extern.extern
def normcdfinv(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_normcdfinvf", core.float32),
                               (core.float64,): ("__nv_normcdfinv", core.float64),
                               }, _builder)


@extern.extern
def normcdf(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_normcdff", core.float32),
                               (core.float64,): ("__nv_normcdf", core.float64),
                               }, _builder)


@extern.extern
def lgamma(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_lgammaf", core.float32),
                               (core.float64,): ("__nv_lgamma", core.float64),
                               }, _builder)


@extern.extern
def ldexp(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.float32, core.int32,): ("__nv_ldexpf", core.float32),
                               (core.float64, core.int32,): ("__nv_ldexp", core.float64),
                               }, _builder)


@extern.extern
def scalbn(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.float32, core.int32,): ("__nv_scalbnf", core.float32),
                               (core.float64, core.int32,): ("__nv_scalbn", core.float64),
                               }, _builder)


@extern.extern
def fmod(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.float32, core.float32,): ("__nv_fmodf", core.float32),
                               (core.float64, core.float64,): ("__nv_fmod", core.float64),
                               }, _builder)


@extern.extern
def remainder(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.float32, core.float32,): ("__nv_remainderf", core.float32),
                               (core.float64, core.float64,): ("__nv_remainder", core.float64),
                               }, _builder)


@extern.extern
def fma(arg0, arg1, arg2, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, arg2, ],
                              {(core.float32, core.float32, core.float32,): ("__nv_fmaf", core.float32),
                               (core.float64, core.float64, core.float64,): ("__nv_fma", core.float64),
                               }, _builder)


@extern.extern
def pow(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.float32, core.int32,): ("__nv_powif", core.float32),
                               (core.float64, core.int32,): ("__nv_powi", core.float64),
                               (core.float32, core.float32,): ("__nv_powf", core.float32),
                               (core.float64, core.float64,): ("__nv_pow", core.float64),
                               }, _builder)


@extern.extern
def tgamma(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_tgammaf", core.float32),
                               (core.float64,): ("__nv_tgamma", core.float64),
                               }, _builder)


@extern.extern
def round(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_roundf", core.float32),
                               (core.float64,): ("__nv_round", core.float64),
                               }, _builder)


@extern.extern
def llround(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_llroundf", core.int64),
                               (core.float64,): ("__nv_llround", core.int64),
                               }, _builder)


@extern.extern
def fdim(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.float32, core.float32,): ("__nv_fdimf", core.float32),
                               (core.float64, core.float64,): ("__nv_fdim", core.float64),
                               }, _builder)


@extern.extern
def ilogb(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_ilogbf", core.int32),
                               (core.float64,): ("__nv_ilogb", core.int32),
                               }, _builder)


@extern.extern
def logb(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float32,): ("__nv_logbf", core.float32),
                               (core.float64,): ("__nv_logb", core.float64),
                               }, _builder)


@extern.extern
def isfinited(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.float64,): ("__nv_isfinited", core.int32),
                               }, _builder)
