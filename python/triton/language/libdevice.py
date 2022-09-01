import os

from . import core, extern

LIBDEVICE_PATH = os.path.dirname(
    os.path.abspath(__file__)) + "/libdevice.10.bc"


@extern.extern
def clz(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("int32"),): ("__nv_clz", core.dtype("int32")),
                               (core.dtype("int64"),): ("__nv_clzll", core.dtype("int32")),
                               }, _builder)


@extern.extern
def popc(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("int32"),): ("__nv_popc", core.dtype("int32")),
                               (core.dtype("int64"),): ("__nv_popcll", core.dtype("int32")),
                               }, _builder)


@extern.extern
def byte_perm(arg0, arg1, arg2, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, arg2, ],
                              {(core.dtype("int32"), core.dtype("int32"), core.dtype("int32"),): ("__nv_byte_perm", core.dtype("int32")),
                               }, _builder)


@extern.extern
def min(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("int32"), core.dtype("int32"),): ("__nv_min", core.dtype("int32")),
                               (core.dtype("uint32"), core.dtype("uint32"),): ("__nv_umin", core.dtype("uint32")),
                               (core.dtype("int64"), core.dtype("int64"),): ("__nv_llmin", core.dtype("int64")),
                               (core.dtype("uint64"), core.dtype("uint64"),): ("__nv_ullmin", core.dtype("uint64")),
                               (core.dtype("fp32"), core.dtype("fp32"),): ("__nv_fminf", core.dtype("fp32")),
                               (core.dtype("fp64"), core.dtype("fp64"),): ("__nv_fmin", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def max(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("int32"), core.dtype("int32"),): ("__nv_max", core.dtype("int32")),
                               (core.dtype("uint32"), core.dtype("uint32"),): ("__nv_umax", core.dtype("uint32")),
                               (core.dtype("int64"), core.dtype("int64"),): ("__nv_llmax", core.dtype("int64")),
                               (core.dtype("uint64"), core.dtype("uint64"),): ("__nv_ullmax", core.dtype("uint64")),
                               (core.dtype("fp32"), core.dtype("fp32"),): ("__nv_fmaxf", core.dtype("fp32")),
                               (core.dtype("fp64"), core.dtype("fp64"),): ("__nv_fmax", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def mulhi(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("int32"), core.dtype("int32"),): ("__nv_mulhi", core.dtype("int32")),
                               (core.dtype("uint32"), core.dtype("uint32"),): ("__nv_umulhi", core.dtype("uint32")),
                               }, _builder)


@extern.extern
def mul64hi(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("int64"), core.dtype("int64"),): ("__nv_mul64hi", core.dtype("int64")),
                               (core.dtype("uint64"), core.dtype("uint64"),): ("__nv_umul64hi", core.dtype("uint64")),
                               }, _builder)


@extern.extern
def mul24(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("int32"), core.dtype("int32"),): ("__nv_mul24", core.dtype("int32")),
                               (core.dtype("uint32"), core.dtype("uint32"),): ("__nv_umul24", core.dtype("uint32")),
                               }, _builder)


@extern.extern
def brev(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("int32"),): ("__nv_brev", core.dtype("int32")),
                               (core.dtype("int64"),): ("__nv_brevll", core.dtype("int64")),
                               }, _builder)


@extern.extern
def sad(arg0, arg1, arg2, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, arg2, ],
                              {(core.dtype("int32"), core.dtype("int32"), core.dtype("uint32"),): ("__nv_sad", core.dtype("int32")),
                               (core.dtype("uint32"), core.dtype("uint32"), core.dtype("uint32"),): ("__nv_usad", core.dtype("uint32")),
                               }, _builder)


@extern.extern
def abs(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("int32"),): ("__nv_abs", core.dtype("int32")),
                               (core.dtype("int64"),): ("__nv_llabs", core.dtype("int64")),
                               (core.dtype("fp32"),): ("__nv_fabsf", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_fabs", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def floor(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_floorf", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_floor", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def rcp64h(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp64"),): ("__nv_rcp64h", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def rsqrt(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_rsqrtf", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_rsqrt", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def ceil(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp64"),): ("__nv_ceil", core.dtype("fp64")),
                               (core.dtype("fp32"),): ("__nv_ceilf", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def trunc(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp64"),): ("__nv_trunc", core.dtype("fp64")),
                               (core.dtype("fp32"),): ("__nv_truncf", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def exp2(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_exp2f", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_exp2", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def saturatef(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_saturatef", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def fmaf_rn(arg0, arg1, arg2, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, arg2, ],
                              {(core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32"),): ("__nv_fmaf_rn", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def fmaf_rz(arg0, arg1, arg2, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, arg2, ],
                              {(core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32"),): ("__nv_fmaf_rz", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def fmaf_rd(arg0, arg1, arg2, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, arg2, ],
                              {(core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32"),): ("__nv_fmaf_rd", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def fmaf_ru(arg0, arg1, arg2, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, arg2, ],
                              {(core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32"),): ("__nv_fmaf_ru", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def fmaf_ieee_rn(arg0, arg1, arg2, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, arg2, ],
                              {(core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32"),): ("__nv_fmaf_ieee_rn", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def fmaf_ieee_rz(arg0, arg1, arg2, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, arg2, ],
                              {(core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32"),): ("__nv_fmaf_ieee_rz", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def fmaf_ieee_rd(arg0, arg1, arg2, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, arg2, ],
                              {(core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32"),): ("__nv_fmaf_ieee_rd", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def fmaf_ieee_ru(arg0, arg1, arg2, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, arg2, ],
                              {(core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32"),): ("__nv_fmaf_ieee_ru", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def fma_rn(arg0, arg1, arg2, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, arg2, ],
                              {(core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64"),): ("__nv_fma_rn", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def fma_rz(arg0, arg1, arg2, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, arg2, ],
                              {(core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64"),): ("__nv_fma_rz", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def fma_rd(arg0, arg1, arg2, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, arg2, ],
                              {(core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64"),): ("__nv_fma_rd", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def fma_ru(arg0, arg1, arg2, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, arg2, ],
                              {(core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64"),): ("__nv_fma_ru", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def fast_fdividef(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp32"), core.dtype("fp32"),): ("__nv_fast_fdividef", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def fdiv_rn(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp32"), core.dtype("fp32"),): ("__nv_fdiv_rn", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def fdiv_rz(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp32"), core.dtype("fp32"),): ("__nv_fdiv_rz", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def fdiv_rd(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp32"), core.dtype("fp32"),): ("__nv_fdiv_rd", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def fdiv_ru(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp32"), core.dtype("fp32"),): ("__nv_fdiv_ru", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def frcp_rn(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_frcp_rn", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def frcp_rz(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_frcp_rz", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def frcp_rd(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_frcp_rd", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def frcp_ru(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_frcp_ru", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def fsqrt_rn(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_fsqrt_rn", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def fsqrt_rz(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_fsqrt_rz", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def fsqrt_rd(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_fsqrt_rd", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def fsqrt_ru(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_fsqrt_ru", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def ddiv_rn(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp64"), core.dtype("fp64"),): ("__nv_ddiv_rn", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def ddiv_rz(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp64"), core.dtype("fp64"),): ("__nv_ddiv_rz", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def ddiv_rd(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp64"), core.dtype("fp64"),): ("__nv_ddiv_rd", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def ddiv_ru(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp64"), core.dtype("fp64"),): ("__nv_ddiv_ru", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def drcp_rn(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp64"),): ("__nv_drcp_rn", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def drcp_rz(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp64"),): ("__nv_drcp_rz", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def drcp_rd(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp64"),): ("__nv_drcp_rd", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def drcp_ru(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp64"),): ("__nv_drcp_ru", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def dsqrt_rn(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp64"),): ("__nv_dsqrt_rn", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def dsqrt_rz(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp64"),): ("__nv_dsqrt_rz", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def dsqrt_rd(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp64"),): ("__nv_dsqrt_rd", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def dsqrt_ru(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp64"),): ("__nv_dsqrt_ru", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def sqrt(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_sqrtf", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_sqrt", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def dadd_rn(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp64"), core.dtype("fp64"),): ("__nv_dadd_rn", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def dadd_rz(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp64"), core.dtype("fp64"),): ("__nv_dadd_rz", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def dadd_rd(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp64"), core.dtype("fp64"),): ("__nv_dadd_rd", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def dadd_ru(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp64"), core.dtype("fp64"),): ("__nv_dadd_ru", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def dmul_rn(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp64"), core.dtype("fp64"),): ("__nv_dmul_rn", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def dmul_rz(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp64"), core.dtype("fp64"),): ("__nv_dmul_rz", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def dmul_rd(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp64"), core.dtype("fp64"),): ("__nv_dmul_rd", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def dmul_ru(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp64"), core.dtype("fp64"),): ("__nv_dmul_ru", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def fadd_rd(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp32"), core.dtype("fp32"),): ("__nv_fadd_rd", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def fadd_ru(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp32"), core.dtype("fp32"),): ("__nv_fadd_ru", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def fmul_rd(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp32"), core.dtype("fp32"),): ("__nv_fmul_rd", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def fmul_ru(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp32"), core.dtype("fp32"),): ("__nv_fmul_ru", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def fadd_rn(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp32"), core.dtype("fp32"),): ("__nv_fadd_rn", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def fadd_rz(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp32"), core.dtype("fp32"),): ("__nv_fadd_rz", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def fmul_rn(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp32"), core.dtype("fp32"),): ("__nv_fmul_rn", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def fmul_rz(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp32"), core.dtype("fp32"),): ("__nv_fmul_rz", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def double2float_rn(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp64"),): ("__nv_double2float_rn", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def double2float_rz(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp64"),): ("__nv_double2float_rz", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def double2float_rd(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp64"),): ("__nv_double2float_rd", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def double2float_ru(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp64"),): ("__nv_double2float_ru", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def double2int_rn(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp64"),): ("__nv_double2int_rn", core.dtype("int32")),
                               }, _builder)


@extern.extern
def double2int_rz(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp64"),): ("__nv_double2int_rz", core.dtype("int32")),
                               }, _builder)


@extern.extern
def double2int_rd(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp64"),): ("__nv_double2int_rd", core.dtype("int32")),
                               }, _builder)


@extern.extern
def double2int_ru(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp64"),): ("__nv_double2int_ru", core.dtype("int32")),
                               }, _builder)


@extern.extern
def double2uint_rn(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp64"),): ("__nv_double2uint_rn", core.dtype("int32")),
                               }, _builder)


@extern.extern
def double2uint_rz(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp64"),): ("__nv_double2uint_rz", core.dtype("int32")),
                               }, _builder)


@extern.extern
def double2uint_rd(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp64"),): ("__nv_double2uint_rd", core.dtype("int32")),
                               }, _builder)


@extern.extern
def double2uint_ru(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp64"),): ("__nv_double2uint_ru", core.dtype("int32")),
                               }, _builder)


@extern.extern
def int2double_rn(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("int32"),): ("__nv_int2double_rn", core.dtype("fp64")),
                               (core.dtype("uint32"),): ("__nv_uint2double_rn", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def float2int_rn(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_float2int_rn", core.dtype("int32")),
                               }, _builder)


@extern.extern
def float2int_rz(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_float2int_rz", core.dtype("int32")),
                               }, _builder)


@extern.extern
def float2int_rd(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_float2int_rd", core.dtype("int32")),
                               }, _builder)


@extern.extern
def float2int_ru(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_float2int_ru", core.dtype("int32")),
                               }, _builder)


@extern.extern
def float2uint_rn(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_float2uint_rn", core.dtype("int32")),
                               }, _builder)


@extern.extern
def float2uint_rz(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_float2uint_rz", core.dtype("int32")),
                               }, _builder)


@extern.extern
def float2uint_rd(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_float2uint_rd", core.dtype("int32")),
                               }, _builder)


@extern.extern
def float2uint_ru(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_float2uint_ru", core.dtype("int32")),
                               }, _builder)


@extern.extern
def int2float_rn(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("int32"),): ("__nv_int2float_rn", core.dtype("fp32")),
                               (core.dtype("uint32"),): ("__nv_uint2float_rn", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def int2float_rz(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("int32"),): ("__nv_int2float_rz", core.dtype("fp32")),
                               (core.dtype("uint32"),): ("__nv_uint2float_rz", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def int2float_rd(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("int32"),): ("__nv_int2float_rd", core.dtype("fp32")),
                               (core.dtype("uint32"),): ("__nv_uint2float_rd", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def int2float_ru(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("int32"),): ("__nv_int2float_ru", core.dtype("fp32")),
                               (core.dtype("uint32"),): ("__nv_uint2float_ru", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def hiloint2double(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("int32"), core.dtype("int32"),): ("__nv_hiloint2double", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def double2loint(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp64"),): ("__nv_double2loint", core.dtype("int32")),
                               }, _builder)


@extern.extern
def double2hiint(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp64"),): ("__nv_double2hiint", core.dtype("int32")),
                               }, _builder)


@extern.extern
def float2ll_rn(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_float2ll_rn", core.dtype("int64")),
                               }, _builder)


@extern.extern
def float2ll_rz(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_float2ll_rz", core.dtype("int64")),
                               }, _builder)


@extern.extern
def float2ll_rd(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_float2ll_rd", core.dtype("int64")),
                               }, _builder)


@extern.extern
def float2ll_ru(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_float2ll_ru", core.dtype("int64")),
                               }, _builder)


@extern.extern
def float2ull_rn(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_float2ull_rn", core.dtype("int64")),
                               }, _builder)


@extern.extern
def float2ull_rz(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_float2ull_rz", core.dtype("int64")),
                               }, _builder)


@extern.extern
def float2ull_rd(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_float2ull_rd", core.dtype("int64")),
                               }, _builder)


@extern.extern
def float2ull_ru(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_float2ull_ru", core.dtype("int64")),
                               }, _builder)


@extern.extern
def double2ll_rn(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp64"),): ("__nv_double2ll_rn", core.dtype("int64")),
                               }, _builder)


@extern.extern
def double2ll_rz(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp64"),): ("__nv_double2ll_rz", core.dtype("int64")),
                               }, _builder)


@extern.extern
def double2ll_rd(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp64"),): ("__nv_double2ll_rd", core.dtype("int64")),
                               }, _builder)


@extern.extern
def double2ll_ru(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp64"),): ("__nv_double2ll_ru", core.dtype("int64")),
                               }, _builder)


@extern.extern
def double2ull_rn(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp64"),): ("__nv_double2ull_rn", core.dtype("int64")),
                               }, _builder)


@extern.extern
def double2ull_rz(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp64"),): ("__nv_double2ull_rz", core.dtype("int64")),
                               }, _builder)


@extern.extern
def double2ull_rd(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp64"),): ("__nv_double2ull_rd", core.dtype("int64")),
                               }, _builder)


@extern.extern
def double2ull_ru(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp64"),): ("__nv_double2ull_ru", core.dtype("int64")),
                               }, _builder)


@extern.extern
def ll2float_rn(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("int64"),): ("__nv_ll2float_rn", core.dtype("fp32")),
                               (core.dtype("uint64"),): ("__nv_ull2float_rn", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def ll2float_rz(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("int64"),): ("__nv_ll2float_rz", core.dtype("fp32")),
                               (core.dtype("uint64"),): ("__nv_ull2float_rz", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def ll2float_rd(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("int64"),): ("__nv_ll2float_rd", core.dtype("fp32")),
                               (core.dtype("uint64"),): ("__nv_ull2float_rd", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def ll2float_ru(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("int64"),): ("__nv_ll2float_ru", core.dtype("fp32")),
                               (core.dtype("uint64"),): ("__nv_ull2float_ru", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def ll2double_rn(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("int64"),): ("__nv_ll2double_rn", core.dtype("fp64")),
                               (core.dtype("uint64"),): ("__nv_ull2double_rn", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def ll2double_rz(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("int64"),): ("__nv_ll2double_rz", core.dtype("fp64")),
                               (core.dtype("uint64"),): ("__nv_ull2double_rz", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def ll2double_rd(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("int64"),): ("__nv_ll2double_rd", core.dtype("fp64")),
                               (core.dtype("uint64"),): ("__nv_ull2double_rd", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def ll2double_ru(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("int64"),): ("__nv_ll2double_ru", core.dtype("fp64")),
                               (core.dtype("uint64"),): ("__nv_ull2double_ru", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def int_as_float(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("int32"),): ("__nv_int_as_float", core.dtype("fp32")),
                               (core.dtype("uint32"),): ("__nv_uint_as_float", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def float_as_int(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_float_as_int", core.dtype("int32")),
                               }, _builder)


@extern.extern
def float_as_uint(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_float_as_uint", core.dtype("int32")),
                               }, _builder)


@extern.extern
def longlong_as_double(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("int64"),): ("__nv_longlong_as_double", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def double_as_longlong(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp64"),): ("__nv_double_as_longlong", core.dtype("int64")),
                               }, _builder)


@extern.extern
def fast_sinf(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_fast_sinf", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def fast_cosf(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_fast_cosf", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def fast_log2f(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_fast_log2f", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def fast_logf(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_fast_logf", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def fast_expf(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_fast_expf", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def fast_tanf(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_fast_tanf", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def fast_exp10f(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_fast_exp10f", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def fast_log10f(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_fast_log10f", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def pow(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp32"), core.dtype("fp32"),): ("__nv_fast_powf", core.dtype("fp32")),
                               (core.dtype("fp32"), core.dtype("fp32"),): ("__nv_powf", core.dtype("fp32")),
                               (core.dtype("fp64"), core.dtype("fp64"),): ("__nv_pow", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def hadd(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("int32"), core.dtype("int32"),): ("__nv_hadd", core.dtype("int32")),
                               (core.dtype("uint32"), core.dtype("uint32"),): ("__nv_uhadd", core.dtype("uint32")),
                               }, _builder)


@extern.extern
def rhadd(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("int32"), core.dtype("int32"),): ("__nv_rhadd", core.dtype("int32")),
                               (core.dtype("uint32"), core.dtype("uint32"),): ("__nv_urhadd", core.dtype("uint32")),
                               }, _builder)


@extern.extern
def fsub_rn(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp32"), core.dtype("fp32"),): ("__nv_fsub_rn", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def fsub_rz(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp32"), core.dtype("fp32"),): ("__nv_fsub_rz", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def fsub_rd(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp32"), core.dtype("fp32"),): ("__nv_fsub_rd", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def fsub_ru(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp32"), core.dtype("fp32"),): ("__nv_fsub_ru", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def frsqrt_rn(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_frsqrt_rn", core.dtype("fp32")),
                               }, _builder)


@extern.extern
def ffs(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("int32"),): ("__nv_ffs", core.dtype("int32")),
                               (core.dtype("int64"),): ("__nv_ffsll", core.dtype("int32")),
                               }, _builder)


@extern.extern
def rint(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_rintf", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_rint", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def llrint(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_llrintf", core.dtype("int64")),
                               (core.dtype("fp64"),): ("__nv_llrint", core.dtype("int64")),
                               }, _builder)


@extern.extern
def nearbyint(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_nearbyintf", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_nearbyint", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def isnanf(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_isnanf", core.dtype("int32")),
                               }, _builder)


@extern.extern
def signbitf(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_signbitf", core.dtype("int32")),
                               }, _builder)


@extern.extern
def copysign(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp32"), core.dtype("fp32"),): ("__nv_copysignf", core.dtype("fp32")),
                               (core.dtype("fp64"), core.dtype("fp64"),): ("__nv_copysign", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def finitef(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_finitef", core.dtype("int32")),
                               }, _builder)


@extern.extern
def isinff(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_isinff", core.dtype("int32")),
                               }, _builder)


@extern.extern
def nextafter(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp32"), core.dtype("fp32"),): ("__nv_nextafterf", core.dtype("fp32")),
                               (core.dtype("fp64"), core.dtype("fp64"),): ("__nv_nextafter", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def sin(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_sinf", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_sin", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def cos(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_cosf", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_cos", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def sinpi(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_sinpif", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_sinpi", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def cospi(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_cospif", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_cospi", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def tan(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_tanf", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_tan", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def log2(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_log2f", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_log2", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def exp(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_expf", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_exp", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def exp10(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_exp10f", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_exp10", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def cosh(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_coshf", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_cosh", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def sinh(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_sinhf", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_sinh", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def tanh(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_tanhf", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_tanh", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def atan2(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp32"), core.dtype("fp32"),): ("__nv_atan2f", core.dtype("fp32")),
                               (core.dtype("fp64"), core.dtype("fp64"),): ("__nv_atan2", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def atan(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_atanf", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_atan", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def asin(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_asinf", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_asin", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def acos(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_acosf", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_acos", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def log(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_logf", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_log", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def log10(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_log10f", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_log10", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def log1p(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_log1pf", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_log1p", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def acosh(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_acoshf", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_acosh", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def asinh(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_asinhf", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_asinh", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def atanh(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_atanhf", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_atanh", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def expm1(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_expm1f", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_expm1", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def hypot(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp32"), core.dtype("fp32"),): ("__nv_hypotf", core.dtype("fp32")),
                               (core.dtype("fp64"), core.dtype("fp64"),): ("__nv_hypot", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def rhypot(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp32"), core.dtype("fp32"),): ("__nv_rhypotf", core.dtype("fp32")),
                               (core.dtype("fp64"), core.dtype("fp64"),): ("__nv_rhypot", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def norm3d(arg0, arg1, arg2, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, arg2, ],
                              {(core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32"),): ("__nv_norm3df", core.dtype("fp32")),
                               (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64"),): ("__nv_norm3d", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def rnorm3d(arg0, arg1, arg2, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, arg2, ],
                              {(core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32"),): ("__nv_rnorm3df", core.dtype("fp32")),
                               (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64"),): ("__nv_rnorm3d", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def norm4d(arg0, arg1, arg2, arg3, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, arg2, arg3, ],
                              {(core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32"),): ("__nv_norm4df", core.dtype("fp32")),
                               (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64"),): ("__nv_norm4d", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def rnorm4d(arg0, arg1, arg2, arg3, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, arg2, arg3, ],
                              {(core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32"),): ("__nv_rnorm4df", core.dtype("fp32")),
                               (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64"),): ("__nv_rnorm4d", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def cbrt(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_cbrtf", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_cbrt", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def rcbrt(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_rcbrtf", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_rcbrt", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def j0(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_j0f", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_j0", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def j1(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_j1f", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_j1", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def y0(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_y0f", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_y0", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def y1(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_y1f", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_y1", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def yn(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("int32"), core.dtype("fp32"),): ("__nv_ynf", core.dtype("fp32")),
                               (core.dtype("int32"), core.dtype("fp64"),): ("__nv_yn", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def jn(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("int32"), core.dtype("fp32"),): ("__nv_jnf", core.dtype("fp32")),
                               (core.dtype("int32"), core.dtype("fp64"),): ("__nv_jn", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def cyl_bessel_i0(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_cyl_bessel_i0f", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_cyl_bessel_i0", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def cyl_bessel_i1(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_cyl_bessel_i1f", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_cyl_bessel_i1", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def erf(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_erff", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_erf", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def erfinv(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_erfinvf", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_erfinv", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def erfc(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_erfcf", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_erfc", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def erfcx(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_erfcxf", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_erfcx", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def erfcinv(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_erfcinvf", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_erfcinv", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def normcdfinv(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_normcdfinvf", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_normcdfinv", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def normcdf(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_normcdff", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_normcdf", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def lgamma(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_lgammaf", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_lgamma", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def ldexp(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp32"), core.dtype("int32"),): ("__nv_ldexpf", core.dtype("fp32")),
                               (core.dtype("fp64"), core.dtype("int32"),): ("__nv_ldexp", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def scalbn(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp32"), core.dtype("int32"),): ("__nv_scalbnf", core.dtype("fp32")),
                               (core.dtype("fp64"), core.dtype("int32"),): ("__nv_scalbn", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def fmod(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp32"), core.dtype("fp32"),): ("__nv_fmodf", core.dtype("fp32")),
                               (core.dtype("fp64"), core.dtype("fp64"),): ("__nv_fmod", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def remainder(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp32"), core.dtype("fp32"),): ("__nv_remainderf", core.dtype("fp32")),
                               (core.dtype("fp64"), core.dtype("fp64"),): ("__nv_remainder", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def fma(arg0, arg1, arg2, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, arg2, ],
                              {(core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32"),): ("__nv_fmaf", core.dtype("fp32")),
                               (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64"),): ("__nv_fma", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def powi(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp32"), core.dtype("int32"),): ("__nv_powif", core.dtype("fp32")),
                               (core.dtype("fp64"), core.dtype("int32"),): ("__nv_powi", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def tgamma(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_tgammaf", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_tgamma", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def round(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_roundf", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_round", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def llround(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_llroundf", core.dtype("int64")),
                               (core.dtype("fp64"),): ("__nv_llround", core.dtype("int64")),
                               }, _builder)


@extern.extern
def fdim(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp32"), core.dtype("fp32"),): ("__nv_fdimf", core.dtype("fp32")),
                               (core.dtype("fp64"), core.dtype("fp64"),): ("__nv_fdim", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def ilogb(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_ilogbf", core.dtype("int32")),
                               (core.dtype("fp64"),): ("__nv_ilogb", core.dtype("int32")),
                               }, _builder)


@extern.extern
def logb(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp32"),): ("__nv_logbf", core.dtype("fp32")),
                               (core.dtype("fp64"),): ("__nv_logb", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def signbitd(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp64"),): ("__nv_signbitd", core.dtype("int32")),
                               }, _builder)


@extern.extern
def isfinited(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp64"),): ("__nv_isfinited", core.dtype("int32")),
                               }, _builder)


@extern.extern
def isinfd(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp64"),): ("__nv_isinfd", core.dtype("int32")),
                               }, _builder)


@extern.extern
def isnand(arg0, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, ],
                              {(core.dtype("fp64"),): ("__nv_isnand", core.dtype("int32")),
                               }, _builder)


@extern.extern
def dsub_rn(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp64"), core.dtype("fp64"),): ("__nv_dsub_rn", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def dsub_rz(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp64"), core.dtype("fp64"),): ("__nv_dsub_rz", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def dsub_ru(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp64"), core.dtype("fp64"),): ("__nv_dsub_ru", core.dtype("fp64")),
                               }, _builder)


@extern.extern
def dsub_rd(arg0, arg1, _builder=None):
    return extern.elementwise("libdevice", LIBDEVICE_PATH, [arg0, arg1, ],
                              {(core.dtype("fp64"), core.dtype("fp64"),): ("__nv_dsub_rd", core.dtype("fp64")),
                               }, _builder)
