import os

from .. import impl
from . import extern

import triton.language as tl

LIBDEVICE_PATH = os.path.dirname(os.path.abspath(__file__)) + "/libdevice.10.bc"


@impl.extern
def clz(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("int32"),): ("__nv_clz", tl.dtype("int32")),
            (tl.dtype("int64"),): ("__nv_clzll", tl.dtype("int32")),
        },
        _builder,
    )


@impl.extern
def popc(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("int32"),): ("__nv_popc", tl.dtype("int32")),
            (tl.dtype("int64"),): ("__nv_popcll", tl.dtype("int32")),
        },
        _builder,
    )


@impl.extern
def byte_perm(arg0, arg1, arg2, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
            arg2,
        ],
        {
            (
                tl.dtype("int32"),
                tl.dtype("int32"),
                tl.dtype("int32"),
            ): ("__nv_byte_perm", tl.dtype("int32")),
        },
        _builder,
    )


@impl.extern
def min(arg0, arg1, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
        ],
        {
            (
                tl.dtype("int32"),
                tl.dtype("int32"),
            ): ("__nv_min", tl.dtype("int32")),
            (
                tl.dtype("uint32"),
                tl.dtype("uint32"),
            ): ("__nv_umin", tl.dtype("uint32")),
            (
                tl.dtype("int64"),
                tl.dtype("int64"),
            ): ("__nv_llmin", tl.dtype("int64")),
            (
                tl.dtype("uint64"),
                tl.dtype("uint64"),
            ): ("__nv_ullmin", tl.dtype("uint64")),
            (
                tl.dtype("fp32"),
                tl.dtype("fp32"),
            ): ("__nv_fminf", tl.dtype("fp32")),
            (
                tl.dtype("fp64"),
                tl.dtype("fp64"),
            ): ("__nv_fmin", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def max(arg0, arg1, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
        ],
        {
            (
                tl.dtype("int32"),
                tl.dtype("int32"),
            ): ("__nv_max", tl.dtype("int32")),
            (
                tl.dtype("uint32"),
                tl.dtype("uint32"),
            ): ("__nv_umax", tl.dtype("uint32")),
            (
                tl.dtype("int64"),
                tl.dtype("int64"),
            ): ("__nv_llmax", tl.dtype("int64")),
            (
                tl.dtype("uint64"),
                tl.dtype("uint64"),
            ): ("__nv_ullmax", tl.dtype("uint64")),
            (
                tl.dtype("fp32"),
                tl.dtype("fp32"),
            ): ("__nv_fmaxf", tl.dtype("fp32")),
            (
                tl.dtype("fp64"),
                tl.dtype("fp64"),
            ): ("__nv_fmax", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def mulhi(arg0, arg1, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
        ],
        {
            (
                tl.dtype("int32"),
                tl.dtype("int32"),
            ): ("__nv_mulhi", tl.dtype("int32")),
            (
                tl.dtype("uint32"),
                tl.dtype("uint32"),
            ): ("__nv_umulhi", tl.dtype("uint32")),
            (
                tl.dtype("int64"),
                tl.dtype("int64"),
            ): ("__nv_mul64hi", tl.dtype("int64")),
            (
                tl.dtype("uint64"),
                tl.dtype("uint64"),
            ): ("__nv_umul64hi", tl.dtype("uint64")),
        },
        _builder,
    )


@impl.extern
def mul24(arg0, arg1, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
        ],
        {
            (
                tl.dtype("int32"),
                tl.dtype("int32"),
            ): ("__nv_mul24", tl.dtype("int32")),
            (
                tl.dtype("uint32"),
                tl.dtype("uint32"),
            ): ("__nv_umul24", tl.dtype("uint32")),
        },
        _builder,
    )


@impl.extern
def brev(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("int32"),): ("__nv_brev", tl.dtype("int32")),
            (tl.dtype("int64"),): ("__nv_brevll", tl.dtype("int64")),
        },
        _builder,
    )


@impl.extern
def sad(arg0, arg1, arg2, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
            arg2,
        ],
        {
            (
                tl.dtype("int32"),
                tl.dtype("int32"),
                tl.dtype("uint32"),
            ): ("__nv_sad", tl.dtype("int32")),
            (
                tl.dtype("uint32"),
                tl.dtype("uint32"),
                tl.dtype("uint32"),
            ): ("__nv_usad", tl.dtype("uint32")),
        },
        _builder,
    )


@impl.extern
def abs(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("int32"),): ("__nv_abs", tl.dtype("int32")),
            (tl.dtype("int64"),): ("__nv_llabs", tl.dtype("int64")),
            (tl.dtype("fp32"),): ("__nv_fabsf", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_fabs", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def floor(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_floorf", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_floor", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def rcp64h(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp64"),): ("__nv_rcp64h", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def rsqrt(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_rsqrtf", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_rsqrt", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def ceil(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp64"),): ("__nv_ceil", tl.dtype("fp64")),
            (tl.dtype("fp32"),): ("__nv_ceilf", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def trunc(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp64"),): ("__nv_trunc", tl.dtype("fp64")),
            (tl.dtype("fp32"),): ("__nv_truncf", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def exp2(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_exp2f", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_exp2", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def saturatef(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_saturatef", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def fma_rn(arg0, arg1, arg2, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
            arg2,
        ],
        {
            (
                tl.dtype("fp32"),
                tl.dtype("fp32"),
                tl.dtype("fp32"),
            ): ("__nv_fmaf_rn", tl.dtype("fp32")),
            (
                tl.dtype("fp64"),
                tl.dtype("fp64"),
                tl.dtype("fp64"),
            ): ("__nv_fma_rn", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def fma_rz(arg0, arg1, arg2, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
            arg2,
        ],
        {
            (
                tl.dtype("fp32"),
                tl.dtype("fp32"),
                tl.dtype("fp32"),
            ): ("__nv_fmaf_rz", tl.dtype("fp32")),
            (
                tl.dtype("fp64"),
                tl.dtype("fp64"),
                tl.dtype("fp64"),
            ): ("__nv_fma_rz", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def fma_rd(arg0, arg1, arg2, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
            arg2,
        ],
        {
            (
                tl.dtype("fp32"),
                tl.dtype("fp32"),
                tl.dtype("fp32"),
            ): ("__nv_fmaf_rd", tl.dtype("fp32")),
            (
                tl.dtype("fp64"),
                tl.dtype("fp64"),
                tl.dtype("fp64"),
            ): ("__nv_fma_rd", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def fma_ru(arg0, arg1, arg2, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
            arg2,
        ],
        {
            (
                tl.dtype("fp32"),
                tl.dtype("fp32"),
                tl.dtype("fp32"),
            ): ("__nv_fmaf_ru", tl.dtype("fp32")),
            (
                tl.dtype("fp64"),
                tl.dtype("fp64"),
                tl.dtype("fp64"),
            ): ("__nv_fma_ru", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def fast_dividef(arg0, arg1, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
        ],
        {
            (
                tl.dtype("fp32"),
                tl.dtype("fp32"),
            ): ("__nv_fast_fdividef", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def div_rn(arg0, arg1, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
        ],
        {
            (
                tl.dtype("fp32"),
                tl.dtype("fp32"),
            ): ("__nv_fdiv_rn", tl.dtype("fp32")),
            (
                tl.dtype("fp64"),
                tl.dtype("fp64"),
            ): ("__nv_ddiv_rn", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def div_rz(arg0, arg1, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
        ],
        {
            (
                tl.dtype("fp32"),
                tl.dtype("fp32"),
            ): ("__nv_fdiv_rz", tl.dtype("fp32")),
            (
                tl.dtype("fp64"),
                tl.dtype("fp64"),
            ): ("__nv_ddiv_rz", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def div_rd(arg0, arg1, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
        ],
        {
            (
                tl.dtype("fp32"),
                tl.dtype("fp32"),
            ): ("__nv_fdiv_rd", tl.dtype("fp32")),
            (
                tl.dtype("fp64"),
                tl.dtype("fp64"),
            ): ("__nv_ddiv_rd", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def div_ru(arg0, arg1, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
        ],
        {
            (
                tl.dtype("fp32"),
                tl.dtype("fp32"),
            ): ("__nv_fdiv_ru", tl.dtype("fp32")),
            (
                tl.dtype("fp64"),
                tl.dtype("fp64"),
            ): ("__nv_ddiv_ru", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def rcp_rn(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_frcp_rn", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_drcp_rn", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def rcp_rz(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_frcp_rz", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_drcp_rz", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def rcp_rd(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_frcp_rd", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_drcp_rd", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def rcp_ru(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_frcp_ru", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_drcp_ru", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def sqrt_rn(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_fsqrt_rn", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_dsqrt_rn", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def sqrt_rz(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_fsqrt_rz", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_dsqrt_rz", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def sqrt_rd(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_fsqrt_rd", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_dsqrt_rd", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def sqrt_ru(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_fsqrt_ru", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_dsqrt_ru", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def sqrt(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_sqrtf", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_sqrt", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def add_rn(arg0, arg1, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
        ],
        {
            (
                tl.dtype("fp64"),
                tl.dtype("fp64"),
            ): ("__nv_dadd_rn", tl.dtype("fp64")),
            (
                tl.dtype("fp32"),
                tl.dtype("fp32"),
            ): ("__nv_fadd_rn", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def add_rz(arg0, arg1, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
        ],
        {
            (
                tl.dtype("fp64"),
                tl.dtype("fp64"),
            ): ("__nv_dadd_rz", tl.dtype("fp64")),
            (
                tl.dtype("fp32"),
                tl.dtype("fp32"),
            ): ("__nv_fadd_rz", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def add_rd(arg0, arg1, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
        ],
        {
            (
                tl.dtype("fp64"),
                tl.dtype("fp64"),
            ): ("__nv_dadd_rd", tl.dtype("fp64")),
            (
                tl.dtype("fp32"),
                tl.dtype("fp32"),
            ): ("__nv_fadd_rd", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def add_ru(arg0, arg1, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
        ],
        {
            (
                tl.dtype("fp64"),
                tl.dtype("fp64"),
            ): ("__nv_dadd_ru", tl.dtype("fp64")),
            (
                tl.dtype("fp32"),
                tl.dtype("fp32"),
            ): ("__nv_fadd_ru", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def mul_rn(arg0, arg1, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
        ],
        {
            (
                tl.dtype("fp64"),
                tl.dtype("fp64"),
            ): ("__nv_dmul_rn", tl.dtype("fp64")),
            (
                tl.dtype("fp32"),
                tl.dtype("fp32"),
            ): ("__nv_fmul_rn", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def mul_rz(arg0, arg1, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
        ],
        {
            (
                tl.dtype("fp64"),
                tl.dtype("fp64"),
            ): ("__nv_dmul_rz", tl.dtype("fp64")),
            (
                tl.dtype("fp32"),
                tl.dtype("fp32"),
            ): ("__nv_fmul_rz", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def mul_rd(arg0, arg1, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
        ],
        {
            (
                tl.dtype("fp64"),
                tl.dtype("fp64"),
            ): ("__nv_dmul_rd", tl.dtype("fp64")),
            (
                tl.dtype("fp32"),
                tl.dtype("fp32"),
            ): ("__nv_fmul_rd", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def mul_ru(arg0, arg1, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
        ],
        {
            (
                tl.dtype("fp64"),
                tl.dtype("fp64"),
            ): ("__nv_dmul_ru", tl.dtype("fp64")),
            (
                tl.dtype("fp32"),
                tl.dtype("fp32"),
            ): ("__nv_fmul_ru", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def double2float_rn(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp64"),): ("__nv_double2float_rn", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def double2float_rz(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp64"),): ("__nv_double2float_rz", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def double2float_rd(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp64"),): ("__nv_double2float_rd", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def double2float_ru(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp64"),): ("__nv_double2float_ru", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def double2int_rn(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp64"),): ("__nv_double2int_rn", tl.dtype("int32")),
        },
        _builder,
    )


@impl.extern
def double2int_rz(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp64"),): ("__nv_double2int_rz", tl.dtype("int32")),
        },
        _builder,
    )


@impl.extern
def double2int_rd(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp64"),): ("__nv_double2int_rd", tl.dtype("int32")),
        },
        _builder,
    )


@impl.extern
def double2int_ru(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp64"),): ("__nv_double2int_ru", tl.dtype("int32")),
        },
        _builder,
    )


@impl.extern
def double2uint_rn(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp64"),): ("__nv_double2uint_rn", tl.dtype("int32")),
        },
        _builder,
    )


@impl.extern
def double2uint_rz(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp64"),): ("__nv_double2uint_rz", tl.dtype("int32")),
        },
        _builder,
    )


@impl.extern
def double2uint_rd(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp64"),): ("__nv_double2uint_rd", tl.dtype("int32")),
        },
        _builder,
    )


@impl.extern
def double2uint_ru(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp64"),): ("__nv_double2uint_ru", tl.dtype("int32")),
        },
        _builder,
    )


@impl.extern
def int2double_rn(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("int32"),): ("__nv_int2double_rn", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def uint2double_rn(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("uint32"),): ("__nv_uint2double_rn", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def float2int_rn(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_float2int_rn", tl.dtype("int32")),
        },
        _builder,
    )


@impl.extern
def float2int_rz(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_float2int_rz", tl.dtype("int32")),
        },
        _builder,
    )


@impl.extern
def float2int_rd(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_float2int_rd", tl.dtype("int32")),
        },
        _builder,
    )


@impl.extern
def float2int_ru(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_float2int_ru", tl.dtype("int32")),
        },
        _builder,
    )


@impl.extern
def float2uint_rn(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_float2uint_rn", tl.dtype("int32")),
        },
        _builder,
    )


@impl.extern
def float2uint_rz(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_float2uint_rz", tl.dtype("int32")),
        },
        _builder,
    )


@impl.extern
def float2uint_rd(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_float2uint_rd", tl.dtype("int32")),
        },
        _builder,
    )


@impl.extern
def float2uint_ru(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_float2uint_ru", tl.dtype("int32")),
        },
        _builder,
    )


@impl.extern
def int2float_rn(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("int32"),): ("__nv_int2float_rn", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def int2float_rz(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("int32"),): ("__nv_int2float_rz", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def int2float_rd(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("int32"),): ("__nv_int2float_rd", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def int2float_ru(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("int32"),): ("__nv_int2float_ru", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def uint2float_rn(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("uint32"),): ("__nv_uint2float_rn", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def uint2float_rz(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("uint32"),): ("__nv_uint2float_rz", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def uint2float_rd(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("uint32"),): ("__nv_uint2float_rd", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def uint2float_ru(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("uint32"),): ("__nv_uint2float_ru", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def hiloint2double(arg0, arg1, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
        ],
        {
            (
                tl.dtype("int32"),
                tl.dtype("int32"),
            ): ("__nv_hiloint2double", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def double2loint(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp64"),): ("__nv_double2loint", tl.dtype("int32")),
        },
        _builder,
    )


@impl.extern
def double2hiint(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp64"),): ("__nv_double2hiint", tl.dtype("int32")),
        },
        _builder,
    )


@impl.extern
def float2ll_rn(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_float2ll_rn", tl.dtype("int64")),
        },
        _builder,
    )


@impl.extern
def float2ll_rz(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_float2ll_rz", tl.dtype("int64")),
        },
        _builder,
    )


@impl.extern
def float2ll_rd(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_float2ll_rd", tl.dtype("int64")),
        },
        _builder,
    )


@impl.extern
def float2ll_ru(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_float2ll_ru", tl.dtype("int64")),
        },
        _builder,
    )


@impl.extern
def float2ull_rn(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_float2ull_rn", tl.dtype("int64")),
        },
        _builder,
    )


@impl.extern
def float2ull_rz(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_float2ull_rz", tl.dtype("int64")),
        },
        _builder,
    )


@impl.extern
def float2ull_rd(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_float2ull_rd", tl.dtype("int64")),
        },
        _builder,
    )


@impl.extern
def float2ull_ru(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_float2ull_ru", tl.dtype("int64")),
        },
        _builder,
    )


@impl.extern
def double2ll_rn(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp64"),): ("__nv_double2ll_rn", tl.dtype("int64")),
        },
        _builder,
    )


@impl.extern
def double2ll_rz(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp64"),): ("__nv_double2ll_rz", tl.dtype("int64")),
        },
        _builder,
    )


@impl.extern
def double2ll_rd(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp64"),): ("__nv_double2ll_rd", tl.dtype("int64")),
        },
        _builder,
    )


@impl.extern
def double2ll_ru(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp64"),): ("__nv_double2ll_ru", tl.dtype("int64")),
        },
        _builder,
    )


@impl.extern
def double2ull_rn(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp64"),): ("__nv_double2ull_rn", tl.dtype("int64")),
        },
        _builder,
    )


@impl.extern
def double2ull_rz(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp64"),): ("__nv_double2ull_rz", tl.dtype("int64")),
        },
        _builder,
    )


@impl.extern
def double2ull_rd(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp64"),): ("__nv_double2ull_rd", tl.dtype("int64")),
        },
        _builder,
    )


@impl.extern
def double2ull_ru(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp64"),): ("__nv_double2ull_ru", tl.dtype("int64")),
        },
        _builder,
    )


@impl.extern
def ll2float_rn(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("int64"),): ("__nv_ll2float_rn", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def ll2float_rz(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("int64"),): ("__nv_ll2float_rz", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def ll2float_rd(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("int64"),): ("__nv_ll2float_rd", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def ll2float_ru(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("int64"),): ("__nv_ll2float_ru", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def ull2float_rn(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("uint64"),): ("__nv_ull2float_rn", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def ull2float_rz(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("uint64"),): ("__nv_ull2float_rz", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def ull2float_rd(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("uint64"),): ("__nv_ull2float_rd", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def ull2float_ru(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("uint64"),): ("__nv_ull2float_ru", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def ll2double_rn(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("int64"),): ("__nv_ll2double_rn", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def ll2double_rz(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("int64"),): ("__nv_ll2double_rz", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def ll2double_rd(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("int64"),): ("__nv_ll2double_rd", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def ll2double_ru(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("int64"),): ("__nv_ll2double_ru", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def ull2double_rn(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("uint64"),): ("__nv_ull2double_rn", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def ull2double_rz(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("uint64"),): ("__nv_ull2double_rz", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def ull2double_rd(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("uint64"),): ("__nv_ull2double_rd", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def ull2double_ru(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("uint64"),): ("__nv_ull2double_ru", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def int_as_float(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("int32"),): ("__nv_int_as_float", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def float_as_int(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_float_as_int", tl.dtype("int32")),
        },
        _builder,
    )


@impl.extern
def uint_as_float(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("uint32"),): ("__nv_uint_as_float", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def float_as_uint(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_float_as_uint", tl.dtype("int32")),
        },
        _builder,
    )


@impl.extern
def longlong_as_double(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("int64"),): ("__nv_longlong_as_double", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def double_as_longlong(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp64"),): ("__nv_double_as_longlong", tl.dtype("int64")),
        },
        _builder,
    )


@impl.extern
def fast_sinf(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_fast_sinf", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def fast_cosf(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_fast_cosf", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def fast_log2f(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_fast_log2f", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def fast_logf(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_fast_logf", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def fast_expf(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_fast_expf", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def fast_tanf(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_fast_tanf", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def fast_exp10f(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_fast_exp10f", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def fast_log10f(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_fast_log10f", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def fast_powf(arg0, arg1, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
        ],
        {
            (
                tl.dtype("fp32"),
                tl.dtype("fp32"),
            ): ("__nv_fast_powf", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def hadd(arg0, arg1, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
        ],
        {
            (
                tl.dtype("int32"),
                tl.dtype("int32"),
            ): ("__nv_hadd", tl.dtype("int32")),
            (
                tl.dtype("uint32"),
                tl.dtype("uint32"),
            ): ("__nv_uhadd", tl.dtype("uint32")),
        },
        _builder,
    )


@impl.extern
def rhadd(arg0, arg1, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
        ],
        {
            (
                tl.dtype("int32"),
                tl.dtype("int32"),
            ): ("__nv_rhadd", tl.dtype("int32")),
            (
                tl.dtype("uint32"),
                tl.dtype("uint32"),
            ): ("__nv_urhadd", tl.dtype("uint32")),
        },
        _builder,
    )


@impl.extern
def sub_rn(arg0, arg1, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
        ],
        {
            (
                tl.dtype("fp32"),
                tl.dtype("fp32"),
            ): ("__nv_fsub_rn", tl.dtype("fp32")),
            (
                tl.dtype("fp64"),
                tl.dtype("fp64"),
            ): ("__nv_dsub_rn", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def sub_rz(arg0, arg1, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
        ],
        {
            (
                tl.dtype("fp32"),
                tl.dtype("fp32"),
            ): ("__nv_fsub_rz", tl.dtype("fp32")),
            (
                tl.dtype("fp64"),
                tl.dtype("fp64"),
            ): ("__nv_dsub_rz", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def sub_rd(arg0, arg1, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
        ],
        {
            (
                tl.dtype("fp32"),
                tl.dtype("fp32"),
            ): ("__nv_fsub_rd", tl.dtype("fp32")),
            (
                tl.dtype("fp64"),
                tl.dtype("fp64"),
            ): ("__nv_dsub_rd", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def sub_ru(arg0, arg1, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
        ],
        {
            (
                tl.dtype("fp32"),
                tl.dtype("fp32"),
            ): ("__nv_fsub_ru", tl.dtype("fp32")),
            (
                tl.dtype("fp64"),
                tl.dtype("fp64"),
            ): ("__nv_dsub_ru", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def rsqrt_rn(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_frsqrt_rn", tl.dtype("fp32")),
        },
        _builder,
    )


@impl.extern
def ffs(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("int32"),): ("__nv_ffs", tl.dtype("int32")),
            (tl.dtype("int64"),): ("__nv_ffsll", tl.dtype("int32")),
        },
        _builder,
    )


@impl.extern
def rint(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_rintf", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_rint", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def llrint(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_llrintf", tl.dtype("int64")),
            (tl.dtype("fp64"),): ("__nv_llrint", tl.dtype("int64")),
        },
        _builder,
    )


@impl.extern
def nearbyint(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_nearbyintf", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_nearbyint", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def isnan(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_isnanf", tl.dtype("int32")),
            (tl.dtype("fp64"),): ("__nv_isnand", tl.dtype("int32")),
        },
        _builder,
    )


@impl.extern
def signbit(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_signbitf", tl.dtype("int32")),
            (tl.dtype("fp64"),): ("__nv_signbitd", tl.dtype("int32")),
        },
        _builder,
    )


@impl.extern
def copysign(arg0, arg1, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
        ],
        {
            (
                tl.dtype("fp32"),
                tl.dtype("fp32"),
            ): ("__nv_copysignf", tl.dtype("fp32")),
            (
                tl.dtype("fp64"),
                tl.dtype("fp64"),
            ): ("__nv_copysign", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def finitef(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_finitef", tl.dtype("int32")),
        },
        _builder,
    )


@impl.extern
def isinf(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_isinff", tl.dtype("int32")),
            (tl.dtype("fp64"),): ("__nv_isinfd", tl.dtype("int32")),
        },
        _builder,
    )


@impl.extern
def nextafter(arg0, arg1, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
        ],
        {
            (
                tl.dtype("fp32"),
                tl.dtype("fp32"),
            ): ("__nv_nextafterf", tl.dtype("fp32")),
            (
                tl.dtype("fp64"),
                tl.dtype("fp64"),
            ): ("__nv_nextafter", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def sin(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_sinf", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_sin", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def cos(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_cosf", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_cos", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def sinpi(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_sinpif", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_sinpi", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def cospi(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_cospif", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_cospi", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def tan(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_tanf", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_tan", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def log2(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_log2f", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_log2", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def exp(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_expf", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_exp", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def exp10(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_exp10f", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_exp10", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def cosh(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_coshf", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_cosh", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def sinh(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_sinhf", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_sinh", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def tanh(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_tanhf", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_tanh", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def atan2(arg0, arg1, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
        ],
        {
            (
                tl.dtype("fp32"),
                tl.dtype("fp32"),
            ): ("__nv_atan2f", tl.dtype("fp32")),
            (
                tl.dtype("fp64"),
                tl.dtype("fp64"),
            ): ("__nv_atan2", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def atan(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_atanf", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_atan", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def asin(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_asinf", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_asin", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def acos(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_acosf", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_acos", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def log(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_logf", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_log", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def log10(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_log10f", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_log10", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def log1p(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_log1pf", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_log1p", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def acosh(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_acoshf", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_acosh", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def asinh(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_asinhf", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_asinh", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def atanh(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_atanhf", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_atanh", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def expm1(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_expm1f", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_expm1", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def hypot(arg0, arg1, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
        ],
        {
            (
                tl.dtype("fp32"),
                tl.dtype("fp32"),
            ): ("__nv_hypotf", tl.dtype("fp32")),
            (
                tl.dtype("fp64"),
                tl.dtype("fp64"),
            ): ("__nv_hypot", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def rhypot(arg0, arg1, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
        ],
        {
            (
                tl.dtype("fp32"),
                tl.dtype("fp32"),
            ): ("__nv_rhypotf", tl.dtype("fp32")),
            (
                tl.dtype("fp64"),
                tl.dtype("fp64"),
            ): ("__nv_rhypot", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def norm3d(arg0, arg1, arg2, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
            arg2,
        ],
        {
            (
                tl.dtype("fp32"),
                tl.dtype("fp32"),
                tl.dtype("fp32"),
            ): ("__nv_norm3df", tl.dtype("fp32")),
            (
                tl.dtype("fp64"),
                tl.dtype("fp64"),
                tl.dtype("fp64"),
            ): ("__nv_norm3d", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def rnorm3d(arg0, arg1, arg2, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
            arg2,
        ],
        {
            (
                tl.dtype("fp32"),
                tl.dtype("fp32"),
                tl.dtype("fp32"),
            ): ("__nv_rnorm3df", tl.dtype("fp32")),
            (
                tl.dtype("fp64"),
                tl.dtype("fp64"),
                tl.dtype("fp64"),
            ): ("__nv_rnorm3d", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def norm4d(arg0, arg1, arg2, arg3, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
            arg2,
            arg3,
        ],
        {
            (
                tl.dtype("fp32"),
                tl.dtype("fp32"),
                tl.dtype("fp32"),
                tl.dtype("fp32"),
            ): ("__nv_norm4df", tl.dtype("fp32")),
            (
                tl.dtype("fp64"),
                tl.dtype("fp64"),
                tl.dtype("fp64"),
                tl.dtype("fp64"),
            ): ("__nv_norm4d", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def rnorm4d(arg0, arg1, arg2, arg3, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
            arg2,
            arg3,
        ],
        {
            (
                tl.dtype("fp32"),
                tl.dtype("fp32"),
                tl.dtype("fp32"),
                tl.dtype("fp32"),
            ): ("__nv_rnorm4df", tl.dtype("fp32")),
            (
                tl.dtype("fp64"),
                tl.dtype("fp64"),
                tl.dtype("fp64"),
                tl.dtype("fp64"),
            ): ("__nv_rnorm4d", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def cbrt(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_cbrtf", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_cbrt", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def rcbrt(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_rcbrtf", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_rcbrt", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def j0(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_j0f", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_j0", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def j1(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_j1f", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_j1", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def y0(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_y0f", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_y0", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def y1(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_y1f", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_y1", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def yn(arg0, arg1, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
        ],
        {
            (
                tl.dtype("int32"),
                tl.dtype("fp32"),
            ): ("__nv_ynf", tl.dtype("fp32")),
            (
                tl.dtype("int32"),
                tl.dtype("fp64"),
            ): ("__nv_yn", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def jn(arg0, arg1, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
        ],
        {
            (
                tl.dtype("int32"),
                tl.dtype("fp32"),
            ): ("__nv_jnf", tl.dtype("fp32")),
            (
                tl.dtype("int32"),
                tl.dtype("fp64"),
            ): ("__nv_jn", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def cyl_bessel_i0(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_cyl_bessel_i0f", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_cyl_bessel_i0", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def cyl_bessel_i1(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_cyl_bessel_i1f", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_cyl_bessel_i1", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def erf(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_erff", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_erf", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def erfinv(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_erfinvf", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_erfinv", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def erfc(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_erfcf", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_erfc", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def erfcx(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_erfcxf", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_erfcx", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def erfcinv(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_erfcinvf", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_erfcinv", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def normcdfinv(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_normcdfinvf", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_normcdfinv", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def normcdf(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_normcdff", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_normcdf", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def lgamma(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_lgammaf", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_lgamma", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def ldexp(arg0, arg1, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
        ],
        {
            (
                tl.dtype("fp32"),
                tl.dtype("int32"),
            ): ("__nv_ldexpf", tl.dtype("fp32")),
            (
                tl.dtype("fp64"),
                tl.dtype("int32"),
            ): ("__nv_ldexp", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def scalbn(arg0, arg1, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
        ],
        {
            (
                tl.dtype("fp32"),
                tl.dtype("int32"),
            ): ("__nv_scalbnf", tl.dtype("fp32")),
            (
                tl.dtype("fp64"),
                tl.dtype("int32"),
            ): ("__nv_scalbn", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def fmod(arg0, arg1, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
        ],
        {
            (
                tl.dtype("fp32"),
                tl.dtype("fp32"),
            ): ("__nv_fmodf", tl.dtype("fp32")),
            (
                tl.dtype("fp64"),
                tl.dtype("fp64"),
            ): ("__nv_fmod", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def remainder(arg0, arg1, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
        ],
        {
            (
                tl.dtype("fp32"),
                tl.dtype("fp32"),
            ): ("__nv_remainderf", tl.dtype("fp32")),
            (
                tl.dtype("fp64"),
                tl.dtype("fp64"),
            ): ("__nv_remainder", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def fma(arg0, arg1, arg2, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
            arg2,
        ],
        {
            (
                tl.dtype("fp32"),
                tl.dtype("fp32"),
                tl.dtype("fp32"),
            ): ("__nv_fmaf", tl.dtype("fp32")),
            (
                tl.dtype("fp64"),
                tl.dtype("fp64"),
                tl.dtype("fp64"),
            ): ("__nv_fma", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def pow(arg0, arg1, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
        ],
        {
            (
                tl.dtype("fp32"),
                tl.dtype("int32"),
            ): ("__nv_powif", tl.dtype("fp32")),
            (
                tl.dtype("fp64"),
                tl.dtype("int32"),
            ): ("__nv_powi", tl.dtype("fp64")),
            (
                tl.dtype("fp32"),
                tl.dtype("fp32"),
            ): ("__nv_powf", tl.dtype("fp32")),
            (
                tl.dtype("fp64"),
                tl.dtype("fp64"),
            ): ("__nv_pow", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def tgamma(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_tgammaf", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_tgamma", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def round(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_roundf", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_round", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def llround(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_llroundf", tl.dtype("int64")),
            (tl.dtype("fp64"),): ("__nv_llround", tl.dtype("int64")),
        },
        _builder,
    )


@impl.extern
def fdim(arg0, arg1, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
            arg1,
        ],
        {
            (
                tl.dtype("fp32"),
                tl.dtype("fp32"),
            ): ("__nv_fdimf", tl.dtype("fp32")),
            (
                tl.dtype("fp64"),
                tl.dtype("fp64"),
            ): ("__nv_fdim", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def ilogb(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_ilogbf", tl.dtype("int32")),
            (tl.dtype("fp64"),): ("__nv_ilogb", tl.dtype("int32")),
        },
        _builder,
    )


@impl.extern
def logb(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp32"),): ("__nv_logbf", tl.dtype("fp32")),
            (tl.dtype("fp64"),): ("__nv_logb", tl.dtype("fp64")),
        },
        _builder,
    )


@impl.extern
def isfinited(arg0, _builder=None):
    return extern.elementwise(
        "libdevice",
        LIBDEVICE_PATH,
        [
            arg0,
        ],
        {
            (tl.dtype("fp64"),): ("__nv_isfinited", tl.dtype("int32")),
        },
        _builder,
    )
