import os

import triton
from triton import base
from triton import dispatch

LIBDEVICE_PATH = os.path.dirname(os.path.abspath(__file__)) + "/libdevice.10.bc"


@triton.extern
def clz(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.int32,): ("__nv_clz", base.int32),
            (base.int64,): ("__nv_clzll", base.int32),
        },
        _builder=_builder,
    )


@triton.extern
def popc(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.int32,): ("__nv_popc", base.int32),
            (base.int64,): ("__nv_popcll", base.int32),
        },
        _builder=_builder,
    )


@triton.extern
def byte_perm(arg0, arg1, arg2, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
            arg2,
        ],
        arg_type_symbol_dict={
            (
                base.int32,
                base.int32,
                base.int32,
            ): ("__nv_byte_perm", base.int32),
        },
        _builder=_builder,
    )


@triton.extern
def min(arg0, arg1, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
        ],
        arg_type_symbol_dict={
            (
                base.int32,
                base.int32,
            ): ("__nv_min", base.int32),
            (
                base.uint32,
                base.uint32,
            ): ("__nv_umin", base.uint32),
            (
                base.int64,
                base.int64,
            ): ("__nv_llmin", base.int64),
            (
                base.uint64,
                base.uint64,
            ): ("__nv_ullmin", base.uint64),
            (
                base.float32,
                base.float32,
            ): ("__nv_fminf", base.float32),
            (
                base.float64,
                base.float64,
            ): ("__nv_fmin", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def max(arg0, arg1, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
        ],
        arg_type_symbol_dict={
            (
                base.int32,
                base.int32,
            ): ("__nv_max", base.int32),
            (
                base.uint32,
                base.uint32,
            ): ("__nv_umax", base.uint32),
            (
                base.int64,
                base.int64,
            ): ("__nv_llmax", base.int64),
            (
                base.uint64,
                base.uint64,
            ): ("__nv_ullmax", base.uint64),
            (
                base.float32,
                base.float32,
            ): ("__nv_fmaxf", base.float32),
            (
                base.float64,
                base.float64,
            ): ("__nv_fmax", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def mulhi(arg0, arg1, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
        ],
        arg_type_symbol_dict={
            (
                base.int32,
                base.int32,
            ): ("__nv_mulhi", base.int32),
            (
                base.uint32,
                base.uint32,
            ): ("__nv_umulhi", base.uint32),
            (
                base.int64,
                base.int64,
            ): ("__nv_mul64hi", base.int64),
            (
                base.uint64,
                base.uint64,
            ): ("__nv_umul64hi", base.uint64),
        },
        _builder=_builder,
    )


@triton.extern
def mul24(arg0, arg1, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
        ],
        arg_type_symbol_dict={
            (
                base.int32,
                base.int32,
            ): ("__nv_mul24", base.int32),
            (
                base.uint32,
                base.uint32,
            ): ("__nv_umul24", base.uint32),
        },
        _builder=_builder,
    )


@triton.extern
def brev(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.int32,): ("__nv_brev", base.int32),
            (base.int64,): ("__nv_brevll", base.int64),
        },
        _builder=_builder,
    )


@triton.extern
def sad(arg0, arg1, arg2, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
            arg2,
        ],
        arg_type_symbol_dict={
            (
                base.int32,
                base.int32,
                base.uint32,
            ): ("__nv_sad", base.int32),
            (
                base.uint32,
                base.uint32,
                base.uint32,
            ): ("__nv_usad", base.uint32),
        },
        _builder=_builder,
    )


@triton.extern
def abs(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.int32,): ("__nv_abs", base.int32),
            (base.int64,): ("__nv_llabs", base.int64),
            (base.float32,): ("__nv_fabsf", base.float32),
            (base.float64,): ("__nv_fabs", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def floor(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_floorf", base.float32),
            (base.float64,): ("__nv_floor", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def rcp64h(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float64,): ("__nv_rcp64h", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def rsqrt(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_rsqrtf", base.float32),
            (base.float64,): ("__nv_rsqrt", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def ceil(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float64,): ("__nv_ceil", base.float64),
            (base.float32,): ("__nv_ceilf", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def trunc(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float64,): ("__nv_trunc", base.float64),
            (base.float32,): ("__nv_truncf", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def exp2(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_exp2f", base.float32),
            (base.float64,): ("__nv_exp2", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def saturatef(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_saturatef", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def fma_rn(arg0, arg1, arg2, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
            arg2,
        ],
        arg_type_symbol_dict={
            (
                base.float32,
                base.float32,
                base.float32,
            ): ("__nv_fmaf_rn", base.float32),
            (
                base.float64,
                base.float64,
                base.float64,
            ): ("__nv_fma_rn", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def fma_rz(arg0, arg1, arg2, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
            arg2,
        ],
        arg_type_symbol_dict={
            (
                base.float32,
                base.float32,
                base.float32,
            ): ("__nv_fmaf_rz", base.float32),
            (
                base.float64,
                base.float64,
                base.float64,
            ): ("__nv_fma_rz", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def fma_rd(arg0, arg1, arg2, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
            arg2,
        ],
        arg_type_symbol_dict={
            (
                base.float32,
                base.float32,
                base.float32,
            ): ("__nv_fmaf_rd", base.float32),
            (
                base.float64,
                base.float64,
                base.float64,
            ): ("__nv_fma_rd", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def fma_ru(arg0, arg1, arg2, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
            arg2,
        ],
        arg_type_symbol_dict={
            (
                base.float32,
                base.float32,
                base.float32,
            ): ("__nv_fmaf_ru", base.float32),
            (
                base.float64,
                base.float64,
                base.float64,
            ): ("__nv_fma_ru", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def fast_dividef(arg0, arg1, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
        ],
        arg_type_symbol_dict={
            (
                base.float32,
                base.float32,
            ): ("__nv_fast_fdividef", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def div_rn(arg0, arg1, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
        ],
        arg_type_symbol_dict={
            (
                base.float32,
                base.float32,
            ): ("__nv_fdiv_rn", base.float32),
            (
                base.float64,
                base.float64,
            ): ("__nv_ddiv_rn", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def div_rz(arg0, arg1, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
        ],
        arg_type_symbol_dict={
            (
                base.float32,
                base.float32,
            ): ("__nv_fdiv_rz", base.float32),
            (
                base.float64,
                base.float64,
            ): ("__nv_ddiv_rz", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def div_rd(arg0, arg1, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
        ],
        arg_type_symbol_dict={
            (
                base.float32,
                base.float32,
            ): ("__nv_fdiv_rd", base.float32),
            (
                base.float64,
                base.float64,
            ): ("__nv_ddiv_rd", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def div_ru(arg0, arg1, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
        ],
        arg_type_symbol_dict={
            (
                base.float32,
                base.float32,
            ): ("__nv_fdiv_ru", base.float32),
            (
                base.float64,
                base.float64,
            ): ("__nv_ddiv_ru", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def rcp_rn(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_frcp_rn", base.float32),
            (base.float64,): ("__nv_drcp_rn", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def rcp_rz(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_frcp_rz", base.float32),
            (base.float64,): ("__nv_drcp_rz", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def rcp_rd(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_frcp_rd", base.float32),
            (base.float64,): ("__nv_drcp_rd", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def rcp_ru(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_frcp_ru", base.float32),
            (base.float64,): ("__nv_drcp_ru", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def sqrt_rn(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_fsqrt_rn", base.float32),
            (base.float64,): ("__nv_dsqrt_rn", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def sqrt_rz(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_fsqrt_rz", base.float32),
            (base.float64,): ("__nv_dsqrt_rz", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def sqrt_rd(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_fsqrt_rd", base.float32),
            (base.float64,): ("__nv_dsqrt_rd", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def sqrt_ru(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_fsqrt_ru", base.float32),
            (base.float64,): ("__nv_dsqrt_ru", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def sqrt(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_sqrtf", base.float32),
            (base.float64,): ("__nv_sqrt", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def add_rn(arg0, arg1, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
        ],
        arg_type_symbol_dict={
            (
                base.float64,
                base.float64,
            ): ("__nv_dadd_rn", base.float64),
            (
                base.float32,
                base.float32,
            ): ("__nv_fadd_rn", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def add_rz(arg0, arg1, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
        ],
        arg_type_symbol_dict={
            (
                base.float64,
                base.float64,
            ): ("__nv_dadd_rz", base.float64),
            (
                base.float32,
                base.float32,
            ): ("__nv_fadd_rz", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def add_rd(arg0, arg1, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
        ],
        arg_type_symbol_dict={
            (
                base.float64,
                base.float64,
            ): ("__nv_dadd_rd", base.float64),
            (
                base.float32,
                base.float32,
            ): ("__nv_fadd_rd", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def add_ru(arg0, arg1, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
        ],
        arg_type_symbol_dict={
            (
                base.float64,
                base.float64,
            ): ("__nv_dadd_ru", base.float64),
            (
                base.float32,
                base.float32,
            ): ("__nv_fadd_ru", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def mul_rn(arg0, arg1, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
        ],
        arg_type_symbol_dict={
            (
                base.float64,
                base.float64,
            ): ("__nv_dmul_rn", base.float64),
            (
                base.float32,
                base.float32,
            ): ("__nv_fmul_rn", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def mul_rz(arg0, arg1, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
        ],
        arg_type_symbol_dict={
            (
                base.float64,
                base.float64,
            ): ("__nv_dmul_rz", base.float64),
            (
                base.float32,
                base.float32,
            ): ("__nv_fmul_rz", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def mul_rd(arg0, arg1, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
        ],
        arg_type_symbol_dict={
            (
                base.float64,
                base.float64,
            ): ("__nv_dmul_rd", base.float64),
            (
                base.float32,
                base.float32,
            ): ("__nv_fmul_rd", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def mul_ru(arg0, arg1, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
        ],
        arg_type_symbol_dict={
            (
                base.float64,
                base.float64,
            ): ("__nv_dmul_ru", base.float64),
            (
                base.float32,
                base.float32,
            ): ("__nv_fmul_ru", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def double2float_rn(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float64,): ("__nv_double2float_rn", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def double2float_rz(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float64,): ("__nv_double2float_rz", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def double2float_rd(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float64,): ("__nv_double2float_rd", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def double2float_ru(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float64,): ("__nv_double2float_ru", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def double2int_rn(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float64,): ("__nv_double2int_rn", base.int32),
        },
        _builder=_builder,
    )


@triton.extern
def double2int_rz(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float64,): ("__nv_double2int_rz", base.int32),
        },
        _builder=_builder,
    )


@triton.extern
def double2int_rd(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float64,): ("__nv_double2int_rd", base.int32),
        },
        _builder=_builder,
    )


@triton.extern
def double2int_ru(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float64,): ("__nv_double2int_ru", base.int32),
        },
        _builder=_builder,
    )


@triton.extern
def double2uint_rn(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float64,): ("__nv_double2uint_rn", base.int32),
        },
        _builder=_builder,
    )


@triton.extern
def double2uint_rz(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float64,): ("__nv_double2uint_rz", base.int32),
        },
        _builder=_builder,
    )


@triton.extern
def double2uint_rd(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float64,): ("__nv_double2uint_rd", base.int32),
        },
        _builder=_builder,
    )


@triton.extern
def double2uint_ru(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float64,): ("__nv_double2uint_ru", base.int32),
        },
        _builder=_builder,
    )


@triton.extern
def int2double_rn(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.int32,): ("__nv_int2double_rn", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def uint2double_rn(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.uint32,): ("__nv_uint2double_rn", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def float2int_rn(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_float2int_rn", base.int32),
        },
        _builder=_builder,
    )


@triton.extern
def float2int_rz(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_float2int_rz", base.int32),
        },
        _builder=_builder,
    )


@triton.extern
def float2int_rd(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_float2int_rd", base.int32),
        },
        _builder=_builder,
    )


@triton.extern
def float2int_ru(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_float2int_ru", base.int32),
        },
        _builder=_builder,
    )


@triton.extern
def float2uint_rn(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_float2uint_rn", base.int32),
        },
        _builder=_builder,
    )


@triton.extern
def float2uint_rz(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_float2uint_rz", base.int32),
        },
        _builder=_builder,
    )


@triton.extern
def float2uint_rd(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_float2uint_rd", base.int32),
        },
        _builder=_builder,
    )


@triton.extern
def float2uint_ru(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_float2uint_ru", base.int32),
        },
        _builder=_builder,
    )


@triton.extern
def int2float_rn(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.int32,): ("__nv_int2float_rn", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def int2float_rz(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.int32,): ("__nv_int2float_rz", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def int2float_rd(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.int32,): ("__nv_int2float_rd", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def int2float_ru(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.int32,): ("__nv_int2float_ru", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def uint2float_rn(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.uint32,): ("__nv_uint2float_rn", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def uint2float_rz(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.uint32,): ("__nv_uint2float_rz", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def uint2float_rd(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.uint32,): ("__nv_uint2float_rd", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def uint2float_ru(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.uint32,): ("__nv_uint2float_ru", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def hiloint2double(arg0, arg1, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
        ],
        arg_type_symbol_dict={
            (
                base.int32,
                base.int32,
            ): ("__nv_hiloint2double", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def double2loint(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float64,): ("__nv_double2loint", base.int32),
        },
        _builder=_builder,
    )


@triton.extern
def double2hiint(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float64,): ("__nv_double2hiint", base.int32),
        },
        _builder=_builder,
    )


@triton.extern
def float2ll_rn(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_float2ll_rn", base.int64),
        },
        _builder=_builder,
    )


@triton.extern
def float2ll_rz(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_float2ll_rz", base.int64),
        },
        _builder=_builder,
    )


@triton.extern
def float2ll_rd(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_float2ll_rd", base.int64),
        },
        _builder=_builder,
    )


@triton.extern
def float2ll_ru(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_float2ll_ru", base.int64),
        },
        _builder=_builder,
    )


@triton.extern
def float2ull_rn(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_float2ull_rn", base.int64),
        },
        _builder=_builder,
    )


@triton.extern
def float2ull_rz(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_float2ull_rz", base.int64),
        },
        _builder=_builder,
    )


@triton.extern
def float2ull_rd(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_float2ull_rd", base.int64),
        },
        _builder=_builder,
    )


@triton.extern
def float2ull_ru(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_float2ull_ru", base.int64),
        },
        _builder=_builder,
    )


@triton.extern
def double2ll_rn(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float64,): ("__nv_double2ll_rn", base.int64),
        },
        _builder=_builder,
    )


@triton.extern
def double2ll_rz(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float64,): ("__nv_double2ll_rz", base.int64),
        },
        _builder=_builder,
    )


@triton.extern
def double2ll_rd(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float64,): ("__nv_double2ll_rd", base.int64),
        },
        _builder=_builder,
    )


@triton.extern
def double2ll_ru(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float64,): ("__nv_double2ll_ru", base.int64),
        },
        _builder=_builder,
    )


@triton.extern
def double2ull_rn(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float64,): ("__nv_double2ull_rn", base.int64),
        },
        _builder=_builder,
    )


@triton.extern
def double2ull_rz(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float64,): ("__nv_double2ull_rz", base.int64),
        },
        _builder=_builder,
    )


@triton.extern
def double2ull_rd(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float64,): ("__nv_double2ull_rd", base.int64),
        },
        _builder=_builder,
    )


@triton.extern
def double2ull_ru(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float64,): ("__nv_double2ull_ru", base.int64),
        },
        _builder=_builder,
    )


@triton.extern
def ll2float_rn(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.int64,): ("__nv_ll2float_rn", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def ll2float_rz(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.int64,): ("__nv_ll2float_rz", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def ll2float_rd(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.int64,): ("__nv_ll2float_rd", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def ll2float_ru(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.int64,): ("__nv_ll2float_ru", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def ull2float_rn(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.uint64,): ("__nv_ull2float_rn", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def ull2float_rz(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.uint64,): ("__nv_ull2float_rz", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def ull2float_rd(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.uint64,): ("__nv_ull2float_rd", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def ull2float_ru(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.uint64,): ("__nv_ull2float_ru", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def ll2double_rn(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.int64,): ("__nv_ll2double_rn", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def ll2double_rz(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.int64,): ("__nv_ll2double_rz", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def ll2double_rd(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.int64,): ("__nv_ll2double_rd", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def ll2double_ru(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.int64,): ("__nv_ll2double_ru", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def ull2double_rn(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.uint64,): ("__nv_ull2double_rn", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def ull2double_rz(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.uint64,): ("__nv_ull2double_rz", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def ull2double_rd(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.uint64,): ("__nv_ull2double_rd", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def ull2double_ru(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.uint64,): ("__nv_ull2double_ru", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def int_as_float(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.int32,): ("__nv_int_as_float", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def float_as_int(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_float_as_int", base.int32),
        },
        _builder=_builder,
    )


@triton.extern
def uint_as_float(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.uint32,): ("__nv_uint_as_float", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def float_as_uint(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_float_as_uint", base.int32),
        },
        _builder=_builder,
    )


@triton.extern
def longlong_as_double(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.int64,): ("__nv_longlong_as_double", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def double_as_longlong(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float64,): ("__nv_double_as_longlong", base.int64),
        },
        _builder=_builder,
    )


@triton.extern
def fast_sinf(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_fast_sinf", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def fast_cosf(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_fast_cosf", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def fast_log2f(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_fast_log2f", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def fast_logf(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_fast_logf", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def fast_expf(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_fast_expf", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def fast_tanf(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_fast_tanf", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def fast_exp10f(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_fast_exp10f", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def fast_log10f(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_fast_log10f", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def fast_powf(arg0, arg1, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
        ],
        arg_type_symbol_dict={
            (
                base.float32,
                base.float32,
            ): ("__nv_fast_powf", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def hadd(arg0, arg1, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
        ],
        arg_type_symbol_dict={
            (
                base.int32,
                base.int32,
            ): ("__nv_hadd", base.int32),
            (
                base.uint32,
                base.uint32,
            ): ("__nv_uhadd", base.uint32),
        },
        _builder=_builder,
    )


@triton.extern
def rhadd(arg0, arg1, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
        ],
        arg_type_symbol_dict={
            (
                base.int32,
                base.int32,
            ): ("__nv_rhadd", base.int32),
            (
                base.uint32,
                base.uint32,
            ): ("__nv_urhadd", base.uint32),
        },
        _builder=_builder,
    )


@triton.extern
def sub_rn(arg0, arg1, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
        ],
        arg_type_symbol_dict={
            (
                base.float32,
                base.float32,
            ): ("__nv_fsub_rn", base.float32),
            (
                base.float64,
                base.float64,
            ): ("__nv_dsub_rn", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def sub_rz(arg0, arg1, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
        ],
        arg_type_symbol_dict={
            (
                base.float32,
                base.float32,
            ): ("__nv_fsub_rz", base.float32),
            (
                base.float64,
                base.float64,
            ): ("__nv_dsub_rz", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def sub_rd(arg0, arg1, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
        ],
        arg_type_symbol_dict={
            (
                base.float32,
                base.float32,
            ): ("__nv_fsub_rd", base.float32),
            (
                base.float64,
                base.float64,
            ): ("__nv_dsub_rd", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def sub_ru(arg0, arg1, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
        ],
        arg_type_symbol_dict={
            (
                base.float32,
                base.float32,
            ): ("__nv_fsub_ru", base.float32),
            (
                base.float64,
                base.float64,
            ): ("__nv_dsub_ru", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def rsqrt_rn(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_frsqrt_rn", base.float32),
        },
        _builder=_builder,
    )


@triton.extern
def ffs(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.int32,): ("__nv_ffs", base.int32),
            (base.int64,): ("__nv_ffsll", base.int32),
        },
        _builder=_builder,
    )


@triton.extern
def rint(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_rintf", base.float32),
            (base.float64,): ("__nv_rint", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def llrint(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_llrintf", base.int64),
            (base.float64,): ("__nv_llrint", base.int64),
        },
        _builder=_builder,
    )


@triton.extern
def nearbyint(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_nearbyintf", base.float32),
            (base.float64,): ("__nv_nearbyint", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def isnan(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_isnanf", base.int32),
            (base.float64,): ("__nv_isnand", base.int32),
        },
        _builder=_builder,
    )


@triton.extern
def signbit(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_signbitf", base.int32),
            (base.float64,): ("__nv_signbitd", base.int32),
        },
        _builder=_builder,
    )


@triton.extern
def copysign(arg0, arg1, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
        ],
        arg_type_symbol_dict={
            (
                base.float32,
                base.float32,
            ): ("__nv_copysignf", base.float32),
            (
                base.float64,
                base.float64,
            ): ("__nv_copysign", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def finitef(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_finitef", base.int32),
        },
        _builder=_builder,
    )


@triton.extern
def isinf(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_isinff", base.int32),
            (base.float64,): ("__nv_isinfd", base.int32),
        },
        _builder=_builder,
    )


@triton.extern
def nextafter(arg0, arg1, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
        ],
        arg_type_symbol_dict={
            (
                base.float32,
                base.float32,
            ): ("__nv_nextafterf", base.float32),
            (
                base.float64,
                base.float64,
            ): ("__nv_nextafter", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def sin(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_sinf", base.float32),
            (base.float64,): ("__nv_sin", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def cos(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_cosf", base.float32),
            (base.float64,): ("__nv_cos", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def sinpi(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_sinpif", base.float32),
            (base.float64,): ("__nv_sinpi", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def cospi(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_cospif", base.float32),
            (base.float64,): ("__nv_cospi", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def tan(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_tanf", base.float32),
            (base.float64,): ("__nv_tan", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def log2(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_log2f", base.float32),
            (base.float64,): ("__nv_log2", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def exp(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_expf", base.float32),
            (base.float64,): ("__nv_exp", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def exp10(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_exp10f", base.float32),
            (base.float64,): ("__nv_exp10", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def cosh(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_coshf", base.float32),
            (base.float64,): ("__nv_cosh", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def sinh(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_sinhf", base.float32),
            (base.float64,): ("__nv_sinh", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def tanh(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_tanhf", base.float32),
            (base.float64,): ("__nv_tanh", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def atan2(arg0, arg1, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
        ],
        arg_type_symbol_dict={
            (
                base.float32,
                base.float32,
            ): ("__nv_atan2f", base.float32),
            (
                base.float64,
                base.float64,
            ): ("__nv_atan2", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def atan(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_atanf", base.float32),
            (base.float64,): ("__nv_atan", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def asin(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_asinf", base.float32),
            (base.float64,): ("__nv_asin", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def acos(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_acosf", base.float32),
            (base.float64,): ("__nv_acos", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def log(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_logf", base.float32),
            (base.float64,): ("__nv_log", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def log10(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_log10f", base.float32),
            (base.float64,): ("__nv_log10", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def log1p(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_log1pf", base.float32),
            (base.float64,): ("__nv_log1p", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def acosh(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_acoshf", base.float32),
            (base.float64,): ("__nv_acosh", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def asinh(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_asinhf", base.float32),
            (base.float64,): ("__nv_asinh", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def atanh(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_atanhf", base.float32),
            (base.float64,): ("__nv_atanh", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def expm1(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_expm1f", base.float32),
            (base.float64,): ("__nv_expm1", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def hypot(arg0, arg1, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
        ],
        arg_type_symbol_dict={
            (
                base.float32,
                base.float32,
            ): ("__nv_hypotf", base.float32),
            (
                base.float64,
                base.float64,
            ): ("__nv_hypot", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def rhypot(arg0, arg1, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
        ],
        arg_type_symbol_dict={
            (
                base.float32,
                base.float32,
            ): ("__nv_rhypotf", base.float32),
            (
                base.float64,
                base.float64,
            ): ("__nv_rhypot", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def norm3d(arg0, arg1, arg2, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
            arg2,
        ],
        arg_type_symbol_dict={
            (
                base.float32,
                base.float32,
                base.float32,
            ): ("__nv_norm3df", base.float32),
            (
                base.float64,
                base.float64,
                base.float64,
            ): ("__nv_norm3d", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def rnorm3d(arg0, arg1, arg2, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
            arg2,
        ],
        arg_type_symbol_dict={
            (
                base.float32,
                base.float32,
                base.float32,
            ): ("__nv_rnorm3df", base.float32),
            (
                base.float64,
                base.float64,
                base.float64,
            ): ("__nv_rnorm3d", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def norm4d(arg0, arg1, arg2, arg3, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
            arg2,
            arg3,
        ],
        arg_type_symbol_dict={
            (
                base.float32,
                base.float32,
                base.float32,
                base.float32,
            ): ("__nv_norm4df", base.float32),
            (
                base.float64,
                base.float64,
                base.float64,
                base.float64,
            ): ("__nv_norm4d", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def rnorm4d(arg0, arg1, arg2, arg3, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
            arg2,
            arg3,
        ],
        arg_type_symbol_dict={
            (
                base.float32,
                base.float32,
                base.float32,
                base.float32,
            ): ("__nv_rnorm4df", base.float32),
            (
                base.float64,
                base.float64,
                base.float64,
                base.float64,
            ): ("__nv_rnorm4d", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def cbrt(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_cbrtf", base.float32),
            (base.float64,): ("__nv_cbrt", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def rcbrt(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_rcbrtf", base.float32),
            (base.float64,): ("__nv_rcbrt", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def j0(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_j0f", base.float32),
            (base.float64,): ("__nv_j0", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def j1(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_j1f", base.float32),
            (base.float64,): ("__nv_j1", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def y0(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_y0f", base.float32),
            (base.float64,): ("__nv_y0", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def y1(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_y1f", base.float32),
            (base.float64,): ("__nv_y1", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def yn(arg0, arg1, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
        ],
        arg_type_symbol_dict={
            (
                base.int32,
                base.float32,
            ): ("__nv_ynf", base.float32),
            (
                base.int32,
                base.float64,
            ): ("__nv_yn", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def jn(arg0, arg1, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
        ],
        arg_type_symbol_dict={
            (
                base.int32,
                base.float32,
            ): ("__nv_jnf", base.float32),
            (
                base.int32,
                base.float64,
            ): ("__nv_jn", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def cyl_bessel_i0(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_cyl_bessel_i0f", base.float32),
            (base.float64,): ("__nv_cyl_bessel_i0", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def cyl_bessel_i1(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_cyl_bessel_i1f", base.float32),
            (base.float64,): ("__nv_cyl_bessel_i1", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def erf(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_erff", base.float32),
            (base.float64,): ("__nv_erf", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def erfinv(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_erfinvf", base.float32),
            (base.float64,): ("__nv_erfinv", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def erfc(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_erfcf", base.float32),
            (base.float64,): ("__nv_erfc", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def erfcx(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_erfcxf", base.float32),
            (base.float64,): ("__nv_erfcx", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def erfcinv(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_erfcinvf", base.float32),
            (base.float64,): ("__nv_erfcinv", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def normcdfinv(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_normcdfinvf", base.float32),
            (base.float64,): ("__nv_normcdfinv", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def normcdf(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_normcdff", base.float32),
            (base.float64,): ("__nv_normcdf", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def lgamma(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_lgammaf", base.float32),
            (base.float64,): ("__nv_lgamma", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def ldexp(arg0, arg1, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
        ],
        arg_type_symbol_dict={
            (
                base.float32,
                base.int32,
            ): ("__nv_ldexpf", base.float32),
            (
                base.float64,
                base.int32,
            ): ("__nv_ldexp", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def scalbn(arg0, arg1, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
        ],
        arg_type_symbol_dict={
            (
                base.float32,
                base.int32,
            ): ("__nv_scalbnf", base.float32),
            (
                base.float64,
                base.int32,
            ): ("__nv_scalbn", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def fmod(arg0, arg1, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
        ],
        arg_type_symbol_dict={
            (
                base.float32,
                base.float32,
            ): ("__nv_fmodf", base.float32),
            (
                base.float64,
                base.float64,
            ): ("__nv_fmod", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def remainder(arg0, arg1, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
        ],
        arg_type_symbol_dict={
            (
                base.float32,
                base.float32,
            ): ("__nv_remainderf", base.float32),
            (
                base.float64,
                base.float64,
            ): ("__nv_remainder", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def fma(arg0, arg1, arg2, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
            arg2,
        ],
        arg_type_symbol_dict={
            (
                base.float32,
                base.float32,
                base.float32,
            ): ("__nv_fmaf", base.float32),
            (
                base.float64,
                base.float64,
                base.float64,
            ): ("__nv_fma", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def pow(arg0, arg1, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
        ],
        arg_type_symbol_dict={
            (
                base.float32,
                base.int32,
            ): ("__nv_powif", base.float32),
            (
                base.float64,
                base.int32,
            ): ("__nv_powi", base.float64),
            (
                base.float32,
                base.float32,
            ): ("__nv_powf", base.float32),
            (
                base.float64,
                base.float64,
            ): ("__nv_pow", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def tgamma(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_tgammaf", base.float32),
            (base.float64,): ("__nv_tgamma", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def round(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_roundf", base.float32),
            (base.float64,): ("__nv_round", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def llround(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_llroundf", base.int64),
            (base.float64,): ("__nv_llround", base.int64),
        },
        _builder=_builder,
    )


@triton.extern
def fdim(arg0, arg1, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
            arg1,
        ],
        arg_type_symbol_dict={
            (
                base.float32,
                base.float32,
            ): ("__nv_fdimf", base.float32),
            (
                base.float64,
                base.float64,
            ): ("__nv_fdim", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def ilogb(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_ilogbf", base.int32),
            (base.float64,): ("__nv_ilogb", base.int32),
        },
        _builder=_builder,
    )


@triton.extern
def logb(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float32,): ("__nv_logbf", base.float32),
            (base.float64,): ("__nv_logb", base.float64),
        },
        _builder=_builder,
    )


@triton.extern
def isfinited(arg0, _builder: triton.ir.builder = None):
    return dispatch.elementwise(
        lib_name="libdevice",
        lib_path=LIBDEVICE_PATH,
        args=[
            arg0,
        ],
        arg_type_symbol_dict={
            (base.float64,): ("__nv_isfinited", base.int32),
        },
        _builder=_builder,
    )
