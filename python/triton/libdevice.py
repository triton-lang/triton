import os

from . import unroll

LIBDEVICE_PATH = os.path.dirname(
    os.path.abspath(__file__)) + "/libdevice.10.bc"


@unroll.extern
def clz(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.int32,): ("__nv_clz", unroll.int32),
                                                    (unroll.int64,): ("__nv_clzll", unroll.int32),
                                                    }, _builder=_builder)


@unroll.extern
def popc(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.int32,): ("__nv_popc", unroll.int32),
                                                    (unroll.int64,): ("__nv_popcll", unroll.int32),
                                                    }, _builder=_builder)


@unroll.extern
def byte_perm(arg0, arg1, arg2, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, arg2, ],
                              arg_type_symbol_dict={
                                  (unroll.int32, unroll.int32, unroll.int32,): ("__nv_byte_perm", unroll.int32),
                                  }, _builder=_builder)


@unroll.extern
def min(arg0, arg1, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, ],
                              arg_type_symbol_dict={(unroll.int32, unroll.int32,): ("__nv_min", unroll.int32),
                                                    (unroll.uint32, unroll.uint32,): ("__nv_umin", unroll.uint32),
                                                    (unroll.int64, unroll.int64,): ("__nv_llmin", unroll.int64),
                                                    (unroll.uint64, unroll.uint64,): ("__nv_ullmin", unroll.uint64),
                                                    (unroll.float32, unroll.float32,): ("__nv_fminf", unroll.float32),
                                                    (unroll.float64, unroll.float64,): ("__nv_fmin", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def max(arg0, arg1, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, ],
                              arg_type_symbol_dict={(unroll.int32, unroll.int32,): ("__nv_max", unroll.int32),
                                                    (unroll.uint32, unroll.uint32,): ("__nv_umax", unroll.uint32),
                                                    (unroll.int64, unroll.int64,): ("__nv_llmax", unroll.int64),
                                                    (unroll.uint64, unroll.uint64,): ("__nv_ullmax", unroll.uint64),
                                                    (unroll.float32, unroll.float32,): ("__nv_fmaxf", unroll.float32),
                                                    (unroll.float64, unroll.float64,): ("__nv_fmax", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def mulhi(arg0, arg1, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, ],
                              arg_type_symbol_dict={(unroll.int32, unroll.int32,): ("__nv_mulhi", unroll.int32),
                                                    (unroll.uint32, unroll.uint32,): ("__nv_umulhi", unroll.uint32),
                                                    (unroll.int64, unroll.int64,): ("__nv_mul64hi", unroll.int64),
                                                    (unroll.uint64, unroll.uint64,): ("__nv_umul64hi", unroll.uint64),
                                                    }, _builder=_builder)


@unroll.extern
def mul24(arg0, arg1, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, ],
                              arg_type_symbol_dict={(unroll.int32, unroll.int32,): ("__nv_mul24", unroll.int32),
                                                    (unroll.uint32, unroll.uint32,): ("__nv_umul24", unroll.uint32),
                                                    }, _builder=_builder)


@unroll.extern
def brev(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.int32,): ("__nv_brev", unroll.int32),
                                                    (unroll.int64,): ("__nv_brevll", unroll.int64),
                                                    }, _builder=_builder)


@unroll.extern
def sad(arg0, arg1, arg2, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, arg2, ],
                              arg_type_symbol_dict={
                                  (unroll.int32, unroll.int32, unroll.uint32,): ("__nv_sad", unroll.int32),
                                  (unroll.uint32, unroll.uint32, unroll.uint32,): ("__nv_usad", unroll.uint32),
                                  }, _builder=_builder)


@unroll.extern
def abs(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.int32,): ("__nv_abs", unroll.int32),
                                                    (unroll.int64,): ("__nv_llabs", unroll.int64),
                                                    (unroll.float32,): ("__nv_fabsf", unroll.float32),
                                                    (unroll.float64,): ("__nv_fabs", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def floor(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_floorf", unroll.float32),
                                                    (unroll.float64,): ("__nv_floor", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def rcp64h(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float64,): ("__nv_rcp64h", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def rsqrt(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_rsqrtf", unroll.float32),
                                                    (unroll.float64,): ("__nv_rsqrt", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def ceil(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float64,): ("__nv_ceil", unroll.float64),
                                                    (unroll.float32,): ("__nv_ceilf", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def trunc(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float64,): ("__nv_trunc", unroll.float64),
                                                    (unroll.float32,): ("__nv_truncf", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def exp2(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_exp2f", unroll.float32),
                                                    (unroll.float64,): ("__nv_exp2", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def saturatef(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_saturatef", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def fma_rn(arg0, arg1, arg2, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, arg2, ],
                              arg_type_symbol_dict={
                                  (unroll.float32, unroll.float32, unroll.float32,): ("__nv_fmaf_rn", unroll.float32),
                                  (unroll.float64, unroll.float64, unroll.float64,): ("__nv_fma_rn", unroll.float64),
                                  }, _builder=_builder)


@unroll.extern
def fma_rz(arg0, arg1, arg2, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, arg2, ],
                              arg_type_symbol_dict={
                                  (unroll.float32, unroll.float32, unroll.float32,): ("__nv_fmaf_rz", unroll.float32),
                                  (unroll.float64, unroll.float64, unroll.float64,): ("__nv_fma_rz", unroll.float64),
                                  }, _builder=_builder)


@unroll.extern
def fma_rd(arg0, arg1, arg2, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, arg2, ],
                              arg_type_symbol_dict={
                                  (unroll.float32, unroll.float32, unroll.float32,): ("__nv_fmaf_rd", unroll.float32),
                                  (unroll.float64, unroll.float64, unroll.float64,): ("__nv_fma_rd", unroll.float64),
                                  }, _builder=_builder)


@unroll.extern
def fma_ru(arg0, arg1, arg2, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, arg2, ],
                              arg_type_symbol_dict={
                                  (unroll.float32, unroll.float32, unroll.float32,): ("__nv_fmaf_ru", unroll.float32),
                                  (unroll.float64, unroll.float64, unroll.float64,): ("__nv_fma_ru", unroll.float64),
                                  }, _builder=_builder)


@unroll.extern
def fast_dividef(arg0, arg1, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, ], arg_type_symbol_dict={
        (unroll.float32, unroll.float32,): ("__nv_fast_fdividef", unroll.float32),
        }, _builder=_builder)


@unroll.extern
def div_rn(arg0, arg1, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, ],
                              arg_type_symbol_dict={(unroll.float32, unroll.float32,): ("__nv_fdiv_rn", unroll.float32),
                                                    (unroll.float64, unroll.float64,): ("__nv_ddiv_rn", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def div_rz(arg0, arg1, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, ],
                              arg_type_symbol_dict={(unroll.float32, unroll.float32,): ("__nv_fdiv_rz", unroll.float32),
                                                    (unroll.float64, unroll.float64,): ("__nv_ddiv_rz", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def div_rd(arg0, arg1, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, ],
                              arg_type_symbol_dict={(unroll.float32, unroll.float32,): ("__nv_fdiv_rd", unroll.float32),
                                                    (unroll.float64, unroll.float64,): ("__nv_ddiv_rd", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def div_ru(arg0, arg1, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, ],
                              arg_type_symbol_dict={(unroll.float32, unroll.float32,): ("__nv_fdiv_ru", unroll.float32),
                                                    (unroll.float64, unroll.float64,): ("__nv_ddiv_ru", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def rcp_rn(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_frcp_rn", unroll.float32),
                                                    (unroll.float64,): ("__nv_drcp_rn", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def rcp_rz(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_frcp_rz", unroll.float32),
                                                    (unroll.float64,): ("__nv_drcp_rz", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def rcp_rd(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_frcp_rd", unroll.float32),
                                                    (unroll.float64,): ("__nv_drcp_rd", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def rcp_ru(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_frcp_ru", unroll.float32),
                                                    (unroll.float64,): ("__nv_drcp_ru", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def sqrt_rn(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_fsqrt_rn", unroll.float32),
                                                    (unroll.float64,): ("__nv_dsqrt_rn", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def sqrt_rz(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_fsqrt_rz", unroll.float32),
                                                    (unroll.float64,): ("__nv_dsqrt_rz", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def sqrt_rd(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_fsqrt_rd", unroll.float32),
                                                    (unroll.float64,): ("__nv_dsqrt_rd", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def sqrt_ru(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_fsqrt_ru", unroll.float32),
                                                    (unroll.float64,): ("__nv_dsqrt_ru", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def sqrt(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_sqrtf", unroll.float32),
                                                    (unroll.float64,): ("__nv_sqrt", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def add_rn(arg0, arg1, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, ],
                              arg_type_symbol_dict={(unroll.float64, unroll.float64,): ("__nv_dadd_rn", unroll.float64),
                                                    (unroll.float32, unroll.float32,): ("__nv_fadd_rn", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def add_rz(arg0, arg1, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, ],
                              arg_type_symbol_dict={(unroll.float64, unroll.float64,): ("__nv_dadd_rz", unroll.float64),
                                                    (unroll.float32, unroll.float32,): ("__nv_fadd_rz", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def add_rd(arg0, arg1, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, ],
                              arg_type_symbol_dict={(unroll.float64, unroll.float64,): ("__nv_dadd_rd", unroll.float64),
                                                    (unroll.float32, unroll.float32,): ("__nv_fadd_rd", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def add_ru(arg0, arg1, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, ],
                              arg_type_symbol_dict={(unroll.float64, unroll.float64,): ("__nv_dadd_ru", unroll.float64),
                                                    (unroll.float32, unroll.float32,): ("__nv_fadd_ru", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def mul_rn(arg0, arg1, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, ],
                              arg_type_symbol_dict={(unroll.float64, unroll.float64,): ("__nv_dmul_rn", unroll.float64),
                                                    (unroll.float32, unroll.float32,): ("__nv_fmul_rn", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def mul_rz(arg0, arg1, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, ],
                              arg_type_symbol_dict={(unroll.float64, unroll.float64,): ("__nv_dmul_rz", unroll.float64),
                                                    (unroll.float32, unroll.float32,): ("__nv_fmul_rz", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def mul_rd(arg0, arg1, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, ],
                              arg_type_symbol_dict={(unroll.float64, unroll.float64,): ("__nv_dmul_rd", unroll.float64),
                                                    (unroll.float32, unroll.float32,): ("__nv_fmul_rd", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def mul_ru(arg0, arg1, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, ],
                              arg_type_symbol_dict={(unroll.float64, unroll.float64,): ("__nv_dmul_ru", unroll.float64),
                                                    (unroll.float32, unroll.float32,): ("__nv_fmul_ru", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def double2float_rn(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float64,): ("__nv_double2float_rn", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def double2float_rz(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float64,): ("__nv_double2float_rz", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def double2float_rd(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float64,): ("__nv_double2float_rd", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def double2float_ru(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float64,): ("__nv_double2float_ru", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def double2int_rn(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float64,): ("__nv_double2int_rn", unroll.int32),
                                                    }, _builder=_builder)


@unroll.extern
def double2int_rz(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float64,): ("__nv_double2int_rz", unroll.int32),
                                                    }, _builder=_builder)


@unroll.extern
def double2int_rd(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float64,): ("__nv_double2int_rd", unroll.int32),
                                                    }, _builder=_builder)


@unroll.extern
def double2int_ru(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float64,): ("__nv_double2int_ru", unroll.int32),
                                                    }, _builder=_builder)


@unroll.extern
def double2uint_rn(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float64,): ("__nv_double2uint_rn", unroll.int32),
                                                    }, _builder=_builder)


@unroll.extern
def double2uint_rz(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float64,): ("__nv_double2uint_rz", unroll.int32),
                                                    }, _builder=_builder)


@unroll.extern
def double2uint_rd(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float64,): ("__nv_double2uint_rd", unroll.int32),
                                                    }, _builder=_builder)


@unroll.extern
def double2uint_ru(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float64,): ("__nv_double2uint_ru", unroll.int32),
                                                    }, _builder=_builder)


@unroll.extern
def int2double_rn(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.int32,): ("__nv_int2double_rn", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def uint2double_rn(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.uint32,): ("__nv_uint2double_rn", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def float2int_rn(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_float2int_rn", unroll.int32),
                                                    }, _builder=_builder)


@unroll.extern
def float2int_rz(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_float2int_rz", unroll.int32),
                                                    }, _builder=_builder)


@unroll.extern
def float2int_rd(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_float2int_rd", unroll.int32),
                                                    }, _builder=_builder)


@unroll.extern
def float2int_ru(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_float2int_ru", unroll.int32),
                                                    }, _builder=_builder)


@unroll.extern
def float2uint_rn(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_float2uint_rn", unroll.int32),
                                                    }, _builder=_builder)


@unroll.extern
def float2uint_rz(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_float2uint_rz", unroll.int32),
                                                    }, _builder=_builder)


@unroll.extern
def float2uint_rd(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_float2uint_rd", unroll.int32),
                                                    }, _builder=_builder)


@unroll.extern
def float2uint_ru(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_float2uint_ru", unroll.int32),
                                                    }, _builder=_builder)


@unroll.extern
def int2float_rn(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.int32,): ("__nv_int2float_rn", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def int2float_rz(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.int32,): ("__nv_int2float_rz", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def int2float_rd(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.int32,): ("__nv_int2float_rd", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def int2float_ru(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.int32,): ("__nv_int2float_ru", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def uint2float_rn(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.uint32,): ("__nv_uint2float_rn", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def uint2float_rz(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.uint32,): ("__nv_uint2float_rz", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def uint2float_rd(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.uint32,): ("__nv_uint2float_rd", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def uint2float_ru(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.uint32,): ("__nv_uint2float_ru", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def hiloint2double(arg0, arg1, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, ], arg_type_symbol_dict={
        (unroll.int32, unroll.int32,): ("__nv_hiloint2double", unroll.float64),
        }, _builder=_builder)


@unroll.extern
def double2loint(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float64,): ("__nv_double2loint", unroll.int32),
                                                    }, _builder=_builder)


@unroll.extern
def double2hiint(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float64,): ("__nv_double2hiint", unroll.int32),
                                                    }, _builder=_builder)


@unroll.extern
def float2ll_rn(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_float2ll_rn", unroll.int64),
                                                    }, _builder=_builder)


@unroll.extern
def float2ll_rz(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_float2ll_rz", unroll.int64),
                                                    }, _builder=_builder)


@unroll.extern
def float2ll_rd(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_float2ll_rd", unroll.int64),
                                                    }, _builder=_builder)


@unroll.extern
def float2ll_ru(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_float2ll_ru", unroll.int64),
                                                    }, _builder=_builder)


@unroll.extern
def float2ull_rn(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_float2ull_rn", unroll.int64),
                                                    }, _builder=_builder)


@unroll.extern
def float2ull_rz(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_float2ull_rz", unroll.int64),
                                                    }, _builder=_builder)


@unroll.extern
def float2ull_rd(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_float2ull_rd", unroll.int64),
                                                    }, _builder=_builder)


@unroll.extern
def float2ull_ru(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_float2ull_ru", unroll.int64),
                                                    }, _builder=_builder)


@unroll.extern
def double2ll_rn(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float64,): ("__nv_double2ll_rn", unroll.int64),
                                                    }, _builder=_builder)


@unroll.extern
def double2ll_rz(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float64,): ("__nv_double2ll_rz", unroll.int64),
                                                    }, _builder=_builder)


@unroll.extern
def double2ll_rd(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float64,): ("__nv_double2ll_rd", unroll.int64),
                                                    }, _builder=_builder)


@unroll.extern
def double2ll_ru(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float64,): ("__nv_double2ll_ru", unroll.int64),
                                                    }, _builder=_builder)


@unroll.extern
def double2ull_rn(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float64,): ("__nv_double2ull_rn", unroll.int64),
                                                    }, _builder=_builder)


@unroll.extern
def double2ull_rz(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float64,): ("__nv_double2ull_rz", unroll.int64),
                                                    }, _builder=_builder)


@unroll.extern
def double2ull_rd(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float64,): ("__nv_double2ull_rd", unroll.int64),
                                                    }, _builder=_builder)


@unroll.extern
def double2ull_ru(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float64,): ("__nv_double2ull_ru", unroll.int64),
                                                    }, _builder=_builder)


@unroll.extern
def ll2float_rn(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.int64,): ("__nv_ll2float_rn", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def ll2float_rz(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.int64,): ("__nv_ll2float_rz", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def ll2float_rd(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.int64,): ("__nv_ll2float_rd", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def ll2float_ru(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.int64,): ("__nv_ll2float_ru", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def ull2float_rn(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.uint64,): ("__nv_ull2float_rn", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def ull2float_rz(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.uint64,): ("__nv_ull2float_rz", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def ull2float_rd(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.uint64,): ("__nv_ull2float_rd", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def ull2float_ru(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.uint64,): ("__nv_ull2float_ru", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def ll2double_rn(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.int64,): ("__nv_ll2double_rn", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def ll2double_rz(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.int64,): ("__nv_ll2double_rz", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def ll2double_rd(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.int64,): ("__nv_ll2double_rd", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def ll2double_ru(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.int64,): ("__nv_ll2double_ru", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def ull2double_rn(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.uint64,): ("__nv_ull2double_rn", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def ull2double_rz(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.uint64,): ("__nv_ull2double_rz", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def ull2double_rd(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.uint64,): ("__nv_ull2double_rd", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def ull2double_ru(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.uint64,): ("__nv_ull2double_ru", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def int_as_float(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.int32,): ("__nv_int_as_float", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def float_as_int(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_float_as_int", unroll.int32),
                                                    }, _builder=_builder)


@unroll.extern
def uint_as_float(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.uint32,): ("__nv_uint_as_float", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def float_as_uint(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_float_as_uint", unroll.int32),
                                                    }, _builder=_builder)


@unroll.extern
def longlong_as_double(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.int64,): ("__nv_longlong_as_double", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def double_as_longlong(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float64,): ("__nv_double_as_longlong", unroll.int64),
                                                    }, _builder=_builder)


@unroll.extern
def fast_sinf(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_fast_sinf", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def fast_cosf(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_fast_cosf", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def fast_log2f(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_fast_log2f", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def fast_logf(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_fast_logf", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def fast_expf(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_fast_expf", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def fast_tanf(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_fast_tanf", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def fast_exp10f(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_fast_exp10f", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def fast_log10f(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_fast_log10f", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def fast_powf(arg0, arg1, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, ], arg_type_symbol_dict={
        (unroll.float32, unroll.float32,): ("__nv_fast_powf", unroll.float32),
        }, _builder=_builder)


@unroll.extern
def hadd(arg0, arg1, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, ],
                              arg_type_symbol_dict={(unroll.int32, unroll.int32,): ("__nv_hadd", unroll.int32),
                                                    (unroll.uint32, unroll.uint32,): ("__nv_uhadd", unroll.uint32),
                                                    }, _builder=_builder)


@unroll.extern
def rhadd(arg0, arg1, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, ],
                              arg_type_symbol_dict={(unroll.int32, unroll.int32,): ("__nv_rhadd", unroll.int32),
                                                    (unroll.uint32, unroll.uint32,): ("__nv_urhadd", unroll.uint32),
                                                    }, _builder=_builder)


@unroll.extern
def sub_rn(arg0, arg1, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, ],
                              arg_type_symbol_dict={(unroll.float32, unroll.float32,): ("__nv_fsub_rn", unroll.float32),
                                                    (unroll.float64, unroll.float64,): ("__nv_dsub_rn", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def sub_rz(arg0, arg1, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, ],
                              arg_type_symbol_dict={(unroll.float32, unroll.float32,): ("__nv_fsub_rz", unroll.float32),
                                                    (unroll.float64, unroll.float64,): ("__nv_dsub_rz", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def sub_rd(arg0, arg1, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, ],
                              arg_type_symbol_dict={(unroll.float32, unroll.float32,): ("__nv_fsub_rd", unroll.float32),
                                                    (unroll.float64, unroll.float64,): ("__nv_dsub_rd", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def sub_ru(arg0, arg1, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, ],
                              arg_type_symbol_dict={(unroll.float32, unroll.float32,): ("__nv_fsub_ru", unroll.float32),
                                                    (unroll.float64, unroll.float64,): ("__nv_dsub_ru", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def rsqrt_rn(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_frsqrt_rn", unroll.float32),
                                                    }, _builder=_builder)


@unroll.extern
def ffs(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.int32,): ("__nv_ffs", unroll.int32),
                                                    (unroll.int64,): ("__nv_ffsll", unroll.int32),
                                                    }, _builder=_builder)


@unroll.extern
def rint(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_rintf", unroll.float32),
                                                    (unroll.float64,): ("__nv_rint", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def llrint(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_llrintf", unroll.int64),
                                                    (unroll.float64,): ("__nv_llrint", unroll.int64),
                                                    }, _builder=_builder)


@unroll.extern
def nearbyint(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_nearbyintf", unroll.float32),
                                                    (unroll.float64,): ("__nv_nearbyint", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def isnan(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_isnanf", unroll.int32),
                                                    (unroll.float64,): ("__nv_isnand", unroll.int32),
                                                    }, _builder=_builder)


@unroll.extern
def signbit(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_signbitf", unroll.int32),
                                                    (unroll.float64,): ("__nv_signbitd", unroll.int32),
                                                    }, _builder=_builder)


@unroll.extern
def copysign(arg0, arg1, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, ], arg_type_symbol_dict={
        (unroll.float32, unroll.float32,): ("__nv_copysignf", unroll.float32),
        (unroll.float64, unroll.float64,): ("__nv_copysign", unroll.float64),
        }, _builder=_builder)


@unroll.extern
def finitef(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_finitef", unroll.int32),
                                                    }, _builder=_builder)


@unroll.extern
def isinf(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_isinff", unroll.int32),
                                                    (unroll.float64,): ("__nv_isinfd", unroll.int32),
                                                    }, _builder=_builder)


@unroll.extern
def nextafter(arg0, arg1, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, ], arg_type_symbol_dict={
        (unroll.float32, unroll.float32,): ("__nv_nextafterf", unroll.float32),
        (unroll.float64, unroll.float64,): ("__nv_nextafter", unroll.float64),
        }, _builder=_builder)


@unroll.extern
def sin(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_sinf", unroll.float32),
                                                    (unroll.float64,): ("__nv_sin", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def cos(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_cosf", unroll.float32),
                                                    (unroll.float64,): ("__nv_cos", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def sinpi(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_sinpif", unroll.float32),
                                                    (unroll.float64,): ("__nv_sinpi", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def cospi(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_cospif", unroll.float32),
                                                    (unroll.float64,): ("__nv_cospi", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def tan(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_tanf", unroll.float32),
                                                    (unroll.float64,): ("__nv_tan", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def log2(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_log2f", unroll.float32),
                                                    (unroll.float64,): ("__nv_log2", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def exp(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_expf", unroll.float32),
                                                    (unroll.float64,): ("__nv_exp", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def exp10(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_exp10f", unroll.float32),
                                                    (unroll.float64,): ("__nv_exp10", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def cosh(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_coshf", unroll.float32),
                                                    (unroll.float64,): ("__nv_cosh", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def sinh(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_sinhf", unroll.float32),
                                                    (unroll.float64,): ("__nv_sinh", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def tanh(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_tanhf", unroll.float32),
                                                    (unroll.float64,): ("__nv_tanh", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def atan2(arg0, arg1, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, ],
                              arg_type_symbol_dict={(unroll.float32, unroll.float32,): ("__nv_atan2f", unroll.float32),
                                                    (unroll.float64, unroll.float64,): ("__nv_atan2", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def atan(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_atanf", unroll.float32),
                                                    (unroll.float64,): ("__nv_atan", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def asin(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_asinf", unroll.float32),
                                                    (unroll.float64,): ("__nv_asin", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def acos(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_acosf", unroll.float32),
                                                    (unroll.float64,): ("__nv_acos", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def log(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_logf", unroll.float32),
                                                    (unroll.float64,): ("__nv_log", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def log10(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_log10f", unroll.float32),
                                                    (unroll.float64,): ("__nv_log10", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def log1p(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_log1pf", unroll.float32),
                                                    (unroll.float64,): ("__nv_log1p", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def acosh(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_acoshf", unroll.float32),
                                                    (unroll.float64,): ("__nv_acosh", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def asinh(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_asinhf", unroll.float32),
                                                    (unroll.float64,): ("__nv_asinh", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def atanh(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_atanhf", unroll.float32),
                                                    (unroll.float64,): ("__nv_atanh", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def expm1(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_expm1f", unroll.float32),
                                                    (unroll.float64,): ("__nv_expm1", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def hypot(arg0, arg1, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, ],
                              arg_type_symbol_dict={(unroll.float32, unroll.float32,): ("__nv_hypotf", unroll.float32),
                                                    (unroll.float64, unroll.float64,): ("__nv_hypot", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def rhypot(arg0, arg1, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, ],
                              arg_type_symbol_dict={(unroll.float32, unroll.float32,): ("__nv_rhypotf", unroll.float32),
                                                    (unroll.float64, unroll.float64,): ("__nv_rhypot", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def norm3d(arg0, arg1, arg2, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, arg2, ],
                              arg_type_symbol_dict={
                                  (unroll.float32, unroll.float32, unroll.float32,): ("__nv_norm3df", unroll.float32),
                                  (unroll.float64, unroll.float64, unroll.float64,): ("__nv_norm3d", unroll.float64),
                                  }, _builder=_builder)


@unroll.extern
def rnorm3d(arg0, arg1, arg2, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, arg2, ],
                              arg_type_symbol_dict={
                                  (unroll.float32, unroll.float32, unroll.float32,): ("__nv_rnorm3df", unroll.float32),
                                  (unroll.float64, unroll.float64, unroll.float64,): ("__nv_rnorm3d", unroll.float64),
                                  }, _builder=_builder)


@unroll.extern
def norm4d(arg0, arg1, arg2, arg3, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, arg2, arg3, ],
                              arg_type_symbol_dict={(unroll.float32, unroll.float32, unroll.float32, unroll.float32,): (
                              "__nv_norm4df", unroll.float32),
                                                    (unroll.float64, unroll.float64, unroll.float64, unroll.float64,): (
                                                    "__nv_norm4d", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def rnorm4d(arg0, arg1, arg2, arg3, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, arg2, arg3, ],
                              arg_type_symbol_dict={(unroll.float32, unroll.float32, unroll.float32, unroll.float32,): (
                              "__nv_rnorm4df", unroll.float32),
                                                    (unroll.float64, unroll.float64, unroll.float64, unroll.float64,): (
                                                    "__nv_rnorm4d", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def cbrt(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_cbrtf", unroll.float32),
                                                    (unroll.float64,): ("__nv_cbrt", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def rcbrt(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_rcbrtf", unroll.float32),
                                                    (unroll.float64,): ("__nv_rcbrt", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def j0(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_j0f", unroll.float32),
                                                    (unroll.float64,): ("__nv_j0", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def j1(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_j1f", unroll.float32),
                                                    (unroll.float64,): ("__nv_j1", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def y0(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_y0f", unroll.float32),
                                                    (unroll.float64,): ("__nv_y0", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def y1(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_y1f", unroll.float32),
                                                    (unroll.float64,): ("__nv_y1", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def yn(arg0, arg1, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, ],
                              arg_type_symbol_dict={(unroll.int32, unroll.float32,): ("__nv_ynf", unroll.float32),
                                                    (unroll.int32, unroll.float64,): ("__nv_yn", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def jn(arg0, arg1, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, ],
                              arg_type_symbol_dict={(unroll.int32, unroll.float32,): ("__nv_jnf", unroll.float32),
                                                    (unroll.int32, unroll.float64,): ("__nv_jn", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def cyl_bessel_i0(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_cyl_bessel_i0f", unroll.float32),
                                                    (unroll.float64,): ("__nv_cyl_bessel_i0", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def cyl_bessel_i1(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_cyl_bessel_i1f", unroll.float32),
                                                    (unroll.float64,): ("__nv_cyl_bessel_i1", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def erf(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_erff", unroll.float32),
                                                    (unroll.float64,): ("__nv_erf", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def erfinv(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_erfinvf", unroll.float32),
                                                    (unroll.float64,): ("__nv_erfinv", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def erfc(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_erfcf", unroll.float32),
                                                    (unroll.float64,): ("__nv_erfc", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def erfcx(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_erfcxf", unroll.float32),
                                                    (unroll.float64,): ("__nv_erfcx", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def erfcinv(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_erfcinvf", unroll.float32),
                                                    (unroll.float64,): ("__nv_erfcinv", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def normcdfinv(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_normcdfinvf", unroll.float32),
                                                    (unroll.float64,): ("__nv_normcdfinv", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def normcdf(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_normcdff", unroll.float32),
                                                    (unroll.float64,): ("__nv_normcdf", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def lgamma(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_lgammaf", unroll.float32),
                                                    (unroll.float64,): ("__nv_lgamma", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def ldexp(arg0, arg1, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, ],
                              arg_type_symbol_dict={(unroll.float32, unroll.int32,): ("__nv_ldexpf", unroll.float32),
                                                    (unroll.float64, unroll.int32,): ("__nv_ldexp", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def scalbn(arg0, arg1, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, ],
                              arg_type_symbol_dict={(unroll.float32, unroll.int32,): ("__nv_scalbnf", unroll.float32),
                                                    (unroll.float64, unroll.int32,): ("__nv_scalbn", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def fmod(arg0, arg1, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, ],
                              arg_type_symbol_dict={(unroll.float32, unroll.float32,): ("__nv_fmodf", unroll.float32),
                                                    (unroll.float64, unroll.float64,): ("__nv_fmod", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def remainder(arg0, arg1, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, ], arg_type_symbol_dict={
        (unroll.float32, unroll.float32,): ("__nv_remainderf", unroll.float32),
        (unroll.float64, unroll.float64,): ("__nv_remainder", unroll.float64),
        }, _builder=_builder)


@unroll.extern
def fma(arg0, arg1, arg2, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, arg2, ],
                              arg_type_symbol_dict={
                                  (unroll.float32, unroll.float32, unroll.float32,): ("__nv_fmaf", unroll.float32),
                                  (unroll.float64, unroll.float64, unroll.float64,): ("__nv_fma", unroll.float64),
                                  }, _builder=_builder)


@unroll.extern
def pow(arg0, arg1, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, ],
                              arg_type_symbol_dict={(unroll.float32, unroll.int32,): ("__nv_powif", unroll.float32),
                                                    (unroll.float64, unroll.int32,): ("__nv_powi", unroll.float64),
                                                    (unroll.float32, unroll.float32,): ("__nv_powf", unroll.float32),
                                                    (unroll.float64, unroll.float64,): ("__nv_pow", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def tgamma(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_tgammaf", unroll.float32),
                                                    (unroll.float64,): ("__nv_tgamma", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def round(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_roundf", unroll.float32),
                                                    (unroll.float64,): ("__nv_round", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def llround(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_llroundf", unroll.int64),
                                                    (unroll.float64,): ("__nv_llround", unroll.int64),
                                                    }, _builder=_builder)


@unroll.extern
def fdim(arg0, arg1, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, arg1, ],
                              arg_type_symbol_dict={(unroll.float32, unroll.float32,): ("__nv_fdimf", unroll.float32),
                                                    (unroll.float64, unroll.float64,): ("__nv_fdim", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def ilogb(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_ilogbf", unroll.int32),
                                                    (unroll.float64,): ("__nv_ilogb", unroll.int32),
                                                    }, _builder=_builder)


@unroll.extern
def logb(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float32,): ("__nv_logbf", unroll.float32),
                                                    (unroll.float64,): ("__nv_logb", unroll.float64),
                                                    }, _builder=_builder)


@unroll.extern
def isfinited(arg0, _builder=None):
    return unroll.elementwise(lib_name="libdevice", lib_path=LIBDEVICE_PATH, args=[arg0, ],
                              arg_type_symbol_dict={(unroll.float64,): ("__nv_isfinited", unroll.int32),
                                                    }, _builder=_builder)
