from enum import Enum
from triton.language import core


class RoundingMode(Enum):
    rn = 0  # rte
    rz = 1  # rtz
    rd = 2  # rtn
    ru = 3  # rtp
    reserve0 = 4
    reserve1 = 5


@core.extern
def clz(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def popc(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def byte_perm(arg0, arg1, arg2, _builder=None):
    raise NotImplementedError


@core.extern
def mulhi(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
            arg1,
        ], {
            (
                core.dtype("int32"),
                core.dtype("int32"),
            ): ("__mt_mulhi", core.dtype("int32")),
            (
                core.dtype("uint32"),
                core.dtype("uint32"),
            ): ("__mt_umulhi", core.dtype("uint32")),
            (
                core.dtype("int64"),
                core.dtype("int64"),
            ): ("__mt_mul64hi", core.dtype("int64")),
            (
                core.dtype("uint64"),
                core.dtype("uint64"),
            ): ("__mt_umul64hi", core.dtype("uint64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def mul24(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
            arg1,
        ], {
            (
                core.dtype("int32"),
                core.dtype("int32"),
            ): ("__mt_mul24", core.dtype("int32")),
            (
                core.dtype("uint32"),
                core.dtype("uint32"),
            ): ("__mt_umul24", core.dtype("uint32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def brev(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def sad(arg0, arg1, arg2, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
            arg1,
            arg2,
        ], {
            (
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("uint32"),
            ): ("__mt_sad", core.dtype("int32")),
            (
                core.dtype("uint32"),
                core.dtype("uint32"),
                core.dtype("uint32"),
            ): ("__mt_usad", core.dtype("uint32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def abs(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("int32"), ): ("__mt_abs_i32", core.dtype("int32")),
            # FIXME mtcc not support abs(int64)
            # (core.dtype("int64"),): ("__nv_llabs", core.dtype("int64")),
            (
                core.dtype("fp32"), ): ("__mt_fabs_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__mt_fabs_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def floor(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__mt_floor_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__mt_floor_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def rcp64h(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def rsqrt(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_rsqrtf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_rsqrt", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ceil(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp64"), ): ("__nv_ceil", core.dtype("fp64")),
            (core.dtype("fp32"), ): ("__nv_ceilf", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def trunc(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp64"), ): ("__mt_trunc_f64", core.dtype("fp64")),  # FIXME: maybe bad perf
            (core.dtype("fp32"), ): ("__mt_trunc_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def exp2(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {(core.dtype("fp32"), ): ("__mt_exp2_f32", core.dtype("fp32")), (core.dtype("fp64"), ):
                         ("__mt_exp2_f64", core.dtype("fp64")),  # FIXME: maybe bad perf
                         }, is_pure=True, _builder=_builder)


@core.extern
def saturatef(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def fma_rn(arg0, arg1, arg2, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
            arg1,
            arg2,
        ], {
            (
                core.dtype("fp32"),
                core.dtype("fp32"),
                core.dtype("fp32"),
            ): ("__mt_fmaf_rn_f32", core.dtype("fp32")),
            # FIXME mtcc not support __mt_fmaf_rn_f64
            # (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64"),): ("__nv_fma_rn", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def fma_rz(arg0, arg1, arg2, _builder=None):
    raise NotImplementedError


@core.extern
def fma_rd(arg0, arg1, arg2, _builder=None):
    raise NotImplementedError


@core.extern
def fma_ru(arg0, arg1, arg2, _builder=None):
    raise NotImplementedError


@core.extern
def fast_dividef(arg0, arg1, _builder=None):
    return core.extern_elementwise("", "", [
        arg0,
        arg1,
    ], {
        (
            core.dtype("fp32"),
            core.dtype("fp32"),
        ): ("__mt_fast_fdivide_f32", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def div_rn(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
            arg1,
        ], {
            (
                core.dtype("fp32"),
                core.dtype("fp32"),
            ): ("__mt_div_rte_f32", core.dtype("fp32")),
            (
                core.dtype("fp64"),
                core.dtype("fp64"),
            ): ("__mt_div_rte_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def div_rz(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
            arg1,
        ], {
            (
                core.dtype("fp32"),
                core.dtype("fp32"),
            ): ("__mt_div_rtz_f32", core.dtype("fp32")),
            (
                core.dtype("fp64"),
                core.dtype("fp64"),
            ): ("__mt_div_rtz_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def div_rd(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
            arg1,
        ], {
            (
                core.dtype("fp32"),
                core.dtype("fp32"),
            ): ("__mt_div_rtn_f32", core.dtype("fp32")),
            (
                core.dtype("fp64"),
                core.dtype("fp64"),
            ): ("__mt_div_rtn_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def div_ru(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
            arg1,
        ], {
            # FIXME mtcc not yet support __mt_div_rtp_f32.
            # (core.dtype("fp32"), core.dtype("fp32"),): ("__mt_div_rtp_f32", core.dtype("fp32")),
            (
                core.dtype("fp64"),
                core.dtype("fp64"),
            ): ("__mt_div_rtp_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def rcp_rn(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def rcp_rz(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def rcp_rd(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def rcp_ru(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def sqrt_rn(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def sqrt_rz(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def sqrt_rd(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def sqrt_ru(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def sqrt(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__mt_sqrt_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__mt_sqrt_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def add_rn(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
            arg1,
        ], {
            (
                core.dtype("fp64"),
                core.dtype("fp64"),
            ): ("__mt_add_rte_f64", core.dtype("fp64")),
            # FIXME mtcc not yet support __mt_add_rte_f32.
            # (core.dtype("fp32"), core.dtype("fp32"),): ("__mt_add_rte_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def add_rz(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
            arg1,
        ], {
            (
                core.dtype("fp64"),
                core.dtype("fp64"),
            ): ("__mt_add_rtz_f64", core.dtype("fp64")),
            # FIXME mtcc not yet support __mt_add_rtz_f32.
            # (core.dtype("fp32"), core.dtype("fp32"),): ("__mt_add_rtz_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def add_rd(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
            arg1,
        ], {
            (
                core.dtype("fp64"),
                core.dtype("fp64"),
            ): ("__mt_add_rtn_f64", core.dtype("fp64")),
            # FIXME mtcc not yet support __mt_add_rtn_f32.
            # (core.dtype("fp32"), core.dtype("fp32"),): ("__mt_add_rtn_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def add_ru(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
            arg1,
        ], {
            (
                core.dtype("fp64"),
                core.dtype("fp64"),
            ): ("__mt_add_rtp_f64", core.dtype("fp64")),
            # FIXME mtcc not yet support __mt_add_rtp_f32.
            # (core.dtype("fp32"), core.dtype("fp32"),): ("__mt_add_rtp_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def mul_rn(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
            arg1,
        ], {
            (
                core.dtype("fp64"),
                core.dtype("fp64"),
            ): ("__mt_mul_rte_f64", core.dtype("fp64")),
            # FIXME mtcc not yet support __mt_mul_rte_f32.
            # (core.dtype("fp32"), core.dtype("fp32"),): ("__mt_mul_rte_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def mul_rz(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
            arg1,
        ], {
            (
                core.dtype("fp64"),
                core.dtype("fp64"),
            ): ("__mt_mul_rtz_f64", core.dtype("fp64")),
            # FIXME mtcc not yet support __mt_mul_rtz_f32.
            # (core.dtype("fp32"), core.dtype("fp32"),): ("__mt_mul_rtz_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def mul_rd(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
            arg1,
        ], {
            (
                core.dtype("fp64"),
                core.dtype("fp64"),
            ): ("__mt_mul_rtn_f64", core.dtype("fp64")),
            # FIXME mtcc not yet support __mt_mul_rtn_f32.
            # (core.dtype("fp32"), core.dtype("fp32"),): ("__mt_mul_rtn_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def mul_ru(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
            arg1,
        ], {
            (
                core.dtype("fp64"),
                core.dtype("fp64"),
            ): ("__mt_mul_rtp_f64", core.dtype("fp64")),
            # FIXME mtcc not yet support __mt_mul_rtp_f32.
            # (core.dtype("fp32"), core.dtype("fp32"),): ("__mt_mul_rtp_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def double2float_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2float_rn", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2float_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2float_rz", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2float_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2float_rd", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2float_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2float_ru", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2int_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2int_rn", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2int_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2int_rz", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2int_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2int_rd", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2int_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2int_ru", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2uint_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2uint_rn", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2uint_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2uint_rz", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2uint_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2uint_rd", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2uint_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2uint_ru", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def int2double_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [
        arg0,
    ], {
        (core.dtype("int32"), ): ("__mt_i32_to_f64", core.dtype("fp64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def uint2double_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [
        arg0,
    ], {
        (core.dtype("uint32"), ): ("__mt_ui32_to_f64", core.dtype("fp64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2int_rn(arg0, _builder=None):
    raise NotImplementedError
    # TODO make sure __mt_f32_to_i32 eq to __nv_float2int_rn, which rounds to nearest.
    return core.extern_elementwise("", "", [
        arg0,
    ], {
        (core.dtype("fp32"), ): ("__mt_f32_to_i32", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2int_rz(arg0, _builder=None):
    raise NotImplementedError
    return core.extern_elementwise("", "", [
        arg0,
    ], {
        (core.dtype("fp32"), ): ("xxx", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2int_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [
        arg0,
    ], {
        (core.dtype("fp32"), ): ("__mt_f32_to_i32_rd", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2int_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [
        arg0,
    ], {
        (core.dtype("fp32"), ): ("__mt_f32_to_i32_ru", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2uint_rn(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def float2uint_rz(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def float2uint_rd(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def float2uint_ru(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def int2float_rn(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def int2float_rz(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def int2float_rd(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def int2float_ru(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def uint2float_rn(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def uint2float_rz(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def uint2float_rd(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def uint2float_ru(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def hiloint2double(arg0, arg1, _builder=None):
    raise NotImplementedError


@core.extern
def double2loint(arg0, _builder=None):
    # FIXME(lingfeng.qiu): It seems like this function is missed in libdevice.bc of musa.
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2loint", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2hiint(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def float2ll_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float2ll_rn", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2ll_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float2ll_rz", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2ll_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float2ll_rd", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2ll_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float2ll_ru", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2ull_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [
        arg0,
    ], {
        (core.dtype("fp32"), ): ("__mt_f32_to_ll_rn", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2ull_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [
        arg0,
    ], {
        (core.dtype("fp32"), ): ("__mt_f32_to_ll_rz", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2ull_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [
        arg0,
    ], {
        (core.dtype("fp32"), ): ("__mt_f32_to_ll_rd", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2ull_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [
        arg0,
    ], {
        (core.dtype("fp32"), ): ("__mt_f32_to_ll_ru", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2ll_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2ll_rn", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2ll_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2ll_rz", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2ll_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2ll_rd", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2ll_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2ll_ru", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2ull_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2ull_rn", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2ull_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2ull_rz", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2ull_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2ull_rd", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2ull_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2ull_ru", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ll2float_rn(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def ll2float_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [
        arg0,
    ], {
        (core.dtype("int64"), ): ("__mt_ll_to_f32_rz", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ll2float_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [
        arg0,
    ], {
        (core.dtype("int64"), ): ("__mt_ll_to_f32_rd", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ll2float_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [
        arg0,
    ], {
        (core.dtype("int64"), ): ("__mt_ll_to_f32_ru", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ull2float_rn(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def ull2float_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [
        arg0,
    ], {
        (core.dtype("uint64"), ): ("__mt_ull_to_f32_rz", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ull2float_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [
        arg0,
    ], {
        (core.dtype("uint64"), ): ("__mt_ull_to_f32_rd", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ull2float_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [
        arg0,
    ], {
        (core.dtype("uint64"), ): ("__mt_ull_to_f32_ru", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ll2double_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0, RoundingMode.rn], {
        (core.dtype("int64"), core.dtype("int8")): ("__mt_i64_to_f64", core.dtype("fp64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ll2double_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0, RoundingMode.rz], {
        (core.dtype("int64"), core.dtype("int8")): ("__mt_i64_to_f64", core.dtype("fp64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ll2double_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0, RoundingMode.rd], {
        (core.dtype("int64"), core.dtype("int8")): ("__mt_i64_to_f64", core.dtype("fp64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ll2double_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0, RoundingMode.ru], {
        (core.dtype("int64"), core.dtype("int8")): ("__mt_i64_to_f64", core.dtype("fp64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ull2double_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0, RoundingMode.rn], {
        (core.dtype("int64"), core.dtype("int8")): ("__mt_ui64_to_f64", core.dtype("fp64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ull2double_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0, RoundingMode.rz], {
        (core.dtype("int64"), core.dtype("int8")): ("__mt_ui64_to_f64", core.dtype("fp64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ull2double_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0, RoundingMode.rd], {
        (core.dtype("int64"), core.dtype("int8")): ("__mt_ui64_to_f64", core.dtype("fp64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ull2double_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0, RoundingMode.ru], {
        (core.dtype("int64"), core.dtype("int8")): ("__mt_ui64_to_f64", core.dtype("fp64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def int_as_float(arg0, _builder=None):
    return core.extern_elementwise("", "", [
        arg0,
    ], {
        (core.dtype("int32"), ): ("__mt_int_as_float", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float_as_int(arg0, _builder=None):
    return core.extern_elementwise("", "", [
        arg0,
    ], {
        (core.dtype("fp32"), ): ("__mt_float_as_int", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def uint_as_float(arg0, _builder=None):
    return core.extern_elementwise("", "", [
        arg0,
    ], {
        (core.dtype("uint32"), ): ("__mt_uint_as_float", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float_as_uint(arg0, _builder=None):
    return core.extern_elementwise("", "", [
        arg0,
    ], {
        (core.dtype("fp32"), ): ("__mt_float_as_uint", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def longlong_as_double(arg0, _builder=None):
    return core.extern_elementwise("", "", [
        arg0,
    ], {
        (core.dtype("int64"), ): ("__mt_longlong_as_double", core.dtype("fp64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double_as_longlong(arg0, _builder=None):
    return core.extern_elementwise("", "", [
        arg0,
    ], {
        (core.dtype("fp64"), ): ("__mt_double_as_longlong", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


# TODO It seems lack of fast_math in mtcc.


@core.extern
def fast_sinf(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def fast_cosf(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def fast_log2f(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def fast_logf(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def fast_expf(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def fast_tanf(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def fast_exp10f(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def fast_log10f(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def fast_powf(arg0, arg1, _builder=None):
    raise NotImplementedError


@core.extern
def hadd(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
            arg1,
        ], {
            (
                core.dtype("int32"),
                core.dtype("int32"),
            ): ("__mt_hadd", core.dtype("int32")),
            (
                core.dtype("uint32"),
                core.dtype("uint32"),
            ): ("__mt_uhadd", core.dtype("uint32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def rhadd(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
            arg1,
        ], {
            (
                core.dtype("int32"),
                core.dtype("int32"),
            ): ("__mt_rhadd", core.dtype("int32")),
            (
                core.dtype("uint32"),
                core.dtype("uint32"),
            ): ("__mt_urhadd", core.dtype("uint32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def sub_rn(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
            arg1,
        ], {
            (
                core.dtype("fp32"),
                core.dtype("fp32"),
            ): ("__mt_fsub_rn_f32", core.dtype("fp32")),
            (
                core.dtype("fp64"),
                core.dtype("fp64"),
            ): ("__mt_sub_rte_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def sub_rz(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
            arg1,
        ], {
            # FIXME mtcc not yet support __mt_sub_rtz_f32.
            # (core.dtype("fp32"), core.dtype("fp32"),): ("__mt_sub_rtz_f32", core.dtype("fp32")),
            (
                core.dtype("fp64"),
                core.dtype("fp64"),
            ): ("__mt_sub_rtz_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def sub_rd(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
            arg1,
        ], {
            # FIXME mtcc not yet support __mt_sub_rtn_f32.
            # (core.dtype("fp32"), core.dtype("fp32"),): ("__mt_sub_rtn_f32", core.dtype("fp32")),
            (
                core.dtype("fp64"),
                core.dtype("fp64"),
            ): ("__mt_sub_rtn_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def sub_ru(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
            arg1,
        ], {
            # FIXME mtcc not yet support __mt_sub_rtp_f32.
            # (core.dtype("fp32"), core.dtype("fp32"),): ("__mt_sub_rtp_f32", core.dtype("fp32")),
            (
                core.dtype("fp64"),
                core.dtype("fp64"),
            ): ("__mt_sub_rtp_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def rsqrt_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [
        arg0,
    ], {
        (core.dtype("fp32"), ): ("__mt_rsqrt_rn_f32", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ffs(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("int32"), ): ("__mt_ffs_i32", core.dtype("int32")),
            (core.dtype("int64"), ): ("__mt_ffsll_i64", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def rint(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__nv_rintf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_rint", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def llrint(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__nv_llrintf", core.dtype("int64")),
            (core.dtype("fp64"), ): ("__nv_llrint", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def nearbyint(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def isnan(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__mt_isnan_f32", core.dtype("int32")),
            (core.dtype("fp64"), ): ("__mt_isnan_f64", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def signbit(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__mt_signbit_f32", core.dtype("int1")),
            (core.dtype("fp64"), ): ("__mt_signbit_f64", core.dtype("int1")),
        }, is_pure=True, _builder=_builder)


@core.extern
def copysign(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__nv_copysignf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__nv_copysign", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def finitef(arg0, _builder=None):
    return core.extern_elementwise("", "", [
        arg0,
    ], {
        (core.dtype("fp32"), ): ("__mt_isfinite_f32", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def isinf(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__mt_isinf_f32", core.dtype("int32")),
            (core.dtype("fp64"), ): ("__mt_isinf_f64", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def nextafter(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
            arg1,
        ], {
            (
                core.dtype("fp32"),
                core.dtype("fp32"),
            ): ("__mt_nextafter_f32", core.dtype("fp32")),
            (
                core.dtype("fp64"),
                core.dtype("fp64"),
            ): ("__mt_nextafter_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def sin(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_sinf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_sin", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def cos(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_cosf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_cos", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def sinpi(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__mt_sinpi_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__mt_sinpi_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def cospi(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            # FIXME mtcc not yet support __mt_cospi_f32.
            # (core.dtype("fp32"),): ("__mt_cospi_f32", core.dtype("fp32")),
            (
                core.dtype("fp64"), ): ("__mt_cospi_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def tan(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__mt_tan_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__mt_tan_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def log2(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_log2f", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_log2", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def exp(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_expf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_exp", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def exp10(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            # FIXME mtcc not yet support __mt_exp10_f32.
            # (core.dtype("fp32"),): ("__mt_exp10_f32", core.dtype("fp32")),
            (
                core.dtype("fp64"), ): ("__mt_exp10_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def cosh(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_coshf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_cosh", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def sinh(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__mt_sinh_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__mt_sinh_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def tanh(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0],
        {(core.dtype("fp32"), ):
         ("__mt_tanh_f32",
          core.dtype("fp32")),  # FIXME: mtcc should wrap the libdevice func to support mp_32 hw supported tanhf
         (core.dtype("fp64"), ): ("__mt_tanh_f64", core.dtype("fp64")),  # FIXME: maybe bad perf
         }, is_pure=True, _builder=_builder)


@core.extern
def atan2(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__nv_atan2f", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__nv_atan2", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def atan(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_atanf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_atan", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def asin(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_asinf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_asin", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def acos(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_acosf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_acos", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def log(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__mt_log_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__mt_log_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def log10(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__mt_log10_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__mt_log10_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def log1p(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__mt_log1p_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__mt_log1p_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def acosh(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_acoshf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_acosh", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def asinh(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_asinhf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_asinh", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def atanh(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_atanhf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_atanh", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def expm1(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_expm1f", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_expm1", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def hypot(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__nv_hypotf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__nv_hypot", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def rhypot(arg0, arg1, _builder=None):
    raise NotImplementedError


@core.extern
def norm3d(arg0, arg1, arg2, _builder=None):
    raise NotImplementedError


@core.extern
def rnorm3d(arg0, arg1, arg2, _builder=None):
    raise NotImplementedError


@core.extern
def norm4d(arg0, arg1, arg2, arg3, _builder=None):
    raise NotImplementedError


@core.extern
def rnorm4d(arg0, arg1, arg2, arg3, _builder=None):
    raise NotImplementedError


@core.extern
def cbrt(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__mt_cbrt_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__mt_cbrt_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def rcbrt(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def j0(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def j1(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def y0(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def y1(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def yn(arg0, arg1, _builder=None):
    raise NotImplementedError


@core.extern
def jn(arg0, arg1, _builder=None):
    raise NotImplementedError


@core.extern
def cyl_bessel_i0(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def cyl_bessel_i1(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def erf(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {(core.dtype("fp32"), ): ("__mt_erf_f32", core.dtype("fp32")), (core.dtype("fp64"), ):
                         ("__mt_erf_f64", core.dtype("fp64")),  # FIXME: maybe bad perf
                         }, is_pure=True, _builder=_builder)


@core.extern
def erfinv(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__mt_erfinv_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__mt_erfinv_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def erfc(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__mt_erfc_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__mt_erfc_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def erfcx(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def erfcinv(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__mt_erfcinv_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__mt_erfcinv_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def normcdfinv(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def normcdf(arg0, _builder=None):
    raise NotImplementedError


@core.extern
def lgamma(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__mt_lgamma_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__mt_lgamma_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ldexp(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {(core.dtype("fp32"), core.dtype("int32")): ("__mt_ldexp_f32", core.dtype("fp32")),
                               (core.dtype("fp64"), core.dtype("int32")):
                               ("__mt_ldexp_f64", core.dtype("fp64")),  # FIXME: maybe bad perf
                               }, is_pure=True, _builder=_builder)


@core.extern
def scalbn(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
            arg1,
        ], {
            (
                core.dtype("fp32"),
                core.dtype("int32"),
            ): ("__mt_scalbn_f32", core.dtype("fp32")),
            (
                core.dtype("fp64"),
                core.dtype("int32"),
            ): ("__mt_scalbn_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def fmod(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
            arg1,
        ], {
            (
                core.dtype("fp32"),
                core.dtype("fp32"),
            ): ("__mt_fmod_f32", core.dtype("fp32")),
            (
                core.dtype("fp64"),
                core.dtype("fp64"),
            ): ("__mt_fmod_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def remainder(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
            arg1,
        ], {
            (
                core.dtype("fp32"),
                core.dtype("fp32"),
            ): ("__mt_remainder_f32", core.dtype("fp32")),
            (
                core.dtype("fp64"),
                core.dtype("fp64"),
            ): ("__mt_remainder_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def fma(arg0, arg1, arg2, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("__nv_fmaf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("__nv_fma", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def pow(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
            arg1,
        ], {
            (
                core.dtype("fp32"),
                core.dtype("int32"),
            ): ("__mt_pown_f32", core.dtype("fp32")),
            (
                core.dtype("fp64"),
                core.dtype("int32"),
            ): ("__mt_pown_f64", core.dtype("fp64")),
            (
                core.dtype("fp32"),
                core.dtype("fp32"),
            ): ("__mt_pow_f32", core.dtype("fp32")),
            (
                core.dtype("fp64"),
                core.dtype("fp64"),
            ): ("__mt_pow_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def tgamma(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__mt_tgamma_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__mt_tgamma_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def round(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__mt_round_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__mt_round_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def llround(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__mt_llround_f32", core.dtype("int64")),
            (core.dtype("fp64"), ): ("__mt_llround_f64", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def fdim(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fdimf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__nv_fdim", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ilogb(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_ilogbf", core.dtype("int32")),
            (core.dtype("fp64"), ): ("__nv_ilogb", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def logb(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__mt_logb_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__mt_logb_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def isfinited(arg0, _builder=None):
    return core.extern_elementwise("", "", [
        arg0,
    ], {
        (core.dtype("fp64"), ): ("__mt_isfinite_f64", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


# device capability >= 31
@core.extern
def fast_gelu(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__mt_tt_gelu_f32", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


# device capability >= 31
@core.extern
def fast_tanh(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__mt_tt_tanh_f32", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)
