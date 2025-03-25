from triton.language import core


@core.extern
def clz(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("Unsupported", core.dtype("int32")),
            (core.dtype("int64"), ): ("Unsupported", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def popc(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("Unsupported", core.dtype("int32")),
            (core.dtype("int64"), ): ("Unsupported", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def byte_perm(arg0, arg1, arg2, _builder=None):
    return core.extern_elementwise("", "", [arg0, arg1, arg2], {
        (core.dtype("int32"), core.dtype("int32"), core.dtype("int32")): ("Unsupported", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def mulhi(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("int32")): ("Unsupported", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32")): ("_ZN3xpu6umulhiEjj", core.dtype("uint32")),
            (core.dtype("int64"), core.dtype("int64")): ("Unsupported", core.dtype("int64")),
            (core.dtype("uint64"), core.dtype("uint64")): ("Unsupported", core.dtype("uint64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def mul24(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("int32")): ("Unsupported", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32")): ("Unsupported", core.dtype("uint32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def brev(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("Unsupported", core.dtype("int32")),
            (core.dtype("int64"), ): ("Unsupported", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def sad(arg0, arg1, arg2, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("int32"), core.dtype("int32"), core.dtype("uint32")): ("Unsupported", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32"), core.dtype("uint32")): ("Unsupported", core.dtype("uint32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def abs(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("Unsupported", core.dtype("int32")),
            (core.dtype("int64"), ): ("Unsupported", core.dtype("int64")),
            (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def floor(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu9xpu_floorEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("_ZN3xpu9xpu_floorEd", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def rcp64h(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def rsqrt(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("_ZN3xpu6hrsqrtEDF16_", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("_ZN3xpu6rsqrtfEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("_ZN3xpu6rsqrtfEf", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ceil(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp64"), ): ("_ZN3xpu8xpu_ceilEd", core.dtype("fp64")),
            (core.dtype("fp32"), ): ("_ZN3xpu8xpu_ceilEf", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def trunc(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
            (core.dtype("fp32"), ): ("_ZN3xpu6truncfEf", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def exp2(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu5exp2fEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def saturatef(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def fma_rn(arg0, arg1, arg2, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def fma_rz(arg0, arg1, arg2, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def fma_rd(arg0, arg1, arg2, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def fma_ru(arg0, arg1, arg2, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def fast_dividef(arg0, arg1, _builder=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("Unsupported", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def div_rn(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("_ZN3xpu9__fdiv_rnEff", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def div_rz(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("_ZN3xpu9__fdiv_rzEff", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def div_rd(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def div_ru(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def rcp_rn(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def rcp_rz(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def rcp_rd(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def rcp_ru(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def sqrt_rn(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu10__fsqrt_rnEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("_ZN3xpu10__dsqrt_rnEd", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def sqrt_rz(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def sqrt_rd(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def sqrt_ru(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def sqrt(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu8xpu_sqrtEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("_ZN3xpu8xpu_sqrtEd", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def add_rn(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("Unsupported", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def add_rz(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("Unsupported", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def add_rd(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("Unsupported", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def add_ru(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("Unsupported", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def mul_rn(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("Unsupported", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def mul_rz(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("Unsupported", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def mul_rd(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("Unsupported", core.dtype("fp32")),
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
            ): ("Unsupported", core.dtype("fp64")),
            (
                core.dtype("fp32"),
                core.dtype("fp32"),
            ): ("Unsupported", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def double2float_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2float_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2float_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2float_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2int_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2int_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2int_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2int_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2uint_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2uint_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2uint_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2uint_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def int2double_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("Unsupported", core.dtype("fp64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def uint2double_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("Unsupported", core.dtype("fp64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2int_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2int_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2int_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2int_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2uint_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2uint_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2uint_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2uint_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def int2float_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("Unsupported", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def int2float_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("Unsupported", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def int2float_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("Unsupported", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def int2float_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("Unsupported", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def uint2float_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("Unsupported", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def uint2float_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("Unsupported", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def uint2float_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("Unsupported", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def uint2float_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("Unsupported", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def hiloint2double(arg0, arg1, _builder=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("int32"), core.dtype("int32")): ("Unsupported", core.dtype("fp64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2loint(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2hiint(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2ll_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2ll_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2ll_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2ll_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2ull_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2ull_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2ull_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2ull_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2ll_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2ll_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2ll_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2ll_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2ull_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2ull_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2ull_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2ull_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ll2float_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("Unsupported", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ll2float_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("Unsupported", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ll2float_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("Unsupported", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ll2float_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("Unsupported", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ull2float_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("Unsupported", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ull2float_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("Unsupported", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ull2float_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("Unsupported", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ull2float_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("Unsupported", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ll2double_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("Unsupported", core.dtype("fp64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ll2double_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("Unsupported", core.dtype("fp64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ll2double_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("Unsupported", core.dtype("fp64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ll2double_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("Unsupported", core.dtype("fp64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ull2double_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("Unsupported", core.dtype("fp64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ull2double_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("Unsupported", core.dtype("fp64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ull2double_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("Unsupported", core.dtype("fp64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ull2double_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("Unsupported", core.dtype("fp64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def int_as_float(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("Unsupported", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float_as_int(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def uint_as_float(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("Unsupported", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float_as_uint(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def longlong_as_double(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("Unsupported", core.dtype("fp64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double_as_longlong(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def fast_sinf(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def fast_cosf(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def fast_log2f(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def fast_logf(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def fast_expf(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def fast_tanf(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def fast_exp10f(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def fast_log10f(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def fast_powf(arg0, arg1, _builder=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("Unsupported", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def hadd(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("int32")): ("Unsupported", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32")): ("Unsupported", core.dtype("uint32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def rhadd(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("int32")): ("Unsupported", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32")): ("Unsupported", core.dtype("uint32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def sub_rn(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def sub_rz(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def sub_rd(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def sub_ru(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def rsqrt_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [
        arg0,
    ], {
        (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ffs(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("int32"), ): ("Unsupported", core.dtype("int32")),
            (core.dtype("int64"), ): ("Unsupported", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def rint(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("_ZN3xpu4rintEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def llrint(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("Unsupported", core.dtype("int64")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def nearbyint(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("_ZN3xpu9nearbyintEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def isnan(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("_ZN3xpu6hisnanEDF16_", core.dtype("int32")),
            (core.dtype("fp32"), ): ("_ZN3xpu5isnanEf", core.dtype("int32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def signbit(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("_ZN3xpu10__signbitfEf", core.dtype("int32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def copysign(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def finitef(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("_ZN3xpu7hfiniteEDF16_", core.dtype("int16")),
            (core.dtype("fp32"), ): ("_ZN3xpu7finitefEf", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def isinf(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("_ZN3xpu4hsinEDF16_", core.dtype("int32")),
            (core.dtype("fp32"), ): ("_ZN3xpu5isinfEf", core.dtype("int32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def nextafter(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def sin(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu4sinfEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def cos(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu4cosfEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def sinpi(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def cospi(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def tan(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu4tanfEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def log2(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("_ZN3xpu5hlog2EDF16_", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("_ZN3xpu5log2fEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def exp(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def exp10(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def cosh(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu5coshfEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def sinh(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu5sinhfEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def tanh(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("_ZN3xpu5htanhEDF16_", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("_ZN3xpu5tanhfEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def atan2(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("_ZN3xpu6atan2fEff", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def atan(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu5atanfEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def asin(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu5asinfEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def acos(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu5acosfEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def log(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def log10(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu6log10fEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def log1p(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu6log1pfEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def acosh(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu6acoshfEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def asinh(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu6asinhfEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def atanh(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu6atanhfEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def expm1(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu6expm1fEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def hypot(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def rhypot(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def norm3d(arg0, arg1, arg2, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def rnorm3d(arg0, arg1, arg2, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def norm4d(arg0, arg1, arg2, arg3, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2, arg3], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")):
            ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")):
            ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def rnorm4d(arg0, arg1, arg2, arg3, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2, arg3], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")):
            ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")):
            ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def cbrt(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def rcbrt(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def j0(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def j1(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def y0(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def y1(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def yn(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("fp32")): ("Unsupported", core.dtype("fp32")),
            (core.dtype("int32"), core.dtype("fp64")): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def jn(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("fp32")): ("Unsupported", core.dtype("fp32")),
            (core.dtype("int32"), core.dtype("fp64")): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def cyl_bessel_i0(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def cyl_bessel_i1(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def erf(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu3erfEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def erfinv(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu6erfinvEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def erfc(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu4erfcEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def erfcx(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def erfcinv(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def normcdfinv(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def normcdf(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def lgamma(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ldexp(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("int32")): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("int32")): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def scalbn(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("int32")): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("int32")): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def fmod(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("_ZN3xpu5fmodfEff", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def remainder(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def fma(arg0, arg1, arg2, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("_ZN3xpu3fmaEfff", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def pow(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("int32")): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("int32")): ("Unsupported", core.dtype("fp64")),
            (core.dtype("fp16"), core.dtype("fp16")): ("_ZN3xpu4hpowEDF16_DF16_", core.dtype("fp16")),
            (core.dtype("fp32"), core.dtype("fp32")): ("_ZN3xpu3powEff", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def tgamma(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def round(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def llround(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported", core.dtype("int64")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def fdim(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ilogb(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported", core.dtype("int32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def logb(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def isfinited(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def xpu_trunc_div(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("_ZN3xpu9xpu_truncEff", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)
