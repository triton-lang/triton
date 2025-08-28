################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
from triton.language import core
from triton_dist.language import core as dist_core


@core.extern
def abs(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("__triton_hip_iabs", core.dtype("int32")),
            (core.dtype("int64"), ): ("__triton_hip_iabs", core.dtype("int64")),
            (core.dtype("fp32"), ): ("__triton_hip_fabs", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__triton_hip_fabs", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def floor(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_floor_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_floor_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rsqrt(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_rsqrt_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_rsqrt_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def ceil(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_ceil_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_ceil_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def trunc(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_trunc_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_trunc_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def exp2(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_exp2_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_exp2_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def exp(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_exp_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_exp_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_expf(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__triton_hip_fast_expf", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_dividef(arg0, arg1, _semantic=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__triton_hip_fast_fdividef", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def sqrt(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_sqrt_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_sqrt_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def llrint(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__triton_hip_llrint", core.dtype("int64")),
            (core.dtype("fp64"), ): ("__triton_hip_llrint", core.dtype("int64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def nearbyint(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__ocml_nearbyint_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_nearbyint_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def isnan(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__ocml_isnan_f32", core.dtype("int32")),
            (core.dtype("fp64"), ): ("__ocml_isnan_f64", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic).to(core.int1, _semantic=_semantic)


@core.extern
def signbit(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__ocml_signbit_f32", core.dtype("int32")),
            (core.dtype("fp64"), ): ("__ocml_signbit_f64", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def copysign(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ocml_copysign_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ocml_copysign_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def isinf(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_isinf_f32", core.dtype("int32")),
            (core.dtype("fp64"), ): ("__ocml_isinf_f64", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic).to(core.int1, _semantic=_semantic)


@core.extern
def nextafter(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ocml_nextafter_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ocml_nextafter_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sin(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_sin_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_sin_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def cos(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_cos_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_cos_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def tan(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_tan_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_tan_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def log2(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_log2_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_log2_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def cosh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_cosh_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_cosh_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sinh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_sinh_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_sinh_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def tanh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_tanh_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_tanh_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def atan2(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ocml_atan2_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ocml_atan2_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def atan(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_atan_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_atan_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def asin(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_asin_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_asin_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def acos(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_acos_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_acos_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def log(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_log_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_log_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def log10(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_log10_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_log10_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def log1p(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_log1p_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_log1p_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def acosh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_acosh_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_acosh_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def asinh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_asinh_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_asinh_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def atanh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_atanh_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_atanh_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def expm1(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_expm1_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_expm1_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def hypot(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ocml_hypot_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ocml_hypot_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def j0(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_j0_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_j0_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def j1(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_j1_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_j1_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def y0(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_y0_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_y0_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def y1(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_y1_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_y1_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def cyl_bessel_i0(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_i0_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_i0_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def cyl_bessel_i1(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_i1_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_i1_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def erf(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_erf_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_erf_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def erfinv(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_erfinv_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_erfinv_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def erfc(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_erfc_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_erfc_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def erfcx(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_erfcx_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_erfcx_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def lgamma(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_lgamma_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_lgamma_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def ldexp(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("int32")): ("__ocml_ldexp_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("int32")): ("__ocml_ldexp_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fmod(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ocml_fmod_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ocml_fmod_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fma(arg0, arg1, arg2, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("__ocml_fma_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("__ocml_fma_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def pow(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("int32")): ("__ocml_pown_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("int32")): ("__ocml_pown_f64", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__ocml_pow_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ocml_pow_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def ilogb(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_ilogb_f32", core.dtype("int32")),
            (core.dtype("fp64"), ): ("__ocml_ilogb_f64", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def thread_idx(axis, _semantic=None):
    assert axis <= 2 and axis >= 0
    axis_to_xyz = ["x", "y", "z"]
    calleeName = f"llvm.amdgcn.workitem.id.{axis_to_xyz[axis]}"
    return core.extern_elementwise("", "", [], {
        (): (calleeName, core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def __syncthreads(_semantic=None):
    return core.tensor(_semantic.builder.create_barrier(), core.void)


@core.extern
def load_acquire_workgroup(arg0, _semantic=None):
    return dist_core.extern_elementwise("", "", [arg0], {
        (core.pointer_type(core.int32), ): ("__triton_hip_load_acquire_workgroup", core.int32),
    }, is_pure=False, _semantic=_semantic)


@core.extern
def load_relaxed_workgroup(arg0, _semantic=None):
    return dist_core.extern_elementwise("", "", [arg0], {
        (core.pointer_type(core.int32), ): ("__triton_hip_load_relaxed_workgroup", core.int32),
    }, is_pure=False, _semantic=_semantic)


@core.extern
def load_acquire_agent(arg0, _semantic=None):
    return dist_core.extern_elementwise("", "", [arg0], {
        (core.pointer_type(core.int32), ): ("__triton_hip_load_acquire_agent", core.int32),
    }, is_pure=False, _semantic=_semantic)


@core.extern
def load_relaxed_agent(arg0, _semantic=None):
    return dist_core.extern_elementwise("", "", [arg0], {
        (core.pointer_type(core.int32), ): ("__triton_hip_load_relaxed_agent", core.int32),
    }, is_pure=False, _semantic=_semantic)


@core.extern
def load_acquire_system(arg0, _semantic=None):
    return dist_core.extern_elementwise("", "", [arg0], {
        (core.pointer_type(core.int32), ): ("__triton_hip_load_acquire_system", core.int32),
    }, is_pure=False, _semantic=_semantic)


@core.extern
def load_relaxed_system(arg0, _semantic=None):
    return dist_core.extern_elementwise("", "", [arg0], {
        (core.pointer_type(core.int32), ): ("__triton_hip_load_relaxed_system", core.int32),
    }, is_pure=False, _semantic=_semantic)


@core.extern
def store_release_workgroup(arg0, _semantic=None):
    return dist_core.extern_elementwise("", "", [arg0], {
        (core.pointer_type(core.int32), ): ("__triton_hip_store_release_workgroup", core.int32),
    }, is_pure=False, _semantic=_semantic)


@core.extern
def store_relaxed_workgroup(arg0, _semantic=None):
    return dist_core.extern_elementwise("", "", [arg0], {
        (core.pointer_type(core.int32), ): ("__triton_hip_store_relaxed_workgroup", core.int32),
    }, is_pure=False, _semantic=_semantic)


@core.extern
def store_release_agent(arg0, arg1, _semantic=None):
    return dist_core.extern_elementwise("", "", [arg0, arg1], {
        (core.pointer_type(core.int32), core.int32): ("__triton_hip_store_release_agent", core.int32),
    }, is_pure=False, _semantic=_semantic)


@core.extern
def store_relaxed_agent(arg0, _semantic=None):
    return dist_core.extern_elementwise("", "", [arg0], {
        (core.pointer_type(core.int32), ): ("__triton_hip_store_relaxed_agent", core.int32),
    }, is_pure=False, _semantic=_semantic)


@core.extern
def store_release_system(arg0, arg1, _semantic=None):
    return dist_core.extern_elementwise("", "", [arg0, arg1], {
        (core.pointer_type(core.int32), core.int32): ("__triton_hip_store_release_system", core.int32),
    }, is_pure=False, _semantic=_semantic)


@core.extern
def store_relaxed_system(arg0, arg1, _semantic=None):
    return dist_core.extern_elementwise("", "", [arg0, arg1], {
        (core.pointer_type(core.int32), core.int32): ("__triton_hip_store_relaxed_system", core.int32),
    }, is_pure=False, _semantic=_semantic)


@core.extern
def syncthreads(_semantic=None):
    return core.extern_elementwise("", "", [], {
        (): ("__triton_hip_syncthreads", core.uint64),
    }, is_pure=False, _semantic=_semantic)


@core.extern
def red_add_release_agent(arg0, arg1, _semantic=None):
    return dist_core.extern_elementwise("", "", [arg0, arg1], {
        (core.pointer_type(core.int32), core.int32): ("__triton_hip_red_add_release_agent", core.int32),
    }, is_pure=False, _semantic=_semantic)


@core.extern
def red_add_release_system(arg0, arg1, _semantic=None):
    return dist_core.extern_elementwise("", "", [arg0, arg1], {
        (core.pointer_type(core.int32), core.int32): ("__triton_hip_red_add_release_system", core.int32),
    }, is_pure=False, _semantic=_semantic)


@core.extern
def atom_add_acquire_agent(arg0, arg1, _semantic=None):
    return dist_core.extern_elementwise("", "", [arg0, arg1], {
        (core.pointer_type(core.int32), core.int32): ("__triton_hip_atom_add_acquire_agent", core.int32),
    }, is_pure=False, _semantic=_semantic)


@core.extern
def atom_add_relaxed_agent(arg0, arg1, _semantic=None):
    return dist_core.extern_elementwise("", "", [arg0, arg1], {
        (core.pointer_type(core.int32), core.int32): ("__triton_hip_atom_add_relaxed_agent", core.int32),
    }, is_pure=False, _semantic=_semantic)


@core.extern
def atom_add_acqrel_agent(arg0, arg1, _semantic=None):
    return dist_core.extern_elementwise("", "", [arg0, arg1], {
        (core.pointer_type(core.int32), core.int32): ("__triton_hip_atom_add_acqrel_agent", core.int32),
    }, is_pure=False, _semantic=_semantic)


@core.extern
def atom_add_acquire_system(arg0, arg1, _semantic=None):
    return dist_core.extern_elementwise("", "", [arg0, arg1], {
        (core.pointer_type(core.int32), core.int32): ("__triton_hip_atom_add_acquire_system", core.int32),
    }, is_pure=False, _semantic=_semantic)


@core.extern
def atom_add_relaxed_system(arg0, arg1, _semantic=None):
    return dist_core.extern_elementwise("", "", [arg0, arg1], {
        (core.pointer_type(core.int32), core.int32): ("__triton_hip_atom_add_relaxed_system", core.int32),
    }, is_pure=False, _semantic=_semantic)


@core.extern
def atom_add_acqrel_system(arg0, arg1, _semantic=None):
    return dist_core.extern_elementwise("", "", [arg0, arg1], {
        (core.pointer_type(core.int32), core.int32): ("__triton_hip_atom_add_acqrel_system", core.int32),
    }, is_pure=False, _semantic=_semantic)


@core.extern
def atom_cas_acquire_relaxed_agent(arg0, arg1, arg2, _semantic=None):
    return dist_core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.pointer_type(core.int32), core.pointer_type(core.int32), core.int32):
            ("__triton_hip_atom_cas_acquire_relaxed_agent", core.int32),
        }, is_pure=False, _semantic=_semantic)


@core.extern
def atom_cas_release_relaxed_agent(arg0, arg1, arg2, _semantic=None):
    return dist_core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.pointer_type(core.int32), core.pointer_type(core.int32), core.int32):
            ("__triton_hip_atom_cas_release_relaxed_agent", core.int32),
        }, is_pure=False, _semantic=_semantic)


@core.extern
def atom_cas_relaxed_relaxed_agent(arg0, arg1, arg2, _semantic=None):
    return dist_core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.pointer_type(core.int32), core.pointer_type(core.int32), core.int32):
            ("__triton_hip_atom_cas_relaxed_relaxed_agent", core.int32),
        }, is_pure=False, _semantic=_semantic)


@core.extern
def atom_cas_acqrel_relaxed_agent(arg0, arg1, arg2, _semantic=None):
    return dist_core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.pointer_type(core.int32), core.pointer_type(core.int32), core.int32):
            ("__triton_hip_atom_cas_acqrel_relaxed_agent", core.uint64),
        }, is_pure=False, _semantic=_semantic)


@core.extern
def atom_cas_acquire_relaxed_system(arg0, arg1, arg2, _semantic=None):
    return dist_core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.pointer_type(core.int32), core.pointer_type(core.int32), core.int32):
            ("__triton_hip_atom_cas_acquire_relaxed_system", core.uint64),
        }, is_pure=False, _semantic=_semantic)


@core.extern
def atom_cas_release_relaxed_system(arg0, arg1, arg2, _semantic=None):
    return dist_core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.pointer_type(core.int32), core.pointer_type(core.int32), core.int32):
            ("__triton_hip_atom_cas_release_relaxed_system", core.uint64),
        }, is_pure=False, _semantic=_semantic)


@core.extern
def atom_cas_relaxed_relaxed_system(arg0, arg1, arg2, _semantic=None):
    return dist_core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.pointer_type(core.int32), core.pointer_type(core.int32), core.int32):
            ("__triton_hip_atom_cas_relaxed_relaxed_system", core.uint64),
        }, is_pure=False, _semantic=_semantic)


@core.extern
def atom_cas_acqrel_relaxed_system(arg0, arg1, arg2, _semantic=None):
    return dist_core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.pointer_type(core.int32), core.pointer_type(core.int32), core.int32):
            ("__triton_hip_atom_cas_acqrel_relaxed_system", core.uint64),
        }, is_pure=False, _semantic=_semantic)
