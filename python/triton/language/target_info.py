from triton.runtime import driver
from triton.runtime.jit import constexpr_function
from triton.backends import backends

if "amd" in backends:
    from triton._C.libtriton import amd

__all__ = ["current_target"]


def current_target():
    try:
        active_driver = driver.active
    except RuntimeError:
        # If there is no active driver, return None
        return None
    return active_driver.get_current_target()


current_target.__triton_builtin__ = True


@constexpr_function
def is_cuda():
    target = current_target()
    return target is not None and target.backend == "cuda"


@constexpr_function
def cuda_capability_geq(major, minor=0):
    """
    Determines whether we have compute capability >= (major, minor) and
    returns this as a constexpr boolean. This can be used for guarding
    inline asm implementations that require a certain compute capability.
    """
    target = current_target()
    if target is None or target.backend != "cuda":
        return False
    assert isinstance(target.arch, int)
    return target.arch >= major * 10 + minor


@constexpr_function
def is_hip():
    target = current_target()
    return target is not None and target.backend == "hip"


@constexpr_function
def is_hip_cdna2():
    target = current_target()
    return is_hip() and amd.is_hip_cdna2(target.arch)


@constexpr_function
def is_hip_cdna3():
    target = current_target()
    return is_hip() and amd.is_hip_cdna3(target.arch)


@constexpr_function
def is_hip_cdna4():
    target = current_target()
    return is_hip() and amd.is_hip_cdna4(target.arch)


@constexpr_function
def is_hip_rdna3():
    target = current_target()
    return is_hip() and amd.is_hip_rdna3(target.arch)


@constexpr_function
def is_hip_rdna4():
    target = current_target()
    return is_hip() and amd.is_hip_rdna4(target.arch)


@constexpr_function
def is_hip_gfx1250():
    target = current_target()
    return is_hip() and amd.is_hip_gfx1250(target.arch)


@constexpr_function
def is_hip_cdna():
    target = current_target()
    return is_hip() and amd.is_hip_cdna(target.arch)


@constexpr_function
def is_hip_rdna():
    target = current_target()
    return is_hip() and amd.is_hip_rdna(target.arch)


@constexpr_function
def get_hip_lds_size():
    target = current_target()
    return amd.get_hip_lds_size(target.arch) if is_hip() else 0


@constexpr_function
def hip_supports_bf16xX():
    """
    Returns true if current architecture fully supports bf16x3 and bf16x6 precision
    """
    target = current_target()
    return is_hip() and amd.hip_supports_bf16xX(target.arch)


@constexpr_function
def hip_supports_float8_uz():
    """
    Returns true if current architecture supports float8e4b8 and float8e5b16(aka E4M3FNUZ and E5M2FNUZ)
    """
    target = current_target()
    return is_hip() and amd.hip_supports_float8_uz(target.arch)


@constexpr_function
def hip_supports_cast_inf_clamping():
    target = current_target()
    return is_hip() and amd.hip_supports_cast_inf_clamping(target.arch)


@constexpr_function
def hip_supports_descriptor_scatter():
    target = current_target()
    return is_hip() and amd.hip_supports_descriptor_scatter(target.arch)


@constexpr_function
def hip_supports_f8e4m3fn_cast():
    target = current_target()
    return is_hip() and amd.hip_supports_f8e4m3fn_cast(target.arch)


@constexpr_function
def hip_supports_vdot():
    target = current_target()
    return is_hip() and amd.hip_supports_vdot(target.arch)


@constexpr_function
def hip_supports_f8e5():
    target = current_target()
    return is_hip() and amd.hip_supports_f8e5(target.arch)


@constexpr_function
def hip_supports_f8e4nv():
    target = current_target()
    return is_hip() and amd.hip_supports_f8e4nv(target.arch)


@constexpr_function
def hip_supports_f8e4m3():
    target = current_target()
    return is_hip() and amd.hip_supports_f8e4m3(target.arch)


@constexpr_function
def hip_supports_kpack():
    target = current_target()
    return is_hip() and amd.hip_supports_kpack(target.arch)


@constexpr_function
def hip_supports_scaled_dot():
    target = current_target()
    return is_hip() and amd.hip_supports_scaled_dot(target.arch)


@constexpr_function
def hip_wmma_version():
    target = current_target()
    return amd.hip_wmma_version(target.arch) if is_hip() else 0


@constexpr_function
def hip_supports_mxfp_dot():
    target = current_target()
    return is_hip() and amd.hip_supports_mxfp_dot(target.arch)


@constexpr_function
def hip_supports_mn_pack_scales():
    target = current_target()
    return is_hip() and amd.hip_supports_mn_pack_scales(target.arch)


@constexpr_function
def get_cdna_version():
    """
    Gets the CDNA architecture generation (1-4), or -1 if not CDNA.
    """
    target = current_target()
    return amd.get_cdna_version(target.arch) if is_hip() else -1


@constexpr_function
def get_rdna_version():
    """
    Gets the RDNA architecture generation (1-4), or -1 if not RDNA.
    """
    target = current_target()
    return amd.get_rdna_version(target.arch) if is_hip() else -1
