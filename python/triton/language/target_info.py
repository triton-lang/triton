from triton.runtime import driver
from triton.runtime.jit import constexpr_function

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
    return False if target is None else target.backend == "hip"


@constexpr_function
def is_hip_cdna2():
    target = current_target()
    return target is not None and target.backend == 'hip' and target.arch == 'gfx90a'


@constexpr_function
def is_hip_cdna3():
    target = current_target()
    return target is not None and target.backend == 'hip' and target.arch == 'gfx942'


@constexpr_function
def is_hip_cdna4():
    target = current_target()
    return target is not None and target.backend == 'hip' and target.arch == 'gfx950'


@constexpr_function
def is_hip_rdna3():
    target = current_target()
    return target is not None and target.backend == 'hip' and 'gfx11' in target.arch


@constexpr_function
def is_hip_rdna4():
    target = current_target()
    # check for gfx120 instead of gfx12, to avoid matching gfx1250
    return target is not None and target.backend == 'hip' and 'gfx120' in target.arch


@constexpr_function
def is_hip_gfx1250():
    target = current_target()
    return target is not None and target.backend == 'hip' and 'gfx1250' in target.arch


@constexpr_function
def is_hip_cdna3_or_newer():
    return is_hip_cdna3() or is_hip_cdna4()


@constexpr_function
def is_hip_cdna():
    return is_hip_cdna2() or is_hip_cdna3() or is_hip_cdna4()


@constexpr_function
def is_hip_rdna():
    return is_hip_rdna3() or is_hip_rdna4()


@constexpr_function
def get_hip_lds_size():
    return 163840 if is_hip_cdna4() else 65536


@constexpr_function
def hip_supports_bf16xX():
    """
    Returns true if current architecture fully supports bf16x3 and bf16x6 precision
    """
    if is_hip_gfx1250():
        return False
    return True


@constexpr_function
def hip_supports_float8_uz():
    """
    Returns true if current architecture supports float8e4b8 and float8e5b16(aka E4M3FNUZ and E5M2FNUZ)
    """
    return is_hip_cdna3()


@constexpr_function
def hip_supports_cast_inf_clamping():
    return not (is_hip_rdna4() or is_hip_gfx1250())


@constexpr_function
def hip_supports_descriptor_scatter():
    return is_hip_gfx1250()


@constexpr_function
def hip_supports_f8e4m3fn_cast():
    return is_hip_cdna3() or is_hip_cdna4() or is_hip_gfx1250()


@constexpr_function
def hip_supports_vdot():
    return is_hip_cdna() or is_hip_rdna() or is_hip_gfx1250() or is_hip_gfx1250()


@constexpr_function
def hip_supports_f8e5():
    return is_hip_cdna4() or is_hip_rdna4() or is_hip_gfx1250()


@constexpr_function
def hip_supports_f8e4nv():
    return is_hip_cdna4() or is_hip_rdna4() or is_hip_gfx1250()


@constexpr_function
def hip_supports_f8e4m3():
    return is_hip_cdna3() or is_hip_cdna4() or is_hip_rdna3() or is_hip_rdna4() or is_hip_gfx1250()


@constexpr_function
def hip_supports_kpack():
    return is_hip_cdna2() or is_hip_cdna3()


@constexpr_function
def hip_supports_scaled_dot():
    return is_hip_cdna() or is_hip_rdna3() or is_hip_rdna4() or is_hip_gfx1250()


@constexpr_function
def hip_wmma_version():
    if is_hip_rdna3():
        return 1
    if is_hip_cdna4():
        return 2
    if is_hip_gfx1250():
        return 3


@constexpr_function
def hip_supports_mxfp_dot():
    return is_hip_cdna4() or is_hip_gfx1250()


@constexpr_function
def hip_supports_mn_pack_scales():
    return is_hip_cdna4()


@constexpr_function
def get_cdna_version():
    """
    Gets the AMD architecture version, i.e. CDNA3 or CDNA4, currently
    only supports 3 (gfx942) or 4 (gfx950). Returns -1 if it is not AMD
    hardware or unsupported architecture
    """
    target = current_target()
    if target.backend != 'hip':
        return -1
    if target.arch == 'gfx942':
        return 3
    if target.arch == 'gfx950':
        return 4
    return -1


@constexpr_function
def get_rdna_version():
    """
    Gets the AMD architecture version, i.e. RDNA3 or RDNA4, by matching
    gfx11* (RDNA3) or gfx12* (RDNA4). Returns -1 if it is not AMD
    hardware or unsupported architecture.
    """
    target = current_target()
    if target.backend != 'hip':
        return -1
    if target.arch.startswith('gfx11'):
        return 3
    if target.arch.startswith('gfx12') and not target.arch.startswith('gfx125'):
        return 4
    return -1
