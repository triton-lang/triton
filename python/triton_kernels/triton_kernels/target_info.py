import torch
import triton

cached_capabilities = {}


def is_hip():
    if "is_hip" not in cached_capabilities:
        cached_capabilities["is_hip"] = torch.cuda.is_available() and bool(torch.version.hip)
    return cached_capabilities["is_hip"]


def cuda_capability_geq(major, minor=0):
    """
    Determines whether we have compute capability >= (major, minor) and
    returns this as a constexpr boolean. This can be used for guarding
    inline asm implementations that require a certain compute capability.
    """
    if is_hip():
        return False
    if "cuda" not in cached_capabilities:
        if torch.cuda.is_available():
            cached_capabilities["cuda"] = torch.cuda.get_device_capability()
        else:
            cached_capabilities["cuda"] = (0, 0)
    return cached_capabilities["cuda"] >= (major, minor)


def get_cdna_version():
    """
    Gets the AMD architecture version, i.e. CDNA3 or CDNA4, currently
    only supports 3 (gfx942) or 4 (gfx950). Returns -1 if it is not AMD
    hardware or unsupported architecture
    """
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != 'hip':
        return -1
    if target.arch == 'gfx942':
        return 3
    if target.arch == 'gfx950':
        return 4
    return -1


def num_sms():
    return torch.cuda.get_device_properties(0).multi_processor_count
