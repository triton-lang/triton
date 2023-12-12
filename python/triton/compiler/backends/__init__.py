from .cuda import CUDABackend
from .hip import HIPBackend


def make_backend(target):
    return {"cuda": CUDABackend, "hip": HIPBackend}[target[0]](target)
