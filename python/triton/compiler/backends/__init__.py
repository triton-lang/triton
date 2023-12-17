from .cuda import CUDABackend


def make_backend(target):
    return {"cuda": CUDABackend}[target[0]](target)
