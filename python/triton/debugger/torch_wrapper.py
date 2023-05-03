try:
    import torch as _torch
except ImportError:
    _torch = None


class TorchWrapper:
    """
    Helps in making torch an optional dependency
    """

    def __getattr__(self, name):
        if _torch is None:
            raise ImportError("Triton requires PyTorch to be installed")
        return getattr(_torch, name)


torch = TorchWrapper()
