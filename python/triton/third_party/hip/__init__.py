from triton.common.backend import register_backend
from .hip_backend import HIPBackend

# register backend
register_backend("hip", HIPBackend)
