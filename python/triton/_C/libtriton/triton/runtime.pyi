from enum import Enum

CUDA: backend
HOST: backend
ROCM: backend

class backend(Enum):
    CUDA = ...
    HOST = ...
    ROCM = ...
