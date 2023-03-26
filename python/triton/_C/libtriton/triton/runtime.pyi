from enum import IntEnum

HOST: backend
CUDA: backend
ROCM: backend

class backend(IntEnum):
    HOST = 0
    CUDA = 1
    ROCM = 2
