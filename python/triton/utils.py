from __future__ import annotations, division

import functools
import hashlib
import os
import subprocess

import torch

import triton


@functools.lru_cache()
def version_key() -> str:
    """
    Compute a code+ptxas content version key for compiled artifacts.

    :return: a version key string, cached.
    """
    # find the triton install location.
    package_dir = os.path.dirname(triton.__file__)

    # find all python code.
    code_paths = []
    for root, dirs, files in os.walk(package_dir):
        for f in files:
            if f.endswith(".py") or f.endswith(".so") or f.endswith(".bc"):
                code_paths.append(f"{root}/{f}")

        if "__pycache__" in dirs:
            dirs.remove("__pycache__")

    # coerce a stable sort of the paths
    code_paths = sorted(code_paths)

    # compute a common hash of all code.
    hasher = hashlib.md5()
    # sorted, for stable order
    for path in code_paths:
        hasher.update(open(path, "rb").read())
    code_hash = hasher.hexdigest()

    # ptxas version
    try:
        ptxas_version = hashlib.md5(
            subprocess.check_output(["ptxas", "--version"])
        ).hexdigest()
    except Exception:
        ptxas_version = "noptxas"

    return f'{"-".join(triton.__version__.split("."))}-{ptxas_version}-{code_hash}'


def cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 greater than or equal to n"""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    return n


class MockTensor:
    """
    Can be used in place of real tensors when calling:
        kernel.warmup(MockTensor(torch.float32), ...)
    """

    @staticmethod
    def wrap_dtype(arg):
        if isinstance(arg, torch.dtype):
            return MockTensor(arg)
        return arg

    def __init__(self, dtype):
        self.dtype = dtype

    def data_ptr(self):
        return 0  # optimistically assumes multiple of 16
