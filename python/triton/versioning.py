from __future__ import division, annotations

import os
import functools
import hashlib
import subprocess

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
