from __future__ import division, annotations

import functools
import hashlib
import subprocess

import triton


@functools.lru_cache()
def version_key():
    contents = []

    # frontend
    with open(__file__, "rb") as f:
        contents += [hashlib.md5(f.read()).hexdigest()]

    # TODO(crutcher): walk/hash packages

    # ptxas version
    try:
        ptxas_version = hashlib.md5(
            subprocess.check_output(["ptxas", "--version"])
        ).hexdigest()
    except Exception:
        ptxas_version = ""
    return "-".join(triton.__version__) + "-" + ptxas_version + "-" + "-".join(contents)
